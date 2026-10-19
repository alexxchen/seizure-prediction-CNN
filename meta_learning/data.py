"""Lazy split loading, source audits, and cached centered/resampled waveforms."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from meta_learning.crops import WINDOW_SAMPLES, crop_segments, negative_crop_pool, random_crops
from meta_learning.waveform import cached_waveforms
from meta_learning.common import digest, signal_audit, validate_sample_table, write_json
from prepare import load_dataset
import sample_pipeline


@dataclass
class FeatureSplit:
    mouse: str
    split: str
    inputs: np.ndarray
    labels: np.ndarray
    table: object
    crop_segments: np.ndarray = None
    crop_max_offsets: np.ndarray = None
    crop_row_contexts: np.ndarray = None
    negative_pool: object = None

    @property
    def fingerprint(self):
        columns = ["mouse", "day", "raw_file", "start_row", "stop_row", "label"]
        return digest(self.table[columns].to_dict("records"))

    def subset_days(self, days):
        """Restrict to whole recording days, so overlapping windows never cross the boundary."""
        mask = self.table.day.isin(days).to_numpy()
        if mask.all():
            return self
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError(f"{self.mouse}: no {self.split} windows on days {list(days)}")
        inputs = None if self.inputs is None else np.asarray(self.inputs[indices])
        contexts = None
        if self.crop_segments is not None:
            contexts = indices if self.crop_row_contexts is None else self.crop_row_contexts[indices]
        return FeatureSplit(self.mouse, self.split, inputs, np.asarray(self.labels)[indices],
                            self.table.iloc[indices].reset_index(drop=True), self.crop_segments,
                            self.crop_max_offsets, contexts, self.negative_pool)

    def scale(self, indices=None):
        # Preserve waveform amplitude before the neural frontend.
        # Retain the saved-scale interface for personalization and inference.
        return 1.0

    def batch(self, indices, scale, device):
        if self.inputs is None:
            raise ValueError("Fixed-window inference requires DataStore.load, not load_crops")
        if scale != 1.0:
            raise ValueError("Waveform frontends require identity normalization")
        values = np.array(self.inputs[indices], dtype=np.float32, copy=True)
        labels = np.array(self.labels[indices], dtype=np.int64, copy=True)
        return torch.from_numpy(values).to(device), torch.from_numpy(labels).to(device)


    def random_crop_batch(self, indices, rng, device):
        """Training-only: a fresh random crop of each row's context segment."""
        if self.crop_segments is None:
            raise ValueError("Random crops require DataStore.load_crops")
        contexts = indices if self.crop_row_contexts is None else self.crop_row_contexts[indices]
        labels = np.array(self.labels[indices], dtype=np.int64, copy=True)
        if self.negative_pool is None:
            if (np.asarray(contexts) < 0).any():
                raise ValueError("Negative crops require a complete safe-interval pool")
            values = random_crops(self.crop_segments, self.crop_max_offsets, contexts, rng)
        else:
            values = np.empty((len(indices), WINDOW_SAMPLES), dtype=np.float32)
            positive, negative = labels == 1, labels == 0
            if positive.any():
                values[positive] = random_crops(self.crop_segments, self.crop_max_offsets, contexts[positive], rng)
            if negative.any():
                days = self.table.iloc[np.asarray(indices)[negative]].day.to_numpy()
                values[negative] = self.negative_pool.sample(days, rng)
        return torch.from_numpy(values).to(device), torch.from_numpy(labels).to(device)


class DataStore:
    """Construct all sample tables, but read signal values only for requested splits."""

    def __init__(self, prepared_dir, data_dir, cache_dir, artifact_dir):
        self.dataset = load_dataset(prepared_dir, data_dir)
        self.cache_dir, self.artifact_dir = Path(cache_dir), Path(artifact_dir)
        self.tables, self.loaded = {}, {}

    def sample_table(self, mouse, split):
        """Prepare split metadata without reading fixed windows or crop contexts."""
        if mouse not in self.tables:
            tables, selection = sample_pipeline.build_tables(self.dataset, mouse)
            for part, table in tables.items():
                validate_sample_table(table, self.dataset, mouse, part)
            self.tables[mouse] = tables
            selection_path = self.artifact_dir / mouse / "event_selection.csv"
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection.to_csv(selection_path, index=False)
            summary = {part: dict(samples=len(table), positives=int((table.label == 1).sum()),
                                  negatives=int((table.label == 0).sum()),
                                  positive_fraction=float((table.label == 1).mean()),
                                  eligible_events=int(((selection.selection == "selected") &
                                      selection.day.isin(table.day.unique())).sum()))
                       for part, table in tables.items()}
            write_json(self.artifact_dir / mouse / "sampling_summary.json", summary)
            for part, table in tables.items():
                values = FeatureSplit(mouse, part, None, table.label.to_numpy(), table)
                write_json(self.artifact_dir / mouse / f"{part}_provenance.json",
                           dict(fingerprint=values.fingerprint, samples=len(table),
                                days=sorted(table.day.unique().tolist())))
                table.to_csv(self.artifact_dir / mouse / f"{part}_samples.csv", index=False)
        return self.tables[mouse][split]

    def load(self, mouse, split):
        key = (mouse, split)
        if key in self.loaded:
            return self.loaded[key]
        table = self.sample_table(mouse, split)
        raw, labels = sample_pipeline.materialize(self.dataset, mouse, split, table,
                                                  self.cache_dir / "samples")
        write_json(self.artifact_dir / mouse / f"{split}_signal_audit.json", signal_audit(raw))
        waveforms = cached_waveforms(raw, self.cache_dir / "waveforms")
        result = FeatureSplit(mouse, split, waveforms, labels, table)
        self.loaded[key] = result
        return result

    def load_crops(self, mouse, split, days=None):
        """Read only unique crop contexts on requested days; skip fixed-window caches."""
        days = None if days is None else tuple(sorted(days))
        key = (mouse, split, "crops", days)
        if key not in self.loaded:
            table = self.sample_table(mouse, split)
            if days is not None:
                table = table[table.day.isin(days)].reset_index(drop=True)
            if table.empty:
                raise ValueError(f"{mouse}: no {split} crop windows on days {days}")
            segments, offsets, contexts = crop_segments(self.dataset, mouse, split, table, self.cache_dir / "crops")
            negatives = negative_crop_pool(self.dataset, mouse, split, table.day.unique(),
                                           self.cache_dir / "negative-crops")
            self.loaded[key] = FeatureSplit(mouse, split, None, table.label.to_numpy(), table,
                                            segments, offsets, contexts, negatives)
            suffix = "all" if days is None else "-".join(map(str, days))
            write_json(self.artifact_dir / mouse / f"{split}_crop_days-{suffix}_provenance.json",
                       dict(fingerprint=self.loaded[key].fingerprint, samples=len(table),
                            days=sorted(table.day.unique().tolist()), unique_contexts=len(segments),
                            fixed_windows_materialized=False, negative_pool=negatives.describe()))
        return self.loaded[key]
