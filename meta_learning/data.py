"""Lazy split loading, source audits, and cached centered/resampled waveforms."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from meta_learning.crops import crop_segments, random_crops
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
        segments = None if self.crop_segments is None else np.asarray(self.crop_segments[indices])
        offsets = None if self.crop_max_offsets is None else np.asarray(self.crop_max_offsets)[indices]
        return FeatureSplit(self.mouse, self.split, inputs, np.asarray(self.labels)[indices],
                            self.table.iloc[indices].reset_index(drop=True), segments, offsets)

    def scale(self, indices=None):
        # Preserve absolute spectral amplitude before the neural log transform.
        # Retain the saved-scale interface for personalization and inference.
        return 1.0

    def batch(self, indices, scale, device):
        if scale != 1.0:
            raise ValueError("The neural STFT frontend requires identity normalization")
        values = np.array(self.inputs[indices], dtype=np.float32, copy=True)
        labels = np.array(self.labels[indices], dtype=np.int64, copy=True)
        return torch.from_numpy(values).to(device), torch.from_numpy(labels).to(device)


    def random_crop_batch(self, indices, rng, device):
        """Training-only: a fresh random crop of each row's context segment."""
        if self.crop_segments is None:
            raise ValueError("Random crops require DataStore.load_crops")
        values = random_crops(self.crop_segments, self.crop_max_offsets, indices, rng)
        labels = np.array(self.labels[indices], dtype=np.int64, copy=True)
        return torch.from_numpy(values).to(device), torch.from_numpy(labels).to(device)


class DataStore:
    """Construct all sample tables, but read signal values only for requested splits."""

    def __init__(self, prepared_dir, data_dir, cache_dir, artifact_dir):
        self.dataset = load_dataset(prepared_dir, data_dir)
        self.cache_dir, self.artifact_dir = Path(cache_dir), Path(artifact_dir)
        self.tables, self.loaded = {}, {}

    def load(self, mouse, split):
        key = (mouse, split)
        if key in self.loaded:
            return self.loaded[key]
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
        table = self.tables[mouse][split]
        raw, labels = sample_pipeline.materialize(self.dataset, mouse, split, table,
                                                  self.cache_dir / "samples")
        write_json(self.artifact_dir / mouse / f"{split}_signal_audit.json", signal_audit(raw))
        waveforms = cached_waveforms(raw, self.cache_dir / "waveforms")
        result = FeatureSplit(mouse, split, waveforms, labels, table)
        write_json(self.artifact_dir / mouse / f"{split}_provenance.json",
                   dict(fingerprint=result.fingerprint, samples=len(labels), days=sorted(table.day.unique().tolist())))
        table.to_csv(self.artifact_dir / mouse / f"{split}_samples.csv", index=False)
        self.loaded[key] = result
        return result

    def load_crops(self, mouse, split):
        """The same split with cached context segments for training-time random crops."""
        key = (mouse, split, "crops")
        if key not in self.loaded:
            base = self.load(mouse, split)
            segments, offsets = crop_segments(self.dataset, mouse, split, base.table, self.cache_dir / "crops")
            self.loaded[key] = replace(base, crop_segments=segments, crop_max_offsets=offsets)
        return self.loaded[key]
