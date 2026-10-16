"""Lazy split loading, source audits, cached fixed spectra, and train-only scaling."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from learnable_frontend import fixed_power
from meta_learning.common import digest, signal_audit, validate_sample_table, write_json
from prepare import load_dataset
import sample_pipeline


@dataclass
class FeatureSplit:
    mouse: str
    split: str
    power: np.ndarray
    labels: np.ndarray
    table: object

    @property
    def fingerprint(self):
        columns = ["mouse", "day", "raw_file", "start_row", "stop_row", "label"]
        return digest(self.table[columns].to_dict("records"))

    def scale(self, indices=None):
        values = self.power if indices is None else self.power[indices]
        return max(float(np.mean(values, dtype=np.float64)), 1e-12)

    def batch(self, indices, scale, device):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Normalization scale must be finite and positive")
        values = np.array(self.power[indices], dtype=np.float32, copy=True) / np.float32(scale)
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
        power = fixed_power(raw, self.cache_dir / "spectra")
        result = FeatureSplit(mouse, split, power, labels, table)
        write_json(self.artifact_dir / mouse / f"{split}_provenance.json",
                   dict(fingerprint=result.fingerprint, samples=len(labels), days=sorted(table.day.unique().tolist())))
        table.to_csv(self.artifact_dir / mouse / f"{split}_samples.csv", index=False)
        self.loaded[key] = result
        return result
