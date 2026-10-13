"""Read-only access to the private continuous LFP recordings and event labels."""

import hashlib
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SOURCE_HZ = 500
TARGETS = ["X1", "X2", "X4", "X9", "X10", "X14", "X15"]
DAY_SPLITS = {"train": list(range(8)), "validation": [8, 11], "test": [12, 13]}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_sources(root, annotations):
    root = Path(root)
    sources = {}
    for path in sorted(root.glob("*.parquet")):
        if path.name == annotations:
            continue
        match = re.search(r"_(X\d+)_\d+_Days_(\d+)_interpolated\.parquet$", path.name)
        if match is None:
            raise ValueError(f"Unrecognized recording identity: {path.name}")
        parquet = pq.ParquetFile(path)
        if not {"pi_time_ms", "lfp"}.issubset(parquet.schema_arrow.names):
            raise ValueError(f"Missing time or LFP columns: {path.name}")
        column = parquet.schema_arrow.get_field_index("pi_time_ms")
        first = parquet.metadata.row_group(0).column(column).statistics
        last = parquet.metadata.row_group(parquet.num_row_groups - 1).column(column).statistics
        if not first or not last or not first.has_min_max or not last.has_min_max:
            raise ValueError(f"Missing timestamp metadata: {path.name}")
        stat = path.stat()
        sources[path.name] = dict(mouse=match.group(1), day=int(match.group(2)),
            rows=parquet.metadata.num_rows, start_ms=int(first.min), end_ms=int(last.max),
            size=stat.st_size, mtime_ns=stat.st_mtime_ns,
            columns=parquet.schema_arrow.names)
    if not sources:
        raise ValueError("No continuous recordings found")
    return dict(annotation_sha256=file_hash(root / annotations), sources=sources)


class SourceDataset:
    def __init__(self, root, annotations, expected_signature):
        self.root = Path(root)
        self.annotations = annotations
        self.signature = inspect_sources(root, annotations)
        if self.signature != expected_signature:
            raise ValueError("Raw recordings or source annotations changed")
        self.sources = self.signature["sources"]
        self.events = pq.read_table(self.root / annotations).to_pandas()
        required = {"mouse", "day", "raw_file", "epoch_id", "start_row", "end_row",
                    "start_pi_time_ms", "end_pi_time_ms"}
        if not required.issubset(self.events.columns) or self.events[list(required)].isna().any().any():
            raise ValueError("Missing source annotation fields")
        if self.events.duplicated(["raw_file", "epoch_id"]).any():
            raise ValueError("Duplicate source events")
        for event in self.events.itertuples():
            source = self.sources.get(event.raw_file)
            if source is None or source["mouse"] != event.mouse or source["day"] != int(event.day):
                raise ValueError("Annotation recording identity mismatch")
            if not 0 <= event.start_row <= event.end_row < source["rows"]:
                raise ValueError("Annotation rows outside the recording")
            if (int(event.start_pi_time_ms) != source["start_ms"] + 2 * int(event.start_row)
                    or int(event.end_pi_time_ms) != source["start_ms"] + 2 * int(event.end_row)):
                raise ValueError("Annotation timestamps and rows are misaligned")

    def reader(self, target, split):
        allowed = {name for name, s in self.sources.items()
                   if s["mouse"] == target and s["day"] in DAY_SPLITS[split]}
        return SignalReader(self.root, self.sources, allowed)


class SignalReader:
    """Read selected rows without silently repairing or rejecting LFP artifacts."""

    def __init__(self, root, sources, allowed):
        self.root, self.sources, self.allowed = Path(root), sources, allowed
        self.filename, self.groups = None, OrderedDict()

    def read(self, filename, start, stop):
        if filename not in self.allowed:
            raise ValueError("Signal read crossed the fixed mouse/day split")
        if not 0 <= start < stop <= self.sources[filename]["rows"]:
            raise ValueError("Signal read outside source bounds")
        if self.filename != filename:
            self.parquet = pq.ParquetFile(self.root / filename)
            sizes = [self.parquet.metadata.row_group(i).num_rows for i in range(self.parquet.num_row_groups)]
            self.offsets = np.r_[0, np.cumsum(sizes)]
            self.groups.clear()
            self.filename = filename
        first = np.searchsorted(self.offsets, start, side="right") - 1
        last = np.searchsorted(self.offsets, stop - 1, side="right") - 1
        pieces = []
        for index in range(first, last + 1):
            if index not in self.groups:
                self.groups[index] = self.parquet.read_row_group(index, columns=["lfp"]).column("lfp").to_numpy()
            signal = self.groups[index]
            self.groups.move_to_end(index)
            left = max(start, self.offsets[index]) - self.offsets[index]
            right = min(stop, self.offsets[index + 1]) - self.offsets[index]
            pieces.append(signal[left:right])
            while len(self.groups) > 12:
                self.groups.popitem(last=False)
        return np.concatenate(pieces)
