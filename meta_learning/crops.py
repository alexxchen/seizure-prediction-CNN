"""Context segments for training-time random crops, preprocessed like fixed windows.

Each positive event shares one cached 7-3 minute context. Negative sampling uses
all eligible safe intervals of the same mouse and day, with probability proportional
to their number of legal crop starts. Contexts have their mean removed and are
resampled to 256 Hz once; a crop then takes 112 s from a segment and removes the
crop's own mean, matching `waveform.preprocess_batch` up to resampling edges.
"""

import hashlib
import json
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

import sample_pipeline
from source_data import DAY_SPLITS, SOURCE_HZ, file_hash
from meta_learning.common import write_json
from meta_learning.waveform import WAVEFORM_CONFIG

TARGET_HZ = WAVEFORM_CONFIG["target_hz"]
WINDOW_SAMPLES = int(round(sample_pipeline.SAMPLE_CONFIG["window_seconds"] * TARGET_HZ))
SEGMENT_SAMPLES = int(round((sample_pipeline.SAMPLE_CONFIG["positive_earlier_minutes"] -
                             sample_pipeline.SAMPLE_CONFIG["positive_later_minutes"]) * 60 * TARGET_HZ))


def crop_segments(dataset, target, split, rows, cache_dir):
    """Return positive contexts and a row mapping (-1 for negative placeholder rows)."""
    positive_positions = np.flatnonzero(rows.label.to_numpy() == 1)
    positive_rows = rows.iloc[positive_positions]
    ranges = sample_pipeline.crop_ranges(dataset, target, positive_rows)
    contexts, unique_positions = {}, []
    row_contexts = np.full(len(rows), -1, dtype=np.int64)
    for position, (filename, (first, last)) in enumerate(zip(positive_rows.raw_file, ranges)):
        key = (filename, int(first), int(last))
        if key not in contexts:
            contexts[key] = len(unique_positions)
            unique_positions.append(position)
        row_contexts[positive_positions[position]] = contexts[key]
    window_rows, _ = sample_pipeline.sample_offsets()
    signature = dict(version=3, source=dataset.signature, sample_code=file_hash(sample_pipeline.__file__),
                     crop_code=file_hash(__file__), waveform=WAVEFORM_CONFIG, crop=sample_pipeline.CROP_CONFIG,
                     sample=sample_pipeline.SAMPLE_CONFIG,
                     rows_sha256=hashlib.sha256(rows.to_csv(index=False).encode()).hexdigest(),
                     ranges_sha256=hashlib.sha256(ranges.tobytes()).hexdigest())
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / key
    segments_path, offsets_path, manifest = cache / "segments.npy", cache / "max_offsets.npy", cache / "manifest.json"
    row_contexts_path = cache / "row_contexts.npy"
    if manifest.exists():
        stored = json.loads(manifest.read_text())
        if stored["signature"] != signature or any(file_hash(cache / name) != digest
                                                   for name, digest in stored["files"].items()):
            raise ValueError("Crop segment cache changed")
        return np.load(segments_path, mmap_mode="r"), np.load(offsets_path), np.load(row_contexts_path)
    cache.mkdir(parents=True, exist_ok=True)
    segments = np.lib.format.open_memmap(str(segments_path) + ".tmp", mode="w+", dtype=np.float32,
                                         shape=(len(unique_positions), SEGMENT_SAMPLES))
    max_offsets = np.empty(len(unique_positions), dtype=np.int64)
    reader = dataset.reader(target, split)
    for index, position in enumerate(unique_positions):
        row = positive_rows.iloc[position]
        first, last = ranges[position]
        signal = sample_pipeline.clean_signal(reader.read(row.raw_file, int(first), int(last) + window_rows))
        signal = signal - np.float32(signal.mean(dtype=np.float64))
        values = resample_poly(signal, TARGET_HZ, SOURCE_HZ).astype(np.float32)
        offset = int((last - first) * TARGET_HZ // SOURCE_HZ)
        if offset + WINDOW_SAMPLES > len(values) or len(values) > SEGMENT_SAMPLES or not np.isfinite(values).all():
            raise ValueError("Invalid crop segment")
        segments[index] = 0.0
        segments[index, :len(values)] = values
        max_offsets[index] = offset
    segments.flush()
    del segments
    Path(str(segments_path) + ".tmp").replace(segments_path)
    np.save(offsets_path, max_offsets, allow_pickle=False)
    np.save(row_contexts_path, row_contexts, allow_pickle=False)
    write_json(manifest, dict(signature=signature, files={path.name: file_hash(path)
                                                         for path in [segments_path, offsets_path, row_contexts_path]}))
    return np.load(segments_path, mmap_mode="r"), np.load(offsets_path), row_contexts


# Align streamed resampling blocks and halos to the rational resampling phase.
RESAMPLE_DOWN = SOURCE_HZ // gcd(SOURCE_HZ, TARGET_HZ)
NEGATIVE_BLOCK_ROWS = RESAMPLE_DOWN * 8000
NEGATIVE_HALO_ROWS = RESAMPLE_DOWN * 2


def write_resampled_interval(reader, filename, first, last, output):
    """Resample a long interval with bounded RAM and the same phase as a full pass."""
    length, total = last - first, 0.0
    for start in range(first, last, NEGATIVE_BLOCK_ROWS):
        signal = sample_pipeline.clean_signal(reader.read(filename, start, min(start + NEGATIVE_BLOCK_ROWS, last)))
        if not np.isfinite(signal).all():
            raise ValueError("Non-finite negative crop context")
        total += signal.sum(dtype=np.float64)
    mean = np.float32(total / length)
    for start in range(0, length, NEGATIVE_BLOCK_ROWS):
        left = max(0, start - NEGATIVE_HALO_ROWS)
        right = min(length, start + NEGATIVE_BLOCK_ROWS + NEGATIVE_HALO_ROWS)
        signal = sample_pipeline.clean_signal(reader.read(filename, first + left, first + right)) - mean
        values = resample_poly(signal, TARGET_HZ, SOURCE_HZ).astype(np.float32)
        begin = start * TARGET_HZ // SOURCE_HZ
        end = min(len(output), (start + NEGATIVE_BLOCK_ROWS) * TARGET_HZ // SOURCE_HZ)
        offset = (start - left) * TARGET_HZ // SOURCE_HZ
        chunk = values[offset:offset + end - begin]
        if len(chunk) != end - begin or not np.isfinite(chunk).all():
            raise ValueError("Invalid resampled negative crop context")
        output[begin:end] = chunk


@dataclass
class NegativeCropPool:
    values: np.ndarray
    intervals: list
    fingerprint: str

    def draw_starts(self, days, rng):
        """Uniformly draw from the union of all legal 256-Hz starts on each day."""
        days = np.asarray(days, dtype=np.int64)
        contexts, starts = np.empty(len(days), dtype=np.int64), np.empty(len(days), dtype=np.int64)
        for day in np.unique(days):
            choices = np.array([i for i, interval in enumerate(self.intervals) if interval["day"] == day], dtype=np.int64)
            if not len(choices):
                raise ValueError(f"No eligible negative crop intervals on Day {day}")
            counts = np.array([self.intervals[i]["start_count"] for i in choices], dtype=np.int64)
            ends = counts.cumsum()
            positions = np.flatnonzero(days == day)
            draws = rng.integers(0, int(ends[-1]), size=len(positions))
            selected = np.searchsorted(ends, draws, side="right")
            contexts[positions] = choices[selected]
            starts[positions] = draws - np.r_[0, ends[:-1]][selected]
        return contexts, starts

    def sample(self, days, rng):
        contexts, starts = self.draw_starts(days, rng)
        crops = np.stack([np.asarray(self.values[self.intervals[context]["cache_start"] + start:
                                                self.intervals[context]["cache_start"] + start + WINDOW_SAMPLES],
                                     dtype=np.float32)
                          for context, start in zip(contexts, starts)])
        return crops - crops.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)

    def describe(self):
        return dict(fingerprint=self.fingerprint, intervals=len(self.intervals),
                    days={str(day): dict(intervals=sum(row["day"] == day for row in self.intervals),
                          legal_starts=sum(row["start_count"] for row in self.intervals if row["day"] == day))
                          for day in sorted({row["day"] for row in self.intervals})})


def negative_crop_pool(dataset, target, split, days, cache_dir):
    """Cache every eligible safe interval once, independently of fixed negative windows."""
    days = tuple(sorted(int(day) for day in days))
    if not set(days).issubset(DAY_SPLITS[split]):
        raise ValueError("Negative crop days crossed the recording-day partition")
    window_rows, _ = sample_pipeline.sample_offsets()
    intervals, total_samples = [], 0
    for filename, regions in sample_pipeline.safe_regions(dataset, target).items():
        source = dataset.sources[filename]
        if source["day"] not in days:
            continue
        for first, last in regions:
            if last - first < window_rows:
                continue
            samples = -(-(last - first) * TARGET_HZ // SOURCE_HZ)
            max_start = (last - first - window_rows) * TARGET_HZ // SOURCE_HZ
            intervals.append(dict(raw_file=filename, day=int(source["day"]), start_row=int(first),
                stop_row=int(last), cache_start=total_samples, cache_stop=total_samples + samples,
                start_count=int(max_start + 1)))
            total_samples += samples
    if not intervals or set(days) != {row["day"] for row in intervals}:
        raise ValueError(f"{target}: missing eligible negative crop intervals on days {days}")
    signature = dict(version=1, source=dataset.signature, mouse=target, split=split, days=days,
        sample_code=file_hash(sample_pipeline.__file__), crop_code=file_hash(__file__),
        sample=sample_pipeline.SAMPLE_CONFIG, crop=sample_pipeline.CROP_CONFIG,
        waveform=WAVEFORM_CONFIG, intervals=intervals)
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    directory = Path(cache_dir) / fingerprint
    values_path, manifest_path = directory / "waveforms.npy", directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint or file_hash(values_path) != manifest["sha256"]:
            raise ValueError("Negative crop pool cache changed")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        temporary = values_path.with_suffix(".npy.tmp")
        values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(total_samples,))
        reader = dataset.reader(target, split)
        for row in intervals:
            write_resampled_interval(reader, row["raw_file"], row["start_row"], row["stop_row"],
                                     values[row["cache_start"]:row["cache_stop"]])
        values.flush()
        del values
        temporary.replace(values_path)
        write_json(manifest_path, dict(fingerprint=fingerprint, signature=signature, sha256=file_hash(values_path)))
    return NegativeCropPool(np.load(values_path, mmap_mode="r"), intervals, fingerprint)


def random_crops(segments, max_offsets, indices, rng):
    """One uniformly placed 112 s crop per selected row, with the crop's own mean removed."""
    indices = np.asarray(indices, dtype=np.int64)
    starts = rng.integers(0, np.asarray(max_offsets)[indices] + 1)
    # Slice row by row so a memory-mapped cache never loads whole segments.
    crops = np.stack([np.asarray(segments[index, start:start + WINDOW_SAMPLES], dtype=np.float32)
                      for index, start in zip(indices, starts)])
    return crops - crops.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
