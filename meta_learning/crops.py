"""Context segments for training-time random crops, preprocessed like fixed windows.

Each sample row gets the source segment covering every allowed crop start
(`sample_pipeline.crop_ranges`): its 7-3 minute positive interval, or its
seizure-free negative neighbourhood. Segments have their mean removed and are
resampled to 256 Hz once; a crop then takes 112 s from a segment and removes the
crop's own mean, matching `waveform.preprocess_batch` up to resampling edges.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

import sample_pipeline
from source_data import SOURCE_HZ, file_hash
from meta_learning.common import write_json
from meta_learning.waveform import WAVEFORM_CONFIG

TARGET_HZ = WAVEFORM_CONFIG["target_hz"]
WINDOW_SAMPLES = int(round(sample_pipeline.SAMPLE_CONFIG["window_seconds"] * TARGET_HZ))
SEGMENT_SAMPLES = int(round((sample_pipeline.SAMPLE_CONFIG["positive_earlier_minutes"] -
                             sample_pipeline.SAMPLE_CONFIG["positive_later_minutes"]) * 60 * TARGET_HZ))


def crop_segments(dataset, target, split, rows, cache_dir):
    """(segments [rows, SEGMENT_SAMPLES], max_offsets [rows]) at 256 Hz; cached by content."""
    ranges = sample_pipeline.crop_ranges(dataset, target, rows)
    window_rows, _ = sample_pipeline.sample_offsets()
    signature = dict(source=dataset.signature, sample_code=file_hash(sample_pipeline.__file__),
                     crop_code=file_hash(__file__), waveform=WAVEFORM_CONFIG, crop=sample_pipeline.CROP_CONFIG,
                     sample=sample_pipeline.SAMPLE_CONFIG,
                     rows_sha256=hashlib.sha256(rows.to_csv(index=False).encode()).hexdigest(),
                     ranges_sha256=hashlib.sha256(ranges.tobytes()).hexdigest())
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / key
    segments_path, offsets_path, manifest = cache / "segments.npy", cache / "max_offsets.npy", cache / "manifest.json"
    if manifest.exists():
        stored = json.loads(manifest.read_text())
        if stored["signature"] != signature or any(file_hash(cache / name) != digest
                                                   for name, digest in stored["files"].items()):
            raise ValueError("Crop segment cache changed")
        return np.load(segments_path, mmap_mode="r"), np.load(offsets_path)
    cache.mkdir(parents=True, exist_ok=True)
    segments = np.lib.format.open_memmap(str(segments_path) + ".tmp", mode="w+", dtype=np.float32,
                                         shape=(len(rows), SEGMENT_SAMPLES))
    max_offsets = np.empty(len(rows), dtype=np.int64)
    reader = dataset.reader(target, split)
    for index, (row, (first, last)) in enumerate(zip(rows.itertuples(), ranges)):
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
    write_json(manifest, dict(signature=signature, files={path.name: file_hash(path)
                                                         for path in [segments_path, offsets_path]}))
    return np.load(segments_path, mmap_mode="r"), np.load(offsets_path)


def random_crops(segments, max_offsets, indices, rng):
    """One uniformly placed 112 s crop per selected row, with the crop's own mean removed."""
    indices = np.asarray(indices, dtype=np.int64)
    starts = rng.integers(0, np.asarray(max_offsets)[indices] + 1)
    # Slice row by row so a memory-mapped cache never loads whole segments.
    crops = np.stack([np.asarray(segments[index, start:start + WINDOW_SAMPLES], dtype=np.float32)
                      for index, start in zip(indices, starts)])
    return crops - crops.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
