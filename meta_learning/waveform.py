"""Fixed, sample-local waveform preprocessing and its cache; no spectral features.

Each window has its own mean removed and is resampled from 500 to 256 Hz. Only
these deterministic waveforms are cached; every spectral or learned transform
runs inside the frontend. The cache key covers this file only, so editing the
trainable frontend does not invalidate cached waveforms.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from source_data import file_hash
from meta_learning.common import write_json


WAVEFORM_CONFIG = dict(source_hz=500, target_hz=256, centering="per-window mean", resampling="resample_poly(256, 500)",
                       end_padding="zeros to the next multiple of 256 samples", dtype="float32")


def preprocess_batch(raw):
    """Center each window and resample it; amplitude is otherwise unchanged."""
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite (windows, waveform_samples) inputs")
    values = values - values.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    values = resample_poly(values, WAVEFORM_CONFIG["target_hz"], WAVEFORM_CONFIG["source_hz"], axis=1)
    values = np.pad(values, ((0, 0), (0, (-values.shape[1]) % WAVEFORM_CONFIG["target_hz"])))
    return values.astype(np.float32)


def cached_waveforms(raw, cache):
    """Cache only deterministic preprocessing; never cache outputs of trainable operations."""
    raw_hash = file_hash(raw.filename) if hasattr(raw, "filename") else hashlib.sha256(raw.tobytes()).hexdigest()
    identity = dict(raw_sha256=raw_hash, source_sha256=file_hash(__file__), config=WAVEFORM_CONFIG,
                    runtime={name: importlib.metadata.version(name) for name in ["numpy", "scipy"]})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path, manifest_path = cache / f"{key}.npy", cache / f"{key}.json"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        if stored["identity"] != identity or file_hash(path) != stored["sha256"]:
            raise ValueError("Waveform cache changed")
        return np.load(path, mmap_mode="r", allow_pickle=False)
    first = preprocess_batch(raw[:1])
    temporary = path.with_suffix(".npy.tmp")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(raw), *first.shape[1:]))
    for start in range(0, len(raw), 32):
        values = preprocess_batch(raw[start:start + 32])
        if not np.isfinite(values).all():
            raise ValueError("Non-finite preprocessed waveform")
        output[start:start + len(values)] = values
    output.flush()
    del output
    temporary.replace(path)
    write_json(manifest_path, dict(identity=identity, sha256=file_hash(path)))
    return np.load(path, mmap_mode="r", allow_pickle=False)
