"""Editable input pipeline for the fixed MouseLFP CNN.

Transform each candidate window into (1, 1, 55, 114) features.
Fit population statistics on training windows only.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import stft
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

PREPROCESS = dict(target_hz=256, frame_length=256,
                  normalization="none", epsilon=1e-6,
                  layout="time_frequency", frequency_hz=[1, 40], row_interpolation="nearest",
                  temporal_pooling="half_medians", waveform_centering="window_mean")


def center_waveform(signal):
    """Remove this window's constant offset while preserving its AC amplitude."""
    signal = np.asarray(signal, dtype=np.float32)
    return signal - np.float32(signal.mean(dtype=np.float64))


def preprocess_window(signal):
    """Use the same signal representation for every split's candidate windows."""
    spectrum = stft.spectrogram(
        resample_poly(center_waveform(signal), PREPROCESS["target_hz"], 500)[:, None],
        framelength=PREPROCESS["frame_length"], centered=False)
    if spectrum.ndim == 2:
        spectrum = spectrum[:, :, None]
    values = np.abs(np.transpose(spectrum, (2, 1, 0))) + PREPROCESS["epsilon"]
    frequencies = np.fft.rfftfreq(PREPROCESS["frame_length"],
                                 d=1 / PREPROCESS["target_hz"])
    low, high = PREPROCESS["frequency_hz"]
    values = values[:, :, (frequencies >= low) & (frequencies <= high)]
    matrix = np.maximum(np.log10(values), 0)[0].astype(np.float32)
    # Retain ordered spectral evolution while pooling transient frames.
    matrix = np.stack([np.median(half, axis=0)
                       for half in np.array_split(matrix, 2, axis=0)])
    return arrange_input(matrix)


def arrange_input(matrix):
    """Map a two-dimensional representation onto the fixed CNN input grid."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite, nonempty representation matrix")
    layout = PREPROCESS["layout"]
    if layout == "frequency_time":
        matrix = matrix.T
    elif layout != "time_frequency":
        raise ValueError("Unknown input layout")
    tensor = torch.from_numpy(np.ascontiguousarray(matrix))[None, None]
    if matrix.shape != (55, 114):
        # Keep measured temporal profiles constant; interpolate frequency only.
        tensor = torch.nn.functional.interpolate(tensor, size=(matrix.shape[0], 114),
                                                  mode="bilinear", align_corners=False)
        tensor = torch.nn.functional.interpolate(tensor, size=(55, 114), mode="nearest")
    return tensor.numpy()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def features(raw, cache):
    """Cache only deterministic, sample-local transforms under their source hash."""
    # A source fingerprint prevents stale features after arbitrary code edits.
    source_hash = file_hash(__file__)
    input_hash = file_hash(raw.filename) if hasattr(raw, "filename") else hashlib.sha256(raw.tobytes()).hexdigest()
    runtime = json.dumps({name: importlib.metadata.version(name)
                          for name in ["numpy", "scipy", "stft"]}, sort_keys=True)
    config = json.dumps(PREPROCESS, sort_keys=True)
    key = hashlib.sha256((source_hash + input_hash + runtime + config).encode()).hexdigest()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{key}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r", allow_pickle=False)
    first = preprocess_window(np.asarray(raw[0]))
    result = np.lib.format.open_memmap(str(path) + ".tmp", mode="w+",
                                       dtype=np.float32, shape=(len(raw), *first.shape))
    for index, signal in enumerate(raw):
        sample = first if index == 0 else preprocess_window(np.asarray(signal))
        if sample.shape != first.shape or not np.isfinite(sample).all():
            raise ValueError("Invalid preprocessed window")
        result[index] = sample
    result.flush()
    del result
    Path(str(path) + ".tmp").replace(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def normalization_stats(train_features):
    mode = PREPROCESS["normalization"]
    if mode == "none":
        return np.array(0, dtype=np.float32), np.array(1, dtype=np.float32)
    if mode != "train_frequency_zscore":
        raise ValueError("Unknown normalization recipe")
    # Fit per-frequency statistics exclusively on training windows.
    total = np.zeros(train_features.shape[-1], dtype=np.float64)
    squares = np.zeros_like(total)
    count = 0
    for start in range(0, len(train_features), 256):
        batch = np.asarray(train_features[start:start + 256], dtype=np.float64)
        axes = tuple(range(batch.ndim - 1))
        total += batch.sum(axis=axes)
        squares += np.square(batch).sum(axis=axes)
        count += int(np.prod(batch.shape[:-1]))
    mean = total / count
    scale = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
    return mean.astype(np.float32), np.maximum(scale, PREPROCESS["epsilon"]).astype(np.float32)


class FeatureDataset(Dataset):
    def __init__(self, values, labels, mean, scale):
        self.values, self.labels, self.mean, self.scale = values, labels, mean, scale

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        sample = np.array(self.values[index], dtype=np.float32, copy=True)
        tensor = torch.from_numpy((sample - self.mean) / self.scale)
        return tensor if self.labels is None else (tensor, int(self.labels[index]))
