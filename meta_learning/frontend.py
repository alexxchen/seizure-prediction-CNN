"""A smooth residual convolutional network between fixed STFT and the task CNN.

Only centered/resampled STFT magnitudes are cached. The neural frontend retains
the time-frequency map and learns local corrections without rank selection.
"""

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.fftpack import fft
from scipy.signal import resample_poly
from scipy.signal.windows import cosine
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from source_data import file_hash
from meta_learning.common import write_json


SPECTRUM_CONFIG = dict(source_hz=500, target_hz=256, frame_length=256,
                       hop_length=128, frequency_hz=[1, 40], center=False,
                       window="cosine_symmetric", end_padding="next_frame_multiple",
                       representation="unnormalized_magnitude")
FRONTEND_CONFIG = dict(architecture="residual_conv2d", input_frequency_bins=40,
    hidden_channels=16, kernels=[3, 3, 1], activation="gelu", input_transform="log1p_base10",
    residual_scale=0.1, output_init_std=1e-3, temporal_pooling="adaptive_average",
    time_bins=55, frequency_bins=114, interpolation="bilinear", normalization="none",
    checkpoint_batch_size=32,
    initialization="Seeded neural weights; small residual around full log-STFT map")


def fixed_spectrum_batch(raw):
    """Reproduce stft 0.5.2's selected magnitude basis using pinned SciPy only."""
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite (windows, waveform_samples) inputs")
    values = values - values.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    values = resample_poly(values, 256, 500, axis=1)
    length, hop = SPECTRUM_CONFIG["frame_length"], SPECTRUM_CONFIG["hop_length"]
    values = np.pad(values, ((0, 0), (0, (-values.shape[1]) % length)))
    frames = np.lib.stride_tricks.sliding_window_view(values, length, axis=1)[:, ::hop]
    spectrum = fft(frames * cosine(length), axis=-1)[..., :length // 2 + 1]
    frequencies = np.fft.rfftfreq(length, d=1 / SPECTRUM_CONFIG["target_hz"])
    low, high = SPECTRUM_CONFIG["frequency_hz"]
    magnitude = np.abs(spectrum[..., (frequencies >= low) & (frequencies <= high)])
    return magnitude.transpose(0, 2, 1).astype(np.float32)


def fixed_spectrum(raw, cache):
    """Cache the fixed basis; never cache outputs of trainable operations."""
    raw_hash = file_hash(raw.filename) if hasattr(raw, "filename") else hashlib.sha256(raw.tobytes()).hexdigest()
    identity = dict(raw_sha256=raw_hash, source_sha256=file_hash(__file__),
                    config=SPECTRUM_CONFIG,
                    runtime={name: importlib.metadata.version(name) for name in ["torch", "numpy", "scipy"]})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{key}.npy"
    manifest_path = cache / f"{key}.json"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        if stored["identity"] != identity or file_hash(path) != stored["sha256"]:
            raise ValueError("Fixed spectrum cache changed")
        return np.load(path, mmap_mode="r", allow_pickle=False)
    first = fixed_spectrum_batch(raw[:1])
    temporary = path.with_suffix(".npy.tmp")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                       shape=(len(raw), *first.shape[1:]))
    for start in range(0, len(raw), 32):
        values = fixed_spectrum_batch(raw[start:start + 32])
        if not np.isfinite(values).all():
            raise ValueError("Non-finite fixed spectrum")
        output[start:start + len(values)] = values
    output.flush()
    del output
    temporary.replace(path)
    write_json(manifest_path, dict(identity=identity, sha256=file_hash(path)))
    return np.load(path, mmap_mode="r", allow_pickle=False)


class LogSTFTFrontend(nn.Module):
    """Parameter-free log-STFT control using the same CNN grid as the network."""

    @staticmethod
    def log_map(magnitude):
        return (torch.log1p(magnitude) / math.log(10)).transpose(1, 2).unsqueeze(1)

    @staticmethod
    def cnn_grid(features):
        pooled = F.adaptive_avg_pool2d(features, (FRONTEND_CONFIG["time_bins"], features.shape[-1]))
        return F.interpolate(pooled, size=(FRONTEND_CONFIG["time_bins"], FRONTEND_CONFIG["frequency_bins"]),
                             mode="bilinear", align_corners=False).unsqueeze(1)

    def forward(self, magnitude):
        return self.cnn_grid(self.log_map(magnitude))


class SpectralFrontend(LogSTFTFrontend):
    """Learn a dense time-frequency map with smooth neural residual corrections."""

    def __init__(self, seed=0):
        super().__init__()
        c = FRONTEND_CONFIG
        frequencies = torch.fft.rfftfreq(SPECTRUM_CONFIG["frame_length"], d=1 / 256)
        self.register_buffer("frequencies", frequencies[(frequencies >= 1) & (frequencies <= 40)])
        # Initialize CPU layers without advancing caller CPU/CUDA RNG streams,
        # keeping the reference task CNN unchanged when the frontend is replaced.
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            torch.set_rng_state(generator.get_state())
            hidden = c["hidden_channels"]
            self.network = nn.Sequential(nn.Conv2d(1, hidden, 3, padding=1), nn.GELU(),
                                         nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                                         nn.Conv2d(hidden, 1, 1))
            # A small nonzero head lets every layer receive gradients from step 1.
            nn.init.normal_(self.network[-1].weight, std=c["output_init_std"])
            nn.init.zeros_(self.network[-1].bias)

    def describe(self):
        return dict(FRONTEND_CONFIG, parameters=sum(parameter.numel() for parameter in self.parameters()))

    def _forward_batch(self, magnitude):
        # Magnitudes are nonnegative by construction; no hard floor is needed.
        base = self.log_map(magnitude)
        features = base + FRONTEND_CONFIG["residual_scale"] * self.network(base)
        return self.cnn_grid(features)

    def forward(self, magnitude):
        outputs = []
        differentiable = torch.is_grad_enabled() and (
            magnitude.requires_grad or any(parameter.requires_grad for parameter in self.parameters()))
        for batch in magnitude.split(FRONTEND_CONFIG["checkpoint_batch_size"]):
            # Non-reentrant checkpointing also supports autograd.grad and inputs
            # without requires_grad. It recomputes all neural activations exactly.
            outputs.append(checkpoint(self._forward_batch, batch, use_reentrant=False)
                           if differentiable else self._forward_batch(batch))
        return torch.cat(outputs)
