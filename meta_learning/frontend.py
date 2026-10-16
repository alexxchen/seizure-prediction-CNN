"""A gated, learnable spectral frontend between fixed STFT and the task CNN.

Only centered/resampled STFT magnitudes are cached. The frontend learns
compression, sample-local normalization, global feature modulation, local and
long-range time-frequency corrections, and temporal aggregation. Every learned
stage starts at (or within a small residual of) the log-STFT average-pooling map.
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
FRONTEND_CONFIG = dict(architecture="gated_spectral_frontend", input_frequency_bins=40,
    input_frames=223,
    compression="Per-frequency Box-Cox of (1 + magnitude) in log10 units; lambda=0 is log1p",
    compression_bound=0.5,
    normalization="Per-sample, per-frequency gated centering/scaling over time; gates start at 0",
    normalization_epsilon=1e-4,
    film="Per-sample time mean/std per frequency -> MLP -> per-frequency scale/shift",
    film_hidden=32,
    hidden_channels=16, time_dilations=[1, 2, 4, 8, 16, 32, 64], kernel=[3, 3], activation="gelu",
    temporal_aggregation="Learnable linear frames->time_bins map initialized to adaptive average pooling",
    time_bins=55, frequency_bins=40, head_initialization="zeros", checkpoint_batch_size=32,
    initialization="Seeded; exact log-STFT average-pooling map at step 0 (identity gates, zero heads)")


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
    """Parameter-free log-STFT control on the same native CNN grid as the network."""

    @staticmethod
    def log_map(magnitude):
        return (torch.log1p(magnitude) / math.log(10)).transpose(1, 2).unsqueeze(1)

    @staticmethod
    def cnn_grid(features):
        pooled = F.adaptive_avg_pool2d(features, (FRONTEND_CONFIG["time_bins"], features.shape[-1]))
        return pooled.unsqueeze(1)

    def forward(self, magnitude):
        return self.cnn_grid(self.log_map(magnitude))


def box_cox_log10(magnitude, lam):
    """Box-Cox of (1 + magnitude) in log10 units; lam == 0 is exactly log1p / ln(10)."""
    log_value = torch.log1p(magnitude)
    small = lam.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(lam), lam)
    exact = torch.expm1(lam * log_value) / safe
    series = log_value + 0.5 * lam * log_value.square()
    return torch.where(small, series, exact) / math.log(10)


class SpectralFrontend(nn.Module):
    """Gated compression, normalization, FiLM, residual conv network, and time aggregation."""

    def __init__(self, seed=0):
        super().__init__()
        c = FRONTEND_CONFIG
        frequencies = torch.fft.rfftfreq(SPECTRUM_CONFIG["frame_length"], d=1 / 256)
        self.register_buffer("frequencies", frequencies[(frequencies >= 1) & (frequencies <= 40)])
        bins, frames, hidden = c["input_frequency_bins"], c["input_frames"], c["hidden_channels"]
        # Initialize CPU layers without advancing caller CPU/CUDA RNG streams,
        # keeping the reference task CNN unchanged when the frontend is replaced.
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            torch.set_rng_state(generator.get_state())
            # (1) Compression exponent per frequency; zero is the log1p map.
            self.compression_raw = nn.Parameter(torch.zeros(bins))
            # (2) Sample-local centering/scaling gates; zero preserves absolute amplitude.
            self.center_gate = nn.Parameter(torch.zeros(bins))
            self.scale_gate = nn.Parameter(torch.zeros(bins))
            # (4) Global per-sample statistics modulate each frequency.
            self.film = nn.Sequential(nn.Linear(2 * bins, c["film_hidden"]), nn.GELU(),
                                      nn.Linear(c["film_hidden"], 2 * bins))
            # (3) Local frequency and long-range dilated time convolutions.
            self.stem = nn.Conv2d(1, hidden, 3, padding=1, padding_mode="replicate")
            self.blocks = nn.ModuleList(
                nn.Conv2d(hidden, hidden, 3, padding=(dilation, 1), dilation=(dilation, 1),
                          padding_mode="replicate")
                for dilation in c["time_dilations"])
            self.head = nn.Conv2d(hidden, 1, 1)
            # Zero heads make the initial map exactly log-STFT. Head weights receive
            # gradients immediately; inner FiLM/conv layers follow once heads move.
            for layer in (self.film[-1], self.head):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        # (5) Linear temporal aggregation starts as adaptive average pooling.
        reference = F.adaptive_avg_pool1d(torch.eye(frames).unsqueeze(0), c["time_bins"])[0].T
        self.register_buffer("aggregation_reference", reference.contiguous())
        self.aggregation_delta = nn.Parameter(torch.zeros_like(reference))

    def compression_exponent(self):
        return FRONTEND_CONFIG["compression_bound"] * torch.tanh(self.compression_raw)

    def describe(self):
        with torch.no_grad():
            exponent = self.compression_exponent()
            gates = dict(compression_exponent=exponent, center_gate=self.center_gate,
                         scale_gate=self.scale_gate)
            summary = {name: dict(mean=float(value.mean()), min=float(value.min()), max=float(value.max()))
                       for name, value in gates.items()}
            summary["aggregation_delta_norm"] = float(self.aggregation_delta.norm())
        return dict(FRONTEND_CONFIG, parameters=sum(parameter.numel() for parameter in self.parameters()),
                    state=summary)

    def _forward_batch(self, magnitude):
        c = FRONTEND_CONFIG
        # (1) Magnitudes are nonnegative by construction; no hard floor is needed.
        compressed = box_cox_log10(magnitude.transpose(1, 2).unsqueeze(1), self.compression_exponent())
        mean = compressed.mean(dim=2, keepdim=True)
        std = (compressed.var(dim=2, unbiased=False, keepdim=True) + c["normalization_epsilon"]).sqrt()
        # (2) Per-sample statistics only; features never depend on batch companions.
        normalized = (compressed - self.center_gate * mean) * torch.exp(-self.scale_gate * std.log())
        # (4) FiLM from statistics of the compressed, unnormalized map.
        scale, shift = self.film(torch.cat([mean.flatten(1), std.flatten(1)], dim=1)).chunk(2, dim=1)
        features = normalized * (1 + scale[:, None, None]) + shift[:, None, None]
        # (3) Residual network; time dilations reach the full 112-second window.
        hidden = self.stem(features)
        for block in self.blocks:
            hidden = hidden + block(F.gelu(hidden))
        features = features + self.head(F.gelu(hidden))
        # (5) Learned linear map over time; frequencies stay on their native 1 Hz grid.
        aggregation = self.aggregation_reference + self.aggregation_delta
        return torch.einsum("ot,nctf->ncof", aggregation, features).unsqueeze(1)

    def forward(self, magnitude):
        expected = (FRONTEND_CONFIG["input_frequency_bins"], FRONTEND_CONFIG["input_frames"])
        if tuple(magnitude.shape[1:]) != expected:
            raise ValueError(f"Expected STFT magnitudes shaped (N, {expected[0]}, {expected[1]})")
        outputs = []
        differentiable = torch.is_grad_enabled() and (
            magnitude.requires_grad or any(parameter.requires_grad for parameter in self.parameters()))
        for batch in magnitude.split(FRONTEND_CONFIG["checkpoint_batch_size"]):
            # Non-reentrant checkpointing also supports autograd.grad and inputs
            # without requires_grad. It recomputes all neural activations exactly.
            outputs.append(checkpoint(self._forward_batch, batch, use_reentrant=False)
                           if differentiable else self._forward_batch(batch))
        return torch.cat(outputs)
