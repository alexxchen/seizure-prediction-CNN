"""A generic convolution + MLP frontend between fixed STFT and the task CNN.

Only centered/resampled STFT magnitudes are cached. A strided time-frequency
convolution trunk and a per-time-bin MLP learn a residual correction to the
average-pooled log-STFT map. The zero-initialized output layer makes step 0
equal the log-STFT baseline.
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
FRONTEND_CONFIG = dict(architecture="conv_mlp_residual", input_frequency_bins=40,
    input_frames=223, input_transform="log1p_base10",
    conv="Conv2d 1->16 3x3, then two Conv2d 16->16 3x3 with time stride 2 (223->112->56 frames); GELU",
    conv_channels=16, time_strides=[2, 2],
    mlp="Per time bin: LayerNorm(16x40) -> Linear(640->64) -> GELU -> Linear(64->40), zero-initialized",
    mlp_hidden=64, residual_base="adaptive average pooling of the log-STFT map to the output grid",
    time_bins=56, frequency_bins=40, checkpoint_batch_size=32,
    initialization="Seeded; zero output layer, so step 0 equals the log-STFT average-pooling map")


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


class SpectralFrontend(nn.Module):
    """Strided conv trunk plus per-time-bin MLP, added to the pooled log-STFT map."""

    def __init__(self, seed=0):
        super().__init__()
        c = FRONTEND_CONFIG
        frequencies = torch.fft.rfftfreq(SPECTRUM_CONFIG["frame_length"], d=1 / 256)
        self.register_buffer("frequencies", frequencies[(frequencies >= 1) & (frequencies <= 40)])
        channels, bins = c["conv_channels"], c["input_frequency_bins"]
        # Initialize CPU layers without advancing caller CPU/CUDA RNG streams,
        # keeping the reference task CNN unchanged when the frontend is replaced.
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            torch.set_rng_state(generator.get_state())
            layers = [nn.Conv2d(1, channels, 3, padding=1), nn.GELU()]
            for stride in c["time_strides"]:
                layers += [nn.Conv2d(channels, channels, 3, stride=(stride, 1), padding=1), nn.GELU()]
            self.conv = nn.Sequential(*layers)
            self.mlp = nn.Sequential(nn.LayerNorm(channels * bins), nn.Linear(channels * bins, c["mlp_hidden"]),
                                     nn.GELU(), nn.Linear(c["mlp_hidden"], bins))
            # Zero output: the initial map is exactly the pooled log-STFT baseline.
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def describe(self):
        with torch.no_grad():
            state = {name: float(parameter.norm()) for name, parameter in self.named_parameters()
                     if name.endswith("weight")}
        return dict(FRONTEND_CONFIG, parameters=sum(parameter.numel() for parameter in self.parameters()),
                    state=dict(weight_norms=state))

    def _forward_batch(self, magnitude):
        base = LogSTFTFrontend.log_map(magnitude)
        hidden = self.conv(base)
        if hidden.shape[2] != FRONTEND_CONFIG["time_bins"]:
            raise ValueError("Convolution strides do not produce the configured time bins")
        # One token per output time bin: all channels and frequencies of that bin.
        tokens = hidden.permute(0, 2, 1, 3).flatten(2)
        residual = self.mlp(tokens).unsqueeze(1)
        return (LogSTFTFrontend.cnn_grid(base).squeeze(1) + residual).unsqueeze(1)

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
