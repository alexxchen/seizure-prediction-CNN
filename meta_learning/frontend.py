"""A convolutional waveform frontend between preprocessed LFP windows and the task CNN.

No spectrum is precomputed. A learnable 1-D convolution filterbank, initialized to
cosine-windowed cosine/sine pairs, forms a log-magnitude time-frequency map inside
the network. Two zero-initialized residual branches refine that map: a 2-D conv +
MLP over the map and a strided 1-D conv stack over the raw waveform. At step 0 the
output equals the log-STFT baseline, which uses the same filterbank frozen.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from meta_learning.waveform import WAVEFORM_CONFIG


FILTERBANK_CONFIG = dict(sample_hz=256, frame_length=256, hop_length=128, frequency_hz=[1, 40],
    initialization="Cosine-window (scipy 'cosine', symmetric) cos/sin pairs: an unnormalized 256-point DFT",
    magnitude="sqrt(real^2 + imag^2 + 1e-6)", compression="log1p / ln(10)", learnable=True)
FRONTEND_CONFIG = dict(architecture="waveform_conv", input_samples=28672, input_frames=223,
    input_frequency_bins=40,
    filterbank="Conv1d 1->80 (40 cos + 40 sin), kernel 256, stride 128, no bias; excluded from weight decay",
    map_branch="Conv2d 1->16 3x3, two Conv2d 16->16 3x3 with time stride 2 (223->56); per time bin "
               "LayerNorm(640) -> Linear(640->64) -> GELU -> Linear(64->40), zero-initialized",
    waveform_branch="Conv1d 1->32 k10 s4 + GroupNorm(1) -> 32 k8 s4 -> 64 k8 s4 -> 64 k4 s2 -> 64 k4 s4 "
                    "(28672->56 samples, GELU); per bin LayerNorm(64) -> Linear(64->64) -> GELU -> "
                    "Linear(64->40), zero-initialized",
    residual_base="log-magnitude map average-pooled to the output grid",
    time_bins=56, frequency_bins=40, checkpoint_batch_size=32,
    initialization="Seeded; exact log-STFT average-pooling map at step 0 (DFT filterbank, zero output layers)")


def filterbank_reference():
    """Windowed DFT rows for 1-40 Hz; magnitude equals |STFT| of the cosine-windowed frame."""
    length, rate = FILTERBANK_CONFIG["frame_length"], FILTERBANK_CONFIG["sample_hz"]
    low, high = FILTERBANK_CONFIG["frequency_hz"]
    n = torch.arange(length, dtype=torch.float64)
    window = torch.sin(math.pi * (n + 0.5) / length)
    frequencies = torch.arange(low, high + 1, dtype=torch.float64) * length / rate
    phase = 2 * math.pi * frequencies[:, None] * n[None] / length
    weights = torch.cat([window * torch.cos(phase), -window * torch.sin(phase)])
    return weights.unsqueeze(1).float()


def log_magnitude_map(waveform, filterbank):
    """(N, samples) -> (N, 1, frames, frequencies) log10(1 + |filter response|)."""
    response = F.conv1d(waveform.unsqueeze(1), filterbank, stride=FILTERBANK_CONFIG["hop_length"])
    real, imag = response.chunk(2, dim=1)
    magnitude = torch.sqrt(real.square() + imag.square() + 1e-6)
    return (torch.log1p(magnitude) / math.log(10)).transpose(1, 2).unsqueeze(1)


def cnn_grid(features):
    pooled = F.adaptive_avg_pool2d(features, (FRONTEND_CONFIG["time_bins"], features.shape[-1]))
    return pooled.unsqueeze(1)


def check_waveform(waveform):
    if waveform.ndim != 2 or waveform.shape[1] != FRONTEND_CONFIG["input_samples"]:
        raise ValueError(f"Expected preprocessed waveforms shaped (N, {FRONTEND_CONFIG['input_samples']}) "
                         f"at {WAVEFORM_CONFIG['target_hz']} Hz")


class LogSTFTFrontend(nn.Module):
    """Parameter-free control: the frozen DFT filterbank on the same native CNN grid."""

    def __init__(self):
        super().__init__()
        self.register_buffer("filterbank", filterbank_reference())

    def forward(self, waveform):
        check_waveform(waveform)
        return cnn_grid(log_magnitude_map(waveform, self.filterbank))


class WaveformFrontend(nn.Module):
    """Learnable filterbank map plus zero-initialized map and raw-waveform conv residuals."""

    def __init__(self, seed=0):
        super().__init__()
        bins = FRONTEND_CONFIG["input_frequency_bins"]
        reference = filterbank_reference()
        self.register_buffer("filterbank_reference", reference.clone())
        self.filterbank = nn.Parameter(reference)
        # Initialize CPU layers without advancing caller CPU/CUDA RNG streams,
        # keeping the reference task CNN unchanged when the frontend is replaced.
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            torch.set_rng_state(generator.get_state())
            self.map_conv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(),
                                          nn.Conv2d(16, 16, 3, stride=(2, 1), padding=1), nn.GELU(),
                                          nn.Conv2d(16, 16, 3, stride=(2, 1), padding=1), nn.GELU())
            self.map_mlp = nn.Sequential(nn.LayerNorm(16 * bins), nn.Linear(16 * bins, 64), nn.GELU(),
                                         nn.Linear(64, bins))
            self.waveform_conv = nn.Sequential(
                nn.Conv1d(1, 32, 10, stride=4, padding=3), nn.GroupNorm(1, 32), nn.GELU(),
                nn.Conv1d(32, 32, 8, stride=4, padding=2), nn.GELU(),
                nn.Conv1d(32, 64, 8, stride=4, padding=2), nn.GELU(),
                nn.Conv1d(64, 64, 4, stride=2, padding=1), nn.GELU(),
                nn.Conv1d(64, 64, 4, stride=4), nn.GELU())
            self.waveform_mlp = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, bins))
            # Zero output layers: the initial map is exactly the pooled log-STFT baseline.
            for layer in (self.map_mlp[-1], self.waveform_mlp[-1]):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def decay_parameters(self):
        """Weight matrices that receive decoupled weight decay (not the filterbank, biases, or norms)."""
        return [parameter for name, parameter in self.named_parameters()
                if parameter.ndim > 1 and name != "filterbank"]

    def describe(self):
        with torch.no_grad():
            state = {name: float(parameter.norm()) for name, parameter in self.named_parameters()
                     if name.endswith("weight")}
            drift = float((self.filterbank - self.filterbank_reference).norm() / self.filterbank_reference.norm())
        return dict(FRONTEND_CONFIG, filterbank_config=FILTERBANK_CONFIG,
                    parameters=sum(parameter.numel() for parameter in self.parameters()),
                    state=dict(weight_norms=state, filterbank_relative_drift=drift))

    def _forward_batch(self, waveform):
        spectral = log_magnitude_map(waveform, self.filterbank)
        hidden = self.map_conv(spectral)
        waveform_hidden = self.waveform_conv(waveform.unsqueeze(1))
        bins = FRONTEND_CONFIG["time_bins"]
        if hidden.shape[2] != bins or waveform_hidden.shape[2] != bins:
            raise ValueError("Convolution strides do not produce the configured time bins")
        # One token per output time bin from each branch.
        map_residual = self.map_mlp(hidden.permute(0, 2, 1, 3).flatten(2))
        waveform_residual = self.waveform_mlp(waveform_hidden.transpose(1, 2))
        return (cnn_grid(spectral).squeeze(1) + (map_residual + waveform_residual).unsqueeze(1)).unsqueeze(1)

    def forward(self, waveform):
        check_waveform(waveform)
        outputs = []
        differentiable = torch.is_grad_enabled() and (
            waveform.requires_grad or any(parameter.requires_grad for parameter in self.parameters()))
        for batch in waveform.split(FRONTEND_CONFIG["checkpoint_batch_size"]):
            # Non-reentrant checkpointing also supports autograd.grad and inputs
            # without requires_grad. It recomputes all neural activations exactly.
            outputs.append(checkpoint(self._forward_batch, batch, use_reentrant=False)
                           if differentiable else self._forward_batch(batch))
        return torch.cat(outputs)
