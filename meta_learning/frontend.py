"""A single, randomly initialized convolutional network from waveform to the task CNN.

No spectrum is precomputed and no stage is hand-designed: a pre-activation 1-D
ResNet with strided downsampling maps each preprocessed waveform to a
(channels, time) feature sequence for the 1-D task CNN. The parameter-free log-STFT control keeps
a frozen cosine-window DFT filterbank on the same cached waveforms.
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
FRONTEND_CONFIG = dict(architecture="waveform_resnet1d", input_samples=28672,
    input_normalization="Per-window RMS after existing waveform mean removal; no fitted statistics",
    rms_numerics="Peak-scaled RMS computation; dtype-tiny floor and zero output for silence",
    stem="Conv1d(1->16, k8, s4)", stages=[[16, 4], [32, 4], [48, 4], [64, 2], [64, 1]], norm_groups=8,
    stage_note="Each [channels, stride]: one residual block at that width, then a strided Conv1d to the next "
               "width (28672 -> 7168 -> 1792 -> 448 -> 112 -> 56); the last stage adds a dilation-2 block",
    block="Pre-activation: GroupNorm(8) -> GELU -> Conv1d k3 (dilation d) -> GroupNorm(8) -> GELU -> Conv1d k3 "
          "(zero-initialized), added to the residual stream",
    head="GroupNorm(8) -> GELU -> Conv1d(64 -> output_channels, k1); output is a (channels, time) sequence",
    output_channels=64, time_bins=56,
    initialization="Seeded PyTorch defaults (random); zero-initialized last conv in each residual block; "
                   "no alignment with log-STFT")


def rms_normalize_waveform(waveform):
    """Return unit-RMS waveforms using only each centered window's own samples."""
    tiny = torch.finfo(waveform.dtype).tiny
    magnitude = waveform.detach().abs().amax(dim=1, keepdim=True)
    peak = magnitude.clamp_min(tiny)
    # Peak scaling avoids overflow/underflow when recording units differ.
    relative = waveform / peak
    relative_rms = relative.square().mean(dim=1, keepdim=True).clamp_min(tiny).sqrt()
    rms = (peak * relative_rms).clamp_min(tiny)
    return torch.where(magnitude > 0, waveform / rms, torch.zeros_like(waveform))


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


def log_magnitude_sequence(waveform, filterbank):
    """(N, samples) -> (N, frequencies, frames) log10(1 + |filter response|)."""
    response = F.conv1d(waveform.unsqueeze(1), filterbank, stride=FILTERBANK_CONFIG["hop_length"])
    real, imag = response.chunk(2, dim=1)
    magnitude = torch.sqrt(real.square() + imag.square() + 1e-6)
    return torch.log1p(magnitude) / math.log(10)


def check_waveform(waveform):
    if waveform.ndim != 2 or waveform.shape[1] != FRONTEND_CONFIG["input_samples"]:
        raise ValueError(f"Expected preprocessed waveforms shaped (N, {FRONTEND_CONFIG['input_samples']}) "
                         f"at {WAVEFORM_CONFIG['target_hz']} Hz")


class LogSTFTFrontend(nn.Module):
    """Parameter-free control: 40 frozen DFT log-magnitude bands as channels, average-pooled to 56 bins."""

    output_shape = (40, FRONTEND_CONFIG["time_bins"])

    def __init__(self):
        super().__init__()
        self.register_buffer("filterbank", filterbank_reference())

    def forward(self, waveform):
        check_waveform(waveform)
        return F.adaptive_avg_pool1d(log_magnitude_sequence(waveform, self.filterbank), FRONTEND_CONFIG["time_bins"])


class ResidualBlock1d(nn.Module):
    """Pre-activation residual block; the zero-initialized last conv starts it as the identity."""

    def __init__(self, channels, dilation=1):
        super().__init__()
        groups = FRONTEND_CONFIG["norm_groups"]
        self.branch = nn.Sequential(nn.GroupNorm(groups, channels), nn.GELU(),
                                    nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                                    nn.GroupNorm(groups, channels), nn.GELU(),
                                    nn.Conv1d(channels, channels, 3, padding=1))
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self, values):
        return values + self.branch(values)


class WaveformFrontend(nn.Module):
    """One generic 1-D ResNet from waveform to a (channels, time) feature sequence."""

    output_shape = (FRONTEND_CONFIG["output_channels"], FRONTEND_CONFIG["time_bins"])

    def __init__(self, seed=0, batch_size=32):
        super().__init__()
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Frontend batch size must be a positive integer")
        self.batch_size = batch_size
        c = FRONTEND_CONFIG
        # Initialize CPU layers without advancing caller CPU/CUDA RNG streams,
        # keeping the reference task CNN unchanged when the frontend is replaced.
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            torch.set_rng_state(generator.get_state())
            layers = [nn.Conv1d(1, c["stages"][0][0], 8, stride=4, padding=2)]
            for index, (channels, stride) in enumerate(c["stages"]):
                layers.append(ResidualBlock1d(channels))
                if index + 1 < len(c["stages"]):
                    layers.append(nn.Conv1d(channels, c["stages"][index + 1][0], stride, stride=stride))
            layers.append(ResidualBlock1d(c["stages"][-1][0], dilation=2))
            self.body = nn.Sequential(*layers)
            width = c["stages"][-1][0]
            self.head = nn.Sequential(nn.GroupNorm(c["norm_groups"], width), nn.GELU(),
                                      nn.Conv1d(width, c["output_channels"], 1))

    def decay_parameters(self):
        """Weight matrices that receive decoupled weight decay (not biases or norms)."""
        return [parameter for parameter in self.parameters() if parameter.ndim > 1]

    def describe(self):
        with torch.no_grad():
            state = {name: float(parameter.norm()) for name, parameter in self.named_parameters()
                     if name.endswith("weight") and parameter.ndim > 1}
        return dict(FRONTEND_CONFIG, checkpoint_batch_size=self.batch_size,
                    parameters=sum(parameter.numel() for parameter in self.parameters()),
                    state=dict(weight_norms=state))

    def _forward_batch(self, waveform):
        c = FRONTEND_CONFIG
        features = self.head(self.body(rms_normalize_waveform(waveform).unsqueeze(1)))
        if features.shape[2] != c["time_bins"]:
            raise ValueError("Convolution strides do not produce the configured time bins")
        return features

    def forward(self, waveform):
        check_waveform(waveform)
        outputs = []
        differentiable = torch.is_grad_enabled() and (
            waveform.requires_grad or any(parameter.requires_grad for parameter in self.parameters()))
        for batch in waveform.split(self.batch_size):
            # Non-reentrant checkpointing also supports autograd.grad and inputs
            # without requires_grad. It recomputes all activations exactly.
            outputs.append(checkpoint(self._forward_batch, batch, use_reentrant=False)
                           if differentiable else self._forward_batch(batch))
        return torch.cat(outputs)
