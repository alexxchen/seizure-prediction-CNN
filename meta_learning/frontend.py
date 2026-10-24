"""A single, randomly initialized convolutional network from waveform to the task CNN.

No spectrum is precomputed and no stage is hand-designed: a pre-activation 1-D
ResNet with strided downsampling maps each preprocessed waveform to a
(channels, time) feature sequence for the 1-D task CNN.
"""

import torch
from torch import nn

from meta_learning.waveform import WAVEFORM_CONFIG


FRONTEND_CONFIG = dict(architecture="waveform_resnet1d", input_samples=28672,
    input_normalization="Per-window RMS after existing waveform mean removal; no fitted statistics",
    rms_numerics="Peak-scaled RMS computation; dtype-tiny floor and zero output for silence",
    stem="Conv1d(1->16, k8, s4)", stages=[[16, 2], [32, 2], [48, 7], [64, 1], [64, 1]], norm_groups=8,
    stage_note="Each [channels, stride]: one residual block at that width, then a strided Conv1d to the next "
               "width with overlapping kernel 2*stride+1 and padding stride "
               "(28672 -> 7168 -> 3584 -> 1792 -> 256 -> 256); the last stage adds a dilation-2 block",
    block="Pre-activation: GroupNorm(8) -> GELU -> Conv1d k3 (dilation d) -> GroupNorm(8) -> GELU -> Conv1d k3 "
          "(zero-initialized), added to the residual stream",
    head="GroupNorm(8) -> GELU -> Conv1d(64 -> output_channels, k1); output is a (channels, time) sequence",
    output_channels=64, time_bins=256,
    initialization="Seeded PyTorch defaults (random); zero-initialized last conv in each residual block")


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


def check_waveform(waveform):
    if waveform.ndim != 2 or waveform.shape[1] != FRONTEND_CONFIG["input_samples"]:
        raise ValueError(f"Expected preprocessed waveforms shaped (N, {FRONTEND_CONFIG['input_samples']}) "
                         f"at {WAVEFORM_CONFIG['target_hz']} Hz")


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

    def __init__(self, seed=0):
        super().__init__()
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
                    layers.append(nn.Conv1d(channels, c["stages"][index + 1][0], 2 * stride + 1,
                                            stride=stride, padding=stride))
            layers.append(ResidualBlock1d(c["stages"][-1][0], dilation=2))
            self.body = nn.Sequential(*layers)
            width = c["stages"][-1][0]
            self.head = nn.Sequential(nn.GroupNorm(c["norm_groups"], width), nn.GELU(),
                                      nn.Conv1d(width, c["output_channels"], 1))

    def forward(self, waveform):
        check_waveform(waveform)
        c = FRONTEND_CONFIG
        features = self.head(self.body(rms_normalize_waveform(waveform).unsqueeze(1)))
        if features.shape[2] != c["time_bins"]:
            raise ValueError("Convolution strides do not produce the configured time bins")
        return features
