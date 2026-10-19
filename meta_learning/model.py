"""A shared waveform frontend and an independently trained CNN classifier."""

import torch
from torch import nn

from meta_learning.frontend import WaveformFrontend
from models.cnn import SeizureCNN


class MetaModel(nn.Module):
    def __init__(self, cnn_seed=None):
        super().__init__()
        self.frontend = WaveformFrontend(seed=cnn_seed if cnn_seed is not None else 0)
        self.cnn = SeizureCNN(self.cnn_input_shape())
        if cnn_seed is not None:
            self.reset_classifier(cnn_seed)

    def cnn_input_shape(self):
        """The 1-D CNN consumes (channels, time); both follow the attached frontend."""
        return tuple(self.frontend.output_shape)

    def use_frontend(self, frontend, seed):
        """Attach another frontend and a fresh reference CNN shaped for its output."""
        self.frontend = frontend
        self.reset_classifier(seed)

    def reset_classifier(self, seed):
        """Use an unlearned, reproducible reference without advancing caller RNGs."""
        parameter = next(self.cnn.parameters())
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.cnn = SeizureCNN(self.cnn_input_shape()).to(device=parameter.device, dtype=parameter.dtype)
        self.train(self.training)

    def forward(self, waveform):
        return self.cnn(self.frontend(waveform))
