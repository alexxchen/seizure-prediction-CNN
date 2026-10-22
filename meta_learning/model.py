"""A shared waveform frontend and an independently trained CNN classifier."""

import torch
from torch import nn

from meta_learning.frontend import WaveformFrontend
from models.cnn import SeizureCNN


class MetaModel(nn.Module):
    def __init__(self, cnn_seed=0, frontend_batch_size=32):
        super().__init__()
        self.frontend = WaveformFrontend(seed=cnn_seed, batch_size=frontend_batch_size)
        self.cnn = self._new_classifier(cnn_seed)

    def cnn_input_shape(self):
        """The 1-D CNN consumes (channels, time); both follow the attached frontend."""
        return tuple(self.frontend.output_shape)

    def _new_classifier(self, seed):
        """Initialize one CPU classifier without advancing caller CPU/CUDA RNGs."""
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            torch.set_rng_state(generator.get_state())
            return SeizureCNN(self.cnn_input_shape())

    def reset_classifier(self, seed):
        """Replace task weights and BatchNorm state with a fresh unlearned reference."""
        parameter = next(self.cnn.parameters())
        self.cnn = self._new_classifier(seed).to(device=parameter.device, dtype=parameter.dtype)
        self.cnn.train(self.training)

    def forward(self, waveform):
        return self.cnn(self.frontend(waveform))
