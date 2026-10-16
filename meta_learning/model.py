"""Shared spectral preprocessing and an independently trained CNN classifier."""

import torch
from torch import nn

from meta_learning.frontend import SpectralFrontend
from models.cnn import SeizureCNN

CNN_INPUT_SHAPE = (1, 1, 55, 114)


class MetaModel(nn.Module):
    def __init__(self, cnn_seed=None):
        super().__init__()
        self.frontend = SpectralFrontend(seed=cnn_seed if cnn_seed is not None else 0)
        self.cnn = SeizureCNN(CNN_INPUT_SHAPE)
        if cnn_seed is not None:
            self.reset_classifier(cnn_seed)

    def reset_classifier(self, seed):
        """Use an unlearned, reproducible reference without advancing caller RNGs."""
        parameter = next(self.cnn.parameters())
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.cnn = SeizureCNN(CNN_INPUT_SHAPE).to(device=parameter.device, dtype=parameter.dtype)
        self.train(self.training)

    def forward(self, magnitude):
        return self.cnn(self.frontend(magnitude))
