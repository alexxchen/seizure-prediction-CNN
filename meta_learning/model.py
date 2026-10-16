"""Shared spectral frontend and a CNN initialization adapted per mouse/task."""

from torch import nn

from meta_learning.frontend import SpectralFrontend
from models.cnn import AxisBatchNorm, SeizureCNN

CNN_INPUT_SHAPE = (1, 1, 55, 114)


class MetaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = SpectralFrontend()
        self.cnn = SeizureCNN(CNN_INPUT_SHAPE)

    def train(self, mode=True):
        super().train(mode)
        # Fixed running statistics make query predictions independent of query
        # companions and avoid mutable/transductive BatchNorm state in episodes.
        # Affine normalization weights remain trainable in every algorithm.
        for layer in self.modules():
            if isinstance(layer, AxisBatchNorm):
                layer.eval()
        return self

    def forward(self, power):
        return self.cnn(self.frontend(power))
