"""Editable task CNN architecture; support normalization remains protocol-owned."""

import torch
from torch import nn


from models.normalization import AxisBatchNorm


class SeizureCNN(nn.Module):
    """Consume (N, channels, time) feature sequences and return class logits.

    A 1-D counterpart of the former 2-D MouseLFP CNN: channels are features with no
    spatial order, and time is the only convolved axis. Temporal mean pooling
    summarizes pattern strength independently of its position in the window.
    """

    def __init__(self, input_shape, nb_classes=2):
        super().__init__()
        self.input_shape = tuple(int(value) for value in input_shape)
        if len(self.input_shape) != 2 or any(value < 1 for value in self.input_shape):
            raise ValueError("Expected input shape (channels, time)")
        channels, time_bins = self.input_shape
        self.normal1 = AxisBatchNorm(channels, axis=1)
        self.conv1 = nn.Conv1d(channels, 16, 5, stride=2, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.normal2 = AxisBatchNorm(16, axis=1)
        self.conv2 = nn.Conv1d(16, 16, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.normal3 = AxisBatchNorm(16, axis=1)
        self.conv3 = nn.Conv1d(16, 16, 3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.0)
        self.drop2 = nn.Dropout(0.0)
        self.dens1 = nn.Linear(16, 64)
        self.dens2 = nn.Linear(64, nb_classes)

    def forward(self, inputs):
        features = self.pool1(torch.relu(self.conv1(self.normal1(inputs))))
        features = self.pool2(torch.relu(self.conv2(self.normal2(features))))
        features = self.pool3(torch.relu(self.conv3(self.normal3(features))))
        features = self.drop1(features.mean(dim=-1))
        features = self.drop2(torch.relu(self.dens1(features)))
        # CrossEntropyLoss consumes logits; softmax is applied during prediction.
        return self.dens2(features)
