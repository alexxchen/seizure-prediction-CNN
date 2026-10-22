"""1-D CNN backbone and axis normalization used by the meta-learner."""

import torch
from torch import nn


class AxisBatchNorm(nn.Module):
    """Match Keras BatchNormalization, including its biased running variance."""

    def __init__(self, features, axis, epsilon=1e-3, momentum=0.99):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
        self.momentum = momentum
        # Inner-loop functional fitting differentiates the support-only EMA.
        # Ordinary model training keeps the usual detached running updates.
        self.differentiable_stats = False
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.ones(features))

    def forward(self, inputs):
        shape = [1] * inputs.ndim
        shape[self.axis] = -1
        if self.training:
            axes = tuple(index for index in range(inputs.ndim) if index != self.axis)
            variance, mean = torch.var_mean(inputs, dim=axes, unbiased=False)
            if self.differentiable_stats:
                self.running_mean = torch.lerp(self.running_mean, mean, 1.0 - self.momentum)
                self.running_var = torch.lerp(self.running_var, variance, 1.0 - self.momentum)
            else:
                with torch.no_grad():
                    self.running_mean.lerp_(mean.detach(), 1.0 - self.momentum)
                    self.running_var.lerp_(variance.detach(), 1.0 - self.momentum)
        else:
            mean, variance = self.running_mean, self.running_var
        normalized = (inputs - mean.reshape(shape)) * torch.rsqrt(variance.reshape(shape) + self.epsilon)
        return normalized * self.weight.reshape(shape) + self.bias.reshape(shape)


class SeizureCNN(nn.Module):
    """Consume (N, channels, time) feature sequences and return class logits.

    A 1-D counterpart of the former 2-D MouseLFP CNN: channels are features with no
    spatial order, and time is the only convolved axis. Flattening before the dense
    layers keeps where in the window each pattern occurs.
    """

    def __init__(self, input_shape, nb_classes=2):
        super().__init__()
        self.input_shape = tuple(int(value) for value in input_shape)
        if len(self.input_shape) != 2 or any(value < 1 for value in self.input_shape):
            raise ValueError("Expected input shape (channels, time)")
        channels, time_bins = self.input_shape
        self.normal1 = AxisBatchNorm(channels, axis=1)
        self.conv1 = nn.Conv1d(channels, 32, 5, stride=2, padding=2)
        self.pool1 = nn.MaxPool1d(2, ceil_mode=True)
        self.normal2 = AxisBatchNorm(32, axis=1)
        self.conv2 = nn.Conv1d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2, ceil_mode=True)
        self.normal3 = AxisBatchNorm(32, axis=1)
        self.conv3 = nn.Conv1d(32, 32, 3, padding=1)
        self.pool3 = nn.MaxPool1d(2, ceil_mode=True)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        time_output = (time_bins - 1) // 2 + 1
        for _ in range(3):
            time_output = -(-time_output // 2)
        self.dens1 = nn.Linear(32 * time_output, 128)
        self.dens2 = nn.Linear(128, nb_classes)
        for layer in (self.conv1, self.conv2, self.conv3, self.dens1, self.dens2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        features = self.pool1(torch.relu(self.conv1(self.normal1(inputs))))
        features = self.pool2(torch.relu(self.conv2(self.normal2(features))))
        features = self.pool3(torch.relu(self.conv3(self.normal3(features))))
        features = self.drop1(torch.flatten(features, start_dim=1))
        features = self.drop2(torch.sigmoid(self.dens1(features)))
        # CrossEntropyLoss consumes logits; softmax is applied during prediction.
        return self.dens2(features)
