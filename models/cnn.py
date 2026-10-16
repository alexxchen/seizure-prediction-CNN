"""CNN backbone and axis normalization used by the meta-learner."""

import torch
from torch import nn


class AxisBatchNorm(nn.Module):
    """Match Keras BatchNormalization, including its biased running variance."""

    def __init__(self, features, axis, epsilon=1e-3, momentum=0.99):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
        self.momentum = momentum
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
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), 1.0 - self.momentum)
                self.running_var.lerp_(variance.detach(), 1.0 - self.momentum)
        else:
            mean, variance = self.running_mean, self.running_var
        normalized = (inputs - mean.reshape(shape)) * torch.rsqrt(variance.reshape(shape) + self.epsilon)
        return normalized * self.weight.reshape(shape) + self.bias.reshape(shape)


class SeizureCNN(nn.Module):
    """Consume (N, 1, electrodes, time, frequency) and return class logits."""

    def __init__(self, input_shape, nb_classes=2):
        super().__init__()
        self.input_shape = tuple(int(value) for value in input_shape)
        if len(self.input_shape) != 4 or self.input_shape[0] != 1:
            raise ValueError("Expected input shape (1, electrodes, time, frequency)")
        if any(value < 1 for value in self.input_shape):
            raise ValueError("Input dimensions must be positive")
        electrodes, time_bins, frequency_bins = self.input_shape[1:]
        self.normal1 = AxisBatchNorm(electrodes, axis=2)
        self.conv1 = nn.Conv3d(1, 16, (electrodes, 5, 5), stride=(1, 2, 2))
        # For kernel == stride == 2, ceil_mode matches TensorFlow SAME pooling.
        self.pool1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), ceil_mode=True)
        self.normal2 = AxisBatchNorm(16, axis=1)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.normal3 = AxisBatchNorm(32, axis=1)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

        def output_bins(size):
            size = (size - 5) // 2 + 1
            size = (size + 1) // 2
            size = (size - 2) // 2
            return (size - 2) // 2

        time_output, frequency_output = output_bins(time_bins), output_bins(frequency_bins)
        if min(time_output, frequency_output) < 1:
            raise ValueError("Time and frequency dimensions are too small for the CNN")
        self.dens1 = nn.Linear(64 * time_output * frequency_output, 128)
        self.dens2 = nn.Linear(128, nb_classes)
        for layer in (self.conv1, self.conv2, self.conv3, self.dens1, self.dens2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        features = self.pool1(torch.relu(self.conv1(self.normal1(inputs))))
        features = features.squeeze(2)
        features = self.pool2(torch.relu(self.conv2(self.normal2(features))))
        features = self.pool3(torch.relu(self.conv3(self.normal3(features))))
        features = self.drop1(torch.flatten(features, start_dim=1))
        features = self.drop2(torch.sigmoid(self.dens1(features)))
        # CrossEntropyLoss consumes logits; softmax is applied during prediction.
        return self.dens2(features)
