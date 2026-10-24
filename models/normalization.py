"""Locked support-only BatchNorm implementation for differentiable CNN fits."""

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
