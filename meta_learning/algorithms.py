"""Differentiable classifier adaptation and the matched pooled-training control."""

import torch
from torch.func import functional_call
from torch.nn import functional as F


def meta_objective(model, support, query, config):
    """Differentiate query loss THROUGH support-driven classifier updates.

    The frontend is shared in the inner loop. Both the frontend parameters and
    the initial CNN parameters receive exact higher-order outer gradients.
    No support update mutates the base module or its buffers.
    """
    parameters = dict(model.named_parameters())
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    names = [name for name in parameters if name.startswith("cnn.")]
    if not names:
        raise ValueError("The model needs cnn.* classifier parameters")
    model.train()
    for _ in range(config.inner_steps):
        logits = functional_call(model, (parameters, buffers), (support[0],), strict=True)
        loss = F.cross_entropy(logits, support[1], label_smoothing=config.label_smoothing)
        gradients = torch.autograd.grad(loss, [parameters[name] for name in names], create_graph=True)
        updates = {name: parameters[name] - config.inner_lr * gradient
                   for name, gradient in zip(names, gradients)}
        parameters = dict(parameters, **updates)
    model.eval()
    logits = functional_call(model, (parameters, buffers), (query[0],), strict=True)
    return F.cross_entropy(logits, query[1], label_smoothing=config.label_smoothing)


def pooled_objective(model, support, query, config):
    """Use exactly the same sampled examples and scale, without an inner loop."""
    model.train()
    inputs = torch.cat([support[0], query[0]])
    labels = torch.cat([support[1], query[1]])
    return F.cross_entropy(model(inputs), labels, label_smoothing=config.label_smoothing)
