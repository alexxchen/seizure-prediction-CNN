"""Frontend-only bilevel learning and the pooled representation control."""

import copy

import torch
from torch.nn import functional as F

from meta_learning.inner import classifier_loss, fit_classifier


def meta_objective(model, support, query, config, *, return_diagnostics=False, progress=None):
    """Fit a private CNN for a fixed epoch budget, then differentiate query loss.

    Only the frontend is a meta-parameter. Every completed inner optimizer update is
    differentiable; activation checkpointing reduces stored activations without
    replacing the full frontend hypergradient with a first-order approximation.
    """
    features = model.frontend(support[0])
    fit = fit_classifier(model.cnn, features, support[1], config, differentiable=True, progress=progress)
    evaluation = copy.deepcopy(model.cnn).eval()
    result = classifier_loss(evaluation, fit.parameters, fit.buffers,
                             model.frontend(query[0]), query[1], config,
                             return_accuracy=return_diagnostics)
    if not return_diagnostics:
        return result
    loss, accuracy = result
    fit.report["query"] = dict(loss=float(loss.detach()), accuracy=accuracy, samples=len(query[1]))
    return loss, fit.report


def pooled_objective(model, support, query, config, *, event_weights):
    """Use the same examples, averaging their losses equally across represented events."""
    model.train()
    for layer in model.cnn.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = config.classifier_dropout
    inputs = torch.cat([support[0], query[0]])
    labels = torch.cat([support[1], query[1]])
    losses = F.cross_entropy(model(inputs), labels, label_smoothing=config.label_smoothing, reduction="none")
    weights = torch.cat(event_weights)
    return (losses * weights).sum() / weights.sum()
