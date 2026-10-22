"""Frontend-only bilevel meta-learning."""

import copy

from meta_learning.inner import classifier_loss, fit_classifier


def meta_objective(model, support, query, config, *, return_diagnostics=False, progress=None,
                   query_weights=None):
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
                             return_accuracy=return_diagnostics, sample_weights=query_weights)
    if not return_diagnostics:
        return result
    loss, accuracy = result
    fit.report["query"] = dict(loss=float(loss.detach()), accuracy=accuracy, samples=len(query[1]))
    return loss, fit.report
