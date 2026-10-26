"""Frontend-only bilevel meta-learning."""

import copy

from meta_learning.inner import classifier_loss, fit_classifier


def meta_objective(model, support, query, config, *, return_diagnostics=False, progress=None,
                   query_weights=None):
    """Differentiate query loss plus weighted support loss after the final update.

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
    query_loss = result[0] if return_diagnostics else result
    # Re-evaluate the entire fitted support at the final weights in eval mode.
    # Keep both the feature path and the full inner-update hypergradient intact.
    support_loss = classifier_loss(evaluation, fit.parameters, fit.buffers,
                                   features, support[1], config) if (
                                       config.support_loss_weight or return_diagnostics) else None
    loss = (query_loss + config.support_loss_weight * support_loss
            if config.support_loss_weight else query_loss)
    if not return_diagnostics:
        return loss
    _, accuracy = result
    fit.report["query"] = dict(loss=float(query_loss.detach()), accuracy=accuracy, samples=len(query[1]))
    fit.report["outer_objective"] = dict(
        query_loss=float(query_loss.detach()), support_loss_after_last_update=float(support_loss.detach()),
        support_loss_weight=config.support_loss_weight,
        weighted_support_loss=float(support_loss.detach()) * config.support_loss_weight,
        total_loss=float(loss.detach()))
    return loss, fit.report
