"""Frontend-only bilevel meta-learning."""

import copy

import torch

from meta_learning.contrastive import supervised_contrastive_loss
from meta_learning.episodes import SupportPool
from meta_learning.inner import FeatureEpochs, classifier_loss, fit_classifier, fitted_support_loss


def meta_objective(model, support, query, config, *, return_diagnostics=False, progress=None,
                   query_weights=None, crop_validation=None):
    """Differentiate query, final-support and optional supervised contrastive frontend losses.

    Only the frontend is a meta-parameter. Every completed inner optimizer update is
    differentiable; activation checkpointing reduces stored activations without
    replacing the full frontend hypergradient with a first-order approximation.
    """
    features = (FeatureEpochs(frontend=model.frontend, support=support) if isinstance(support, SupportPool)
                else model.frontend(support[0]))
    labels = None if isinstance(support, SupportPool) else support[1]
    fit = fit_classifier(model.cnn, features, labels, config, differentiable=True, progress=progress)
    evaluation = copy.deepcopy(model.cnn).eval()
    query_features = model.frontend(query[0])
    result = classifier_loss(evaluation, fit.parameters, fit.buffers,
                             query_features, query[1], config,
                             return_accuracy=return_diagnostics, sample_weights=query_weights)
    query_loss = result[0] if return_diagnostics else result
    # Re-evaluate all support pool slots equally at the final weights in eval mode.
    # Keep both the feature path and the full inner-update hypergradient intact.
    support_loss = fitted_support_loss(model.cnn, fit, config, differentiable=True) if (
        config.support_loss_weight or return_diagnostics) else None
    loss = (query_loss + config.support_loss_weight * support_loss
            if config.support_loss_weight else query_loss)
    contrastive_loss = query_loss.new_zeros(())
    contrastive_report = dict(enabled=False)
    if config.contrastive_loss_weight:
        support_features, support_labels = fit.source.fitted(differentiable=True)
        projected = model.frontend.project(support_features)
        contrastive_loss, contrastive_report = supervised_contrastive_loss(projected, support_labels)
        contrastive_report.update(enabled=True, scope="Current mouse's complete support pool only; no query views",
            embedding="L2-normalized MLP projection of temporal channel means",
            source_dimensions=support_features[0].numel())
        loss = loss + config.contrastive_loss_weight * contrastive_loss
    if not return_diagnostics:
        return loss
    _, accuracy = result
    fit.report["query"] = dict(loss=float(query_loss.detach()), accuracy=accuracy, samples=len(query[1]))
    fit.report["outer_objective"] = dict(
        query_loss=float(query_loss.detach()), support_loss_after_last_update=float(support_loss.detach()),
        support_loss_weight=config.support_loss_weight,
        weighted_support_loss=float(support_loss.detach()) * config.support_loss_weight,
        contrastive_loss=float(contrastive_loss.detach()), contrastive_loss_weight=config.contrastive_loss_weight,
        weighted_contrastive_loss=float(contrastive_loss.detach()) * config.contrastive_loss_weight,
        contrastive=contrastive_report,
        total_loss=float(loss.detach()))
    if crop_validation is not None:
        # Frozen-fit diagnostics are separate from all differentiable objective terms.
        fit.report.update(crop_validation(fit))
    return loss, fit.report
