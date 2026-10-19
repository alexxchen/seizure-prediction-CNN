"""Train a fresh per-mouse CNN on a frozen shared frontend; inference never fits."""

import copy

import numpy as np
import torch

from meta_learning.episodes import crops_support, support_indices, support_inputs
from meta_learning.inner import fit_classifier


def adapt_model(initial, training, config, device, seed, *, return_diagnostics=False, progress=None):
    if training.split != "train" or not training.table.day.between(0, 7).all():
        raise ValueError("Adaptation may only use Days 0-7")
    model = copy.deepcopy(initial).to(device)
    # Never transfer learned classifier weights, including the pooled control's
    # auxiliary head. Only preprocessing is shared between mice and stages.
    model.reset_classifier(config.seed)
    model.frontend.requires_grad_(False)
    model.cnn.requires_grad_(True)
    scale = training.scale()
    # Validation checkpointing cannot change subsequent meta-training RNG draws. The support
    # policy (fixed windows, or K seeded random crops per event and class) is shared with meta inner fits.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        planned = support_indices(training, np.arange(len(training.labels)), config)
        features, labels = [], []
        with torch.no_grad():
            for start in range(0, len(planned), config.batch_size):
                indices = planned[start:start + config.batch_size]
                inputs, targets = support_inputs(training, indices, config, rng, device)
                features.append(model.frontend(inputs))
                labels.append(targets)
        fit = fit_classifier(model.cnn, torch.cat(features), torch.cat(labels), config,
                             differentiable=False, progress=progress)
        model.cnn.load_state_dict(dict(fit.parameters, **fit.buffers))
        fit.report.update(support_crops=config.support_crops if crops_support(config) else 0,
                          support_crop_unit="per_event_per_class" if crops_support(config) else None)
    result = model.eval(), scale
    return (*result, fit.report) if return_diagnostics else result


def predict(model, split, scale, batch_size, device):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(split.labels), batch_size):
            indices = np.arange(start, min(start + batch_size, len(split.labels)))
            inputs, _ = split.batch(indices, scale, device)
            result.append(model(inputs).softmax(dim=1).cpu().numpy())
    return np.concatenate(result)
