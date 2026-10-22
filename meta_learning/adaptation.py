"""Train a fresh per-mouse CNN on a frozen shared frontend; inference never fits."""

import copy

import numpy as np
import torch

from meta_learning.episodes import crops_support, support_indices, support_inputs
from meta_learning.inner import fit_classifier


def support_features(model, training, config, rng, device):
    """Encode at the frontend batch size while preserving the support crop RNG order."""
    planned = support_indices(training, np.arange(len(training.labels)), config)
    features, labels, pending = [], [], None
    with torch.no_grad():
        # Keep crop draws in the same batches as before; changing frontend_batch_size
        # only changes feature computation, never the sampled training windows.
        for start in range(0, len(planned), config.batch_size):
            indices = planned[start:start + config.batch_size]
            inputs, targets = support_inputs(training, indices, config, rng, torch.device("cpu"))
            labels.append(targets.to(device))
            pending = inputs if pending is None else torch.cat((pending, inputs))
            while len(pending) >= config.frontend_batch_size:
                features.append(model.frontend(pending[:config.frontend_batch_size].to(device)))
                pending = pending[config.frontend_batch_size:]
        if pending is not None and len(pending):
            features.append(model.frontend(pending.to(device)))
    return torch.cat(features), torch.cat(labels)


def adapt_model(initial, training, config, device, seed, *, progress=None):
    if training.split != "train" or not training.table.day.between(0, 7).all():
        raise ValueError("Adaptation may only use Days 0-7")
    model = copy.deepcopy(initial).to(device)
    # Only the frontend is shared between mice; every fit starts from a fresh CNN.
    model.reset_classifier(config.seed)
    model.frontend.requires_grad_(False)
    model.cnn.requires_grad_(True)
    # Validation checkpointing cannot change subsequent meta-training RNG draws. The support
    # policy (fixed windows, or K seeded random crops per event and class) is shared with meta inner fits.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        features, labels = support_features(model, training, config, rng, device)
        fit = fit_classifier(model.cnn, features, labels, config,
                             differentiable=False, progress=progress)
        model.cnn.load_state_dict(dict(fit.parameters, **fit.buffers))
        fit.report.update(support_crops=config.support_crops if crops_support(config) else 0,
                          support_crop_unit="per_event_per_class" if crops_support(config) else None)
    return model.eval(), fit.report


def predict(model, split, batch_size, device):
    model.eval()
    frontend_batch_size = model.frontend.batch_size
    result = []
    with torch.inference_mode():
        for start in range(0, len(split.labels), frontend_batch_size):
            indices = np.arange(start, min(start + frontend_batch_size, len(split.labels)))
            inputs, _ = split.batch(indices, device)
            features = model.frontend(inputs)
            for batch in features.split(batch_size):
                result.append(model.cnn(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(result)
