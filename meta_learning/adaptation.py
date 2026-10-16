"""Fixed-budget personalization using training days; inference never adapts."""

import copy

import numpy as np
import torch
from torch.nn import functional as F


def adapt_model(initial, training, config, device, seed):
    if training.split != "train" or not training.table.day.between(0, 7).all():
        raise ValueError("Adaptation may only use Days 0-7")
    model = copy.deepcopy(initial).to(device)
    model.frontend.requires_grad_(config.adaptation_frontend)
    model.cnn.requires_grad_(True)
    scale = training.scale()
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.adaptation_lr)
    # Validation checkpointing cannot change subsequent meta-training RNG draws.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        offset, order = 0, rng.permutation(len(training.labels))
        for _ in range(config.adaptation_steps):
            model.train()
            indices = order[offset:offset + config.batch_size]
            inputs, labels = training.batch(indices, scale, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), labels, label_smoothing=config.label_smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite adaptation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            offset += len(indices)
            if offset == len(order):
                offset, order = 0, rng.permutation(len(training.labels))
    return model.eval(), scale


def predict(model, split, scale, batch_size, device):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(split.labels), batch_size):
            indices = np.arange(start, min(start + batch_size, len(split.labels)))
            inputs, _ = split.batch(indices, scale, device)
            result.append(model(inputs).softmax(dim=1).cpu().numpy())
    return np.concatenate(result)
