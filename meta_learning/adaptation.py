"""Train a fresh per-mouse CNN on a frozen shared frontend; inference never fits."""

import copy

import numpy as np
import torch

from meta_learning.episodes import SupportPool, balanced_indices, crops_support, event_crop_indices, support_indices, support_inputs
from meta_learning.common import digest
from meta_learning.inner import FeatureEpochs, classifier_loss, fit_classifier, fitted_support_loss


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


def validate_fresh_crops(model, split, config, rng, device, *, role, parameters=None, buffers=None,
                         reference=None):
    """Score one new support/query draw with frozen weights, EMA, and caller RNGs."""
    if role not in {"support", "query"}:
        raise ValueError("Fresh crop validation accepts only support or query")
    days = range(8) if role == "support" else config.query_validation_days
    expected_split = "train" if role == "support" else "validation"
    if split.split != expected_split or split.table.empty or not split.table.day.isin(days).all():
        raise ValueError(f"Fresh {role} crops require that role's recording days")
    count = config.support_crops if role == "support" else config.query_crops
    cap = config.support_per_class if role == "support" else config.query_per_class
    # Role separation gives independent diagnostic streams without consuming any fit/query draws.
    rng_sha256 = digest(dict(role=role, state=rng.bit_generator.state))
    diagnostic_rng = np.random.default_rng(int(rng_sha256[:16], 16))
    planned = event_crop_indices(split, np.arange(len(split.labels)), count)
    planned = balanced_indices(split.labels, planned, cap, diagnostic_rng)
    evaluation = copy.deepcopy(model.cnn).eval()
    parameters = {name: value.detach() for name, value in (
        dict(model.cnn.named_parameters()) if parameters is None else parameters).items()}
    buffers = {name: value.detach().clone() for name, value in (
        dict(model.cnn.named_buffers()) if buffers is None else buffers).items()}
    device = torch.device(device)
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == "cuda" else [])
    total_loss, correct, positives = 0.0, 0, 0
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        for start in range(0, len(planned), config.batch_size):
            inputs, labels = split.random_crop_batch(planned[start:start + config.batch_size],
                                                     diagnostic_rng, device)
            features = model.frontend(inputs)
            loss, accuracy = classifier_loss(evaluation, parameters, buffers, features, labels,
                                             config, return_accuracy=True)
            total_loss += float(loss) * len(labels)
            correct += round(accuracy * len(labels))
            positives += int((labels == 1).sum())
    loss, accuracy = total_loss / len(planned), correct / len(planned)
    report = dict(role=f"{role}_crop_validation", days=sorted(int(day) for day in split.table.day.unique()),
                samples=len(planned), positives=positives, negatives=len(planned) - positives,
                positive_fraction=positives / len(planned), crops=count,
                crop_count_unit="per_event_per_class", label_smoothing=config.label_smoothing,
                loss=loss, accuracy=accuracy,
                sampling="One new draw after fitting; independent role-keyed crop RNG; windows may overlap",
                rng_state_sha256=rng_sha256, source_fingerprint=split.fingerprint,
                use="Diagnostic only; no gradients, updates, statistics fitting, or checkpoint selection")
    if reference is not None:
        report.update(accuracy_gap=reference["accuracy"] - accuracy, loss_gap=loss - reference["loss"])
    return report


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
        if crops_support(config):
            features = FeatureEpochs(frontend=model.frontend,
                                     support=SupportPool(training, np.arange(len(training.labels)), config, rng))
            labels = None
        else:
            features, labels = support_features(model, training, config, rng, device)
        fit = fit_classifier(model.cnn, features, labels, config,
                             differentiable=False, progress=progress)
        fitted_support_loss(model.cnn, fit, config, differentiable=False)
        if crops_support(config):
            fit.report["support_crop_validation"] = validate_fresh_crops(
                model, training, config, rng, device, role="support", parameters=fit.parameters,
                buffers=fit.buffers, reference=dict(accuracy=fit.report["last_support_accuracy"],
                                                  loss=fit.report["last_support_loss"]))
            if progress:
                progress(dict(support_crop_validation=fit.report["support_crop_validation"]))
        model.cnn.load_state_dict(dict(fit.parameters, **fit.buffers))
        fit.report.update(support_crops=config.support_crops if crops_support(config) else 0,
                          support_pool_crops=config.support_pool_crops if crops_support(config) else 0,
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
