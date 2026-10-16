"""Epoch-based, support-only CNN fitting with exact checkpointed hypergradients."""

import copy
from dataclasses import dataclass
import math

import torch
from torch.func import functional_call
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from models.cnn import AxisBatchNorm


@dataclass
class ClassifierFit:
    parameters: dict
    buffers: dict
    report: dict


def classifier_loss(classifier, parameters, buffers, features, labels, config, *, return_accuracy=False):
    """Sample-weighted loss; evaluation never mutates classifier or BN state."""
    total = features.new_zeros(())
    correct = features.new_zeros(())
    for start in range(0, len(labels), config.batch_size):
        stop = start + config.batch_size
        logits = functional_call(classifier, (parameters, buffers), (features[start:stop],), strict=True)
        total = total + F.cross_entropy(logits, labels[start:stop], reduction="sum",
                                       label_smoothing=config.label_smoothing)
        if return_accuracy:
            predicted = (logits[:, 1] >= logits[:, 0]).long()
            correct = correct + (predicted == labels[start:stop]).sum()
    loss = total / len(labels)
    return (loss, float(correct / len(labels))) if return_accuracy else loss


def training_classifier(reference, config):
    training = copy.deepcopy(reference).train()
    for layer in training.modules():
        if isinstance(layer, AxisBatchNorm):
            layer.eval()
        elif isinstance(layer, torch.nn.Dropout):
            layer.p = config.classifier_dropout
    return training


def update_segment(training, names, buffers, features, labels, state, batches, config, differentiable, step=0):
    count = len(names)
    parameters, momentum = state[:count], state[count:2 * count]
    variance = state[2 * count:]
    for indices in batches:
        logits = functional_call(training, (dict(zip(names, parameters)), buffers),
                                 (features[indices],), strict=True)
        loss = F.cross_entropy(logits, labels[indices], label_smoothing=config.label_smoothing)
        gradients = torch.autograd.grad(loss, parameters, create_graph=differentiable)
        norm = torch.linalg.vector_norm(torch.stack([gradient.norm() for gradient in gradients]))
        clip = (config.gradient_clip / (norm + 1e-6)).clamp(max=1.0)
        gradients = tuple(gradient * clip for gradient in gradients)
        step += 1
        if config.inner_optimizer == "adam":
            momentum = tuple(config.inner_beta1 * value + (1 - config.inner_beta1) * gradient
                             for value, gradient in zip(momentum, gradients))
            variance = tuple(config.inner_beta2 * value + (1 - config.inner_beta2) * gradient.square()
                             for value, gradient in zip(variance, gradients))
            # Standard Adam values, with a finite zero subgradient when v == 0.
            # Taking sqrt(0) directly creates NaNs in the outer hypergradient.
            roots = tuple(torch.where(value > 0,
                          torch.sqrt(torch.where(value > 0, value, torch.ones_like(value))),
                          torch.zeros_like(value)) for value in variance)
            parameters = tuple(parameter - (config.inner_lr / (1 - config.inner_beta1 ** step)) * first /
                               (root / math.sqrt(1 - config.inner_beta2 ** step) + config.inner_epsilon)
                               for parameter, first, root in zip(parameters, momentum, roots))
        else:
            momentum = tuple(config.inner_momentum * value + gradient
                             for value, gradient in zip(momentum, gradients))
            parameters = tuple(parameter - config.inner_lr * value
                               for parameter, value in zip(parameters, momentum))
        if not differentiable:
            parameters = tuple(value.detach().requires_grad_(True) for value in parameters)
            momentum = tuple(value.detach() for value in momentum)
            variance = tuple(value.detach() for value in variance)
    return (*parameters, *momentum, *variance)


def run_fit(reference, features, labels, config, differentiable, progress, segments=None):
    training = training_classifier(reference, config)
    evaluation = copy.deepcopy(training).eval()
    names = tuple(name for name, _ in reference.named_parameters())
    weights = tuple(parameter.detach().clone().requires_grad_(True)
                    for parameter in reference.parameters())
    velocity = tuple(torch.zeros_like(parameter) for parameter in weights)
    variance = tuple(torch.zeros_like(parameter) for parameter in weights) if config.inner_optimizer == "adam" else ()
    buffers = {name: value.detach().clone() for name, value in reference.named_buffers()}
    if not differentiable:
        features = features.detach()

    def measure(parameters):
        with torch.no_grad():
            loss, accuracy = classifier_loss(evaluation, dict(zip(names, parameters)), buffers,
                                             features.detach(), labels, config, return_accuracy=True)
            value = float(loss)
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite classifier support loss")
        return value, accuracy

    initial_loss, initial_accuracy = measure(weights)
    best_loss = previous_loss = initial_loss
    best_epoch, stable, steps = 0, 0, 0
    history = []
    stop_reason = "max_epochs"
    for epoch in range(1, config.inner_max_epochs + 1):
        order = torch.randperm(len(labels), device=features.device)
        batches = tuple(order[start:start + config.batch_size]
                        for start in range(0, len(labels), config.batch_size))
        segment_size = config.inner_checkpoint_steps or 1
        for start in range(0, len(batches), segment_size):
            selected = batches[start:start + segment_size]
            if segments is not None:
                segments.append(dict(state=tuple(value.detach().cpu().clone() for value in (*weights, *velocity, *variance)),
                    batches=tuple(indices.cpu() for indices in selected), cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(features.device) if features.is_cuda else None, step=steps))
            state = update_segment(training, names, buffers, features, labels, (*weights, *velocity, *variance),
                                   selected, config, differentiable, step=steps)
            weights, velocity, variance = state[:len(names)], state[len(names):2 * len(names)], state[2 * len(names):]
            steps += len(selected)
        loss, accuracy = measure(weights)
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
        relative_change = abs(previous_loss - loss) / max(abs(previous_loss), 1e-12)
        stable = stable + 1 if relative_change <= config.inner_relative_tolerance else 0
        previous_loss = loss
        row = dict(epoch=epoch, optimizer_steps=steps, support_loss=loss,
                   support_accuracy=accuracy, best_support_loss=best_loss,
                   relative_loss_change=relative_change, stable_epochs=stable)
        history.append(row)
        reached_patience = epoch >= config.inner_min_epochs and stable >= config.inner_patience
        if progress and (epoch == 1 or epoch % config.inner_log_every == 0 or
                         reached_patience or epoch == config.inner_max_epochs):
            progress(dict(row, selected_epoch=epoch))
        if reached_patience:
            stop_reason = "support_plateau"
            break
    report = dict(samples=len(labels), epochs_run=len(history), optimizer_steps=steps,
                  selected_epoch=len(history), best_epoch=best_epoch, initial_support_loss=initial_loss,
                  optimizer=config.inner_optimizer, classifier_dropout=config.classifier_dropout,
                  initial_support_accuracy=initial_accuracy, last_support_accuracy=history[-1]["support_accuracy"],
                  best_support_loss=best_loss, last_support_loss=history[-1]["support_loss"],
                  stop_reason=stop_reason, plateau_reached=stop_reason == "support_plateau",
                  history=history)
    return ClassifierFit(dict(zip(names, weights)), buffers, report)


class _ReplayClassifierFit(torch.autograd.Function):
    """Reverse the complete optimizer trajectory a segment at a time.

    Forward fitting saves detached CNN and optimizer-state boundaries in CPU memory.
    Backward propagates adjoints through every segment, accumulating the exact
    support-feature gradient. GPU activation memory depends on segment length,
    while CPU checkpoint storage grows with the number of segments. No gradient
    is returned for the unlearned CNN reference. Higher outer derivatives are
    intentionally unsupported; inner second derivatives are computed in replay.
    """

    @staticmethod
    def forward(ctx, features, labels, reference, config, progress, report):
        ctx.segments = []
        with torch.enable_grad():
            fit = run_fit(reference, features, labels, config, False, progress, ctx.segments)
        ctx.state_size = len(ctx.segments[0]["state"])
        boundaries = tuple(value for segment in ctx.segments for value in segment.pop("state"))
        ctx.save_for_backward(features, labels, *boundaries)
        ctx.training = training_classifier(reference, config)
        ctx.names = tuple(fit.parameters)
        ctx.buffers, ctx.config = fit.buffers, config
        report.update(fit.report)
        return tuple(value.detach() for value in fit.parameters.values())

    @staticmethod
    @once_differentiable
    def backward(ctx, *weight_gradients):
        features, labels, *boundaries = ctx.saved_tensors
        count, device = len(ctx.names), features.device
        # Saved boundary values also supply shapes for unused output gradients.
        reference_state = boundaries[-ctx.state_size:]
        adjoints = tuple(gradient if gradient is not None else torch.zeros_like(reference_state[i], device=device)
                         for i, gradient in enumerate(weight_gradients))
        adjoints += tuple(torch.zeros_like(value, device=device) for value in reference_state[count:])
        feature_gradient = torch.zeros_like(features)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if features.is_cuda else []
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            for position in reversed(range(len(ctx.segments))):
                segment = ctx.segments[position]
                torch.set_rng_state(segment["cpu_rng"])
                if features.is_cuda:
                    torch.cuda.set_rng_state(segment["cuda_rng"], device)
                state = boundaries[position * ctx.state_size:(position + 1) * ctx.state_size]
                initial = tuple(value.to(device).detach().requires_grad_(True) for value in state)
                inputs = features.detach().requires_grad_(True)
                batches = tuple(indices.to(device) for indices in segment["batches"])
                final = update_segment(ctx.training, ctx.names, ctx.buffers, inputs, labels,
                                       initial, batches, ctx.config, True, step=segment["step"])
                gradients = torch.autograd.grad(final, (inputs, *initial), grad_outputs=adjoints,
                                                allow_unused=True)
                if gradients[0] is not None:
                    feature_gradient.add_(gradients[0])
                adjoints = tuple(value.detach() if value is not None else torch.zeros_like(initial[i])
                                 for i, value in enumerate(gradients[1:]))
                del initial, inputs, final, gradients
        return feature_gradient, None, None, None, None, None


def fit_classifier(reference, features, labels, config, *, differentiable, progress=None):
    """Train full epochs to support patience/cap; optionally replay exact gradients.

    CNN weights and all optimizer moments persist within a task. Stopping is discrete and held
    fixed during differentiation. A zero checkpoint interval enables ordinary
    full unrolling for small reference checks; final per-mouse fitting needs no
    hypergradient or stored training trajectory.
    """
    if not len(labels) or len(features) != len(labels):
        raise ValueError("Classifier fitting requires nonempty matching support features/labels")
    if differentiable and config.inner_checkpoint_steps:
        report = {}
        weights = _ReplayClassifierFit.apply(features, labels, reference, config, progress, report)
        buffers = {name: value.detach().clone() for name, value in reference.named_buffers()}
        return ClassifierFit(dict(zip((name for name, _ in reference.named_parameters()), weights)), buffers, report)
    return run_fit(reference, features, labels, config, differentiable, progress)
