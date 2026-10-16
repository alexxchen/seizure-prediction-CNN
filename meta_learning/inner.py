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
            layer.differentiable_stats = True
        elif isinstance(layer, torch.nn.Dropout):
            layer.p = config.classifier_dropout
    return training


class _SupportForward(torch.nn.Module):
    """Return functional buffers before functional_call restores module state."""

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, features):
        logits = self.classifier(features)
        return logits, tuple(self.classifier.buffers())


def update_segment(training, names, buffers, features, labels, state, batches, config, differentiable, step=0):
    count = len(names)
    optimizer_size = count * (3 if config.inner_optimizer == "adam" else 2)
    parameters, momentum = state[:count], state[count:2 * count]
    variance = state[2 * count:optimizer_size]
    # Functional BatchNorm replaces these values out of place on every support
    # forward. Include the EMA state in the trajectory and its hypergradient.
    buffers = dict(zip(buffers, state[optimizer_size:]))
    forward = _SupportForward(training)
    for indices in batches:
        logits, updated_buffers = functional_call(forward,
            ({f"classifier.{name}": value for name, value in zip(names, parameters)},
             {f"classifier.{name}": value for name, value in buffers.items()}),
            (features[indices],), strict=True)
        buffers = dict(zip(buffers, updated_buffers))
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
            buffers = {name: value.detach() for name, value in buffers.items()}
    return (*parameters, *momentum, *variance, *buffers.values())


def run_fit(reference, features, labels, config, differentiable, progress, segments=None):
    training = training_classifier(reference, config)
    evaluation = copy.deepcopy(training).eval()
    names = tuple(name for name, _ in reference.named_parameters())
    weights = tuple(parameter.detach().clone().requires_grad_(True)
                    for parameter in reference.parameters())
    velocity = tuple(torch.zeros_like(parameter) for parameter in weights)
    variance = tuple(torch.zeros_like(parameter) for parameter in weights) if config.inner_optimizer == "adam" else ()
    buffers = {name: value.detach().clone() for name, value in reference.named_buffers()}
    optimizer_size = len(names) * (3 if config.inner_optimizer == "adam" else 2)
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
    best_epoch, steps = 0, 0
    history = []
    for epoch in range(1, config.inner_epochs + 1):
        order = torch.randperm(len(labels), device=features.device)
        batches = tuple(order[start:start + config.batch_size]
                        for start in range(0, len(labels), config.batch_size))
        segment_size = config.inner_checkpoint_steps or 1
        for start in range(0, len(batches), segment_size):
            selected = batches[start:start + segment_size]
            if segments is not None:
                segments.append(dict(state=tuple(value.detach().cpu().clone() for value in
                    (*weights, *velocity, *variance, *buffers.values())),
                    batches=tuple(indices.cpu() for indices in selected), cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(features.device) if features.is_cuda else None, step=steps))
            state = update_segment(training, names, buffers, features, labels,
                                   (*weights, *velocity, *variance, *buffers.values()),
                                   selected, config, differentiable, step=steps)
            weights, velocity = state[:len(names)], state[len(names):2 * len(names)]
            variance = state[2 * len(names):optimizer_size]
            buffers = dict(zip(buffers, state[optimizer_size:]))
            steps += len(selected)
        loss, accuracy = measure(weights)
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
        relative_change = abs(previous_loss - loss) / max(abs(previous_loss), 1e-12)
        previous_loss = loss
        row = dict(epoch=epoch, optimizer_steps=steps, support_loss=loss,
                   support_accuracy=accuracy, best_support_loss=best_loss,
                   relative_loss_change=relative_change)
        history.append(row)
        if progress and (epoch == config.inner_epochs or
                         (config.inner_log_every > 0 and (epoch == 1 or epoch % config.inner_log_every == 0))):
            progress(dict(row, selected_epoch=epoch))
    report = dict(samples=len(labels), epochs_requested=config.inner_epochs,
                  epochs_run=len(history), optimizer_steps=steps,
                  selected_epoch=len(history), best_epoch=best_epoch, initial_support_loss=initial_loss,
                  optimizer=config.inner_optimizer, classifier_dropout=config.classifier_dropout,
                  batchnorm="Support batch statistics during training; support EMA during evaluation",
                  initial_support_accuracy=initial_accuracy, last_support_accuracy=history[-1]["support_accuracy"],
                  best_support_loss=best_loss, last_support_loss=history[-1]["support_loss"],
                  stop_reason="fixed_epochs",
                  history=history)
    return ClassifierFit(dict(zip(names, weights)), buffers, report)


class _ReplayClassifierFit(torch.autograd.Function):
    """Reverse the complete optimizer trajectory a segment at a time.

    Forward fitting saves detached CNN, optimizer, and BatchNorm boundaries in CPU memory.
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
        ctx.buffer_names, ctx.config = tuple(fit.buffers), config
        ctx.optimizer_size = len(ctx.names) * (3 if config.inner_optimizer == "adam" else 2)
        report.update(fit.report)
        return tuple(value.detach() for value in (*fit.parameters.values(), *fit.buffers.values()))

    @staticmethod
    @once_differentiable
    def backward(ctx, *output_gradients):
        features, labels, *boundaries = ctx.saved_tensors
        count, device = len(ctx.names), features.device
        # Saved boundary values also supply shapes for unused output gradients.
        reference_state = boundaries[-ctx.state_size:]

        def output_adjoint(index, value):
            gradient = output_gradients[index]
            return gradient if gradient is not None else torch.zeros_like(value, device=device)

        adjoints = tuple(output_adjoint(i, reference_state[i]) for i in range(count))
        adjoints += tuple(torch.zeros_like(value, device=device)
                          for value in reference_state[count:ctx.optimizer_size])
        adjoints += tuple(output_adjoint(count + i, value)
                          for i, value in enumerate(reference_state[ctx.optimizer_size:]))
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
                final = update_segment(ctx.training, ctx.names, ctx.buffer_names, inputs, labels,
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
    """Train exactly the configured full epochs; optionally replay exact gradients.

    CNN weights, optimizer moments, and support BatchNorm statistics persist within a task. Support metrics
    are diagnostic only; the final epoch supplies the classifier. A zero checkpoint interval enables ordinary
    full unrolling for small reference checks; final per-mouse fitting needs no
    hypergradient or stored training trajectory.
    """
    if not len(labels) or len(features) != len(labels):
        raise ValueError("Classifier fitting requires nonempty matching support features/labels")
    if differentiable and config.inner_checkpoint_steps:
        report = {}
        state = _ReplayClassifierFit.apply(features, labels, reference, config, progress, report)
        count = sum(1 for _ in reference.parameters())
        return ClassifierFit(dict(zip((name for name, _ in reference.named_parameters()), state[:count])),
                             dict(zip((name for name, _ in reference.named_buffers()), state[count:])), report)
    return run_fit(reference, features, labels, config, differentiable, progress)
