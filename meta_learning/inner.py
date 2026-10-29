"""Epoch-based, support-only CNN fitting with exact checkpointed hypergradients."""

import copy
from dataclasses import dataclass
import math

import torch
from torch.func import functional_call
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from models.normalization import AxisBatchNorm
from meta_learning.common import digest


@dataclass
class ClassifierFit:
    parameters: dict
    buffers: dict
    report: dict
    source: object = None


def classifier_loss(classifier, parameters, buffers, features, labels, config, *, return_accuracy=False,
                    sample_weights=None):
    """Optionally event-weighted loss; evaluation never mutates classifier or BN state."""
    total = features.new_zeros(())
    correct = features.new_zeros(())
    for start in range(0, len(labels), config.batch_size):
        stop = start + config.batch_size
        logits = functional_call(classifier, (parameters, buffers), (features[start:stop],), strict=True)
        losses = F.cross_entropy(logits, labels[start:stop], reduction="none",
                                 label_smoothing=config.label_smoothing)
        weights = 1 if sample_weights is None else sample_weights[start:stop]
        total = total + (losses * weights).sum()
        if return_accuracy:
            predicted = (logits[:, 1] >= logits[:, 0]).long()
            correct = correct + ((predicted == labels[start:stop]) * weights).sum()
    denominator = len(labels) if sample_weights is None else sample_weights.sum()
    loss = total / denominator
    return (loss, float(correct / denominator)) if return_accuracy else loss


def training_classifier(reference, config):
    training = copy.deepcopy(reference).train()
    for layer in training.modules():
        if isinstance(layer, AxisBatchNorm):
            layer.differentiable_stats = True
        elif isinstance(layer, torch.nn.modules.dropout._DropoutNd):
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


def update_segment(training, names, buffers, features, labels, state, batches, config, differentiable):
    count = len(names)
    parameters = state[:count]
    # Functional BatchNorm replaces these values out of place on every support
    # forward. Include the EMA state in the trajectory and its hypergradient.
    buffers = dict(zip(buffers, state[count:]))
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
        if config.inner_weight_decay:
            # Match plain SGD's zero-centered L2 term after loss-gradient clipping.
            gradients = tuple(gradient + config.inner_weight_decay * parameter
                              for gradient, parameter in zip(gradients, parameters))
        parameters = tuple(parameter - config.inner_lr * gradient
                           for parameter, gradient in zip(parameters, gradients))
        if not differentiable:
            parameters = tuple(value.detach().requires_grad_(True) for value in parameters)
            buffers = {name: value.detach() for name, value in buffers.items()}
    return (*parameters, *buffers.values())


class _PoolFeatures(torch.autograd.Function):
    """Encode once; apply the summed feature adjoint to the frontend in chunks."""

    @staticmethod
    def forward(ctx, frontend, support, *parameters):
        ctx.frontend, ctx.support = frontend, support
        ctx.save_for_backward(*parameters)
        device = parameters[0].device
        return torch.cat([frontend(batch.to(device)) for batch in support.waveforms()])

    @staticmethod
    @once_differentiable
    def backward(ctx, feature_gradient):
        parameters = ctx.saved_tensors
        active = tuple((index, value) for index, value in enumerate(parameters) if value.requires_grad)
        gradients = [torch.zeros_like(value) for value in parameters]
        device = parameters[0].device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            start = 0
            for batch in ctx.support.waveforms(replay=True):
                features = ctx.frontend(batch.to(device))
                values = torch.autograd.grad(features, tuple(value for _, value in active),
                    grad_outputs=feature_gradient[start:start + len(batch)], allow_unused=True)
                for (index, _), value in zip(active, values):
                    if value is not None:
                        gradients[index].add_(value)
                start += len(batch)
        return None, None, *gradients


class FeatureEpochs:
    """Cached pool features with replayable selections, or fixed support features."""

    def __init__(self, features=None, labels=None, *, frontend=None, support=None):
        self.support = support
        self.sampled = support is not None
        if self.sampled:
            features = _PoolFeatures.apply(frontend, support, *frontend.parameters())
            labels = torch.as_tensor(support.labels, device=features.device)
        self.features, self.labels = features, labels
        self.inputs = (features,)
        self.device = features.device
        self.samples = len(support) if self.sampled else len(labels)
        self.epoch_indices = []

    def batch(self, epoch, *, differentiable=False, replay=False):
        features = self.features if differentiable else self.features.detach()
        if not self.sampled:
            return features, self.labels
        if not replay and epoch == len(self.epoch_indices):
            self.epoch_indices.append(torch.as_tensor(self.support.draw(epoch), device=self.device))
        indices = self.epoch_indices[epoch]
        return features[indices], self.labels[indices]

    def fitted(self, *, differentiable=False):
        features = self.features if differentiable else self.features.detach()
        return features, self.labels

    def add_gradient(self, total, epoch, gradient):
        if self.sampled:
            total.index_add_(0, self.epoch_indices[epoch], gradient)
        else:
            total.add_(gradient)


def run_fit(reference, source, config, differentiable, progress, segments=None):
    training = training_classifier(reference, config)
    evaluation = copy.deepcopy(training).eval()
    names = tuple(name for name, _ in reference.named_parameters())
    weights = tuple(parameter.detach().clone().requires_grad_(True) for parameter in reference.parameters())
    buffers = {name: value.detach().clone() for name, value in reference.named_buffers()}

    def measure(features, labels):
        with torch.no_grad():
            loss, accuracy = classifier_loss(evaluation, dict(zip(names, weights)), buffers,
                                             features.detach(), labels, config, return_accuracy=True)
        if not math.isfinite(float(loss)):
            raise FloatingPointError("Non-finite classifier support loss")
        return float(loss), accuracy

    steps, history = 0, []
    initial_loss = initial_accuracy = previous_loss = None
    for epoch in range(config.inner_epochs):
        features, labels = source.batch(epoch, differentiable=differentiable)
        if not len(labels) or len(features) != source.samples or len(labels) != source.samples:
            raise ValueError("Every support epoch requires the same nonempty sample count")
        if epoch == 0:
            initial_loss, initial_accuracy = measure(features, labels)
            previous_loss = initial_loss
        order = torch.randperm(len(labels), device=features.device)
        batches = tuple(order[start:start + config.batch_size] for start in range(0, len(labels), config.batch_size))
        segment_size = config.inner_checkpoint_steps or 1
        for start in range(0, len(batches), segment_size):
            selected = batches[start:start + segment_size]
            if segments is not None:
                segments.append(dict(epoch=epoch,
                    state=tuple(value.detach().cpu().clone() for value in (*weights, *buffers.values())),
                    batches=tuple(indices.cpu() for indices in selected), cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(features.device) if features.is_cuda else None))
            state = update_segment(training, names, buffers, features, labels,
                                   (*weights, *buffers.values()), selected, config, differentiable)
            weights = state[:len(names)]
            buffers = dict(zip(buffers, state[len(names):]))
            steps += len(selected)
        completed = epoch + 1
        if completed == config.inner_epochs or (config.inner_log_every > 0 and
                (completed == 1 or completed % config.inner_log_every == 0)):
            loss, accuracy = measure(features, labels)
            row = dict(epoch=completed, optimizer_steps=steps, support_loss=loss, support_accuracy=accuracy,
                       relative_loss_change=abs(previous_loss-loss)/max(abs(previous_loss), 1e-12))
            previous_loss = loss
            history.append(row)
            if progress:
                progress(row)
        del features, labels
    report = dict(samples=source.samples, epochs_requested=config.inner_epochs, epochs_run=config.inner_epochs,
        optimizer_steps=steps, initial_support_loss=initial_loss, initial_support_accuracy=initial_accuracy,
        diagnostic_epochs=[row["epoch"] for row in history], diagnostic_evaluations=1+len(history),
        optimizer="sgd", classifier_dropout=config.classifier_dropout,
        batchnorm="Support batch statistics during training; support EMA during evaluation",
        support_sampling="Per-epoch draws without replacement from a fit-local crop pool" if source.sampled else "Fixed support windows",
        support_pool_crops=config.support_pool_crops if source.sampled else 0,
        support_pool_samples=len(source.labels) if source.sampled else 0,
        support_pool_rng_sha256=digest(source.support.state) if source.sampled else None,
        support_pool_selections_sha256=digest([row.tolist() for row in source.support.selections]) if source.sampled else None,
        last_support_accuracy=history[-1]["support_accuracy"], last_support_loss=history[-1]["support_loss"],
        last_support_scope="Last epoch draw" if source.sampled else "All fixed support windows",
        stop_reason="fixed_epochs", history=history)
    return ClassifierFit(dict(zip(names, weights)), buffers, report, source)


class _ReplayClassifierFit(torch.autograd.Function):
    """Replay every CNN/BN update and sum adjoints into the shared feature pool."""

    @staticmethod
    def forward(ctx, reference, source, config, progress, report, *inputs):
        ctx.segments = []
        with torch.enable_grad():
            fit = run_fit(reference, source, config, False, progress, ctx.segments)
        ctx.state_size = len(ctx.segments[0]["state"])
        boundaries = tuple(value for segment in ctx.segments for value in segment.pop("state"))
        ctx.input_count = len(inputs)
        ctx.save_for_backward(*inputs, *boundaries)
        ctx.source, ctx.config = source, config
        ctx.training = training_classifier(reference, config)
        ctx.names, ctx.buffer_names = tuple(fit.parameters), tuple(fit.buffers)
        report.update(fit.report)
        return tuple(value.detach() for value in (*fit.parameters.values(), *fit.buffers.values()))

    @staticmethod
    @once_differentiable
    def backward(ctx, *output_gradients):
        saved = ctx.saved_tensors
        inputs, boundaries = saved[:ctx.input_count], saved[ctx.input_count:]
        count, device = len(ctx.names), ctx.source.device
        last = boundaries[-ctx.state_size:]
        adjoints = tuple(value if value is not None else torch.zeros_like(last[i], device=device)
                         for i, value in enumerate(output_gradients))
        input_gradients = [torch.zeros_like(value) for value in inputs]
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            for epoch in reversed(range(ctx.config.inner_epochs)):
                features, labels = ctx.source.batch(epoch, replay=True)
                feature_gradient = torch.zeros_like(features)
                positions = [i for i, segment in enumerate(ctx.segments) if segment["epoch"] == epoch]
                for position in reversed(positions):
                    segment = ctx.segments[position]
                    torch.set_rng_state(segment["cpu_rng"])
                    if device.type == "cuda":
                        torch.cuda.set_rng_state(segment["cuda_rng"], device)
                    state = boundaries[position*ctx.state_size:(position+1)*ctx.state_size]
                    initial = tuple(value.to(device).detach().requires_grad_(True) for value in state)
                    values = features.detach().requires_grad_(True)
                    batches = tuple(indices.to(device) for indices in segment["batches"])
                    final = update_segment(ctx.training, ctx.names, ctx.buffer_names, values, labels,
                                           initial, batches, ctx.config, True)
                    gradients = torch.autograd.grad(final, (values, *initial), grad_outputs=adjoints, allow_unused=True)
                    if gradients[0] is not None:
                        feature_gradient.add_(gradients[0])
                    adjoints = tuple(value.detach() if value is not None else torch.zeros_like(initial[i])
                                     for i, value in enumerate(gradients[1:]))
                    del initial, values, final, gradients
                ctx.source.add_gradient(input_gradients[0], epoch, feature_gradient)
                del features, labels, feature_gradient
        return None, None, None, None, None, *input_gradients


def fit_classifier(reference, features, labels, config, *, differentiable, progress=None):
    """One fixed-epoch SGD solver for fixed features or support pool selections.

    Replay retains all CNN/BN boundaries, epoch indices and RNG states, sharing
    one feature pool across every epoch. No hypergradients are cut.
    """
    source = features if isinstance(features, FeatureEpochs) else FeatureEpochs(features, labels)
    if not source.samples:
        raise ValueError("Classifier fitting requires nonempty support")
    if differentiable and config.inner_checkpoint_steps:
        report = {}
        state = _ReplayClassifierFit.apply(reference, source, config, progress, report, *source.inputs)
        count = sum(1 for _ in reference.parameters())
        return ClassifierFit(dict(zip((name for name, _ in reference.named_parameters()), state[:count])),
            dict(zip((name for name, _ in reference.named_buffers()), state[count:])), report, source)
    return run_fit(reference, source, config, differentiable, progress)


class _ReplaySupportLoss(torch.autograd.Function):
    """Uniform final-weight loss over the whole pool, with bounded-memory exact VJPs."""

    @staticmethod
    def forward(ctx, source, evaluation, names, buffer_names, config, report, *values):
        ctx.source, ctx.evaluation, ctx.config = source, evaluation, config
        ctx.names, ctx.buffer_names = names, buffer_names
        ctx.state_count = len(names)+len(buffer_names)
        ctx.save_for_backward(*values)
        parameters = dict(zip(names, values[:len(names)]))
        buffers = dict(zip(buffer_names, values[len(names):ctx.state_count]))
        features, labels = source.fitted()
        loss, accuracy = classifier_loss(evaluation, parameters, buffers, features, labels, config,
                                         return_accuracy=True)
        report.update(accuracy=accuracy)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        saved = ctx.saved_tensors
        state = tuple(value.detach().requires_grad_(True) for value in saved[:ctx.state_count])
        gradients = [torch.zeros_like(value) for value in saved]
        parameters = dict(zip(ctx.names, state[:len(ctx.names)]))
        buffers = dict(zip(ctx.buffer_names, state[len(ctx.names):]))
        features, labels = ctx.source.fitted()
        total_samples = len(labels)
        device = ctx.source.device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            for start in range(0, len(labels), ctx.config.batch_size):
                part = features[start:start + ctx.config.batch_size].detach().requires_grad_(True)
                targets = labels[start:start + ctx.config.batch_size]
                loss = classifier_loss(ctx.evaluation, parameters, buffers, part, targets, ctx.config)
                values = torch.autograd.grad(loss, (part, *state),
                    grad_outputs=upstream * (len(targets) / total_samples), allow_unused=True)
                if values[0] is not None:
                    gradients[ctx.state_count][start:start + len(targets)] = values[0]
                for total, value in zip(gradients[:ctx.state_count], values[1:]):
                    if value is not None:
                        total.add_(value)
        return None, None, None, None, None, None, *gradients


def fitted_support_loss(reference, fit, config, *, differentiable):
    """Average every support pool slot equally at the final CNN/BN state."""
    evaluation = copy.deepcopy(reference).eval()
    source = fit.source
    if differentiable and config.inner_checkpoint_steps and source.sampled:
        report = {}
        loss = _ReplaySupportLoss.apply(source, evaluation, tuple(fit.parameters), tuple(fit.buffers),
            config, report, *fit.parameters.values(), *fit.buffers.values(), *source.inputs)
        accuracy = report["accuracy"]
    else:
        with torch.set_grad_enabled(differentiable):
            features, labels = source.fitted(differentiable=differentiable)
            loss, accuracy = classifier_loss(evaluation, fit.parameters, fit.buffers, features, labels,
                                             config, return_accuracy=True)
    draws = len(source.epoch_indices) if source.sampled else 1
    samples = len(source.labels)
    fit.report.update(last_support_loss=float(loss.detach()), last_support_accuracy=accuracy,
        last_support_scope="All support pool crops with equal weights at final weights" if source.sampled else "All fixed support windows",
        fitted_support_samples=samples, support_draws=draws,
        support_training_draw_samples=source.samples * draws,
        support_pool_slots_used=int((source.support.usage > 0).sum()) if source.sampled else samples)
    return loss
