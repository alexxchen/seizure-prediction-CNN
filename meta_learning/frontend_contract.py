"""Fixed waveform input and checks for editable frontend/CNN architectures."""

import json

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from meta_learning.frontend import FRONTEND_CONFIG, WaveformFrontend

INPUT_SAMPLES = 28672
CNN_INPUT_SHAPE = tuple(WaveformFrontend.output_shape)
if len(CNN_INPUT_SHAPE) != 2 or any(type(value) is not int or value < 1 for value in CNN_INPUT_SHAPE):
    raise ValueError("WaveformFrontend.output_shape must declare positive integer (channels, time) dimensions")
if type(WaveformFrontend.projection_dim) is not int or WaveformFrontend.projection_dim < 1:
    raise ValueError("WaveformFrontend.projection_dim must declare a positive integer")


class Frontend(nn.Module):
    """Own initialization, chunking, and replay independently of candidate internals."""

    output_shape = CNN_INPUT_SHAPE
    projection_dim = WaveformFrontend.projection_dim

    def __init__(self, seed=0, batch_size=32):
        super().__init__()
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Frontend batch size must be a positive integer")
        self.batch_size = batch_size
        if any(not isinstance(FRONTEND_CONFIG.get(key), str) or not FRONTEND_CONFIG[key]
               for key in ("architecture", "input_normalization")):
            raise ValueError("Frontend metadata requires architecture and input_normalization descriptions")
        json.dumps(FRONTEND_CONFIG, allow_nan=False)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            self.network = WaveformFrontend(seed=seed)
        if not list(self.network.parameters()):
            raise ValueError("The frontend must expose trainable meta-parameters")

    def _forward_batch(self, waveform):
        features = self.network(waveform)
        if features.shape != (len(waveform), *CNN_INPUT_SHAPE):
            raise ValueError(f"Frontend must emit (N, {CNN_INPUT_SHAPE[0]}, {CNN_INPUT_SHAPE[1]})")
        return features

    def forward(self, waveform):
        if waveform.ndim != 2 or waveform.shape[1] != INPUT_SAMPLES or not len(waveform):
            raise ValueError(f"Expected nonempty preprocessed waveforms shaped (N, {INPUT_SAMPLES})")
        differentiable = torch.is_grad_enabled() and (
            waveform.requires_grad or any(p.requires_grad for p in self.parameters()))
        return torch.cat([checkpoint(self._forward_batch, batch, use_reentrant=False,
                                     preserve_rng_state=True)
                          if differentiable else self._forward_batch(batch)
                          for batch in waveform.split(self.batch_size)])

    def decay_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.ndim > 1]

    def _project_batch(self, features):
        projected = self.network.project(features)
        if projected.shape != (len(features), self.projection_dim):
            raise ValueError(f"Frontend projection must emit (N, {self.projection_dim})")
        return projected

    def project(self, features):
        """Apply the candidate's sample-local projection to cached features; never encode waveforms again."""
        if features.ndim != 3 or features.shape[1:] != CNN_INPUT_SHAPE or not len(features):
            raise ValueError(f"Expected nonempty frontend features shaped (N, {CNN_INPUT_SHAPE})")
        differentiable = torch.is_grad_enabled() and (
            features.requires_grad or any(parameter.requires_grad for parameter in self.parameters()))
        return torch.cat([checkpoint(self._project_batch, batch, use_reentrant=False, preserve_rng_state=True)
                          if differentiable else self._project_batch(batch)
                          for batch in features.split(self.batch_size)])

    def describe(self):
        return dict(FRONTEND_CONFIG, checkpoint_batch_size=self.batch_size,
                    parameters=sum(p.numel() for p in self.parameters()),
                    state=dict(weight_norms={name: float(p.detach().norm())
                               for name, p in self.named_parameters() if p.ndim > 1}))


def check_candidate(device="cpu"):
    """Finite synthetic checks; no private recordings or training jobs are used.

    These checks detect common contract violations, not adversarial candidate code.
    All transformations must remain sample-local and stateless across calls.
    """
    from meta_learning.model import MetaModel
    from dataclasses import replace
    from pathlib import Path
    from meta_learning.config import Config
    from meta_learning.algorithms import meta_objective

    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if devices else []
        model = MetaModel(cnn_seed=17, frontend_batch_size=2)
        if not torch.equal(before, torch.get_rng_state()) or any(
                not torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all() if devices else [])):
            raise ValueError("Seeded initialization advanced caller RNG streams")
        reference = MetaModel(cnn_seed=17, frontend_batch_size=1)
        if any(not torch.equal(value, reference.state_dict()[name])
               for name, value in model.state_dict().items()):
            raise ValueError("Seeded initialization is not reproducible")
        model = model.to(device)
        frontend = model.frontend
        if any(isinstance(layer, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd))
               for layer in frontend.modules()):
            raise ValueError("Frontend must be sample-local and deterministic; no BatchNorm or dropout")
        state = {name: value.detach().clone() for name, value in frontend.state_dict().items()}
        x = torch.randn(2, INPUT_SAMPLES, device=device)
        mode_features = None
        mode_projection = None
        for training in (True, False):
            frontend.train(training)
            rng_before = torch.get_rng_state().clone()
            device_rng_before = torch.cuda.get_rng_state_all() if devices else []
            with torch.no_grad():
                together = frontend(x)
                projection = frontend.project(together)
                frontend.batch_size = 1
                alone = frontend(x)
                projection_alone = frontend.project(alone)
                frontend.batch_size = 2
                changed = frontend(torch.stack((x[0], x[1] * 7 + 3)))
                projection_changed = frontend.project(changed)
                silent = frontend(torch.zeros_like(x[:1]))
                projection_silent = frontend.project(silent)
            if not torch.equal(rng_before, torch.get_rng_state()) or any(
                    not torch.equal(a, b) for a, b in zip(
                        device_rng_before, torch.cuda.get_rng_state_all() if devices else [])):
                raise ValueError("Frontend forward consumed random numbers")
            if not all(torch.isfinite(v).all() for v in (
                    together, alone, changed, silent, projection, projection_alone, projection_changed, projection_silent)):
                raise ValueError("Frontend emitted non-finite features")
            if not torch.allclose(together, alone, atol=2e-5, rtol=2e-4) or not torch.allclose(
                    together[:1], changed[:1], atol=2e-5, rtol=2e-4):
                raise ValueError("Frontend features depend on other samples or chunk size")
            if not torch.allclose(projection, projection_alone, atol=2e-5, rtol=2e-4) or not torch.allclose(
                    projection[:1], projection_changed[:1], atol=2e-5, rtol=2e-4):
                raise ValueError("Frontend projection depends on other samples or chunk size")
            if mode_features is not None and not torch.allclose(together, mode_features, atol=2e-5, rtol=2e-4):
                raise ValueError("Frontend features differ between training and evaluation")
            if mode_projection is not None and not torch.allclose(projection, mode_projection, atol=2e-5, rtol=2e-4):
                raise ValueError("Frontend projection differs between training and evaluation")
            mode_features = together
            mode_projection = projection
            if any(not torch.equal(value, frontend.state_dict()[name]) for name, value in state.items()):
                raise ValueError("Frontend mutated persistent state during inference")
        frontend.train()
        values = x[:1].detach().requires_grad_(True)
        loss = frontend(values).square().mean()
        gradients = torch.autograd.grad(loss, (values, *frontend.parameters()), allow_unused=True)
        if not all(g is None or torch.isfinite(g).all() for g in gradients):
            raise ValueError("Non-finite frontend gradient")
        if gradients[0] is None or not gradients[0].abs().sum() or not any(
                g is not None and g.abs().sum() > 0 for g in gradients[1:]):
            raise ValueError("Frontend disconnected its waveform or meta-parameters")
        if any(isinstance(layer, nn.modules.batchnorm._BatchNorm) for layer in model.cnn.modules()):
            raise ValueError("CNN BatchNorm must use models.normalization.AxisBatchNorm for full hypergradients")
        reference_state = {name: value.detach().clone() for name, value in model.cnn.state_dict().items()}
        model.cnn.eval()
        with torch.no_grad():
            logits = model.cnn(together)
            repeated = model.cnn(together)
        if logits.shape != (len(x), 2) or not torch.isfinite(logits).all() or not torch.equal(logits, repeated):
            raise ValueError("CNN must emit finite deterministic (N, 2) logits in evaluation mode")
        # This tiny synthetic fit checks architecture compatibility, not candidate performance.
        # Real experiments always complete their declared configuration budget.
        config = Config.load(Path(__file__).resolve().parent.parent / "configs/meta.json")
        diagnostic = replace(config, inner_epochs=1, batch_size=2, inner_checkpoint_steps=1)
        labels = torch.tensor([0, 1], device=device)
        objective = meta_objective(model, (x, labels), (x.flip(0), labels.flip(0)), diagnostic)
        hypergradients = torch.autograd.grad(objective, tuple(frontend.parameters()), allow_unused=True)
        if not torch.isfinite(objective) or not all(
                gradient is None or torch.isfinite(gradient).all() for gradient in hypergradients):
            raise ValueError("CNN fit produced a non-finite frontend hypergradient")
        if not any(gradient is not None and gradient.abs().sum() > 0 for gradient in hypergradients):
            raise ValueError("CNN fit disconnected the frontend hypergradient")
        if any(not torch.equal(value, model.cnn.state_dict()[name]) for name, value in reference_state.items()):
            raise ValueError("CNN evaluation or fitting mutated its unlearned reference state")
        return dict(input_samples=INPUT_SAMPLES, output_shape=list(CNN_INPUT_SHAPE),
                    projection_dim=frontend.projection_dim,
                    parameters=sum(p.numel() for p in frontend.parameters()),
                    cnn_parameters=sum(p.numel() for p in model.cnn.parameters()), device=str(device),
                    coverage="Synthetic feature/projection shapes, initialization RNG, chunk/sample isolation, state, silence, "
                             "frontend gradients, CNN logits and checkpointed query, final-support and supervised contrastive hypergradients")


if __name__ == "__main__":
    import sys
    print(json.dumps(check_candidate(sys.argv[1] if len(sys.argv) > 1 else "cpu")))
