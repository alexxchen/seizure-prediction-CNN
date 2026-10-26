"""Synthetic regression checks for the final-support outer objective."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from meta_learning.algorithms import meta_objective
from meta_learning.config import Config
from models.normalization import AxisBatchNorm


class ToyModel(nn.Module):
    def __init__(self, batchnorm=False):
        super().__init__()
        self.frontend = nn.Linear(2, 3)
        self.cnn = (nn.Sequential(AxisBatchNorm(3, axis=1), nn.Tanh(), nn.Dropout(), nn.Linear(3, 2))
                    if batchnorm else nn.Linear(3, 2))
        self.double()


class MetaObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(231)
        self.support = (torch.randn(6, 2, dtype=torch.double), torch.tensor([0, 0, 1, 1, 0, 1]))
        self.query = (torch.randn(4, 2, dtype=torch.double), torch.tensor([1, 0, 1, 0]))
        self.weights = torch.tensor([1., 2., 3., 2.], dtype=torch.double)
        self.config = Config(inner_epochs=3, batch_size=2, inner_lr=0.07,
                             inner_checkpoint_steps=0, label_smoothing=0.1, gradient_clip=1e6)

    def evaluate(self, model, config):
        torch.manual_seed(71)
        loss, report = meta_objective(model, self.support, self.query, config,
                                      query_weights=self.weights, return_diagnostics=True)
        rng = torch.get_rng_state().clone()
        gradients = torch.autograd.grad(loss, tuple(model.frontend.parameters()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()), "Replay changed the caller RNG")
        return loss.detach(), torch.cat([g.flatten() for g in gradients]), report

    def manual_components(self, model):
        """Independent plain-SGD reference, including the final minibatch update."""
        features = model.frontend(self.support[0])
        weight = model.cnn.weight.detach().clone().requires_grad_()
        bias = model.cnn.bias.detach().clone().requires_grad_()
        torch.manual_seed(71)
        for _ in range(self.config.inner_epochs):
            order = torch.randperm(len(features))
            for indices in order.split(self.config.batch_size):
                before = (weight, bias)
                loss = F.cross_entropy(F.linear(features[indices], weight, bias), self.support[1][indices],
                                       label_smoothing=self.config.label_smoothing)
                gradients = torch.autograd.grad(loss, before, create_graph=True)
                weight, bias = tuple(p - self.config.inner_lr * g for p, g in zip(before, gradients))
        support_loss = F.cross_entropy(F.linear(features, weight, bias), self.support[1],
                                       label_smoothing=self.config.label_smoothing)
        query_losses = F.cross_entropy(F.linear(model.frontend(self.query[0]), weight, bias), self.query[1],
                                       label_smoothing=self.config.label_smoothing, reduction="none")
        query_loss = (query_losses * self.weights).sum() / self.weights.sum()
        pre_update = F.cross_entropy(F.linear(features, *before), self.support[1],
                                    label_smoothing=self.config.label_smoothing)
        last_batch = F.cross_entropy(F.linear(features[indices], weight, bias), self.support[1][indices],
                                    label_smoothing=self.config.label_smoothing)
        return query_loss, support_loss, pre_update, last_batch

    def test_full_support_after_last_update_and_coefficient_gradients(self):
        model = ToyModel()
        for coefficient in (0.0, 0.3, 1.0):
            with self.subTest(coefficient=coefficient):
                query, support, before, last_batch = self.manual_components(model)
                self.assertGreater(abs(float((support - before).detach())), 1e-5)
                self.assertGreater(abs(float((support - last_batch).detach())), 1e-5)
                expected = query + coefficient * support
                expected_gradients = torch.autograd.grad(expected, tuple(model.frontend.parameters()))
                actual, gradients, report = self.evaluate(model, replace(self.config, support_loss_weight=coefficient))
                torch.testing.assert_close(actual, expected.detach(), atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(gradients, torch.cat([g.flatten() for g in expected_gradients]),
                                           atol=1e-12, rtol=1e-12)
                self.assertAlmostEqual(report["query"]["loss"], float(query.detach()), places=12)
                self.assertAlmostEqual(report["outer_objective"]["support_loss_after_last_update"],
                                       float(support.detach()), places=12)
                self.assertAlmostEqual(report["outer_objective"]["total_loss"], float(actual), places=12)
        torch.manual_seed(71)
        plain = meta_objective(model, self.support, self.query, replace(self.config, support_loss_weight=0),
                               query_weights=self.weights)
        query, _, _, _ = self.manual_components(model)
        self.assertTrue(torch.equal(plain, query), "Zero must preserve the query-only objective exactly")

    def test_checkpoint_replay_with_batchnorm_dropout_and_state_isolation(self):
        model = ToyModel(batchnorm=True)
        state = copy.deepcopy(model.state_dict())
        config = replace(self.config, classifier_dropout=0.2, support_loss_weight=0.4)
        reference, reference_grad, _ = self.evaluate(model, config)
        replay, replay_grad, report = self.evaluate(model, replace(config, inner_checkpoint_steps=2))
        torch.testing.assert_close(replay, reference, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(replay_grad, reference_grad, atol=1e-10, rtol=1e-10)
        self.assertTrue(torch.isfinite(replay_grad).all())
        self.assertGreater(float(replay_grad.abs().sum()), 0)
        self.assertEqual(report["epochs_run"], config.inner_epochs)
        self.assertEqual(report["optimizer_steps"], 9)
        self.assertAlmostEqual(report["calibrated_support_loss"],
                               report["outer_objective"]["support_loss_after_last_update"], places=12)
        _, query_grad, _ = self.evaluate(model, replace(config, support_loss_weight=0))
        self.assertGreater(float((query_grad - replay_grad).abs().sum()), 1e-6)
        for name, value in state.items():
            self.assertTrue(torch.equal(model.state_dict()[name], value), name)
        self.assertTrue(all(parameter.grad is None for parameter in model.cnn.parameters()))

    def test_full_hypergradient_matches_finite_difference(self):
        model = ToyModel(batchnorm=True)
        config = replace(self.config, support_loss_weight=0.4, inner_checkpoint_steps=2)
        _, gradient, _ = self.evaluate(model, config)
        parameter = model.frontend.weight
        origin = parameter[0, 0].item()
        epsilon = 1e-5
        values = []
        for delta in (epsilon, -epsilon):
            with torch.no_grad():
                parameter[0, 0] = origin + delta
            value, _, _ = self.evaluate(model, config)
            values.append(value)
        with torch.no_grad():
            parameter[0, 0] = origin
        numerical = (values[0] - values[1]) / (2 * epsilon)
        torch.testing.assert_close(gradient[0], numerical, atol=1e-7, rtol=1e-5)

    def test_coefficient_validation_and_roundtrip(self):
        for value in (0, 0.1, 2.0):
            config = replace(self.config, support_loss_weight=value).validate()
            self.assertEqual(Config.from_dict(config.to_dict()), config)
        for value in (-1, float("nan"), float("inf"), True, "0.1", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "support_loss_weight"):
                replace(self.config, support_loss_weight=value).validate()

    def test_protocol_allows_coefficient_candidates_and_locks_other_fields(self):
        import research
        root = Path(__file__).resolve().parents[1]
        lock = research.verify_protocol(root)
        self.assertEqual(lock["version"], 5)
        self.assertIn("support_loss_weight", research.EDITABLE_CONFIG_FIELDS)
        self.assertNotIn("support_loss_weight", lock["fixed_config"])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in (*lock["fixed_sha256"], *research.EDITABLE, "protocol.lock.json"):
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, destination)
            path = target / "configs/meta.json"
            config = json.loads(path.read_text())
            config["support_loss_weight"] = 0.75
            path.write_text(json.dumps(config))
            research.verify_protocol(target)
            config["seed"] += 1
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Fixed research configuration changed"):
                research.verify_protocol(target)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
