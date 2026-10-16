"""Checks for true hypergradients, temporal isolation, resumption, and inference."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn

from meta_learning.algorithms import meta_objective
from meta_learning.config import Config
from meta_learning.compare import compare_runs
from meta_learning.data import FeatureSplit
from meta_learning.engine import evaluate_run, train_run
from meta_learning.episodes import chronological_episodes, episode_batch
from meta_learning.model import MetaModel


def synthetic_split(mouse, split, days):
    records = [dict(mouse=mouse, day=day, raw_file=f"{mouse}-{day}", start_row=i * 10,
                    stop_row=i * 10 + 5, label=i % 2) for day in days for i in range(4)]
    table = pd.DataFrame(records)
    power = np.random.default_rng(12).uniform(.5, 1.5, (len(table), 79, 8)).astype(np.float32)
    return FeatureSplit(mouse, split, power, table.label.to_numpy(), table)


class FakeStore:
    def __init__(self, signature):
        self.dataset = SimpleNamespace(signature=signature)
        self.accesses = []
        self.values = {part: synthetic_split("X1", part, days) for part, days in
                       {"train": [0, 1, 2, 3], "validation": [8, 11], "test": [12, 13]}.items()}

    def load(self, mouse, split):
        self.accesses.append(split)
        return self.values[split]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.cnn = nn.Linear(1, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.frontend.weight.fill_(.7)
            self.cnn.weight.copy_(torch.tensor([[-.4], [.6]], dtype=torch.float64))

    def forward(self, inputs):
        return self.cnn(self.frontend(inputs))


class MetaLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_hypergradient_matches_finite_difference_and_includes_adaptation(self):
        model = TinyModel()
        config = Config(inner_steps=2, inner_lr=.2, label_smoothing=0)
        support = (torch.tensor([[-2.], [-1.], [1.], [2.]], dtype=torch.float64), torch.tensor([0, 0, 1, 1]))
        query = (torch.tensor([[-3.], [-.3], [.8], [2.4]], dtype=torch.float64), torch.tensor([0, 0, 1, 1]))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        loss = meta_objective(model, support, query, config)
        actual = torch.autograd.grad(loss, model.frontend.weight)[0].item()
        values = []
        epsilon = 1e-6
        for offset in [epsilon, -epsilon]:
            with torch.no_grad():
                model.frontend.weight.copy_(before["frontend.weight"] + offset)
            values.append(meta_objective(model, support, query, config).item())
        with torch.no_grad():
            model.frontend.weight.copy_(before["frontend.weight"])
        self.assertAlmostEqual(actual, (values[0] - values[1]) / (2 * epsilon), places=7)
        # A first-order shortcut would lose the frontend -> inner update path.
        weight = model.cnn.weight
        for _ in range(config.inner_steps):
            logits = torch.nn.functional.linear(model.frontend(support[0]), weight)
            gradient = torch.autograd.grad(torch.nn.functional.cross_entropy(logits, support[1]), weight)[0]
            weight = weight - config.inner_lr * gradient.detach()
        logits = torch.nn.functional.linear(model.frontend(query[0]), weight)
        shortcut = torch.autograd.grad(torch.nn.functional.cross_entropy(logits, query[1]), model.frontend.weight)[0].item()
        self.assertGreater(abs(actual - shortcut), 1e-4)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_chronology_and_support_only_normalization(self):
        split = synthetic_split("X1", "train", [0, 1, 2, 3])
        episodes = chronological_episodes(split)
        self.assertEqual(len(episodes), 2)
        for episode in episodes:
            self.assertLess(max(episode.support_days), min(episode.query_days))
            self.assertFalse(np.intersect1d(episode.support_indices, episode.query_indices).size)
        config = Config(support_per_class=2, query_per_class=2)
        support1, _ = episode_batch(split, episodes[0], config, np.random.default_rng(1), "cpu")
        split.power[episodes[0].query_indices] *= 1e6
        support2, _ = episode_batch(split, episodes[0], config, np.random.default_rng(1), "cpu")
        torch.testing.assert_close(support1[0], support2[0], rtol=0, atol=0)
        split.table.loc[0, "day"] = 11
        with self.assertRaises(ValueError):
            chronological_episodes(split)

    def test_query_prediction_has_no_batch_companion_dependence(self):
        torch.manual_seed(0)
        model = MetaModel().eval()
        power = torch.rand(3, 79, 8)
        with torch.no_grad():
            alone, together = model(power[:1]), model(power)[:1]
        torch.testing.assert_close(alone, together, rtol=1e-5, atol=1e-6)
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        config = Config(inner_steps=1, support_per_class=1, query_per_class=1)
        objective = meta_objective(model, (power[:2], torch.tensor([0, 1])),
                                   (power[1:], torch.tensor([1, 0])), config)
        objective.backward()
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.frontend.parameters()))

    def test_resume_is_exact_and_evaluation_never_refits(self):
        from meta_learning import engine
        config = Config(targets=("X1",), outer_steps=2, tasks_per_step=1, inner_steps=1,
                        eval_every=1, checkpoint_every=1, adaptation_steps=1,
                        support_per_class=1, query_per_class=1, batch_size=4)
        signature = dict(annotation_sha256="synthetic", sources={})
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "manifest.json").write_text(json.dumps(dict(signature=signature,
                day_splits={"train": list(range(8)), "validation": [8, 11], "test": [12, 13]})))
            full_store = FakeStore(signature)
            full = root / "full"
            train_run(config, full_store, full, prepared, device)
            self.assertNotIn("test", full_store.accesses)
            partial = root / "partial"
            original_validate = engine.validate

            def interrupt(*args, **kwargs):
                if args[4] == 2:
                    raise InterruptedError("Simulated interruption")
                return original_validate(*args, **kwargs)

            with mock.patch.object(engine, "validate", side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    train_run(config, FakeStore(signature), partial, prepared, device)
            train_run(config, FakeStore(signature), partial, prepared, device, resume=True)
            completed_status = (partial / "status.json").read_bytes()
            train_run(config, FakeStore(signature), partial, prepared, device, resume=True)
            self.assertEqual((partial / "status.json").read_bytes(), completed_status)
            checkpoints = [torch.load(path / "checkpoints" / "last.pt", weights_only=True) for path in [full, partial]]
            for name, value in checkpoints[0]["model"].items():
                torch.testing.assert_close(value, checkpoints[1]["model"][name], rtol=0, atol=0)
            self.assertEqual(checkpoints[0]["history"], checkpoints[1]["history"])
            self.assertEqual(checkpoints[0]["episode_rng"], checkpoints[1]["episode_rng"])
            validation_store = FakeStore(signature)
            with mock.patch.object(engine, "adapt_model", side_effect=AssertionError("Evaluation must not fit")):
                report = evaluate_run(validation_store, partial, device)
            self.assertEqual(validation_store.accesses, ["validation"])
            selected = json.loads((partial / "selection.json").read_text())
            self.assertEqual(report["aggregate"], selected["aggregate"])
            pooled_config = Config.from_dict(dict(config.to_dict(), algorithm="pooled"))
            pooled = root / "pooled"
            train_run(pooled_config, FakeStore(signature), pooled, prepared, device)
            comparison = compare_runs(partial, pooled)
            self.assertEqual(set(comparison["aggregate"]), {"meta", "pooled"})
            with self.assertRaises(ValueError):
                compare_runs(partial, full)
            # Test-split coverage here uses synthetic arrays exclusively.
            test_store = FakeStore(signature)
            evaluate_run(test_store, partial, device, "test")
            self.assertEqual(test_store.accesses, ["test"])
            checkpoint_path = partial / selected["adapted_models"]["X1"]["path"]
            with checkpoint_path.open("ab") as file:
                file.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                evaluate_run(FakeStore(signature), partial, device)

    def test_invalid_configuration_is_rejected(self):
        for changes in [dict(inner_steps=0), dict(outer_lr=float("nan")),
                        dict(targets=["X1", "X1"]), dict(algorithm="search"), dict(unknown=True)]:
            with self.assertRaises(ValueError):
                Config.from_dict(changes)


if __name__ == "__main__":
    unittest.main()
