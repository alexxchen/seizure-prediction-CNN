"""Behavioral checks for spectral coordinates, learning, and checkpoint reuse."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from meta_learning import frontend
from meta_learning.model import CNN_INPUT_SHAPE, MetaModel
from models.cnn import SeizureCNN


class LearnableFrontendTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_fixed_basis_preserves_frequency_and_removes_dc(self):
        times = np.arange(2000, dtype=np.float32) / 500
        signal = np.sin(2 * np.pi * 10 * times)[None]
        power = frontend.fixed_power_batch(signal)
        shifted = frontend.fixed_power_batch(signal + 100)
        coordinates = frontend.SpectralFrontend().frequencies.numpy()
        self.assertAlmostEqual(float(coordinates[power.mean(axis=(0, 2)).argmax()]), 10)
        np.testing.assert_allclose(power, shifted, rtol=1e-3, atol=1e-4)
        self.assertTrue(np.isfinite(power).all())

    def test_shared_initialization_and_actual_frontend_gradients(self):
        torch.manual_seed(0)
        reference = SeizureCNN(CNN_INPUT_SHAPE)
        torch.manual_seed(0)
        frozen = MetaModel()
        frozen.frontend.requires_grad_(False)
        torch.manual_seed(0)
        learned = MetaModel()
        for name, parameter in reference.state_dict().items():
            self.assertTrue(torch.equal(parameter, learned.cnn.state_dict()[name]))
        power = torch.rand(4, 79, 64)
        frozen.eval()
        learned.eval()
        self.assertEqual(tuple(learned.frontend(power).shape), (4, 1, 1, 55, 114))
        torch.testing.assert_close(frozen(power), learned(power), rtol=0, atol=0)
        labels = torch.tensor([0, 1, 0, 1])
        optimizer = torch.optim.Adam(learned.parameters(), lr=5e-4)
        initial = {name: p.detach().clone() for name, p in learned.frontend.named_parameters()}
        torch.nn.functional.cross_entropy(learned(power), labels).backward()
        for name, parameter in learned.frontend.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        optimizer.step()
        for name, parameter in learned.frontend.named_parameters():
            self.assertFalse(torch.equal(parameter, initial[name]), name)
        torch.nn.functional.cross_entropy(frozen(power), labels).backward()
        self.assertTrue(all(p.grad is None for p in frozen.frontend.parameters()))

    def test_fixed_basis_cache_reuse_and_integrity(self):
        rng = np.random.default_rng(3)
        raw = rng.normal(size=(8, 2000)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            values = frontend.fixed_power(raw, cache)
            np.testing.assert_array_equal(values, frontend.fixed_power_batch(raw))
            with mock.patch.object(frontend, "fixed_power_batch", side_effect=AssertionError("Cache was not reused")):
                np.testing.assert_array_equal(values, frontend.fixed_power(raw, cache))
            with Path(values.filename).open("ab") as file:
                file.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "cache changed"):
                frontend.fixed_power(raw, cache)


if __name__ == "__main__":
    unittest.main()
