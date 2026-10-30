"""Standard SupCon pairing, exact gradients and zero-weight isolation."""

import copy
from dataclasses import replace
import unittest

import torch
from torch import nn

from meta_learning.algorithms import meta_objective
from meta_learning.config import Config
from meta_learning.contrastive import TEMPERATURE, frontend_embedding, supervised_contrastive_loss
from models.normalization import AxisBatchNorm


class ContrastiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matches_independent_pairwise_reference_and_chunk_gradients(self):
        features = torch.randn(8, 3, 5, generator=torch.Generator().manual_seed(2), dtype=torch.double,
                               requires_grad=True)
        labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
        full = features.reshape(len(features), -1)
        z = full / full.norm(dim=1, keepdim=True).clamp_min(1e-6)
        terms = []
        for i in range(len(labels)):
            candidates = [j for j in range(len(labels)) if j != i]
            positives = [j for j in candidates if labels[i] == labels[j]]
            if positives:
                scores = torch.stack([torch.dot(z[i], z[j]) / TEMPERATURE for j in candidates])
                numerator = torch.stack([torch.dot(z[i], z[j]) / TEMPERATURE for j in positives]).mean()
                terms.append(scores.logsumexp(0) - numerator)
        expected = torch.stack(terms).mean()
        reference_gradient = torch.autograd.grad(expected, features)[0]
        for chunk_size in (1, 3, 256):
            actual, report = supervised_contrastive_loss(features, labels, chunk_size=chunk_size)
            gradient = torch.autograd.grad(actual, features)[0]
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(gradient, reference_gradient, atol=1e-11, rtol=1e-11)
            self.assertEqual(report['valid_anchors'], 8)
            self.assertEqual(report['embedding_dimensions'], 15)

    def test_temporal_structure_is_retained_despite_identical_channel_moments(self):
        # Opposite temporal patterns have identical per-channel means and variances.
        pattern = torch.tensor([1., 0., -1., 0.], dtype=torch.double)
        features = torch.stack((pattern, pattern, -pattern, -pattern)).unsqueeze(1)
        torch.testing.assert_close(features.mean(-1), torch.zeros(4, 1, dtype=torch.double))
        torch.testing.assert_close(features.var(-1), features.var(-1)[:1].expand(4, 1))
        loss, _ = supervised_contrastive_loss(features, torch.tensor([0, 0, 1, 1]))
        mixed, _ = supervised_contrastive_loss(features, torch.tensor([0, 1, 0, 1]))
        self.assertLess(float(loss), 1e-6)
        self.assertGreater(float(mixed), 1.)

    def test_full_baseline_shape_and_chunked_gradients(self):
        features = torch.randn(4, 32, 256, generator=torch.Generator().manual_seed(6),
                               dtype=torch.double, requires_grad=True)
        z = frontend_embedding(features)
        self.assertEqual(z.shape, (4, 8192))
        torch.testing.assert_close(z.norm(dim=1), torch.ones(4, dtype=torch.double))
        labels = torch.tensor([0, 0, 1, 1])
        whole, report = supervised_contrastive_loss(features, labels, chunk_size=4)
        chunked, _ = supervised_contrastive_loss(features, labels, chunk_size=1)
        torch.testing.assert_close(whole, chunked)
        expected = torch.autograd.grad(whole, features)[0]
        actual = torch.autograd.grad(chunked, features)[0]
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(float(actual.abs().sum()), 0.)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(report['embedding_dimensions'], 8192)

    def test_same_event_crop_views_remain_positive_partners(self):
        # Two views per class can originate from one event/interval per class.
        # Standard SupCon must retain all four anchors and one positive each.
        features = torch.zeros(4, 2, 3, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([1, 1, 0, 0])
        loss, report = supervised_contrastive_loss(features, labels)
        self.assertEqual(report['valid_anchors_by_class'], {'0': 2, '1': 2})
        self.assertEqual(report['skipped_anchors'], 0)
        # Three candidates per anchor: its other view and the opposite-class views.
        torch.testing.assert_close(loss, torch.tensor(3., dtype=torch.double).log(), atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, features)[0]).all())

    def test_missing_partners_and_silence(self):
        features = torch.zeros(3, 2, 3, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([1, 0, 0])
        loss, report = supervised_contrastive_loss(features, labels)
        self.assertEqual(report['valid_anchors_by_class'], {'0': 2, '1': 0})
        self.assertEqual(report['skipped_anchors'], 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, features)[0]).all())
        zero, report = supervised_contrastive_loss(features[:2], torch.tensor([1, 0]))
        self.assertEqual(report['valid_anchors'], 0)
        self.assertEqual(float(zero.detach()), 0)
        torch.testing.assert_close(torch.autograd.grad(zero, features)[0], torch.zeros_like(features))

    def test_class_separation_and_permutation_invariance(self):
        labels = torch.tensor([1, 1, 0, 0])
        good = torch.tensor([[2., 0], [2, 0], [0, 2], [0, 2]])
        bad = good[[0, 2, 1, 3]]
        low, _ = supervised_contrastive_loss(good, labels)
        high, _ = supervised_contrastive_loss(bad, labels)
        self.assertLess(float(low), float(high))
        order = torch.tensor([3, 0, 2, 1])
        permuted, _ = supervised_contrastive_loss(good[order], labels[order])
        torch.testing.assert_close(low, permuted)

    def test_meta_coefficient_gradient_and_checkpoint_replay(self):
        torch.manual_seed(5)
        model = nn.Module()
        model.frontend = nn.Linear(2, 3).double()
        model.cnn = nn.Sequential(AxisBatchNorm(3, axis=1), nn.Tanh(), nn.Dropout(), nn.Linear(3, 2)).double()
        state = copy.deepcopy(model.state_dict())
        support = (torch.randn(6, 2, dtype=torch.double), torch.tensor([0, 0, 1, 1, 0, 1]))
        query = (torch.randn(4, 2, dtype=torch.double), torch.tensor([1, 0, 1, 0]))
        config = Config(inner_epochs=2, batch_size=2, inner_lr=.02, classifier_dropout=.1,
                        inner_checkpoint_steps=0, support_loss_weight=.4, label_smoothing=0)

        def evaluate(coefficient, checkpoint_steps):
            torch.manual_seed(8)
            loss, report = meta_objective(model, support, query,
                replace(config, contrastive_loss_weight=coefficient, inner_checkpoint_steps=checkpoint_steps),
                return_diagnostics=True)
            rng = torch.get_rng_state().clone()
            grad = torch.autograd.grad(loss, tuple(model.frontend.parameters()))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            return loss.detach(), torch.cat([value.flatten() for value in grad]), report

        baseline, base_gradient, off = evaluate(0, 0)
        raw, _ = supervised_contrastive_loss(model.frontend(torch.cat((support[0], query[0]))),
            torch.cat((support[1], query[1])))
        direct = torch.cat([value.flatten() for value in torch.autograd.grad(raw, tuple(model.frontend.parameters()))])
        for coefficient in (.1, .3):
            full, gradient, report = evaluate(coefficient, 0)
            replay, replay_gradient, _ = evaluate(coefficient, 2)
            torch.testing.assert_close(full, baseline + coefficient * raw.detach(), atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(gradient, base_gradient + coefficient * direct, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(full, replay, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(gradient, replay_gradient, atol=1e-10, rtol=1e-10)
            self.assertAlmostEqual(report['outer_objective']['weighted_contrastive_loss'],
                                   coefficient * float(raw.detach()), places=12)
        self.assertFalse(off['outer_objective']['contrastive']['enabled'])
        for name, value in state.items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.cnn.parameters()))

    def test_coefficient_validation_and_roundtrip(self):
        for value in (0, .1, 2.):
            config = replace(Config(), contrastive_loss_weight=value).validate()
            self.assertEqual(Config.from_dict(config.to_dict()), config)
        for value in (-1, float('nan'), float('inf'), True, '0.1', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'contrastive_loss_weight'):
                replace(Config(), contrastive_loss_weight=value).validate()


if __name__ == '__main__':
    unittest.main()
