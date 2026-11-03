"""Outer LR endpoints, disabled parity, validation and exact optimizer resumption."""

import copy
from dataclasses import replace
import unittest

import torch

from meta_learning.config import Config
from meta_learning.engine import build_outer_scheduler


class OuterScheduleTests(unittest.TestCase):
    def test_configuration_validation_and_roundtrip(self):
        self.assertEqual(Config().outer_lr_scheduler, 'cosine')
        for kind in ('none', 'cosine'):
            config = Config(outer_steps=8, outer_lr_scheduler=kind).validate()
            self.assertEqual(Config.from_dict(config.to_dict()), config)
        for field, invalid in (
                ('outer_lr_scheduler', ('linear', True, None, [])),
                ('outer_lr_warmup_steps', (-1, 8, 1.5, True)),
                ('outer_lr_min_ratio', (-.1, 1.1, float('nan'), float('inf'), True))):
            for value in invalid:
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    replace(Config(outer_steps=8, outer_lr_scheduler='cosine'), **{field: value}).validate()
        with self.assertRaisesRegex(ValueError, 'requires the cosine scheduler'):
            Config(outer_lr_scheduler='none', outer_lr_warmup_steps=1).validate()

    def test_cosine_update_rates_and_post_final_minimum(self):
        config = Config(outer_steps=4, outer_lr=1, outer_lr_scheduler='cosine', outer_lr_min_ratio=.1)
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.AdamW([parameter], lr=config.outer_lr)
        scheduler = build_outer_scheduler(optimizer, config)
        rates = []
        for _ in range(4):
            rates.append(optimizer.param_groups[0]['lr'])
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
        for actual, expected in zip(rates, [1., .8681980515339464, .55, .23180194846605365]):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .1, places=12)

    def test_warmup_and_resume_during_both_schedule_phases(self):
        config = Config(outer_steps=8, outer_lr=1, outer_lr_scheduler='cosine',
                        outer_lr_warmup_steps=2, outer_lr_min_ratio=0)

        def run(interrupt_at=None):
            parameter = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.double))
            optimizer = torch.optim.AdamW([parameter], lr=config.outer_lr)
            scheduler = build_outer_scheduler(optimizer, config)
            rates = []
            for step in range(config.outer_steps):
                if step == interrupt_at:
                    saved = copy.deepcopy((parameter.detach(), optimizer.state_dict(), scheduler.state_dict()))
                    parameter = torch.nn.Parameter(saved[0].clone())
                    optimizer = torch.optim.AdamW([parameter], lr=config.outer_lr)
                    scheduler = build_outer_scheduler(optimizer, config)
                    scheduler.load_state_dict(saved[2])
                    optimizer.load_state_dict(saved[1])
                rates.append(optimizer.param_groups[0]['lr'])
                parameter.grad = torch.tensor([step + 1., 2. - step], dtype=torch.double)
                optimizer.step()
                scheduler.step()
            return parameter.detach(), rates, scheduler.state_dict(), optimizer.param_groups[0]['lr']

        expected = run()
        for step in (1, 4):
            actual = run(step)
            torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
            self.assertEqual(actual[1:], expected[1:])
        self.assertEqual(expected[1][:3], [.5, 1., 1.])
        self.assertAlmostEqual(expected[1][-1], .06698729810778065, places=12)
        self.assertEqual(expected[-1], 0)

    def test_disabled_scheduler_preserves_plain_adamw_updates_and_rng(self):
        expected = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.double))
        actual = torch.nn.Parameter(expected.detach().clone())
        reference = torch.optim.AdamW([expected], lr=.001)
        optimizer = torch.optim.AdamW([actual], lr=.001)
        rng = torch.get_rng_state().clone()
        self.assertIsNone(build_outer_scheduler(optimizer, Config(outer_lr=.001, outer_lr_scheduler='none')))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for step in range(4):
            expected.grad = torch.tensor([step + 1., 2. - step], dtype=torch.double)
            actual.grad = expected.grad.clone()
            reference.step()
            optimizer.step()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual([{k: v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups],
                         [{k: v for k, v in group.items() if k != 'params'} for group in reference.param_groups])


if __name__ == '__main__':
    unittest.main()
