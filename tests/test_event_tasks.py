"""Event isolation, lazy source reuse, full-gradient training, resume, and inference replay."""

import contextlib
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from meta_learning import cli, engine
from meta_learning.algorithms import meta_objective
from meta_learning.config import Config, TASK_DAYS
from meta_learning.data import FeatureSplit
from meta_learning.episodes import episode_batch, episode_from_events, event_sample_weights, split_episode
from models.normalization import AxisBatchNorm


class CropSplit(FeatureSplit):
    def random_crop_batch(self, indices, rng, device):
        values, labels = self.batch(indices, device)
        return values + torch.as_tensor(rng.normal(0, 0.01, values.shape), dtype=values.dtype, device=device), labels


def source_part(mouse, part, days, events=2):
    rows, values = [], []
    for day in days:
        for event in range(events):
            for label in (0, 1):
                for window in range(2):
                    rows.append(dict(mouse=mouse, day=day, raw_file=f'{mouse}-{day}.parquet',
                        start_row=event * 100 + label * 10 + window, stop_row=event * 100 + label * 10 + window + 1,
                        label=label, event_index=day * 10 + event, window=window))
                    values.append([2 * label - 1 + day / 100, event / 10 + window / 100])
    table = pd.DataFrame(rows)
    return CropSplit(mouse, part, np.asarray(values, dtype=np.float32), table.label.to_numpy(), table)


def development_pool(mouse='X1'):
    return FeatureSplit.join_development([source_part(mouse, 'train', range(8)),
                                          source_part(mouse, 'validation', [8, 11])])


class TinyFrontend(nn.Linear):
    projection_dim = 2

    def __init__(self, batch_size):
        super().__init__(2, 3)
        self.batch_size = batch_size
        self.projection_head = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, self.projection_dim))

    def project(self, features):
        return self.projection_head(features)

    def decay_parameters(self):
        return (self.weight,)

    def describe(self):
        return dict(state=dict(weight_norm=float(self.weight.detach().norm())))


class TinyModel(nn.Module):
    def __init__(self, cnn_seed=0, frontend_batch_size=5):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cnn_seed)
            self.frontend = TinyFrontend(frontend_batch_size)
        self.reset_classifier(cnn_seed)

    def reset_classifier(self, seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.cnn = nn.Sequential(AxisBatchNorm(3, axis=1), nn.Tanh(), nn.Dropout(), nn.Linear(3, 2))


class TinyStore:
    def __init__(self, invert_test=False):
        self.pool = development_pool()
        self.test = source_part('X1', 'test', [12, 13])
        if invert_test:
            self.test.labels = 1 - self.test.labels
            self.test.table['label'] = self.test.labels
        self.dataset = type('Dataset', (), dict(signature={'synthetic': True}))()

    def load_task_pool(self, mouse, *, crops=False):
        assert mouse == 'X1'
        return self.pool

    def load(self, mouse, part):
        assert mouse == 'X1' and part == 'test'
        return self.test


class MultiMouseStore:
    def __init__(self):
        self.pools = {mouse: FeatureSplit.join_development([
            source_part(mouse, 'train', range(8), events=index + 1),
            source_part(mouse, 'validation', [8, 11], events=index + 1)])
            for index, mouse in enumerate(('X1', 'X2'))}
        self.tests = {mouse: source_part(mouse, 'test', [12, 13]) for mouse in self.pools}
        self.dataset = type('Dataset', (), dict(signature={'synthetic': 'multiple_mice'}))()

    def load_task_pool(self, mouse, *, crops=False):
        return self.pools[mouse]

    def load(self, mouse, part):
        assert part == 'test'
        return self.tests[mouse]


class EventTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complete_event_groups_and_constant_budgets_across_redraws(self):
        pool, config, rng = development_pool(), Config(), np.random.default_rng(31)
        memberships = set()
        for _ in range(30):
            episode = split_episode(pool, config, rng)
            self.assertEqual(len(episode.query_events), 4)
            self.assertEqual(len(episode.support_events), 16)
            self.assertEqual(len(episode.query_indices), 16)
            self.assertFalse(set(episode.query_events) & set(episode.support_events))
            for key, indices in pool.table.groupby(['day', 'event_index']).indices.items():
                self.assertTrue(set(indices) <= set(episode.query_indices) or
                                set(indices) <= set(episode.support_indices))
            memberships.add(episode.query_events)
            weights = event_sample_weights(pool, episode.query_indices, torch.device('cpu'))
            for key in episode.query_events:
                rows = pool.table.iloc[episode.query_indices]
                self.assertEqual(float(weights[(rows.day.eq(key[0]) & rows.event_index.eq(key[1])).to_numpy()].sum()), 1)
        self.assertGreater(len(memberships), 20)

    def test_partition_rng_restore_and_event_reconstruction(self):
        pool, config, rng = development_pool(), Config(), np.random.default_rng(9)
        state = copy.deepcopy(rng.bit_generator.state)
        first = split_episode(pool, config, rng)
        rng.bit_generator.state = state
        self.assertEqual(first.describe(), split_episode(pool, config, rng).describe())
        restored = episode_from_events(pool, first.support_events, first.query_events)
        np.testing.assert_array_equal(restored.query_indices, first.query_indices)
        with self.assertRaises(ValueError):
            episode_from_events(pool, first.support_events, first.support_events)
        with self.assertRaises(ValueError):
            FeatureSplit.join_development([source_part('X1', 'test', [12, 13])])
        with self.assertRaises(ValueError):
            replace(config, task_days=(0, 12)).validate()
        with self.assertRaises(ValueError):
            Config.from_dict({'query_validation_days': [8, 11]})

    def test_lazy_join_and_subset_preserve_source_mapping_and_crop_rng(self):
        pool = development_pool()
        self.assertIsNone(pool.inputs)
        indices = np.array([78, 1, 66, 33, 0])
        subset = pool.subset_indices(indices)
        expected = np.concatenate([part.inputs for part in pool.parts])[indices]
        values, labels = subset.batch(np.arange(len(indices)), torch.device('cpu'))
        np.testing.assert_array_equal(values.numpy(), expected)
        np.testing.assert_array_equal(labels.numpy(), pool.labels[indices])
        rng_a, rng_b = np.random.default_rng(8), np.random.default_rng(8)
        a, _ = pool.random_crop_batch(indices, rng_a, torch.device('cpu'))
        b, _ = subset.random_crop_batch(np.arange(len(indices)), rng_b, torch.device('cpu'))
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertEqual(rng_a.bit_generator.state, rng_b.bit_generator.state)

    def config(self):
        return Config(targets=('X1',), outer_steps=2, inner_epochs=2, inner_checkpoint_steps=2,
                      batch_size=8, frontend_batch_size=5, support_crops=1, support_pool_crops=2,
                      query_crops=2, eval_every=1, checkpoint_every=1, classifier_dropout=0.1,
                      support_loss_weight=1, contrastive_loss_weight=0.1, label_smoothing=0)

    def run_training(self, directory, store, *, resume=False, interrupt=False, config=None):
        original = engine.progress_event
        def progress(stage, row, **kwargs):
            original(stage, row, **kwargs)
            if interrupt and stage == 'outer_step' and row['step'] == 1:
                raise RuntimeError('Synthetic interrupt after committed step')
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(engine, 'MetaModel', TinyModel), \
                patch.object(engine, 'source_hashes', return_value={}), \
                patch.object(engine, 'verify_prepared', return_value={'signature': store.dataset.signature}), \
                patch.object(engine, 'progress_event', side_effect=progress):
            return engine.train_run(config or self.config(), store, directory, '.', torch.device('cpu'), resume)

    def test_uniform_mouse_sampling_with_replacement_and_rng_restore(self):
        mice = ('X1', 'X2', 'X9', 'X14')
        rng = np.random.default_rng(27)
        state = copy.deepcopy(rng.bit_generator.state)
        draws = [engine.sample_mouse(mice, rng) for _ in range(10000)]
        rng.bit_generator.state = state
        self.assertEqual(draws, [engine.sample_mouse(mice, rng) for _ in range(10000)])
        frequencies = {mouse: sum(m == mouse for m, _ in draws) / len(draws) for mouse in mice}
        for frequency in frequencies.values():
            self.assertLess(abs(frequency - 0.25), 0.015)
        self.assertTrue(any(draws[i][0] == draws[i + 1][0] for i in range(len(draws) - 1)))
        self.assertTrue(all(info == {'probability': 0.25} for _, info in draws))

    def test_command_line_plan_has_single_mouse_policy_without_event_weights(self):
        store = MultiMouseStore()
        tables = {mouse: {part: source_part(mouse, part, days) for part, days in
            [('train', range(8)), ('validation', [8, 11]), ('test', [12, 13])]}
            for mouse in store.pools}
        config = replace(self.config(), targets=tuple(store.pools))
        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()
            with patch('sys.argv', ['train.py', '--run-dir', temporary, '--plan-only']), \
                    patch.object(cli.Config, 'load', return_value=config), \
                    patch.object(cli, 'DataStore', return_value=store), \
                    patch.object(cli.sample_pipeline, 'build_tables',
                        side_effect=lambda dataset, mouse: ({part: split.table for part, split in tables[mouse].items()}, None)), \
                    patch.object(cli, 'train_run', side_effect=AssertionError('Planning must not train')), \
                    contextlib.redirect_stdout(output):
                cli.train_main()
            preview = json.loads(output.getvalue())
            self.assertEqual(preview['tasks_per_step'], 1)
            self.assertFalse(preview['training'])
            self.assertNotIn('outer_task_weights', preview)
            tasks = json.loads((Path(temporary) / 'episodes.json').read_text())
            self.assertEqual(set(tasks), set(config.targets))
            for mouse in config.targets:
                self.assertNotIn('outer_weight', tasks[mouse][0])
                self.assertEqual(len(tasks[mouse][0]['query_event_ids']), 4)
                self.assertEqual(len(tasks[mouse][0]['support_event_ids']), 16)

    def test_single_mouse_updates_all_mouse_evaluation_and_exact_resume(self):
        config = replace(self.config(), targets=('X1', 'X2'), outer_steps=6, inner_epochs=1,
                         batch_size=16, inner_checkpoint_steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_training(root / 'complete', MultiMouseStore(), config=config)
            with self.assertRaisesRegex(RuntimeError, 'Synthetic interrupt'):
                self.run_training(root / 'resumed', MultiMouseStore(), config=config, interrupt=True)
            self.run_training(root / 'resumed', MultiMouseStore(), config=config, resume=True)
            a = torch.load(root / 'complete/checkpoints/last.pt', weights_only=True)
            b = torch.load(root / 'resumed/checkpoints/last.pt', weights_only=True)
            self.assertEqual(a['history'], b['history'])
            self.assertEqual(a['episode_rng'], b['episode_rng'])
            self.assertEqual(a['mouse_rng'], b['mouse_rng'])
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
            for key, state in a['optimizer']['state'].items():
                self.assertEqual(int(state['step']), config.outer_steps)
                for name, value in state.items():
                    torch.testing.assert_close(value, b['optimizer']['state'][key][name], atol=0, rtol=0)
            self.assertEqual(set(a['selection']['subjects']), set(config.targets))
            self.assertEqual(set(a['selection']['test']['subjects']), set(config.targets))
            expected_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 19001]))
            chosen = []
            for row in a['history']:
                self.assertEqual(row['tasks'], 1)
                self.assertEqual(row['inner']['fixed_epoch_tasks'], 1)
                self.assertEqual(len(row['mice']), 1)
                mouse = row['mice'][0]
                chosen.append(mouse)
                expected_mouse, info = engine.sample_mouse(config.targets, expected_rng)
                self.assertEqual(mouse, expected_mouse)
                self.assertEqual(info, row['mouse_sampling'])
                self.assertEqual(set(row['query']['subjects']), {mouse})
                self.assertLessEqual(row['query']['aggregate']['accuracy'], 1)
                self.assertEqual(set(row['episodes']), set(config.targets))
                inner = list((root / 'complete/inner' / f"step-{row['step']:06d}").glob('*.json'))
                self.assertEqual(len(inner), 1)
                fit = json.loads(inner[0].read_text())
                self.assertEqual(fit['mouse'], mouse)
                objective = fit['outer_objective']['total_loss']
                self.assertAlmostEqual(row['outer_loss'], objective, places=6)
                self.assertNotIn('outer_weight', fit)
                self.assertNotIn('outer_task_weights', row)
                for other in config.targets:
                    saved = row['episodes'][other]
                    pool = MultiMouseStore().pools[other]
                    restored = episode_from_events(pool, saved['support_event_ids'], saved['query_event_ids'])
                    self.assertEqual(len(restored.query_events), (config.targets.index(other) + 1) * 2)
                    self.assertFalse(set(restored.query_events) & set(restored.support_events))
            self.assertEqual(set(chosen), set(config.targets))
            for mouse in config.targets:
                partitions = [row['episodes'][mouse]['query_event_ids'] for row in a['history']]
                self.assertTrue(any(partitions[0] != value for value in partitions[1:]))

    def test_checkpointed_pool_training_resume_and_selected_prediction_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete, resumed = root / 'complete', root / 'resumed'
            selection = self.run_training(complete, TinyStore())
            with self.assertRaisesRegex(RuntimeError, 'Synthetic interrupt'):
                self.run_training(resumed, TinyStore(), interrupt=True)
            resumed_selection = self.run_training(resumed, TinyStore(), resume=True)
            a = torch.load(complete / 'checkpoints/last.pt', weights_only=True)
            b = torch.load(resumed / 'checkpoints/last.pt', weights_only=True)
            self.assertEqual(a['history'], b['history'])
            self.assertEqual(a['episode_rng'], b['episode_rng'])
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
            self.assertEqual(selection['aggregate'], resumed_selection['aggregate'])
            self.assertNotEqual(a['history'][0]['episodes'], a['history'][1]['episodes'])
            self.assertGreater(a['history'][0]['gradient_norm'], 0)
            for split in ('query', 'test'):
                with patch.object(engine, 'MetaModel', TinyModel), \
                        patch.object(engine, 'source_hashes', return_value={}), \
                        patch.object(engine, 'adapt_model', side_effect=AssertionError('Replay must not fit')):
                    replay = engine.evaluate_run(TinyStore(), complete, torch.device('cpu'), split)
                expected = selection if split == 'query' else selection['test']
                self.assertEqual(replay['aggregate'], expected['aggregate'])
                for mouse in ('X1',):
                    saved = complete / selection['adapted_models'][mouse]['path']
                    before = pd.read_csv(saved.parent / f'predictions_{split}.csv')
                    after = pd.read_csv(complete / 'evaluation' / split / mouse / 'predictions.csv')
                    pd.testing.assert_frame_equal(before, after)

    def test_test_labels_never_change_frontend_cnn_bn_or_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self.run_training(root / 'baseline', TinyStore())
            inverted = self.run_training(root / 'inverted', TinyStore(invert_test=True))
            self.assertEqual(baseline['step'], inverted['step'])
            self.assertEqual(baseline['aggregate'], inverted['aggregate'])
            self.assertNotEqual(baseline['test']['aggregate']['loss'], inverted['test']['aggregate']['loss'])
            for name, directory in [('baseline', baseline), ('inverted', inverted)]:
                path = root / name / directory['adapted_models']['X1']['path']
                state = torch.load(path, weights_only=True)['model']
                if name == 'baseline':
                    expected = state
                else:
                    for key in expected:
                        torch.testing.assert_close(state[key], expected[key], atol=0, rtol=0)

    def test_contrastive_pool_full_hypergradient_and_sampling_parity(self):
        config = replace(self.config(), classifier_dropout=0, inner_checkpoint_steps=0)
        split, model = development_pool(), TinyModel().double()
        episode = split_episode(split, config, np.random.default_rng(3))

        def evaluate(checkpoint_steps, coefficient=.1):
            rng = np.random.default_rng(42)
            support, query, weights = episode_batch(split, split, episode,
                replace(config, contrastive_loss_weight=coefficient), rng, torch.device('cpu'))
            query = (query[0].double(), query[1])
            # Pool waveforms are float32 by contract; match the toy frontend's precision.
            waveforms = support.waveforms
            support.waveforms = lambda **kwargs: (batch.double() for batch in waveforms(**kwargs))
            torch.manual_seed(6)
            loss = meta_objective(model, support, query,
                replace(config, inner_checkpoint_steps=checkpoint_steps, contrastive_loss_weight=coefficient),
                query_weights=weights.double())
            torch_rng = torch.get_rng_state().clone()
            gradient = torch.autograd.grad(loss, tuple(model.frontend.parameters()))
            self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
            return loss.detach(), gradient, copy.deepcopy(rng.bit_generator.state), support

        full, gradient, state, pool = evaluate(0)
        replay, replay_gradient, replay_state, _ = evaluate(2)
        torch.testing.assert_close(full, replay, atol=1e-12, rtol=1e-12)
        for expected, actual in zip(gradient, replay_gradient):
            torch.testing.assert_close(expected, actual, atol=1e-10, rtol=1e-10)
        self.assertEqual(state, replay_state)
        _, _, disabled_state, disabled_pool = evaluate(2, 0)
        self.assertEqual(state, disabled_state)
        np.testing.assert_array_equal(pool.indices, disabled_pool.indices)
        np.testing.assert_array_equal(pool.selections, disabled_pool.selections)
        origin, epsilon = model.frontend.weight[0, 0].item(), 1e-5
        values = []
        for delta in (epsilon, -epsilon):
            with torch.no_grad():
                model.frontend.weight[0, 0] = origin + delta
            values.append(evaluate(2)[0])
        with torch.no_grad():
            model.frontend.weight[0, 0] = origin
        torch.testing.assert_close(gradient[0][0, 0], (values[0] - values[1]) / (2 * epsilon), atol=1e-7, rtol=1e-5)
        projection = model.frontend.projection_head[0].weight
        origin = projection[0, 0].item()
        values = []
        for delta in (epsilon, -epsilon):
            with torch.no_grad():
                projection[0, 0] = origin + delta
            values.append(evaluate(2)[0])
        with torch.no_grad():
            projection[0, 0] = origin
        torch.testing.assert_close(gradient[2][0, 0], (values[0] - values[1]) / (2 * epsilon), atol=1e-7, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
