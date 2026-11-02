"""Per-mouse random event tasks on non-test days; test is inference only."""

from dataclasses import dataclass
import copy
import math

import numpy as np
import torch

from meta_learning.config import TASK_DAYS


@dataclass(frozen=True)
class Episode:
    mouse: str
    support_indices: np.ndarray
    query_indices: np.ndarray
    support_days: tuple
    query_days: tuple
    support_events: tuple
    query_events: tuple

    def describe(self):
        return dict(mouse=self.mouse, support_split="support", query_split="query",
                    support_days=list(self.support_days), query_days=list(self.query_days),
                    support_samples=len(self.support_indices), query_samples=len(self.query_indices),
                    support_event_ids=[list(key) for key in self.support_events],
                    query_event_ids=[list(key) for key in self.query_events])


def event_groups(pool):
    """An event includes all positive windows and its matched negative-row budgets."""
    if (pool.split != "development" or pool.table.empty or
            not pool.table.mouse.eq(pool.mouse).all() or not pool.table.day.isin(TASK_DAYS).all()):
        raise ValueError("Event tasks require one mouse's non-test development pool")
    groups = pool.table.groupby(["day", "event_index"], sort=True).indices
    if any(set(pool.labels[indices]) != {0, 1} for indices in groups.values()):
        raise ValueError("Each event needs positive windows and matched negatives")
    return {tuple(map(int, key)): indices for key, indices in groups.items()}


def episode_from_events(pool, support_events, query_events):
    """Restore a complete, disjoint event partition without drawing random numbers."""
    groups = event_groups(pool)
    support_events = tuple(sorted(tuple(map(int, key)) for key in support_events))
    query_events = tuple(sorted(tuple(map(int, key)) for key in query_events))
    support_set, query_set = set(support_events), set(query_events)
    if (not support_set or not query_set or support_set & query_set or
            len(support_set) != len(support_events) or len(query_set) != len(query_events) or
            support_set | query_set != set(groups)):
        raise ValueError("Support/query must partition every eligible event exactly once")
    indices = lambda keys: np.sort(np.concatenate([groups[key] for key in keys]))
    return Episode(pool.mouse, indices(support_events), indices(query_events),
                   tuple(sorted({key[0] for key in support_events})),
                   tuple(sorted({key[0] for key in query_events})), support_events, query_events)


def split_episode(pool, config, rng):
    """Redraw event membership each outer step; event counts stay fixed per mouse."""
    if not pool.table.day.isin(config.task_days).all():
        raise ValueError("Task pool crossed configured task_days")
    groups = event_groups(pool)
    keys = tuple(groups)
    if len(keys) < 2:
        raise ValueError("Random event tasks require at least two eligible events")
    count = min(len(keys) - 1, max(1, math.ceil(len(keys) * config.query_event_fraction)))
    selected = set(rng.choice(len(keys), size=count, replace=False).tolist())
    return episode_from_events(pool, [key for i, key in enumerate(keys) if i not in selected],
                               [key for i, key in enumerate(keys) if i in selected])


def balanced_indices(labels, available, per_class, rng):
    chosen = []
    for label in (0, 1):
        candidates = available[labels[available] == label]
        if not len(candidates):
            raise ValueError("An episode needs both classes")
        selected = candidates if per_class is None else rng.choice(
            candidates, size=min(per_class, len(candidates)), replace=False)
        chosen.extend(selected.tolist())
    return rng.permutation(np.asarray(chosen, dtype=np.int64))


def crops_support(config):
    """Support crops apply identically to meta inner fits and per-mouse fits."""
    return config.augment_random_crop and config.random_crop_scope in {"both", "support"}


def crops_query(config):
    return config.augment_random_crop and config.random_crop_scope in {"both", "query"}


def event_crop_indices(split, indices, count):
    """Choose K crop contexts per event and class, independent of fixed-window count.

    Positive rows share their event's interval. Negative rows carry only the event's
    budget and recording day; materialization draws from that day's complete safe
    interval pool. Every event contributes K positives and K negatives.
    """
    indices = np.asarray(indices, dtype=np.int64)
    groups = split.table.iloc[indices].groupby(["mouse", "day", "event_index", "label"], sort=False)
    return np.concatenate([np.repeat(indices[positions[0]], count)
                           for positions in groups.indices.values()])


def support_indices(split, indices, config):
    """Plan the complete support before batching, so events are counted only once."""
    if crops_support(config):
        return event_crop_indices(split, indices, config.support_crops)
    return np.asarray(indices, dtype=np.int64)


def support_inputs(split, indices, config, rng, device):
    """Materialize already planned support contexts, or unchanged fixed windows."""
    if crops_support(config):
        return split.random_crop_batch(indices, rng, device)
    return split.batch(indices, device)


class SupportPool:
    """One legal crop pool per fit, with event/class-balanced epoch selections."""

    def __init__(self, split, indices, config, rng):
        event_groups(split)
        if not split.table.day.isin(config.task_days).all():
            raise ValueError("Support crop pools require non-test task days")
        self.split, self.config, self.rng = split, config, rng
        self.indices = event_crop_indices(split, indices, config.support_pool_crops)
        self.labels = split.labels[self.indices]
        self.groups = len(self.indices) // config.support_pool_crops
        self.state = None
        self.selections = []
        self.usage = np.zeros(len(self.indices), dtype=np.int64)

    def __len__(self):
        counts = np.bincount(self.labels, minlength=2) // self.config.support_pool_crops * self.config.support_crops
        return int(sum(counts if self.config.support_per_class is None else
                       np.minimum(counts, self.config.support_per_class)))

    def waveforms(self, *, replay=False):
        """Stream frontend chunks; crop RNG calls depend only on CNN batch size."""
        if replay:
            if self.state is None:
                raise ValueError("Cannot replay an unmaterialized support pool")
            rng = np.random.default_rng()
            rng.bit_generator.state = copy.deepcopy(self.state)
        else:
            if self.state is not None:
                raise ValueError("A support pool is materialized exactly once per fit")
            self.state = copy.deepcopy(self.rng.bit_generator.state)
            rng = self.rng
        pending = None
        for start in range(0, len(self.indices), self.config.batch_size):
            indices = self.indices[start:start + self.config.batch_size]
            inputs, _ = support_inputs(self.split, indices, self.config, rng, torch.device("cpu"))
            pending = inputs if pending is None else torch.cat((pending, inputs))
            while len(pending) >= self.config.frontend_batch_size:
                yield pending[:self.config.frontend_batch_size]
                pending = pending[self.config.frontend_batch_size:]
        if pending is not None and len(pending):
            yield pending

    def draw(self, epoch):
        if epoch == len(self.selections):
            if self.state is None:
                raise ValueError("Materialize the support pool before drawing epochs")
            size = self.config.support_pool_crops
            selected = np.concatenate([group * size + self.rng.choice(
                size, self.config.support_crops, replace=False) for group in range(self.groups)])
            if self.config.support_per_class is not None:
                selected = balanced_indices(self.labels, selected, self.config.support_per_class, self.rng)
            self.selections.append(selected)
            np.add.at(self.usage, selected, 1)
        return self.selections[epoch]


def event_sample_weights(split, indices, device):
    """Each represented event has unit total weight, including its matched negatives."""
    rows = split.table.iloc[indices]
    counts = rows.groupby(["mouse", "day", "event_index"], sort=False).label.transform("size")
    return torch.as_tensor(1.0 / counts.to_numpy(dtype=np.float32), device=device)


def episode_batch(support_split, query_split, episode, config, rng, device):
    support = episode.support_indices
    if not crops_support(config):
        support = balanced_indices(support_split.labels, support, config.support_per_class, rng)
    crop_query = crops_query(config)
    query = (event_crop_indices(query_split, episode.query_indices, config.query_crops)
             if crop_query else episode.query_indices)
    query = balanced_indices(query_split.labels, query, config.query_per_class, rng)
    # Query is drawn once; the support pool is streamed when fitting starts.
    # Pool positions and epoch selections share the checkpointed task RNG.
    if crops_support(config):
        support_batch = SupportPool(support_split, support, config, rng)
    else:
        support_batch = support_inputs(support_split, support, config, rng, device)
    query_batch = (query_split.random_crop_batch(query, rng, device) if crop_query else
                   query_split.batch(query, device))
    return support_batch, query_batch, event_sample_weights(query_split, query, device)
