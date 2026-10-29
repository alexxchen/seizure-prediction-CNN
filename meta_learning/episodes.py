"""Fixed per-mouse tasks: training support and outer-training query; test is inference only."""

from dataclasses import dataclass
import copy

import numpy as np
import torch

from source_data import DAY_SPLITS


@dataclass(frozen=True)
class Episode:
    mouse: str
    support_indices: np.ndarray
    query_indices: np.ndarray
    support_days: tuple
    query_days: tuple

    def describe(self):
        return dict(mouse=self.mouse, support_split="support", query_split="query",
                    support_days=list(self.support_days), query_days=list(self.query_days),
                    support_samples=len(self.support_indices), query_samples=len(self.query_indices))


def split_episode(support, query):
    """Use the complete official training/validation partitions as one task."""
    if support.mouse != query.mouse:
        raise ValueError("Support and query must belong to the same mouse")
    for values, expected in [(support, "train"), (query, "validation")]:
        if (values.split != expected or values.table.empty or
                not values.table.day.isin(DAY_SPLITS[expected]).all()):
            raise ValueError(f"Expected nonempty {expected} partition with its fixed recording days")
        if not values.table.mouse.eq(values.mouse).all():
            raise ValueError("Support and query must belong to the same mouse")
        if set(values.labels) != {0, 1}:
            raise ValueError(f"{values.mouse}: {expected} partition needs both classes")
    support_days = tuple(sorted(int(day) for day in support.table.day.unique()))
    query_days = tuple(sorted(int(day) for day in query.table.day.unique()))
    if max(support_days) >= min(query_days):
        raise ValueError("Support days must precede query days")
    return Episode(support.mouse, np.arange(len(support.labels)), np.arange(len(query.labels)),
                   support_days, query_days)


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
        if split.split != "train" or not split.table.day.between(0, 7).all():
            raise ValueError("Support crop pools require training Days 0-7")
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


def query_task_weights(queries, episodes):
    """Weight fixed mouse tasks by independent eligible query events, before crops/caps."""
    counts = {}
    for mouse, episode in episodes.items():
        query = queries[mouse]
        rows = query.table.iloc[episode.query_indices]
        if (query.split != "validation" or not rows.mouse.eq(mouse).all()
                or not rows.day.isin(episode.query_days).all()):
            raise ValueError("Outer weights require the task's query partition")
        positives = rows[rows.label == 1]
        count = len(positives.drop_duplicates(["mouse", "day", "event_index"]))
        if not count:
            raise ValueError(f"{mouse}: outer weighting requires positive query events")
        counts[mouse] = count
    total = sum(counts.values())
    if not total:
        raise ValueError("Outer weighting requires at least one query task")
    return counts, {mouse: count / total for mouse, count in counts.items()}


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
