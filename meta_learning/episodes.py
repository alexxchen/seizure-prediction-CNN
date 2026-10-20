"""Fixed per-mouse tasks: training support and outer-training query; test is inference only."""

from dataclasses import dataclass

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
    """Support crops apply identically to meta inner fits and per-mouse fits (never log-STFT)."""
    return (config.augment_random_crop and config.random_crop_scope in {"both", "support"}
            and config.algorithm != "log_stft")


def crops_query(config):
    return (config.augment_random_crop and config.random_crop_scope in {"both", "query"}
            and config.algorithm != "log_stft")


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
    return split.batch(indices, split.scale(indices), device)


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


def episode_batch(support_split, query_split, episode, config, rng, device, *, return_event_weights=False):
    support = support_indices(support_split, episode.support_indices, config)
    support = balanced_indices(support_split.labels, support, config.support_per_class, rng)
    crop_query = crops_query(config)
    query = (event_crop_indices(query_split, episode.query_indices, config.query_crops)
             if crop_query else episode.query_indices)
    query = balanced_indices(query_split.labels, query, config.query_per_class, rng)
    # Identity scaling preserves absolute amplitude before the neural frontend.
    scale = support_split.scale(support)
    # Crop positions come from the episode RNG, so checkpoints resume them exactly and a
    # fixed inner_shuffle_seed also fixes them.
    # Cropped sides contribute K examples per event and class before optional sample caps.
    support_batch = support_inputs(support_split, support, config, rng, device)
    query_batch = (query_split.random_crop_batch(query, rng, device) if crop_query else
                   query_split.batch(query, scale, device))
    if return_event_weights:
        return support_batch, query_batch, (event_sample_weights(support_split, support, device),
                                           event_sample_weights(query_split, query, device))
    return support_batch, query_batch
