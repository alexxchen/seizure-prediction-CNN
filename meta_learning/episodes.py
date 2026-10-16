"""Fixed per-mouse tasks: training support and validation query; test is inference only."""

from dataclasses import dataclass

import numpy as np

from source_data import DAY_SPLITS


@dataclass(frozen=True)
class Episode:
    mouse: str
    support_indices: np.ndarray
    query_indices: np.ndarray
    support_days: tuple
    query_days: tuple

    def describe(self):
        return dict(mouse=self.mouse, support_split="train", query_split="validation",
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


def episode_batch(support_split, query_split, episode, config, rng, device):
    support = balanced_indices(support_split.labels, episode.support_indices, config.support_per_class, rng)
    query = balanced_indices(query_split.labels, episode.query_indices, config.query_per_class, rng)
    # Identity scaling preserves absolute amplitude before the neural frontend.
    scale = support_split.scale(support)
    return support_split.batch(support, scale, device), query_split.batch(query, scale, device)
