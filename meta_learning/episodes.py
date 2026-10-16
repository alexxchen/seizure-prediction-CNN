"""Rolling chronological tasks within a mouse; official validation days are excluded."""

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
        return dict(mouse=self.mouse, support_days=list(self.support_days), query_days=list(self.query_days),
                    support_samples=len(self.support_indices), query_samples=len(self.query_indices))


def chronological_episodes(split, min_support_days=2, query_days=1):
    if split.split != "train" or not split.table.day.isin(DAY_SPLITS["train"]).all():
        raise ValueError("Meta-training tasks must come exclusively from training days")
    if not split.table.mouse.eq(split.mouse).all():
        raise ValueError("Support and query must belong to the same mouse")
    days = sorted(int(day) for day in split.table.day.unique())
    tasks = []
    for position in range(min_support_days, len(days) - query_days + 1):
        support_days, later_days = tuple(days[:position]), tuple(days[position:position + query_days])
        support = np.flatnonzero(split.table.day.isin(support_days).to_numpy())
        query = np.flatnonzero(split.table.day.isin(later_days).to_numpy())
        if set(split.labels[support]) != {0, 1} or set(split.labels[query]) != {0, 1}:
            continue
        # Entire recording days are disjoint, including all overlapping source windows.
        if max(support_days) >= min(later_days) or np.intersect1d(support, query).size:
            raise ValueError("Chronological support/query separation failed")
        tasks.append(Episode(split.mouse, support, query, support_days, later_days))
    if not tasks:
        raise ValueError(f"{split.mouse}: insufficient labeled training days for chronological episodes")
    return tasks


def balanced_indices(labels, available, per_class, rng):
    chosen = []
    for label in (0, 1):
        candidates = available[labels[available] == label]
        if not len(candidates):
            raise ValueError("An episode needs both classes")
        chosen.extend(rng.choice(candidates, size=min(per_class, len(candidates)), replace=False).tolist())
    return rng.permutation(np.asarray(chosen, dtype=np.int64))


def episode_batch(split, episode, config, rng, device):
    support = balanced_indices(split.labels, episode.support_indices, config.support_per_class, rng)
    query = balanced_indices(split.labels, episode.query_indices, config.query_per_class, rng)
    # Query values never participate in estimating normalization.
    scale = split.scale(support)
    return split.batch(support, scale, device), split.batch(query, scale, device)
