"""Frozen window geometry copied exactly from the existing MouseLFP trainer."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from utils.mouse_lfp import (SOURCE_HZ, WINDOW_ROWS, negative_windows,
                             positive_exclusion_overlaps)

DEFAULT_BANDS = [(5, 1)]
WINDOWS = 10
DAY_SPLITS = {"train": list(range(8)), "validation": [8, 11], "test": [12, 13]}


def windows_for_bands(bands, windows=None):
    counts = {(earlier - later) * 60 * SOURCE_HZ // WINDOW_ROWS for earlier, later in bands}
    if len(counts) != 1 or min(counts) < 1:
        raise ValueError("Run intervals with different window counts separately")
    available = counts.pop()
    if windows is not None:
        if windows < 1 or windows > available:
            raise ValueError(f"Select between 1 and {available} windows per event")
        return windows
    return available


def positive_window_offsets(earlier, later, windows=None):
    count = windows_for_bands([(earlier, later)], windows)
    if windows is None:
        return np.arange(count, dtype=np.int64) * WINDOW_ROWS
    return np.linspace(0, (earlier-later)*60*SOURCE_HZ-WINDOW_ROWS,
                       count, dtype=np.int64)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def same_day_negative_indices(events, negatives, windows=WINDOWS):
    """Assign unique negatives from each event's actual recording day."""
    result = np.empty((len(events), windows), dtype=np.int64)
    for day, group in events.groupby("day", sort=True):
        candidates = negatives[negatives.day == day].sort_values(["raw_file", "start_row"])
        count = len(group) * windows
        if len(candidates) < count:
            raise ValueError(f"Day {day}: need {count} cached negatives, have {len(candidates)}")
        indices = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
        result[group.index.to_numpy()] = candidates.index.to_numpy()[indices].reshape(-1, windows)
    if len(np.unique(result)) != result.size:
        raise ValueError("Repeated negative window in day matching")
    return result


def day_groups(events):
    groups = {name: np.flatnonzero(events.day.isin(days).to_numpy())
              for name, days in DAY_SPLITS.items()}
    if any(len(indices) == 0 for indices in groups.values()):
        raise ValueError("Each date split must contain at least one selected event")
    if sorted(np.concatenate(list(groups.values())).tolist()) != list(range(len(events))):
        raise ValueError("Date splits must cover each event exactly once")
    return groups


def matched_negative_indices(events, negatives, windows):
    """Use the first eight entries of the existing ten-window daily grid."""
    return same_day_negative_indices(events, negatives, WINDOWS)[:, :windows]


def negative_options(data):
    return dict(preictal_minutes=10, postictal_minutes=10)


def validate_positive_windows(data, rows, output, name):
    policy = getattr(data, 'signature', {}).get('positive_exclusion')
    if policy is None:
        return
    overlaps = positive_exclusion_overlaps(rows, data.events, data.sources,
                                          preictal_minutes=policy['preictal_minutes'],
                                          postictal_minutes=policy['postictal_minutes'])
    if overlaps.any():
        raise ValueError('Positive windows overlap another annotated seizure exclusion zone')
    write_json(output/f'{name}_exclusion_audit.json', dict(
        windows=len(rows), overlapping_windows=int(overlaps.sum()), policy=policy))


def positive_window_rows(events, bands=DEFAULT_BANDS, windows=None):
    records = []
    for event_index, event in enumerate(events.itertuples()):
        for band_index, (earlier, later) in enumerate(bands):
            first = int(event.start_row) - earlier * 60 * SOURCE_HZ
            for window, offset in enumerate(positive_window_offsets(earlier, later, windows)):
                start = first + int(offset)
                stop = start + WINDOW_ROWS
                if start < 0 or stop > int(event.start_row) - later * 60 * SOURCE_HZ:
                    raise ValueError("Positive window crosses its scan interval")
                records.append(dict(event_index=event_index, band_index=band_index,
                                    earlier_minutes=earlier, later_minutes=later, window=window,
                                    mouse=event.mouse, day=int(event.day), raw_file=event.raw_file,
                                    epoch_id=int(event.epoch_id), start_row=start, stop_row=stop))
    return pd.DataFrame(records)


def training_window_offsets(earlier, later, windows_per_event, stride_seconds):
    """Append a dense training grid while retaining every reference window."""
    if not 1 <= stride_seconds < WINDOW_ROWS // SOURCE_HZ:
        raise ValueError("Training stride must be between 1 and 27 seconds")
    reference = positive_window_offsets(earlier, later, windows_per_event)
    grid = np.arange(0, (earlier-later)*60*SOURCE_HZ-WINDOW_ROWS+1,
                     stride_seconds*SOURCE_HZ, dtype=np.int64)
    return np.r_[reference, grid[~np.isin(grid, reference)]]


def training_overlap_rows(events, groups, bands, windows_per_event, stride_seconds):
    records = []
    for event_index in groups['train']:
        event = events.loc[event_index]
        for band_index, (earlier, later) in enumerate(bands):
            first = int(event.start_row)-earlier*60*SOURCE_HZ
            offsets = training_window_offsets(earlier, later, windows_per_event, stride_seconds)
            reference_count = windows_for_bands([(earlier, later)], windows_per_event)
            for window, offset in enumerate(offsets):
                start = first+int(offset)
                stop = start+WINDOW_ROWS
                if start < 0 or stop > int(event.start_row)-later*60*SOURCE_HZ:
                    raise ValueError("Training window crosses its positive interval")
                records.append(dict(event_index=int(event_index), band_index=band_index,
                                    earlier_minutes=earlier, later_minutes=later, window=window,
                                    mouse=event.mouse, day=int(event.day), raw_file=event.raw_file,
                                    epoch_id=int(event.epoch_id), start_row=start, stop_row=stop,
                                    split='train', reference_window=window < reference_count))
    return pd.DataFrame(records)


def training_overlap_negatives(data, target, events, groups, output, train_windows, base_windows,
                               stride_seconds):
    train_events = events.loc[groups['train']].reset_index(drop=True)
    reference = pd.read_csv(output/'matched_negative_windows.csv')
    reference = reference[reference.event_index.isin(groups['train'])].copy()
    pool = negative_windows(data.events, data.sources, target, None,
                            stride_rows=stride_seconds*SOURCE_HZ, **negative_options(data))
    pool = pool[pool.day.isin(DAY_SPLITS['train'])].copy()
    used = set(zip(reference.raw_file, reference.start_row))
    remaining = pool[[key not in used for key in zip(pool.raw_file, pool.start_row)]]
    extra_count = train_windows-base_windows
    indices = same_day_negative_indices(train_events, remaining, extra_count)
    extra = pool.loc[indices.ravel()].copy().reset_index(drop=True)
    extra['event_index'] = np.repeat(groups['train'], extra_count)
    extra['window'] = np.tile(np.arange(base_windows, train_windows), len(train_events))
    extra['event_day'] = train_events.day.to_numpy()[np.repeat(np.arange(len(train_events)), extra_count)]
    extra['split'] = 'train'
    extra['same_day'] = extra.day == extra.event_day
    matched = pd.concat([reference, extra], ignore_index=True).sort_values(
        ['event_index', 'window']).reset_index(drop=True)
    matched['reference_window'] = matched.window < base_windows
    if matched.duplicated(['raw_file', 'start_row']).any():
        raise ValueError("Repeated training negative window")
    if not matched.day.isin(DAY_SPLITS['train']).all():
        raise ValueError("Training negative crossed the date split")
    if not matched.same_day.all():
        raise ValueError("Training negative crossed the event's recording day")
    return matched
