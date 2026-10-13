"""Train seven mouse LFP models with the fixed 5-1-minute protocol."""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, log_loss, roc_auc_score)

from models.cnn import ConvNN, resolve_device
from utils.mouse_lfp import (FEATURE_SHAPE, SOURCE_HZ, TARGETS, WINDOW_ROWS,
                             MouseLFPData, ParquetSignalReader,
                             negative_windows, window_features, positive_exclusion_overlaps)


# Each tuple is (earlier, later) in minutes before onset.
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


def negative_cache_suffix(data):
    return '_negpre10post10_sameday'


def positive_cache_suffix(data):
    policy = getattr(data, 'signature', {}).get('positive_exclusion')
    if policy is None:
        return ''
    earlier, later = policy['interval']
    return (f"_pospre{policy['preictal_minutes']}post{policy['postictal_minutes']}"
            f"_guard{earlier}_{later}")


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


def prepare_scan(data, target, output, bands=DEFAULT_BANDS, windows_per_event=8):
    windows = windows_for_bands(bands, windows_per_event)
    events = data.selected[data.selected.mouse == target].reset_index(drop=True)
    negatives = negative_windows(data.events, data.sources, target, None, **negative_options(data))
    events.to_csv(output / f"events_{target}.csv", index=False)
    negatives.to_csv(output / f"negative_windows_{target}.csv", index=False)
    indices = matched_negative_indices(events, negatives, windows)
    groups = day_groups(events)
    matched = negatives.iloc[indices.ravel()].copy().reset_index(drop=True)
    matched["event_index"] = np.repeat(np.arange(len(events)), windows)
    matched["window"] = np.tile(np.arange(windows), len(events))
    matched["event_day"] = events.day.to_numpy()[matched.event_index.to_numpy()]
    day_split = {day: part for part, days in DAY_SPLITS.items() for day in days}
    matched["split"] = matched.event_day.map(day_split)
    matched["same_day"] = matched.day == matched.event_day
    if not matched.same_day.all():
        raise ValueError("A negative source crossed the event's recording day")
    write_json(output / "negative_matching_summary.json", dict(
        total=len(matched), same_day=len(matched), other_day_same_split=0,
        by_split={part: dict(total=len(group), same_day=len(group), other_day_same_split=0)
                  for part, group in matched.groupby("split")}))
    matched.to_csv(output / "matched_negative_windows.csv", index=False)
    positive_rows = positive_window_rows(events, bands, windows_per_event)
    validate_positive_windows(data, positive_rows, output, 'positive')
    positive_rows.to_csv(output / "positive_windows.csv", index=False)
    write_json(output / "split.json", dict(days=DAY_SPLITS,
               event_indices={name: value.tolist() for name, value in groups.items()},
               event_counts={name: len(value) for name, value in groups.items()}))

    suffix = "_" + "_".join(f"{a}_{b}" for a, b in bands)
    if windows_per_event is not None:
        suffix += f"_w{windows}"
    suffix += negative_cache_suffix(data)
    suffix += positive_cache_suffix(data)
    scan_version = 3
    cache = Path(data.settings["cachedir"]) / f"MouseLFP_scan_v{scan_version}{suffix}" / target
    cache.mkdir(parents=True, exist_ok=True)
    signature = dict(version=scan_version, base=data.signature, bands=bands, windows=windows,
                     matched_negative_indices=indices.tolist())
    signature["negative_matching"] = "strict same mouse and day"
    signature = json.loads(json.dumps(signature))
    stamp = cache / "manifest.json"
    paths = [cache / "preictal.npy", cache / "interictal.npy"]
    shapes = [(len(events), len(bands), windows, *FEATURE_SHAPE),
              (len(events), windows, *FEATURE_SHAPE)]
    if stamp.exists() and all(path.exists() for path in paths):
        arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths]
        if (json.loads(stamp.read_text()) == signature and
                all(array.shape == shape and array.dtype == np.float32
                    for array, shape in zip(arrays, shapes))):
            print(f"{target}: exact reference cache loaded", flush=True)
            return arrays, groups
    stamp.unlink(missing_ok=True)
    arrays = [np.lib.format.open_memmap(str(path) + ".tmp", mode="w+", dtype=np.float32,
                                       shape=shape) for path, shape in zip(paths, shapes)]
    reader = ParquetSignalReader(data.root)
    for index, event in enumerate(events.itertuples()):
        for band_index in range(len(bands)):
            earlier, later = bands[band_index]
            first = int(event.start_row) - earlier * 60 * SOURCE_HZ
            offsets = positive_window_offsets(earlier, later, windows_per_event)
            signal = reader.read(event.raw_file, first, first + int(offsets[-1]) + WINDOW_ROWS)
            for window, left in enumerate(offsets):
                arrays[0][index, band_index, window] = window_features(signal[left:left + WINDOW_ROWS])
        if (index + 1) % 25 == 0 or index + 1 == len(events):
            print(f"{target}: scan features {index + 1}/{len(events)} events", flush=True)
    for index, window in enumerate(matched.itertuples()):
        signal = reader.read(window.raw_file, int(window.start_row), int(window.stop_row))
        arrays[1][index // windows, index % windows] = window_features(signal)
        if (index + 1) % 250 == 0 or index + 1 == len(matched):
            print(f"{target}: negative features {index + 1}/{len(matched)} windows", flush=True)
    for array, path in zip(arrays, paths):
        array.flush()
        Path(str(path) + ".tmp").replace(path)
    write_json(stamp, signature)
    return [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths], groups


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


def prepare_training_overlap(data, target, output, arrays, groups, bands,
                             windows_per_event, stride_seconds):
    events = data.selected[data.selected.mouse == target].reset_index(drop=True)
    positive_rows = training_overlap_rows(events, groups, bands, windows_per_event, stride_seconds)
    validate_positive_windows(data, positive_rows, output, 'train_positive')
    counts = {len(training_window_offsets(a, b, windows_per_event, stride_seconds)) for a, b in bands}
    if len(counts) != 1:
        raise ValueError("Run training intervals with different window counts separately")
    train_windows = counts.pop()
    base_windows = windows_for_bands(bands, windows_per_event)
    matched = training_overlap_negatives(data, target, events, groups, output, train_windows,
                                         base_windows, stride_seconds)
    positive_rows.to_csv(output/'train_positive_windows.csv', index=False)
    matched.to_csv(output/'train_matched_negative_windows.csv', index=False)
    write_json(output/'training_overlap_summary.json', dict(
        window_seconds=28, stride_seconds=stride_seconds,
        reference_windows_per_event=base_windows, training_windows_per_event=train_windows,
        training_events=len(groups['train']), positive_windows_per_band=len(matched),
        negative_windows=len(matched), same_day_negatives=int(matched.same_day.sum()),
        other_day_same_train_split=int((~matched.same_day).sum()),
        augmentation_split='train', evaluation_windows_unchanged=True))
    suffix = '_'.join(f'{a}_{b}' for a, b in bands)+f'_w{base_windows}_train_s{stride_seconds}'
    suffix += negative_cache_suffix(data)
    suffix += positive_cache_suffix(data)
    cache = Path(data.settings['cachedir'])/f'MouseLFP_scan_v4_{suffix}'/target
    cache.mkdir(parents=True, exist_ok=True)
    signature = json.loads(json.dumps(dict(
        version=4, base=data.signature, bands=bands, evaluation_windows=base_windows,
        stride_seconds=stride_seconds, training_event_indices=groups['train'].tolist(),
        positive_rows_sha256=hashlib.sha256(positive_rows.to_csv(index=False).encode()).hexdigest(),
        negative_rows_sha256=hashlib.sha256(matched.to_csv(index=False).encode()).hexdigest())))
    paths = [cache/'train_preictal.npy', cache/'train_interictal.npy']
    shapes = [(len(groups['train']), len(bands), train_windows, *FEATURE_SHAPE),
              (len(groups['train']), train_windows, *FEATURE_SHAPE)]
    stamp = cache/'manifest.json'
    if stamp.exists() and all(path.exists() for path in paths):
        stored = [np.load(path, mmap_mode='r', allow_pickle=False) for path in paths]
        if (json.loads(stamp.read_text()) == signature and
                all(array.shape == shape and array.dtype == np.float32
                    for array, shape in zip(stored, shapes))):
            print(f'{target}: exact training-overlap cache loaded', flush=True)
            return stored, train_windows
    stamp.unlink(missing_ok=True)
    result = [np.lib.format.open_memmap(str(path)+'.tmp', mode='w+', dtype=np.float32,
                                      shape=shape) for path, shape in zip(paths, shapes)]
    result[0][:, :, :base_windows] = arrays[0][groups['train']]
    result[1][:, :base_windows] = arrays[1][groups['train']]
    reader = ParquetSignalReader(data.root)
    for local_index, event_index in enumerate(groups['train']):
        event = events.loc[event_index]
        for band_index, (earlier, later) in enumerate(bands):
            first = int(event.start_row)-earlier*60*SOURCE_HZ
            signal = reader.read(event.raw_file, first, int(event.start_row)-later*60*SOURCE_HZ)
            offsets = training_window_offsets(earlier, later, windows_per_event, stride_seconds)
            for window in range(base_windows, train_windows):
                left = int(offsets[window])
                result[0][local_index, band_index, window] = window_features(signal[left:left+WINDOW_ROWS])
        if (local_index+1) % 25 == 0 or local_index+1 == len(groups['train']):
            print(f'{target}: overlap positive features {local_index+1}/{len(groups["train"])} events', flush=True)
    extras = matched[~matched.reference_window]
    local_indices = {int(index): i for i, index in enumerate(groups['train'])}
    for index, row in enumerate(extras.itertuples()):
        signal = reader.read(row.raw_file, int(row.start_row), int(row.stop_row))
        result[1][local_indices[int(row.event_index)], int(row.window)] = window_features(signal)
        if (index+1) % 500 == 0 or index+1 == len(extras):
            print(f'{target}: overlap negative features {index+1}/{len(extras)} windows', flush=True)
    for array, path in zip(result, paths):
        array.flush()
        Path(str(path)+'.tmp').replace(path)
    write_json(stamp, signature)
    return [np.load(path, mmap_mode='r', allow_pickle=False) for path in paths], train_windows


def assemble(arrays, band_index, indices):
    positive = arrays[0][indices, band_index].reshape(-1, *FEATURE_SHAPE)
    negative = arrays[1][indices].reshape(-1, *FEATURE_SHAPE)
    X = np.concatenate([positive, negative])
    y = np.r_[np.ones(len(positive), dtype=np.int64), np.zeros(len(negative), dtype=np.int64)]
    return X, y


def evaluate(network, X, y, path, indices, windows=WINDOWS):
    if len(y) != len(indices) * windows * 2:
        raise ValueError("Prediction provenance does not match the interval's window count")
    probabilities = network.predict_proba(X)[:, 1]
    predicted = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    event_indices = np.repeat(indices, windows)
    window_indices = np.tile(np.arange(windows), len(indices))
    pd.DataFrame(dict(event_index=np.tile(event_indices, 2), window=np.tile(window_indices, 2),
                      label=y, probability=probabilities, predicted=predicted)).to_csv(path, index=False)
    return dict(samples=len(y), positive_samples=int((y == 1).sum()),
                negative_samples=int((y == 0).sum()),
                accuracy=float(accuracy_score(y, predicted)), auc=float(roc_auc_score(y, probabilities)),
                average_precision=float(average_precision_score(y, probabilities)),
                loss=float(log_loss(y, probabilities, labels=[0, 1])),
                sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                confusion_matrix=matrix.tolist())


def summarize(output):
    rows = []
    for path in sorted(output.glob("X*/band_*/metrics.json")):
        result = json.loads(path.read_text())
        row = {key: result[key] for key in ["subject", "earlier_minutes", "later_minutes",
                                          "midpoint_minutes", "epochs_run", "best_epoch", "stop_reason"]}
        for part in ["train", "validation", "test"]:
            row.update({f"{part}_{key}": result[part][key]
                        for key in ["samples", "accuracy", "auc", "loss", "sensitivity", "specificity"]})
        rows.append(row)
    if not rows:
        return
    table = pd.DataFrame(rows).sort_values(["earlier_minutes", "subject"], ascending=[False, True])
    temporary = output / "summary.csv.tmp"
    table.to_csv(temporary, index=False)
    temporary.replace(output / "summary.csv")
    columns = [f"{part}_{metric}" for part in ["train", "validation", "test"]
               for metric in ["accuracy", "auc"]]
    aggregate = table.groupby(["earlier_minutes", "later_minutes", "midpoint_minutes"])[columns].agg(
        ["mean", "std", "count"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.reset_index().sort_values("midpoint_minutes", ascending=False).to_csv(
        output / "aggregate.csv", index=False)


def main(args):
    windows = windows_for_bands(args.bands, args.windows_per_event)
    device = resolve_device(args.device)
    settings = json.loads((Path(__file__).parent / "SETTINGS_MouseLFP.json").read_text())
    if args.data_dir is not None:
        settings["datadir"] = str(args.data_dir)
    if args.cache_dir is not None:
        settings["cachedir"] = str(args.cache_dir)
    settings.update(negative_preictal_minutes=10, negative_postictal_minutes=10,
                    negative_matching='same-day', positive_other_preictal_minutes=0,
                    positive_other_postictal_minutes=5, positive_exclusion_band=[5, 1])
    output = args.result_dir
    output.mkdir(parents=True, exist_ok=True)
    data = MouseLFPData(settings)
    source_dir = Path(__file__).parent
    configuration = dict(targets=args.targets, bands=args.bands, windows_per_event_per_band=windows,
                         day_splits=DAY_SPLITS, epochs=args.epochs, patience=args.patience,
                         batch_size=args.batch_size, seed=args.seed, monitor="val_loss",
                         early_stopping=args.early_stopping,
                         threshold=0.5, architecture="existing SeizureCNN", optimizer="existing Adam",
                         lfp_unit="unknown; preserve stored raw amplitude",
                         event_cohort=args.event_cohort, selected_event_counts=data.counts,
                         prediction_horizon_minutes=min(b for a, b in args.bands),
                         negative_reference_windows_per_event=WINDOWS,
                         negative_matching=args.negative_matching,
                         negative_preictal_minutes=args.negative_preictal_minutes,
                         negative_postictal_minutes=args.negative_postictal_minutes,
                         positive_exclusion=data.signature.get('positive_exclusion'),
                         normalization=dict(mode="none", version=1,
                                            fit_split='train', epsilon=1e-6,
                                            negative_values_retained=True),
                         dataset_signature=data.signature,
                         code_sha256={name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                                      for name in ["train_mouse_lfp.py", "models/cnn.py", "utils/mouse_lfp.py"]})
    if args.windows_per_event is not None:
        configuration['positive_window_sampling'] = "uniform across full interval; no overlap"
    if args.train_stride_seconds is not None:
        configuration['training_overlap'] = dict(
            window_seconds=28, stride_seconds=args.train_stride_seconds,
            reference_windows_per_event=windows,
            training_windows_per_event=len(training_window_offsets(
                *args.bands[0], args.windows_per_event, args.train_stride_seconds)),
            evaluation_windows_unchanged=True)
    config_path = output / "scan_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(configuration)):
        raise ValueError("Existing training configuration differs; use a new result directory")
    write_json(config_path, configuration)
    data.write_manifest(output)
    total = len(args.targets) * len(args.bands)
    write_json(output / "progress.json", dict(status="preparing", completed=0, total=total))
    for target in args.targets:
        subject_output = output / target
        subject_output.mkdir(parents=True, exist_ok=True)
        completed = len(list(output.glob("X*/band_*/metrics.json")))
        write_json(output / "progress.json", dict(status="preparing", completed=completed,
                   total=total, latest_subject=target))
        arrays, groups = prepare_scan(data, target, subject_output, args.bands, args.windows_per_event)
        train_arrays, train_windows = arrays, windows
        train_indices = groups['train']
        if args.train_stride_seconds is not None:
            train_arrays, train_windows = prepare_training_overlap(
                data, target, subject_output, arrays, groups, args.bands,
                args.windows_per_event, args.train_stride_seconds)
            train_indices = np.arange(len(groups['train']))
        for band_index, (earlier, later) in enumerate(args.bands):
            run_output = subject_output / f"band_{earlier}_{later}"
            run_output.mkdir(parents=True, exist_ok=True)
            if (run_output / "metrics.json").exists():
                print(f"{target} {earlier}-{later} min: already complete", flush=True)
                continue
            seed_everything(args.seed)
            train_X, train_y = assemble(train_arrays, band_index, train_indices)
            val_X, val_y = assemble(arrays, band_index, groups["validation"])
            print(f"\nTRAIN {target}: {earlier}-{later} minutes before onset", flush=True)
            network = ConvNN(target, device=device, epochs=args.epochs, batch_size=args.batch_size,
                             mode="scan", patience=args.patience, seed=args.seed,
                             early_stopping=args.early_stopping,
                             monitor="val_loss", track_validation_metrics=True,
                             history_path=run_output / "history.json",
                             checkpoint_dir=run_output / "checkpoints").setup(train_X.shape)
            write_json(run_output / "status.json", dict(status="training", subject=target,
                        earlier_minutes=earlier, later_minutes=later))
            write_json(output / "progress.json", dict(status="training", completed=completed,
                       total=total, latest_subject=target, latest_band=[earlier, later]))
            network.fit(train_X, train_y, val_X, val_y)
            result = dict(subject=target, earlier_minutes=earlier, later_minutes=later,
                          midpoint_minutes=(earlier + later) / 2, epochs_run=len(network.history),
                          best_epoch=network.best_epoch, best_val_loss=network.best_score,
                          stop_reason=network.stop_reason, seed=args.seed,
                          checkpoint=str(network.checkpoint_path),
                          normalization="none",
                          training_code_sha256=configuration["code_sha256"])
            for part, indices in groups.items():
                if part == 'train' and args.train_stride_seconds is not None:
                    X, y = assemble(train_arrays, band_index, train_indices)
                    result[part] = evaluate(network, X, y, run_output / 'predictions_train.csv',
                                            indices, train_windows)
                    X, y = assemble(arrays, band_index, indices)
                    result['train_reference'] = evaluate(
                        network, X, y, run_output/'predictions_train_reference.csv', indices, windows)
                    result['training_windows_per_event'] = train_windows
                    result['evaluation_windows_per_event'] = windows
                else:
                    X, y = assemble(arrays, band_index, indices)
                    result[part] = evaluate(network, X, y, run_output / f"predictions_{part}.csv", indices, windows)
            write_json(run_output / "metrics.json", result)
            write_json(run_output / "status.json", dict(status="complete"))
            summarize(output)
            completed = len(list(output.glob("X*/band_*/metrics.json")))
            write_json(output / "progress.json", dict(status="running", completed=completed,
                       total=total, latest_subject=target, latest_band=[earlier, later]))
            print(f"COMPLETE {completed}/{total}: {target} {earlier}-{later} min; "
                  f"validation accuracy={result['validation']['accuracy']:.4f}, "
                  f"test accuracy={result['test']['accuracy']:.4f}", flush=True)
    summarize(output)
    write_json(output / "progress.json", dict(status="complete", completed=total, total=total))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', default=','.join(TARGETS), help='Default: all seven mice')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--result-dir', type=Path, default=Path('docker-output-mouse-training') /
                        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') / 'MouseLFP')
    args = parser.parse_args()
    args.targets = args.targets.split(',')
    if not args.targets or len(set(args.targets)) != len(args.targets) or any(t not in TARGETS for t in args.targets):
        parser.error('Select distinct MouseLFP subject IDs')
    if min(args.epochs, args.batch_size) < 1 or args.seed < 0:
        parser.error('Epochs and batch size must be positive; seed must be nonnegative')
    args.bands = DEFAULT_BANDS
    args.windows_per_event = 8
    args.train_stride_seconds = 14
    args.event_cohort = 'fixed978'
    args.negative_matching = 'same-day'
    args.negative_preictal_minutes = args.negative_postictal_minutes = 10
    args.patience, args.early_stopping = 10, False
    return args


if __name__ == '__main__':
    main(parse_args())
