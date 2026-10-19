"""Editable sample construction and cleaning, shared by every day split."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from source_data import DAY_SPLITS, SOURCE_HZ, file_hash

# One policy applies to train, validation, and test. Source labels stay unchanged.
SAMPLE_CONFIG = dict(window_seconds=112, stride_seconds=56, reference_windows=8,
    positive_earlier_minutes=7, positive_later_minutes=3, event_history_minutes=10,
    positive_other_preictal_minutes=0, positive_other_postictal_minutes=5,
    negative_preictal_minutes=10, negative_postictal_minutes=10,
    negative_recording_scope="annotated_only")


def sample_offsets():
    c = SAMPLE_CONFIG
    window_rows = int(round(c["window_seconds"] * SOURCE_HZ))
    stride_rows = int(round(c["stride_seconds"] * SOURCE_HZ))
    span = int(round((c["positive_earlier_minutes"] - c["positive_later_minutes"]) * 60 * SOURCE_HZ))
    if (not 0 <= c["positive_later_minutes"] < c["positive_earlier_minutes"]
            or window_rows < 1 or window_rows > span or stride_rows < 1 or c["reference_windows"] < 1):
        raise ValueError("Invalid sample window, stride, or positive interval")
    count = min(int(c["reference_windows"]), span // window_rows)
    reference = np.linspace(0, span - window_rows, count, dtype=np.int64)
    dense = np.arange(0, span - window_rows + 1, stride_rows, dtype=np.int64)
    return window_rows, np.r_[reference, dense[~np.isin(dense, reference)]]


def selected_events(dataset, target):
    c = SAMPLE_CONFIG
    records = []
    events = dataset.events[dataset.events.mouse == target]
    before = int(round(c["positive_other_preictal_minutes"] * 60_000))
    after = int(round(c["positive_other_postictal_minutes"] * 60_000))
    if min(before, after, c["event_history_minutes"]) < 0:
        raise ValueError("Negative exclusion or history duration")
    forbidden_start = events.start_pi_time_ms.to_numpy(dtype=np.int64) - before
    forbidden_stop = events.end_pi_time_ms.to_numpy(dtype=np.int64) + 2 + after
    keys = list(zip(events.raw_file, events.epoch_id))
    for filename, group in events.groupby("raw_file", sort=True):
        previous_end = None
        for event in group.sort_values("start_row").to_dict("records"):
            source = dataset.sources[filename]
            history = int(event["start_row"] - c["event_history_minutes"] * 60 * SOURCE_HZ)
            start = int(event["start_row"] - c["positive_earlier_minutes"] * 60 * SOURCE_HZ)
            stop = int(event["start_row"] - c["positive_later_minutes"] * 60 * SOURCE_HZ)
            reason = "selected"
            if history < 0 or start < 0 or stop > source["rows"]:
                reason = "insufficient_history"
            elif previous_end is not None and history <= previous_end:
                reason = "previous_seizure_in_history"
            else:
                left, right = source["start_ms"] + 2 * start, source["start_ms"] + 2 * stop
                overlap = (left < forbidden_stop) & (right > forbidden_start)
                own = keys.index((filename, event["epoch_id"]))
                overlap[own] = False
                if overlap.any():
                    reason = "other_seizure_exclusion"
            event.update(selection=reason, positive_start_row=start)
            records.append(event)
            previous_end = max(int(event["end_row"]), previous_end if previous_end is not None else -1)
    return pd.DataFrame(records).sort_values(["mouse", "day", "start_row", "raw_file"]).reset_index(drop=True)


def safe_regions(dataset, target):
    """Per recording, row ranges outside every seizure margin and positive interval."""
    c = SAMPLE_CONFIG
    events = dataset.events[dataset.events.mouse == target]
    before = int(round(c["negative_preictal_minutes"] * 60_000))
    after = int(round(c["negative_postictal_minutes"] * 60_000))
    if min(before, after) < 0:
        raise ValueError("Negative seizure exclusion duration")
    zones = [(int(e.start_pi_time_ms) - before, int(e.end_pi_time_ms) + 2 + after)
             for e in events.sort_values("start_pi_time_ms").itertuples()]
    # Positive intervals remain unavailable for negative labels when margins change.
    zones.extend((int(e.start_pi_time_ms - c["positive_earlier_minutes"] * 60_000),
                  int(e.start_pi_time_ms - c["positive_later_minutes"] * 60_000))
                 for e in events.itertuples())
    zones.sort()
    annotated = set(events.raw_file)
    scope = c["negative_recording_scope"]
    if scope not in {"annotated_only", "all_recordings"}:
        raise ValueError("Unknown negative recording scope")
    regions = {}
    for filename, source in sorted(dataset.sources.items()):
        if source["mouse"] != target or (scope == "annotated_only" and filename not in annotated):
            continue
        cursor, safe = 0, []
        for left_ms, right_ms in zones:
            left = max(0, (left_ms - source["start_ms"] + 1) // 2)
            right = min(source["rows"], (right_ms - source["start_ms"] + 1) // 2)
            if right <= 0:
                continue
            if left >= source["rows"]:
                break
            if left > cursor:
                safe.append((cursor, left))
            cursor = max(cursor, right)
        if cursor < source["rows"]:
            safe.append((cursor, source["rows"]))
        regions[filename] = safe
    return regions


def negative_pool(dataset, target, window_rows):
    stride_rows = int(round(SAMPLE_CONFIG["stride_seconds"] * SOURCE_HZ))
    records = []
    for filename, safe in safe_regions(dataset, target).items():
        source = dataset.sources[filename]
        for left, right in safe:
            for start in range(left, right - window_rows + 1, stride_rows):
                records.append(dict(mouse=target, day=source["day"], raw_file=filename,
                                    start_row=start, stop_row=start + window_rows))
    return pd.DataFrame(records, columns=["mouse", "day", "raw_file", "start_row", "stop_row"])


# Training-time random crops: alternative real windows with the same label definition.
CROP_CONFIG = dict(positive="Any 112 s window inside the event's 7-3 minute positive interval (128 s of start positions)",
                   negative="All legal 112 s starts in eligible seizure-free intervals of the same mouse and day",
                   negative_sampling="Uniform over legal starts, weighting each interval by its number of starts")


def crop_ranges(dataset, target, rows):
    """Inclusive (first, last) source start rows that keep each row's label definition."""
    window_rows, offsets = sample_offsets()
    span = int(round((SAMPLE_CONFIG["positive_earlier_minutes"] - SAMPLE_CONFIG["positive_later_minutes"]) * 60 * SOURCE_HZ))
    regions = safe_regions(dataset, target)
    ranges = []
    for row in rows.itertuples():
        if row.label == 1:
            first = int(row.start_row - offsets[int(row.window)])
            last = first + span - window_rows
        else:
            region = [(left, right) for left, right in regions.get(row.raw_file, [])
                      if left <= row.start_row and row.stop_row <= right]
            if len(region) != 1:
                raise ValueError("Negative window is not inside exactly one safe region")
            left, right = region[0]
            first, last = left, right - window_rows
        if not first <= row.start_row <= last or first < 0 or last + window_rows > dataset.sources[row.raw_file]["rows"]:
            raise ValueError("Crop range must contain the fixed window and stay inside the recording")
        ranges.append((first, last))
    return np.asarray(ranges, dtype=np.int64).reshape(-1, 2)


def build_tables(dataset, target):
    """Construct all splits with the same size, stride, class definitions, and ratio."""
    window_rows, offsets = sample_offsets()
    selection = selected_events(dataset, target)
    events = selection[selection.selection == "selected"].reset_index(drop=True)
    if not len(events):
        raise ValueError(f"No eligible events for {target}")
    positive = []
    for index, event in enumerate(events.itertuples()):
        for window, offset in enumerate(offsets):
            start = int(event.positive_start_row + offset)
            positive.append(dict(mouse=target, day=int(event.day), raw_file=event.raw_file,
                start_row=start, stop_row=start + window_rows, epoch_id=int(event.epoch_id),
                event_index=index, window=window, label=1))
    positive = pd.DataFrame(positive)
    pool = negative_pool(dataset, target, window_rows)
    negative = []
    for day, group in events.groupby("day", sort=True):
        candidates = pool[pool.day == day].sort_values(["raw_file", "start_row"])
        count = len(group) * len(offsets)
        if len(candidates) < count:
            raise ValueError(f"{target} Day {day}: need {count} unique negatives, found {len(candidates)}")
        indices = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
        matched = candidates.iloc[indices].copy()
        matched["event_index"] = np.repeat(group.index.to_numpy(), len(offsets))
        matched["window"] = np.tile(np.arange(len(offsets)), len(group))
        matched["epoch_id"], matched["label"] = -1, 0
        negative.append(matched)
    negative = pd.concat(negative).sort_values(["event_index", "window"])
    tables = {}
    for split, days in DAY_SPLITS.items():
        rows = pd.concat([positive[positive.day.isin(days)], negative[negative.day.isin(days)]], ignore_index=True)
        if not len(rows):
            raise ValueError(f"{target}: empty {split} split")
        rows["split"], rows["sample_index"] = split, np.arange(len(rows))
        tables[split] = rows
    return tables, selection


def clean_signal(signal):
    """Apply the same waveform cleaning rule to all day splits."""
    return np.asarray(signal, dtype=np.float32)


def materialize(dataset, target, split, rows, cache_dir):
    """Cache candidate windows without altering source files or annotations."""
    window_rows, _ = sample_offsets()
    rows_csv = rows.to_csv(index=False)
    signature = dict(source=dataset.signature, code_sha256=file_hash(__file__),
                     config=SAMPLE_CONFIG, rows_sha256=hashlib.sha256(rows_csv.encode()).hexdigest())
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / key
    cache.mkdir(parents=True, exist_ok=True)
    raw_path, labels_path, rows_path = cache / "raw.npy", cache / "labels.npy", cache / "windows.csv"
    stamp = cache / "manifest.json"
    if stamp.exists():
        stored = json.loads(stamp.read_text())
        if stored["signature"] != signature:
            raise ValueError("Sample cache signature mismatch")
        for name, digest in stored["files"].items():
            if file_hash(cache / name) != digest:
                raise ValueError("Candidate sample cache changed")
    else:
        raw = np.lib.format.open_memmap(str(raw_path) + ".tmp", mode="w+", dtype=np.float32,
                                       shape=(len(rows), window_rows))
        reader = dataset.reader(target, split)
        for index, row in enumerate(rows.itertuples()):
            signal = clean_signal(reader.read(row.raw_file, int(row.start_row), int(row.stop_row)))
            if signal.shape != (window_rows,):
                raise ValueError("Cleaning must preserve the configured signal window length")
            raw[index] = signal
        raw.flush()
        del raw
        Path(str(raw_path) + ".tmp").replace(raw_path)
        np.save(labels_path, rows.label.to_numpy(dtype=np.int64), allow_pickle=False)
        rows_path.write_text(rows_csv)
        stamp.write_text(json.dumps(dict(signature=signature,
            files={p.name: file_hash(p) for p in [raw_path, labels_path, rows_path]}), indent=2) + "\n")
    return (np.load(raw_path, mmap_mode="r", allow_pickle=False),
            np.load(labels_path, mmap_mode="r", allow_pickle=False))
