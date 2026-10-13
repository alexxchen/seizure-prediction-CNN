"""Single-channel, 500-Hz mouse LFP with auditable event/window selection.

Positive windows cover 5-1 minutes before onset and avoid other seizures
and their following five minutes. Negatives exclude all seizures with
ten-minute margins before onset and after the end.
Events and their matched negatives stay grouped across dataset partitions.
"""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import resample_poly
import stft


TARGETS = ["X1", "X2", "X4", "X9", "X10", "X14", "X15"]
EXPECTED_COUNTS = dict(zip(TARGETS, [137, 259, 59, 267, 61, 169, 26]))
SOURCE_HZ = 500
WINDOW_ROWS = 28 * SOURCE_HZ
FEATURE_SHAPE = (1, 1, 55, 114)
CACHE_VERSION = 2


def positive_exclusion_overlaps(windows, events, sources, margin_minutes=10, *,
                                preictal_minutes=None, postictal_minutes=None):
    """Identify positives overlapping another seizure's pre/ictal/post zone.

    All annotations, including rejected events and adjacent source files, are
    used in absolute time. Only the window's own target event is exempt.
    Intervals are half-open; the annotated end row is inclusive.
    """
    before = margin_minutes if preictal_minutes is None else preictal_minutes
    after = margin_minutes if postictal_minutes is None else postictal_minutes
    if any(not isinstance(value, (int, np.integer)) or value < 0 for value in [before, after]):
        raise ValueError('Positive exclusion margin must be a nonnegative integer')
    result = np.zeros(len(windows), dtype=bool)
    coordinates = windows.reset_index(drop=True)
    for mouse, group in coordinates.groupby('mouse', sort=False):
        labels = events[events.mouse == mouse].reset_index(drop=True)
        keys = {(row.raw_file, int(row.epoch_id)): index for index, row in enumerate(labels.itertuples())}
        own = np.asarray([keys[(row.raw_file, int(row.epoch_id))] for row in group.itertuples()])
        origins = group.raw_file.map({name: source['start_ms'] for name, source in sources.items()})
        starts = origins.to_numpy(dtype=np.int64) + 2*group.start_row.to_numpy(dtype=np.int64)
        stops = origins.to_numpy(dtype=np.int64) + 2*group.stop_row.to_numpy(dtype=np.int64)
        forbidden_starts = labels.start_pi_time_ms.to_numpy(dtype=np.int64) - before*60_000
        forbidden_stops = labels.end_pi_time_ms.to_numpy(dtype=np.int64) + 2 + after*60_000
        for left in range(0, len(group), 512):
            right = min(left+512, len(group))
            overlap = ((starts[left:right, None] < forbidden_stops[None, :]) &
                       (stops[left:right, None] > forbidden_starts[None, :]))
            overlap[np.arange(right-left), own[left:right]] = False
            result[group.index.to_numpy()[left:right]] = overlap.any(axis=1)
    return result


def select_events(events, sources, earlier_minutes=10, later_minutes=5,
                  other_seizure_margin_minutes=None, exclusion_band=None,
                  other_seizure_preictal_minutes=None, other_seizure_postictal_minutes=None):
    """Select complete, isolated preictal intervals within their source file."""
    if not 0 <= later_minutes < earlier_minutes:
        raise ValueError("Invalid preictal interval")
    earlier_rows = earlier_minutes * 60 * SOURCE_HZ
    later_rows = later_minutes * 60 * SOURCE_HZ
    records = []
    for filename, group in events.groupby("raw_file", sort=True):
        source = sources[filename]
        previous_end = None
        for event in group.sort_values("start_row").to_dict("records"):
            start, end = int(event["start_row"]), int(event["end_row"])
            if not 0 <= start <= end < source["rows"]:
                raise ValueError(f"Annotation rows out of bounds: {filename}")
            if (event["start_pi_time_ms"] != source["start_ms"] + 2 * start or
                    event["end_pi_time_ms"] != source["start_ms"] + 2 * end):
                raise ValueError(f"Annotation timestamps do not match rows: {filename}")
            window_start = start - earlier_rows
            if window_start < 0:
                reason = "insufficient_history_in_file"
            elif previous_end is not None and window_start <= previous_end:
                reason = f"previous_seizure_within_{earlier_minutes}min"
            else:
                reason = "selected"
            event.update(selection=reason, preictal_start_row=window_start,
                         preictal_stop_row=start - later_rows)
            records.append(event)
            previous_end = max(end, previous_end if previous_end is not None else end)
    result = pd.DataFrame(records).sort_values(
        ["mouse", "day", "start_row", "raw_file"]).reset_index(drop=True)
    if other_seizure_margin_minutes is not None:
        guard_earlier, guard_later = exclusion_band or (earlier_minutes, later_minutes)
        if not 0 <= guard_later < guard_earlier:
            raise ValueError('Invalid positive exclusion interval')
        candidates = result[result.selection == 'selected'].copy()
        candidates['start_row'] = candidates.start_row - guard_earlier*60*SOURCE_HZ
        candidates['stop_row'] = result.loc[candidates.index, 'start_row'] - guard_later*60*SOURCE_HZ
        before = other_seizure_margin_minutes if other_seizure_preictal_minutes is None else other_seizure_preictal_minutes
        after = other_seizure_margin_minutes if other_seizure_postictal_minutes is None else other_seizure_postictal_minutes
        mask = positive_exclusion_overlaps(candidates, events, sources,
                                           preictal_minutes=before, postictal_minutes=after)
        reason = f'other_seizure_prepost_{before}min' if before == after else f'other_seizure_pre{before}_post{after}min'
        result.loc[candidates.index[mask], 'selection'] = reason
        result.loc[candidates.index[candidates.start_row < 0], 'selection'] = 'insufficient_history_in_exclusion_band'
    return result


def negative_windows(events, sources, target, count=None, stride_rows=WINDOW_ROWS,
                     preictal_minutes=10, postictal_minutes=10):
    """Sample unique, chronological negatives from that mouse's labeled files.

Use absolute timestamps from all events, including rejected prediction events
and events in adjacent files, as exclusion zones.
Files without event records are omitted because processing coverage is unknown.
The default stride preserves nonoverlapping evaluation/reference windows.
"""
    if not isinstance(stride_rows, (int, np.integer)) or not 1 <= stride_rows <= WINDOW_ROWS:
        raise ValueError("Negative stride must be between 1 and WINDOW_ROWS")
    if any(not isinstance(value, (int, np.integer)) or value < 0
           for value in (preictal_minutes, postictal_minutes)):
        raise ValueError("Negative exclusion margins must be nonnegative integer minutes")
    candidates = []
    subject = events[events.mouse == target]
    exclusions = [
        (int(event.start_pi_time_ms) - preictal_minutes * 60 * 1000,
         int(event.end_pi_time_ms) + 2 + postictal_minutes * 60 * 1000)
        for event in subject.sort_values("start_pi_time_ms").itertuples()
    ]
    for filename, group in subject.groupby("raw_file", sort=False):
        cursor = 0
        rows = sources[filename]["rows"]
        origin_ms = int(sources[filename]["start_ms"])
        safe = []
        for start_ms, stop_ms in exclusions:
            # Ceil division maps the absolute, half-open exclusion interval
            # onto this file's 2-ms sample grid, even if file origins differ.
            start = max(0, (start_ms - origin_ms + 1) // 2)
            stop = min(rows, (stop_ms - origin_ms + 1) // 2)
            if stop <= 0:
                continue
            if start >= rows:
                break
            if start > cursor:
                safe.append((cursor, start))
            cursor = max(cursor, stop)
        if cursor < rows:
            safe.append((cursor, rows))
        for start, stop in safe:
            for row in range(start, stop - WINDOW_ROWS + 1, stride_rows):
                candidates.append(dict(mouse=target, day=int(group.day.iloc[0]),
                                       raw_file=filename, start_row=row,
                                       stop_row=row + WINDOW_ROWS))
    candidates.sort(key=lambda window: (window["day"], window["raw_file"], window["start_row"]))
    if count is None:
        return pd.DataFrame(candidates, columns=["mouse", "day", "raw_file", "start_row", "stop_row"])
    if len(candidates) < count:
        raise ValueError(f"{target}: need {count} negative windows, found {len(candidates)}")
    indices = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
    return pd.DataFrame([candidates[index] for index in indices])


class ParquetSignalReader:
    """Read only time/LFP columns, retaining a small cache of row groups."""

    def __init__(self, root):
        self.root = Path(root)
        self.filename = None
        self.groups = OrderedDict()

    def read(self, filename, start, stop):
        if filename != self.filename:
            self.parquet = pq.ParquetFile(self.root / filename)
            sizes = [self.parquet.metadata.row_group(i).num_rows
                     for i in range(self.parquet.num_row_groups)]
            self.offsets = np.r_[0, np.cumsum(sizes)]
            self.filename = filename
            self.groups.clear()
        if not 0 <= start < stop <= self.offsets[-1]:
            raise ValueError(f"Signal window out of bounds: {filename}, {start}:{stop}")
        first = np.searchsorted(self.offsets, start, side="right") - 1
        last = np.searchsorted(self.offsets, stop - 1, side="right") - 1
        pieces = []
        for index in range(first, last + 1):
            if index not in self.groups:
                table = self.parquet.read_row_group(index, columns=["pi_time_ms", "lfp"])
                times = table.column("pi_time_ms").to_numpy()
                signal = table.column("lfp").to_numpy()
                if (not np.all(np.diff(times) == 2) or
                        not np.isfinite(signal).all()):
                    raise ValueError(f"Invalid timestamps or LFP values: {filename}, row group {index}")
                self.groups[index] = signal
            signal = self.groups[index]
            self.groups.move_to_end(index)
            left = max(start, self.offsets[index]) - self.offsets[index]
            right = min(stop, self.offsets[index + 1]) - self.offsets[index]
            pieces.append(signal[left:right])
            while len(self.groups) > 12:
                self.groups.popitem(last=False)
        return np.concatenate(pieces)


def window_features(signal):
    """Use the existing 256-Hz CHBMIT feature convention on one LFP window."""
    if signal.shape != (WINDOW_ROWS,):
        raise ValueError("Expected a 28-second, 500-Hz LFP window")
    resampled = resample_poly(signal, 64, 125)[:, None]
    spectrum = stft.spectrogram(resampled, framelength=256, centered=False)
    # stft 0.5.2 squeezes the electrode axis when there is only one channel.
    if spectrum.ndim == 2:
        spectrum = spectrum[:, :, None]
    features = np.abs(np.transpose(spectrum, (2, 1, 0))) + 1e-6
    features = np.concatenate((features[:, :, 1:57], features[:, :, 64:117],
                               features[:, :, 124:]), axis=-1)
    features = np.maximum(np.log10(features), 0)[None].astype(np.float32)
    if features.shape != FEATURE_SHAPE or not np.isfinite(features).all():
        raise ValueError("Invalid single-channel spectrogram")
    return features


class MouseLFPData:
    def __init__(self, settings):
        self.settings = settings
        self.preictal_band = (10, 5)
        self.root = Path(settings["datadir"])
        annotation_path = self.root / settings["annotations"]
        self.events = pq.read_table(annotation_path).to_pandas()
        required = {"mouse", "day", "raw_file", "epoch_id", "start_row", "end_row",
                    "start_pi_time_ms", "end_pi_time_ms"}
        if not required.issubset(self.events.columns) or self.events[list(required)].isna().any().any():
            raise ValueError("Missing annotation fields or values")
        if self.events.duplicated(["raw_file", "epoch_id"]).any():
            raise ValueError("Duplicate event identifiers")
        self.sources = {}
        for filename in sorted(self.events.raw_file.unique()):
            if Path(filename).name != filename:
                raise ValueError("Annotation filenames must be basenames")
            path = self.root / filename
            parquet = pq.ParquetFile(path)
            if not {"pi_time_ms", "lfp"}.issubset(parquet.schema_arrow.names):
                raise ValueError(f"Missing time or LFP column: {filename}")
            time_column = parquet.schema_arrow.get_field_index("pi_time_ms")
            previous_stop = None
            start_ms = None
            for index in range(parquet.num_row_groups):
                group = parquet.metadata.row_group(index)
                stats = group.column(time_column).statistics
                if (stats is None or not stats.has_min_max or stats.null_count or
                        stats.max - stats.min != 2 * (group.num_rows - 1) or
                        (previous_stop is not None and stats.min != previous_stop + 2)):
                    raise ValueError(f"Expected contiguous 500-Hz timestamps: {filename}")
                if start_ms is None:
                    start_ms = stats.min
                previous_stop = stats.max
            stat = path.stat()
            self.sources[filename] = dict(rows=parquet.metadata.num_rows, start_ms=start_ms,
                                          size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        self.positive_margin_minutes = 5
        self.positive_preictal_minutes = 0
        self.positive_exclusion_band = (5, 1)
        self.selection = select_events(self.events, self.sources, *self.preictal_band,
                                       other_seizure_margin_minutes=self.positive_margin_minutes,
                                       exclusion_band=self.positive_exclusion_band,
                                       other_seizure_preictal_minutes=self.positive_preictal_minutes,
                                       other_seizure_postictal_minutes=self.positive_margin_minutes)
        self.selected = self.selection[self.selection.selection == "selected"].reset_index(drop=True)
        self.counts = self.selected.groupby("mouse").size().to_dict()
        if self.counts != EXPECTED_COUNTS:
            raise ValueError(f"Expected the current 978 events {EXPECTED_COUNTS}; found {self.counts}")
        self.signature = dict(
            version=CACHE_VERSION,
            annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
            sources=self.sources, feature_shape=FEATURE_SHAPE,
            source_hz=500, target_hz=256, window_seconds=28,
            preictal_minutes=5, horizon_minutes=5, postictal_minutes=10,
            negative_exclusion="absolute_time_across_files",
            event_selection_interval=[10, 5], negative_preictal_minutes=10,
            event_selection='isolated history; entire positive interval avoids all other seizure exclusion zones',
            positive_exclusion=dict(
                version=2, preictal_minutes=0, postictal_minutes=5, interval=[5, 1],
                annotation_scope='all events of same mouse, absolute time across files',
                target_event_exempt=True, selection='reject entire event on any interval overlap'))
        # Normalize tuples/NumPy values to the JSON representation used on disk.
        self.signature = json.loads(json.dumps(self.signature))

    def write_manifest(self, result_dir):
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        self.selection.to_csv(result_dir / "event_selection.csv", index=False)
        summary = dict(total_events=len(self.events), selected_events=len(self.selected),
                       counts=self.counts,
                       exclusion_counts=self.selection.selection.value_counts().to_dict(),
                       preprocessing=self.signature)
        (result_dir / "data_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"MouseLFP: selected {len(self.selected)}/{len(self.events)} events; {self.counts}", flush=True)

