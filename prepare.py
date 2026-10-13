"""Frozen MouseLFP protocol: materialize the existing windows as raw signals.

Run once before research. Preprocessing belongs in train.py, after this boundary.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_protocol import (DAY_SPLITS, DEFAULT_BANDS, day_groups,
                             matched_negative_indices, negative_options,
                             positive_window_rows, training_overlap_negatives,
                             training_overlap_rows, validate_positive_windows,
                             write_json)
from utils.mouse_lfp import (EXPECTED_COUNTS, SOURCE_HZ, TARGETS, WINDOW_ROWS,
                             MouseLFPData, ParquetSignalReader, negative_windows)

ROOT = Path(__file__).resolve().parent
PROTOCOL = dict(
    version=1, dataset="MouseLFP", targets=TARGETS, event_counts=EXPECTED_COUNTS,
    total_events=978, positive_interval_minutes=[5, 1], source_hz=500,
    window_seconds=28, window_rows=14000, reference_windows_per_class=8,
    negative_matching_grid=10, training_stride_seconds=14,
    training_windows_per_class=23, day_splits=DAY_SPLITS,
    split_events=dict(train=649, validation=168, test=161),
    split_samples=dict(train=29854, validation=2688, test=2576),
    negative_exclusion_minutes=dict(before_onset=10, after_end=10),
    positive_other_seizure_exclusion_minutes=dict(before_onset=0, after_end=5),
    event_eligibility_interval_minutes=[10, 5],
    score="macro_validation_accuracy", direction="maximize",
)
FROZEN_FILES = ["prepare.py", "evaluate.py", "protocol.lock.json",
                "research_protocol.py", "utils/mouse_lfp.py", "utils/__init__.py",
                "SETTINGS_MouseLFP.json"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_digest():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def verify_protocol():
    lock = json.loads((ROOT / "protocol.lock.json").read_text())
    if lock["protocol"] != PROTOCOL:
        raise ValueError("Fixed data/window/split protocol differs from protocol.lock.json")
    for name, expected in lock["files"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Frozen file changed: {name}. Restore the fixed protocol.")


def verify_dataset_signature(signature):
    lock = json.loads((ROOT / "protocol.lock.json").read_text())
    if signature != lock["dataset_signature"]:
        raise ValueError("Dataset annotations/source metadata differ from the fixed original dataset")


def frozen_hashes():
    verify_protocol()
    return {name: sha256_file(ROOT / name) for name in FROZEN_FILES}


def window_tables(data, target, output):
    """Reuse original event selection, offsets and same-day negative assignment."""
    events = data.selected[data.selected.mouse == target].reset_index(drop=True)
    groups = day_groups(events)
    negatives = negative_windows(data.events, data.sources, target, None,
                                 **negative_options(data))
    indices = matched_negative_indices(events, negatives, 8)
    matched = negatives.iloc[indices.ravel()].copy().reset_index(drop=True)
    matched["event_index"] = np.repeat(np.arange(len(events)), 8)
    matched["window"] = np.tile(np.arange(8), len(events))
    matched["event_day"] = events.day.to_numpy()[matched.event_index.to_numpy()]
    matched["split"] = matched.event_day.map({d: s for s, ds in DAY_SPLITS.items() for d in ds})
    matched["same_day"] = matched.day == matched.event_day
    if not matched.same_day.all():
        raise ValueError("Negative window crossed the event day")
    matched.to_csv(output / "matched_negative_windows.csv", index=False)
    positives = positive_window_rows(events, DEFAULT_BANDS, 8)
    validate_positive_windows(data, positives, output, "reference_positive")
    train_positive = training_overlap_rows(events, groups, DEFAULT_BANDS, 8, 14)
    validate_positive_windows(data, train_positive, output, "train_positive")
    train_negative = training_overlap_negatives(data, target, events, groups, output,
                                               23, 8, 14)
    events.to_csv(output / "events.csv", index=False)
    write_json(output / "split.json", dict(days=DAY_SPLITS,
               event_indices={k: v.tolist() for k, v in groups.items()}))
    tables = {}
    for split, group in groups.items():
        positive = train_positive if split == "train" else positives[positives.event_index.isin(group)]
        negative = train_negative if split == "train" else matched[matched.event_index.isin(group)]
        positive = positive.sort_values(["event_index", "window"]).copy()
        negative = negative.sort_values(["event_index", "window"]).copy()
        positive["label"], negative["label"] = 1, 0
        rows = pd.concat([positive, negative], ignore_index=True)
        rows["split"] = split
        rows["sample_index"] = np.arange(len(rows))
        expected = len(group) * (23 if split == "train" else 8) * 2
        if len(rows) != expected or not (rows.stop_row - rows.start_row == WINDOW_ROWS).all():
            raise ValueError("Fixed window count/length changed")
        if not rows.day.isin(DAY_SPLITS[split]).all():
            raise ValueError("Window crossed the date split")
        tables[split] = rows
    return tables, {k: len(v) for k, v in groups.items()}


def prepare(data_dir, output, targets):
    verify_protocol()
    settings = json.loads((ROOT / "SETTINGS_MouseLFP.json").read_text())
    if data_dir is not None:
        settings["datadir"] = str(data_dir)
    data = MouseLFPData(settings)
    verify_dataset_signature(data.signature)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol_sha256=protocol_digest(), dataset_signature=data.signature,
                    targets=targets, frozen_code=frozen_hashes())
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("Prepared dataset identity changed; use a fresh output directory")
        verify_prepared(output)
        print(f"Prepared windows already verified: {output}", flush=True)
        return previous
    # A partial preparation can be rebuilt; manifest is published only on success.
    data.write_manifest(output)
    manifest = dict(identity=identity, subjects={})
    reader = ParquetSignalReader(data.root)
    for target in targets:
        subject_dir = output / target
        subject_dir.mkdir(exist_ok=True)
        tables, event_counts = window_tables(data, target, subject_dir)
        subject = dict(event_counts=event_counts, splits={})
        for split, rows in tables.items():
            raw_path = subject_dir / f"{split}_raw.npy"
            label_path = subject_dir / f"{split}_labels.npy"
            rows_path = subject_dir / f"{split}_windows.csv"
            raw = np.lib.format.open_memmap(str(raw_path) + ".tmp", mode="w+",
                                           dtype=np.float32, shape=(len(rows), WINDOW_ROWS))
            for i, row in enumerate(rows.itertuples()):
                raw[i] = reader.read(row.raw_file, int(row.start_row), int(row.stop_row))
                if (i + 1) % 1000 == 0:
                    print(f"{target} {split}: {i + 1}/{len(rows)} raw windows", flush=True)
            raw.flush()
            del raw
            Path(str(raw_path) + ".tmp").replace(raw_path)
            np.save(label_path, rows.label.to_numpy(dtype=np.int64), allow_pickle=False)
            rows.to_csv(rows_path, index=False)
            subject["splits"][split] = dict(samples=len(rows),
                files={p.name: sha256_file(p) for p in [raw_path, label_path, rows_path]})
            print(f"Prepared {target} {split}: {len(rows)}", flush=True)
        manifest["subjects"][target] = subject
    if targets == TARGETS:
        for split in DAY_SPLITS:
            if sum(s["event_counts"][split] for s in manifest["subjects"].values()) != PROTOCOL["split_events"][split]:
                raise ValueError("Fixed full-cohort event count changed")
            if sum(s["splits"][split]["samples"] for s in manifest["subjects"].values()) != PROTOCOL["split_samples"][split]:
                raise ValueError("Fixed full-cohort sample count changed")
    write_json(manifest_path, manifest)
    return manifest


def verify_prepared(root, splits=("train", "validation", "test")):
    verify_protocol()
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    identity = manifest["identity"]
    verify_dataset_signature(identity["dataset_signature"])
    if identity["protocol_sha256"] != protocol_digest() or identity["frozen_code"] != frozen_hashes():
        raise ValueError("Prepared data was built with a different fixed protocol")
    if set(manifest["subjects"]) != set(identity["targets"]):
        raise ValueError("Prepared subject list changed")
    for target, subject in manifest["subjects"].items():
        for split in splits:
            info = subject["splits"][split]
            for name, digest in info["files"].items():
                if sha256_file(root / target / name) != digest:
                    raise ValueError(f"Prepared data changed: {target}/{name}")
            raw, labels = load_split(root, target, split)
            expected = subject["event_counts"][split] * (46 if split == "train" else 16)
            if raw.shape != (expected, WINDOW_ROWS) or raw.dtype != np.float32 or len(labels) != expected:
                raise ValueError("Prepared window shape/count changed")
            if not np.array_equal(labels, np.r_[np.ones(expected // 2), np.zeros(expected // 2)]):
                raise ValueError("Prepared labels/order changed")
    return manifest


def load_split(root, target, split):
    if split not in DAY_SPLITS:
        raise ValueError("Unknown split")
    subject = Path(root) / target
    return (np.load(subject / f"{split}_raw.npy", mmap_mode="r", allow_pickle=False),
            np.load(subject / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False))


def parse_targets(text):
    targets = text.split(",")
    if not targets or len(set(targets)) != len(targets) or any(t not in TARGETS for t in targets):
        raise ValueError("Select distinct MouseLFP subjects")
    return [t for t in TARGETS if t in targets]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("autoresearch-data"))
    parser.add_argument("--targets", default=",".join(TARGETS), help="Subset is a smoke dataset")
    args = parser.parse_args()
    prepare(args.data_dir, args.output, parse_targets(args.targets))
