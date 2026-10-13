"""Freeze the private raw source inventory and day splits, without fixing windows."""

import argparse
import hashlib
import json
from pathlib import Path

from research_protocol import write_json
from source_data import DAY_SPLITS, SOURCE_HZ, TARGETS, SourceDataset, file_hash

ROOT = Path(__file__).resolve().parent
PROTOCOL = dict(version=2, dataset="MouseLFP", source_hz=SOURCE_HZ, targets=TARGETS,
                day_splits=DAY_SPLITS, score="macro_validation_accuracy", direction="maximize")
FROZEN_FILES = ["prepare.py", "evaluate.py", "source_data.py", "protocol.lock.json",
                "research_protocol.py", "utils/mouse_lfp.py", "utils/__init__.py",
                "SETTINGS_MouseLFP.json", "train.py", "models/cnn.py", "models/__init__.py"]
sha256_file = file_hash


def protocol_digest():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def verify_protocol():
    lock = json.loads((ROOT / "protocol.lock.json").read_text())
    if lock["protocol"] != PROTOCOL:
        raise ValueError("Fixed raw source/day-split protocol changed")
    for name, expected in lock["files"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Fixed source, CNN, trainer, or evaluator changed: {name}")


def frozen_hashes():
    verify_protocol()
    return {name: sha256_file(ROOT / name) for name in FROZEN_FILES}


def source_settings(data_dir=None):
    settings = json.loads((ROOT / "SETTINGS_MouseLFP.json").read_text())
    if data_dir is not None:
        settings["datadir"] = str(data_dir)
    return settings


def load_dataset(prepared_dir, data_dir=None):
    manifest = verify_prepared(prepared_dir)
    settings = source_settings(data_dir)
    return SourceDataset(settings["datadir"], settings["annotations"], manifest["identity"]["dataset_signature"])


def verify_prepared(root, splits=None):
    verify_protocol()
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    lock = json.loads((ROOT / "protocol.lock.json").read_text())
    identity = manifest["identity"]
    if (identity["protocol_sha256"] != protocol_digest()
            or identity["dataset_signature"] != lock["dataset_signature"]
            or identity["frozen_code"] != frozen_hashes()):
        raise ValueError("Source inventory was prepared for a different research protocol")
    return manifest


def prepare(data_dir, output, targets):
    verify_protocol()
    settings = source_settings(data_dir)
    lock = json.loads((ROOT / "protocol.lock.json").read_text())
    dataset = SourceDataset(settings["datadir"], settings["annotations"], lock["dataset_signature"])
    identity = dict(protocol_sha256=protocol_digest(), dataset_signature=dataset.signature,
                    targets=targets, frozen_code=frozen_hashes())
    manifest = dict(version=2, identity=identity, source_events=len(dataset.events),
        source_event_counts=dataset.events.groupby("mouse").size().to_dict(),
        day_splits=DAY_SPLITS,
        recordings={target: [name for name, s in dataset.sources.items() if s["mouse"] == target]
                    for target in targets})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Existing source inventory differs; use a fresh prepared directory")
        print(f"Source inventory verified: {output}", flush=True)
    else:
        write_json(path, manifest)
        print(f"Prepared source inventory: {len(dataset.sources)} recordings, {len(dataset.events)} events", flush=True)
    return manifest


def parse_targets(text):
    targets = text.split(",")
    if not targets or len(set(targets)) != len(targets) or any(t not in TARGETS for t in targets):
        raise ValueError("Select distinct MouseLFP subjects")
    return [t for t in TARGETS if t in targets]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("autoresearch-data"))
    parser.add_argument("--targets", default=",".join(TARGETS))
    args = parser.parse_args()
    prepare(args.data_dir, args.output, parse_targets(args.targets))
