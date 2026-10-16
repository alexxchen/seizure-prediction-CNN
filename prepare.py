"""Verify read-only MouseLFP sources independently of model/training code."""

import argparse
import json
from pathlib import Path

from meta_learning.common import write_json
from source_data import DAY_SPLITS, SOURCE_HZ, TARGETS, SourceDataset, file_hash, inspect_sources

ROOT = Path(__file__).resolve().parent
sha256_file = file_hash


def source_settings(data_dir=None):
    settings = json.loads((ROOT / "SETTINGS_MouseLFP.json").read_text())
    if data_dir is not None:
        settings["datadir"] = str(data_dir)
    return settings


def verify_prepared(root, splits=None):
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    # Read existing source inventories without inheriting their old code locks.
    signature = manifest.get("signature", manifest.get("identity", {}).get("dataset_signature"))
    if not signature or set(signature) != {"annotation_sha256", "sources"}:
        raise ValueError("Invalid source inventory")
    if manifest.get("day_splits") != DAY_SPLITS:
        raise ValueError("Source inventory day splits differ")
    return manifest


def inventory_signature(manifest):
    return manifest.get("signature", manifest.get("identity", {}).get("dataset_signature"))


def load_dataset(prepared_dir, data_dir=None):
    manifest = verify_prepared(prepared_dir)
    settings = source_settings(data_dir)
    return SourceDataset(settings["datadir"], settings["annotations"], inventory_signature(manifest))


def parse_targets(text):
    targets = text.split(",")
    if not targets or len(set(targets)) != len(targets) or any(t not in TARGETS for t in targets):
        raise ValueError("Choose distinct MouseLFP subjects")
    return [target for target in TARGETS if target in targets]


def prepare(data_dir, output, targets):
    settings = source_settings(data_dir)
    output = Path(output)
    if (output / "manifest.json").exists():
        manifest = verify_prepared(output)
        SourceDataset(settings["datadir"], settings["annotations"], inventory_signature(manifest))
        print(f"Source inventory verified: {output}")
        return manifest
    signature = inspect_sources(settings["datadir"], settings["annotations"])
    dataset = SourceDataset(settings["datadir"], settings["annotations"], signature)
    manifest = dict(version=3, source_hz=SOURCE_HZ, annotations=settings["annotations"],
                    targets=targets, signature=signature, day_splits=DAY_SPLITS,
                    source_events=len(dataset.events),
                    source_event_counts=dataset.events.groupby("mouse").size().to_dict())
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {len(dataset.sources)} recordings and {len(dataset.events)} source events: {output}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("prepared"))
    parser.add_argument("--targets", type=parse_targets, default=TARGETS)
    args = parser.parse_args()
    prepare(args.data_dir, args.output, args.targets)
