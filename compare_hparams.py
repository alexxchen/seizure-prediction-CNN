"""Run a finite baseline/candidate comparison on MouseLFP validation days.

This entry point runs a fixed comparison without evaluating test signals.
The supervised reference trainer and data pipelines stay unchanged on disk.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from evaluate import prediction_metrics, seed_everything, validate_sample_table
from prepare import load_dataset, parse_targets, sha256_file
from meta_learning.common import write_json
from source_data import DAY_SPLITS, TARGETS
import input_pipeline
import sample_pipeline
import train


ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ["compare_hparams.py", "train.py", "models/cnn.py", "models/__init__.py",
                "sample_pipeline.py", "input_pipeline.py", "source_data.py", "prepare.py",
                "evaluate.py", "meta_learning/common.py",
                "SETTINGS_MouseLFP.json", "requirements.txt", "Dockerfile"]


def source_hashes():
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def configurations(args):
    baseline = dict(learning_rate=train.LEARNING_RATE, batch_size=train.BATCH_SIZE,
                    weight_decay=train.WEIGHT_DECAY, label_smoothing=train.LABEL_SMOOTHING,
                    dropout=0.5, max_epochs=train.MAX_EPOCHS)
    candidate = dict(baseline, learning_rate=args.learning_rate, batch_size=args.batch_size,
                     weight_decay=args.weight_decay, label_smoothing=args.label_smoothing,
                     dropout=args.dropout)
    if candidate == baseline:
        raise ValueError("Candidate must differ from the baseline")
    return dict(baseline=baseline, candidate=candidate)


def configure_trainer(config):
    """Apply explicit overrides only inside this trial's fresh Python process."""
    train.LEARNING_RATE = config["learning_rate"]
    train.BATCH_SIZE = config["batch_size"]
    train.WEIGHT_DECAY = config["weight_decay"]
    train.LABEL_SMOOTHING = config["label_smoothing"]
    original_builder = train.build_model

    def build_model(input_shape):
        model = original_builder(input_shape)
        model.drop1.p = model.drop2.p = config["dropout"]
        return model

    train.build_model = build_model


def signal_audit(raw):
    """Describe a deterministic subset without cleaning or rejecting samples."""
    indices = np.linspace(0, len(raw) - 1, min(128, len(raw)), dtype=np.int64)
    values = np.asarray(raw[indices], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite raw LFP in the signal-quality audit")
    deviations = values.std(axis=1)
    return dict(sampled_windows=len(indices), total_windows=len(raw),
                finite=True, minimum=float(values.min()), maximum=float(values.max()),
                constant_windows=int((deviations == 0).sum()),
                window_std_quantiles=np.quantile(deviations, [0, .25, .5, .75, 1]).tolist(),
                scope="Uniform window subset; source row/timestamp alignment verified by SourceDataset")


def worker(args):
    metadata = json.loads((args.output / "comparison.json").read_text())
    if source_hashes() != metadata["source_sha256"]:
        raise ValueError("Project source changed after the comparison started")
    configure_trainer(metadata["configurations"][args.variant])
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dataset = load_dataset(args.prepared_dir, args.data_dir)
    tables, selection = sample_pipeline.build_tables(dataset, args.subject)
    for split, table in tables.items():
        validate_sample_table(table, dataset, args.subject, split)
    work = args.output / "trials" / args.variant / args.subject
    work.mkdir(parents=True, exist_ok=True)
    selection.to_csv(work / "event_selection.csv", index=False)
    summary = {}
    for split, table in tables.items():
        positives = int((table.label == 1).sum())
        negatives = int((table.label == 0).sum())
        summary[split] = dict(samples=len(table), positives=positives, negatives=negatives,
                              positive_fraction=positives / len(table),
                              eligible_events=int(((selection.selection == "selected") &
                                  selection.day.isin(DAY_SPLITS[split])).sum()))
    write_json(work / "sampling_summary.json", summary)
    write_json(work / "hyperparameters.json", metadata["configurations"][args.variant])
    cache = args.output / "sample-cache"
    train_raw, train_y = sample_pipeline.materialize(
        dataset, args.subject, "train", tables["train"], cache)
    val_raw, val_y = sample_pipeline.materialize(
        dataset, args.subject, "validation", tables["validation"], cache)
    write_json(work / "signal_quality_audit.json", dict(
        train=signal_audit(train_raw), validation=signal_audit(val_raw)))
    predictor = train.fit(train_raw, train_y, val_raw, val_y, work, device,
                          budget_seconds=args.budget_seconds, seed=args.seed)
    metrics, probabilities = prediction_metrics(val_y, predictor.predict_proba(val_raw))
    provenance = tables["validation"].copy()
    columns = ["mouse", "day", "raw_file", "start_row", "stop_row", "label"]
    metrics["evaluation_protocol_sha256"] = hashlib.sha256(
        provenance[columns].to_csv(index=False).encode()).hexdigest()
    provenance["probability"] = probabilities
    provenance["predicted"] = (probabilities >= 0.5).astype(np.int64)
    provenance.to_csv(work / "predictions_validation.csv", index=False)
    metrics["peak_vram_mb"] = (torch.cuda.max_memory_allocated(device) / 2**20
                               if device.type == "cuda" else 0.0)
    history = json.loads((work / "history.json").read_text())
    metrics["epochs_run"] = len(history)
    checkpoint = torch.load(work / "checkpoint.pt", map_location="cpu", weights_only=True)
    metrics["best_epoch"] = int(checkpoint["epoch"])
    if source_hashes() != metadata["source_sha256"]:
        raise ValueError("Project source changed during the trial")
    write_json(work / "validation_metrics.json", metrics)
    print(f"{args.subject} {args.variant}: validation_accuracy={metrics['accuracy']:.9f}", flush=True)


def compare(args):
    configs = configurations(args)
    # Refuse reuse so a finite comparison cannot mix old checkpoints or settings.
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    for name in SOURCE_FILES:
        destination = args.output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    if source_hashes() != hashes:
        raise ValueError("Project source changed while snapshotting")
    write_json(args.output / "comparison.json", dict(
        mode="finite_hyperparameter_comparison", targets=args.targets,
        configurations=configs, budget_seconds_per_mouse=args.budget_seconds, seed=args.seed,
        device=args.device, day_splits=DAY_SPLITS, threshold=0.5,
        score="mean_validation_accuracy", sample_config=sample_pipeline.SAMPLE_CONFIG,
        sample_offsets_rows=sample_pipeline.sample_offsets()[1].tolist(),
        preprocessing=input_pipeline.PREPROCESS, source_sha256=hashes,
        prepared_manifest_sha256=sha256_file(args.prepared_dir / "manifest.json"),
        runtime={name: importlib.metadata.version(name) for name in
                 ["torch", "numpy", "scipy", "stft", "scikit-learn", "pyarrow"]}))
    results = {variant: {} for variant in configs}
    started = time.monotonic()
    for subject in args.targets:
        for variant in configs:
            work = args.output / "trials" / variant / subject
            work.mkdir(parents=True)
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--output", str(args.output), "--prepared-dir", str(args.prepared_dir),
                       "--data-dir", str(args.data_dir), "--device", args.device,
                       "--budget-seconds", str(args.budget_seconds), "--seed", str(args.seed),
                       "--subject", subject, "--variant", variant]
            print(f"Starting {subject} {variant}", flush=True)
            with (work / "train.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=args.budget_seconds + 600)
            metrics = json.loads((work / "validation_metrics.json").read_text())
            results[variant][subject] = metrics
            if variant == "candidate" and metrics["evaluation_protocol_sha256"] != \
                    results["baseline"][subject]["evaluation_protocol_sha256"]:
                raise ValueError("Baseline and candidate validation samples differ")
            print(f"Finished {subject} {variant}: accuracy={metrics['accuracy']:.9f}", flush=True)
            write_json(args.output / "progress.json", results)
    means = {variant: float(np.mean([row["accuracy"] for row in mice.values()]))
             for variant, mice in results.items()}
    delta = means["candidate"] - means["baseline"]
    write_json(args.output / "result.json", dict(
        validation_accuracy=means, candidate_minus_baseline=delta,
        improved=delta > 1e-6, full_cohort=args.targets == TARGETS,
        subjects=results, wall_seconds=time.monotonic() - started))
    print(json.dumps(dict(validation_accuracy=means, candidate_minus_baseline=delta,
                          improved=delta > 1e-6)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/MouseLFP"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", type=parse_targets, default=TARGETS)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--budget-seconds", type=float, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=train.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=train.BATCH_SIZE)
    parser.add_argument("--weight-decay", type=float, default=train.WEIGHT_DECAY)
    parser.add_argument("--label-smoothing", type=float, default=train.LABEL_SMOOTHING)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--subject", choices=TARGETS, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=["baseline", "candidate"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0 or
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            not math.isfinite(args.weight_decay) or args.weight_decay < 0 or
            args.batch_size < 1 or args.seed < 0 or
            not 0 <= args.dropout < 1 or not 0 <= args.label_smoothing < 1):
        parser.error("Invalid training hyperparameters")
    if bool(args.subject) != bool(args.variant):
        parser.error("Worker requires both --subject and --variant")
    worker(args) if args.subject else compare(args)


if __name__ == "__main__":
    main()
