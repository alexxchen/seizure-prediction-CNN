"""Finite original/frozen/learned frontend comparison on validation days only."""

import argparse
import csv
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

from compare_hparams import signal_audit
from evaluate import prediction_metrics, seed_everything, validate_sample_table
from prepare import load_dataset, parse_targets, sha256_file, verify_prepared
from meta_learning.common import write_json
from source_data import DAY_SPLITS, TARGETS
import input_pipeline
import learnable_frontend
import sample_pipeline
import train


ROOT = Path(__file__).resolve().parent
VARIANTS = ["original", "frozen", "learned"]
SOURCE_FILES = ["compare_frontends.py", "learnable_frontend.py", "compare_hparams.py",
                "train.py", "models/cnn.py", "models/__init__.py", "sample_pipeline.py",
                "input_pipeline.py", "source_data.py", "prepare.py", "evaluate.py",
                "meta_learning/common.py", "SETTINGS_MouseLFP.json",
                "utils/mouse_lfp.py", "utils/__init__.py", "requirements.txt", "Dockerfile"]


def source_hashes():
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def worker(args):
    metadata = json.loads((args.output / "comparison.json").read_text())
    if source_hashes() != metadata["source_sha256"]:
        raise ValueError("Project source changed after comparison setup")
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
    summary = {split: dict(samples=len(table), positives=int((table.label == 1).sum()),
                           negatives=int((table.label == 0).sum()),
                           positive_fraction=float((table.label == 1).mean()),
                           eligible_events=int(((selection.selection == "selected") &
                               selection.day.isin(DAY_SPLITS[split])).sum()))
               for split, table in tables.items()}
    write_json(work / "sampling_summary.json", summary)
    raw_train, y_train = sample_pipeline.materialize(
        dataset, args.subject, "train", tables["train"], args.sample_cache)
    raw_val, y_val = sample_pipeline.materialize(
        dataset, args.subject, "validation", tables["validation"], args.sample_cache)
    write_json(work / "signal_quality_audit.json", dict(
        train=signal_audit(raw_train), validation=signal_audit(raw_val)))
    if args.variant == "original":
        predictor = train.fit(raw_train, y_train, raw_val, y_val, work, device,
                              budget_seconds=args.budget_seconds, seed=args.seed)
    else:
        predictor = learnable_frontend.fit(raw_train, y_train, raw_val, y_val, work, device,
            budget_seconds=args.budget_seconds, seed=args.seed, learnable=args.variant == "learned")
    metrics, probabilities = prediction_metrics(y_val, predictor.predict_proba(raw_val))
    provenance = tables["validation"].copy()
    columns = ["mouse", "day", "raw_file", "start_row", "stop_row", "label"]
    metrics["evaluation_protocol_sha256"] = hashlib.sha256(
        provenance[columns].to_csv(index=False).encode()).hexdigest()
    if not np.array_equal(y_val, provenance.label.to_numpy()):
        raise ValueError("Validation labels differ from sample provenance")
    provenance["probability"] = probabilities
    provenance["predicted"] = (probabilities >= 0.5).astype(np.int64)
    provenance.to_csv(work / "predictions_validation.csv", index=False)
    history = json.loads((work / "history.json").read_text())
    checkpoint = torch.load(work / "checkpoint.pt", map_location="cpu", weights_only=True)
    metrics.update(epochs_run=len(history), best_epoch=int(checkpoint["epoch"]),
                   training_seconds=history[-1]["training_seconds"],
                   peak_vram_mb=(torch.cuda.max_memory_allocated(device) / 2**20
                                 if device.type == "cuda" else 0.0),
                   cnn_parameters=sum(p.numel() for p in (predictor.model if args.variant == "original"
                                                         else predictor.model.cnn).parameters()),
                   frontend_parameters=(0 if args.variant == "original" else
                                        sum(p.numel() for p in predictor.model.frontend.parameters())))
    if abs(metrics["accuracy"] - checkpoint["val_accuracy"]) > 1e-12:
        raise ValueError("Reloaded checkpoint does not reproduce its validation accuracy")
    if source_hashes() != metadata["source_sha256"]:
        raise ValueError("Project source changed during the trial")
    write_json(work / "validation_metrics.json", metrics)


def compare(args):
    verify_prepared(args.prepared_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    for name in SOURCE_FILES:
        path = args.output / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    if source_hashes() != hashes:
        raise ValueError("Project source changed while snapshotting")
    offsets = sample_pipeline.sample_offsets()[1]
    write_json(args.output / "comparison.json", dict(
        mode="finite_frontend_comparison", variants=VARIANTS, targets=args.targets,
        score="macro_validation_accuracy", threshold=0.5, seed=args.seed, device=args.device,
        budget_seconds_per_mouse=args.budget_seconds, max_epochs=train.MAX_EPOCHS,
        training_controls=dict(learning_rate=train.LEARNING_RATE, batch_size=train.BATCH_SIZE,
                               weight_decay=train.WEIGHT_DECAY, label_smoothing=train.LABEL_SMOOTHING,
                               dropout=0.5, checkpoint="validation_accuracy_then_lower_loss"),
        day_splits=DAY_SPLITS, cnn_input_shape=list(train.CNN_INPUT_SHAPE),
        source_sha256=hashes, sample_config=sample_pipeline.SAMPLE_CONFIG,
        sample_offsets_seconds=(offsets / 500).tolist(), original_preprocess=input_pipeline.PREPROCESS,
        fixed_spectrum=learnable_frontend.SPECTRUM_CONFIG, frontend=learnable_frontend.FRONTEND_CONFIG,
        normalization="Original unchanged; frozen/learned divide power by training mean power only",
        task_definition_changed=False, prepared_manifest_sha256=sha256_file(args.prepared_dir / "manifest.json"),
        runtime={name: importlib.metadata.version(name) for name in
                 ["torch", "numpy", "scipy", "stft", "scikit-learn", "pyarrow"]}))
    results = {variant: {} for variant in VARIANTS}
    started = time.monotonic()
    for subject in args.targets:
        for variant in VARIANTS:
            work = args.output / "trials" / variant / subject
            work.mkdir(parents=True)
            print(f"Starting {subject} {variant}", flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
                       "--prepared-dir", str(args.prepared_dir), "--data-dir", str(args.data_dir),
                       "--sample-cache", str(args.sample_cache), "--budget-seconds", str(args.budget_seconds),
                       "--seed", str(args.seed), "--device", args.device, "--subject", subject, "--variant", variant]
            with (work / "train.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=args.budget_seconds + 600)
            metrics = json.loads((work / "validation_metrics.json").read_text())
            if variant != "original":
                if metrics["evaluation_protocol_sha256"] != results["original"][subject]["evaluation_protocol_sha256"]:
                    raise ValueError("Comparison arms have different validation samples")
                if (work / "sampling_summary.json").read_text() != (
                        args.output / "trials" / "original" / subject / "sampling_summary.json").read_text():
                    raise ValueError("Comparison arms have different sample counts")
            results[variant][subject] = metrics
            write_json(args.output / "progress.json", results)
            print(f"Finished {subject} {variant}: accuracy={metrics['accuracy']:.9f} "
                  f"epochs={metrics['epochs_run']}", flush=True)
    aggregate = {variant: {metric: float(np.mean([row[metric] for row in mice.values()]))
                          for metric in ["accuracy", "auc", "sensitivity", "specificity", "loss"]}
                 for variant, mice in results.items()}
    protocols = {mouse: results["original"][mouse]["evaluation_protocol_sha256"] for mouse in args.targets}
    result = dict(aggregate=aggregate, subjects=results, full_cohort=args.targets == TARGETS,
                  learned_minus_original=aggregate["learned"]["accuracy"] - aggregate["original"]["accuracy"],
                  learned_minus_frozen=aggregate["learned"]["accuracy"] - aggregate["frozen"]["accuracy"],
                  evaluation_protocol_sha256=hashlib.sha256(json.dumps(protocols, sort_keys=True).encode()).hexdigest(),
                  wall_seconds=time.monotonic() - started, test_metrics_computed=False)
    write_json(args.output / "result.json", result)
    rows = []
    for mouse in args.targets:
        for variant in VARIANTS:
            summary = json.loads((args.output / "trials" / variant / mouse / "sampling_summary.json").read_text())
            rows.append(dict(mouse=mouse, variant=variant, **results[variant][mouse],
                train_samples=summary["train"]["samples"], validation_samples=summary["validation"]["samples"],
                test_defined_samples=summary["test"]["samples"], positive_fraction=summary["train"]["positive_fraction"],
                train_events=summary["train"]["eligible_events"], validation_events=summary["validation"]["eligible_events"],
                window_seconds=sample_pipeline.SAMPLE_CONFIG["window_seconds"],
                stride_seconds=sample_pipeline.SAMPLE_CONFIG["stride_seconds"]))
    with (args.output / "comparison.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(aggregate=aggregate, learned_minus_original=result["learned_minus_original"],
                          learned_minus_frozen=result["learned_minus_frozen"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/MouseLFP"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-cache", type=Path)
    parser.add_argument("--targets", type=parse_targets, default=TARGETS)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--budget-seconds", type=float, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subject", choices=TARGETS, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0 or args.seed < 0:
        parser.error("Budget must be finite and positive; seed must be nonnegative")
    if bool(args.subject) != bool(args.variant):
        parser.error("Worker requires both --subject and --variant")
    args.sample_cache = args.sample_cache or args.output / "sample-cache"
    worker(args) if args.subject else compare(args)


if __name__ == "__main__":
    main()
