"""Fixed evaluator and experiment ledger for agent-driven MouseLFP research."""

import argparse
import csv
import fcntl
import importlib.util
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, log_loss, roc_auc_score)

from prepare import (ROOT, TARGETS, frozen_hashes, load_dataset, parse_targets,
                     sha256_file, verify_prepared)
from research_protocol import write_json
from source_data import DAY_SPLITS

# Wall-clock watchdog includes import/preprocessing/evaluation overhead.
STARTUP_ALLOWANCE_SECONDS = 180
MIN_IMPROVEMENT = 1e-6
LEDGER_COLUMNS = ["experiment", "commit", "source_sha256", "validation_accuracy",
                  "peak_vram_mb", "wall_seconds", "status", "description"]


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Research supports cpu or cuda devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def source_files(root):
    root = Path(root)
    return [root / "train.py", root / "sample_pipeline.py", root / "input_pipeline.py",
            root / "requirements.txt", root / "Dockerfile",
            *sorted((root / "models").rglob("*.py"))]


def source_hashes(root):
    return {str(p.relative_to(root)): sha256_file(p) for p in source_files(root)}


def source_digest(hashes):
    import hashlib
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def git_commit():
    commit = os.environ.get("AUTORESEARCH_COMMIT")
    if commit:
        return commit
    if shutil.which("git") is None:
        return "unknown"
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def import_recipe(source):
    # Preparation imports the legacy CNN. Resolve models against the saved recipe.
    for name in list(sys.modules):
        if name in {"input_pipeline", "sample_pipeline", "models"} or name.startswith("models."):
            del sys.modules[name]
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("research_recipe", source / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_sample_table(rows, dataset, target, split):
    required = ["mouse", "day", "raw_file", "start_row", "stop_row", "label"]
    if not set(required).issubset(rows.columns) or rows[required].isna().any().any() or not len(rows):
        raise ValueError("Missing candidate sample provenance")
    if not rows.mouse.eq(target).all() or not rows.day.isin(DAY_SPLITS[split]).all():
        raise ValueError("Candidate samples crossed the fixed mouse/day partition")
    for column in ["day", "start_row", "stop_row", "label"]:
        values = rows[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Candidate sample indices and labels must be integers")
    if set(rows.label.unique()) != {0, 1} or rows.duplicated(["raw_file", "start_row", "stop_row"]).any():
        raise ValueError("Both classes and unique source windows are required")
    for row in rows.itertuples():
        source = dataset.sources.get(row.raw_file)
        if (source is None or source["mouse"] != target or source["day"] != int(row.day)
                or not 0 <= row.start_row < row.stop_row <= source["rows"]):
            raise ValueError("Candidate sample provenance does not match the raw inventory")
    return rows


def prediction_metrics(labels, probabilities):
    probabilities = np.asarray(probabilities)
    if probabilities.shape == (len(labels), 2):
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("Predictor must return normalized finite class probabilities")
        probabilities = probabilities[:, 1]
    if (probabilities.shape != (len(labels),) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()):
        raise ValueError("Predictor must return one probability per original window")
    predicted = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    result = dict(samples=len(labels), auc=float(roc_auc_score(labels, probabilities)),
                  average_precision=float(average_precision_score(labels, probabilities)),
                  loss=float(log_loss(labels, probabilities, labels=[0, 1])),
                  accuracy=float(accuracy_score(labels, predicted)),
                  sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                  confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]])
    return result, probabilities


def worker(args):
    source = args.experiment_dir / "source"
    metadata = json.loads((args.experiment_dir / "experiment.json").read_text())
    if source_hashes(source) != metadata["source_hashes"]:
        raise ValueError("Experiment source snapshot changed")
    seed_everything(args.seed)
    recipe = import_recipe(source)
    sampler = importlib.import_module("sample_pipeline")
    dataset = load_dataset(args.prepared_dir, args.data_dir)
    tables, selection = sampler.build_tables(dataset, args.subject)
    if set(tables) != set(DAY_SPLITS):
        raise ValueError("A shared candidate recipe must construct all three day splits")
    for part, table in tables.items():
        validate_sample_table(table, dataset, args.subject, part)
    lengths = {int(stop - start) for table in tables.values()
               for start, stop in zip(table.start_row, table.stop_row)}
    if len(lengths) != 1:
        raise ValueError("Train, validation, and test must share the same window size")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    work_dir = args.experiment_dir / args.subject
    work_dir.mkdir(exist_ok=True)
    cache = args.experiment_dir.parent.parent / "sample-cache"
    if not args.final:
        write_json(work_dir / "sample_config.json", sampler.SAMPLE_CONFIG)
        selection.to_csv(work_dir / "event_selection.csv", index=False)
    summary = {part: dict(samples=len(table), positive_samples=int((table.label == 1).sum()),
                         negative_samples=int((table.label == 0).sum()),
                         window_rows=sorted((table.stop_row - table.start_row).unique().tolist()))
               for part, table in tables.items()}
    if not args.final:
        write_json(work_dir / "sampling_summary.json", summary)
    if args.final:
        predictor = recipe.load_predictor(work_dir, device)
        split = "test"
    else:
        train_raw, train_y = sampler.materialize(dataset, args.subject, "train", tables["train"], cache)
        val_raw, val_y = sampler.materialize(dataset, args.subject, "validation", tables["validation"], cache)
        predictor = recipe.fit(train_raw, train_y, val_raw, val_y,
                               work_dir=work_dir, device=device,
                               budget_seconds=args.budget_seconds, seed=args.seed)
        split = "validation"
    raw, labels = sampler.materialize(dataset, args.subject, split, tables[split], cache)
    result, probabilities = prediction_metrics(labels, predictor.predict_proba(raw))
    provenance = tables[split].copy()
    if not np.array_equal(provenance.label.to_numpy(), labels):
        raise ValueError("Prediction labels differ from fixed window provenance")
    provenance["probability"] = probabilities
    provenance["predicted"] = (probabilities >= 0.5).astype(np.int64)
    provenance.to_csv(work_dir / f"predictions_{split}.csv", index=False)
    result["peak_vram_mb"] = (torch.cuda.max_memory_allocated(device) / 2 ** 20
                               if device.type == "cuda" else 0.0)
    result["sample_config"] = sampler.SAMPLE_CONFIG
    columns = ["mouse", "day", "raw_file", "start_row", "stop_row", "label"]
    result["evaluation_protocol_sha256"] = hashlib.sha256(
        provenance[columns].to_csv(index=False).encode()).hexdigest()
    write_json(work_dir / f"{split}_metrics.json", result)


@contextmanager
def session_lock(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".session.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another experiment is using this research session") from None
        yield


def session_config(args, manifest):
    import importlib.metadata
    missing = set(args.targets) - set(manifest["recordings"])
    if missing:
        raise ValueError(f"Prepare these subjects first: {sorted(missing)}")
    return dict(version=4, prepared_manifest_sha256=sha256_file(args.prepared_dir / "manifest.json"),
                targets=args.targets, budget_seconds=args.budget_seconds, seed=args.seed,
                device=str(resolve_device(args.device)), score="macro_validation_accuracy",
                threshold=0.5,
                research_scope="source_data_and_input", cnn_input_shape=[1, 1, 55, 114],
                min_improvement=MIN_IMPROVEMENT, frozen_code=frozen_hashes(),
                runtime={name: importlib.metadata.version(name)
                         for name in ["torch", "numpy", "scipy", "stft", "scikit-learn"]},
                scope="full-cohort" if args.targets == TARGETS else "smoke")


def ensure_session(args, manifest):
    config = session_config(args, manifest)
    path = args.run_dir / "session.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("Research session data/targets/budget/seed/device/scoring changed; use a new --run-dir")
    else:
        write_json(path, config)
    return config


def append_ledger(run_dir, row):
    path = run_dir / "results.tsv"
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS, delimiter="\t")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run_subject(args, experiment_dir, target, final=False):
    command = [sys.executable, str(ROOT / "evaluate.py"), "_worker",
               "--prepared-dir", str(args.prepared_dir),
               "--experiment-dir", str(experiment_dir), "--subject", target,
               "--device", args.device, "--seed", str(args.seed),
               "--budget-seconds", str(args.budget_seconds)]
    if args.data_dir is not None:
        command.extend(["--data-dir", str(args.data_dir)])
    if final:
        command.append("--final")
    log_path = experiment_dir / target / ("test.log" if final else "train.log")
    log_path.parent.mkdir(exist_ok=True)
    started = time.monotonic()
    with log_path.open("w") as log:
        try:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           timeout=args.budget_seconds + STARTUP_ALLOWANCE_SECONDS, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            tail = log_path.read_text(errors="replace").splitlines()[-20:]
            raise RuntimeError(f"{target} failed ({error}); see {log_path}\n" + "\n".join(tail)) from error
    split = "test" if final else "validation"
    result = json.loads((experiment_dir / target / f"{split}_metrics.json").read_text())
    if not math.isfinite(result["accuracy"]):
        raise ValueError("Non-finite score")
    result["wall_seconds"] = time.monotonic() - started
    if not final:
        # Preserve everything written by fit that might be needed for inference.
        subject_dir = experiment_dir / target
        ignored = {"train.log", "validation_metrics.json", "predictions_validation.csv"}
        result["artifacts"] = {str(p.relative_to(subject_dir)): sha256_file(p)
                               for p in sorted(subject_dir.rglob("*"))
                               if p.is_file() and p.name not in ignored}
        if not result["artifacts"]:
            raise ValueError("Recipe must save artifacts for load_predictor and final evaluation")
    print(f"{target}: {split}_accuracy={result['accuracy']:.6f} "
          f"{split}_auc={result['auc']:.6f} wall_seconds={result['wall_seconds']:.1f}", flush=True)
    return result


def experiment(args):
    with session_lock(args.run_dir):
        manifest = verify_prepared(args.prepared_dir, splits=("train", "validation"))
        session = ensure_session(args, manifest)
        if (args.run_dir / "final.json").exists() or (args.run_dir / "final_started.json").exists():
            raise ValueError("This session has entered final test evaluation; start a new session for research")
        experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        experiment_dir = args.run_dir / "experiments" / experiment_id
        source = experiment_dir / "source"
        source.mkdir(parents=True)
        hashes = source_hashes(ROOT)
        for relative in hashes:
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        if source_hashes(source) != hashes:
            raise ValueError("Research code changed while snapshotting")
        commit = git_commit()
        metadata = dict(experiment=experiment_id, description=args.description, commit=commit,
                        source_hashes=hashes, session=session)
        write_json(experiment_dir / "experiment.json", metadata)
        write_json(experiment_dir / "status.json", dict(status="running"))
        row = dict(experiment=experiment_id, commit=commit, source_sha256=source_digest(hashes),
                   validation_accuracy="", peak_vram_mb=0, wall_seconds=0,
                   status="crash", description=args.description)
        started = time.monotonic()
        try:
            results = {target: run_subject(args, experiment_dir, target) for target in args.targets}
            verify_prepared(args.prepared_dir, splits=("train", "validation"))
            if frozen_hashes() != session["frozen_code"]:
                raise ValueError("Fixed protocol changed during the experiment")
            score = float(np.mean([r["accuracy"] for r in results.values()]))
            best_path = args.run_dir / "best.json"
            best = json.loads(best_path.read_text()) if best_path.exists() else None
            status = "keep" if best is None or score > best["validation_accuracy"] + MIN_IMPROVEMENT else "discard"
            result = dict(experiment=experiment_id, validation_accuracy=score,
                          validation_auc=float(np.mean([r["auc"] for r in results.values()])),
                          threshold=0.5, status=status,
                          scope=session["scope"], subjects=results, source_sha256=source_digest(hashes))
            result["evaluation_protocol_sha256"] = hashlib.sha256(json.dumps(
                {t: r["evaluation_protocol_sha256"] for t, r in results.items()},
                sort_keys=True).encode()).hexdigest()
            write_json(experiment_dir / "result.json", result)
            if status == "keep":
                write_json(best_path, result)
            row.update(validation_accuracy=f"{score:.9f}", status=status,
                       peak_vram_mb=max(r["peak_vram_mb"] for r in results.values()))
            print(f"validation_accuracy: {score:.9f}\nstatus: {status}\nexperiment: {experiment_id}", flush=True)
        except BaseException as error:
            write_json(experiment_dir / "status.json", dict(status="crash", error=str(error)))
            raise
        finally:
            row["wall_seconds"] = round(time.monotonic() - started, 3)
            append_ledger(args.run_dir, row)
        write_json(experiment_dir / "status.json", dict(status=status))


def finalize(args):
    with session_lock(args.run_dir):
        session = json.loads((args.run_dir / "session.json").read_text())
        manifest = verify_prepared(args.prepared_dir)
        # Read the budget/subjects/seed from the selected research session.
        args.targets, args.seed = session["targets"], session["seed"]
        args.budget_seconds, args.device = session["budget_seconds"], session["device"]
        if session_config(args, manifest) != session:
            raise ValueError("Final test data or protocol differs from the research session")
        best = json.loads((args.run_dir / "best.json").read_text())
        experiment_dir = args.run_dir / "experiments" / best["experiment"]
        for target, result in best["subjects"].items():
            for relative, expected in result["artifacts"].items():
                if sha256_file(experiment_dir / target / relative) != expected:
                    raise ValueError(f"Selected checkpoint/artifact changed: {target}/{relative}")
        destination = args.run_dir / "final.json"
        if destination.exists():
            print(destination.read_text(), end="")
            return
        write_json(args.run_dir / "final_started.json", dict(experiment=best["experiment"]))
        results = {}
        for target in args.targets:
            path = experiment_dir / target / "test_metrics.json"
            # Interrupted final evaluations resume without retraining or model selection.
            results[target] = json.loads(path.read_text()) if path.exists() else run_subject(args, experiment_dir, target, final=True)
        verify_prepared(args.prepared_dir)
        result = dict(experiment=best["experiment"], validation_accuracy=best["validation_accuracy"],
                      test_accuracy=float(np.mean([r["accuracy"] for r in results.values()])),
                      threshold=0.5, validation_auc=best["validation_auc"],
                      test_auc=float(np.mean([r["auc"] for r in results.values()])),
                      scope=session["scope"], subjects=results)
        write_json(destination, result)
        print(json.dumps(result, indent=2), flush=True)


def restore(args):
    """Recover the best recipe, keeping a backup of the current editable files."""
    with session_lock(args.run_dir):
        frozen_hashes()
        session = json.loads((args.run_dir / "session.json").read_text())
        if (session.get("score") != "macro_validation_accuracy" or session.get("threshold") != 0.5
                or session.get("research_scope") != "source_data_and_input"):
            raise ValueError("Archived session has a different research scope; restore from a data-pipeline session")
        best = json.loads((args.run_dir / "best.json").read_text())
        experiment_dir = args.run_dir / "experiments" / best["experiment"]
        metadata = json.loads((experiment_dir / "experiment.json").read_text())
        source = experiment_dir / "source"
        if source_hashes(source) != metadata["source_hashes"]:
            raise ValueError("Best recipe snapshot changed")
        backup = args.run_dir / "restore-backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup.mkdir(parents=True)
        for name in ["sample_pipeline.py", "input_pipeline.py"]:
            shutil.copy2(ROOT / name, backup / name)
            shutil.copy2(source / name, ROOT / name)
        print(f"Restored {best['experiment']}; previous recipe backed up at {backup}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["experiment", "finalize", "restore", "_worker"])
    parser.add_argument("--prepared-dir", type=Path, default=Path("autoresearch-data"))
    parser.add_argument("--data-dir", type=Path, help="Private continuous recordings and source labels")
    parser.add_argument("--run-dir", type=Path, default=Path("autoresearch-runs/default"))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budget-seconds", type=float, default=300,
                        help="Fixed training budget per mouse; keep constant within a session")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--description", default="baseline")
    parser.add_argument("--experiment-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--subject", choices=TARGETS, help=argparse.SUPPRESS)
    parser.add_argument("--final", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.targets = parse_targets(args.targets)
    args.prepared_dir, args.run_dir = args.prepared_dir.resolve(), args.run_dir.resolve()
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0 or args.seed < 0:
        parser.error("Budget must be finite and positive; seed must be nonnegative")
    if args.action == "_worker" and (args.subject is None or args.experiment_dir is None):
        parser.error("Worker requires an experiment snapshot and subject")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    {"experiment": experiment, "finalize": finalize, "restore": restore, "_worker": worker}[arguments.action](arguments)
