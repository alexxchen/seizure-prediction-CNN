"""Finite training, exact checkpoint resumption, model selection, and inference."""

import importlib.metadata
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from meta_learning.frontend import FRONTEND_CONFIG, SPECTRUM_CONFIG
from meta_learning.adaptation import adapt_model, predict
from meta_learning.algorithms import meta_objective, pooled_objective
from meta_learning.common import digest, prediction_metrics, save_checkpoint, seed_everything, write_json
from meta_learning.config import Config
from meta_learning.episodes import chronological_episodes, episode_batch
from meta_learning.model import CNN_INPUT_SHAPE, MetaModel
from prepare import verify_prepared
from source_data import DAY_SPLITS, file_hash
import sample_pipeline

ROOT = Path(__file__).resolve().parent.parent


def source_hashes():
    paths = [ROOT / name for name in ["train.py", "evaluate.py", "prepare.py", "source_data.py",
             "sample_pipeline.py", "requirements.txt", "Dockerfile",
             "SETTINGS_MouseLFP.json", "run_meta.sh"]]
    paths += list((ROOT / "meta_learning").glob("*.py")) + list((ROOT / "models").glob("*.py"))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(paths)}


def runtime_identity(device):
    return dict(packages={name: importlib.metadata.version(name) for name in
                          ["torch", "numpy", "scipy", "pandas", "scikit-learn", "pyarrow"]},
                device=str(device), cuda=torch.version.cuda, cpu_threads=torch.get_num_threads(),
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def checkpoint(model, optimizer, step, rng, history, selection, identity, elapsed_seconds, peak_vram_mb):
    return dict(model=cpu_state(model), optimizer=optimizer.state_dict(), step=step,
                episode_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python_rng=random.getstate(), history=history, selection=selection, identity=identity,
                training_wall_seconds=elapsed_seconds, peak_vram_mb=peak_vram_mb)


def validate(initial, store, config, device, step, run_dir, identity):
    subjects, predictions, adapted = {}, {}, {}
    for position, mouse in enumerate(config.targets):
        training, validation = store.load(mouse, "train"), store.load(mouse, "validation")
        model, scale = adapt_model(initial, training, config, device, config.seed + 10000 + position)
        metrics, probabilities = prediction_metrics(validation.labels, predict(
            model, validation, scale, config.batch_size, device))
        metrics["evaluation_protocol_sha256"] = validation.fingerprint
        subjects[mouse] = metrics
        table = validation.table.copy()
        table["probability"], table["predicted"] = probabilities, (probabilities >= 0.5).astype(int)
        predictions[mouse] = table
        adapted[mouse] = dict(model=cpu_state(model), scale=scale, mouse=mouse,
                              training_fingerprint=training.fingerprint, validation_fingerprint=validation.fingerprint,
                              identity=identity, step=step, config=config.to_dict())
    aggregate = {key: float(np.mean([row[key] for row in subjects.values()]))
                 for key in ["accuracy", "auc", "sensitivity", "specificity", "loss"]}
    return dict(step=step, aggregate=aggregate, subjects=subjects), predictions, adapted


def train_run(config, store, run_dir, prepared_dir, device, resume=False):
    config.validate()
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run.json"
    if not resume and (metadata_path.exists() or (run_dir / "checkpoints").exists()):
        raise ValueError("Use a new run directory or --resume")
    if resume:
        previous = json.loads(metadata_path.read_text())["provenance"]
        if (previous["config"] != config.to_dict() or previous["source_sha256"] != source_hashes() or
                previous["runtime"] != runtime_identity(device)):
            raise ValueError("Resume requires identical config, sources, and runtime")
    training = {mouse: store.load(mouse, "train") for mouse in config.targets}
    episodes = {mouse: chronological_episodes(split, config.min_support_days, config.query_days)
                for mouse, split in training.items()}
    provenance = dict(config=config.to_dict(), source_sha256=source_hashes(),
                      source_inventory_sha256=digest(verify_prepared(prepared_dir)["signature"]),
                      training_fingerprints={mouse: split.fingerprint for mouse, split in training.items()},
                      runtime=runtime_identity(device), sample_config=sample_pipeline.SAMPLE_CONFIG,
                      offsets_seconds=(sample_pipeline.sample_offsets()[1] / 500).tolist(),
                      spectrum_config=SPECTRUM_CONFIG, frontend_config=FRONTEND_CONFIG,
                      day_splits=DAY_SPLITS, cnn_input_shape=list(CNN_INPUT_SHAPE),
                      normalization="Support-only mean power per episode; Days 0-7 mean power for adaptation",
                      batchnorm="Fixed running statistics; trainable affine weights; no query-batch dependence",
                      inner_updates="CNN only", outer_updates="Shared frontend and CNN initialization",
                      meta_gradient="Full differentiation through inner SGD steps")
    identity = digest(provenance)
    if resume:
        if not metadata_path.exists() or json.loads(metadata_path.read_text())["identity"] != identity:
            raise ValueError("Resume requires identical config, sources, data, and runtime")
    else:
        if metadata_path.exists() or (run_dir / "checkpoints").exists():
            raise ValueError("Use a new run directory or --resume")
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
            git_commit = commit.stdout.strip() if commit.returncode == 0 else "unknown"
        except FileNotFoundError:
            git_commit = "unknown"
        write_json(metadata_path, dict(identity=identity, provenance=provenance, git_commit=git_commit))
        write_json(run_dir / "config.json", config.to_dict())
        for name in provenance["source_sha256"]:
            destination = run_dir / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        if source_hashes() != provenance["source_sha256"]:
            raise ValueError("Source files changed while snapshotting")
    write_json(run_dir / "episodes.json", {mouse: [task.describe() for task in tasks]
                                         for mouse, tasks in episodes.items()})
    seed_everything(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = MetaModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.outer_lr)
    rng = np.random.default_rng(config.seed)
    start_step, history, selection, elapsed_seconds, previous_peak = 0, [], None, 0.0, 0.0
    if resume:
        saved = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Checkpoint belongs to another run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step, history, selection = saved["step"], saved["history"], saved["selection"]
        elapsed_seconds, previous_peak = saved["training_wall_seconds"], saved["peak_vram_mb"]
        rng.bit_generator.state = saved["episode_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        if selection:
            write_json(run_dir / "selection.json", selection)
        if start_step == config.outer_steps:
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get("status") != "complete":
                write_json(status_path, dict(status="complete", step=start_step, selection=selection,
                    training_wall_seconds=elapsed_seconds, peak_vram_mb=previous_peak))
            return selection
    started = time.monotonic()
    def peak_memory():
        current = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        return max(previous_peak, current)
    try:
        for step in range(start_step + 1, config.outer_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(config.tasks_per_step):
                # Equal mouse weights match the macro validation objective.
                mouse = config.targets[int(rng.integers(len(config.targets)))]
                episode = episodes[mouse][int(rng.integers(len(episodes[mouse])))]
                support, query = episode_batch(training[mouse], episode, config, rng, device)
                objective = meta_objective if config.algorithm == "meta" else pooled_objective
                loss = objective(model, support, query, config)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite outer loss")
                (loss / config.tasks_per_step).backward()
                losses.append(float(loss.detach()))
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            row = dict(step=step, outer_loss=float(np.mean(losses)), gradient_norm=float(grad_norm))
            improved = False
            if step % config.eval_every == 0 or step == config.outer_steps:
                report, predictions, adapted = validate(model, store, config, device, step, run_dir, identity)
                row["validation"] = report["aggregate"]
                score = report["aggregate"]
                previous = selection["aggregate"] if selection else None
                improved = previous is None or score["accuracy"] > previous["accuracy"] or (
                    score["accuracy"] == previous["accuracy"] and score["loss"] < previous["loss"])
                if improved:
                    directory = run_dir / "selected" / f"step-{step:06d}"
                    models = {}
                    for mouse, state in adapted.items():
                        path = directory / mouse / "checkpoint.pt"
                        save_checkpoint(path, state)
                        models[mouse] = dict(path=str(path.relative_to(run_dir)), sha256=file_hash(path))
                        predictions[mouse].to_csv(path.parent / "predictions_validation.csv", index=False)
                        write_json(path.parent / "validation_metrics.json", report["subjects"][mouse])
                    selection = dict(report, identity=identity, adapted_models=models)
                    save_checkpoint(directory / "meta.pt", dict(model=cpu_state(model), identity=identity, step=step))
                write_json(run_dir / "validation" / f"step-{step:06d}.json", report)
            history.append(row)
            if step % config.checkpoint_every == 0 or "validation" in row:
                save_checkpoint(run_dir / "checkpoints" / "last.pt",
                                checkpoint(model, optimizer, step, rng, history, selection, identity,
                                           elapsed_seconds + time.monotonic() - started, peak_memory()))
            if improved:
                write_json(run_dir / "selection.json", selection)
            write_json(run_dir / "history.json", history)
            write_json(run_dir / "status.json", dict(status="training", step=step, target_steps=config.outer_steps))
            if step == 1 or step % 10 == 0 or "validation" in row:
                print(json.dumps(row), flush=True)
        write_json(run_dir / "status.json", dict(status="complete", step=config.outer_steps,
                   training_wall_seconds=elapsed_seconds + time.monotonic() - started,
                   peak_vram_mb=peak_memory(), selection=selection))
    except BaseException as error:
        write_json(run_dir / "status.json", dict(status="interrupted", error=type(error).__name__, message=str(error)))
        raise
    return selection


def evaluate_run(store, run_dir, device, split="validation"):
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test")
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "run.json").read_text())
    if metadata["provenance"]["source_sha256"] != source_hashes():
        raise ValueError("Evaluate with the run's saved source version")
    if metadata["provenance"]["source_inventory_sha256"] != digest(store.dataset.signature):
        raise ValueError("Evaluation sources differ from training sources")
    selection = json.loads((run_dir / "selection.json").read_text())
    if selection["identity"] != metadata["identity"]:
        raise ValueError("Selection belongs to another run")
    config = Config.from_dict(metadata["provenance"]["config"])
    seed_everything(config.seed)
    subjects = {}
    for mouse in config.targets:
        record = selection["adapted_models"][mouse]
        path = run_dir / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError("Selected checkpoint changed")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["identity"] != metadata["identity"] or state["mouse"] != mouse:
            raise ValueError("Selected checkpoint identity mismatch")
        model = MetaModel().to(device)
        model.load_state_dict(state["model"])
        values = store.load(mouse, split)
        if split == "validation" and values.fingerprint != state["validation_fingerprint"]:
            raise ValueError("Validation sample provenance changed")
        metrics, probabilities = prediction_metrics(values.labels, predict(model, values, state["scale"], config.batch_size, device))
        metrics["evaluation_protocol_sha256"] = values.fingerprint
        subjects[mouse] = metrics
        destination = run_dir / "evaluation" / split / mouse
        destination.mkdir(parents=True, exist_ok=True)
        table = values.table.copy()
        table["probability"], table["predicted"] = probabilities, (probabilities >= 0.5).astype(int)
        table.to_csv(destination / "predictions.csv", index=False)
        write_json(destination / "metrics.json", metrics)
    report = dict(split=split, selected_step=selection["step"], identity=metadata["identity"],
                  aggregate={key: float(np.mean([row[key] for row in subjects.values()]))
                             for key in ["accuracy", "auc", "sensitivity", "specificity", "loss"]},
                  subjects=subjects, evaluation_protocol_sha256=digest(
                      {mouse: row["evaluation_protocol_sha256"] for mouse, row in subjects.items()}))
    write_json(run_dir / "evaluation" / split / "result.json", report)
    return report
