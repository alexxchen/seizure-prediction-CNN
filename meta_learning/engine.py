"""Finite training, exact checkpoint resumption, model selection, and inference."""

import copy
import contextlib
import importlib.metadata
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from meta_learning.frontend import FILTERBANK_CONFIG, FRONTEND_CONFIG, LogSTFTFrontend, WaveformFrontend
from meta_learning.waveform import WAVEFORM_CONFIG
from meta_learning.adaptation import adapt_model, predict
from meta_learning.algorithms import meta_objective, pooled_objective
from meta_learning.common import digest, prediction_metrics, save_checkpoint, seed_everything, write_json
from meta_learning.config import Config
from meta_learning.episodes import crops_support, split_episode, episode_batch
from meta_learning.model import MetaModel
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


@contextlib.contextmanager
def inner_shuffle_rng(seed, device):
    """Seed inner shuffles for one task without changing the caller's RNG streams."""
    if seed is None:
        yield
        return
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def fit_progress(stage, **context):
    return lambda row: print(json.dumps(dict(stage=stage, **context, **row)), flush=True)


def validation_roles(config):
    """Query days supply outer gradients; disjoint selection days (if any) pick checkpoints."""
    roles = [("query", "validation", config.query_validation_days)]
    if config.selection_validation_days is not None:
        roles.append(("selection", "validation", config.selection_validation_days))
    return roles + [("eval", "test", None)]


def selection_criterion(config):
    if config.selection_validation_days is None:
        return "Minimum mean query log loss; test metrics are reporting only"
    return (f"Minimum mean log loss on validation Days {list(config.selection_validation_days)}, disjoint from "
            f"query Days {list(config.query_validation_days)}; test metrics are reporting only")


def report_roles(report):
    """Role -> report; the root is the selection report (the query report without selection days)."""
    roles = {report["role"]: report, "eval": report["eval"]}
    if "query" in report:
        roles["query"] = report["query"]
    return roles


def validate(initial, store, config, device, step, run_dir, identity, *, fit_stage="per_mouse_fit"):
    """Fit on support only; score query, selection, and held-out eval with the same classifier.

    The root aggregate/subjects select checkpoints: selection days when configured,
    otherwise the query days. The nested eval report is test-only inference.
    """
    roles = validation_roles(config)
    subjects = {role: {} for role, _, _ in roles}
    predictions, adapted = {}, {}
    for position, mouse in enumerate(config.targets):
        # Per-mouse fits share the meta inner fits' support policy, including support crops.
        training = store.load_crops(mouse, "train") if crops_support(config) else store.load(mouse, "train")
        model, scale, fit = adapt_model(initial, training, config, device, config.seed + 10000 + position,
                                        return_diagnostics=True,
                                        progress=fit_progress(fit_stage, step=step, mouse=mouse))
        predictions[mouse], fingerprints = {}, {}
        for role, part, days in roles:
            values = store.load(mouse, part)
            if (values.split != part or values.mouse != mouse or
                    not values.table.day.isin(DAY_SPLITS[part]).all()):
                raise ValueError(f"Invalid {part} partition for {mouse}")
            # Full-split fingerprints let the separate evaluation command replay checkpoints.
            fingerprints[f"{part}_fingerprint"] = values.fingerprint
            if days is not None:
                values = values.subset_days(days)
            metrics, probabilities = prediction_metrics(values.labels, predict(
                model, values, scale, config.batch_size, device))
            metrics.update(evaluation_protocol_sha256=values.fingerprint, classifier_fit=fit,
                           days=sorted(int(day) for day in values.table.day.unique()))
            subjects[role][mouse] = metrics
            fingerprints[f"{role}_fingerprint"] = values.fingerprint
            table = values.table.copy()
            table["probability"], table["predicted"] = probabilities, (probabilities >= 0.5).astype(int)
            predictions[mouse][role] = table
            fit_progress(f"{fit_stage}_{role}", step=step, mouse=mouse)(
                {key: metrics[key] for key in ["samples", "loss", "accuracy", "auc", "sensitivity", "specificity"]})
        adapted[mouse] = dict(model=cpu_state(model), scale=scale, mouse=mouse,
                              training_fingerprint=training.fingerprint, **fingerprints,
                              identity=identity, step=step, config=config.to_dict(), classifier_fit=fit)
    reports = {}
    for role, part, days in roles:
        aggregate = {key: float(np.mean([row[key] for row in subjects[role].values()]))
                     for key in ["accuracy", "auc", "sensitivity", "specificity", "loss"]}
        reports[role] = dict(step=step, split=part, role=role, days=list(days or DAY_SPLITS[part]),
                             aggregate=aggregate, subjects=subjects[role])
    if "selection" in reports:
        return dict(reports["selection"], query=reports["query"], eval=reports["eval"]), predictions, adapted
    return dict(reports["query"], eval=reports["eval"]), predictions, adapted


def save_predictions(directory, predictions, metrics):
    for role, table in predictions.items():
        table.to_csv(directory / f"predictions_{role}.csv", index=False)
        write_json(directory / f"{role}_metrics.json", metrics[role])


def mouse_metrics(report, mouse):
    return {role: value["subjects"][mouse] for role, value in report_roles(report).items()}


def query_improves(score, previous):
    """Checkpoint selection uses the mean log loss of the selection report."""
    return previous is None or score["loss"] < previous["loss"]


def log_stft_baseline(initial, store, config, device, run_dir, identity):
    """Fit the standalone log_stft algorithm; reuse its completed report on resume."""
    directory = Path(run_dir) / "baseline" / "log_stft"
    report_path = directory / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["identity"] != identity or set(report["subjects"]) != set(config.targets):
            raise ValueError("Pure log-STFT baseline belongs to another run")
        for record in report["adapted_models"].values():
            if file_hash(Path(run_dir) / record["path"]) != record["sha256"]:
                raise ValueError("Pure log-STFT baseline checkpoint changed")
        reused = True
    else:
        baseline = copy.deepcopy(initial)
        baseline.use_frontend(LogSTFTFrontend().to(device), config.seed)
        started = time.monotonic()
        # Per-mouse fitting is reproducible and does not change caller RNG state.
        with torch.random.fork_rng():
            report, predictions, adapted = validate(baseline, store, config, device, 0,
                run_dir, identity, fit_stage="baseline_log_stft_fit")
        records = {}
        for mouse, state in adapted.items():
            path = directory / mouse / "checkpoint.pt"
            state["frontend_type"] = "pure_log_stft"
            save_checkpoint(path, state)
            records[mouse] = dict(path=str(path.relative_to(run_dir)), sha256=file_hash(path))
            save_predictions(path.parent, predictions[mouse], mouse_metrics(report, mouse))
        report.update(identity=identity, frontend_type="pure_log_stft", frontend_parameters=0,
                      description="Frozen cosine-window DFT filterbank log1p magnitude, average-pooled to 40 band channels x 56 time bins for the 1-D CNN; no learned stages; "
                                  "fresh per-mouse CNN trained on Days 0-7; " +
                                  ", ".join(f"{role} Days {value['days']}" for role, value in report_roles(report).items()),
                      adapted_models=records, wall_seconds=time.monotonic() - started)
        # Publish the completion marker only after all per-mouse artifacts exist.
        write_json(report_path, report)
        reused = False
    keys = ["samples", "loss", "accuracy", "auc", "sensitivity", "specificity"]
    for mouse in report["subjects"]:
        print(json.dumps(dict(stage="baseline_log_stft_mouse", step=0, mouse=mouse, reused=reused,
                              **{role: {key: metrics[key] for key in keys}
                                 for role, metrics in mouse_metrics(report, mouse).items()})), flush=True)
    print(json.dumps(dict(stage="baseline_log_stft", step=0, reused=reused,
                          **{role: value["aggregate"] for role, value in report_roles(report).items()})), flush=True)
    return report


def run_log_stft(model, store, config, device, run_dir, identity):
    """One support-only CNN fit per mouse, without an outer optimizer or loop."""
    status_path = run_dir / "status.json"
    previous_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    try:
        if previous_status.get("status") != "complete":
            write_json(status_path, dict(status="log_stft_fit", step=0, target_steps=0))
        report = log_stft_baseline(model, store, config, device, run_dir, identity)
        selection = dict(report, criterion="Single fixed log-STFT fit; no outer updates or checkpoint search")
        selection_path = run_dir / "selection.json"
        if previous_status.get("status") == "complete" and selection_path.exists():
            if json.loads(selection_path.read_text()) != selection:
                raise ValueError("Standalone baseline selection changed")
            return selection
        write_json(run_dir / "selection.json", selection)
        write_json(run_dir / "history.json", [dict(step=0, algorithm="log_stft",
                   **{("query_refit" if role == "query" else role): value["aggregate"]
                      for role, value in report_roles(report).items()})])
        peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        peak = max(peak, previous_status.get("peak_vram_mb", 0.0))
        write_json(status_path, dict(status="complete", step=0, target_steps=0,
                   training_wall_seconds=report["wall_seconds"], peak_vram_mb=peak, selection=selection))
        return selection
    except BaseException as error:
        write_json(run_dir / "status.json", dict(status="interrupted", error=type(error).__name__, message=str(error)))
        raise


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
    load = store.load_crops if config.augment_random_crop and config.algorithm != "log_stft" else store.load
    training = {mouse: load(mouse, "train") for mouse in config.targets}
    # Only the configured query days supply outer gradients; selection days stay held out.
    queries = {mouse: load(mouse, "validation").subset_days(config.query_validation_days)
               for mouse in config.targets}
    episodes = {mouse: split_episode(training[mouse], queries[mouse]) for mouse in config.targets}
    provenance = dict(config=config.to_dict(), source_sha256=source_hashes(),
                      source_inventory_sha256=digest(verify_prepared(prepared_dir)["signature"]),
                      training_fingerprints={mouse: split.fingerprint for mouse, split in training.items()},
                      query_fingerprints={mouse: split.fingerprint for mouse, split in queries.items()},
                      split_roles=dict(support="train Days 0-7",
                                       query=f"validation Days {list(config.query_validation_days)}",
                                       selection=(None if config.selection_validation_days is None else
                                                  f"validation Days {list(config.selection_validation_days)}"),
                                       eval="test Days 12/13"),
                      selection_criterion=("Single fixed log-STFT fit; no checkpoint search"
                          if config.algorithm == "log_stft" else selection_criterion(config)),
                      outer_optimizer=(f"AdamW lr {config.outer_lr}; decoupled weight decay {config.outer_weight_decay} "
                                       "on frontend weight matrices; frontend biases, GroupNorm, and any auxiliary CNN undecayed"),
                      runtime=runtime_identity(device), sample_config=sample_pipeline.SAMPLE_CONFIG,
                      offsets_seconds=(sample_pipeline.sample_offsets()[1] / 500).tolist(),
                      waveform_config=WAVEFORM_CONFIG, filterbank_config=FILTERBANK_CONFIG,
                      frontend_config=(dict(architecture="pure_log_stft", input_transform="frozen DFT filterbank, log1p_base10 magnitude",
                          temporal_pooling="adaptive_average", time_bins=LogSTFTFrontend.output_shape[1],
                          frequency_bins=LogSTFTFrontend.output_shape[0], parameters=0)
                          if config.algorithm == "log_stft" else FRONTEND_CONFIG),
                      baseline_policy="Standalone algorithm=log_stft only; no automatic startup baseline",
                      day_splits=DAY_SPLITS, cnn_input_shape=list((LogSTFTFrontend if config.algorithm == "log_stft" else WaveformFrontend).output_shape),
                      cnn="1-D CNN: three Conv1d/AxisBatchNorm/max-pool stages (16, 32, 64), flatten, dense 128, 2 logits",
                      task_schedule=dict(mice=("Every configured mouse fitted once; no outer loop"
                                             if config.algorithm == "log_stft" else
                                             "Every configured mouse exactly once per outer step"),
                                         mouse_order=list(config.targets),
                                         tasks_per_step=0 if config.algorithm == "log_stft" else len(config.targets),
                                         episodes="One fixed train-support/validation-query task per mouse", gradient_reduction="Mean per mouse"),
                      normalization="Identity (scale=1) before the neural frontend; no fitted statistics",
                      batchnorm=("Support batch statistics during training; query/test use exact support statistics of the final "
                                 "weights, recalibrated differentiably after each fit" if config.batchnorm_recalibration else
                                 "Support batch statistics and per-task running EMA; query/test use frozen support EMA; full EMA hypergradient"),
                      augmentation=(dict(crop_config=sample_pipeline.CROP_CONFIG,
                                         scope=config.random_crop_scope, query_crops=config.query_crops,
                                         support_crops=config.support_crops,
                                         policy="Meta/pooled training batches only (support, query, or both per scope): "
                                                "fresh uniformly placed 112 s crops of each row's context segment every "
                                                "outer step from the episode RNG; query_crops independent crops per query "
                                                "window, loss averaged; fixed windows for per-mouse fitting, selection, and eval")
                                    if config.augment_random_crop else "None"),
                      inner_shuffles=("Fresh window order and minibatch shuffles every outer step"
                                      if config.inner_shuffle_seed is None else
                                      f"Fixed per task: seed {config.inner_shuffle_seed} + task index; "
                                      "identical window order and minibatch shuffles every outer step"),
                      inner_updates=(f"Task-local CNN {config.inner_optimizer}; exactly {config.inner_epochs} full epochs "
                                     + (f"over {config.support_crops} random crops per support window (meta inner fits "
                                        "from the episode RNG; per-mouse fits from a fixed per-mouse seed)"
                                        if crops_support(config) else "over fixed support windows")
                                     + " in both meta inner and per-mouse fits"
                                     if config.algorithm in {"meta", "log_stft"} else "None"),
                      outer_updates={"meta": "Shared frontend only",
                                     "pooled": "Frontend and auxiliary CNN; CNN discarded before per-mouse fitting",
                                     "log_stft": "None; fixed frontend and one CNN fit per mouse"}[config.algorithm],
                      classifier_seed=config.seed,
                      frontend_seed=config.seed,
                      adaptation="Fresh CNN; frozen frontend; same fixed epoch budget as inner training",
                      inner_stopping="Fixed epoch budget; no early stopping; final epoch classifier; support metrics are diagnostic only",
                      meta_gradient=("Full unrolled frontend hypergradient through every fixed-epoch update with RNG-preserving segment checkpointing"
                                     if config.algorithm == "meta" else
                                     "None" if config.algorithm == "log_stft" else "Direct supervised gradient"))
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
    write_json(run_dir / "episodes.json", {mouse: [task.describe()] for mouse, task in episodes.items()})
    seed_everything(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = MetaModel(cnn_seed=config.seed).to(device)
    if config.algorithm == "log_stft":
        return run_log_stft(model, store, config, device, run_dir, identity)
    model.cnn.requires_grad_(config.algorithm == "pooled")
    outer_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    # Decoupled weight decay applies only to frontend weight matrices (not biases or norms).
    decayed = {id(parameter) for parameter in model.frontend.decay_parameters()}
    groups = [dict(params=[p for p in outer_parameters if id(p) in decayed], weight_decay=config.outer_weight_decay),
              dict(params=[p for p in outer_parameters if id(p) not in decayed], weight_decay=0.0)]
    optimizer = torch.optim.AdamW([group for group in groups if group["params"]], lr=config.outer_lr)
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
    if not resume:
        save_checkpoint(run_dir / "checkpoints" / "last.pt",
                        checkpoint(model, optimizer, 0, rng, history, selection, identity, 0.0, previous_peak))
    started = time.monotonic()
    def peak_memory():
        current = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        return max(previous_peak, current)
    try:
        for step in range(start_step + 1, config.outer_steps + 1):
            write_json(run_dir / "status.json", dict(status="training", step=step - 1,
                       active_step=step, target_steps=config.outer_steps))
            optimizer.zero_grad(set_to_none=True)
            losses, inner_reports = [], []
            for task_index, mouse in enumerate(config.targets):
                # Accumulate every mouse's equally weighted gradient before one outer update.
                episode = episodes[mouse]
                fixed_seed = (None if config.inner_shuffle_seed is None else
                              config.inner_shuffle_seed + task_index)
                # A fixed seed reuses the same window order and inner minibatch shuffles every
                # outer step, so the meta objective is a deterministic function of the frontend.
                task_rng = rng if fixed_seed is None else np.random.default_rng(fixed_seed)
                support, query = episode_batch(training[mouse], queries[mouse], episode, config, task_rng, device)
                if config.algorithm == "meta":
                    with inner_shuffle_rng(fixed_seed, device):
                        loss, fit = meta_objective(model, support, query, config, return_diagnostics=True,
                            progress=fit_progress("inner_fit", step=step, task=task_index, mouse=mouse))
                    fit.update(inner_shuffle_seed=fixed_seed)
                    fit.update(mouse=mouse, support_days=list(episode.support_days),
                               query_days=list(episode.query_days), query_samples=len(query[1]))
                    write_json(run_dir / "inner" / f"step-{step:06d}" / f"task-{task_index:03d}.json", fit)
                    inner_reports.append(fit)
                    fit_progress("query_task", step=step, task=task_index, mouse=mouse,
                                 query_split="validation", query_days=list(episode.query_days))(fit["query"])
                else:
                    loss = pooled_objective(model, support, query, config)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite outer loss")
                (loss / len(config.targets)).backward()
                losses.append(float(loss.detach()))
            grad_norm = torch.nn.utils.clip_grad_norm_(outer_parameters, config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            row = dict(step=step, mice=list(config.targets), tasks=len(losses),
                       outer_loss=float(np.mean(losses)), gradient_norm=float(grad_norm))
            if inner_reports:
                subjects = {fit["mouse"]: fit["query"] for fit in inner_reports}
                aggregate = {key: float(np.mean([metrics[key] for metrics in subjects.values()]))
                             for key in ["loss", "accuracy"]}
                row["query"] = dict(split="validation", role="query", timing="before_outer_update",
                                    aggregate=aggregate, subjects=subjects)
                fit_progress("query_step", step=step)(row["query"])
                row["inner"] = dict(epochs_per_task=config.inner_epochs, support_crops=config.support_crops,
                                    mean_epochs=float(np.mean([fit["epochs_run"] for fit in inner_reports])),
                                    mean_selected_epoch=float(np.mean([fit["selected_epoch"] for fit in inner_reports])),
                                    optimizer_steps=sum(fit["optimizer_steps"] for fit in inner_reports),
                                    fixed_epoch_tasks=sum(fit["stop_reason"] == "fixed_epochs" for fit in inner_reports))
            improved = False
            if step % config.eval_every == 0 or step == config.outer_steps:
                report, predictions, adapted = validate(model, store, config, device, step, run_dir, identity)
                roles = report_roles(report)
                row["query_refit"] = roles["query"]["aggregate"]
                if "selection" in roles:
                    row["selection"] = roles["selection"]["aggregate"]
                row["eval"] = report["eval"]["aggregate"]
                row["frontend"] = model.frontend.describe()["state"]
                score = report["aggregate"]
                previous = selection["aggregate"] if selection else None
                improved = query_improves(score, previous)
                fit_progress("query_eval_step", step=step)(
                    {role: value["aggregate"] for role, value in roles.items()})
                if improved:
                    directory = run_dir / "selected" / f"step-{step:06d}"
                    models = {}
                    for mouse, state in adapted.items():
                        path = directory / mouse / "checkpoint.pt"
                        save_checkpoint(path, state)
                        models[mouse] = dict(path=str(path.relative_to(run_dir)), sha256=file_hash(path))
                        save_predictions(path.parent, predictions[mouse], mouse_metrics(report, mouse))
                    selection = dict(report, identity=identity, adapted_models=models,
                                     criterion=selection_criterion(config))
                    save_checkpoint(directory / "frontend.pt", dict(frontend=cpu_state(model.frontend),
                                    identity=identity, step=step))
                for role, value in roles.items():
                    write_json(run_dir / role / f"step-{step:06d}.json",
                               {k: v for k, v in value.items() if k not in {"eval", "query"}})
            history.append(row)
            if step % config.checkpoint_every == 0 or "query_refit" in row:
                save_checkpoint(run_dir / "checkpoints" / "last.pt",
                                checkpoint(model, optimizer, step, rng, history, selection, identity,
                                           elapsed_seconds + time.monotonic() - started, peak_memory()))
            if improved:
                write_json(run_dir / "selection.json", selection)
            write_json(run_dir / "history.json", history)
            write_json(run_dir / "status.json", dict(status="training", step=step, target_steps=config.outer_steps))
            if step == 1 or step % 10 == 0 or "query_refit" in row:
                print(json.dumps(row), flush=True)
        write_json(run_dir / "status.json", dict(status="complete", step=config.outer_steps,
                   training_wall_seconds=elapsed_seconds + time.monotonic() - started,
                   peak_vram_mb=peak_memory(), selection=selection))
    except BaseException as error:
        write_json(run_dir / "status.json", dict(status="interrupted", error=type(error).__name__, message=str(error)))
        raise
    return selection


def evaluate_run(store, run_dir, device, split="test"):
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
        if config.algorithm == "log_stft":
            model.use_frontend(LogSTFTFrontend().to(device), config.seed)
        model.load_state_dict(state["model"])
        values = store.load(mouse, split)
        if values.fingerprint != state[f"{split}_fingerprint"]:
            raise ValueError(f"{split} sample provenance changed")
        metrics, probabilities = prediction_metrics(values.labels, predict(model, values, state["scale"], config.batch_size, device))
        metrics["evaluation_protocol_sha256"] = values.fingerprint
        metrics["classifier_fit"] = state["classifier_fit"]
        subjects[mouse] = metrics
        destination = run_dir / "evaluation" / split / mouse
        destination.mkdir(parents=True, exist_ok=True)
        table = values.table.copy()
        table["probability"], table["predicted"] = probabilities, (probabilities >= 0.5).astype(int)
        table.to_csv(destination / "predictions.csv", index=False)
        write_json(destination / "metrics.json", metrics)
    report = dict(split=split, role="eval" if split == "test" else "query", selected_step=selection["step"], identity=metadata["identity"],
                  aggregate={key: float(np.mean([row[key] for row in subjects.values()]))
                             for key in ["accuracy", "auc", "sensitivity", "specificity", "loss"]},
                  subjects=subjects, evaluation_protocol_sha256=digest(
                      {mouse: row["evaluation_protocol_sha256"] for mouse, row in subjects.items()}))
    write_json(run_dir / "evaluation" / split / "result.json", report)
    return report
