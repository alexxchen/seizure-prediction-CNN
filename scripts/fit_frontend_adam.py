"""Finite Adam personalization of a frozen archived frontend, selected by validation loss."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def probabilities(cnn, features, batch_size):
    cnn.eval()
    with torch.no_grad():
        return torch.cat([cnn(batch).softmax(1) for batch in features.split(batch_size)])


def fit_cnn(cnn, support, labels, validation, validation_labels, config, *, epochs, lr,
            resume=None, progress=None):
    """Train all epochs; validation chooses calibrated copies without changing the solver.

    Adam moments and BatchNorm EMA persist across epochs. Each candidate receives
    exact support-only BatchNorm calibration on a copy; the training EMA is untouched.
    Resume restores Adam, training buffers, best candidate, history, and shuffle RNG.
    """
    from meta_learning.common import prediction_metrics
    from meta_learning.inner import calibrate_batchnorm

    if epochs < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Use a positive epoch budget and finite positive Adam learning rate")
    if (not len(labels) or not len(validation_labels) or len(support) != len(labels)
            or len(validation) != len(validation_labels)):
        raise ValueError("Nonempty matching support/validation features and labels are required")
    support, validation = support.detach(), validation.detach()
    for layer in cnn.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = config.classifier_dropout
    anchor = [parameter.detach().clone() for parameter in cnn.parameters()]
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    history, best, start, state = [], None, 0, None
    if resume is not None:
        if resume["epochs_requested"] != epochs or resume["adam_lr"] != lr:
            raise ValueError("Resume budget or Adam learning rate differs")
        cnn.load_state_dict(resume["cnn"])
        optimizer.load_state_dict(resume["optimizer"])
        history, best, start = copy.deepcopy(resume["history"]), resume["best"], resume["epoch"]
        state = resume
        torch.set_rng_state(resume["torch_rng"])
        if support.is_cuda:
            torch.cuda.set_rng_state(resume["cuda_rng"], support.device)
    for epoch in range(start + 1, epochs + 1):
        cnn.train()
        order = torch.randperm(len(labels), device=support.device)
        for indices in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(cnn(support[indices]), labels[indices],
                                   label_smoothing=config.label_smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Adam support loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), config.gradient_clip, error_if_nonfinite=True)
            if config.inner_weight_decay:
                for parameter, origin in zip(cnn.parameters(), anchor):
                    parameter.grad.add_(parameter.detach() - origin, alpha=config.inner_weight_decay)
            optimizer.step()
        candidate = copy.deepcopy(cnn).eval()
        with torch.no_grad():
            calibrated = calibrate_batchnorm(candidate, dict(candidate.named_parameters()),
                                             dict(candidate.named_buffers()), support)
            candidate.load_state_dict(dict(candidate.named_parameters(), **calibrated))
            support_probabilities = probabilities(candidate, support, config.batch_size)
            validation_probabilities = probabilities(candidate, validation, config.batch_size)
        support_metrics, _ = prediction_metrics(labels.cpu().numpy(), support_probabilities.cpu().numpy())
        validation_metrics, _ = prediction_metrics(validation_labels.cpu().numpy(),
                                                   validation_probabilities.cpu().numpy())
        if not math.isfinite(validation_metrics["loss"]):
            raise FloatingPointError("Non-finite Adam validation loss")
        row = dict(epoch=epoch, support_loss=support_metrics["loss"],
                   support_accuracy=support_metrics["accuracy"], val_loss=validation_metrics["loss"],
                   val_accuracy=validation_metrics["accuracy"])
        history.append(row)
        if best is None or row["val_loss"] < best["loss"]:
            best = dict(epoch=epoch, loss=row["val_loss"], cnn=cpu_tree(candidate.state_dict()))
        state = dict(epoch=epoch, epochs_requested=epochs, adam_lr=lr, cnn=cpu_tree(cnn.state_dict()),
                     optimizer=cpu_tree(optimizer.state_dict()), best=best, history=history,
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(support.device) if support.is_cuda else None)
        if progress is not None:
            progress(row, state)
    cnn.load_state_dict(best["cnn"])
    return cnn.eval(), state


def encode(model, values, config, device):
    features = []
    with torch.no_grad():
        for start in range(0, len(values.labels), config.frontend_batch_size):
            inputs, _ = values.batch(np.arange(start, min(start + config.frontend_batch_size,
                                                         len(values.labels))), 1.0, device)
            features.append(model.frontend(inputs))
    return torch.cat(features), torch.tensor(np.array(values.labels, copy=True), device=device, dtype=torch.long)


def plot_history(history, path, mouse, selected_epoch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, metric in zip(axes, ("loss", "accuracy")):
        for role in ("support", "val"):
            axis.plot([row["epoch"] for row in history], [row[f"{role}_{metric}"] for row in history], label=role)
        axis.axvline(selected_epoch, color="gray", linestyle="--", label="selected epoch")
        axis.set(xlabel="Epoch", ylabel=metric.capitalize(), title=f"{mouse}: {metric}")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)


def main(args):
    source_run, output = args.source_run.resolve(), args.run_dir.resolve()
    metadata = json.loads((source_run / "run.json").read_text())
    selection = json.loads((source_run / "selection.json").read_text())
    if selection["identity"] != metadata["identity"]:
        raise ValueError("Selection belongs to another source run")
    source = source_run / "source"
    for name, expected in metadata["provenance"]["source_sha256"].items():
        if file_hash(source / name) != expected:
            raise ValueError(f"Archived source changed: {name}")
    # All model, sample, crop, and calibration code comes from the verified source run.
    sys.path.insert(0, str(source))
    from meta_learning.adaptation import predict, support_features
    from meta_learning.common import digest, prediction_metrics, resolve_device, save_checkpoint, seed_everything, write_json
    from meta_learning.config import Config
    from meta_learning.data import DataStore
    from meta_learning.engine import runtime_identity
    from meta_learning.episodes import crops_support
    from meta_learning.model import MetaModel
    from source_data import TARGETS

    config = Config.from_dict(metadata["provenance"]["config"])
    if config.algorithm != "meta":
        raise ValueError("This experiment requires a learned meta frontend")
    targets = args.targets.split(",") if args.targets else list(TARGETS)
    if not targets or len(set(targets)) != len(targets) or set(targets) - set(TARGETS):
        raise ValueError("Choose distinct supported mice")
    device = resolve_device(args.device)
    runtime = runtime_identity(device)
    if runtime["packages"] != metadata["provenance"]["runtime"]["packages"]:
        raise ValueError("Use the source run's package versions")
    frontend_path = source_run / "selected" / f"step-{selection['step']:06d}" / "frontend.pt"
    # Verify that the standalone frontend matches the hash-verified selected models.
    frontend_state = torch.load(frontend_path, weights_only=True, map_location="cpu")
    if frontend_state["identity"] != metadata["identity"] or frontend_state["step"] != selection["step"]:
        raise ValueError("Frontend checkpoint identity mismatch")
    tracked = {str(source_run / name): file_hash(source_run / name) for name in ("run.json", "selection.json")}
    tracked[str(frontend_path)] = file_hash(frontend_path)
    for mouse in targets:
        record = selection["adapted_models"][mouse]
        path = source_run / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Source checkpoint changed: {mouse}")
        archived = torch.load(path, weights_only=True, map_location="cpu")
        if archived["identity"] != metadata["identity"] or archived["mouse"] != mouse:
            raise ValueError("Source adapted checkpoint identity mismatch")
        attached = {key.removeprefix("frontend."): value for key, value in archived["model"].items()
                    if key.startswith("frontend.")}
        if attached.keys() != frontend_state["frontend"].keys() or any(
                not torch.equal(value, attached[key]) for key, value in frontend_state["frontend"].items()):
            raise ValueError("Frontend differs from selected adapted checkpoints")
        tracked[str(path)] = record["sha256"]
    experiment = dict(source_identity=metadata["identity"], source_step=selection["step"],
                      input_sha256=tracked, targets=targets, epochs=args.epochs, optimizer="adam", lr=args.lr,
                      source_config=config.to_dict(), validation_days=[8, 11], test_days=[12, 13],
                      selection="Per-mouse minimum validation log loss across all completed epochs",
                      validation_scope="Days 8/11 already supervised the source frontend; reused for CNN epoch selection",
                      batchnorm="Train EMA; each candidate calibrated on support only, without changing training EMA",
                      script_sha256=file_hash(__file__), runtime=runtime)
    identity = digest(experiment)
    if args.resume:
        previous = json.loads((output / "run.json").read_text())
        if previous["identity"] != identity:
            raise ValueError("Resume identity differs from the saved experiment")
    else:
        output.mkdir(parents=True, exist_ok=False)
        shutil.copytree(source, output / "source")
        shutil.copyfile(__file__, output / Path(__file__).name)
        write_json(output / "run.json", dict(identity=identity, experiment=experiment))
        save_checkpoint(output / "frontend.pt", frontend_state)
    store = DataStore(args.prepared_dir, args.data_dir, args.cache_dir, output / "data")
    if digest(store.dataset.signature) != metadata["provenance"]["source_inventory_sha256"]:
        raise ValueError("Source inventory differs from frontend training")
    seed_everything(config.seed)
    started, subjects = time.monotonic(), {}
    for mouse in targets:
        directory = output / mouse
        directory.mkdir(exist_ok=True)
        report_path = directory / "report.json"
        if args.resume and report_path.exists():
            report = json.loads(report_path.read_text())
            if report["identity"] != identity or file_hash(directory / "checkpoint.pt") != report["checkpoint_sha256"]:
                raise ValueError("Completed mouse checkpoint changed")
            subjects[mouse] = report
            continue
        mouse_started = time.monotonic()
        seed = config.seed + 10000 + list(TARGETS).index(mouse)
        torch.manual_seed(seed)
        with torch.random.fork_rng():
            model = MetaModel(cnn_seed=config.seed, frontend_batch_size=config.frontend_batch_size)
        model.frontend.load_state_dict(frontend_state["frontend"], strict=True)
        model.to(device).eval()
        model.frontend.requires_grad_(False)
        training = store.load_crops(mouse, "train") if crops_support(config) else store.load(mouse, "train")
        if not training.table.day.between(0, 7).all():
            raise ValueError("CNN training must use Days 0-7 only")
        if training.fingerprint != metadata["provenance"]["training_fingerprints"][mouse]:
            raise ValueError("Support sample fingerprint changed")
        validation = store.load(mouse, "validation").subset_days((8, 11))
        if validation.fingerprint != metadata["provenance"]["query_fingerprints"][mouse]:
            raise ValueError("Validation sample fingerprint changed")
        print(f"{mouse}: encoding seeded support crops and fixed validation windows", flush=True)
        features, labels = support_features(model, training, config, np.random.default_rng(seed), device)
        val_features, val_labels = encode(model, validation, config, device)
        last_path = directory / "last.pt"
        resume = torch.load(last_path, weights_only=True, map_location="cpu") if args.resume and last_path.exists() else None
        if resume is not None and resume["identity"] != identity:
            raise ValueError("Mouse resume checkpoint identity mismatch")

        def progress(row, state):
            save_checkpoint(last_path, dict(state, identity=identity, mouse=mouse, seed=seed))
            write_json(directory / "history.json", state["history"])
            write_json(output / "status.json", dict(state="running", mouse=mouse, epoch=row["epoch"],
                                                     selected_epoch=state["best"]["epoch"]))
            print(json.dumps(dict(mouse=mouse, **row, selected_epoch=state["best"]["epoch"])), flush=True)

        model.cnn, state = fit_cnn(model.cnn, features, labels, val_features, val_labels, config,
                                   epochs=args.epochs, lr=args.lr, resume=resume, progress=progress)
        del val_features, val_labels
        if any(not torch.equal(value.cpu(), frontend_state["frontend"][key])
               for key, value in model.frontend.state_dict().items()):
            raise ValueError("Frozen frontend changed")
        model.eval()
        selected_path = directory / "checkpoint.pt"
        save_checkpoint(selected_path, dict(model=cpu_tree(model.state_dict()), identity=identity, mouse=mouse,
                                           source_identity=metadata["identity"], source_step=selection["step"],
                                           scale=1.0, selected_epoch=state["best"]["epoch"], epochs_run=state["epoch"],
                                           optimizer="adam", adam_lr=args.lr, seed=seed,
                                           training_fingerprint=training.fingerprint,
                                           validation_fingerprint=validation.fingerprint,
                                           source_config=config.to_dict()))
        metrics = {}
        # The test partition is first materialized after CNN epoch selection is finished.
        for role, values in (("val", validation), ("test", store.load(mouse, "test"))):
            buffers = cpu_tree(dict(model.named_buffers()))
            predictions = predict(model, values, 1.0, config.batch_size, device,
                                  frontend_batch_size=config.frontend_batch_size)
            score, positive = prediction_metrics(values.labels, predictions)
            if any(not torch.equal(value.cpu(), buffers[key]) for key, value in model.named_buffers()):
                raise ValueError("Inference changed BatchNorm buffers")
            if role == "val" and not np.isclose(score["loss"], state["best"]["loss"], atol=5e-6, rtol=1e-5):
                raise ValueError("Selected validation predictions failed replay")
            table = values.table.copy()
            table["probability"], table["predicted"] = positive, (positive >= 0.5).astype(int)
            table.to_csv(directory / f"predictions_{role}.csv", index=False)
            metrics[role] = dict(score, fingerprint=values.fingerprint,
                                 days=sorted(int(day) for day in values.table.day.unique()))
            write_json(directory / f"{role}_metrics.json", metrics[role])
        report = dict(identity=identity, mouse=mouse, seed=seed, epochs_run=state["epoch"],
                      selected_epoch=state["best"]["epoch"], best_val_loss=state["best"]["loss"],
                      optimizer="adam", adam_lr=args.lr, support_samples=len(labels),
                      support_positives=int((labels == 1).sum()), support_positive_fraction=float((labels == 1).float().mean()),
                      support_events=int(len(training.table[training.table.label == 1].drop_duplicates(["day", "event_index"]))),
                      support_crops=config.support_crops if crops_support(config) else 0,
                      training_fingerprint=training.fingerprint, metrics=metrics,
                      checkpoint_sha256=file_hash(selected_path), wall_seconds=time.monotonic() - mouse_started)
        plot_history(state["history"], directory / "convergence.png", mouse, report["selected_epoch"])
        write_json(report_path, report)
        subjects[mouse] = report
        print(json.dumps(dict(stage="mouse_complete", **report)), flush=True)
        store.loaded.clear()
        del features, labels, model, training, validation, state
    if any(file_hash(path) != expected for path, expected in tracked.items()):
        raise ValueError("Source run artifacts changed during fitting")
    aggregate = {role: {key: float(np.mean([row["metrics"][role][key] for row in subjects.values()]))
                        for key in ("loss", "accuracy", "auc", "sensitivity", "specificity")}
                 for role in ("val", "test")}
    report = dict(identity=identity, experiment=experiment, subjects=subjects, aggregate=aggregate,
                  wall_seconds_this_invocation=time.monotonic() - started,
                  total_mouse_wall_seconds=sum(row["wall_seconds"] for row in subjects.values()),
                  completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "report.json", report)
    lines = ["# Frozen meta frontend: Adam personalization", "",
             "Days 0-7 train each fresh CNN. All requested epochs complete; minimum Days 8/11 log loss selects each CNN.",
             "Days 8/11 previously supervised the source frontend. Days 12/13 are inference only after selection.", "",
             "| Mouse | Selected epoch | Val loss | Val accuracy | Test loss | Test accuracy | Test AUC |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mouse, row in subjects.items():
        val, test = row["metrics"]["val"], row["metrics"]["test"]
        lines.append(f"| {mouse} | {row['selected_epoch']} | {val['loss']:.4f} | {val['accuracy']:.4f} | "
                     f"{test['loss']:.4f} | {test['accuracy']:.4f} | {test['auc']:.4f} |")
    lines.append(f"| Mouse mean | | {aggregate['val']['loss']:.4f} | {aggregate['val']['accuracy']:.4f} | "
                 f"{aggregate['test']['loss']:.4f} | {aggregate['test']['accuracy']:.4f} | {aggregate['test']['auc']:.4f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(output / "status.json", dict(state="complete", targets=targets, aggregate=aggregate))
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/users/jianchen/data/seizure_data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/data/users/jianchen/data/seizure_data-cache/meta-learning"))
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--targets", help="Comma-separated mice; defaults to all seven")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Use positive epochs and a finite positive learning rate")
    args.prepared_dir = args.prepared_dir or args.cache_dir / "prepared"
    main(args)
