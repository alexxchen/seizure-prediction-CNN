"""Plot saved frontend features and replayed classification outcomes without fitting.

Adapted from the input-embedding analysis preserved under
docker-output-input-embeddings/*/MouseLFP/source_snapshot/
analyze_input_embeddings_final.py. PCA uses support reference windows only;
t-SNE jointly lays out all fixed reference windows of each mouse.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def saved_source(run_dir, metadata):
    """Replay the archived implementation, including its preprocessing and model."""
    source = run_dir / "source"
    for name, expected in metadata["provenance"]["source_sha256"].items():
        if file_hash(source / name) != expected:
            raise ValueError(f"Saved source changed: {name}")
    sys.path.insert(0, str(source))
    return source


def annotate(table, probabilities):
    """Keep the binary threshold identical to checkpoint evaluation (ties positive)."""
    result = table.copy()
    probabilities = np.asarray(probabilities)
    if (probabilities.shape != (len(table),) or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
            or not set(table.label.unique()).issubset({0, 1})):
        raise ValueError("Expected one finite binary probability per labeled window")
    result["probability"] = probabilities
    result["predicted"] = (probabilities >= 0.5).astype(int)
    result["misclassified"] = result.predicted != result.label
    result["outcome"] = np.where(result.label == 1,
                                  np.where(result.predicted == 1, "TP", "FN"),
                                  np.where(result.predicted == 1, "FP", "TN"))
    return result


def extract(model, values, batch_size, device):
    """Extract exactly what the fitted CNN consumes, keeping BatchNorm frozen."""
    model.eval()
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    frontend_batch_size = model.frontend.batch_size
    features, probabilities = [], []
    with torch.inference_mode():
        for start in range(0, len(values.labels), frontend_batch_size):
            indices = np.arange(start, min(start + frontend_batch_size, len(values.labels)))
            inputs, _ = values.batch(indices, device)
            output = model.frontend(inputs)
            if not torch.isfinite(output).all():
                raise ValueError("Non-finite frontend features")
            features.append(output.cpu().numpy())
            for batch in output.split(batch_size):
                probabilities.append(model.cnn(batch).softmax(dim=1)[:, 1].cpu().numpy())
    if any(not torch.equal(value, buffers[name]) for name, value in model.named_buffers()):
        raise ValueError("Inference mutated model buffers")
    return np.concatenate(features), np.concatenate(probabilities)


def embed(features, support, components, perplexity, iterations, seed):
    """Flatten the entire channel/time output; do not add feature normalization."""
    flat = features.reshape(len(features), -1)
    count = min(components, int(support.sum()) - 1, flat.shape[1])
    if count < 2 or len(flat) < 4:
        raise ValueError("At least three support windows and four total windows are required")
    pca = PCA(n_components=count, svd_solver="randomized", iterated_power=4, random_state=seed)
    pca.fit(flat[support])
    projected = pca.transform(flat)
    tsne = TSNE(n_components=2, perplexity=min(perplexity, (len(flat) - 1) / 3),
                init="pca", learning_rate="auto", random_state=seed, max_iter=iterations,
                n_jobs=4)
    coordinates = tsne.fit_transform(projected)
    return coordinates, projected, pca, tsne


def scatter(axis, table, role=None):
    if role is not None:
        background = table[table.role != role]
        axis.scatter(background.tsne_1, background.tsne_2, color="#b6bcc6", s=8,
                     alpha=0.16, linewidths=0, rasterized=True)
        table = table[table.role == role]
    for label, color in [(0, "#2878b5"), (1, "#169b62")]:
        group = table[table.label == label]
        axis.scatter(group.tsne_1, group.tsne_2, color=color, s=17, alpha=0.65,
                     linewidths=0, rasterized=True)
    wrong = table[table.misclassified]
    axis.scatter(wrong.tsne_1, wrong.tsne_2, color="#d62738", marker="x", s=42,
                 linewidths=1.1, rasterized=True)
    counts = table.outcome.value_counts()
    title = "All reference windows" if role is None else f"{role.capitalize()} | Days " + "/".join(
        str(day) for day in sorted(table.day.unique()))
    axis.set(title=f"{title}\nN={len(table)} | FP={counts.get('FP', 0)}, FN={counts.get('FN', 0)}",
             xlabel="t-SNE 1", ylabel="t-SNE 2")
    axis.grid(alpha=0.12)


def legend():
    from matplotlib.lines import Line2D

    return [Line2D([], [], linestyle="", marker="o", color="#169b62", label="Positive (true label)"),
            Line2D([], [], linestyle="", marker="o", color="#2878b5", label="Negative (true label)"),
            Line2D([], [], linestyle="", marker="x", color="#d62738", label="Misclassified (FP / FN)")]


def save_figure(figure, path):
    for extension in ("png", "pdf", "svg"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=180, bbox_inches="tight")
    figure.clear()


def plot_mouse(table, mouse, directory, step, feature_shape):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    roles = list(table.role.unique())
    figure = Figure(figsize=(5 * (len(roles) + 1), 5.6))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(roles) + 1, sharex=True, sharey=True)
    for axis, role in zip(axes, [None, *roles]):
        scatter(axis, table, role)
    figure.suptitle(f"{mouse}: frontend {feature_shape[0]} channels x {feature_shape[1]} time bins"
                   f" | selected step {step}\nOne point = one fixed 112 s reference window; shared layout across panels")
    figure.legend(handles=legend(), loc="lower center", ncol=3)
    figure.subplots_adjust(top=0.76, bottom=0.18, wspace=0.23)
    save_figure(figure, directory / f"{mouse}_frontend_tsne")


def main(args):
    run_dir = args.run_dir.resolve()
    metadata = json.loads((run_dir / "run.json").read_text())
    selection = json.loads((run_dir / "selection.json").read_text())
    source = saved_source(run_dir, metadata)
    # Import only after selecting the verified source snapshot.
    from meta_learning.common import digest, resolve_device, seed_everything
    from meta_learning.config import Config
    from meta_learning.data import DataStore
    from meta_learning.engine import validation_roles
    from meta_learning.model import MetaModel

    if selection["identity"] != metadata["identity"]:
        raise ValueError("Selection belongs to another run")
    config = Config.from_dict(metadata["provenance"]["config"])
    seed_everything(config.seed)
    targets = args.targets.split(",") if args.targets else list(config.targets)
    if not targets or len(set(targets)) != len(targets) or set(targets) - set(selection["adapted_models"]):
        raise ValueError("Choose distinct mice with selected adapted checkpoints")
    device = resolve_device(args.device)
    output = (args.output or run_dir / "visualizations" / (
        "frontend-tsne-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / Path(__file__).name)
    store = DataStore(args.prepared_dir, args.data_dir, args.cache_dir, output / "data")
    if digest(store.dataset.signature) != metadata["provenance"]["source_inventory_sha256"]:
        raise ValueError("Source inventory differs from training")
    roles = [("support", "train", tuple(range(8))), *validation_roles(config)]
    tables, reports = {}, {}
    tracked = {str(run_dir / name): file_hash(run_dir / name) for name in ("run.json", "selection.json")}
    for mouse in targets:
        print(f"{mouse}: replaying selected step {selection['step']} on {device}", flush=True)
        record = selection["adapted_models"][mouse]
        path = run_dir / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Selected checkpoint changed: {mouse}")
        tracked[str(path)] = record["sha256"]
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["identity"] != metadata["identity"] or state["mouse"] != mouse or state["step"] != selection["step"]:
            raise ValueError("Checkpoint identity mismatch")
        with torch.random.fork_rng():
            model = MetaModel(cnn_seed=config.seed, frontend_batch_size=config.frontend_batch_size)
        model.load_state_dict(state["model"], strict=True)
        model.to(device).requires_grad_(False)
        rows, arrays, checks = [], [], {}
        for role, part, days in roles:
            values = store.load(mouse, part)
            key = "training_fingerprint" if part == "train" else f"{part}_fingerprint"
            if values.fingerprint != state[key]:
                raise ValueError(f"{mouse}: {part} sample provenance changed")
            values = values.subset_days(days) if days is not None else values
            if role != "support" and values.fingerprint != state[f"{role}_fingerprint"]:
                raise ValueError(f"{mouse}: {role} sample provenance changed")
            features, probabilities = extract(model, values, config.batch_size, device)
            table = annotate(values.table, probabilities)
            table["role"] = role
            check = dict(samples=len(table), days=sorted(int(x) for x in table.day.unique()),
                         fingerprint=values.fingerprint, errors=int(table.misclassified.sum()))
            if role != "support":
                saved_path = path.parent / f"predictions_{role}.csv"
                tracked[str(saved_path)] = file_hash(saved_path)
                previous = pd.read_csv(saved_path)
                pd.testing.assert_frame_equal(values.table.reset_index(drop=True), previous[values.table.columns])
                delta = float(np.max(np.abs(probabilities - previous.probability.to_numpy())))
                if delta > 5e-6 or not np.array_equal(table.predicted, previous.predicted):
                    raise ValueError(f"Saved predictions do not replay: {mouse}/{role}, delta={delta}")
                check["maximum_prediction_replay_error"] = delta
                check["predicted_classes_identical"] = True
            checks[role] = check
            rows.append(table)
            arrays.append(features)
            print(f"  {role}: {len(table)} windows, {check['errors']} errors", flush=True)
        table, features = pd.concat(rows, ignore_index=True), np.concatenate(arrays)
        print(f"{mouse}: support-only PCA, then t-SNE on {len(table)} windows", flush=True)
        coordinates, projected, pca, tsne = embed(features, (table.role == "support").to_numpy(),
                                                 args.components, args.perplexity, args.iterations, args.seed)
        table[["tsne_1", "tsne_2"]] = coordinates
        directory = output / mouse
        directory.mkdir()
        table.to_csv(directory / "embedding_windows.csv", index=False)
        np.savez_compressed(directory / "features.npz", frontend=features, pca=projected,
                            tsne=coordinates, pca_components=pca.components_, pca_mean=pca.mean_,
                            explained_variance_ratio=pca.explained_variance_ratio_)
        plot_mouse(table, mouse, directory, selection["step"], features.shape[1:])
        reports[mouse] = dict(checkpoint=record, checks=checks, feature_shape=list(features.shape[1:]),
                              pca_components=pca.n_components_,
                              pca_variance_retained=float(pca.explained_variance_ratio_.sum()),
                              perplexity=tsne.perplexity, kl_divergence=float(tsne.kl_divergence_),
                              iterations_run=int(tsne.n_iter_))
        tables[mouse] = table
        # Avoid retaining waveform arrays for the entire cohort.
        store.loaded.clear()
        del features, arrays, values, model
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    columns = min(2, len(tables))
    figure = Figure(figsize=(7 * columns, 5 * ((len(tables) + columns - 1) // columns) + 1))
    FigureCanvasAgg(figure)
    axes = figure.subplots((len(tables) + columns - 1) // columns, columns, squeeze=False)
    for axis, (mouse, table) in zip(axes.flat, tables.items()):
        scatter(axis, table, "test")
        axis.set_title(mouse + " | " + axis.get_title())
    for axis in list(axes.flat)[len(tables):]:
        axis.set_visible(False)
    figure.suptitle(f"Frontend t-SNE | selected step {selection['step']} | test outcomes\n"
                   "Each mouse has an independent layout; gray = support/query reference")
    figure.legend(handles=legend(), loc="lower center", ncol=3)
    figure.subplots_adjust(top=0.87, bottom=0.1, hspace=0.4, wspace=0.24)
    save_figure(figure, output / "frontend_tsne_test_all_mice")
    combined = pd.concat(tables.values(), ignore_index=True)
    combined.to_csv(output / "all_embedding_windows.csv", index=False)
    counts = combined.groupby(["mouse", "role", "outcome"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["TP", "TN", "FP", "FN"], fill_value=0)
    counts.to_csv(output / "outcome_counts.csv")
    if any(file_hash(name) != expected for name, expected in tracked.items()):
        raise ValueError("Input run artifacts changed during visualization")
    import importlib.metadata

    report = dict(run=str(run_dir), source_snapshot=str(source), selected_step=selection["step"],
                  algorithm="meta", seed=args.seed, tsne_iterations=args.iterations,
                  pca_fit="support fixed reference windows only", tsne_fit="all fixed reference windows per mouse",
                  feature_transform="flatten full frontend output; no added scaling",
                  labels_used_to_fit_embedding=False, training_performed=False,
                  support_view="fixed reference windows, not the random crops used during fitting",
                  threshold=0.5, subjects=reports, input_sha256=tracked,
                  runtime={name: importlib.metadata.version(name) for name in
                           ["torch", "numpy", "scipy", "scikit-learn", "matplotlib"]})
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(
        f"Frontend feature visualization, selected step {selection['step']}\n\n"
        "Each point is a fixed 112 s window. Green: positive; blue: negative; red x: classification error.\n"
        "FP = false positive; FN = false negative. Errors use the saved adapted CNN at threshold 0.5.\n"
        "PCA is fitted on support reference features only, without additional normalization.\n"
        "t-SNE jointly displays all splits, with no labels supplied. Each mouse is projected separately.\n"
        "The support panel uses fixed reference windows, not the random training crops.\n"
        "Test points influence this visualization layout only. t-SNE distances and cluster sizes are exploratory.\n"
        "No model fitting or BatchNorm recalibration was performed; saved query/test predictions were verified.\n\n"
        "Counts are recorded in outcome_counts.csv; per-window coordinates and probabilities are in embedding_windows.csv.\n")
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/users/jianchen/data/seizure_data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/data/users/jianchen/data/seizure_data-cache/meta-learning"))
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are never overwritten")
    parser.add_argument("--targets", help="Comma-separated subset of mice with selected checkpoints")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.components < 2 or args.iterations < 250 or not np.isfinite(args.perplexity) or args.perplexity <= 0 or args.seed < 0:
        parser.error("Use >=2 PCA components, >=250 iterations, positive perplexity, and a nonnegative seed")
    args.prepared_dir = args.prepared_dir or args.cache_dir / "prepared"
    main(args)
