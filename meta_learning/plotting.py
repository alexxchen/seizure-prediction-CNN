"""Render convergence figures from saved metrics without training or inference."""

import argparse
import json
import math
from pathlib import Path


def metric_series(history, key, metric=None):
    """Keep the recorded step coordinates, including sparse refit evaluations."""
    points = []
    for row in history:
        value = row.get(key)
        if metric is not None:
            value = value.get(metric) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(value):
            points.append((row["step"], value))
    return tuple(zip(*points)) if points else ((), ())


def accuracy_history(run_dir, history, selection):
    """Read support accuracy from the very same CNN fits as each refit evaluation."""
    from statistics import fmean

    rows = []
    for original in history:
        if "query_refit" not in original:
            continue
        row = dict(original)
        report_path = Path(run_dir) / "query" / f"step-{row['step']:06d}.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else (
            selection if selection["step"] == row["step"] else {})
        accuracies = []
        subjects = report.get("subjects", {})
        for metrics in subjects.values():
            fit = metrics.get("classifier_fit", {})
            value = fit["last_support_accuracy"]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                break
            accuracies.append(value)
        if subjects and len(accuracies) == len(subjects):
            row["support_refit"] = dict(accuracy=fmean(accuracies))
        rows.append(row)
    return rows


def split_label(name, days):
    if not days:
        return name
    days = sorted(days)
    if len(days) == 1:
        description = f"Day {days[0]}"
    elif days == list(range(days[0], days[-1] + 1)):
        description = f"Days {days[0]}-{days[-1]}"
    else:
        description = "Days " + "/".join(map(str, days))
    return f"{name} ({description})"


def convergence_figure(history, selection, title, split_days=None):
    """Compare fit, fresh-support, and evaluation accuracies for the same refitted CNNs."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    figure = Figure(figsize=(10, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    split_days = split_days or {}
    for key, name, color, marker in [
        ("support_refit", "Support", "#7b4eaa", "s"),
        ("support_crop_validation_refit", "Fresh support crops", "#ba4c78", "v"),
        ("query_refit", "Query", "#3564a5", "o"),
        ("query_crop_validation_refit", "Fresh query crops", "#579eb0", "P"),
        ("valid", "Valid", "#d18117", "^"),
        ("test", "Test", "#338663", "D"),
    ]:
        steps, values = metric_series(history, key, "accuracy")
        if steps:
            axis.plot(steps, values, color=color, marker=marker, markersize=5, linewidth=2,
                      label=split_label(name, split_days.get(name)))

    selected_step = selection.get("step")
    if selected_step is not None and any(row.get("outer_loss") is not None for row in history):
        axis.axvline(selected_step, color="#777777", linestyle="--", linewidth=1)
        axis.text(selected_step, 1.01, f"Selected: {selected_step}",
                  transform=axis.get_xaxis_transform(), ha="center", fontsize=9, color="#555555")
    axis.set(title=f"{title} — accuracy across outer loops", xlabel="Outer loop", ylabel="Accuracy",
             ylim=(0, 1))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axis.grid(True, alpha=0.2)
    axis.set_axisbelow(True)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(loc="lower right", ncol=2)
    else:
        axis.text(0.5, 0.5, "No recorded refit accuracies", ha="center", va="center",
                  transform=axis.transAxes)
    return figure


def save_convergence_plots(run_dir):
    """Write PNG and SVG artifacts; leave saved histories and checkpoints untouched."""
    run_dir = Path(run_dir)
    history = json.loads((run_dir / "history.json").read_text())
    if not history:
        raise ValueError("Cannot plot an empty training history")
    selection = json.loads((run_dir / "selection.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    split_days = dict(Support=list(range(8)),
                      Query=config["query_validation_days"],
                      Valid=config["selection_validation_days"],
                      Test=selection["test"]["days"])
    rows = accuracy_history(run_dir, history, selection)
    title = ", ".join(sorted(selection.get("subjects", {}))) or run_dir.name
    figure = convergence_figure(rows, selection, title, split_days)
    paths = []
    try:
        for extension in ("png", "svg"):
            path = run_dir / f"convergence.{extension}"
            temporary = path.with_suffix(path.suffix + ".tmp")
            figure.savefig(temporary, format=extension, dpi=160)
            temporary.replace(path)
            paths.append(path)
    finally:
        figure.clear()
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    for path in save_convergence_plots(args.run_dir):
        print(path, flush=True)


if __name__ == "__main__":
    main()
