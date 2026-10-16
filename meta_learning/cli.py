"""Command-line interfaces for planning, finite training, and selected-model evaluation."""

import argparse
import json
from pathlib import Path

from meta_learning.common import resolve_device, write_json
from meta_learning.config import Config
from meta_learning.data import DataStore, FeatureSplit
from meta_learning.engine import evaluate_run, train_run
from meta_learning.episodes import chronological_episodes
from prepare import parse_targets, source_settings
import sample_pipeline

ROOT = Path(__file__).resolve().parent.parent


def data_arguments(parser):
    settings = source_settings()
    parser.add_argument("--data-dir", type=Path, default=Path(settings["datadir"]))
    parser.add_argument("--prepared-dir", type=Path, default=Path("prepared"))
    parser.add_argument("--cache-dir", type=Path, default=Path(settings["cachedir"]) / "meta-learning")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")


def train_main():
    parser = argparse.ArgumentParser(description="Train a cross-day meta-learner or a matched pooled baseline")
    data_arguments(parser)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "meta.json")
    parser.add_argument("--algorithm", choices=["meta", "pooled"])
    parser.add_argument("--steps", type=int, help="Override the finite outer-step budget")
    parser.add_argument("--targets", type=parse_targets)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Audit chronological tasks without reading waveforms or training")
    args = parser.parse_args()
    config = Config.load(args.config).to_dict()
    for name, value in [("algorithm", args.algorithm), ("outer_steps", args.steps), ("targets", args.targets)]:
        if value is not None:
            config[name] = value
    config = Config.from_dict(config)
    if args.plan_only and (args.run_dir / "run.json").exists():
        parser.error("Use a separate plan directory to preserve an existing training run")
    store = DataStore(args.prepared_dir, args.data_dir, args.cache_dir, args.run_dir / "data")
    if args.plan_only:
        tasks = {}
        for mouse in config.targets:
            tables, _ = sample_pipeline.build_tables(store.dataset, mouse)
            table = tables["train"]
            split = FeatureSplit(mouse, "train", None, table.label.to_numpy(), table)
            tasks[mouse] = [task.describe() for task in chronological_episodes(
                split, config.min_support_days, config.query_days)]
        write_json(args.run_dir / "episodes.json", tasks)
        print(json.dumps(dict(training=False, episodes={mouse: len(rows) for mouse, rows in tasks.items()})))
        return
    result = train_run(config, store, args.run_dir, args.prepared_dir, resolve_device(args.device), args.resume)
    print(json.dumps(result), flush=True)


def evaluate_main():
    parser = argparse.ArgumentParser(description="Run inference with selected per-mouse checkpoints; no fitting")
    data_arguments(parser)
    parser.add_argument("--split", choices=["validation", "test"], default="validation",
                        help="Test evaluation requires an explicit --split test")
    args = parser.parse_args()
    store = DataStore(args.prepared_dir, args.data_dir, args.cache_dir, args.run_dir / "data")
    result = evaluate_run(store, args.run_dir, resolve_device(args.device), args.split)
    print(json.dumps(result), flush=True)
