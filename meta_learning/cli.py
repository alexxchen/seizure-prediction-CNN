"""Command-line interfaces for planning, finite training, and selected-model evaluation."""

import argparse
import json
from pathlib import Path

from meta_learning.common import resolve_device, write_json
from meta_learning.config import Config
from meta_learning.data import DataStore, FeatureSplit
from meta_learning.engine import evaluate_run, train_run
from meta_learning.episodes import split_episode, query_task_weights
from meta_learning.training_log import progress_event
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
    parser = argparse.ArgumentParser(description="Train the waveform frontend with meta-learning")
    data_arguments(parser)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "meta.json")
    parser.add_argument("--steps", type=int, help="Override the finite outer-step budget")
    parser.add_argument("--targets", type=parse_targets)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Audit fixed support/query tasks without reading waveforms or training")
    args = parser.parse_args()
    config = Config.load(args.config).to_dict()
    for name, value in [("outer_steps", args.steps), ("targets", args.targets)]:
        if value is not None:
            config[name] = value
    config = Config.from_dict(config)
    if args.plan_only and (args.run_dir / "run.json").exists():
        parser.error("Use a separate plan directory to preserve an existing training run")
    store = DataStore(args.prepared_dir, args.data_dir, args.cache_dir, args.run_dir / "data")
    if args.plan_only:
        tasks, queries, episodes = {}, {}, {}
        for mouse in config.targets:
            tables, _ = sample_pipeline.build_tables(store.dataset, mouse)
            splits = {part: FeatureSplit(mouse, part, None, table.label.to_numpy(), table)
                      for part, table in tables.items()}
            query = splits["validation"].subset_days(config.query_validation_days)
            queries[mouse] = query
            episodes[mouse] = split_episode(splits["train"], query)
            tasks[mouse] = [dict(episodes[mouse].describe(),
                                 valid_days=(None if config.selection_validation_days is None else
                                                 list(config.selection_validation_days)),
                                 test_split="test", test_days=sorted(splits["test"].table.day.unique().tolist()),
                                 test_samples=len(splits["test"].labels))]
        counts, weights = query_task_weights(queries, episodes)
        for mouse in tasks:
            tasks[mouse][0].update(query_events=counts[mouse], outer_weight=weights[mouse])
        write_json(args.run_dir / "episodes.json", tasks)
        print(json.dumps(dict(training=False, episodes={mouse: len(rows) for mouse, rows in tasks.items()},
                              query_events=counts, outer_task_weights=weights)))
        return
    result = train_run(config, store, args.run_dir, args.prepared_dir, resolve_device(args.device), args.resume)
    # Plot only after a successful run, including a completed compatible resume.
    from meta_learning.plotting import save_convergence_plots

    paths = save_convergence_plots(args.run_dir)
    progress_event('convergence_plots', dict(paths=[str(path) for path in paths]), run_dir=args.run_dir)


def evaluate_main():
    parser = argparse.ArgumentParser(description="Replay query, valid, or test using selected checkpoints; no fitting")
    data_arguments(parser)
    parser.add_argument("--split", choices=["query", "valid", "test"], default="test",
                        help="query: outer-training days; valid: checkpoint-selection days; test: development Days 12/13 (default)")
    args = parser.parse_args()
    store = DataStore(args.prepared_dir, args.data_dir, args.cache_dir, args.run_dir / "data")
    result = evaluate_run(store, args.run_dir, resolve_device(args.device), args.split)
    print(json.dumps(result), flush=True)
