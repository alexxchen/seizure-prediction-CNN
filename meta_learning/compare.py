"""Compare completed, matched meta and pooled runs without launching training."""

import argparse
import json
from pathlib import Path

from meta_learning.common import digest, write_json


def compare_runs(first, second):
    directories = [Path(first), Path(second)]
    metadata = [json.loads((path / "run.json").read_text())["provenance"] for path in directories]
    selections = [json.loads((path / "selection.json").read_text()) for path in directories]
    if any(selection["identity"] != digest(provenance) for selection, provenance in zip(selections, metadata)):
        raise ValueError("Selection identity does not match run provenance")
    a, b = [dict(item["config"]) for item in metadata]
    algorithms = [config.pop("algorithm") for config in [a, b]]
    if a != b or set(algorithms) != {"meta", "pooled"}:
        raise ValueError("Compare meta and pooled runs with matching remaining configuration")
    for key in ["training_fingerprints", "query_fingerprints", "split_roles", "selection_criterion", "source_inventory_sha256", "runtime", "sample_config",
                "spectrum_config", "frontend_config", "source_sha256"]:
        if metadata[0][key] != metadata[1][key]:
            raise ValueError(f"Comparison mismatch: {key}")
    for mouse in a["targets"]:
        for reports in [selections, [report["eval"] for report in selections]]:
            if reports[0]["subjects"][mouse]["evaluation_protocol_sha256"] != reports[1]["subjects"][mouse]["evaluation_protocol_sha256"]:
                raise ValueError("Query/eval sample fingerprints differ")
    reports = dict(zip(algorithms, selections))
    statuses = [json.loads((path / "status.json").read_text()) for path in directories]
    if any(status["status"] != "complete" for status in statuses):
        raise ValueError("Compare completed runs")
    return dict(aggregate={algorithm: report["aggregate"] for algorithm, report in reports.items()},
                aggregate_role=metadata[0]["selection_criterion"],
                eval={algorithm: report["eval"]["aggregate"] for algorithm, report in reports.items()},
                eval_meta_minus_pooled=reports["meta"]["eval"]["aggregate"]["accuracy"] - reports["pooled"]["eval"]["aggregate"]["accuracy"],
                meta_minus_pooled=reports["meta"]["aggregate"]["accuracy"] - reports["pooled"]["aggregate"]["accuracy"],
                compute={algorithm: {key: status[key] for key in ["training_wall_seconds", "peak_vram_mb"]}
                         for algorithm, status in zip(algorithms, statuses)},
                query_protocol={mouse: reports["meta"]["subjects"][mouse]["evaluation_protocol_sha256"] for mouse in a["targets"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=2, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_runs(*args.runs)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
