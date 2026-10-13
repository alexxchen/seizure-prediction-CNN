# autoresearch

An AI agent improves MouseLFP seizure prediction by changing code, running experiments, and retaining better validation results. Use English throughout the project.

## Setup

When the user requests research:

1. **Choose a run tag:** use a fresh tag for a new study; reuse the existing session when resuming.
2. **Create the branch:** use `autoresearch/<tag>`. Inspect git status and protect existing user changes.
3. **Read the project:** `README.md` gives context; `prepare.py`, `research_protocol.py`, and `protocol.lock.json` define the fixed data protocol; `evaluate.py` defines scoring; `train.py` and `models/*.py` contain the editable recipe.
4. **Verify data exists:** the default prepared dataset is `/data/users/jianchen/data/seizure_data-cache/autoresearch-raw-v1`. Check for `manifest.json` and all seven mouse directories. Run `./run_research.sh prepare` if preparation is needed.
5. **Select the session:** set `OUTPUT_DIR`, for example `export OUTPUT_DIR="$PWD/autoresearch-runs/<tag>"`. The evaluator creates `session.json` and `results.tsv`. Preserve existing records when resuming. Earlier AUC-based sessions stay archived; create a new session for accuracy-based comparisons.
6. **Run the baseline:** use the initial recipe before making changes. In an existing session, read `best.json` and restore the best recipe before continuing.

Documentation edits and project checks leave the research loop inactive.

## Experimentation

Run each experiment on a single GPU:

```bash
./run_research.sh experiment --description 'baseline' > run.log 2>&1
```

The budget is **300 seconds per mouse**, training seven separate models. Startup, preprocessing, and evaluation add overhead. The initial recipe also stops at 100 epochs. Keep the budget, seed, subjects, data identity, and runtime versions fixed within a session.

**You may change:** `train.py` and `models/*.py`, including preprocessing, architecture, optimizer, losses, regularization, batch size, precision, and the training loop. Dependencies and Docker recipes are editable; environment changes require a new session and matching baseline.

**Keep fixed:** the dataset, event eligibility, labels, raw window coordinates, and splits. Preserve `prepare.py`, `research_protocol.py`, `utils/mouse_lfp.py`, `utils/__init__.py`, `SETTINGS_MouseLFP.json`, `evaluate.py`, `protocol.lock.json`, and all prepared data.

The protocol remains **978 events across seven mice**, positives **5-1 minutes before onset**, **28-second windows at 500 Hz**, eight reference windows per class, and a **14-second training stride** yielding 23 windows per class. Train Days **0-7**, validation Days **8/11**, and test Days **12/13** stay fixed. Preserve the existing seizure exclusions and same-mouse, same-day negative assignments.

**The objective:** maximize `validation_accuracy`, the equal-weight mean of the seven mice's window-level validation accuracies. The fixed evaluator computes accuracy at threshold `0.5`; AUC remains a reported metric. Fit preprocessing statistics on training data only, predict every original validation window, and keep test data sealed during research.

Preserve the `fit`, `load_predictor`, and `predict_proba` interfaces in `train.py`. Save weights and inference statistics in `work_dir`; final evaluation restores them without retraining. Select epoch checkpoints by highest validation accuracy, using lower validation loss to break ties. README contains the full signatures.

**Resource use:** inspect GPU memory and elapsed time alongside the score. Prefer simple, maintainable improvements. Promotion requires an accuracy increase greater than `1e-6`.

## Output format

A successful experiment ends with these fields (example values):

```text
validation_accuracy: 0.750000000
status: keep
experiment: <experiment-id>
```

Extract the summary from the redirected log:

```bash
rg '^(validation_accuracy|status|experiment):' run.log
```

Per-mouse scores and resource usage are in `experiments/<id>/result.json`; histories and worker logs are under `X*/`. Completing all seven mice is required for promotion.

## Logging results

The evaluator appends results to **`$OUTPUT_DIR/results.tsv`** using these tab-separated columns:

```text
experiment	commit	source_sha256	validation_accuracy	peak_vram_mb	wall_seconds	status	description
```

The experiment ID locates source and model snapshots. Commit and source hash identify the recipe, including uncommitted edits. Failed runs have a blank accuracy. Memory is in MiB; elapsed time is in seconds. Status is `keep`, `discard`, or `crash`.

The first completed experiment is `keep`. Later scores must improve on the best by more than `1e-6`; otherwise they are `discard`. Failures preserve `best.json`. Let the evaluator write the ledger. Keep generated results and checkpoints outside version control.

## The experiment loop

Use the dedicated research branch.

**Repeat continuously once research is requested:**

1. Inspect the current branch, best recipe, and recent results.
2. Form a hypothesis and edit the research code, preferably changing one factor first.
3. Commit only the experiment changes, protecting unrelated user work.
4. Run `./run_research.sh experiment --description 'brief hypothesis' > run.log 2>&1`.
5. Read the score, inspect all seven mice, and check the recorded ledger row.
6. If **keep**, retain the recipe as the next starting point.
7. If **discard**, run `./run_research.sh restore` before the next experiment. It restores the best training/model source and backs up current code. Restore dependency edits separately and rebuild when needed.
8. If **crash**, read `status.json` and the affected mouse's `train.log`. Fix an implementation error and retry, or restore the best recipe and try another idea.
9. Repeat and report meaningful progress periodically.

**Timeouts:** each mouse process is terminated after its training budget plus 180 seconds of overhead. Inspect the worker log before retrying.

**Continue autonomously:** after research starts, proceed without routine permission requests for the next experiment. Stop promptly when the user requests it or the task budget is exhausted. When progress stalls, inspect learning curves, revisit near-misses, or try a different preprocessing or architecture hypothesis.

**Final testing:** when the user ends the search and requests testing, run `./run_research.sh finalize`. It loads the best source snapshot, weights, and preprocessing state, evaluates the fixed test split, and seals the session. Test scores never select a recipe.

Reference: [Karpathy autoresearch program.md](https://github.com/karpathy/autoresearch/blob/master/program.md).
