# autoresearch

An LLM conducts its own research on how private continuous MouseLFP recordings and seizure annotations become inputs to the existing CNN.

## Setup

When the user requests experiments, set up the research session:

1. **Choose a run tag**: use a date-based tag, such as `oct13-data`. For a fresh run, both `autoresearch/<tag>` and `autoresearch-runs/<tag>` must be unused. Resume an existing data-pipeline session only when its scope, sources, controls, and runtime match. Keep earlier window-cache and model-search sessions archived.
2. **Create the branch**: inspect the branch and working tree, protect existing user changes, then use `git checkout -b autoresearch/<tag>` from the current project state. Preserve unrelated work and previous experiment records.
3. **Read the in-scope files**: read these files before choosing a hypothesis:
   - `README.md` and `AGENTS.md` — project context and research rules.
   - `prepare.py` and `source_data.py` — fixed source inventory, annotation alignment, recording access, and day splits.
   - `sample_pipeline.py` — editable event eligibility, sample construction, exclusions, window policy, negative selection, and cleaning.
   - `input_pipeline.py` — editable resampling, features, normalization, frequency selection, and input layout.
   - `train.py`, `models/cnn.py`, and `models/__init__.py` — fixed CNN and training controls.
   - `evaluate.py`, `protocol.lock.json`, and `SETTINGS_MouseLFP.json` — fixed evaluation, source identity, snapshots, and session management.
4. **Verify data exists**: check the recordings under `/data/users/jianchen/data/seizure_data` and the manifest under `/data/users/jianchen/data/seizure_data-cache/autoresearch-source-v1`. Run `./run_research.sh prepare` if preparation is needed.
5. **Initialize the results location**: set `OUTPUT_DIR="$PWD/autoresearch-runs/<tag>"`. The evaluator creates `session.json` and initializes `results.tsv` when the first experiment runs. Preserve the ledger when resuming; inspect `best.json` and use `./run_research.sh restore` before continuing an existing session.
6. **Begin the requested research**: verify setup, report the selected session, and establish the baseline. The user's request to run experiments authorizes the loop without another routine confirmation. Documentation edits and project inspections leave training inactive.

## Experimentation

Use one GPU for each experiment. The evaluator trains seven mice sequentially, with a **300-second training allowance per mouse** and a fixed 100-epoch cap. Data assembly and feature construction precede the training timer; validation and checkpoint work during training count toward the allowance. Launch a candidate with:

```bash
./run_research.sh experiment --description 'baseline shared data recipe' > run.log 2>&1
```

**What you CAN do:**

- Edit `sample_pipeline.py` and `input_pipeline.py`. Explore event eligibility, positive and negative definitions, prediction intervals, exclusion margins, recording coverage, class ratio, window duration, stride, cleaning, filtering, resampling, representation, normalization, frequency selection, and layout.
- Apply each candidate's sample definitions, window size, cleaning, and transformations consistently to training, validation, and test. Fit population statistics and adaptive preprocessing parameters on training data only, then reuse them for validation and test through the fixed predictor interface.
- Map different windows and representations to `(1, 1, 55, 114)`. Check physical frequency selection and normalization axes when resampling, transforms, or layouts change.

**What you CANNOT do:**

- Alter the private recordings or source annotations in `seizure_epochs_pitime_seizure_w15_t0.3_min5_gap10.parquet`.
- Change training Days 0-7, validation Days 8/11, or test Days 12/13. The modeled mice remain X1/X2/X4/X9/X10/X14/X15.
- Edit `train.py`, `models/cnn.py`, or `models/__init__.py`. Preserve the CNN, Adam at `5e-4`, batch size 32, label smoothing `0.1`, seed, training budget, epoch cap, and checkpoint rule.
- Edit `prepare.py`, `source_data.py`, `evaluate.py`, `protocol.lock.json`, `SETTINGS_MouseLFP.json`, or supporting legacy data helpers during research. The fixed evaluator defines scoring and promotion.
- Install packages, add dependencies, or change the runtime within a comparison session. An explicitly requested environment change requires a new session and matching baseline.
- Compute test metrics or use test outcomes for recipe selection during the search. Final testing requires the user to end research and request `finalize`.

**Objective: maximize `validation_accuracy`.** It is the equal-weight mean of seven mice's validation accuracies at threshold `0.5`. AUC, sensitivity, specificity, and loss provide supporting evidence. Checkpoints prefer higher accuracy, breaking ties with lower validation loss. The evaluator promotes the first successful baseline and subsequent candidates that improve mean accuracy by more than `1e-6`.

The historical 978-event cohort and 28-second windows are editable derived choices. The starting sampler combines eight reference offsets with a 14-second grid over the 5-1-minute positive interval, yielding 23 windows per event at default settings. It matches same-mouse/day negatives with ten-minute seizure margins and uses the existing event-history exclusions. Inspect actual offsets and selected dates when describing the recipe. The starting input uses 256 Hz resampling, log-STFT, and training-fitted normalization.

Changing eligibility, label horizons, exclusions, or class ratios changes the evaluated population or task. Identify that change alongside the score and fingerprint. Attribute gains to input processing only when the sample and label definitions support that comparison.

**VRAM:** allow moderate increases for useful validation gains. Track peak allocation and favor recipes that execute reliably.

**Simplicity criterion:** prefer clear, compact recipes. Weigh a gain against added branches, special cases, and maintenance cost; removing unnecessary processing is a useful hypothesis. The fixed promotion rule still applies: equal-scoring simplifications receive `discard`. Record their simplicity benefit without overriding the evaluator's status or `best.json`.

**The first run:** establish a baseline with the current shared recipe before candidate edits. A resumed session starts from its saved best. Use all seven mice and the session's fixed training allowance for comparable results; keep smoke checks in separate sessions.

## Output format

A successful run ends with a summary such as:

```text
validation_accuracy: 0.750000000
status: keep
experiment: <experiment-id>
```

Extract the summary from the redirected log:

```bash
rg '^(validation_accuracy|status|experiment):' run.log
```

`$OUTPUT_DIR/experiments/<experiment-id>/result.json` contains the mean score, per-mouse metrics, fingerprints, and peak VRAM. Each `X*/` directory contains sampling configuration, event selection, split counts, training history, checkpoint, validation predictions, and worker logs. Inspect per-mouse and per-day behavior alongside the mean; overlapping windows from one event share evidence.

## Logging results

The evaluator appends completed and failed attempts to `$OUTPUT_DIR/results.tsv`. It uses tabs and these eight columns:

```text
experiment	commit	source_sha256	validation_accuracy	peak_vram_mb	wall_seconds	status	description
```

1. Experiment identifier locating the source snapshot and artifacts.
2. Git commit identifier.
3. SHA-256 digest of the saved research source files.
4. Mean validation accuracy; failed runs leave this field empty.
5. Peak allocated GPU memory in MiB; failed runs may retain the default zero.
6. Total experiment wall time in seconds, including preprocessing and evaluation.
7. Evaluator status: `keep`, `discard`, or `crash`.
8. Short English hypothesis description.

Example with illustrative identifiers and source digests; the crash row has an empty accuracy field:

```text
experiment	commit	source_sha256	validation_accuracy	peak_vram_mb	wall_seconds	status	description
baseline-id	a1b2c3d	<digest>	0.680000000	1200	2300	keep	baseline shared data recipe
candidate-id	b2c3d4e	<digest>	0.700000000	1250	2350	keep	remove redundant positive offsets
discard-id	c3d4e5f	<digest>	0.670000000	1300	2380	discard	transpose the input representation
crash-id	d4e5f6a	<digest>		0	480	crash	feature construction exceeded the worker allowance
```

Let the evaluator write the ledger; inspect its row instead of appending a duplicate or changing its status. Keep generated logs, caches, checkpoints, and results outside version control. Each experiment snapshots both editable pipelines, the fixed trainer/CNN, `requirements.txt`, and `Dockerfile`.

Report each recipe with its score, split sample counts, class ratio, eligible event counts, window duration, actual offsets or stride policy, prediction interval, exclusion settings, and evaluation-protocol fingerprint. Identify task-definition changes. Use `sampling_summary.json`, `event_selection.csv`, `sample_config.json`, and validation predictions as evidence. Record audit findings and interpretation in English with the experiment artifacts.

## The experiment loop

Conduct research on the dedicated branch, such as `autoresearch/oct13-data`.

LOOP FOREVER once the user has requested experiments:

1. Inspect the branch, commit, working tree, saved best recipe, sampling summaries, and recent results. Check that the starting pipeline matches the selected best.
2. Form a data construction, cleaning, representation, normalization, or layout hypothesis. Edit `sample_pipeline.py` and/or `input_pipeline.py`, sharing processing across all splits. Check sample boundaries, labels, coverage, shapes, and training-only fitting before launching.
3. Commit only the candidate's pipeline changes. Preserve unrelated user changes and keep generated experiment artifacts untracked.
4. Run `./run_research.sh experiment --description 'brief hypothesis' > run.log 2>&1`. Redirect training output to the log so it stays out of the conversation context.
5. Read the summary with `rg '^(validation_accuracy|status|experiment):' run.log`. Inspect the result JSON, per-mouse metrics, histories, sample counts, event eligibility, and fingerprints.
6. If the summary is missing or the process failed, inspect `tail -n 50 run.log`, the experiment's `status.json`, and the affected mouse's `train.log`. Repair clear implementation errors and retry; abandon a broken idea after a few unsuccessful fixes.
7. Verify the evaluator's ledger row and preserve the recipe details. For a failure before an experiment record is created, retain the launch log and diagnose setup before retrying. Keep the ledger and experiment artifacts untracked.
8. For `keep`, retain the candidate as the next starting point. Explain the gain with its data definition, including changes to the label horizon or evaluated cohort.
9. For `discard`, run `./run_research.sh restore`. It backs up the current editable files and restores both pipelines from the best snapshot. Recover from broken candidates the same way. Preserve failed attempts, snapshots, the ledger, fixed controls, and unrelated user work; use file-scoped git updates to align the research state with the restored recipe.

Continue from the best recipe and retain candidates promoted by the evaluator. Revisit earlier alternatives sparingly and use previous results to select the next hypothesis.

**Timeout:** each mouse receives its training allowance plus 180 seconds for startup, sample assembly, preprocessing, and evaluation: 480 seconds at the default budget. The evaluator terminates overdue workers and records `crash`. Inspect the worker log before retrying. A full-cohort run can take roughly 35 minutes of training plus overhead; apply the watchdog per mouse rather than a ten-minute limit to the whole experiment.

**Crashes:** fix straightforward import, sample-coordinate, or shape errors and rerun. For OOM, impossible sample coverage, or an impractical representation, restore the best pipelines and try another hypothesis. Preserve failure records and continue searching.

**NEVER STOP:** after the requested loop starts, continue autonomously. Avoid routine requests to continue or choose a stopping point. If ideas run out, reread the pipelines, inspect source-quality findings and per-mouse errors, consult research referenced in the repository, combine promising ideas, or test a different representation. Continue until the user stops research or an explicit task or execution budget is reached. Honor stop requests immediately.

The user may leave the agent running while away. Plan experiment throughput around seven sequential training allowances and report meaningful progress as results arrive.

**Final testing:** when the user ends the search and requests `finalize`, run `./run_research.sh finalize` with the selected session's `OUTPUT_DIR`. The evaluator loads the best saved source snapshot and checkpoints, constructs test samples with that recipe, reuses training-fitted statistics, writes test predictions and `final.json`, and seals the session. Interrupted final evaluation can resume. Test metrics stay outside recipe selection.
