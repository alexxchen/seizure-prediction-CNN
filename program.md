# autoresearch

This experiment lets an LLM search for a better waveform frontend for seizure prediction.

## Setup

To start a study:

1. **Choose a run tag and budget**: use a short tag such as `oct22`. Training needs a user-requested finite candidate count or explicit finite job. Editing the scaffold alone does not authorize training.
2. **Set up the branch and directory**: use `meta-learning/auto-research` or a dedicated study branch based on it. Inspect the working tree and preserve unrelated changes. Set `export OUTPUT_DIR="$PWD/autoresearch-runs/<tag>"` and run `mkdir -p "$OUTPUT_DIR"`. Use a fresh directory.
3. **Read the in-scope files**:
   - `README.md` and `AGENTS.md` — project context and fixed research rules.
   - `meta_learning/frontend.py` — the file you modify.
   - `meta_learning/frontend_contract.py`, `meta_learning/model.py`, and `models/cnn.py` — candidate interface, wrapper, and fixed CNN.
   - `configs/meta.json` — training controls and budget.
   - `research.py`, `run_research.sh`, and `protocol.lock.json` — execution, evaluation, logging, and locked files.
4. **Check the data**: verify that the version-3 source inventory exists. Run `./run_research.sh prepare` if it is missing. Preserve private recordings, annotations, caches, and old sessions.
5. **Check the setup**: run `./run_research.sh check`. The engine creates `results.tsv` and `best.json` when experiments run; leave their contents to the engine.
6. **Start the authorized job**: keep the same output directory, configuration, dependencies, runtime, and watchdog for the study. A changed protocol requires separate review, a fresh session, and a new baseline.

## Experimentation

Each candidate runs the **fixed training budget in `configs/meta.json`**. Launch it with `./run_research.sh experiment --description 'your idea'`. The launcher runs in the foreground using an immutable source snapshot. Read the actual JSON before starting and preserve its explicit overrides: inner/per-mouse fitting uses 40 complete epochs, overriding the Config default of 50. The remaining overrides are documented in README.md.

**What you CAN do:**

- Modify `meta_learning/frontend.py` — the only candidate file. Explore architecture, width, depth, kernels, spectral transforms, nonlinearities, sample-local normalization, and initialization. The initial RMS/GroupNorm ResNet is the baseline; all frontend internals are open.

**What you CANNOT do:**

- Modify fixed files or refresh `protocol.lock.json` to admit a candidate. Keep the CNN, training loop, optimizer controls, sample/crop policy, RNG, and evaluation unchanged.
- Install packages, add files, change dependencies, or load external weights.
- Use dropout, cross-sample or fitted statistics, mutable state, labels, recording metadata, filesystem/data access, training/evaluation imports, or detached learned operations inside the candidate.
- Change data roles or selection rules. Support Days 0-7 alone fit each fresh CNN and its BatchNorm statistics. Query Days 8/11 supply outer gradients; optional disjoint valid days supply selection only. Test Days 12/13 supply inference and cross-candidate scores. Every outer step includes every configured mouse once with the fixed query-event weighting. Preserve full hypergradients, SGD epochs, final support BatchNorm recalibration, and outer AdamW.

Keep `WaveformFrontend(seed=0)` and the JSON-serializable `FRONTEND_CONFIG`. Accept centered/resampled/padded waveforms `(N, 28672)`; emit `(N, 64, 256)` for the fixed three-block, 32-channel CNN. Features must be deterministic, sample-local, stateless, differentiable in waveform and learned parameters, finite for silence, and independent of chunk size and train/eval mode. Put additional transforms inside the frontend; the fixed wrapper handles RNG isolation and chunking.

**The goal: get the lowest `test_loss`.** Score equal-mouse-mean predictive log loss on Days 12/13 at the run's query/valid-selected checkpoint. The first successful baseline is kept. A later candidate is kept only when loss is strictly lower; ties discard. Report accuracy at threshold 0.5 alongside loss. Test scores rank completed candidates; they never select checkpoints within a run. These days are the user-designated development set, so search and final replay report development performance.

**VRAM**: inspect peak memory and runtime alongside loss. Keep the configured training schedule even when a frontend is larger or slower.

**Simplicity criterion**: prefer clear, compact ideas when choosing experiments. Small gains should motivate further simplification. The engine's keep/discard decision still follows strict loss improvement.

**The first run**: establish the baseline by running the existing frontend unchanged.

## Output format

A completed experiment prints a summary like this (example values):

```text
test_loss: 0.650000
test_accuracy: 0.625000
selection_loss: 0.620000
status: keep
experiment: <experiment-id>
```

Extract the score from the log:

```bash
rg '^(test_loss|test_accuracy|selection_loss|status|experiment):' "$OUTPUT_DIR/run.log"
```

The experiment's `result.json` also records selection role/accuracy, selected step, peak VRAM, and wall time. Use the test score at that selected step. Histories, per-mouse predictions, sample/event counts, class ratios, window settings, source fingerprints, AUC, sensitivity, and specificity are available in the saved artifacts. Describe quality audits by their actual coverage; synthetic checks cover the frontend contract.

## Logging results

After each experiment, the engine updates `$OUTPUT_DIR/results.tsv`. It is tab-separated and has 12 columns:

```text
experiment	commit	frontend_sha256	selection_role	selection_loss	selection_accuracy	test_loss	test_accuracy	peak_vram_mb	wall_seconds	status	description
```

1. `experiment`, `commit`, `frontend_sha256`: experiment and source identities.
2. `selection_role`, `selection_loss`, `selection_accuracy`: query/valid checkpoint selection.
3. `test_loss`, `test_accuracy`: development-test score at that checkpoint.
4. `peak_vram_mb`, `wall_seconds`: memory and runtime.
5. `status`: `keep`, `discard`, or `crash`.
6. `description`: a short English description of the idea.

Let the engine maintain the ledger and `best.json`. Preserve failed/discarded results, immutable snapshots, logs, checkpoints, and attempt records. Keep generated artifacts and research sessions out of version control.

## The experiment loop

Work on the study branch and repeat within the user-authorized count or execution budget:

1. Inspect the branch, current frontend, saved best, and recent results.
2. Edit `meta_learning/frontend.py` with one experimental idea.
3. Run `./run_research.sh check`; fix contract failures before training.
4. Run `./run_research.sh experiment --description 'brief hypothesis' > "$OUTPUT_DIR/run.log" 2>&1`. Redirect output so logs stay out of the conversation.
5. Read the summary with `rg '^(test_loss|test_accuracy|selection_loss|status|experiment):' "$OUTPUT_DIR/run.log"`.
6. If the command fails or the summary is missing, read `tail -n 50 "$OUTPUT_DIR/run.log"`, then the saved `contract.log` and `train.log`. Diagnose before retrying.
7. Inspect the saved result and per-mouse metrics. The engine records the outcome in the TSV and updates the best candidate.
8. If loss improved, keep the frontend and build on it.
9. If loss tied, worsened, or the idea failed, run `./run_research.sh restore`. It backs up and restores only the frontend. Preserve unrelated work and all experiment artifacts.

Try ideas, keep improvements, and build the next experiment on what worked. If ideas run short, re-read the frontend and recent results, combine promising changes, or try a different architecture.

**Timeout**: the default watchdog is 86,400 seconds per attempt. Keep it fixed within the session. Timeout records `crash` and preserves checkpoints; partial runs cannot promote. The watchdog never replaces complete epochs. Use separate explicit short jobs for smoke checks.

**Crashes**: fix simple frontend errors and retry within the authorized budget. Skip fundamentally broken ideas. Resume a compatible interrupted experiment with `./run_research.sh experiment --resume <experiment-id>`; it uses its immutable code and restores saved optimizer, BatchNorm, RNG, history, and selection state.

**KEEP GOING WITHIN THE BUDGET**: once authorized, continue through the finite job without asking after each candidate. Report progress, honor stop requests, and stop when the requested count or execution budget is reached.

Replay saved weights without fitting with `./run_research.sh replay --split test`; use `query` or `valid` for the other roles. `valid` requires selection days. When the user requests study completion, run `./run_research.sh finalize` to replay the best development-test score and seal the session.
