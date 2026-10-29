# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current branch. Set `$env:OUTPUT_DIR` to a fresh `autoresearch-runs/<tag>` directory and create it. Preserve unrelated changes and previous studies.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` and `AGENTS.md` — repository context and fixed research rules.
   - `prepare.py`, `research.py`, and `protocol.lock.json` — fixed data preparation, execution, evaluation, and edit boundaries. Do not modify.
   - `meta_learning/frontend.py`, `models/cnn.py`, and `configs/meta.json` — the files you modify. Frontend/CNN architecture and allowed optimization settings.
4. **Verify data exists**: Check the prepared source inventory described in `README.md`. If missing, run `.\run_research.ps1 prepare`. Run `.\run_research.ps1 check` before experiments.
5. **Initialize results.tsv**: Preserve any previous ledger with its study, then create the untracked `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once the user has requested the study, kick off the experimentation. Editing this document does not start training.

## Experimentation

Each experiment runs on a single GPU. The training script completes the **declared budget in `configs/meta.json`** (`outer_steps` and `inner_epochs`). You launch it simply as: `.\run_research.ps1 experiment --timeout-seconds 172800 --description "your idea"`. On Linux, use `./run_research.sh` with the same arguments.

**What you CAN do:**
- Modify `meta_learning/frontend.py` and `models/cnn.py` — model architecture, initialization, and waveform transforms, within the interfaces and model constraints in `AGENTS.md`.
- Modify only the fields listed in `protocol.lock.json` under `editable_config_fields` in `configs/meta.json`.

**What you CANNOT do:**
- Modify fixed files or locked configuration fields. Data loading, crop policy, solvers, normalization, and evaluation remain fixed. Do not refresh `protocol.lock.json` to admit a candidate; infrastructure changes require a separately reviewed protocol and fresh baseline.
- Install new packages or add dependencies. You can only use what's already in `requirements.txt`.
- Modify the evaluation harness or data roles. Support Days 0-7 alone fit each fresh CNN and its BatchNorm statistics; query Days 8/11 supply outer frontend gradients. Preserve full hypergradients and query/valid checkpoint selection. Test Days 12/13 never supply gradients, CNN fitting, statistics, or within-run checkpoint selection.

**The goal is simple: get the lowest test_loss.** Compare equal-mouse-mean log loss on Days 12/13 at each run's query/valid-selected checkpoint, and report threshold-0.5 accuracy alongside it. These days are the user-designated development set. Every candidate must finish its declared budget; record its complete configuration and runtime. Different budgets are not equal-compute comparisons.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful test_loss gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting strictly better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 test_loss improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 test_loss improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep only if the full-precision loss is strictly lower; ties discard.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
test_loss:       0.650000
test_accuracy:   0.625000
selection_loss:  0.620000
peak_vram_mb:    1024.0
experiment:     <experiment-id>
```

The script completes the declared JSON budget, so runtime depends on the candidate. Compare full-precision metrics from `$env:OUTPUT_DIR/experiments/<experiment-id>/result.json`. You can extract the key metric from the log file:

```
Select-String -Path run.log -Pattern "^test_loss:"
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	test_loss	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. test_loss achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	test_loss	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`). Honor explicit user Git instructions: if commits are disabled, retain the immutable candidate snapshot, record its experiment ID with the base commit, and restore only candidate edits on discard. Preserve unrelated changes and all artifacts; never use `git clean`.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune the frontend, CNN, and/or allowed optimization fields with an experimental idea by directly hacking the code.
3. Run `.\run_research.ps1 check`, then git commit unless the user requested otherwise
4. Run the experiment: `.\run_research.ps1 experiment --timeout-seconds 172800 --description "your idea" > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `Select-String -Path run.log -Pattern "^test_loss:|^peak_vram_mb:"`
6. If the summary is missing, check the saved execution status and run `Get-Content run.log -Tail 50` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. Keep the first completed baseline. If test_loss improved (lower), you "advance" the branch, keeping the git commit
9. If test_loss is equal or worse, restore only your candidate changes/commits to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: The current study declares a 172800-second watchdog per attempt; keep it fixed within the study. If a run exceeds it, treat it as a failure (discard and revert). Incomplete runs cannot promote; do not silently reduce the training budget.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. How many experiments complete depends on each candidate's declared budget and runtime. The user then wakes up to experimental results, all completed by you while they slept!

Run one candidate at a time. Wait for completion, analyze and log its outcome, and keep or discard it before choosing the next hypothesis. Do not queue experiments or edit future candidates during a run. Honor any explicit user limit or stop request.
