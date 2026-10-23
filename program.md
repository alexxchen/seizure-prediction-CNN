# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `meta-learning/auto-research`. Start from a clean, committed checkout and preserve unrelated changes separately. Set `export OUTPUT_DIR="$PWD/autoresearch-runs/<tag>"` and run `mkdir -p "$OUTPUT_DIR"`. Use a fresh output directory for this protocol and baseline; preserve old sessions.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` and `AGENTS.md` — repository context and fixed research rules.
   - `meta_learning/frontend_contract.py`, `meta_learning/model.py`, and `models/cnn.py` — the frontend boundary, wrapper, and fixed CNN. Do not modify.
   - `configs/meta.json` — fixed training controls and budget. Do not modify.
   - `research.py`, `run_research.sh`, and `protocol.lock.json` — execution, snapshots, and locked files. Do not modify.
   - `prepare.py` — fixed source preparation. Do not modify.
   - `meta_learning/frontend.py` — the file you modify. Waveform transforms, architecture, normalization, and initialization.
4. **Verify data exists**: Check that the version-3 source inventory exists. If not, run `./run_research.sh prepare`. Preserve private recordings, annotations, caches, and old sessions. Run `./run_research.sh check` before starting experiments.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for the **fixed training budget in `configs/meta.json`**: currently 50 outer steps and 40 complete inner epochs. The JSON overrides the Config defaults of 100 outer steps and 50 inner epochs; preserve its explicit overrides. You launch it simply as: `./run_research.sh experiment --description 'your idea'`.

**What you CAN do:**
- Modify `meta_learning/frontend.py` — this is the only file you edit. Everything inside the frontend is fair game: architecture, width, depth, kernels, spectral transforms, nonlinearities, sample-local normalization, and initialization. The initial RMS/GroupNorm ResNet is your baseline, not an architectural constraint.

**What you CANNOT do:**
- Modify fixed files or refresh `protocol.lock.json` to admit a candidate. Keep the CNN, training loop, optimizers, sample/crop policy, RNG, and evaluation fixed. A deliberate infrastructure change needs a separately reviewed protocol and a fresh session/baseline.
- Install new packages or add dependencies. You can only use what's already in `requirements.txt`. Do not add candidate files or load external weights.
- Modify the evaluation harness or data roles. Support Days 0-7 alone fit each fresh CNN and its BatchNorm statistics. Query Days 8/11 supply outer gradients; optional disjoint valid days select checkpoints only. Test Days 12/13 supply inference and cross-candidate development scores. Preserve full hypergradients, query-event weighting, complete SGD epochs, support-only BatchNorm recalibration, and outer AdamW.
- Use dropout, cross-sample or fitted statistics, mutable state, labels, recording metadata, filesystem/data access, training/evaluation imports, or detached learned operations inside your frontend.

Keep `WaveformFrontend(seed=0)` and the JSON-serializable `FRONTEND_CONFIG`. Accept centered/resampled/padded waveforms `(N, 28672)` and emit `(N, 64, 256)` for the fixed three-block, 32-channel CNN. Your features must be deterministic, sample-local, stateless, differentiable in waveform and learned parameters, finite for silence, and independent of chunk size and train/eval mode. Put additional transforms inside the frontend; the fixed wrapper handles RNG isolation and chunking.

**The goal is simple: get the lowest test_loss.** Compare equal-mouse-mean predictive log loss on Days 12/13 at each run's query/valid-selected checkpoint. Keep your first successful baseline; keep later candidates only for strictly lower loss and discard ties. Report accuracy at threshold 0.5 alongside loss. The human designated Days 12/13 as a development set: use their scores to rank completed candidates, never for gradients, CNN fitting, BatchNorm statistics, or within-run checkpoint selection. Report search and replay scores as development performance.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful test_loss gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting strictly better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 test_loss improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 test_loss improvement from deleting code? Definitely keep. For this study, simplicity does not override the strict loss rule: ties discard.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
test_loss:       0.650000
test_accuracy:   0.625000
selection_loss:  0.620000
peak_vram_mb:    1024.0
experiment:      <experiment-id>
```

Note that the script completes the fixed JSON budget, so runtime varies with your frontend. You can extract the key metric from the log file:

```
grep "^test_loss:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	test_loss	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. test_loss achieved (e.g. 0.650000) — use 0.000000 for crashes; exclude crash rows from comparisons
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	test_loss	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	wider frontend kernels
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

Compare full-precision losses from `experiments/<id>/result.json` before rounding for the TSV. Report threshold-0.5 accuracy alongside loss. The command saves execution status (`complete` or `crash`); you write `keep`, `discard`, or `crash` to the ledger. Preserve snapshots, attempt records, logs, and checkpoints under `$OUTPUT_DIR`, outside version control. Reset only your candidate commits on the dedicated study branch; preserve unrelated changes, the untracked ledger, and all artifacts. Never use `git clean` to discard experiments.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `meta_learning/frontend.py` with an experimental idea by directly hacking the code.
3. Run `./run_research.sh check`, then `git add meta_learning/frontend.py` and `git commit`.
4. Run the experiment: `./run_research.sh experiment --description 'brief hypothesis' > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^test_loss:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If test_loss improved (lower), you "advance" the branch, keeping the git commit
9. If test_loss is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: The watchdog defaults to 12,000 seconds per attempt. Keep it fixed within the study. If a run exceeds it, treat it as a failure (discard and revert). Preserve checkpoints and complete the fixed JSON budget on successful runs; partial runs cannot promote.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. How many experiments you complete depends on the frontend and the fixed meta-training budget. The user then wakes up to experimental results, all completed by you while they slept!

For this study, run one candidate at a time. Wait for it to finish, read its result or failure logs, record your decision, and keep or reset the commit before choosing your next hypothesis. Use the latest result and study history; do not queue or edit future candidates during a run. Honor any explicit limit or stop request from the human. A request to edit this workflow does not itself start training.

Resume compatible saved code with `./run_research.sh experiment --resume <experiment-id>`; preserve its saved optimizer, BatchNorm, RNG, history, and selection state. To change code, commit the fix and start a new experiment. Replay saved weights without fitting with `./run_research.sh replay --experiment <experiment-id> --split test`; use `query` or `valid` for the other roles. `valid` requires selection days. Resolve your kept commit to its saved `experiment.json`. Describe signal-quality audits by their actual coverage; synthetic checks cover the frontend contract.
