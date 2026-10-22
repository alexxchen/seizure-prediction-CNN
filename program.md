# Frontend autoresearch

Explore which shared frontend gives the best development-test performance with
the fixed meta-learning task CNN. Read README.md and AGENTS.md first.

## Authorization and setup

Building this scaffold does not authorize training. Start research only after the
user requests a finite number of candidates or an explicit finite training job.
Honor that count and stop requests. Never start an unbounded experiment loop.

1. Inspect the branch and working tree. Use `meta-learning/auto-research` or a
   dedicated study branch based on it. Preserve unrelated work and old artifacts.
2. Read `meta_learning/frontend.py`, `frontend_contract.py`, `model.py`,
   `models/cnn.py`, `configs/meta.json`, `research.py`, and `protocol.lock.json`.
   Only `meta_learning/frontend.py` may change between candidates.
3. Run `./run_research.sh check`. Review the actual coverage of these synthetic
   checks; they do not audit private recordings or prove absence of leakage.
4. Choose a fresh `OUTPUT_DIR="$PWD/autoresearch-runs/<study>"`. Prepare sources
   with `./run_research.sh prepare` if the version-3 source inventory is missing.
   Do not reuse old single-channel or other-protocol research sessions.
5. Establish a baseline with the current frontend before any candidate edits.
   Every candidate uses the exact current `configs/meta.json`: X1/X2/X9/X14,
   100 outer steps, 40 inner/per-mouse epochs, batch sizes 64/64, support/query
   crops 4/64, and the other explicit JSON overrides documented in README.md.

## Editable frontend

All frontend internals are open: convolutional or other differentiable
architectures, width/depth, receptive fields, spectral transforms, nonlinearities,
sample-local normalization, and random initialization. The initial 131,296-parameter
RMS/GroupNorm ResNet is a baseline, not the allowed architecture family.

Keep the public candidate interface:

```python
FRONTEND_CONFIG = {"architecture": "...", "input_normalization": "..."}

class WaveformFrontend(torch.nn.Module):
    def __init__(self, seed=0):
        ...  # CPU initialization; reproducible from seed; no external weights

    def forward(self, waveform):
        ...  # (N, 28672) -> finite (N, 64, 256)
```

`FRONTEND_CONFIG` must be JSON serializable and accurately describe the candidate.
The fixed wrapper owns initialization RNG isolation, frontend window chunks,
RNG-preserving activation checkpointing, parameter counting, and outer decay
groups. Do not add files or dependencies. Standard-library and existing pinned
numerical/PyTorch dependencies are available.

Input is already centered, resampled to 256 Hz, and zero-padded. Put additional
transforms inside the candidate. Features must use only their own waveform and
learned frontend parameters. Keep features deterministic in train/eval mode,
finite for silent windows, and differentiable through the fixed inner fit. Do not
detach learned operations, read files/labels/recording metadata, import training or
evaluation code, use cross-sample normalization, mutate persistent statistics, or
load pretrained weights. The fixed CNN always consumes exactly 64 feature
channels and 256 time bins. Its three convolution blocks emit 32 channels;
internal frontend representations may differ.

## Fixed controls and objective

Keep the sample recipe, source inventory, train Days 0-7, query Days 8/11,
optional disjoint selection days, and test Days 12/13. Every outer step includes
each configured mouse once, with its eligible query-event weight. Only query
labels supply the outer gradient. Only support supplies CNN SGD updates and
BatchNorm recalibration. Preserve the complete differentiable inner trajectory,
plain SGD, fixed unlearned CNN reference, outer AdamW, crop policy/RNG, full epoch
budget, checkpoint selection, and inference behavior.

**Minimize `test_loss`: equal-mouse-mean ordinary predictive log loss on Days
12/13 at the run's query/valid-selected checkpoint.** Lower is better; strictly
lower promotes, ties discard. The first successful candidate is kept. Report
test accuracy at 0.5, AUC, sensitivity, specificity, runtime, and VRAM alongside
loss. Changing test metrics cannot choose checkpoints inside an individual run.
Test scores choose frontend candidates across completed runs.

The user explicitly made Days 12/13 a development set for this search. Do not
describe the selected score or final replay as untouched generalization. There
is no independent held-out set in this protocol.

Do not edit `protocol.lock.json` to admit a candidate change. Deliberate fixed
infrastructure changes require review, a new protocol, a fresh session, and a
matching baseline. Do not install packages or change the runtime in a session.

## Finite experiment loop

For each candidate remaining in the authorized count:

1. Inspect the saved best, recent results, per-mouse scores, and current code.
   Restore the best before building on it when needed.
2. Form one English hypothesis and edit only `meta_learning/frontend.py`.
3. Run `./run_research.sh check`. Repair contract failures before real training.
4. Run one candidate in the foreground, redirecting output:

   ```bash
   ./run_research.sh experiment --description 'brief frontend hypothesis' > candidate.log 2>&1
   rg '^(test_loss|test_accuracy|selection_loss|status|experiment):' candidate.log
   ```

5. Inspect `experiments/<id>/result.json`, `train.log`, histories, selected
   checkpoint, per-mouse predictions, sampling/event counts, fingerprints, runtime,
   and VRAM. Compare development-test metrics at the selected step, not the best
   test step in a history. Keep signal-quality coverage claims tied to actual audits.
6. Let the evaluator maintain `results.tsv` and `best.json`. Do not relabel failed
   or discarded attempts. Keep successful improvements as the next starting point.
7. For a discard or broken candidate, run `./run_research.sh restore`. It backs up
   the current file before restoring only the best frontend. Preserve all logs,
   source snapshots, interruptions, checkpoints, and unrelated user changes.
8. Report progress and stop when the requested count or execution budget is met.

The fixed 86,400-second watchdog is a maximum per attempt, not an epoch/time
tradeoff. A timeout records `crash` and preserves the last checkpoint; it never
promotes a partial run. Keep the same watchdog within a session. Larger/smaller
explicit watchdogs require a fresh session. Synthetic tests use smaller fixed
budgets independently; never substitute them into production comparisons.

## Recovery, replay, and completion

Interrupted experiments can resume from their own immutable code snapshot:

```bash
./run_research.sh experiment --resume <experiment-id>
```

The existing engine restores compatible frontend/outer optimizer, complete task
BatchNorm state, RNG streams, histories, selection, and epoch budget. Current
working frontend changes cannot change that saved experiment. Mismatched sources,
configuration, runtime, or protocol require a new session.

Replay one role without training:

```bash
./run_research.sh replay --split query
./run_research.sh replay --split test
```

`valid` requires configured selection days. `./run_research.sh finalize` replays
the best candidate's saved weights on test and seals the study. Finalization uses
the same development data already used for search; it does not create a new
independent evaluation. Run it only when the user requests study completion.
