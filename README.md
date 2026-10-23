# MouseLFP frontend autoresearch

Learn a shared waveform frontend for predicting seizure-related labels across recording days of the same mouse. Only frontend weights receive outer updates. Every task and per-mouse evaluation fits a fresh CNN from a fixed, unlearned reference.

## Autoresearch workflow

This branch adapts `seizure-single-channel-auto-research` to shared-frontend meta-learning. The research agent edits only `meta_learning/frontend.py`; the task CNN, fixed reference initialization, SGD fitting, outer solver, data/crops, budgets, and evaluation code are locked. The frontend's entire internal design is editable, including learned or spectral transforms, architecture, sample-local normalization, and initialization. Its output must remain `(N, 64, 256)` so the CNN's architecture and seeded starting weights stay identical.

**Objective: minimize the equal-mouse-mean test log loss on Days 12/13.** The user explicitly designated these days as a development set for frontend research. Accuracy at threshold 0.5, AUC, sensitivity, and specificity are supporting metrics. Each candidate first selects its checkpoint by the existing query/valid log-loss rule; the candidate's development-test score at that checkpoint controls `keep`/`discard`. Test labels never supply training gradients, CNN updates, or within-run checkpoint selection. The first successful baseline is kept; subsequent loss must be strictly lower. Search results and final replay are development results, not an untouched generalization estimate.

Read `program.md` for the agent loop, following [karpathy/autoresearch](https://github.com/karpathy/autoresearch/blob/master/program.md): once the user starts a study, the agent continues experimenting until interrupted. Each candidate is committed before training; improvements retain their commits, and failed/discarded candidates are logged before resetting to their parent. The agent initializes and appends the untracked repository-root `results.tsv` with five columns: `commit`, `test_loss`, `memory_gb`, `status`, and `description`. The runner supplies metrics; the agent decides `keep`/`discard`/`crash` and manages Git history. Editing the workflow does not start a study.

The loop is sequential and driven by feedback: run one candidate, wait for its result, let the agent analyze and record the outcome, retain or reset the commit, then choose the next hypothesis. Each runner invocation returns control to the agent after one experiment. Future candidates are chosen after reviewing the latest result; the agent does not queue variants or edit the next candidate during a run.

Start each study on a fresh `autoresearch/<tag>` branch from a clean, committed `meta-learning/auto-research` checkout. A typical candidate iteration is:

```bash
export OUTPUT_DIR="$PWD/autoresearch-runs/<tag>"
parent=$(git rev-parse HEAD)
# Edit only meta_learning/frontend.py.
./run_research.sh check
git add meta_learning/frontend.py
git commit -m 'frontend hypothesis'
./run_research.sh experiment --description 'frontend hypothesis' > "$OUTPUT_DIR/run.log" 2>&1
# Wait for completion, then let the agent read and analyze the result.
# Append its score and keep/discard/crash decision to results.tsv.
```

Retain an improved candidate's commit. For `discard` or an abandoned `crash`, use `git reset --hard "$parent"` after recording the result and verifying the working tree is clean. Then choose the next hypothesis from the result.

Initialize the output directory and ledger, prepare data if needed, and run the unchanged frontend baseline before testing ideas; see `program.md` for setup. Compare full-precision losses from saved results. The branch tip tracks the best kept frontend; the ledger records all attempts, including discarded/crashed commits. Threshold-0.5 accuracy accompanies loss in progress reports and detailed artifacts. Reset only the current candidate's own commits on the dedicated study branch, preserving unrelated changes, untracked results, and all historical artifacts.

All research commands run in the foreground. `--dry-run` prints a launcher command without creating directories or running Docker. `--timeout-seconds` defaults to 12,000 seconds per candidate attempt; it is a failure watchdog, never early stopping. The configured 50 outer steps and 40 complete inner epochs stay fixed across candidates. Use a new session for a different protocol, source inventory, runtime, or watchdog. A study uses the four configured mice X1/X2/X9/X14.

`protocol.lock.json` checks fixed file hashes and rejects extra executable modules in controlled directories. Synthetic `check` verifies waveform/output shape, initialization RNG preservation, reproducibility, silence, sample/chunk independence, persistent state, and frontend gradients. These checks detect common implementation mistakes; candidates are trusted research code and the checks are not a security sandbox.

Each candidate executes an immutable copied source tree, so later working-tree edits cannot change a running experiment. Artifacts include `session.json`, `experiments/<id>/experiment.json`, `snapshot.json`, `contract.log`, `train.log`, `result.json`, immutable per-attempt records in `attempts/`, and the training artifacts described below. `result.json` contains both query/valid selection metrics and development-test metrics, plus execution status (`complete` or `crash`). Crashes preserve snapshots, logs, and checkpoints. The runner leaves the agent's `results.tsv` untouched and maintains no cross-candidate best record. Resume the same candidate from its saved source with:

```bash
./run_research.sh experiment --resume <experiment-id>
```

Replay a specific experiment's saved code and CNN/BatchNorm weights without fitting:

```bash
./run_research.sh replay --experiment <experiment-id> --split test
```

Choose `query`, `valid`, or `test`; `valid` requires selection days. The ledger's commit identifies the candidate in its saved `experiment.json`. Study continuation uses Git history and the ledger; stopping leaves the session open.

This infrastructure change establishes protocol version 2 for continuous agent research and Git/TSV result management. Its lock records the current committed runtime and launcher updates (PyTorch 2.14.1/CUDA 12.6 and 50 outer steps), alongside the new workflow. Start a fresh session and baseline under this protocol. Existing sessions and snapshots retain their original files and remain available through their original code/runtime.

Local equivalents use the existing pinned dependencies:

```bash
python research.py check
python research.py experiment --run-dir autoresearch-runs/local --data-dir /path/to/raw \
  --prepared-dir /path/to/prepared --cache-dir /path/to/cache --device cuda --description baseline
```

The remaining sections describe the fixed trainer and the initial frontend baseline. Architectural statements about this baseline do not restrict subsequent frontend candidates.

## Data and roles

Supported mice are X1/X2/X4/X9/X10/X14/X15. The current configuration trains X1/X2/X9/X14.

| Role | Days | Use |
| --- | --- | --- |
| Support | 0-7 | Inner CNN updates and support-only BatchNorm calibration |
| Query | 8/11 | Outer frontend gradients and current checkpoint selection |
| Valid | Optional disjoint subset of Days 8/11 | Checkpoint selection only |
| Test (development) | 12/13 | Inference within each run; frontend candidate ranking across runs |

`query_validation_days` chooses query days. `selection_validation_days=null` reuses query loss for checkpoint selection, so the current setup has no independent validation score. For independent selection, use disjoint lists such as query `[8]` and selection `[11]`. The source/cache partition named `validation` contains both days; public metric roles are `query`, `valid`, and `test`.

`prepare.py` verifies recording identities and annotation alignment, then saves a version-3 source inventory. Raw recordings and source annotations remain read-only. The input inventory contains 168 recordings and 2,773 annotated intervals.

The sample recipe uses 112-second windows, positive windows in the 7-3 minutes before an event, balanced same-mouse/day negatives, ten-minute seizure margins, and the exclusions in `sample_pipeline.py`. Fixed positive offsets from the interval start are 0, 128, 56, and 112 seconds. Whole recording days stay together across support, query, selection, and test.

## Frontend and task CNN

Fixed preprocessing removes each window's mean, resamples from 500 to 256 Hz, and zero-pads to 28,672 samples. These waveforms are cached by `meta_learning/waveform.py`; there are no fitted dataset or task normalization statistics.

Inside the frontend, each window is divided by its own RMS. Peak-scaled arithmetic avoids overflow/underflow across input units; a silent window yields a zero normalized waveform with finite gradients. There is no fixed 0.01 multiplier, amplitude branch, or gain augmentation.

```text
waveform / window RMS                        (28,672 samples)
-> Conv1d(1->16, k8, s4)                      7,168
-> ResBlock(16) -> Conv1d(16->32, k2, s2)      3,584
-> ResBlock(32) -> Conv1d(32->48, k2, s2)      1,792
-> ResBlock(48) -> Conv1d(48->64, k7, s7)      256
-> ResBlock(64) -> Conv1d(64->64, k1, s1)      256
-> ResBlock(64) -> ResBlock(64, dilation 2)
-> GroupNorm(8) -> GELU -> Conv1d(64->64, k1)
-> 64 channels x 256 time bins
```

Each pre-activation residual block uses GroupNorm/GELU and two k3 convolutions; its final convolution is zero-initialized. The residual stream carries the RMS-normalized waveform. The frontend has 131,296 parameters, no BatchNorm or dropout, and features depend only on their own window. Seeded frontend and CNN initialization preserve caller CPU/CUDA RNG streams. `frontend_batch_size` controls window chunks independently of CNN `batch_size`.

The task CNN convolves time, mixing feature channels directly: AxisBatchNorm -> Conv1d(64->32, k5, s2) -> ReLU -> max-pool 2 -> AxisBatchNorm -> Conv1d(32->32, k3) -> ReLU -> max-pool 2 -> AxisBatchNorm -> Conv1d(32->32, k3) -> ReLU -> max-pool 2 -> flatten (512) -> dropout -> Linear(512->128) -> sigmoid -> dropout -> Linear(128->2). The CNN derives its dense input size from the frontend shape and has 82,658 parameters. `classifier_dropout` defaults to zero. Inference uses softmax with a positive threshold of 0.5.

The 64 x 256 interface is produced by learned convolution strides with a total
waveform reduction of 112; there is no interpolation or fixed pooling at the
interface. All three CNN blocks emit 32 channels, with a 512-feature dense input.
This reduces the CNN from 177,090 to 82,658 parameters relative to the
64/64/64 classifier, while preserving the frontend and its 64 x 256 output.
This infrastructure update establishes a new protocol with the 64 x 256 boundary
and 32/32/32 CNN. Use a fresh research session and establish a matching baseline;
old session records and immutable source snapshots remain unchanged.

## Meta-training

1. Each mouse has one fixed task: all eligible support Days 0-7 and its configured query days. Every outer step visits each configured mouse once, in target order.
2. Fit a task-local CNN for exactly `inner_epochs` complete shuffled epochs with plain SGD, learning rate `inner_lr`, without momentum. There is no early stopping or best-epoch rollback. The frontend is held fixed during that fit; the complete fit remains differentiable.
3. Query labels supervise the frontend through both the query features and every support-training update. Average query loss within each represented event, then across events. Weight each mouse's gradient by its eligible positive query-event count divided by the total count, before crop expansion and optional per-class caps. Update the frontend once after all tasks.
4. Outer AdamW uses `outer_lr`; decoupled `outer_weight_decay` applies only to frontend weight matrices, excluding biases and normalization parameters. `inner_weight_decay` adds an L2 pull toward detached reference CNN weights after inner gradient clipping.
5. At every `eval_every` interval and the final step, freeze the frontend and fit one fresh CNN per mouse with the same support policy, SGD solver, and epoch budget. Score query, optional valid, and test without gradients.
6. Select the checkpoint with minimum mouse-mean predictive log loss on valid days, or query days when no separate valid days are configured. Test never selects checkpoints within a run; autoresearch uses the selected checkpoint's test loss to rank frontend candidates. Outer training uses label smoothing; selection uses ordinary unsmoothed predictive log loss.

Support training uses normal CNN BatchNorm batch statistics and task-local running means/variances. Query/test use eval mode. Each fit ends with exact full-support BatchNorm recalibration using the final weights (`batchnorm_recalibration=true`); this is differentiable in the meta path. The full support-only EMA trajectory and recalibration contribute to the hypergradient. Query inference and backward replay never mutate saved/reference statistics.

RNG-preserving segment checkpointing keeps the full inner trajectory without truncating hypergradients. `inner_checkpoint_steps=0` disables replay for small diagnostic checks. `inner_shuffle_seed=null` draws fresh window order and minibatch shuffles; an integer fixes each task's sampling and shuffles without advancing caller RNG streams. Support metrics are diagnostic only. `inner_log_every=0` measures initialization and the final epoch; a positive value also measures the first epoch and each configured interval.

### Random crops

`augment_random_crop=true` replaces training windows with real crops. Positive crops stay in their event's complete 7-3 minute interval. Negatives are sampled uniformly over legal 112-second start positions in all eligible seizure-free intervals of the same mouse/day, weighted by legal-start counts and using `sample_pipeline.safe_regions`. Recording boundaries, margins, and the configured recording scope are preserved.

`support_crops` and `query_crops` count crops per event and class. Plan support crops over the complete support before batching. Negative counts are per-event budgets; negative positions are independent of matched fixed-window rows. `random_crop_scope` selects support, query, or both. Optional `support_per_class`/`query_per_class` caps limit generated examples. Event/task weighting is independent of crop counts.

Training crop draws use the checkpointed episode RNG. Per-mouse support fitting uses the same crop policy, with seed `seed + 10000 + target position`; draw crops once and reuse them for all epochs. Query-refit evaluation, valid, and test use fixed windows. `inner_epochs` is never rescaled: K crops per event/class give approximately K/4 times the optimizer steps of the four-window recipe.

Crop preparation uses bounded-memory caches of positive contexts and complete eligible negative intervals. It checks materialized contexts for finite values, rather than auditing complete recordings. Fixed-window quality audits cover only the windows actually loaded.

### Current configuration

| Setting | `configs/meta.json` | Config default |
| --- | --- | --- |
| Targets | X1/X2/X9/X14 | X2 |
| Outer steps | 50 | 100 |
| Inner/per-mouse epochs | 40 | 50 |
| CNN/frontend batch size | 64 / 64 | 32 / 32 |
| Support/query crops | 4 / 64 | 40 / 64 |
| Inner checkpoint segment | 20 updates | 5 updates |
| Query / valid days | 8,11 / none | 8 / 11 |

The JSON config explicitly overrides these defaults. It uses inner SGD lr 0.01, outer AdamW lr 0.001, zero inner/outer weight decay, dropout 0, label smoothing 0.1, gradient clipping 5, evaluation every 10 steps, and checkpointing every 25 steps. Preserve explicitly selected budgets when changing code.

## Commands

The Docker launcher uses `seizure-prediction-cnn:pytorch`, built with PyTorch 2.14.1, CUDA 12.6, and cuDNN 9. This runtime enables cuDNN for deterministic dilated convolutions, including those generated by convolution double backward. The image disables TF32 to keep standalone CUDA checks consistent with the trainer's FP32 policy. Build it with `docker build -t seizure-prediction-cnn:pytorch .`. Existing containers retain their original image; use the original runtime for their replay and compatible resumption. The version-2 protocol locks the current runtime files; earlier sessions retain their original locks in their source snapshots.

Raw data lives under `/data/users/jianchen/data/seizure_data`, with shared caches under `/data/users/jianchen/data/seizure_data-cache/meta-learning`. Override `DATA_DIR`, `CACHE_DIR`, `PREPARED_DIR`, `OUTPUT_DIR`, `IMAGE_NAME`, or `TRAIN_CPU_THREADS` as needed. The launcher's default output is `results/waveform-conv-meta`.

```bash
./run_meta.sh prepare
OUTPUT_DIR="$PWD/results/task-plan" ./run_meta.sh train --plan-only
OUTPUT_DIR="$PWD/results/waveform-conv-meta-rms" ./run_meta.sh
```

`./run_meta.sh` defaults to training with `configs/meta.json`; pass training options directly, such as `./run_meta.sh --resume`. Explicit `train` is also accepted. Training runs detached and appends output to `run.log`; the launcher prints its container ID. Use `tail -f run.log` or `--foreground`. `--dry-run` prints the Docker command without launching it. Use a new output directory for a new configuration or source version.

```bash
OUTPUT_DIR="$PWD/results/waveform-conv-meta-rms" ./run_meta.sh --resume
OUTPUT_DIR="$PWD/results/waveform-conv-meta-rms" ./run_meta.sh query
OUTPUT_DIR="$PWD/results/waveform-conv-meta-rms" ./run_meta.sh valid
OUTPUT_DIR="$PWD/results/waveform-conv-meta-rms" ./run_meta.sh test
```

`query`, `valid`, and `test` each replay one role using selected checkpoints, without fitting or gradients. `valid` requires non-null selection days. Reports go to `evaluation/<role>/result.json`. Configuration accepts current fields only; evaluation accepts `--split query|valid|test`.

Resume requires identical configuration, sources, data, and runtime. It restores frontend weights, outer optimizer, episode RNG, CPU/CUDA/Python RNG, history, and selection. Work after the latest completed outer checkpoint is replayed. A completed run exits without further updates.

For local execution, use the versions in `requirements.txt`:

```bash
python prepare.py --output prepared
python train.py --config configs/meta.json --prepared-dir prepared --cache-dir /path/to/cache --run-dir results/new-run
python evaluate.py --prepared-dir prepared --cache-dir /path/to/cache --run-dir results/new-run --split test
python -m meta_learning.plotting results/new-run
```

## Analysis and final fitting

`scripts/plot_frontend_tsne.py` reads the selected current-schema models, verifies their source snapshots and saved predictions, and plots fixed reference-window frontend features. PCA is fitted on support only; t-SNE jointly lays out each mouse's reference windows. Test participates only in the exploratory layout. Outputs use a new timestamped visualization directory.

```bash
python scripts/plot_frontend_tsne.py --run-dir results/new-run --prepared-dir /path/to/prepared --cache-dir /path/to/cache --device cuda
```

`scripts/fit_frontend_adam.py` is a separate final-fitting experiment: freeze a selected frontend, fit fresh CNNs for 100 complete Adam epochs, and select each CNN epoch by Days 8/11 log loss. Support Days 0-7 alone supply optimizer updates and BatchNorm calibration. Test is scored once after selection. Those query days already supervised the frontend; this is reuse for CNN epoch selection. This script does not change the meta SGD solver.

```bash
python scripts/fit_frontend_adam.py --source-run results/new-run --run-dir results/adam-personalization/new-run --epochs 100 --lr 0.001 --device cuda
```

This separate Adam experiment is outside the locked autoresearch workflow. The Adam job saves replayable weights, optimizer moments, shuffle RNG, histories, and selected validation predictions. Resume requires matching budgets, configuration, and source snapshots.

## Artifacts

- `run.json`, `config.json`, `source/`: configuration, source snapshots/hashes, data identity, runtime, task schedule, and preprocessing policy.
- `episodes.json`, `data/X*/`: event counts, task weights, partition/sample tables, fingerprints, class ratios, and covered quality audits.
- `history.json`, `inner/step-*/task-*.json`: outer/query metrics, fixed epoch counts, inner optimizer steps, and measured support diagnostics.
- `checkpoints/last.pt`, `status.json`: resumable training state and completion/interruption status.
- `query/`, optional `valid/`, `test/`: per-evaluation reports; test metrics never control within-run checkpoint selection.
- `selection.json`, `selected/step-*/`: selected frontend and per-mouse CNN/BatchNorm weights, predictions and metrics.
- `convergence.png`, `convergence.svg`: support/query/valid/test accuracy curves from the same support-only fits.
- `evaluation/<role>/`: inference-only replay outputs.

Private recordings, annotations, and historical experiment artifacts are preserved. Current tools require current configuration and checkpoint schemas.
