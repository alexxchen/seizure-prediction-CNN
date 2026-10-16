# MouseLFP cross-day meta-learning

Learn a shared spectral preprocessing frontend that helps a classifier generalize from earlier to later recording days of the same mouse. Only frontend parameters are meta-learned. CNN classifiers are trained independently within tasks and for each mouse; their initialization is not learned.

This branch contains the preparation, task planning, meta-training, personalization, validation, resume, and explicit final-test workflow, together with its matched pooled-training control.

## Data and partitions

Private input contains 168 continuous recordings and `seizure_epochs_pitime_seizure_w15_t0.3_min5_gap10.parquet`, with 2,773 annotated intervals. The modeled mice are X1/X2/X4/X9/X10/X14/X15.

| Purpose | Days | Permitted use |
| --- | --- | --- |
| Training | 0-7 | Episode support/query, pooled training, personalization, fitted statistics |
| Validation | 8/11 | Checkpoint selection and inference |
| Test | 12/13 | Explicit final inference using selected adapted checkpoints |

Source recordings and annotations stay read-only. `prepare.py` records source identities and verifies row/timestamp alignment independently of model code. Source inventories use version 3 and contain source identity and day partitions only.

The shared sample recipe in `sample_pipeline.py` currently uses 112-second windows, a 7-3-minute positive interval, a 56-second grid plus reference offsets, balanced same-mouse/day negatives, ten-minute negative seizure margins, and existing event-history/postictal exclusions. Actual offsets are 0, 128, 56, and 112 seconds from the positive interval start. Training, validation, and test share these definitions. Changes to sampling alter the evaluated task and must be reported.

## Learning procedure

1. Construct rolling episodes within each mouse using only Days 0-7. Every support day precedes every query day. Whole recording days remain together, so overlapping windows cannot cross support/query boundaries.
2. Include each configured mouse exactly once in every outer step: all seven mice in the full configuration. Choose one chronological episode uniformly within each mouse. By default, use every eligible window on both sides (`support_per_class` and `query_per_class` are `null`). Optional per-class caps sample without replacement. Estimate one power scale from the selected support windows and reuse it for query windows.
3. In each inner loop, train private CNN tensors with minibatch Adam on support labels, starting from fixed unlearned weights determined by the run seed. Train complete shuffled epochs, keeping CNN weights, first/second moments, and optimizer step counts throughout that task. The frontend stays fixed. Stop after support-loss stability or the finite maximum epoch budget, respecting the minimum epochs. Reset the classifier only when starting another task.
4. In the outer loop, differentiate query loss through each completed CNN training trajectory. Accumulate equally weighted gradients from all configured mice (1/7 per mouse in the full configuration), then update the shared frontend once. Tasks run sequentially to bound memory; every task in that step sees the same frontend weights. Detach the reference CNN weights, while retaining the higher-order frontend-to-classifier-training gradient. RNG-preserving segment activation checkpointing recomputes inner activations instead of truncating this gradient. The outer optimizer contains no CNN parameters.
5. At validation intervals, freeze the shared frontend and train a fresh CNN for each mouse using all its Days 0-7. Reset classifier weights from the run seed, including for the pooled control; no learned CNN weights transfer into this stage. Fit the power scale on those training samples only. Use the same optimizer, dropout, epoch budgets, and support-loss stability as inner training.
6. Select the shared frontend and its trained per-mouse classifiers by mean validation accuracy at threshold 0.5, breaking ties with lower mean validation log loss. Selected per-mouse checkpoints are retained exactly as evaluated. The shared `frontend.pt` artifact contains frontend weights only.

Defaults train for at least 10 and at most 50 epochs, with batch size 32, Adam learning rate 0.0005, betas 0.9/0.999, epsilon 1e-8, and classifier dropout 0. Stop when the absolute change in support evaluation loss is at most 0.1% relative to the preceding epoch for five consecutive epochs. A large loss increase resets this stability counter rather than counting as convergence. Support evaluation reads no query or official validation values. Log support accuracy alongside loss to expose poor fitting at a flat loss. The classifier used by the outer objective is the model at the completed stopping epoch, without rollback to an earlier or initial classifier. SGD remains an explicit `inner_optimizer="sgd"` control using `inner_momentum`.

Every meta-training step prints a `query_task` record after each mouse's inner fit and a `query_step` summary after all mice finish, before official validation starts. Records include query loss (the label-smoothed outer objective), accuracy at threshold 0.5, and per-mouse sample counts; summaries average loss and accuracy equally across mice. These metrics describe each adapted task CNN and the frontend before that step's outer update, using training Days 0-7 query windows. They do not control inner stopping or replace official Days 8/11 validation. Query evaluation reuses the outer objective's forward pass. Per-task metrics are saved in `inner/step-*/task-*.json`; per-step summaries and per-mouse metrics are saved under `query` in `history.json` and resumable checkpoint history.

This is a finite-trajectory bilevel representation objective. Patience is an operational stopping criterion, not a proof of mathematical convergence. A task reaching the maximum epoch budget is explicitly recorded as `max_epochs` with `plateau_reached=false`; it still supplies its completed-trajectory query gradient. The stopping decision is held fixed during differentiation. This implementation does not use implicit differentiation or first-order/truncated approximations. Forward fitting saves detached CNN/optimizer-state boundaries, global step counts, and RNG states in CPU memory. Backward replays one segment at a time, carrying parameter and optimizer-state adjoints between segments and accumulating the support-feature gradient. Adam retains both moments and global bias correction; its square-root calculation uses a finite zero subgradient without changing forward update values. GPU activation memory depends on segment length; CPU checkpoint storage grows with trajectory length. The implementation computes the first outer gradient and does not support differentiating that outer gradient again.

The spectral frontend learns 520 parameters for Gaussian frequency bands, power compression, and positive temporal smoothing. Fixed preprocessing centers windows, resamples from 500 to 256 Hz, and computes a two-second STFT power basis over 1-40 Hz. Only this sample-local fixed basis is cached. Phase is excluded from this representation. The existing CNN consumes `(1, 1, 55, 114)` features.

BatchNorm running means and variances remain fixed in all meta, pooled, and personalization stages; affine normalization weights remain trainable. `classifier_dropout` applies consistently to support, pooled, and final per-mouse CNN training and defaults to zero; query evaluation and inference always disable dropout. Query predictions therefore do not depend on unrelated query-batch companions or mutate shared normalization buffers. The backbone architecture retains its original dropout modules; training config controls their probability.

## Setup and Docker commands

The six direct runtime dependencies are pinned in `requirements.txt`: PyTorch, NumPy, SciPy, pandas, PyArrow, and scikit-learn. The existing `seizure-prediction-cnn:pytorch` image can execute the mounted source. Build a fresh image when needed:

```bash
docker build -t seizure-prediction-cnn:pytorch .
./run_meta.sh check
./run_meta.sh prepare
```

`run_meta.sh` uses these defaults:

- `DATA_DIR=/data/users/jianchen/data/seizure_data`
- `CACHE_DIR=/data/users/jianchen/data/seizure_data-cache/meta-learning`
- `PREPARED_DIR=$CACHE_DIR/prepared`
- `OUTPUT_DIR=$PWD/results/frontend-meta`
- `IMAGE_NAME=seizure-prediction-cnn:pytorch`
- `TRAIN_CPU_THREADS=4`

Use `--device cpu` for CPU runs and `--dry-run` to inspect a launch command. Config paths are relative to the mounted project directory. All jobs terminate at their configured step budget.

### Plan chronological tasks

```bash
OUTPUT_DIR="$PWD/results/task-plan" ./run_meta.sh train --plan-only
```

This validates episode construction and writes `episodes.json` without reading waveform windows or training. A mouse with insufficient labeled training days causes an explicit error rather than falling back to random window splitting.

### Train and compare

```bash
OUTPUT_DIR="$PWD/results/frontend-meta" ./run_meta.sh train --config configs/meta.json
OUTPUT_DIR="$PWD/results/frontend-pooled" ./run_meta.sh train --config configs/meta.json --algorithm pooled
```

The pooled representation control uses the same frontend, CNN architecture, task samples, support-fitted scaling, outer-step count, and final per-mouse fitting budget. It learns its frontend using supervised loss on concatenated support/query examples and a jointly trained auxiliary CNN. That auxiliary CNN is discarded before final per-mouse fitting; both methods evaluate their frontend with a freshly trained CNN. Meta-training performs additional forward/backward operations; equal outer-step counts are not equal wall-clock compute.

Compare completed runs without launching additional training:

```bash
python -m meta_learning.compare results/frontend-meta results/frontend-pooled --output results/comparison.json
```

The comparison rejects differences in configuration other than the algorithm, source versions, runtime, training samples, or validation sample fingerprints. It also reports optimization/validation wall time and peak allocated GPU memory for each run.

`configs/meta.json` controls finite budgets, learning rates, task sizes, CNN stopping, smoothing, and gradient clipping. Defaults retain 1,000 outer steps and one task for each of all seven mice per step, with the epoch solver above for each task and final per-mouse CNN. Task count is derived from `targets`; there is no configurable mouse-sampling budget. This is substantially more work than the former three-step inner loop. `inner_checkpoint_steps=5` checkpoints segments of up to five minibatch updates; zero disables checkpointing for small diagnostic comparisons. `inner_log_every=10` prints support progress at ten-epoch intervals, as well as the first and stopping epochs. Use distinct output directories for different configurations or seeds.

### Resume and validation replay

```bash
OUTPUT_DIR="$PWD/results/frontend-meta" ./run_meta.sh train --config configs/meta.json --resume
OUTPUT_DIR="$PWD/results/frontend-meta" ./run_meta.sh validate
```

Resumption restores model, optimizer, task-sampler state, PyTorch CPU/CUDA RNG state, history, and selected checkpoint state. Configuration, source identities, code, and runtime must match. An initial step-zero checkpoint allows interruptions during the first long inner fit to resume. Work after the most recent completed outer checkpoint is replayed; partial inner trajectories are not resumed in place. A completed run with the same configuration exits without further optimization.

Validation replay loads the selected adapted checkpoints and their saved normalization scales. It performs no personalization, gradient updates, or fitting.

### Explicit final testing

After choosing a run and ending model selection:

```bash
OUTPUT_DIR="$PWD/results/frontend-meta" ./run_meta.sh test
```

This is the only launcher action that reads test signals. It uses the selected per-mouse checkpoints and training-fitted scales, writes test predictions, and leaves model-selection records unchanged. The Python interface requires explicit `--split test` for this operation.

### Short integration check

```bash
OUTPUT_DIR="$PWD/results/frontend-meta-smoke" ./run_meta.sh train --config configs/smoke.json
OUTPUT_DIR="$PWD/results/frontend-meta-smoke" ./run_meta.sh validate
```

The smoke configuration uses X15, two outer steps, small per-class sample caps, and one to two epochs for both inner and final per-mouse CNN fitting. It verifies execution rather than estimating convergence or model performance. Keep it separate from full-cohort comparisons. The pooled smoke control uses the same configuration plus `--algorithm pooled`.

## Local execution

Use an environment matching `requirements.txt`. Examples:

```bash
python prepare.py --output prepared
python train.py --prepared-dir prepared --cache-dir /path/to/cache --run-dir results/frontend-meta --config configs/meta.json
python evaluate.py --prepared-dir prepared --cache-dir /path/to/cache --run-dir results/frontend-meta
python -m unittest discover -s tests -v
```

Data paths default to `SETTINGS_MouseLFP.json`. The Docker launcher supplies container paths explicitly.

## Artifacts and reproducibility

Each run retains:

- `run.json`, `config.json`, and `source/`: configuration, source inventory identity, code hashes/snapshots, sampling policy, normalization/BatchNorm policy, and runtime.
- `episodes.json` and `data/X*/`: chronological task definitions, event eligibility, split class counts, sample coordinates/fingerprints, and sampled signal-quality audits.
- `history.json`, `status.json`, and `checkpoints/last.pt`: losses, gradient norms, validation summaries, progress, and resumable training state.
- `inner/step-*/task-*.json`: support-only loss curves, completed epochs/updates, stopping reason, and task dates/counts. Outer history summarizes patience stops and maximum-budget tasks. Per-mouse validation reports and selected checkpoints retain the same CNN-fit diagnostics.
- `selection.json` and `selected/step-*/`: selected shared model, adapted per-mouse models/scales, validation predictions, and metrics.
- `evaluation/validation/` or `evaluation/test/`: inference-only reports and per-window predictions.

Source preparation and finite-window quality audits establish identity, alignment, and the recorded audit findings. Inspect actual signal artifacts before attributing model behavior to physiology. Independent events and recording days, rather than overlapping window counts, determine the available evidence.

Historical experiment records remain available locally and contain their original source snapshots. The earlier `results/meta` run optimized CNN initialization as well as the frontend and used a three-step inner loop; it does not measure this frontend-only epoch objective. Use a new output directory for the changed algorithm. Historical runs can be replayed with their saved source versions and cannot resume under this changed algorithm. The former four-draw mouse-sampling schedule also requires a new output directory. Old `tasks_per_step`, `inner_steps`, `adaptation_steps`, and `adaptation_lr` configuration fields are rejected; the epoch solver now controls both fitting stages.

## Source layout

```text
meta_learning/      Frontend, chronological episodes, meta/pooled objectives, adaptation, and evaluation
models/            CNN backbone and axis normalization
configs/           Full and smoke training configurations
tests/             Gradient, isolation, caching, resumption, and inference checks
source_data.py     Read-only recordings and annotations
sample_pipeline.py Shared sample definitions and raw-window caching
prepare.py         Source inventory preparation
train.py           Training entry point
evaluate.py        Inference entry point
run_meta.sh        Docker launcher
```

## Method references

The outer representation objective follows the bilevel learning formulation in [Franceschi et al. (ICML 2018)](https://proceedings.mlr.press/v80/franceschi18a.html). Unlike initialization-learning methods such as [MAML (ICML 2017)](https://proceedings.mlr.press/v70/finn17a.html), the only meta-parameters here are the frontend weights. The spectral frontend is a compact Gaussian-band design; it retains a fixed STFT basis and is not an implementation of a raw-waveform LEAF or SincNet model.
