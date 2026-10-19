# MouseLFP waveform convolution frontend meta-learning

Learn a shared spectral preprocessing frontend that helps a classifier generalize from earlier to later recording days of the same mouse. Only frontend parameters are meta-learned. CNN classifiers are trained independently within tasks and for each mouse; their initialization is not learned.

This branch contains the preparation, task planning, meta-training, personalization, query/test evaluation, and resume workflow, together with its matched pooled-training control.

## Data and partitions

Private input contains 168 continuous recordings and `seizure_epochs_pitime_seizure_w15_t0.3_min5_gap10.parquet`, with 2,773 annotated intervals. The modeled mice are X1/X2/X4/X9/X10/X14/X15.

| Purpose | Days | Permitted use |
| --- | --- | --- |
| Support (training) | 0-7 | Inner CNN fitting, pooled representation training, personalization, fitted statistics |
| Query (validation) | 8 (`query_validation_days`) | Outer frontend gradients; pooled representation training |
| Selection (validation) | 11 (`selection_validation_days`) | Checkpoint selection only; never supplies gradients |
| Eval (test) | 12/13 | Inference only for standalone baseline or at meta/pooled evaluation intervals; never used for fitting or selection |

Validation is now part of meta-training supervision, so its score measures the query objective rather than independent generalization. Test is the held-out evaluation partition. Source recordings and annotations stay read-only. `prepare.py` records source identities and verifies row/timestamp alignment independently of model code. Source inventories use version 3 and contain source identity and day partitions only.

The shared sample recipe in `sample_pipeline.py` currently uses 112-second windows, a 7-3-minute positive interval, a 56-second grid plus reference offsets, balanced same-mouse/day negatives, ten-minute negative seizure margins, and existing event-history/postictal exclusions. Actual offsets are 0, 128, 56, and 112 seconds from the positive interval start. Training, validation, and test share these definitions. Changes to sampling alter the evaluated task and must be reported.

## Learning procedure

Choose the training algorithm with `--algorithm meta|pooled|log_stft` (or the `algorithm` configuration field). **Meta and pooled runs start directly with outer step 1 and never run a startup baseline.**

Run the pure log-STFT baseline separately when needed:

```bash
OUTPUT_DIR="$PWD/results/log-stft" ./run_meta.sh train --config configs/meta.json --algorithm log_stft
```

`log_stft` applies the frozen DFT filterbank to the cached waveforms, takes `log1p(magnitude) / ln(10)`, and average-pools over time to 56 bins, giving 40 band channels × 56 time bins with zero learnable frontend parameters. It feeds the same 1-D CNN as the learned frontend, with 40 input channels; the learned frontend is not initialized to match it. It fits one fresh CNN per configured mouse on all Days 0-7 using the configured seed, optimizer, fixed epochs, dropout, and batch size, then scores query Days 8/11 and eval Days 12/13. It has no outer optimizer or updates; `outer_steps`, `eval_every`, and `checkpoint_every` do not repeat this fitting. All seven mice are included by the full configuration.

Baseline reports are saved under `baseline/log_stft/report.json`, with per-mouse checkpoints, query/test predictions, and metrics. The report root is the selection-day report (the query report when `selection_validation_days` is null); nested `query` and `eval` fields hold the query-day and test reports. `selection.json` references the fitted baseline checkpoints, so `validate` and `test` replay commands work with the baseline run directory. Its checkpoints identify `frontend_type="pure_log_stft"`. A compatible completed baseline is reused with `--algorithm log_stft --resume` after identity/checkpoint checks. Keep this output directory separate from meta and pooled runs. Learned runs record their own query/eval metrics without automatic baseline-difference fields.

1. Construct one fixed task per mouse: the entire training partition (Days 0-7) is support and the configured validation query days (currently Day 8) are query; the disjoint selection days (currently Day 11) are held out for checkpoint selection. Whole recording days remain together, so overlapping windows cannot cross support/query boundaries. Test Days 12/13 are never included in a task.
2. Include each configured mouse exactly once in every outer step: all seven mice in the full configuration. Reuse that mouse's fixed support/query partitions every step. By default, use every eligible window on both sides (`support_per_class` and `query_per_class` are `null`). Optional per-class caps sample without replacement. Preserve absolute spectral amplitude with identity normalization (scale=1), before the neural frontend; no normalization statistics are fitted.
3. In each inner loop, train private CNN tensors with minibatch SGD and momentum on support labels, starting from fixed unlearned weights determined by the run seed. Train complete shuffled epochs, keeping CNN weights and momentum throughout that task (both moments and step counts when Adam is selected). The frontend stays fixed. Train exactly `inner_epochs` complete epochs as specified in `configs/meta.json`, with no early stopping. Reset the classifier only when starting another task. By default (`inner_shuffle_seed=null`) every outer step draws a fresh support window order and fresh minibatch shuffles from the run's RNG streams. Setting `inner_shuffle_seed` to a nonnegative integer fixes both for each task (seed + task index) at every outer step, so the meta objective becomes a deterministic function of the frontend; this isolates outer-optimization progress from shuffle noise at the risk of fitting one inner trajectory. Fixed shuffles run inside a forked RNG and leave the run's torch and episode RNG streams unchanged. Per-task reports record the seed used. `configs/meta.json` uses fresh shuffles (`inner_shuffle_seed=null`); set an integer only for deterministic diagnostics. On GPU, nondeterministic kernels can still introduce small run-to-run differences.
4. In the outer loop, differentiate query loss through each completed CNN training trajectory. Accumulate equally weighted gradients from all configured mice (1/7 per mouse in the full configuration), then update the shared frontend once. Tasks run sequentially to bound memory; every task in that step sees the same frontend weights. Detach the reference CNN weights, while retaining the higher-order frontend-to-classifier-training gradient. RNG-preserving segment activation checkpointing recomputes inner activations instead of truncating this gradient. The outer optimizer contains no CNN parameters.
5. Every `eval_every` steps and at the final step, freeze the updated shared frontend and train a fresh CNN for each mouse using all its Days 0-7. Reset classifier weights from the run seed, including for the pooled control; no learned CNN weights transfer into this stage. Keep the same identity normalization used in the episodes. Use the same optimizer, dropout, and fixed epoch budget as inner training. Evaluate this CNN on both validation query and test eval without gradients or parameter updates.
6. Select the shared frontend and its trained per-mouse classifiers by minimum mouse-mean log loss on the selection days. Because query days are the outer training objective, selecting on them (`selection_validation_days=null`) picks the most query-fitted frontend; an earlier X2 run reached 100% refit query accuracy while test AUC fell from 0.75 to 0.60. Test metrics are reporting only and cannot influence checkpoint selection. Report accuracy at threshold 0.5 for both partitions. Selected per-mouse checkpoints are retained exactly as evaluated. The shared `frontend.pt` artifact contains frontend weights only.

The full configuration trains for exactly `inner_epochs` epochs (currently 20); the Config class default remains 50. The full configuration uses plain SGD at learning rate 0.001 without momentum, batch size 32, and classifier dropout 0; its outer AdamW learning rate is 0.0001. The Config class default inner learning rate remains 0.0005. Support loss and accuracy are logged for diagnosis and never shorten training. The outer objective uses the classifier after the configured final epoch, without rollback to the best support epoch. Final per-mouse CNN fitting uses the same fixed budget. Adam remains available through `inner_optimizer="adam"` with the configured betas and epsilon. The outer optimizer is AdamW. `outer_weight_decay` (currently 10.0) applies decoupled decay, a per-step shrink of `outer_lr × outer_weight_decay` (1% at lr 0.001), to frontend weight matrices only; biases, LayerNorm parameters, and the pooled control's auxiliary CNN are not decayed.

Every meta-training step prints a `query_task` record after each mouse's inner fit and a `query_step` summary after all mice finish. These use all query-day windows by default, before that step's outer update. Records include the label-smoothed outer loss, accuracy at threshold 0.5, and sample counts; means weight each mouse equally. Query metrics supervise the frontend and never control inner stopping. Per-task metrics are saved in `inner/step-*/task-*.json`; per-step summaries are saved under `query` in `history.json`.

At evaluation intervals, `query_eval_step` prints selection, query, and test means for the updated frontend after fresh support-only CNN fitting. History stores these as `selection`, `query_refit`, and `eval`, and reports are saved under `selection/`, `query/`, and `eval/step-*.json`. The refitted selection loss used for checkpoint selection is ordinary unsmoothed predictive log loss. It differs from the smoothed, pre-update training query loss; both timings and definitions are explicit. The standalone log-STFT algorithm uses the same refitting seeds/budget and predictive metrics.

The inner settings were chosen for a smooth meta objective. With SGD at learning rate 0.01 and momentum 0.9, X2 query loss changed by 0.01-0.2 under frontend perturbations of norm 1e-6 to 1e-4 (5 to 50 epochs), so the hypergradient described local chaos rather than a descent direction; even a fixed shuffle did not make the objective usable. With learning rate 0.001 and no momentum (20 epochs), the loss along the negative hypergradient follows the first-order prediction up to a parameter step of 1e-2 and keeps decreasing at 1e-1. Changing these settings should be rechecked with a similar one-dimensional slice. This is a fixed-trajectory bilevel representation objective. Completing the configured epochs is a training budget, not a claim of convergence. Each fit records `epochs_requested`, `epochs_run`, and `stop_reason="fixed_epochs"`. Every update in the fixed trajectory supplies the query hypergradient; there is no loss-dependent stopping decision. This implementation does not use implicit differentiation or first-order/truncated approximations. Forward fitting saves detached CNN/optimizer/BatchNorm-state boundaries, global step counts, and RNG states in CPU memory. The running-statistic EMA is part of the differentiable inner state; query gradients include the dependence on support-fitted statistics. Replay restores the statistics for each segment and leaves saved/reference statistics unchanged. Backward replays one segment at a time, carrying parameter and optimizer-state adjoints between segments and accumulating the support-feature gradient. Adam retains both moments and global bias correction; its square-root calculation uses a finite zero subgradient without changing forward update values. GPU activation memory depends on segment length; CPU checkpoint storage grows with trajectory length. The implementation computes the first outer gradient and does not support differentiating that outer gradient again.

This branch (`meta-learning/waveform-conv-frontend`) removes the precomputed STFT. The only fixed preprocessing is per window: remove the window mean, resample 500 to 256 Hz (`resample_poly`), and zero-pad to a multiple of 256 samples, giving 28,672 samples for a 112-second window. These waveforms are cached (`cache/waveforms`, keyed on `meta_learning/waveform.py` only, so frontend edits do not invalidate the cache). No dataset or task statistics are fitted before the frontend.

The frontend is one generic, randomly initialized network: a pre-activation 1-D ResNet from waveform to a multi-channel map for the CNN.

```text
waveform (28,672 samples at 256 Hz) × 0.01 (fixed unit conversion)
→ Conv1d(1→16, k8, s4)                                   28,672 → 7,168
→ [ResBlock(16)] → Conv1d(16→32, k4, s4)                 → 1,792
→ [ResBlock(32)] → Conv1d(32→48, k4, s4)                 → 448
→ [ResBlock(48)] → Conv1d(48→64, k4, s4)                 → 112
→ [ResBlock(64)] → Conv1d(64→64, k2, s2)                 → 56
→ [ResBlock(64)] → [ResBlock(64, dilation 2)]
→ GroupNorm(8) → GELU → Conv1d(64→64, k1)
→ feature sequence: 64 channels × 56 time bins → task-local 1-D CNN

ResBlock(C, d): x + Conv1d_k3(GELU(GroupNorm8(Conv1d_k3,dilation d(GELU(GroupNorm8(x))))))
                (the last conv is zero-initialized, so each block starts as the identity)
```

All 130,272 frontend parameters are outer/meta parameters. There is no filterbank, modulus, log compression, fixed pooling, or residual base map: the network decides how to decompose the waveform, and its output is not aligned with log-STFT at step 0. The frontend hands the CNN a `(channels, time)` sequence rather than an image: channels are unordered features, so nothing forces the frontend to arrange them along an artificial second spatial axis (`MetaModel` shapes the reference CNN from the attached frontend's `output_shape`). GroupNorm normalizes each sample's residual-branch inputs only; the unnormalized residual stream keeps absolute amplitude. All computation is per sample, with no BatchNorm or dropout. Weight decay applies to every frontend weight matrix. Frontend initialization follows the run seed without advancing caller CPU/CUDA RNG streams or changing the task CNN reference. It does not load historical trained CNN weights.

`augment_random_crop` (currently true) replaces each meta/pooled training window with a fresh random crop every outer step. A positive row may start anywhere in its event's 7-3 minute interval (128 s of start positions, so every crop is a real positive window). A negative row may start within ±64 s of its fixed position, clipped to its seizure-free safe region, so every crop still satisfies the 10-minute seizure margins and avoids positive intervals (28-128 s of start positions in X2). `meta_learning/crops.py` caches each row's 240 s context segment (mean-removed, resampled to 256 Hz) under `cache/crops`; a crop takes 112 s and removes its own mean. A crop at the fixed position reproduces the cached fixed window to about 3e-5 relative RMS. Crop positions come from the episode RNG, so checkpoints resume them exactly and a fixed `inner_shuffle_seed` also fixes them. Day 11 selection, query refit metrics, test, and the `log_stft` algorithm always use the fixed windows, so sample tables and fingerprints are unchanged. Crops of one event are strongly correlated: they regularize but do not add independent events. `random_crop_scope` (`both`, `query`, or `support`) chooses which side is cropped; uncropped sides use the fixed windows. `query_crops` (K, default 1) draws K independent crops per query window each outer step and averages the outer loss over them, reducing crop noise in the hypergradient (the pooled control then weights query examples K times). Support crops change the inner CNN's training data every step. Per-mouse fitting (the CNN scored on Days 8/11 and 12/13 and saved in selected checkpoints) uses exactly the same support policy as meta inner fits: the same K crops per window, `inner_epochs`, and optimizer, so meta-training optimizes the frontend for the CNN that is actually deployed. Its crops come from a fixed per-mouse seed (`seed + 10000 + mouse index`), so a refit at the same frontend is reproducible; its report records `support_crops`. `support_crops` (K, default 1) gives every support window K independent crops in the same inner fit, so the fitted CNN depends less on one draw of crop positions. Each epoch is then K times longer, and `inner_epochs` is not rescaled: with K=10 and 20 epochs, every CNN fit (meta inner and per-mouse) takes about 10 times the optimizer steps of a fixed-window fit. Choose `inner_epochs` with this in mind.

Random initialization makes the first outer updates large in function space: in a two-step X2 smoke run at `outer_lr=0.001`, one update moved Day 8 query accuracy from 0.50 to 0.83 while Day 11 selection AUC was 0.47. Prefer a smaller `outer_lr` with this frontend and rely on Day 11 selection.

The task CNN is one-dimensional: AxisBatchNorm(channels) → Conv1d(channels→16, k5, s2) → ReLU → max-pool 2 → AxisBatchNorm → Conv1d(16→32, k3) → ReLU → max-pool 2 → AxisBatchNorm → Conv1d(32→64, k3) → ReLU → max-pool 2 → flatten (64 × 4) → dropout → Linear(256→128) → sigmoid → dropout → Linear(128→2). It mirrors the former 2-D MouseLFP CNN's stages but convolves time only; channels are fully mixed by each convolution, and flattening keeps where in the window a pattern occurs. It has about 46,000 parameters with the 64-channel learned frontend and 44,000 with the 40-band log-STFT control. This replaces the 2-D CNN used by all earlier runs, so earlier results are not directly comparable.

Frontend computation uses non-reentrant activation checkpointing in chunks of 32 samples. This bounds retained convolution activations while preserving gradients through all support/query windows and the complete inner training trajectory. The pooled control and final per-mouse fitting use the same neural architecture. A neural frontend does not establish that the resulting bilevel objective is globally smooth or that query loss will decrease monotonically; the task CNN still uses ReLU/max-pooling and the inner optimizer retains its own dynamics. Use a new run directory; historical runs remain reproducible with their saved source snapshots.

CNN BatchNorm uses normal training behavior: support minibatches supply batch statistics and update per-task running means/variances, while query/test inference uses eval mode with support-fitted statistics. With `batchnorm_recalibration=true` (the default), every inner, per-mouse, and baseline fit ends with one training-mode pass over the complete support at zero retention, replacing the running EMA with the exact support mean/variance of the final weights. The EMA starts from (0, 1) and retains 0.99 per update, so after t updates it still holds 0.99^t of its initial value; for short fits (for example 49 updates per X2 epoch) this biased the first layer by several standard deviations. The recalibration pass is differentiable, so the hypergradient includes the dependence of these statistics on the frontend; it uses support data only. Reports record `calibrated_support_loss` and `calibrated_support_accuracy` next to the EMA-based per-epoch diagnostics. Each fresh CNN resets its running statistics; fitted per-mouse statistics are saved with the checkpoint. The existing axis normalization retains its Keras-compatible settings (retention momentum 0.99, epsilon 0.001, biased variance). Pooled representation training updates its auxiliary CNN statistics normally, then discards that CNN before support-only per-mouse fitting. Inner gradient replay includes the complete running-statistic trajectory as well as CNN/optimizer state. `classifier_dropout` applies consistently to support, pooled, and final per-mouse CNN training and defaults to zero; query evaluation and inference always disable dropout. Query predictions therefore do not depend on unrelated query-batch companions or mutate shared normalization buffers. The backbone architecture retains its original dropout modules; training config controls their probability.

## Setup and Docker commands

The runtime dependencies are pinned in `requirements.txt`: PyTorch, NumPy, SciPy, pandas, PyArrow, scikit-learn, and Matplotlib. The existing `seizure-prediction-cnn:pytorch` image can execute the mounted source. Build a fresh image when needed:

```bash
docker build -t seizure-prediction-cnn:pytorch .
./run_meta.sh check
./run_meta.sh prepare
```

`run_meta.sh` uses these defaults:

- `DATA_DIR=/data/users/jianchen/data/seizure_data`
- `CACHE_DIR=/data/users/jianchen/data/seizure_data-cache/meta-learning`
- `PREPARED_DIR=$CACHE_DIR/prepared`
- `OUTPUT_DIR=$PWD/results/waveform-conv-meta`
- `IMAGE_NAME=seizure-prediction-cnn:pytorch`
- `TRAIN_CPU_THREADS=4`

Use `--device cpu` for CPU runs and `--dry-run` to inspect a launch command. Config paths are relative to the mounted project directory. All jobs terminate at their configured step budget.

Run `bash run_meta.sh` to launch training with `configs/meta.json` in a detached Docker container. Explicit `train` commands also run detached by default. The container appends stdout and stderr directly to the project-root `run.log` and continues after the terminal closes. The launcher prints the container ID and commands to check its status or stop it. Use `tail -f run.log` to follow progress, or add `--foreground` to a `train` command to keep the Docker client in the foreground. Other subcommands run in the foreground.

### Plan fixed support/query tasks

```bash
OUTPUT_DIR="$PWD/results/task-plan" ./run_meta.sh train --plan-only
```

This validates train-support/validation-query partition membership and writes one task per mouse to `episodes.json` without reading waveform windows or training. It also records test eval sample counts and days. Empty partitions or missing support/query classes cause explicit errors.

### Train and compare

```bash
OUTPUT_DIR="$PWD/results/waveform-conv-meta" ./run_meta.sh train --config configs/meta.json
OUTPUT_DIR="$PWD/results/neural-stft-train-val-bn-pooled" ./run_meta.sh train --config configs/meta.json --algorithm pooled
```

The pooled representation control uses the same frontend, CNN architecture, task samples, identity scaling, outer-step count, and final per-mouse fitting budget. It learns its frontend using supervised loss on concatenated training/validation examples and a jointly trained auxiliary CNN. That auxiliary CNN is discarded before final per-mouse fitting; both methods evaluate their frontend with a freshly trained CNN. Meta-training performs additional forward/backward operations; equal outer-step counts are not equal wall-clock compute.

Compare completed runs without launching additional training:

```bash
python -m meta_learning.compare results/waveform-conv-meta results/neural-stft-train-val-bn-pooled --output results/comparison.json
```

The comparison rejects differences in configuration other than the algorithm, source versions, runtime, support/query samples, split roles, or query/test sample fingerprints. Query and test comparison metrics are reported separately. It also reports optimization/validation wall time and peak allocated GPU memory for each run.

`configs/meta.json` controls finite budgets, learning rates, task sizes, fixed CNN epoch budgets, smoothing, and gradient clipping. Defaults retain 1,000 outer steps and one task for each of all seven mice per step, with the epoch solver above for each task and final per-mouse CNN. Task count is derived from `targets`; there is no configurable mouse-sampling budget. This is substantially more work than the former three-step inner loop. `inner_checkpoint_steps=5` checkpoints segments of up to five minibatch updates; zero disables checkpointing for small diagnostic comparisons. `inner_log_every=0` prints only the final support summary for each inner or per-mouse fit. A positive value restores progress at that epoch interval, plus the first and final epochs. Complete per-epoch curves remain saved in task reports; per-mouse query metrics and every-step query summaries are still printed. Use distinct output directories for different configurations or seeds.

### Resume and query replay

```bash
OUTPUT_DIR="$PWD/results/waveform-conv-meta" ./run_meta.sh train --config configs/meta.json --resume
OUTPUT_DIR="$PWD/results/waveform-conv-meta" ./run_meta.sh validate
```

Resumption restores model, optimizer, task-sampler state, PyTorch CPU/CUDA RNG state, history, and selected checkpoint state. Configuration, source identities, code, and runtime must match. An initial step-zero checkpoint allows interruptions during the first long inner fit to resume. Work after the most recent completed outer checkpoint is replayed; partial inner trajectories are not resumed in place. A completed run with the same configuration exits without further optimization.

Query replay (`validate`) loads the selected adapted checkpoints and their saved identity scales. It performs no personalization, gradient updates, or fitting.

### Replay held-out evaluation

Meta/pooled training scores test at evaluation intervals; the standalone baseline scores test after its single per-mouse fit. To replay the selected checkpoint:

```bash
OUTPUT_DIR="$PWD/results/waveform-conv-meta" ./run_meta.sh test
```

This loads the selected per-mouse checkpoints and identity scales, writes test predictions, and leaves model-selection records unchanged. The Python inference interface defaults to `--split test`; use `--split validation` to replay the query objective. Neither replay command trains a CNN or fits statistics.

### Short integration check

```bash
OUTPUT_DIR="$PWD/results/waveform-conv-meta-smoke" ./run_meta.sh train --config configs/smoke.json
OUTPUT_DIR="$PWD/results/waveform-conv-meta-smoke" ./run_meta.sh validate
```

The smoke configuration uses X15, two outer steps, small per-class sample caps, and exactly two epochs for both inner and final per-mouse CNN fitting. It verifies execution rather than estimating convergence or model performance. Keep it separate from full-cohort comparisons. The pooled smoke control uses the same configuration plus `--algorithm pooled`.

## Local execution

Use an environment matching `requirements.txt`. Examples:

```bash
python prepare.py --output prepared
python train.py --prepared-dir prepared --cache-dir /path/to/cache --run-dir results/waveform-conv-meta --config configs/meta.json
python evaluate.py --prepared-dir prepared --cache-dir /path/to/cache --run-dir results/waveform-conv-meta
python -m unittest discover -s tests -v
```

Data paths default to `SETTINGS_MouseLFP.json`. The Docker launcher supplies container paths explicitly.

## Artifacts and reproducibility

Each run retains:

- `run.json`, `config.json`, and `source/`: configuration, source inventory identity, code hashes/snapshots, sampling policy, normalization/BatchNorm policy, and runtime.
- `episodes.json` and `data/X*/`: fixed train-support/validation-query task definitions, event eligibility, split class counts, sample coordinates/fingerprints, and sampled signal-quality audits.
- `history.json`, `status.json`, and `checkpoints/last.pt`: losses, gradient norms, query/eval summaries, progress, and resumable training state.
- `convergence.png` and `convergence.svg`: automatically drawn after successful training (including a completed resume). One plot shows accuracy versus outer loop with four lines: Support (Days 0-7), Query (configured query days), Val (configured selection days), and Test (Days 12/13), plus a marker for the selected checkpoint. All four use the same support-fitted CNNs at each recorded evaluation step, averaged equally across mice; support accuracy uses the final recalibrated BatchNorm statistics when available. Query values are post-update refit metrics. Test curves are reporting only. Missing metrics are omitted, and standalone log-STFT has only one point per split at step zero. Regenerate figures without training with `python -m meta_learning.plotting results/waveform-conv-meta`.
- `baseline/log_stft/`: query/test reports, per-mouse CNN checkpoints, and predictions, produced only by `algorithm="log_stft"`. Meta and pooled runs do not create this directory.
- `inner/step-*/task-*.json`: support-only loss curves, completed epochs/updates, stopping reason, and task dates/counts. Outer history records the fixed epochs per task, completed tasks, and optimizer updates. Per-mouse query/eval reports and selected checkpoints retain the same CNN-fit diagnostics.
- `selection.json` and `selected/step-*/`: selected shared model, adapted per-mouse models/scales, `predictions_{query,selection,eval}.csv` and matching metrics; selection uses selection-day loss only.
- `evaluation/validation/` or `evaluation/test/`: inference-only reports and per-window predictions.

Source preparation and finite-window quality audits establish identity, alignment, and the recorded audit findings. Inspect actual signal artifacts before attributing model behavior to physiology. Independent events and recording days, rather than overlapping window counts, determine the available evidence.

Historical experiment records remain available locally and contain their original source snapshots. The earlier `results/meta` run optimized CNN initialization as well as the frontend and used a three-step inner loop; it does not measure this frontend-only epoch objective. Use a new output directory for the changed algorithm. Historical runs can be replayed with their saved source versions and cannot resume under this changed algorithm. The fixed train-support/validation-query/test-eval protocol requires a new output directory and cannot resume former rolling-within-train runs. Restoring training-mode BatchNorm also requires a fresh run; frozen-statistics checkpoints cannot resume under this code. The launcher now defaults to `results/waveform-conv-meta` to keep those runs separate. The former four-draw mouse-sampling schedule also requires a new output directory. Old `min_support_days`, `query_days`, `tasks_per_step`, `inner_steps`, `adaptation_steps`, `adaptation_lr`, `inner_min_epochs`, `inner_max_epochs`, `inner_patience`, and `inner_relative_tolerance` configuration fields are rejected; `inner_epochs` controls the fixed budget of both CNN fitting stages. Runs using the former early stopping policy require a new output directory.

## Source layout

```text
meta_learning/      Frontend, fixed split tasks, meta/pooled objectives, adaptation, and evaluation
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

The outer representation objective follows the bilevel learning formulation in [Franceschi et al. (ICML 2018)](https://proceedings.mlr.press/v80/franceschi18a.html). Unlike initialization-learning methods such as [MAML (ICML 2017)](https://proceedings.mlr.press/v70/finn17a.html), the only meta-parameters here are the frontend weights. The shared frontend is a single randomly initialized 1-D ResNet over preprocessed waveforms, producing a multi-channel map for the task CNN.
