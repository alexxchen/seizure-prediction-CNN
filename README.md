# MouseLFP waveform convolution frontend meta-learning

Learn a shared waveform preprocessing frontend that helps a classifier generalize from earlier to later recording days of the same mouse. Only frontend parameters are meta-learned. CNN classifiers are trained independently within tasks and for each mouse; their initialization is not learned.

This branch contains the preparation, task planning, meta-training, personalization, query/test evaluation, and resume workflow, together with its matched pooled-training control.

## Data and partitions

Private input contains 168 continuous recordings and `seizure_epochs_pitime_seizure_w15_t0.3_min5_gap10.parquet`, with 2,773 annotated intervals. Supported mice are X1/X2/X4/X9/X10/X14/X15; the current configuration trains all seven mice.

| Purpose | Days | Permitted use |
| --- | --- | --- |
| Support (training) | 0-7 | Inner CNN fitting, pooled representation training, personalization, fitted statistics |
| Query (outer training) | 8/11 (`query_validation_days`) | Outer frontend gradients; pooled representation training |
| Valid (checkpoint selection) | Query days (`selection_validation_days=null`) | Reuse query loss for selection; no independent valid set in the current configuration |
| Test | 12/13 | Inference only for standalone baseline or at meta/pooled evaluation intervals; never used for fitting or selection |

Query Days 8/11 supply meta-training supervision. With `selection_validation_days=null`, their post-refit query loss also selects checkpoints; no independent valid metric is emitted. To restore an independent valid set, set `query_validation_days` and `selection_validation_days` to disjoint day lists, for example `[8]` and `[11]`. These two configuration fields control the split without code changes. Test Days 12/13 report final generalization. The cache/source partition named `validation` contains both query and valid days; it is not a metric role. Public metrics use `query`, `valid`, and `test`. Configuration keys `query_validation_days`, `selection_validation_days`, and `eval_every` retain their existing names. Source recordings and annotations stay read-only. `prepare.py` records source identities and verifies row/timestamp alignment independently of model code. Source inventories use version 3 and contain source identity and day partitions only.

The shared sample recipe in `sample_pipeline.py` currently uses 112-second windows, a 7-3-minute positive interval, a 56-second grid plus reference offsets, balanced same-mouse/day negatives, ten-minute negative seizure margins, and existing event-history/postictal exclusions. Actual offsets are 0, 128, 56, and 112 seconds from the positive interval start. Training, validation, and test share these definitions. Changes to sampling alter the evaluated task and must be reported.

## Learning procedure

Choose the training algorithm with `--algorithm meta|pooled|log_stft` (or the `algorithm` configuration field). **Meta and pooled runs start directly with outer step 1 and never run a startup baseline.**

Run the pure log-STFT baseline separately when needed:

```bash
OUTPUT_DIR="$PWD/results/log-stft" ./run_meta.sh train --config configs/meta.json --algorithm log_stft
```

`log_stft` applies the frozen DFT filterbank to the cached waveforms, takes `log1p(magnitude) / ln(10)`, and average-pools over time to 56 bins, giving 40 band channels × 56 time bins with zero learnable frontend parameters. It feeds the same 1-D CNN as the learned frontend, with 40 input channels; the learned frontend is not initialized to match it. It fits one fresh CNN per configured mouse on all Days 0-7 using the configured seed, optimizer, fixed epochs, dropout, and batch size, then scores query Days 8/11 (also used for selection) and test Days 12/13. It has no outer optimizer or updates; `outer_steps`, `eval_every`, and `checkpoint_every` do not repeat this fitting. The current configuration fits X2; pass all seven supported targets explicitly for a cohort run.

Baseline reports are saved under `baseline/log_stft/report.json`, with per-mouse checkpoints, query/test predictions, and metrics. The report root is the selection-day report (the query report when `selection_validation_days` is null); nested `query` and `test` fields hold the query-day and test reports. `selection.json` references the fitted baseline checkpoints, so `validate` and `test` replay commands work with the baseline run directory. Its checkpoints identify `frontend_type="pure_log_stft"`. A compatible completed baseline is reused with `--algorithm log_stft --resume` after identity/checkpoint checks. Keep this output directory separate from meta and pooled runs. Learned runs record their own query/valid/test metrics without automatic baseline-difference fields.

1. Construct one fixed task per mouse: the entire training partition (Days 0-7) is support and the configured validation query days (currently Days 8/11) are query; checkpoint selection currently reuses query (`selection_validation_days=null`). An optional disjoint selection-day list enables independent validation. Whole recording days remain together, so overlapping windows cannot cross support/query boundaries. Test Days 12/13 are never included in a task.
2. Include each configured mouse exactly once in every outer step: all seven in the current configuration. Reuse that mouse's fixed support/query partitions every step. By default, use every eligible window on both sides (`support_per_class` and `query_per_class` are `null`). Optional per-class caps sample without replacement. Preserve absolute spectral amplitude with identity normalization (scale=1), before the neural frontend; no normalization statistics are fitted.
3. In each inner loop, train private CNN tensors with minibatch SGD without momentum on support labels, starting from fixed unlearned weights determined by the run seed. Train complete shuffled epochs, keeping CNN weights and support BatchNorm state throughout that task. Plain SGD has no momentum or optimizer-moment tensors to save or replay. The frontend stays fixed. Train exactly `inner_epochs` complete epochs as specified in `configs/meta.json`, with no early stopping. Reset the classifier only when starting another task. By default (`inner_shuffle_seed=null`) every outer step draws a fresh support window order and fresh minibatch shuffles from the run's RNG streams. Setting `inner_shuffle_seed` to a nonnegative integer fixes both for each task (seed + task index) at every outer step, so the meta objective becomes a deterministic function of the frontend; this isolates outer-optimization progress from shuffle noise at the risk of fitting one inner trajectory. Fixed shuffles run inside a forked RNG and leave the run's torch and episode RNG streams unchanged. Per-task reports record the seed used. `configs/meta.json` uses fresh shuffles (`inner_shuffle_seed=null`); set an integer only for deterministic diagnostics. On GPU, nondeterministic kernels can still introduce small run-to-run differences.
4. In the outer loop, differentiate query loss through each completed CNN training trajectory. Weight each mouse gradient by E_mouse / sum(E_mouse), where E_mouse is its number of unique eligible positive query events on the configured query days, then update the shared frontend once. Count events before crop expansion and optional per-class caps; matched negatives do not add events. Within a meta task, average query losses within each represented event and then across events. Apply the same task weights to the pooled control, retaining its within-task mean over support/query events. Tasks run sequentially to bound memory; every task in that step sees the same frontend weights. Detach the reference CNN weights, while retaining the higher-order frontend-to-classifier-training gradient. RNG-preserving segment activation checkpointing recomputes inner activations instead of truncating this gradient. The outer optimizer contains no CNN parameters.
5. Every `eval_every` steps and at the final step, freeze the updated shared frontend and train a fresh CNN for each mouse using all its Days 0-7. Reset classifier weights from the run seed, including for the pooled control; no learned CNN weights transfer into this stage. Keep the same identity normalization used in the episodes. Use the same optimizer, dropout, and fixed epoch budget as inner training. Evaluate this CNN on query, valid, and test without gradients or parameter updates.
6. Select the shared frontend and its trained per-mouse classifiers by minimum mouse-mean log loss on the selection days. Because query days are the outer training objective, selecting on them (`selection_validation_days=null`) picks the most query-fitted frontend; an earlier X2 run reached 100% refit query accuracy while test AUC fell from 0.75 to 0.60. Test metrics are reporting only and cannot influence checkpoint selection. Report accuracy at threshold 0.5 for both partitions. Selected per-mouse checkpoints are retained exactly as evaluated. The shared `frontend.pt` artifact contains frontend weights only.

`configs/meta.json` currently uses algorithm `meta`, all seven targets, seed 0, 100 outer steps, 40 complete inner epochs, CNN batch size 64, frontend batch size 64, and plain SGD at learning rate 0.01 without momentum, classifier dropout 0, outer AdamW learning rate 0.001 and weight decay 0. Random crops apply to both sides, with 4 support and 64 query crops per event and class; query and checkpoint selection both use Days 8/11 (`selection_validation_days=null`). All eligible examples are used, evaluation runs every 10 steps, checkpointing every 25 steps, label smoothing is 0.1, and gradient clipping is 5. The inner checkpoint segment length is 20 updates, shuffles are fresh (`inner_shuffle_seed=null`), diagnostic logging defaults to 0, and final support BatchNorm recalibration is enabled. All seven mice remain supported through an explicit `targets` configuration or CLI override.

`inner_weight_decay` (default 0) adds an L2 pull toward the fixed unlearned reference weights to every inner SGD step, after loss-gradient clipping: g ← clip(∇ℓ) + λ(θ − θ₀). It applies identically to meta inner fits and per-mouse fits. θ₀ is detached, so the pull carries no hypergradient of its own, and replay reproduces the forward trajectory exactly. It is intended to keep long inner trajectories in a contracting basin rather than drifting toward memorization, which made 14,550-step X2 fits (K=40, 30 epochs, learning rate 0.01) chaotic. Support diagnostics never shorten training. The final configured epoch supplies the classifier, with no best-epoch rollback. Final per-mouse fitting uses the same epoch budget and optimizer. The only inner optimizer is `inner_optimizer="sgd"`, without momentum. Adam and its first/second-moment state are removed from inner fitting; the outer frontend optimizer remains AdamW. Old configurations with `inner_momentum=0` are accepted and converted to plain SGD; nonzero momentum and the Adam inner optimizer are rejected. Nonzero `outer_weight_decay` applies decoupled decay only to frontend weight matrices; biases, GroupNorm parameters, and the pooled auxiliary CNN are excluded.

Every meta-training step prints a `query_task` record after each mouse's inner fit and a `query_step` summary after all mice finish. These use all query-day windows by default, before that step's outer update. Records include the label-smoothed outer loss, accuracy at threshold 0.5, sample counts, eligible query-event counts, and outer task weights; training query means use query-event-proportional task weights. Refit evaluation metrics and checkpoint selection retain mouse-mean aggregation. Query metrics supervise the frontend and never control inner stopping. Per-task metrics are saved in `inner/step-*/task-*.json`; per-step summaries are saved under `query` in `history.json`.

At evaluation intervals, `evaluation_step` prints valid, query, and test means for the updated frontend after fresh support-only CNN fitting. History stores these as `valid`, `query_refit`, and `test`, and reports are saved under `valid/`, `query/`, and `test/step-*.json`. The refitted selection loss used for checkpoint selection is ordinary unsmoothed predictive log loss. It differs from the smoothed, pre-update training query loss; both timings and definitions are explicit. The standalone log-STFT algorithm uses the same refitting seeds/budget and predictive metrics.

Historical smoothness diagnostics used different settings from the current configuration. With SGD at learning rate 0.01 and momentum 0.9, X2 query loss changed by 0.01-0.2 under frontend perturbations of norm 1e-6 to 1e-4 (5 to 50 epochs), so the hypergradient described local chaos rather than a descent direction; even a fixed shuffle did not make the objective usable. With learning rate 0.001 and no momentum (20 epochs), the loss along the negative hypergradient follows the first-order prediction up to a parameter step of 1e-2 and keeps decreasing at 1e-1. Those observations do not establish smoothness for the current learning rates and event-based crops; a short diagnostic slice is needed to assess that configuration. This is a fixed-trajectory bilevel representation objective. Completing the configured epochs is a training budget, not a claim of convergence. Each fit records `epochs_requested`, `epochs_run`, and `stop_reason="fixed_epochs"`. Every update in the fixed trajectory supplies the query hypergradient; there is no loss-dependent stopping decision. This implementation does not use implicit differentiation or first-order/truncated approximations. Forward fitting saves detached CNN/BatchNorm-state boundaries and RNG states in CPU memory; plain SGD needs no optimizer-moment state. The running-statistic EMA is part of the differentiable inner state; query gradients include the dependence on support-fitted statistics. Replay restores the statistics for each segment and leaves saved/reference statistics unchanged. Backward replays one segment at a time, carrying parameter and BatchNorm-state adjoints between segments and accumulating the support-feature gradient. GPU activation memory depends on segment length; CPU checkpoint storage grows with trajectory length. The implementation computes the first outer gradient and does not support differentiating that outer gradient again.

This branch (`meta-learning/waveform-conv-frontend`) removes the precomputed STFT. The only fixed preprocessing is per window: remove the window mean, resample 500 to 256 Hz (`resample_poly`), and zero-pad to a multiple of 256 samples, giving 28,672 samples for a 112-second window. Fixed-window inference and the log-STFT control cache these waveforms (`cache/waveforms`, keyed on `meta_learning/waveform.py` only, so frontend edits do not invalidate the cache). No dataset or task statistics are fitted before the frontend.

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

`augment_random_crop` (currently true) samples real 112-second windows every outer step. Positive crops use each eligible event's complete 7-3 minute interval (128 s of legal start positions). Negative crops sample from **all eligible seizure-free intervals of the same mouse and recording day**, independently of fixed negative windows. Every interval is shortened by 112 s to obtain its legal start range; intervals shorter than a window are excluded. The sampler draws uniformly over the union of legal 256-Hz start positions, so intervals are weighted by the number of legal starts rather than chosen equally. Existing ten-minute seizure margins, positive-interval exclusions, recording boundaries, and `negative_recording_scope="annotated_only"` remain in effect. The negative window stays inside one safe interval and one recording.

`meta_learning/crops.py` caches one positive context per event under `cache/crops` and all eligible negative intervals under `cache/negative-crops`. Variable-length negative intervals share a flat memory-mapped array, with a range index and source coordinates. Long intervals are centered and resampled in aligned blocks with overlapping filter context, keeping preparation memory bounded. The final 112 s crop removes its own mean. Crop loading skips fixed-window waveform caches and reads only requested support/query days, including Day 11 in the current query configuration. Existing caches are retained, and the new format uses a new content key. Interval counts, legal-start counts, and the negative pool fingerprint are recorded in crop provenance.

`support_crops` and `query_crops` count crops **per event and class**, independent of the fixed-window count. The current configuration draws 4 positive and 4 negative support crops per event, and 64 positive and 64 negative query crops per event. Each event's negative count is a bookkeeping budget: those crops can occur anywhere in the same day's eligible negative pool. Context and position draws use the checkpointed episode RNG; a fixed `inner_shuffle_seed` fixes those draws too. Optional `support_per_class` and `query_per_class` caps limit the generated examples per class; without crops, they limit fixed windows. `random_crop_scope` (`both`, `query`, or `support`) chooses which side is cropped. Query loss averages within each represented event and then across events. The pooled control uses the same examples but averages within each event and then equally across represented support and query events within each mouse task. Both algorithms weight task gradients by eligible query-event counts, so crop counts do not change task or event weights. With optional per-class caps, the task coefficient still uses the full eligible query-event count while its loss averages represented events.

Per-mouse fitting uses the same event-based support crop policy, `inner_epochs`, and optimizer as meta inner fitting. Its negative-interval choices, crop positions, and shuffles use a fixed per-mouse seed (`seed + 10000 + mouse index`); its report records `support_crops` and `support_crop_unit="per_event_per_class"`. Plan crop contexts over the full support before batching, so an event crossing a batch boundary is counted once. Crops are drawn once per fit and reused throughout its complete shuffled epochs. With E events and K support crops, a full fit trains on 2 × E × K examples per epoch. Compared with the fixed-window recipe's four positives and four negatives per event, each epoch is approximately K/4 times as long; `inner_epochs` is not rescaled. Independent selection, query-refit evaluation, test, and the standalone `log_stft` algorithm always use fixed windows. Sample tables and fingerprints remain unchanged. Sampling negatives from complete safe intervals changes the training objective and requires a new run directory; previous runs retain their source snapshots. Crops from one event are correlated and do not add independent events.

Random initialization makes the first outer updates large in function space: in a two-step X2 smoke run at `outer_lr=0.001`, one update moved Day 8 query accuracy from 0.50 to 0.83 while Day 11 selection AUC was 0.47. That historical run used a separate Day 11 selection set; the current configuration uses both days for query and selection.

The task CNN is one-dimensional: AxisBatchNorm(channels) → Conv1d(channels→16, k5, s2) → ReLU → max-pool 2 → AxisBatchNorm → Conv1d(16→32, k3) → ReLU → max-pool 2 → AxisBatchNorm → Conv1d(32→64, k3) → ReLU → max-pool 2 → flatten (64 × 4) → dropout → Linear(256→128) → sigmoid → dropout → Linear(128→2). It mirrors the former 2-D MouseLFP CNN's stages but convolves time only; channels are fully mixed by each convolution, and flattening keeps where in the window a pattern occurs. It has about 46,000 parameters with the 64-channel learned frontend and 44,000 with the 40-band log-STFT control. This replaces the 2-D CNN used by all earlier runs, so earlier results are not directly comparable.

Frontend computation uses non-reentrant activation checkpointing with `frontend_batch_size` complete windows per chunk (32 by default). Set `frontend_batch_size` in the JSON configuration independently of the CNN training `batch_size`; for example, use 128 frontend windows while retaining CNN minibatches of 32. The setting is applied to meta/pooled feature extraction, final support feature extraction, and inference, and is recorded in run provenance. Final support crop draws keep their existing sampling order when this setting changes. Larger frontend chunks trade higher peak activation memory for fewer frontend calls; CNN minibatch updates and BatchNorm training batches are unchanged. This bounds retained convolution activations while preserving gradients through all support/query windows and the complete inner training trajectory. The pooled control and final per-mouse fitting use the same neural architecture. A neural frontend does not establish that the resulting bilevel objective is globally smooth or that query loss will decrease monotonically; the task CNN still uses ReLU/max-pooling and the inner optimizer retains its own dynamics. Use a new run directory; historical runs remain reproducible with their saved source snapshots.

CNN BatchNorm uses normal training behavior: support minibatches supply batch statistics and update per-task running means/variances, while query/test inference uses eval mode with support-fitted statistics. With `batchnorm_recalibration=true` (the default), every inner, per-mouse, and baseline fit ends with one training-mode pass over the complete support at zero retention, replacing the running EMA with the exact support mean/variance of the final weights. The EMA starts from (0, 1) and retains 0.99 per update, so after t updates it still holds 0.99^t of its initial value; for short fits (for example 49 updates per X2 epoch) this biased the first layer by several standard deviations. The recalibration pass is differentiable, so the hypergradient includes the dependence of these statistics on the frontend; it uses support data only. Reports record `calibrated_support_loss` and `calibrated_support_accuracy` next to the EMA-based per-epoch diagnostics. Each fresh CNN resets its running statistics; fitted per-mouse statistics are saved with the checkpoint. The existing axis normalization retains its Keras-compatible settings (retention momentum 0.99, epsilon 0.001, biased variance). Pooled representation training updates its auxiliary CNN statistics normally, then discards that CNN before support-only per-mouse fitting. Inner gradient replay includes the complete running-statistic trajectory and CNN state. `classifier_dropout` applies consistently to support, pooled, and final per-mouse CNN training and defaults to zero; query evaluation and inference always disable dropout. Query predictions therefore do not depend on unrelated query-batch companions or mutate shared normalization buffers. The backbone architecture retains its original dropout modules; training config controls their probability.

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

`configs/meta.json` controls finite budgets, learning rates, task sizes, fixed CNN epoch budgets, smoothing, and gradient clipping. Defaults use 100 outer steps and one X2 task per step, with 30 complete epochs for each task and final per-mouse CNN. Explicit JSON settings override Config class defaults (50 epochs and 40 support crops). Task count is derived from `targets`; there is no configurable mouse-sampling budget. This is substantially more work than the former three-step inner loop. `inner_checkpoint_steps=5` checkpoints segments of up to five minibatch updates; zero disables checkpointing for small diagnostic comparisons. `inner_log_every=0` evaluates support diagnostics at initialization and the final epoch only, then reports the final summary. A positive value evaluates and logs the first epoch, that epoch interval, and the final epoch. Reports store only measured diagnostic epochs, while `epochs_run` and `selected_epoch` retain the full configured budget. Final BatchNorm calibration and its support diagnostic are always retained; per-mouse query metrics and every-step query summaries are still printed. Use distinct output directories for different configurations or seeds.

### Resume and query replay

```bash
OUTPUT_DIR="$PWD/results/waveform-conv-meta" ./run_meta.sh train --config configs/meta.json --resume
OUTPUT_DIR="$PWD/results/waveform-conv-meta" ./run_meta.sh validate
```

Resumption restores model, optimizer, task-sampler state, PyTorch CPU/CUDA RNG state, history, and selected checkpoint state. Configuration, source identities, code, and runtime must match. An initial step-zero checkpoint allows interruptions during the first long inner fit to resume. Work after the most recent completed outer checkpoint is replayed; partial inner trajectories are not resumed in place. A completed run with the same configuration exits without further optimization.

Validation replay (`validate`) loads the selected adapted checkpoints and their saved identity scales. With the current configuration it scores query Days 8/11, writing `evaluation/query/`. With independent valid days configured it also writes `evaluation/valid/`. The combined `evaluation/validation/result.json` describes valid, with a nested `query` report (query is the root when no valid days are configured); the days are never merged into a query score. It performs no personalization, gradient updates, or fitting.

### Replay held-out evaluation

Meta/pooled training scores test at evaluation intervals; the standalone baseline scores test after its single per-mouse fit. To replay the selected checkpoint:

```bash
OUTPUT_DIR="$PWD/results/waveform-conv-meta" ./run_meta.sh test
```

This loads the selected per-mouse checkpoints and identity scales, writes test predictions, and leaves model-selection records unchanged. The Python inference interface defaults to `--split test`; use `--split valid` or `--split query` for a single role, or legacy `--split validation` to replay both separately. Neither replay command trains a CNN or fits statistics.

### Frontend feature distributions

Use the selected adapted checkpoints to plot each mouse's full frontend output and classification errors:

```bash
python scripts/plot_frontend_tsne.py --run-dir results/waveform-conv-meta \
  --data-dir /data/users/jianchen/data/seizure_data \
  --cache-dir /data/users/jianchen/data/seizure_data-cache/meta-learning --device cuda
```

Run in the environment matching `requirements.txt`; `--prepared-dir` defaults to `CACHE_DIR/prepared`. The script verifies and imports the run's saved source snapshot, freezes the frontend and fitted CNN, and checks replayed query/valid/test predictions against the saved predictions. No fitting or BatchNorm recalibration occurs. It uses every fixed 112-second reference window, including support reference windows rather than the random crops used during fitting. `--targets X1,X2` optionally restricts the output to selected mice with adapted checkpoints.

Each mouse gets a separate t-SNE layout of flattened frontend features (currently 64 × 56), preceded by 50-component PCA fitted only on support reference features, with no extra scaling. Labels never enter either projection. Support, query, optional valid, and test panels share that mouse's coordinates; green/blue indicate true positive/negative labels and red crosses indicate classification errors at threshold 0.5. Test participates only in the exploratory layout. Coordinates across different mice are independent, and t-SNE cluster sizes and distances do not quantify predictive performance.

PNG/PDF/SVG plots, full features, per-window coordinates/probabilities, FP/FN counts, a script snapshot, and verification metadata go into a new timestamped directory under `RUN_DIR/visualizations/`. `--output` can select a new directory; existing directories are never overwritten. The training configuration and historical run artifacts remain unchanged.

### Transfer the frozen frontend to unseen mice

```bash
python scripts/transfer_frontend.py --source-run results/waveform-conv-meta \
  --run-dir results/transfers/unseen-mice --targets X4,X10,X15 --device cuda
```

This finite job uses the source run's verified saved implementation and selected frontend, trains one fresh CNN per target on Days 0-7, and reports query Days 8/11 and test Days 12/13. The frontend remains frozen; target query/test metrics never select weights. The complete epoch budget, support crop policy, optimizer, dropout, and batch sizes come from the source run's configuration. Per-mouse fit seeds are `seed + 10000 + target position` in the supplied target order. Outputs include checkpoints, predictions, source snapshots, and `report.json`/`report.md` in a new directory. Data/cache options match the visualization script; existing output directories are rejected.

### Short integration check

```bash
./run_meta.sh check
```

The finite synthetic checks cover crop counts, temporal isolation, exact checkpoint replay, inference-only evaluation, diagnostic schedules, and cache sharing. They do not launch a real-data training run. Use a separate configuration and run directory for any explicitly budgeted real-data smoke check.

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
- `convergence.png` and `convergence.svg`: automatically drawn after successful training (including a completed resume). One plot shows accuracy versus outer loop with four lines: Support (Days 0-7), Query (configured query days), Valid (configured selection days), and Test (Days 12/13), plus a marker for the selected checkpoint. All four use the same support-fitted CNNs at each recorded evaluation step, averaged equally across mice; support accuracy uses the final recalibrated BatchNorm statistics when available. Query values are post-update refit metrics. Test curves are reporting only. Missing metrics are omitted, and standalone log-STFT has only one point per split at step zero. Regenerate figures without training with `python -m meta_learning.plotting results/waveform-conv-meta`.
- `baseline/log_stft/`: query/test reports, per-mouse CNN checkpoints, and predictions, produced only by `algorithm="log_stft"`. Meta and pooled runs do not create this directory.
- `inner/step-*/task-*.json`: support-only loss curves, completed epochs/updates, stopping reason, and task dates/counts. Outer history records the fixed epochs per task, completed tasks, and optimizer updates. Per-mouse query/eval reports and selected checkpoints retain the same CNN-fit diagnostics.
- `selection.json` and `selected/step-*/`: selected shared model, adapted per-mouse models/scales, `predictions_{query,valid,test}.csv` and matching metrics; selection uses selection-day loss only.
- `evaluation/query/`, `evaluation/valid/`, or `evaluation/test/`: inference-only reports and per-window predictions.

Source preparation and finite-window quality audits establish identity, alignment, and the recorded audit findings. Fixed-window signal audits are written only when those windows are loaded. Crop preparation checks that every materialized, resampled context is finite; it does not audit complete continuous recordings. Inspect actual signal artifacts before attributing model behavior to physiology. Independent events and recording days, rather than overlapping window counts, determine the available evidence.

Historical experiment records remain available locally and contain their original source snapshots. The earlier `results/meta` run optimized CNN initialization as well as the frontend and used a three-step inner loop; it does not measure this frontend-only epoch objective. Use a new output directory for the changed algorithm. Historical runs can be replayed with their saved source versions and cannot resume under this changed algorithm. The fixed train-support/validation-query/test-eval protocol requires a new output directory and cannot resume former rolling-within-train runs. Restoring training-mode BatchNorm also requires a fresh run; frozen-statistics checkpoints cannot resume under this code. The launcher now defaults to `results/waveform-conv-meta-event-weighted` to keep the query-event-weighted objective separate from historical equal-mouse runs. The former four-draw mouse-sampling schedule also requires a new output directory. Old `min_support_days`, `query_days`, `tasks_per_step`, `inner_steps`, `adaptation_steps`, `adaptation_lr`, `inner_min_epochs`, `inner_max_epochs`, `inner_patience`, and `inner_relative_tolerance` configuration fields are rejected; `inner_epochs` controls the fixed budget of both CNN fitting stages. Runs using the former early stopping policy require a new output directory.

Query-event-proportional outer weighting requires a new run directory and cannot resume the former equal-mouse objective. `run.json`, `episodes.json`, per-task reports, and training history record event counts and normalized task weights. For the current seven-mouse query split, counts are X1=24, X2=26, X4=6, X9=53, X10=7, X14=38, X15=3 (157 total). The new three mice jointly receive 16/157 of the outer objective.

## Source layout

```text
meta_learning/      Frontend, fixed split tasks, meta/pooled objectives, adaptation, and evaluation
models/            CNN backbone and axis normalization
configs/           Current finite training configuration
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
