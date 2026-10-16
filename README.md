# MouseLFP cross-day meta-learning

Learn a spectral frontend that helps a classifier generalize from earlier to later recording days of the same mouse. Tasks are shared across mice; the objective directly measures cross-day generalization.

This branch provides finite preparation, task planning, meta-training, personalization, validation, resume, and explicit final-test commands. Historical experiment directories and supervised reference implementations remain available.

## Data and partitions

Private input contains 168 continuous recordings and `seizure_epochs_pitime_seizure_w15_t0.3_min5_gap10.parquet`, with 2,773 annotated intervals. The modeled mice are X1/X2/X4/X9/X10/X14/X15.

| Purpose | Days | Permitted use |
| --- | --- | --- |
| Training | 0-7 | Episode support/query, pooled training, personalization, fitted statistics |
| Validation | 8/11 | Checkpoint selection and inference |
| Test | 12/13 | Explicit final inference using selected adapted checkpoints |

Source recordings and annotations stay read-only. `prepare.py` records source identities and verifies row/timestamp alignment independently of model code. Existing source inventories can also be read; their old trainer-code hashes do not constrain this branch. New inventories use version 3 and contain source identity and day partitions only.

The shared sample recipe in `sample_pipeline.py` currently uses 112-second windows, a 7-3-minute positive interval, a 56-second grid plus reference offsets, balanced same-mouse/day negatives, ten-minute negative seizure margins, and existing event-history/postictal exclusions. Actual offsets are 0, 128, 56, and 112 seconds from the positive interval start. Training, validation, and test share these definitions. Changes to sampling alter the evaluated task and must be reported.

## Learning procedure

1. Construct rolling episodes within each mouse using only Days 0-7. Every support day precedes every query day. Whole recording days remain together, so overlapping windows cannot cross support/query boundaries.
2. Sample mice uniformly, then episodes uniformly within the mouse. Sample both classes without replacement from each side. Estimate one power scale from the sampled support windows and reuse it for query windows.
3. In the inner loop, adapt CNN parameters with differentiable SGD on support labels. The shared frontend stays fixed during these inner updates.
4. In the outer loop, differentiate query loss through the inner updates and update both the shared frontend and initial CNN parameters. This includes the higher-order frontend-to-adaptation gradient.
5. At validation intervals, independently personalize a copy of the shared model for each mouse using all its Days 0-7. Fit its power scale on those training samples only. Use a fixed number of Adam steps; freeze the frontend by default.
6. Select the shared checkpoint and its adapted per-mouse models by mean validation accuracy at threshold 0.5, breaking ties with lower mean validation log loss. Selected adapted checkpoints are retained exactly as evaluated.

The spectral frontend learns 520 parameters for Gaussian frequency bands, power compression, and positive temporal smoothing. Fixed preprocessing centers windows, resamples from 500 to 256 Hz, and computes a two-second STFT power basis over 1-40 Hz. Only this sample-local fixed basis is cached. Phase is excluded from this representation. The existing CNN consumes `(1, 1, 55, 114)` features.

BatchNorm running means and variances remain fixed in all meta, pooled, and personalization stages; affine normalization weights remain trainable. Dropout is enabled for support/pooled/personalization training and disabled for query evaluation and inference. Query predictions therefore do not depend on unrelated query-batch companions or mutate shared normalization buffers.

## Setup and Docker commands

Dependencies remain in `requirements.txt`; the existing `seizure-prediction-cnn:pytorch` image can execute the mounted source. Build a fresh image when needed:

```bash
docker build -t seizure-prediction-cnn:pytorch .
./run_meta.sh check
./run_meta.sh prepare
```

`run_meta.sh` uses these defaults:

- `DATA_DIR=/data/users/jianchen/data/seizure_data`
- `CACHE_DIR=/data/users/jianchen/data/seizure_data-cache/meta-learning`
- `PREPARED_DIR=$CACHE_DIR/prepared`
- `OUTPUT_DIR=$PWD/results/meta`
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
OUTPUT_DIR="$PWD/results/meta" ./run_meta.sh train --config configs/meta.json
OUTPUT_DIR="$PWD/results/pooled" ./run_meta.sh train --config configs/meta.json --algorithm pooled
```

The pooled control uses the same frontend, CNN, task samples, support-fitted scaling, outer-step count, and final-personalization budget. It minimizes supervised loss on concatenated episode support/query examples without an inner loop. Meta-training performs additional forward/backward operations; equal outer-step counts are not equal wall-clock compute.

Compare completed runs without launching additional training:

```bash
python -m meta_learning.compare results/meta results/pooled --output results/comparison.json
```

The comparison rejects differences in configuration other than the algorithm, source versions, runtime, training samples, or validation sample fingerprints. It also reports optimization/validation wall time and peak allocated GPU memory for each run.

`configs/meta.json` controls finite budgets, learning rates, task sizes, adaptation, smoothing, and gradient clipping. Defaults use 1,000 outer steps, four tasks per step, three inner SGD steps, and 100 Adam steps for per-mouse personalization. These are new workflow defaults, not the old per-mouse wall-clock search allowance. Use distinct output directories for different configurations or seeds.

### Resume and validation replay

```bash
OUTPUT_DIR="$PWD/results/meta" ./run_meta.sh train --config configs/meta.json --resume
OUTPUT_DIR="$PWD/results/meta" ./run_meta.sh validate
```

Resumption restores model, optimizer, task-sampler state, PyTorch CPU/CUDA RNG state, history, and selected checkpoint state. Configuration, source identities, code, and runtime must match. Work after the most recent completed checkpoint is replayed. A completed run with the same configuration exits without further optimization.

Validation replay loads the selected adapted checkpoints and their saved normalization scales. It performs no personalization, gradient updates, or fitting.

### Explicit final testing

After choosing a run and ending model selection:

```bash
OUTPUT_DIR="$PWD/results/meta" ./run_meta.sh test
```

This is the only launcher action that reads test signals. It uses the selected per-mouse checkpoints and training-fitted scales, writes test predictions, and leaves model-selection records unchanged. The Python interface requires explicit `--split test` for this operation.

### Short integration check

```bash
OUTPUT_DIR="$PWD/results/meta-smoke" ./run_meta.sh train --config configs/smoke.json
OUTPUT_DIR="$PWD/results/meta-smoke" ./run_meta.sh validate
```

The smoke configuration uses X15, two outer steps, one inner step, and two personalization steps. It verifies execution rather than estimating model performance. Keep it separate from full-cohort comparisons. The pooled smoke control uses the same configuration plus `--algorithm pooled`.

## Local execution

Use an environment matching `requirements.txt`. Examples:

```bash
python prepare.py --output prepared
python train.py --prepared-dir prepared --cache-dir /path/to/cache --run-dir results/meta --config configs/meta.json
python evaluate.py --prepared-dir prepared --cache-dir /path/to/cache --run-dir results/meta
python -m unittest discover -s tests -v
```

Data paths default to `SETTINGS_MouseLFP.json`. The Docker launcher supplies container paths explicitly.

## Artifacts and reproducibility

Each run retains:

- `run.json`, `config.json`, and `source/`: configuration, source inventory identity, code hashes/snapshots, sampling policy, normalization/BatchNorm policy, and runtime.
- `episodes.json` and `data/X*/`: chronological task definitions, event eligibility, split class counts, sample coordinates/fingerprints, and sampled signal-quality audits.
- `history.json`, `status.json`, and `checkpoints/last.pt`: losses, gradient norms, validation summaries, progress, and resumable training state.
- `selection.json` and `selected/step-*/`: selected shared model, adapted per-mouse models/scales, validation predictions, and metrics.
- `evaluation/validation/` or `evaluation/test/`: inference-only reports and per-window predictions.

Source preparation and finite-window quality audits establish identity, alignment, and the recorded audit findings. Inspect actual signal artifacts before attributing model behavior to physiology. Independent events and recording days, rather than overlapping window counts, determine the available evidence.

The original supervised helpers in `train.py`, finite comparison scripts, legacy training entry points, and historical result folders are retained for reference. The primary `train.py` CLI now starts this meta-learning workflow. Prior records contain the source snapshots needed to reproduce their original environment and code.

## Method references

The outer representation objective follows the bilevel learning formulation in [Franceschi et al. (ICML 2018)](https://proceedings.mlr.press/v80/franceschi18a.html). The shared adaptable initialization is related to [MAML (ICML 2017)](https://proceedings.mlr.press/v70/finn17a.html). The spectral frontend is a compact Gaussian-band design; it retains a fixed STFT basis and is not an implementation of a raw-waveform LEAF or SincNet model.
