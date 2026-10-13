# MouseLFP autoresearch

An autonomous research project built around the current seven-mouse experiment, following the [Karpathy autoresearch](https://github.com/karpathy/autoresearch) workflow: an AI agent edits research code, trains models, compares validation results, and keeps improvements before iterating.

**The dataset, events, windows, and splits are fixed. Preprocessing, model architecture, and training methods are editable.**

## Main entry points

| File | Purpose |
| --- | --- |
| `prepare.py` | Fixed data preparation: raw signal windows, labels, and source coordinates |
| `train.py` | Agent-editable preprocessing, model, optimizer, and training loop; model code can also live in `models/` |
| `program.md` | Research agent objectives, constraints, and experiment loop |

`evaluate.py` is the fixed evaluator. It snapshots source code, trains each mouse separately, computes validation metrics, and records `keep/discard/crash` outcomes and the best experiment. Research uses the validation set; `finalize` evaluates the test set separately.

Use English throughout the project, including documentation, comments, prompts, log messages, and experiment descriptions.

## Fixed data protocol

| Item | Current protocol |
| --- | --- |
| Dataset | MouseLFP, X1/X2/X4/X9/X10/X14/X15, 978 events |
| Events per mouse | 137 / 259 / 59 / 267 / 61 / 169 / 26 |
| Positive interval | **5-1 minutes before onset**, retaining the existing event eligibility and other-seizure exclusions |
| Positive exclusion | Other seizures from onset through **5 minutes after their end**; reject the entire event if its interval overlaps |
| Negative exclusion | Every annotated seizure from **10 minutes before onset** through **10 minutes after its end**, checked across files |
| Negative matching | Unique windows from the same mouse and day, retaining the first eight entries of the existing ten-window grid |
| Raw window | **28 seconds, 500 Hz, 14,000 samples** |
| Window counts | **8 reference windows** per event per class; **14-second** training stride, **23 training windows** per class |
| Train / validation / test | Days **0-7 / 8,11 / 12,13** |
| Event counts | **649 / 168 / 161** |
| Balanced window counts | **29,854 / 2,688 / 2,576** |

`research_protocol.py` preserves the original trainer's window geometry and matching functions. `protocol.lock.json` records fixed code hashes, the annotation hash, and source file metadata. Prepared arrays, labels, and provenance tables have separate SHA-256 checks. Source data is mounted read-only.

Raw windows are stored as float32 and preprocessed by the research code. The initial recipe uses the original 256 Hz resampling, log-STFT, SeizureCNN, Adam, learning rate `5e-4`, batch size 32, and no additional normalization. It reloads the checkpoint with the highest validation accuracy at threshold `0.5`, breaking accuracy ties with lower validation loss. STFT and resampling are editable preprocessing choices; raw window durations and coordinates remain fixed.

## Quick start

The local Docker image uses Python 3.11, PyTorch 2.5.1, and CUDA 12.4. Rebuild it when needed:

```bash
docker build -t seizure-prediction-cnn:pytorch .
```

Prepare all raw windows once, then run the baseline when research is requested:

```bash
./run_research.sh prepare
export OUTPUT_DIR="$PWD/autoresearch-runs/my-study"
./run_research.sh experiment --description 'baseline original STFT CNN Adam' > run.log 2>&1
cat "$OUTPUT_DIR/results.tsv"
cat "$OUTPUT_DIR/best.json"
```

The default training budget is **300 seconds per mouse**, or up to **35 minutes for seven mice, plus startup, preprocessing, and evaluation overhead**. The initial recipe also caps training at 100 epochs. Comparisons within a session use the same data, seed, budget, device, and runtime environment, recorded in `session.json`. Changes to these conditions require a new session.

Default paths:

- `DATA_DIR=/data/users/jianchen/data/seizure_data`
- `PREPARED_DIR=/data/users/jianchen/data/seizure_data-cache/autoresearch-raw-v1`
- `OUTPUT_DIR=<project>/autoresearch-runs/default`
- `IMAGE_NAME=seizure-prediction-cnn:pytorch`

`TRAIN_CPU_THREADS` defaults to 4. Use `--device cpu` for CPU experiments. Explicit CUDA requests fail if GPU access is unavailable. Use `--dry-run` to inspect the launch command.

To start agent-driven research, send the agent:

> Read program.md, establish a baseline, then autonomously iterate on preprocessing, model architecture, and training methods while keeping the dataset, window parameters, and splits fixed. Record each validation result and keep improvements.

The launcher runs one experiment. The AI agent reads `program.md`, generates hypotheses, and edits code.

## Comparison and recovery

The objective is the **arithmetic mean of the seven mice's window-level validation accuracies at threshold `0.5`**, with higher scores preferred. The first successful experiment is `keep`. Later experiments are `keep` only if the mean improves by more than `1e-6`; other successful experiments are `discard`, and failures or timeouts are `crash`. AUC remains a supporting metric in result JSON files. Accuracy-based experiments require a new session; previous AUC-based records remain archived.

```bash
# Run a new experiment after editing train.py or models/*.py.
./run_research.sh experiment --description 'train-only frequency z-score' > run.log 2>&1

# Restore the best recipe after a discard; current code is backed up first.
./run_research.sh restore
```

The initial code also provides `train_frequency_zscore`, selectable through `PREPROCESS` in `train.py`. Fit all statistics on the training set only. Filtering, time-domain input, wavelets, other spectral representations, and arbitrary models are supported through the `fit`, `load_predictor`, and `predict_proba` interfaces.

For training failures, inspect `$OUTPUT_DIR/experiments/<id>/status.json` and `X*/train.log`. Each experiment retains source code, weights, and predictions. Failed experiments preserve the previous best result. Experiments within a session run sequentially.

Dependencies and the Docker environment are editable. Rebuild the image and establish a matching baseline in a new session. Runtime versions and dependency recipes are recorded with experiments.

## Smoke checks

Use a separate X15 dataset and short budget. Results are labeled `smoke`:

```bash
export PREPARED_DIR=/data/users/jianchen/data/seizure_data-cache/autoresearch-raw-smoke-v1
export OUTPUT_DIR="$PWD/autoresearch-runs/smoke"
./run_research.sh prepare --targets X15
./run_research.sh experiment --targets X15 --budget-seconds 2 \
  --description 'pipeline smoke' > run.log 2>&1
```

Smoke checks reduce the subjects included in a run while preserving each selected mouse's original windows, events, and day splits. Full experiments use all seven mice and a separate session.

## Final test evaluation

After the search is complete and final evaluation is requested, run:

```bash
./run_research.sh finalize
```

The evaluator verifies and loads the source snapshot, weights, and preprocessing state selected by `best.json`. It evaluates the original test Days 12/13 and writes `final.json` and each mouse's `predictions_test.csv`. Starting final evaluation seals the session. Rerun the command to resume an interrupted evaluation. Test results are excluded from recipe selection.

## Local execution and checks

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python prepare.py --output autoresearch-data
python evaluate.py experiment --prepared-dir autoresearch-data \
  --run-dir autoresearch-runs/local --device cuda --description baseline
python -m unittest discover -s tests -v
```

Checks cover compatibility with the original window geometry, rejection of fixed protocol edits, baseline preprocessing compatibility, training-only normalization statistics, valid predictions, and fixed session conditions.

## Original training entry points

`train_mouse_lfp.py` / `run_train.sh` retain the original seven-mouse training workflow and existing caches and results. `main.py` / `run.sh` retain training entry points for CHBMIT, FB, and Kaggle2014Pred.

The initial CNN is based on Truong et al. (2018), *Convolutional neural networks for seizure prediction using intracranial and scalp electroencephalogram*, Neural Networks 105, 104-111. DOI: 10.1016/j.neunet.2018.04.018.
