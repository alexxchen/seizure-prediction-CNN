# MouseLFP data-pipeline autoresearch

Explore how private continuous LFP recordings and event annotations should be turned into CNN inputs. Research changes sample construction, cleaning, signal representation, and layout while retaining the existing CNN and training controls.

Use English for project documentation, comments, prompts, logs, and experiment descriptions.

## Source data and research scope

The source folder contains 168 approximately 24-hour signal recordings and the annotation file `seizure_epochs_pitime_seizure_w15_t0.3_min5_gap10.parquet`, with 2,773 event intervals. Recordings contain `pi_time_ms`, `lfp`, and motion channels; their filenames include `interpolated`. This inventory and label alignment do not establish that the signals are clean.

| Fixed | Editable derived choices |
| --- | --- |
| Original recordings and annotation file | Event eligibility and positive/negative sample construction |
| Train Days 0-7, validation Days 8/11, test Days 12/13 | Window size, stride, prediction interval, and exclusion margins |
| Existing SeizureCNN and input grid `(1, 1, 55, 114)` | Cleaning, filtering, resampling, representation, frequency selection, and layout |
| Current trainer, optimizer, loss, seed, and comparison budget | Training-fitted normalization and input transformations |

The seven modeled mice are X1/X2/X4/X9/X10/X14/X15. The historical 978-event cohort, 28-second windows, and previous fixed sample counts are a past data recipe, not immutable raw data.

Each candidate's sampling policy, window size, cleaning, and feature transformations apply consistently to all three day splits. Normalization statistics are fitted on training inputs and reused for validation and test. This preserves a common processing rule while retaining genuine differences between recording days.

## Main files

| File | Role |
| --- | --- |
| `prepare.py`, `source_data.py` | Fixed source inventory, annotation alignment, read-only recording access, and day partitions |
| `sample_pipeline.py` | Editable sample construction, window policy, negative assignment, and signal cleaning |
| `input_pipeline.py` | Editable features, normalization, and CNN input layout |
| `train.py`, `models/cnn.py` | Fixed CNN and training controls |
| `evaluate.py` | Fixed scoring, source snapshots, experiment records, restoration, and final testing |
| `program.md` | Agent setup and autonomous experiment loop |

`protocol.lock.json` binds the raw source inventory, annotation hash, day splits, and fixed infrastructure. Both editable pipeline files are saved with each experiment. Candidate sample caches include source coordinates, labels, configuration, and checksums; original recordings stay read-only.

## Current control recipe

The CNN architecture remains unchanged. Controls retain Adam at `5e-4`, batch size 32, label smoothing `0.1`, and a 100-epoch cap. Checkpoints maximize validation accuracy at threshold `0.5`, using lower validation loss for ties.

The initial shared sample policy uses a 5-1-minute positive interval, 28-second windows, a 14-second grid, same-mouse/day negatives, ten-minute negative seizure margins, and the existing event-history exclusions. Dense windowing now applies equally to training, validation, and test. These sampling settings can be edited in `sample_pipeline.py`.

The current input defaults retain 256 Hz resampling, log-STFT, and training-fitted frequency normalization. `arrange_input` maps alternative representations to the fixed CNN grid. `time_frequency` is the default layout; `frequency_time` transposes the matrix before resizing it to the same grid. A changed window size propagates to all splits before feature construction.

## Setup and baseline

The local Docker image uses Python 3.11, PyTorch 2.5.1, and CUDA 12.4. Rebuild when needed:

```bash
docker build -t seizure-prediction-cnn:pytorch .
```

Prepare the source inventory once. This verifies source identity and annotation alignment; it does not generate a permanent fixed-window dataset:

```bash
./run_research.sh prepare
```

When research is requested, use a fresh session for the new scope:

```bash
export OUTPUT_DIR="$PWD/autoresearch-runs/data-study"
./run_research.sh experiment --description 'baseline shared data recipe' > run.log 2>&1
cat "$OUTPUT_DIR/results.tsv"
cat "$OUTPUT_DIR/best.json"
```

Defaults:

- `DATA_DIR=/data/users/jianchen/data/seizure_data`
- `PREPARED_DIR=/data/users/jianchen/data/seizure_data-cache/autoresearch-source-v1`
- `OUTPUT_DIR=<project>/autoresearch-runs/default`
- `IMAGE_NAME=seizure-prediction-cnn:pytorch`

`TRAIN_CPU_THREADS` defaults to 4. Use `--device cpu` for CPU execution and `--dry-run` to inspect commands. Each mouse has a 300-second training allowance; data assembly, preprocessing, and evaluation add overhead.

Earlier prepared-window directories and model-search results remain archived. Start a fresh session after changing the research scope or runtime; those historical results are not relabeled as new-pipeline runs.

To start the agent:

> Read program.md. Establish a baseline, then explore sample construction, cleaning, representation, normalization, and input layout. Apply each recipe consistently to train, validation, and test while preserving raw sources, day splits, the CNN, and training controls. Record validation accuracy and the associated data definition.

Project checks and documentation work leave the experiment loop inactive.

## Results and recovery

The primary score is the arithmetic mean of seven mice's validation accuracies at threshold `0.5`. AUC, sensitivity, specificity, and loss are supporting metrics. The first successful run is `keep`; later accuracy must improve by more than `1e-6` for promotion. Failed attempts preserve the previous best.

Each result records its sampling configuration, class counts, and evaluation-protocol fingerprint. Interpret scores with those definitions: changing eligibility, exclusions, or the prediction interval also changes the evaluated population or task.

```bash
# After editing sample_pipeline.py or input_pipeline.py:
./run_research.sh experiment --description 'shared window and input hypothesis' > run.log 2>&1

# Recover the best pair of pipeline files, backing up current code first:
./run_research.sh restore
```

Worker logs, sampling summaries, event selection, histories, checkpoints, and predictions are under `$OUTPUT_DIR/experiments/<id>/X*/`.

## Final evaluation

After ending the search and requesting testing:

```bash
./run_research.sh finalize
```

Final evaluation uses the selected sampler, input transformation, training-fitted statistics, and saved weights on Days 12/13. It writes `final.json` and per-window predictions and seals the session. Interrupted final evaluations can resume. Test metrics never guide candidate selection.

## Local execution

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python prepare.py --output autoresearch-data
python evaluate.py experiment --prepared-dir autoresearch-data \
  --run-dir autoresearch-runs/local --device cuda --description baseline
```

The original `train_mouse_lfp.py` / `run_train.sh` and `main.py` / `run.sh` remain available for reproducing earlier MouseLFP and CHBMIT/FB/Kaggle2014Pred workflows. Their old sample definitions are separate from the new research pipeline.

The CNN is based on Truong et al. (2018), *Convolutional neural networks for seizure prediction using intracranial and scalp electroencephalogram*, Neural Networks 105, 104-111. DOI: 10.1016/j.neunet.2018.04.018. The agent workflow follows [Karpathy autoresearch](https://github.com/karpathy/autoresearch).
