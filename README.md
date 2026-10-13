# Convolutional neural networks for seizure prediction

PyTorch implementation of the patient-specific CNN from:

Truong, N. D., et al. (2018). "Convolutional neural networks for seizure prediction
using intracranial and scalp electroencephalogram." *Neural Networks* 105, 104–111.
DOI: 10.1016/j.neunet.2018.04.018.

This branch uses Python 3 and PyTorch with CPU and CUDA training. It preserves
the original electrode-axis normalization, 3D/2D convolution blocks, pooling,
Dropout, sigmoid hidden layer, Adam settings, and combined training/validation
loss used for checkpoint selection and early stopping. Model initialization and
optimizer arithmetic differ between frameworks, so a new training run is not
expected to reproduce the original weights bit for bit.

## Docker quick start

The pinned image uses PyTorch 2.5.1, CUDA 12.4, and cuDNN 9, and supports NVIDIA
A100 GPUs. Install Docker and the NVIDIA Container Toolkit, then build:

```bash
docker build -t seizure-prediction-cnn:pytorch .
```

Run GPU leave-one-seizure-out cross-validation for the downloaded CHBMIT subject 20:

```bash
./run.sh
```

Useful variations:

```bash
./run.sh --epochs 1 --max-folds 1  # Short GPU experiment
./run.sh --mode test             # Temporal training/test split
./run.sh --device cpu            # CPU execution
./run.sh --targets 20,21,23       # Requires those subjects to be downloaded
./run.sh --dry-run               # Print the Docker command
```

Data defaults to `/data/users/jianchen/data/CHBMIT`, caches to
`/data/users/jianchen/data/CHBMIT-cache`, and outputs to `docker-output-pytorch/CHBMIT/`.
Existing numeric caches written with Python 2 / hickle 2.1.0 are supported;
re-downloading or rebuilding them is unnecessary. Raw data is mounted read-only.
The script runs under your user ID and preserves checkpoints, predictions,
training history, and metrics on the host. Override paths with `DATA_DIR`,
`CACHE_DIR`, and `OUTPUT_DIR`; use `IMAGE_NAME` to select a different image.
The script forwards signals with `--init` and provides 1 GiB of shared memory
for optional DataLoader workers.

## Local execution

Use Python 3.11. Install the CUDA build of PyTorch, then the remaining pinned dependencies:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Configure the paths in `SETTINGS_<DATASET>.json`, or override them on the command line.
For CHBMIT, copy the three CSV files from `copy-to-CHBMIT-folder` to its data root.

```bash
python main.py --dataset CHBMIT --mode cv --targets 20 --device cuda \
  --data-dir /data/users/jianchen/data/CHBMIT \
  --cache-dir /data/users/jianchen/data/CHBMIT-cache \
  --result-dir results/CHBMIT
```

Supported datasets: `CHBMIT`, `FB`, and `Kaggle2014Pred`; mouse training uses the entry below.
`cv` holds out one seizure at a time. `test` holds out approximately the last third
of seizure groups in the preprocessed sequence. Existing caches retain their
stored seizure ordering. For newly generated data, the original file iteration
order is retained. Verify temporal ordering when interpreting a `test` experiment.
Defaults are 50 epochs for `cv`, 100 for `test`, batch size 32, and random seed 0.
Use `--device auto` to select CUDA when available, or explicitly request `cuda`,
`cuda:0`, or `cpu`. An explicit CUDA request fails if GPU access is unavailable.
Use `--reuse-checkpoint` to evaluate existing PyTorch weights instead of fitting.

PyTorch checkpoints use `.pt`; legacy Keras `.h5` model weights are not loaded.
Predictions retain the original CSV headers and four-decimal formatting.
Cross-validation checkpoints include fold numbers, and all checkpoints persist.

Official installation reference: [PyTorch 2.5.1 builds](https://pytorch.org/get-started/previous-versions/#v251).

## Seven-mouse LFP training

```bash
./run_train.sh
```

The single mouse training entry runs X1, X2, X4, X9, X10, X14 and X15 with
`train_mouse_lfp.py`, `utils/mouse_lfp.py` and the existing `models/cnn.py`.
The fixed protocol retains 978 events (137/259/59/267/61/169/26 respectively):

- Positive interval: **5–1 minutes before onset**, avoiding other seizures and
  **five minutes after their end**. Reject the whole event if its interval overlaps.
- Negatives: avoid every annotation from **ten minutes before onset to ten minutes
  after seizure end**, across source files; match unique windows within the same mouse/day.
- Input: **28 seconds**, 500 Hz resampled to 256 Hz; original log-STFT features
  of shape `(N, 1, 1, 55, 114)`, with no added normalization.
- Each event supplies eight reference windows per class. A 14-second training
  grid gives 23 windows per class per training event; validation/test keep eight.
- Train Days 0–7; validation Days 8/11; test Days 12/13. Event counts are
  649/168/161, with 29,854/2,688/2,576 balanced windows.
- Existing CNN/Adam, seed 0, batch size 32, 100 epochs, early stopping disabled;
  select and reload the checkpoint with minimum validation loss.

Data and annotation paths are configured in `SETTINGS_MouseLFP.json`. Each launch
writes a fresh `docker-output-mouse-training/<UTC-time>-<PID>/MouseLFP/` directory,
including logs, configurations, source-window coordinates, exclusion checks,
histories, checkpoints, predictions and summary metrics. Original caches remain
compatible and previous result directories are retained.

```bash
./run_train.sh --dry-run
./run_train.sh --targets X15 --epochs 1  # Short training check
```

`DATA_DIR`, `CACHE_DIR`, `OUTPUT_DIR`, `IMAGE_NAME` and `TRAIN_CPU_THREADS` override
runner paths/image/CPU threads. CLI options are limited to subjects, device,
epochs, batch size, seed and paths; the data protocol stays fixed.
Use a fresh output directory for the simplified code; an incompatible saved
run configuration is rejected. `run.sh`/`main.py` retain the original project
training entry for the original datasets, while MouseLFP uses `run_train.sh`.
