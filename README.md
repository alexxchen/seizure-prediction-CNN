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
training history, and metrics on the host. See [Docker instructions](docker/README.md).

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

Supported datasets: `CHBMIT`, `FB`, and `Kaggle2014Pred`.
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

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python docker/smoke_test.py --device cpu
python docker/smoke_test.py --device cuda \
  --cache-dir /data/users/jianchen/data/CHBMIT-cache \
  --data-dir /data/users/jianchen/data/CHBMIT
```

The Docker build runs CPU tests. The CUDA smoke test performs training using real
cached CHBMIT samples, saves a checkpoint, reloads it on CPU, and checks predictions.
The test suite covers electrode-axis normalization, SAME pooling, dataset shapes,
checkpoint round trips, old/new cache formats, and seizure split behavior.

Official installation reference: [PyTorch 2.5.1 builds](https://pytorch.org/get-started/previous-versions/#v251).
