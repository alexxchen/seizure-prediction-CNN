# PyTorch Docker environment

The image pins PyTorch 2.5.1, CUDA 12.4, cuDNN 9, and the Python dependencies in
`requirements.txt`. The build runs CPU model/preprocessing tests and a CPU
training/checkpoint smoke test. NVIDIA A100 training uses the host driver through
the NVIDIA Container Toolkit.

Build from the repository root:

```bash
docker build -t seizure-prediction-cnn:pytorch .
```

The launch script defaults to CHBMIT subject 20 and CUDA:

```bash
./run.sh
./run.sh --epochs 1 --max-folds 1
./run.sh --mode test
./run.sh --device cpu
./run.sh --device cuda:0 --batch-size 64
./run.sh --help
```

For CUDA, the script adds `--gpus all`. CPU execution omits GPU access entirely.
Model inputs, labels, parameters, and the loss run on the selected PyTorch device.
Only each batch is transferred to the GPU, and existing float64 cached features
are converted to float32 sample by sample. The script uses `--init` for signal
forwarding and 1 GiB of shared memory for optional DataLoader workers.

Default host paths:

| Purpose | Host path | Container path |
| --- | --- | --- |
| Raw CHBMIT data | `/data/users/jianchen/data/CHBMIT` | `/data/CHBMIT` (read-only) |
| Preprocessed cache | `/data/users/jianchen/data/CHBMIT-cache` | `/cache/CHBMIT` |
| Experiment outputs | `<repository>/docker-output-pytorch` | `/output` |

The script creates the cache and output directories and runs the container with
your user ID. Override defaults with environment variables:

```bash
DATA_DIR=/path/to/CHBMIT CACHE_DIR=/path/to/cache OUTPUT_DIR=/path/to/output ./run.sh
IMAGE_NAME=seizure-prediction-cnn:pytorch ./run.sh
```

`DEVICE` sets the default device; `CHBMIT_TARGETS` sets the default subjects.
Download additional subjects before using `--targets 20,21,23`.
`--dry-run` prints the complete launch command without starting training.

Outputs under `docker-output-pytorch/CHBMIT/` include:

- `checkpoints/weights_20_cv_fold1.pt`: best combined training/validation loss.
- `preictal_20_1.csv`, `interictal_20_1.csv`: held-out class probabilities.
- `history_20_cv_1.json`: per-epoch losses and accuracy.
- `metrics_cv.json`: completed fold AUC values and checkpoint paths.

The temporal test checkpoint is `checkpoints/weights_20_test.pt`.
Use `./run.sh --mode test --reuse-checkpoint` to evaluate it again.
Checkpoints and output directories use PyTorch-specific names to keep legacy
experiments available. Old Keras `.h5` weights require conversion and are not
accepted as PyTorch checkpoints. Numeric Python 2 hickle caches are read directly.

Validate GPU training on actual downloaded data:

```bash
docker run --rm --gpus all \
  -v /data/users/jianchen/data/CHBMIT:/data/CHBMIT:ro \
  -v /data/users/jianchen/data/CHBMIT-cache:/cache/CHBMIT:ro \
  seizure-prediction-cnn:pytorch \
  python docker/smoke_test.py --device cuda \
    --cache-dir /cache/CHBMIT --data-dir /data/CHBMIT
```

Official version reference: [PyTorch 2.5.1 CUDA builds](https://pytorch.org/get-started/previous-versions/#v251).
