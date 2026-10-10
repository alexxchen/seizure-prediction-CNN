# Migration validation

Validated on an NVIDIA A100 80 GB PCIe with PyTorch 2.5.1+cu124 and CUDA 12.4.

## Automated checks

- 14 tests passed during the Docker build.
- CPU training, checkpoint save/reload, and probability checks passed.
- CUDA training on real cached CHBMIT samples passed; the checkpoint also
  reproduced predictions when reloaded on CPU within `5e-5` tolerance.
- Bash syntax checks and GPU/CPU launch command checks passed.
- The final launch script ran with host UID/GID 1010 and wrote user-owned outputs.

## Keras reference comparison

Compared the original Python 2 / Keras 2.0.6 / TensorFlow 1.4.1 image with this
implementation. Identical convolution, dense, and BatchNorm weights were
transferred into PyTorch before comparing two input samples in evaluation mode.
A 28-second raw EEG window from `chb20_34.edf` was also compared before and after
STFT preprocessing.

| Check | Maximum absolute difference |
| --- | --- |
| Class probabilities | `8.9407e-8` |
| Raw EEG values in microvolts | `5.6843e-14` |
| Preprocessed features | `1.1921e-7` |

Both implementations have 73,438 trainable parameters for CHBMIT's input shape
`(N, 1, 22, 55, 114)`. The comparison checks the migrated computation and channel
selection; it does not imply identical optimization trajectories across frameworks.

## One full CUDA fold

```bash
./run.sh --epochs 1 --max-folds 1
```

Loaded the existing Python 2 caches without regenerating data. Trained on 2,610
samples, validated on 868, and evaluated on 577 held-out samples. Training loss
was `0.181325`, validation loss `1.295493`, and test AUC `0.942617`.
The checkpoint and both prediction CSV files were saved successfully.

This was a one-epoch functional validation. The paper's complete cross-validation
experiment and the FB/Kaggle raw datasets were not run as part of this migration.
