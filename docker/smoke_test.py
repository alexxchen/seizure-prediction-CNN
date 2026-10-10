"""Validate CPU/CUDA training, cache compatibility, and checkpoint reload."""

import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import stft
import torch

from main import seed_everything
from models.cnn import ConvNN
from utils.load_signals import load_signals_CHBMIT


def read_features(cache_dir):
    arrays = []
    for name in ('ictal_20.hickle', 'interictal_20.hickle'):
        with h5py.File(cache_dir / name, 'r') as handle:
            arrays.append(handle['data_0/data_0/data_0'][:2].astype(np.float32))
    return np.concatenate(arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    seed_everything(0)
    torch.set_num_threads(2)
    print(f'PyTorch: {torch.__version__}, CUDA runtime: {torch.version.cuda}')
    if args.cache_dir:
        inputs = read_features(args.cache_dir)
        labels = np.array([1, 1, 0, 0])
    else:
        inputs = np.random.default_rng(0).normal(size=(4, 1, 22, 55, 114)).astype(np.float32)
        labels = np.array([0, 1, 0, 1])
    spectrum = stft.spectrogram(np.zeros((256 * 28, 22)), framelength=256, centered=False)
    assert spectrum.shape == (129, 55, 22)
    if args.data_dir:
        raw = next(load_signals_CHBMIT(str(args.data_dir), '20', 'interictal'))
        assert raw.shape[1] == 22 and np.isfinite(raw).all()
        print(f'Raw EEG validation passed: {raw.shape}')
    with tempfile.TemporaryDirectory() as directory:
        model = ConvNN('smoke', batch_size=2, epochs=1, device=args.device,
                       checkpoint_dir=directory).setup(inputs.shape)
        model.fit(inputs[[0, 2]], labels[[0, 2]], inputs[[1, 3]], labels[[1, 3]])
        assert next(model.model.parameters()).device == model.device
        predictions = model.predict_proba(inputs)
        assert np.isfinite(predictions).all()
        np.testing.assert_allclose(predictions.sum(axis=1), 1, atol=1e-6)
        restored = ConvNN('smoke', batch_size=2, device='cpu', checkpoint_dir=directory).setup(inputs.shape)
        restored.load_trained_weights(model.checkpoint_path)
        np.testing.assert_allclose(restored.predict_proba(inputs), predictions, atol=5e-5, rtol=5e-5)
        assert np.isfinite(model.evaluate(inputs, labels))
    print(f'Environment smoke test passed on {model.device}.')


if __name__ == '__main__':
    main()
