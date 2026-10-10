import numpy as np
import pytest
import torch

from models.cnn import AxisBatchNorm, ConvNN, SeizureCNN, SignalDataset, resolve_device


@pytest.fixture(autouse=True)
def seed_model():
    torch.set_num_threads(2)
    torch.manual_seed(0)


def test_electrode_batch_norm_matches_keras_statistics():
    inputs = torch.arange(48, dtype=torch.float32).reshape(2, 1, 3, 2, 4)
    normalizer = AxisBatchNorm(3, axis=2)
    values = inputs.numpy()
    mean = values.mean(axis=(0, 1, 3, 4))
    variance = values.var(axis=(0, 1, 3, 4))
    expected = (values - mean[None, None, :, None, None]) / np.sqrt(
        variance[None, None, :, None, None] + 1e-3)
    np.testing.assert_allclose(normalizer(inputs).detach().numpy(), expected, atol=1e-6)
    np.testing.assert_allclose(normalizer.running_mean.numpy(), mean * 0.01, atol=1e-6)
    np.testing.assert_allclose(normalizer.running_var.numpy(), 0.99 + variance * 0.01, atol=1e-6)
    normalizer.eval()
    expected_eval = (values - mean[None, None, :, None, None] * 0.01) / np.sqrt(
        0.99 + variance[None, None, :, None, None] * 0.01 + 1e-3)
    np.testing.assert_allclose(normalizer(inputs).detach().numpy(), expected_eval, atol=1e-5)


@pytest.mark.parametrize('shape', [(1, 22, 55, 114), (1, 17, 55, 114),
                                  (1, 6, 55, 114), (1, 16, 59, 100)])
def test_supported_eeg_shapes_and_gradients(shape):
    model = SeizureCNN(shape)
    inputs = torch.randn(2, *shape)
    logits = model(inputs)
    assert logits.shape == (2, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    if shape == (1, 22, 55, 114):
        assert sum(parameter.numel() for parameter in model.parameters()) == 73438


def test_same_pooling_keeps_last_odd_window():
    model = SeizureCNN((1, 22, 55, 114))
    values = -torch.arange(55, dtype=torch.float32).reshape(1, 1, 1, 1, 55) - 1
    pooled = model.pool1(values)
    assert pooled.shape[-1] == 28
    assert pooled[0, 0, 0, 0, -1] == -55


def test_training_checkpoint_and_cpu_reload(tmp_path):
    inputs = np.random.default_rng(0).normal(size=(6, 1, 6, 55, 114))
    labels = np.array([0, 1, 0, 1, 0, 1])
    model = ConvNN('test', batch_size=2, epochs=1, device='cpu', checkpoint_dir=tmp_path)
    model.setup(inputs.shape)
    initial = model.model.conv1.weight.detach().clone()
    model.fit(inputs[:4], labels[:4], inputs[4:], labels[4:])
    assert model.checkpoint_path.exists()
    assert not torch.equal(initial, model.model.conv1.weight)
    predictions = model.predict_proba(inputs)
    np.testing.assert_allclose(predictions.sum(axis=1), 1, atol=1e-6)
    assert 0 <= model.evaluate(inputs, labels) <= 1
    restored = ConvNN('test', device='cpu', checkpoint_dir=tmp_path).setup(inputs.shape)
    restored.load_trained_weights(model.checkpoint_path)
    np.testing.assert_allclose(restored.predict_proba(inputs), predictions, atol=1e-6)
    assert model.predict_proba(inputs[:0]).shape == (0, 2)


def test_nan_features_fail_before_training():
    inputs = np.zeros((2, 1, 6, 55, 114))
    inputs[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        SignalDataset(inputs, np.array([0, 1]))[0]


def test_explicit_cuda_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA was requested'):
        resolve_device('cuda')
    assert resolve_device('auto').type == 'cpu'
