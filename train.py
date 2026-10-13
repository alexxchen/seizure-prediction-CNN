"""Editable research code: raw-window preprocessing, model and optimization.

Contract: fit(...) returns an object with predict_proba(raw_windows).
load_predictor(...) restores that object without retraining for final evaluation.
"""

import hashlib
import json
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import stft
import torch
from scipy.signal import resample_poly
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.cnn import SeizureCNN

# Research knobs. The initial recipe matches the existing log-STFT/CNN/Adam.
PREPROCESS = dict(target_hz=256, frame_length=256, normalization="none", epsilon=1e-6)
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0
MAX_EPOCHS = 100
LABEL_SMOOTHING = 0.0
GRADIENT_CLIP = float("inf")


def preprocess_window(signal):
    """Replace freely; the caller supplies exactly one fixed 28-second window."""
    spectrum = stft.spectrogram(
        resample_poly(signal, PREPROCESS["target_hz"], 500)[:, None],
        framelength=PREPROCESS["frame_length"], centered=False)
    if spectrum.ndim == 2:
        spectrum = spectrum[:, :, None]
    values = np.abs(np.transpose(spectrum, (2, 1, 0))) + PREPROCESS["epsilon"]
    values = np.concatenate((values[:, :, 1:57], values[:, :, 64:117],
                             values[:, :, 124:]), axis=-1)
    return np.maximum(np.log10(values), 0)[None].astype(np.float32)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def features(raw, cache):
    """Cache only deterministic, sample-local transforms under their source hash."""
    # A source fingerprint prevents stale features after arbitrary code edits.
    source_hash = file_hash(__file__)
    input_hash = file_hash(raw.filename) if hasattr(raw, "filename") else hashlib.sha256(raw.tobytes()).hexdigest()
    runtime = json.dumps({name: importlib.metadata.version(name)
                          for name in ["numpy", "scipy", "stft"]}, sort_keys=True)
    key = hashlib.sha256((source_hash + input_hash + runtime).encode()).hexdigest()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{key}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r", allow_pickle=False)
    first = preprocess_window(np.asarray(raw[0]))
    result = np.lib.format.open_memmap(str(path) + ".tmp", mode="w+",
                                       dtype=np.float32, shape=(len(raw), *first.shape))
    for index, signal in enumerate(raw):
        sample = first if index == 0 else preprocess_window(np.asarray(signal))
        if sample.shape != first.shape or not np.isfinite(sample).all():
            raise ValueError("Invalid preprocessed window")
        result[index] = sample
    result.flush()
    del result
    Path(str(path) + ".tmp").replace(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def normalization_stats(train_features):
    mode = PREPROCESS["normalization"]
    if mode == "none":
        return np.array(0, dtype=np.float32), np.array(1, dtype=np.float32)
    if mode != "train_frequency_zscore":
        raise ValueError("Unknown normalization recipe")
    # Fit per-frequency statistics exclusively on training windows.
    total = np.zeros(train_features.shape[-1], dtype=np.float64)
    squares = np.zeros_like(total)
    count = 0
    for start in range(0, len(train_features), 256):
        batch = np.asarray(train_features[start:start + 256], dtype=np.float64)
        axes = tuple(range(batch.ndim - 1))
        total += batch.sum(axis=axes)
        squares += np.square(batch).sum(axis=axes)
        count += int(np.prod(batch.shape[:-1]))
    mean = total / count
    scale = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
    return mean.astype(np.float32), np.maximum(scale, PREPROCESS["epsilon"]).astype(np.float32)


class FeatureDataset(Dataset):
    def __init__(self, values, labels, mean, scale):
        self.values, self.labels, self.mean, self.scale = values, labels, mean, scale

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        sample = np.array(self.values[index], dtype=np.float32, copy=True)
        tensor = torch.from_numpy((sample - self.mean) / self.scale)
        return tensor if self.labels is None else (tensor, int(self.labels[index]))


def build_model(input_shape):
    """Replace with any model that returns two class logits."""
    return SeizureCNN(input_shape)


class Predictor:
    def __init__(self, model, mean, scale, cache, device):
        self.model, self.mean, self.scale = model, mean, scale
        self.cache, self.device = Path(cache), torch.device(device)

    def predict_proba(self, raw):
        values = features(raw, self.cache)
        loader = DataLoader(FeatureDataset(values, None, self.mean, self.scale),
                            batch_size=BATCH_SIZE, shuffle=False)
        self.model.eval()
        output = []
        with torch.inference_mode():
            for batch in loader:
                output.append(self.model(batch.to(self.device)).softmax(dim=1).cpu().numpy())
        return np.concatenate(output)


def fit(train_raw, train_y, validation_raw, validation_y, work_dir, device,
        budget_seconds, seed):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = work_dir.parent.parent.parent / "feature-cache"
    train_features, val_features = features(train_raw, cache), features(validation_raw, cache)
    mean, scale = normalization_stats(train_features)
    model = build_model(train_features.shape[1:]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(FeatureDataset(train_features, train_y, mean, scale),
                              batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    val_loader = DataLoader(FeatureDataset(val_features, validation_y, mean, scale),
                            batch_size=BATCH_SIZE, shuffle=False)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    best_accuracy, best_loss, history = -1.0, float("inf"), []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss, samples = 0.0, 0
        for inputs, labels in train_loader:
            if samples and time.monotonic() - started >= budget_seconds:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
            optimizer.step()
            train_loss += loss.item() * len(labels)
            samples += len(labels)
        model.eval()
        val_loss, val_samples, val_correct = 0.0, 0, 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                labels = labels.to(device)
                logits = model(inputs.to(device))
                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite validation loss")
                val_loss += loss.item() * len(labels)
                predicted = (logits.softmax(dim=1)[:, 1] >= 0.5).long()
                val_correct += (predicted == labels).sum().item()
                val_samples += len(labels)
        val_loss /= val_samples
        val_accuracy = val_correct / val_samples
        history.append(dict(epoch=epoch, loss=train_loss / samples,
                            val_loss=val_loss, val_accuracy=val_accuracy,
                            training_seconds=time.monotonic() - started))
        if (val_accuracy > best_accuracy or
                (val_accuracy == best_accuracy and val_loss < best_loss)):
            best_accuracy, best_loss = val_accuracy, val_loss
            torch.save(dict(model=model.state_dict(), input_shape=list(train_features.shape[1:]),
                            mean=torch.from_numpy(mean), scale=torch.from_numpy(scale),
                            epoch=epoch, val_loss=val_loss, val_accuracy=val_accuracy,
                            monitor="val_accuracy", threshold=0.5), work_dir / "checkpoint.pt")
        (work_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch={epoch} loss={history[-1]['loss']:.6f} val_loss={val_loss:.6f} "
              f"val_accuracy={val_accuracy:.6f}", flush=True)
        if time.monotonic() - started >= budget_seconds:
            break
    return load_predictor(work_dir, device)


def load_predictor(work_dir, device):
    work_dir = Path(work_dir)
    checkpoint = torch.load(work_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = build_model(checkpoint["input_shape"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return Predictor(model, checkpoint["mean"].numpy(), checkpoint["scale"].numpy(),
                     work_dir.parent.parent.parent / "feature-cache", device)


if __name__ == "__main__":
    raise SystemExit("Run ./run_research.sh experiment; the frozen evaluator calls this recipe.")
