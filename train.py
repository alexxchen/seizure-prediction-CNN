"""Fixed CNN and training recipe for input-pipeline research.

Contract: fit(...) returns an object with predict_proba(raw_windows).
load_predictor(...) restores that object without retraining for final evaluation.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.cnn import SeizureCNN
from input_pipeline import FeatureDataset, features, normalization_stats

# Fixed controls for input comparisons. Preserve the current committed recipe.
CNN_INPUT_SHAPE = (1, 1, 55, 114)
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0
MAX_EPOCHS = 100
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP = float("inf")


def build_model(input_shape):
    """Keep the existing CNN and its parameter count fixed across inputs."""
    if tuple(input_shape) != CNN_INPUT_SHAPE:
        raise ValueError(f"Input pipeline must produce {CNN_INPUT_SHAPE}; received {tuple(input_shape)}")
    return SeizureCNN(CNN_INPUT_SHAPE)


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
    for values in (train_features, val_features):
        if values.ndim != 5 or tuple(values.shape[1:]) != CNN_INPUT_SHAPE:
            raise ValueError(f"Input pipeline must preserve the CNN input shape {CNN_INPUT_SHAPE}")
    mean, scale = normalization_stats(train_features)
    # Keep initialization independent of random draws made by data processing.
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
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
