"""Small learned spectral frontend feeding the unchanged MouseLFP CNN.

Only centered/resampled STFT power is cached. Frequency selection, compression,
and temporal smoothing live in forward() and receive classification gradients.
This is a spectral frontend, not a phase-aware raw-waveform network.
"""

import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from input_pipeline import FeatureDataset, file_hash
from models.cnn import SeizureCNN
from meta_learning.common import write_json
import train


SPECTRUM_CONFIG = dict(source_hz=500, target_hz=256, frame_length=512,
                       hop_length=128, frequency_hz=[1, 40], center=False,
                       window="hann", power_normalization="window_squared_sum")
FRONTEND_CONFIG = dict(channels=40, center_bounds_hz=[0.5, 40.5],
                       bandwidth_bounds_hz=[0.5, 4.0], initial_bandwidth_hz=1.0,
                       smoothing_taps=9, time_bins=55, frequency_bins=114)


def fixed_power_batch(raw):
    """Compute a sample-local fixed basis, without temporal aggregation."""
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite (windows, waveform_samples) inputs")
    values = values - values.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    values = resample_poly(values, 256, 500, axis=1)
    window = torch.hann_window(SPECTRUM_CONFIG["frame_length"])
    spectrum = torch.stft(torch.from_numpy(np.ascontiguousarray(values)),
                          n_fft=SPECTRUM_CONFIG["frame_length"],
                          hop_length=SPECTRUM_CONFIG["hop_length"], window=window,
                          center=False, return_complex=True)
    frequencies = torch.fft.rfftfreq(SPECTRUM_CONFIG["frame_length"], d=1 / 256)
    low, high = SPECTRUM_CONFIG["frequency_hz"]
    power = spectrum[:, (frequencies >= low) & (frequencies <= high)].abs().square()
    return (power / window.square().sum()).numpy()


def fixed_power(raw, cache):
    """Cache the fixed basis; never cache outputs of trainable operations."""
    raw_hash = file_hash(raw.filename) if hasattr(raw, "filename") else hashlib.sha256(raw.tobytes()).hexdigest()
    identity = dict(raw_sha256=raw_hash, source_sha256=file_hash(__file__),
                    config=SPECTRUM_CONFIG,
                    runtime={name: importlib.metadata.version(name) for name in ["torch", "numpy", "scipy"]})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{key}.npy"
    manifest_path = cache / f"{key}.json"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        if stored["identity"] != identity or file_hash(path) != stored["sha256"]:
            raise ValueError("Fixed spectrum cache changed")
        return np.load(path, mmap_mode="r", allow_pickle=False)
    first = fixed_power_batch(raw[:1])
    temporary = path.with_suffix(".npy.tmp")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                       shape=(len(raw), *first.shape[1:]))
    for start in range(0, len(raw), 32):
        values = fixed_power_batch(raw[start:start + 32])
        if not np.isfinite(values).all():
            raise ValueError("Non-finite fixed spectrum")
        output[start:start + len(values)] = values
    output.flush()
    del output
    temporary.replace(path)
    write_json(manifest_path, dict(identity=identity, sha256=file_hash(path)))
    return np.load(path, mmap_mode="r", allow_pickle=False)


class SpectralFrontend(nn.Module):
    """Learn normalized Gaussian bands, compression, and positive time kernels."""

    def __init__(self):
        super().__init__()
        c = FRONTEND_CONFIG
        frequencies = torch.fft.rfftfreq(SPECTRUM_CONFIG["frame_length"], d=1 / 256)
        self.register_buffer("frequencies", frequencies[(frequencies >= 1) & (frequencies <= 40)])
        centers = torch.linspace(1, 40, c["channels"])
        self.center_raw = nn.Parameter(torch.logit((centers - 0.5) / 40))
        bandwidth_fraction = (c["initial_bandwidth_hz"] - 0.5) / 3.5
        self.bandwidth_raw = nn.Parameter(torch.full((c["channels"],),
                                         math.log(bandwidth_fraction / (1 - bandwidth_fraction))))
        self.log_gain = nn.Parameter(torch.zeros(c["channels"]))
        self.exponent_raw = nn.Parameter(torch.full((c["channels"],), -math.log(2)))
        offsets = torch.arange(c["smoothing_taps"]) - c["smoothing_taps"] // 2
        self.pool_logits = nn.Parameter((-0.5 * offsets.square().float()).repeat(c["channels"], 1))

    def physical_parameters(self):
        return dict(center_hz=0.5 + 40 * self.center_raw.sigmoid(),
                    bandwidth_hz=0.5 + 3.5 * self.bandwidth_raw.sigmoid(),
                    gain=torch.exp(4 * torch.tanh(self.log_gain / 4)),
                    compression_exponent=0.25 + 0.75 * self.exponent_raw.sigmoid(),
                    smoothing_weights=self.pool_logits.softmax(dim=1))

    def describe(self):
        return {key: value.detach().cpu().tolist() for key, value in self.physical_parameters().items()}

    def forward(self, power):
        p = self.physical_parameters()
        distance = ((self.frequencies[None] - p["center_hz"][:, None]) /
                    p["bandwidth_hz"][:, None])
        weights = torch.exp(-0.5 * distance.square())
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        energy = torch.matmul(weights, power)
        compressed = torch.log1p(energy * p["gain"][None, :, None] + 1e-6)
        compressed = compressed.pow(p["compression_exponent"][None, :, None])
        padding = FRONTEND_CONFIG["smoothing_taps"] // 2
        smoothed = F.conv1d(F.pad(compressed, (padding, padding), mode="replicate"),
                            p["smoothing_weights"][:, None], groups=FRONTEND_CONFIG["channels"])
        pooled = F.adaptive_avg_pool1d(smoothed, FRONTEND_CONFIG["time_bins"])
        matrix = pooled.transpose(1, 2).unsqueeze(1)
        return F.interpolate(matrix, size=(55, 114), mode="bilinear", align_corners=False).unsqueeze(1)


class FrontendCNN(nn.Module):
    def __init__(self, learnable=True):
        super().__init__()
        # Frontend initialization consumes no random draws: all arms share CNN initialization.
        self.frontend = SpectralFrontend()
        self.frontend.requires_grad_(learnable)
        self.cnn = SeizureCNN(train.CNN_INPUT_SHAPE)

    def forward(self, power):
        return self.cnn(self.frontend(power))


class Predictor:
    def __init__(self, model, scale, cache, device):
        self.model, self.scale, self.cache, self.device = model, scale, Path(cache), torch.device(device)

    def predict_proba(self, raw):
        values = fixed_power(raw, self.cache)
        loader = DataLoader(FeatureDataset(values, None, 0, self.scale), batch_size=train.BATCH_SIZE)
        self.model.eval()
        probabilities = []
        with torch.inference_mode():
            for inputs in loader:
                probabilities.append(self.model(inputs.to(self.device)).softmax(dim=1).cpu().numpy())
        return np.concatenate(probabilities)


def fit(train_raw, train_y, validation_raw, validation_y, work_dir, device,
        budget_seconds, seed, learnable=True):
    """Use the original optimizer, loss, cap, time budget, and checkpoint rule."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = work_dir.parent.parent.parent / "fixed-spectrum-cache"
    train_values, val_values = fixed_power(train_raw, cache), fixed_power(validation_raw, cache)
    # One training-fitted scalar preserves amplitude differences between samples/days.
    scale = np.float32(max(float(np.mean(train_values, dtype=np.float64)), 1e-12))
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = FrontendCNN(learnable).to(device)
    initial = {name: parameter.detach().clone() for name, parameter in model.frontend.named_parameters()}
    write_json(work_dir / "frontend_initial.json", model.frontend.describe())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=train.LEARNING_RATE, weight_decay=train.WEIGHT_DECAY,
                                 betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss(label_smoothing=train.LABEL_SMOOTHING)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(FeatureDataset(train_values, train_y, 0, scale),
                              batch_size=train.BATCH_SIZE, shuffle=True, generator=generator)
    val_loader = DataLoader(FeatureDataset(val_values, validation_y, 0, scale), batch_size=train.BATCH_SIZE)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    best_accuracy, best_loss, history = -1.0, float("inf"), []
    for epoch in range(1, train.MAX_EPOCHS + 1):
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
            nn.utils.clip_grad_norm_(model.parameters(), train.GRADIENT_CLIP, error_if_nonfinite=True)
            optimizer.step()
            train_loss += loss.item() * len(labels)
            samples += len(labels)
        model.eval()
        val_loss, correct, count = 0.0, 0, 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                labels = labels.to(device)
                logits = model(inputs.to(device))
                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite validation loss")
                val_loss += loss.item() * len(labels)
                correct += ((logits.softmax(dim=1)[:, 1] >= 0.5).long() == labels).sum().item()
                count += len(labels)
        val_accuracy, val_loss = correct / count, val_loss / count
        history.append(dict(epoch=epoch, loss=train_loss / samples, val_loss=val_loss,
                            val_accuracy=val_accuracy, training_seconds=time.monotonic() - started))
        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
            best_accuracy, best_loss = val_accuracy, val_loss
            torch.save(dict(model=model.state_dict(), scale=float(scale), learnable=learnable,
                            spectrum_config=SPECTRUM_CONFIG, frontend_config=FRONTEND_CONFIG,
                            epoch=epoch, val_accuracy=val_accuracy, val_loss=val_loss,
                            monitor="val_accuracy", threshold=0.5), work_dir / "checkpoint.pt")
        write_json(work_dir / "history.json", history)
        print(f"epoch={epoch} loss={train_loss / samples:.6f} val_loss={val_loss:.6f} "
              f"val_accuracy={val_accuracy:.6f}", flush=True)
        if time.monotonic() - started >= budget_seconds:
            break
    predictor = load_predictor(work_dir, device)
    write_json(work_dir / "frontend_selected.json", predictor.model.frontend.describe())
    movement = {name: float((p.detach() - initial[name]).norm().cpu())
                for name, p in predictor.model.frontend.named_parameters()}
    write_json(work_dir / "frontend_parameter_movement.json", movement)
    return predictor


def load_predictor(work_dir, device):
    work_dir = Path(work_dir)
    checkpoint = torch.load(work_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint["spectrum_config"] != SPECTRUM_CONFIG or checkpoint["frontend_config"] != FRONTEND_CONFIG:
        raise ValueError("Frontend checkpoint configuration differs from this implementation")
    model = FrontendCNN(checkpoint["learnable"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return Predictor(model, np.float32(checkpoint["scale"]),
                     work_dir.parent.parent.parent / "fixed-spectrum-cache", device)
