"""Serialization, provenance, deterministic execution, and binary metrics."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, log_loss, roc_auc_score

from source_data import DAY_SPLITS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu, cuda, or auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu")
    return device


def validate_sample_table(rows, dataset, target, split):
    columns = ["mouse", "day", "raw_file", "start_row", "stop_row", "label"]
    if not set(columns).issubset(rows) or not len(rows) or rows[columns].isna().any().any():
        raise ValueError("Missing sample provenance")
    if not rows.mouse.eq(target).all() or not rows.day.isin(DAY_SPLITS[split]).all():
        raise ValueError("Samples crossed a mouse/day partition")
    for column in ["day", "start_row", "stop_row", "label"]:
        values = rows[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Sample coordinates and labels must be integers")
    if set(rows.label.unique()) != {0, 1} or rows.duplicated(["raw_file", "start_row", "stop_row"]).any():
        raise ValueError("Both classes and unique windows are required")
    for row in rows.itertuples():
        source = dataset.sources.get(row.raw_file)
        if (source is None or source["mouse"] != target or source["day"] != row.day or
                not 0 <= row.start_row < row.stop_row <= source["rows"]):
            raise ValueError("Sample coordinates do not match the source inventory")
    return rows


def prediction_metrics(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if probabilities.shape == (len(labels), 2):
        if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Class probabilities must sum to one")
        probabilities = probabilities[:, 1]
    if (probabilities.shape != labels.shape or not np.isfinite(probabilities).all() or
            (probabilities < 0).any() or (probabilities > 1).any() or set(np.unique(labels)) != {0, 1}):
        raise ValueError("Expected finite binary probabilities and both label classes")
    predicted = probabilities >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    result = dict(samples=len(labels), accuracy=float(np.mean(predicted == labels)),
                  auc=float(roc_auc_score(labels, probabilities)),
                  average_precision=float(average_precision_score(labels, probabilities)),
                  sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                  loss=float(log_loss(labels, probabilities, labels=[0, 1])),
                  confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]])
    return result, probabilities


def signal_audit(raw):
    indices = np.linspace(0, len(raw) - 1, min(128, len(raw)), dtype=np.int64)
    values = np.asarray(raw[indices], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite raw LFP in the signal audit")
    std = values.std(axis=1)
    return dict(sampled_windows=len(indices), total_windows=len(raw), finite=True,
                minimum=float(values.min()), maximum=float(values.max()),
                constant_windows=int((std == 0).sum()),
                window_std_quantiles=np.quantile(std, [0, .25, .5, .75, 1]).tolist())
