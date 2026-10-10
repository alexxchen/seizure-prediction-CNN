"""Run patient-specific PyTorch seizure prediction experiments."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from models.cnn import ConvNN, resolve_device
from utils.load_signals import PrepData
from utils.prep_data import train_val_loo_split, train_val_test_split


DEFAULT_TARGETS = {
    "CHBMIT": ["20", "21", "23"],
    "FB": ["1", "3", "6", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
    "Kaggle2014Pred": ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5", "Patient_1", "Patient_2"],
}


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


def write_predictions(result_dir, target, fold, predictions, labels):
    for name, label in (("preictal", 1), ("interictal", 0)):
        filename = result_dir / f"{name}_{target}_{fold}.csv"
        np.savetxt(filename, predictions[labels == label, 1], fmt="%.4f",
                   header=name, comments="")
        print(f"Wrote {filename}")


def main(dataset="CHBMIT", build_type="cv", targets=None, device="auto", epochs=None,
         batch_size=32, num_workers=0, seed=0, data_dir=None, cache_dir=None,
         result_dir=None, max_folds=None, reuse_checkpoint=False):
    device = resolve_device(device)
    seed_everything(seed)
    project_dir = Path(__file__).resolve().parent
    with (project_dir / f"SETTINGS_{dataset}.json").open() as handle:
        settings = json.load(handle)
    if settings["dataset"] != dataset:
        raise ValueError("Dataset and settings file do not match")
    for key, value in (("datadir", data_dir), ("cachedir", cache_dir), ("resultdir", result_dir)):
        if value is not None:
            settings[key] = str(value)
    for key in ("cachedir", "resultdir"):
        Path(settings[key]).mkdir(parents=True, exist_ok=True)
    result_dir = Path(settings["resultdir"])
    targets = DEFAULT_TARGETS[dataset] if targets is None else targets
    if not targets:
        raise ValueError("Select at least one subject")
    epochs = epochs if epochs is not None else (50 if build_type == "cv" else 100)
    if epochs < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("Epochs and batch size must be positive; workers must be nonnegative")
    if max_folds is not None and max_folds < 1:
        raise ValueError("--max-folds must be positive")
    metrics = []
    print(f"Dataset: {dataset}, mode: {build_type}, subjects: {targets}, device: {device}")
    for target in targets:
        ictal_X, ictal_y = PrepData(target, type="ictal", settings=settings).apply()
        interictal_X, interictal_y = PrepData(target, type="interictal", settings=settings).apply()
        if build_type == "cv":
            folds = train_val_loo_split(ictal_X, ictal_y, interictal_X, interictal_y, 0.25)
        elif build_type == "test":
            folds = [train_val_test_split(ictal_X, ictal_y, interictal_X, interictal_y, 0.25, 0.35)]
        else:
            raise ValueError("Mode must be cv or test")
        for fold, split in enumerate(folds, start=1):
            X_train, y_train, X_val, y_val, X_test, y_test = split
            network = ConvNN(target, batch_size=batch_size, epochs=epochs, mode=build_type,
                             device=device, checkpoint_dir=result_dir / "checkpoints",
                             fold=fold if build_type == "cv" else None,
                             num_workers=num_workers, seed=seed)
            network.setup(X_train.shape)
            if reuse_checkpoint and network.checkpoint_path.exists():
                network.load_trained_weights(network.checkpoint_path)
            else:
                network.fit(X_train, y_train, X_val, y_val)
            predictions = network.predict_proba(X_test)
            auc = float(roc_auc_score(y_test, predictions[:, 1]))
            print(f"Subject {target}, fold {fold}, test AUC: {auc:.6f}")
            write_predictions(result_dir, target, fold, predictions, y_test)
            history_path = result_dir / f"history_{target}_{build_type}_{fold}.json"
            history_path.write_text(json.dumps(network.history, indent=2) + "\n")
            metrics.append(dict(subject=target, fold=fold, auc=auc, device=str(device),
                                checkpoint=str(network.checkpoint_path)))
            (result_dir / f"metrics_{build_type}.json").write_text(json.dumps(metrics, indent=2) + "\n")
            if max_folds is not None and fold >= max_folds:
                break
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cv", "test"), default="cv")
    parser.add_argument("--dataset", choices=tuple(DEFAULT_TARGETS), default="CHBMIT")
    parser.add_argument("--targets", help="Comma-separated subject IDs")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:INDEX")
    parser.add_argument("--epochs", type=int, help="Defaults to 50 for cv and 100 for test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--max-folds", type=int, help="Limit cv folds for a shorter experiment")
    parser.add_argument("--reuse-checkpoint", action="store_true", help="Evaluate an existing .pt checkpoint")
    args = parser.parse_args()
    if args.targets is not None:
        args.targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    return args


if __name__ == "__main__":
    arguments = vars(parse_args())
    arguments["build_type"] = arguments.pop("mode")
    main(**arguments)
