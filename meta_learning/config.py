"""Explicit finite training configuration; no dependency on a search protocol."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

from source_data import DAY_SPLITS, TARGETS


@dataclass(frozen=True)
class Config:
    algorithm: str = "meta"
    targets: tuple = ("X2",)
    seed: int = 0
    outer_steps: int = 100
    inner_epochs: int = 50
    inner_checkpoint_steps: int = 5
    inner_log_every: int = 0
    inner_shuffle_seed: int | None = None
    batchnorm_recalibration: bool = True
    inner_optimizer: str = "sgd"
    inner_lr: float = 0.01
    inner_weight_decay: float = 0.0
    classifier_dropout: float = 0.0
    outer_lr: float = 0.001
    outer_weight_decay: float = 0.0
    augment_random_crop: bool = True
    random_crop_scope: str = "both"
    query_crops: int = 64
    support_crops: int = 40
    query_validation_days: tuple = (8,)
    selection_validation_days: tuple | None = (11,)
    support_per_class: int | None = None
    query_per_class: int | None = None
    eval_every: int = 10
    checkpoint_every: int = 25
    batch_size: int = 32
    frontend_batch_size: int = 32
    label_smoothing: float = 0.1
    gradient_clip: float = 5.0

    def validate(self):
        if self.algorithm not in {"meta", "pooled", "log_stft"}:
            raise ValueError("algorithm must be meta, pooled, or log_stft")
        if self.inner_optimizer != "sgd":
            raise ValueError("inner_optimizer must be sgd (without momentum)")
        if (not self.targets or len(set(self.targets)) != len(self.targets) or
                any(mouse not in TARGETS for mouse in self.targets)):
            raise ValueError("Choose distinct MouseLFP targets")
        integers = ["outer_steps", "inner_epochs",
                    "eval_every", "batch_size", "frontend_batch_size", "checkpoint_every"]
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1 for key in integers):
            raise ValueError("Step counts and sample counts must be positive integers")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Invalid seed")
        if type(self.inner_checkpoint_steps) is not int or self.inner_checkpoint_steps < 0:
            raise ValueError("inner_checkpoint_steps must be a nonnegative integer")
        if type(self.inner_log_every) is not int or self.inner_log_every < 0:
            raise ValueError("inner_log_every must be a nonnegative integer (0 measures/logs only the final epoch)")
        validation_days = set(DAY_SPLITS["validation"])
        query_days = self.query_validation_days
        if (type(query_days) is not tuple or not query_days or len(set(query_days)) != len(query_days) or
                any(type(day) is not int or day not in validation_days for day in query_days)):
            raise ValueError("query_validation_days must be distinct validation days")
        selection_days = self.selection_validation_days
        if selection_days is not None and (
                type(selection_days) is not tuple or not selection_days or
                len(set(selection_days)) != len(selection_days) or set(selection_days) & set(query_days) or
                any(type(day) is not int or day not in validation_days for day in selection_days)):
            raise ValueError("selection_validation_days must be null or distinct validation days disjoint from the query days")
        if (type(self.outer_weight_decay) not in {float, int} or not math.isfinite(self.outer_weight_decay)
                or self.outer_weight_decay < 0):
            raise ValueError("outer_weight_decay must be finite and nonnegative")
        if type(self.augment_random_crop) is not bool:
            raise ValueError("augment_random_crop must be true or false")
        if self.random_crop_scope not in {"both", "query", "support"}:
            raise ValueError("random_crop_scope must be both, query, or support")
        if type(self.query_crops) is not int or self.query_crops < 1:
            raise ValueError("query_crops must be a positive integer")
        if self.query_crops > 1 and not (self.augment_random_crop and self.random_crop_scope in {"both", "query"}):
            raise ValueError("query_crops > 1 requires random crops on the query")
        if type(self.support_crops) is not int or self.support_crops < 1:
            raise ValueError("support_crops must be a positive integer")
        if self.support_crops > 1 and not (self.augment_random_crop and self.random_crop_scope in {"both", "support"}):
            raise ValueError("support_crops > 1 requires random crops on the support")
        if type(self.batchnorm_recalibration) is not bool:
            raise ValueError("batchnorm_recalibration must be true or false")
        if self.inner_shuffle_seed is not None and (
                type(self.inner_shuffle_seed) is not int or self.inner_shuffle_seed < 0):
            raise ValueError("inner_shuffle_seed must be null (fresh shuffles) or a nonnegative integer")
        for key in ["support_per_class", "query_per_class"]:
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{key} must be null (all samples) or a positive integer")
        if (type(self.inner_weight_decay) not in {float, int} or not math.isfinite(self.inner_weight_decay)
                or self.inner_weight_decay < 0):
            raise ValueError("inner_weight_decay must be finite and nonnegative")
        for key in ["inner_lr", "outer_lr", "gradient_clip"]:
            value = getattr(self, key)
            if type(value) not in {float, int} or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ["classifier_dropout", "label_smoothing"]:
            value = getattr(self, key)
            if type(value) not in {float, int} or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{key} must be finite and in [0, 1)")
        return self

    def to_dict(self):
        days = self.selection_validation_days
        return dict(asdict(self), targets=list(self.targets), query_validation_days=list(self.query_validation_days),
                    selection_validation_days=None if days is None else list(days))

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        # Older zero-momentum configurations describe the same plain SGD solver.
        if "inner_momentum" in value:
            momentum = value.pop("inner_momentum")
            if type(momentum) not in {int, float} or momentum != 0:
                raise ValueError("SGD momentum is no longer supported; remove inner_momentum")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        value = dict(value)
        for key in ["targets", "query_validation_days", "selection_validation_days"]:
            if value.get(key) is not None:
                value[key] = tuple(value[key])
        return cls(**value).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
