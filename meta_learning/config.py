"""Explicit finite training configuration; no dependency on a search protocol."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

from source_data import TARGETS


@dataclass(frozen=True)
class Config:
    algorithm: str = "meta"
    targets: tuple = tuple(TARGETS)
    seed: int = 0
    outer_steps: int = 1000
    inner_min_epochs: int = 10
    inner_max_epochs: int = 50
    inner_patience: int = 5
    inner_relative_tolerance: float = 1e-3
    inner_checkpoint_steps: int = 5
    inner_log_every: int = 10
    inner_optimizer: str = "adam"
    inner_lr: float = 5e-4
    inner_momentum: float = 0.9
    inner_beta1: float = 0.9
    inner_beta2: float = 0.999
    inner_epsilon: float = 1e-8
    classifier_dropout: float = 0.0
    outer_lr: float = 5e-4
    support_per_class: int | None = None
    query_per_class: int | None = None
    min_support_days: int = 2
    query_days: int = 1
    eval_every: int = 100
    checkpoint_every: int = 25
    batch_size: int = 32
    label_smoothing: float = 0.1
    gradient_clip: float = 5.0

    def validate(self):
        if self.algorithm not in {"meta", "pooled"}:
            raise ValueError("algorithm must be meta or pooled")
        if self.inner_optimizer not in {"adam", "sgd"}:
            raise ValueError("inner_optimizer must be adam or sgd")
        if (not self.targets or len(set(self.targets)) != len(self.targets) or
                any(mouse not in TARGETS for mouse in self.targets)):
            raise ValueError("Choose distinct MouseLFP targets")
        integers = ["outer_steps", "inner_min_epochs", "inner_max_epochs",
                    "inner_patience", "inner_log_every", "min_support_days", "query_days",
                    "eval_every", "batch_size", "checkpoint_every"]
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1 for key in integers):
            raise ValueError("Step counts, sample counts, and day counts must be positive integers")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Invalid seed")
        if self.inner_min_epochs > self.inner_max_epochs:
            raise ValueError("inner_min_epochs must not exceed inner_max_epochs")
        if type(self.inner_checkpoint_steps) is not int or self.inner_checkpoint_steps < 0:
            raise ValueError("inner_checkpoint_steps must be a nonnegative integer")
        for key in ["support_per_class", "query_per_class"]:
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{key} must be null (all samples) or a positive integer")
        for key in ["inner_lr", "inner_epsilon", "outer_lr", "gradient_clip"]:
            value = getattr(self, key)
            if type(value) not in {float, int} or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ["inner_relative_tolerance", "inner_momentum", "inner_beta1", "inner_beta2",
                    "classifier_dropout", "label_smoothing"]:
            value = getattr(self, key)
            if type(value) not in {float, int} or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{key} must be finite and in [0, 1)")
        return self

    def to_dict(self):
        return dict(asdict(self), targets=list(self.targets))

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        value = dict(value)
        if "targets" in value:
            value["targets"] = tuple(value["targets"])
        return cls(**value).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
