"""Explicit finite training configuration; no dependency on a search protocol."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

from source_data import DAY_SPLITS, TARGETS

TASK_DAYS = tuple(DAY_SPLITS["train"] + DAY_SPLITS["validation"])


@dataclass(frozen=True)
class Config:
    targets: tuple = ("X2",)
    seed: int = 0
    outer_steps: int = 100
    inner_epochs: int = 50
    inner_checkpoint_steps: int = 5
    inner_log_every: int = 0
    inner_shuffle_seed: int | None = None
    inner_lr: float = 0.01
    inner_weight_decay: float = 0.0
    classifier_dropout: float = 0.0
    outer_lr: float = 0.001
    outer_weight_decay: float = 0.0
    support_loss_weight: float = 0.1
    contrastive_loss_weight: float = 0.0
    augment_random_crop: bool = True
    random_crop_scope: str = "both"
    query_crops: int = 64
    support_crops: int = 4
    support_pool_crops: int = 32
    task_days: tuple = TASK_DAYS
    query_event_fraction: float = 0.2
    support_per_class: int | None = None
    query_per_class: int | None = None
    eval_every: int = 10
    checkpoint_every: int = 25
    batch_size: int = 32
    frontend_batch_size: int = 32
    label_smoothing: float = 0.1
    gradient_clip: float = 5.0

    def validate(self):
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
        if (type(self.task_days) is not tuple or not self.task_days or
                len(set(self.task_days)) != len(self.task_days) or
                any(type(day) is not int or day not in TASK_DAYS for day in self.task_days)):
            raise ValueError("task_days must be distinct non-test Days 0-8/11")
        if (type(self.query_event_fraction) not in {float, int} or
                not math.isfinite(self.query_event_fraction) or not 0 < self.query_event_fraction < 1):
            raise ValueError("query_event_fraction must be finite and strictly between zero and one")
        if (type(self.outer_weight_decay) not in {float, int} or not math.isfinite(self.outer_weight_decay)
                or self.outer_weight_decay < 0):
            raise ValueError("outer_weight_decay must be finite and nonnegative")
        if (type(self.support_loss_weight) not in {float, int} or not math.isfinite(self.support_loss_weight)
                or self.support_loss_weight < 0):
            raise ValueError("support_loss_weight must be finite and nonnegative")
        if (type(self.contrastive_loss_weight) not in {float, int} or not math.isfinite(self.contrastive_loss_weight)
                or self.contrastive_loss_weight < 0):
            raise ValueError("contrastive_loss_weight must be finite and nonnegative")
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
        if type(self.support_pool_crops) is not int or self.support_pool_crops < 1:
            raise ValueError("support_pool_crops must be a positive integer")
        if (self.augment_random_crop and self.random_crop_scope in {"both", "support"}
                and self.support_pool_crops < self.support_crops):
            raise ValueError("support_pool_crops must be at least support_crops")
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
        return dict(asdict(self), targets=list(self.targets), task_days=list(self.task_days))

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        for key in ["targets", "task_days"]:
            if value.get(key) is not None:
                value[key] = tuple(value[key])
        return cls(**value).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
