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
    tasks_per_step: int = 4
    inner_steps: int = 3
    inner_lr: float = 0.01
    outer_lr: float = 5e-4
    support_per_class: int = 16
    query_per_class: int = 16
    min_support_days: int = 2
    query_days: int = 1
    eval_every: int = 100
    checkpoint_every: int = 25
    adaptation_steps: int = 100
    adaptation_lr: float = 5e-4
    adaptation_frontend: bool = False
    batch_size: int = 32
    label_smoothing: float = 0.1
    gradient_clip: float = 5.0

    def validate(self):
        if self.algorithm not in {"meta", "pooled"}:
            raise ValueError("algorithm must be meta or pooled")
        if (not self.targets or len(set(self.targets)) != len(self.targets) or
                any(mouse not in TARGETS for mouse in self.targets)):
            raise ValueError("Choose distinct MouseLFP targets")
        integers = ["outer_steps", "tasks_per_step", "inner_steps", "support_per_class",
                    "query_per_class", "min_support_days", "query_days", "eval_every",
                    "adaptation_steps", "batch_size", "checkpoint_every"]
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1 for key in integers):
            raise ValueError("Step counts, sample counts, and day counts must be positive integers")
        if type(self.seed) is not int or self.seed < 0 or type(self.adaptation_frontend) is not bool:
            raise ValueError("Invalid seed or adaptation_frontend")
        for key in ["inner_lr", "outer_lr", "adaptation_lr", "gradient_clip"]:
            value = getattr(self, key)
            if type(value) not in {float, int} or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
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
