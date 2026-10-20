"""Explicit validated training recipe; paths are launch arguments."""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    seed: int = 1234
    epochs: int = 2
    microbatch: int = 2
    accumulation: int = 8
    world_size: int = 2
    english_fraction: float = 0.5
    encoder_lr: float = 5e-6
    head_lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    max_grad_norm: float = 1.0
    amp_init_scale: float = 1024.0
    amp_max_retries: int = 12
    max_len: int = 1024
    head_max_len: int = 256
    validate_every: int = 50
    save_every: int = 50
    snapshot_seconds: int = 900
    log_sync_seconds: int = 300
    patience: int = 3
    english_accuracy_drop: float = 0.02
    english_loss_rise: float = 0.05
    threads: int = 2

    def __post_init__(self):
        for key in (
            "seed",
            "epochs",
            "microbatch",
            "accumulation",
            "world_size",
            "max_len",
            "head_max_len",
            "validate_every",
            "save_every",
            "snapshot_seconds",
            "log_sync_seconds",
            "patience",
            "threads",
            "amp_max_retries",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"{key} must be a positive integer (seed may be zero)")
        for key in ("encoder_lr", "head_lr", "max_grad_norm", "amp_init_scale"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in (
            "english_fraction",
            "warmup_fraction",
            "weight_decay",
            "english_accuracy_drop",
            "english_loss_rise",
        ):
            if not math.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"invalid {key}")
        if not 0 < self.english_fraction < 1 or self.head_max_len + 8 >= self.max_len:
            raise ValueError("invalid replay ratio or head/context budgets")
        if self.amp_max_retries > 16:
            raise ValueError("AMP retry budget must be <=16")
        if self.epochs > 4 or self.world_size not in (1, 2):
            raise ValueError("bounded recipe requires <=4 epochs and 1 or 2 ranks")
        ne = self.global_batch * self.english_fraction
        if ne != round(ne) or ne < 1 or ne >= self.global_batch:
            raise ValueError("replay counts must be exact integers in each effective batch")

    @property
    def global_batch(self):
        return self.microbatch * self.accumulation * self.world_size

    def to_dict(self):
        return asdict(self)


def read_settings(path: Path) -> Settings:
    with path.open("rb") as stream:
        return Settings(**tomllib.load(stream))
