"""The prediction file: one line per scored item (plan 2.3, 3.6).

Every evaluation saves, per item, the ID, the gold distribution, the predicted
distribution, the temperature already applied and the latency. Every later metric,
chart and error analysis is recomputed from these files without retraining, and
they can be published as raw results.

``probs`` is what the model shipped. ``temperature`` records a temperature the model
already applied to get it (1.0 means raw), so a recalibration is never applied twice
by accident.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class PredictionError(ValueError):
    """A prediction file is malformed."""


@dataclass(frozen=True)
class Prediction:
    id: str
    probs: dict[str, float]
    gold: dict[str, float] | None = None
    temperature: float = 1.0
    latency_ms: float | None = None
    error: str | None = None  # a request that failed: counted, never scored
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "probs": self.probs,
            "gold": self.gold,
            "temperature": self.temperature,
            "latency_ms": self.latency_ms,
            "error": self.error,
            **({"extra": self.extra} if self.extra else {}),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Prediction:
        unknown = set(d) - {"id", "probs", "gold", "temperature", "latency_ms", "error", "extra"}
        if unknown:
            raise PredictionError(f"unknown fields: {sorted(unknown)}")
        if not isinstance(d.get("id"), str) or not d["id"]:
            raise PredictionError("a prediction needs an id")
        probs = d.get("probs")
        if not isinstance(probs, dict) or (not probs and not d.get("error")):
            raise PredictionError(f"{d['id']}: probs must be an object of option -> probability")
        return cls(
            id=d["id"],
            probs={str(k): float(v) for k, v in probs.items()},
            gold=d.get("gold"),
            temperature=float(d.get("temperature", 1.0)),
            latency_ms=d.get("latency_ms"),
            error=d.get("error"),
            extra=d.get("extra") or {},
        )


def write_predictions(path: str | Path, predictions: Iterable[Prediction]) -> int:
    """Write all predictions, replacing the file only once writing succeeded."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for p in predictions:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def read_predictions(path: str | Path) -> dict[str, Prediction]:
    """Read a prediction file into ``{id: Prediction}``; errors name file and line."""
    path = Path(path)
    predictions: dict[str, Prediction] = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                p = Prediction.from_dict(json.loads(line))
            except (json.JSONDecodeError, PredictionError) as e:
                raise PredictionError(f"{path}:{n}: {e}") from e
            if p.id in predictions:
                raise PredictionError(f"{path}:{n}: duplicate id {p.id!r}")
            predictions[p.id] = p
    return predictions
