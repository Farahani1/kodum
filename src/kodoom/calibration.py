"""Temperature calibration artifacts (plan 2.2, 3.6).

The artifact follows typed-decision-bench's ``calibration.json`` schema v2 as its
README describes it: a ``temperatures`` map with one temperature per question type
(``choice``, ``noul``, ``score``) and a global ``temperature`` as the fallback, so an
engine serves ``temperatures.get(question_type, temperature)``. A type gets its own
temperature only when it has enough calibration examples; otherwise it falls back to
the global one, fitted on everything.

The field names beyond those two (``schema_version``, ``fitted_on``) are this
project's own and are not part of the community standard. Fit on the calibration
split only, never on test data.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kodoom.metrics import MetricError, fit_temperature

MIN_EXAMPLES_PER_TYPE = 100  # the bench's rule: a type needs at least 100 cases of its own


class CalibrationError(ValueError):
    """A calibration artifact is malformed."""


@dataclass(frozen=True)
class Calibration:
    temperature: float  # global fallback
    temperatures: dict[str, float] = field(default_factory=dict)  # per question type
    fitted_on: dict[str, int] = field(default_factory=dict)  # examples per type, and "all"

    def for_type(self, question_type: str) -> float:
        return self.temperatures.get(question_type, self.temperature)

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "temperature": self.temperature,
            "temperatures": self.temperatures,
            "fitted_on": self.fitted_on,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> Calibration:
        if "temperature" not in d:
            raise CalibrationError("a calibration needs a global 'temperature'")
        values = [d["temperature"], *d.get("temperatures", {}).values()]
        if any(not isinstance(t, int | float) or isinstance(t, bool) or t <= 0 for t in values):
            raise CalibrationError("temperatures must be positive numbers")
        return cls(
            temperature=float(d["temperature"]),
            temperatures={k: float(v) for k, v in d.get("temperatures", {}).items()},
            fitted_on=dict(d.get("fitted_on", {})),
        )


def fit_calibration(
    examples: Sequence[tuple[str, Mapping[str, float], Mapping[str, float]]],
    min_per_type: int = MIN_EXAMPLES_PER_TYPE,
) -> Calibration:
    """Fit from ``(question_type, probs, gold)`` triples of the calibration split."""
    if not examples:
        raise MetricError("no calibration examples")
    by_type: dict[str, list] = defaultdict(list)
    for question_type, probs, gold in examples:
        by_type[question_type].append((probs, gold))
    everything = [(p, g) for _, p, g in examples]
    return Calibration(
        temperature=fit_temperature(everything),
        temperatures={
            t: fit_temperature(items)
            for t, items in sorted(by_type.items())
            if len(items) >= min_per_type
        },
        fitted_on={"all": len(examples), **{t: len(i) for t, i in sorted(by_type.items())}},
    )


def write_calibration(path: str | Path, calibration: Calibration) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(calibration.to_dict(), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_calibration(path: str | Path) -> Calibration:
    try:
        return Calibration.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except json.JSONDecodeError as e:
        raise CalibrationError(f"{path}: invalid JSON: {e}") from e
