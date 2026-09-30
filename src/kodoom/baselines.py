"""Trivial baselines, so every gain is measured against something (plan 2.2).

- ``uniform``: the same probability on every option (chance).
- ``prior``: each option's share of the gold mass in a training file, per question
  type, ignoring the input. It answers "what does a model that never reads the state
  score?" On typed-decisions this baseline is strong on ECE while knowing nothing,
  which is why KL and Brier are read next to ECE.
- ``oracle``: the gold itself, to prove the harness can reach perfect scores.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from kodoom.predictions import Prediction
from kodoom.schema import Record

PRIOR_SMOOTHING = 1e-3  # keeps an option never seen in training possible


def uniform(records: Sequence[Record]) -> list[Prediction]:
    return [
        Prediction(r.id, {o.id: 1 / len(r.options) for o in r.options}, r.gold) for r in records
    ]


def oracle(records: Sequence[Record]) -> list[Prediction]:
    return [
        Prediction(r.id, {o.id: r.gold.get(o.id, 0.0) for o in r.options}, r.gold) for r in records
    ]


def fit_prior(training: Sequence[Record]) -> dict[tuple[str, str], float]:
    """Mean gold mass of each ``(question_type, option id)`` in ``training``."""
    mass: dict[tuple[str, str], float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for r in training:
        count[r.question_type] += 1
        for option_id, p in r.gold.items():
            mass[r.question_type, option_id] += p
    return {key: total / count[key[0]] for key, total in mass.items()}


def prior(records: Sequence[Record], training: Sequence[Record]) -> list[Prediction]:
    table = fit_prior(training)
    predictions = []
    for r in records:
        weights = {
            o.id: table.get((r.question_type, o.id), 0.0) + PRIOR_SMOOTHING for o in r.options
        }
        total = sum(weights.values())
        predictions.append(Prediction(r.id, {k: w / total for k, w in weights.items()}, r.gold))
    return predictions
