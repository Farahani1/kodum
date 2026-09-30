"""Metrics for typed decisions (plan 3.2): accuracy, calibration, distance from gold.

Pure Python, no dependencies. Definitions follow typed-decision-bench (kyr0), so
results are comparable with the community: ``p`` is the predicted distribution and
``y`` the gold distribution over the same options, and every calibration metric
uses the full ``y``, never its argmax.

- accuracy: the predicted top option is a gold top option (gold ties all count)
- soft_accuracy: sum of p * y, the mass p puts on the gold
- nll: -sum y * log p (cross-entropy); brier: sum (p - y)^2
- kl: KL(y || p), the leaderboard's distance from gold; js: Jensen-Shannon (bounded)
- confidence: max p; ece_15: expected calibration error, 15 equal-width bins
- score_mae: |E_p[level] - E_y[level]| on score questions; score_within_one: share <= 1

Ties in p are broken by option order, so results never depend on dict ordering.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

EPS = 1e-12
ECE_BINS = 15
# Fitting a temperature below or above these makes no sense for a softmax over options.
MIN_TEMPERATURE, MAX_TEMPERATURE = 0.05, 20.0


class MetricError(ValueError):
    """A prediction or gold distribution cannot be scored."""


@dataclass(frozen=True)
class ItemMetrics:
    correct: bool
    soft_accuracy: float
    nll: float
    brier: float
    kl: float
    js: float
    confidence: float
    score_error: float | None  # |E_p - E_y| in levels, for ordered (score) questions only


def normalize(probs: Mapping[str, float], options: Sequence[str]) -> list[float]:
    """The prediction as a probability vector in option order (renormalized to sum to 1)."""
    extra = set(probs) - set(options)
    if extra:
        raise MetricError(f"prediction names unknown options: {sorted(extra)}")
    vector = [float(probs.get(o, 0.0)) for o in options]
    if any(p < 0 or math.isnan(p) for p in vector):
        raise MetricError("a probability is negative or not a number")
    total = sum(vector)
    if total <= 0:
        raise MetricError("the probabilities sum to zero")
    return [p / total for p in vector]


def item_metrics(
    options: Sequence[str],
    gold: Mapping[str, float],
    probs: Mapping[str, float],
    *,
    ordered: bool = False,
) -> ItemMetrics:
    """Score one prediction. ``options`` gives the order (levels for ordered questions)."""
    p = normalize(probs, options)
    y = [float(gold.get(o, 0.0)) for o in options]
    top = max(y)
    predicted = max(range(len(p)), key=lambda i: (p[i], -i))  # first maximum wins
    kl = sum(yi * math.log(yi / max(pi, EPS)) for yi, pi in zip(y, p, strict=True) if yi > 0)
    mid = [(yi + pi) / 2 for yi, pi in zip(y, p, strict=True)]
    js = 0.5 * _kl(y, mid) + 0.5 * _kl(p, mid)
    error = None
    if ordered:
        expected_p = sum(i * pi for i, pi in enumerate(p))
        expected_y = sum(i * yi for i, yi in enumerate(y))
        error = abs(expected_p - expected_y)
    return ItemMetrics(
        correct=y[predicted] >= top - EPS,
        soft_accuracy=sum(pi * yi for pi, yi in zip(p, y, strict=True)),
        nll=-sum(yi * math.log(max(pi, EPS)) for yi, pi in zip(y, p, strict=True) if yi > 0),
        brier=sum((pi - yi) ** 2 for pi, yi in zip(p, y, strict=True)),
        kl=kl,
        js=js,
        confidence=max(p),
        score_error=error,
    )


def _kl(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * math.log(x / max(y, EPS)) for x, y in zip(a, b, strict=True) if x > 0)


def ece(confidences: Sequence[float], correct: Sequence[bool], bins: int = ECE_BINS) -> float:
    """Expected calibration error over ``bins`` equal-width confidence bins."""
    n = len(confidences)
    if n == 0:
        return float("nan")
    total, hits, seen = [0.0] * bins, [0.0] * bins, [0] * bins
    for c, ok in zip(confidences, correct, strict=True):
        b = min(int(c * bins), bins - 1)
        total[b] += c
        hits[b] += ok
        seen[b] += 1
    return sum(abs(total[b] - hits[b]) for b in range(bins) if seen[b]) / n


def coverage_curve(
    confidences: Sequence[float], correct: Sequence[bool], points: int = 20
) -> list[tuple[float, float, float]]:
    """Accuracy against coverage if only answers above a confidence are accepted.

    Returns ``(coverage, accuracy, threshold)`` from the most confident answers
    downwards: what share can be automated, and how accurate it is (plan 3.2).
    """
    n = len(confidences)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: -confidences[i])
    curve, hits = [], 0
    checkpoints = {max(1, round(n * (k + 1) / points)) for k in range(points)}
    for rank, i in enumerate(order, start=1):
        hits += correct[i]
        if rank in checkpoints:
            curve.append((rank / n, hits / rank, confidences[i]))
    return curve


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile, for latency percentiles and bootstrap intervals."""
    if not values:
        return float("nan")
    s = sorted(values)
    position = (len(s) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return s[low] + (s[high] - s[low]) * (position - low)


# -- bootstrap ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    groups: Sequence[object] | None = None,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Mean and a percentile confidence interval, resampling whole ``groups``.

    Items of one group (the two halves of a minimal pair, the five questions of one
    case) are not independent, so they are resampled together. Without ``groups``
    every item is its own group.
    """
    return _resample(values, groups, n_boot, seed, level)


def paired_difference_ci(
    a: Sequence[float],
    b: Sequence[float],
    groups: Sequence[object] | None = None,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Mean of ``a - b`` on the same items, with a paired (group-wise) bootstrap interval.

    An interval containing 0 means a tie (plan 3.3).
    """
    if len(a) != len(b):
        raise MetricError("paired comparison needs the same items in both runs")
    return bootstrap_ci(
        [x - y for x, y in zip(a, b, strict=True)], groups, n_boot=n_boot, seed=seed, level=level
    )


def _resample(values, groups, n_boot, seed, level):
    if not values:
        nan = float("nan")
        return nan, nan, nan
    keys = list(range(len(values))) if groups is None else list(groups)
    if len(keys) != len(values):
        raise MetricError("groups and values differ in length")
    sums: dict[object, list[float]] = {}
    for key, v in zip(keys, values, strict=True):
        cell = sums.setdefault(key, [0.0, 0])
        cell[0] += v
        cell[1] += 1
    cells = list(sums.values())
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        picked = [cells[rng.randrange(len(cells))] for _ in cells]
        means.append(sum(c[0] for c in picked) / sum(c[1] for c in picked))
    tail = (1 - level) / 2
    mean = sum(values) / len(values)
    return mean, quantile(means, tail), quantile(means, 1 - tail)


# -- temperature scaling --------------------------------------------------------------------


def apply_temperature(probs: Mapping[str, float], temperature: float) -> dict[str, float]:
    """``softmax(log p / T)``: T > 1 flattens an overconfident answer, T < 1 sharpens it.

    The chosen option never changes; only the stated confidence does.
    """
    if temperature <= 0:
        raise MetricError("temperature must be positive")
    logits = {k: math.log(max(p, EPS)) / temperature for k, p in probs.items()}
    top = max(logits.values())
    weights = {k: math.exp(z - top) for k, z in logits.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def fit_temperature(
    examples: Sequence[tuple[Mapping[str, float], Mapping[str, float]]],
) -> float:
    """The temperature that minimizes mean NLL against the gold, on ``(probs, gold)`` pairs.

    NLL is convex in 1/T, so a golden-section search over log T finds the optimum.
    """
    if not examples:
        raise MetricError("no examples to fit a temperature on")

    def loss(log_t: float) -> float:
        t = math.exp(log_t)
        total = 0.0
        for probs, gold in examples:
            scaled = apply_temperature(probs, t)
            total -= sum(
                y * math.log(max(scaled.get(k, 0.0), EPS)) for k, y in gold.items() if y > 0
            )
        return total / len(examples)

    low, high = math.log(MIN_TEMPERATURE), math.log(MAX_TEMPERATURE)
    ratio = (math.sqrt(5) - 1) / 2
    a, b = low, high
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = loss(c), loss(d)
    while b - a > 1e-5:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = loss(d)
    return math.exp((a + b) / 2)
