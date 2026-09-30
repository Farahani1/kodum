"""Score predictions against gold records and report the plan's breakdowns (plan 3.2).

``evaluate`` joins gold records with a prediction file. Items without a prediction
are counted as missing, failed requests as failed; neither is scored, and both are
reported, so a model cannot look better by skipping hard items. Results can be
broken down by any record field (``question_type``, ``task_family``,
``question_lang`` ...) or ``extra.<key>``, and by whether a task family was seen in
training.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

from kodoom.calibration import Calibration
from kodoom.metrics import (
    ItemMetrics,
    apply_temperature,
    bootstrap_ci,
    coverage_curve,
    ece,
    item_metrics,
    quantile,
)
from kodoom.predictions import Prediction
from kodoom.schema import Record

NAN = float("nan")


@dataclass(frozen=True)
class Scored:
    record: Record
    metrics: ItemMetrics
    latency_ms: float | None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Summary:
    n: int
    accuracy: float
    accuracy_low: float
    accuracy_high: float
    soft_accuracy: float
    nll: float
    brier: float
    kl: float
    js: float
    confidence: float
    ece_15: float
    score_n: int
    score_mae: float
    score_within_one: float
    latency_p50: float
    latency_p95: float

    def to_dict(self) -> dict[str, float | None]:
        return {
            k: None if isinstance(v, float) and math.isnan(v) else v for k, v in vars(self).items()
        }


@dataclass(frozen=True)
class PairReport:
    """Minimal pairs: a pair counts as solved only when both halves are right."""

    pairs: int
    pair_accuracy: float
    item_accuracy: float  # accuracy over the same items, for comparison
    one_right: int  # pairs where exactly one half is right (a guess, or ignoring the fact)


@dataclass
class Evaluation:
    items: list[Scored]
    missing: list[str]  # gold ids that have no prediction
    failed: list[str]  # gold ids whose request failed
    unexpected: list[str]  # prediction ids that match no gold record

    def summary(self, *, n_boot: int = 1000, seed: int = 0) -> Summary:
        return summarize(self.items, n_boot=n_boot, seed=seed)

    def by(self, name: str, *, n_boot: int = 200, seed: int = 0) -> dict[str, Summary]:
        groups: dict[str, list[Scored]] = defaultdict(list)
        for item in self.items:
            groups[group_value(item, name)].append(item)
        return {k: summarize(v, n_boot=n_boot, seed=seed) for k, v in sorted(groups.items())}

    def pairs(self) -> PairReport | None:
        halves: dict[str, list[Scored]] = defaultdict(list)
        for item in self.items:
            pair_id = item.record.extra.get("pair_id")
            if pair_id:
                halves[pair_id].append(item)
        both = [h for h in halves.values() if len(h) == 2]
        if not both:
            return None
        right = [sum(i.metrics.correct for i in h) for h in both]
        return PairReport(
            pairs=len(both),
            pair_accuracy=sum(r == 2 for r in right) / len(both),
            item_accuracy=sum(right) / (2 * len(both)),
            one_right=sum(r == 1 for r in right),
        )

    def coverage(self, points: int = 10) -> list[tuple[float, float, float]]:
        confidences = [i.metrics.confidence for i in self.items]
        return coverage_curve(confidences, [i.metrics.correct for i in self.items], points)


def evaluate(
    records: Sequence[Record],
    predictions: Mapping[str, Prediction],
    *,
    calibration: Calibration | None = None,
    seen_families: Collection[str] | None = None,
) -> Evaluation:
    """Score ``predictions`` against ``records``; optionally recalibrate first."""
    items, missing, failed = [], [], []
    for record in records:
        prediction = predictions.get(record.id)
        if prediction is None:
            missing.append(record.id)
            continue
        if prediction.error:
            failed.append(record.id)
            continue
        probs = prediction.probs
        if calibration is not None:
            probs = apply_temperature(probs, calibration.for_type(record.question_type))
        metrics = item_metrics(
            [o.id for o in record.options],
            record.gold,
            probs,
            ordered=record.question_type == "score",
        )
        tags = {}
        if seen_families is not None:
            tags["family_status"] = "seen" if record.task_family in seen_families else "unseen"
        items.append(Scored(record, metrics, prediction.latency_ms, tags))
    known = {r.id for r in records}
    return Evaluation(items, missing, failed, sorted(set(predictions) - known))


def group_value(item: Scored, name: str) -> str:
    if name in item.tags:
        return item.tags[name]
    if name.startswith("extra."):
        value = item.record.extra.get(name.removeprefix("extra."))
    else:
        value = getattr(item.record, name, None)
    return "(none)" if value is None else str(value)


def summarize(items: Sequence[Scored], *, n_boot: int = 1000, seed: int = 0) -> Summary:
    n = len(items)
    if n == 0:
        return Summary(0, *([NAN] * 10), 0, NAN, NAN, NAN, NAN)  # type: ignore[arg-type]
    m = [i.metrics for i in items]
    correct = [float(x.correct) for x in m]
    groups = [
        i.record.source_id for i in items
    ]  # a case's questions and a pair's halves go together
    accuracy, low, high = bootstrap_ci(correct, groups, n_boot=n_boot, seed=seed)
    score_errors = [x.score_error for x in m if x.score_error is not None]
    latencies = [i.latency_ms for i in items if i.latency_ms is not None]

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    return Summary(
        n=n,
        accuracy=accuracy,
        accuracy_low=low,
        accuracy_high=high,
        soft_accuracy=mean([x.soft_accuracy for x in m]),
        nll=mean([x.nll for x in m]),
        brier=mean([x.brier for x in m]),
        kl=mean([x.kl for x in m]),
        js=mean([x.js for x in m]),
        confidence=mean([x.confidence for x in m]),
        ece_15=ece([x.confidence for x in m], [x.correct for x in m]),
        score_n=len(score_errors),
        score_mae=mean(score_errors) if score_errors else NAN,
        score_within_one=mean([e <= 1 for e in score_errors]) if score_errors else NAN,
        latency_p50=quantile(latencies, 0.5),
        latency_p95=quantile(latencies, 0.95),
    )


COLUMNS = (
    ("n", "n", "{:>6d}"),
    ("accuracy", "acc", "{:>6.3f}"),
    ("soft_accuracy", "soft", "{:>6.3f}"),
    ("nll", "nll", "{:>6.3f}"),
    ("brier", "brier", "{:>6.3f}"),
    ("kl", "kl", "{:>6.3f}"),
    ("confidence", "conf", "{:>6.3f}"),
    ("ece_15", "ece15", "{:>6.3f}"),
)


def format_table(rows: Mapping[str, Summary], title: str) -> str:
    width = max([len(title), *(len(k) for k in rows)]) + 1
    header = " ".join(f"{label:>6}" for _, label, _ in COLUMNS)
    lines = [f"{title.ljust(width)} {header}  acc 95% CI"]
    for name, s in rows.items():
        cells = " ".join(fmt.format(getattr(s, attr)) for attr, _, fmt in COLUMNS)
        interval = f"[{s.accuracy_low:.3f}, {s.accuracy_high:.3f}]" if s.n else ""
        lines.append(f"{name.ljust(width)} {cells}  {interval}")
    return "\n".join(lines)
