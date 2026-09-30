"""helmo/synthetic-typed-decisions as decision records (plan 1.1).

Layout read from the real dataset with ``kodoom inspect`` (commit ``REVISION``): one
JSON-lines file, ``synthetic_train.jsonl``, one row per single-question record with
``topic``, ``state`` (plain text), ``questions`` (a JSON string holding one question,
``q1``) and ``gold`` (a JSON string). There is no id, so the id is the row's position
in the pinned file. The gold has a different shape per type (dataset card):

- ``choice``: ``{"probabilities": {"B": 1.0}}``, over the option ids; a few rows sum
  to less than 1 (0.95), those are scaled to 1 and flagged in ``extra``;
- ``score``: ``{"mean": 2.0, "variance": 0.0}``, a 0-based level, fractional when
  between two levels; there is no distribution, so it is turned into the two-level
  distribution with exactly that mean (the variance stays in ``extra``);
- ``noul``: ``{"noul": 0.15}``, the probability that the statement is true; the
  criteria are empty, so the options are "No" and "Yes".
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from kodoom.schema import Record
from kodoom.sources import get_source
from kodoom.typed_decisions import (
    QUESTION_TYPES,
    TypedDecisionsError,
    _json,
    _options,
    _str,
)

SOURCE = "helmo/synthetic-typed-decisions"
REVISION = "1827dc0d7f5a77f172107bfa25a49d2eda50b32e"
FILENAME = "synthetic_train.jsonl"
QUESTION = "q1"


def row_record(row: dict[str, Any], index: int, *, revision: str = REVISION) -> Record:
    """The record of one row; ``index`` is its 0-based position in the file."""
    where = f"row {index}"
    topic = _str(row, "topic", where)
    state = _str(row, "state", where)
    questions = _json(row, "questions", where)
    gold = _json(row, "gold", where)
    if not isinstance(questions, dict) or set(questions) != {QUESTION}:
        raise TypedDecisionsError(f"{where}: questions must hold exactly {QUESTION!r}")
    spec = questions[QUESTION]
    qtype = spec.get("type") if isinstance(spec, dict) else None
    if qtype not in QUESTION_TYPES:
        raise TypedDecisionsError(f"{where}: type must be one of {QUESTION_TYPES}, got {qtype!r}")
    if not isinstance(gold, dict) or not isinstance(gold.get(QUESTION), dict):
        raise TypedDecisionsError(f"{where}: no gold for {QUESTION!r}")
    options = _options(spec, qtype, where)
    probabilities, extra = _gold(gold[QUESTION], qtype, [o.id for o in options], where)
    return Record(
        id=f"helmo-{index:05d}",
        source_id=f"helmo-{index:05d}",
        source=SOURCE,
        source_revision=revision,
        license=get_source(SOURCE).license,
        split="train",
        origin="synthetic",
        task_family="topics",
        state_lang="en",
        question_lang="en",
        state=state,
        question_type=qtype,
        question_text=_str(spec, "instructions", where),
        options=options,
        gold=probabilities,
        extra={"topic": topic, "row": index, **extra},
    )


def _gold(
    g: dict[str, Any], qtype: str, ids: list[str], where: str
) -> tuple[dict[str, float], dict[str, Any]]:
    if qtype == "noul":
        p = _number(g, "noul", where)
        return {"false": 1.0 - p, "true": p}, {"gold_noul": p}
    if qtype == "choice":
        raw = g.get("probabilities")
        if not isinstance(raw, dict) or not raw:
            raise TypedDecisionsError(f"{where}: a choice gold needs 'probabilities'")
        unknown = set(raw) - set(ids)
        if unknown:
            raise TypedDecisionsError(f"{where}: gold names unknown options {sorted(unknown)}")
        values = {i: float(raw.get(i, 0.0)) for i in ids}
        total = math.fsum(values.values())
        if total <= 0:
            raise TypedDecisionsError(f"{where}: choice gold has no probability mass")
        if abs(total - 1.0) <= 1e-6:
            return values, {}
        # The source's own gold does not always sum to 1 (row 33: 0.95). It is scaled to
        # sum to 1 and the original sum is kept, so the record can be found and dropped.
        return {i: p / total for i, p in values.items()}, {"gold_sum_in_source": total}
    mean = _number(g, "mean", where, high=len(ids) - 1)
    low = math.floor(mean)
    high_share = mean - low
    probabilities = {str(i): 0.0 for i in range(len(ids))}
    probabilities[str(low)] = 1.0 - high_share
    if high_share > 0:
        probabilities[str(low + 1)] = high_share
    return probabilities, {"gold_mean": mean, "gold_variance": g.get("variance")}


def _number(g: dict[str, Any], key: str, where: str, *, high: float = 1.0) -> float:
    value = g.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= high:
        raise TypedDecisionsError(f"{where}: gold {key!r} must be a number in [0, {high}]")
    return float(value)


def read_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise TypedDecisionsError(f"{path}, line {n}: not valid JSON: {e}") from e


def load_records(
    download: Callable[..., str], *, revision: str = REVISION, limit: int | None = None
) -> list[Record]:
    """All records of the pinned file, or the first ``limit``.

    ``download(repo_id=, filename=, repo_type=, revision=)`` returns a local path, as
    ``huggingface_hub.hf_hub_download`` does.
    """
    path = download(repo_id=SOURCE, filename=FILENAME, repo_type="dataset", revision=revision)
    records = []
    for index, row in enumerate(read_rows(path)):
        if limit is not None and index >= limit:
            break
        records.append(row_record(row, index, revision=revision))
    return records
