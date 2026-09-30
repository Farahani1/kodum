"""LocalLLaMA/typed-decisions as decision records (plan 1.1, 1.3).

One row of the dataset is a *case*: a piece of state and five typed questions about
it. Here every question becomes one record, so a case's questions share a
``source_id`` and always land in the same split. The layout below was read from the
real dataset with ``kodoom inspect`` (commit ``REVISION``), not assumed:

- ``state``: a JSON string; kept byte-identical as the record's ``state``;
- ``questions``: JSON ``{name: {type, instructions, criteria}}``; for ``choice`` and
  ``noul`` the criteria map option id -> description (yes/no uses ``"false"`` and
  ``"true"``; a yes/no question may have no criteria at all, then the options are
  "No" and "Yes"), for ``score`` the criteria is the list of level descriptions and the
  option ids are the level indexes ``"0"``, ``"1"``, ...;
- ``gold``: JSON ``{name: {label, confidence, probabilities, ...}}``, a full
  distribution per question over the same option ids;
- ``factors`` and ``label_agreement`` describe how a case was built; they are not
  model input, are kept in ``extra`` and never translated.

The English original is the reference for typed-decisions-fa: same case ids, same
splits, same gold.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from kodoom.schema import Option, Record
from kodoom.sources import get_source

SOURCE = "LocalLLaMA/typed-decisions"
REVISION = "e135720c8fdff7896a4e4068cff624a899d41597"
WORKFLOWS = (
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
)
SPLITS = ("train", "test")
QUESTION_TYPES = ("choice", "noul", "score")


class TypedDecisionsError(ValueError):
    """A row does not have the layout this loader was written for."""


def parquet_name(workflow: str, split: str) -> str:
    return f"{workflow}/{split}-00000-of-00001.parquet"


def case_records(row: Mapping[str, Any], *, revision: str = REVISION) -> list[Record]:
    """The records of one case: one per question, in the order of the source."""
    case_id = _text(row, "id")
    where = f"case {case_id!r}"
    workflow, split = _text(row, "workflow"), _text(row, "split")
    if split not in SPLITS:
        raise TypedDecisionsError(f"{where}: split must be one of {SPLITS}, got {split!r}")
    state = _json_text(row, "state", where)
    questions = _json(row, "questions", where)
    gold = _json(row, "gold", where)
    if not isinstance(questions, dict) or not questions:
        raise TypedDecisionsError(f"{where}: questions must be a non-empty object")
    factors = _json(row, "factors", where, required=False)
    agreement = _json(row, "label_agreement", where, required=False) or {}

    license_ = get_source(SOURCE).license
    records = []
    for name, spec in questions.items():
        here = f"{where}, question {name!r}"
        qtype = spec.get("type") if isinstance(spec, dict) else None
        if qtype not in QUESTION_TYPES:
            raise TypedDecisionsError(
                f"{here}: type must be one of {QUESTION_TYPES}, got {qtype!r}"
            )
        if name not in gold or "probabilities" not in gold[name]:
            raise TypedDecisionsError(f"{here}: no gold probabilities")
        g = gold[name]
        extra: dict[str, Any] = {
            "case_id": case_id,
            "workflow": workflow,
            "question": name,
            "original_split": split,
            "gold_label": g.get("label"),
            "gold_confidence": g.get("confidence"),
            "label_agreement": agreement.get(name),
        }
        if "score" in g:
            extra["gold_score"] = g["score"]
        if factors is not None:
            extra["factors"] = factors
        records.append(
            Record(
                id=f"{case_id}:{name}",
                source_id=case_id,
                source=SOURCE,
                source_revision=revision,
                license=license_,
                split=split,
                origin="synthetic",
                task_family="workflow-" + workflow.replace("_", "-"),
                state_lang="en",
                question_lang="en",
                state=state,
                question_type=qtype,
                question_text=_str(spec, "instructions", here),
                options=_options(spec, qtype, here),
                gold={str(k): float(v) for k, v in g["probabilities"].items()},
                extra=extra,
            )
        )
    return records


def _options(spec: Mapping[str, Any], qtype: str, where: str) -> tuple[Option, ...]:
    criteria = spec.get("criteria")
    if qtype == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise TypedDecisionsError(f"{where}: a score question needs a list of levels")
        return tuple(Option(str(i), str(text)) for i, text in enumerate(criteria))
    if qtype == "noul" and not criteria:
        # Real rows give a yes/no question no criteria (invoice_processing "duplicate": the
        # key is missing; helmo: it is empty).
        return (Option("false", "No"), Option("true", "Yes"))
    if not isinstance(criteria, dict) or len(criteria) < 2:
        raise TypedDecisionsError(f"{where}: {qtype} criteria must map option ids to text")
    if qtype == "noul" and set(criteria) != {"false", "true"}:
        raise TypedDecisionsError(f"{where}: yes/no criteria must be 'false' and 'true'")
    return tuple(Option(str(key), str(text)) for key, text in criteria.items())


def read_parquet_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - exercised on machines without pyarrow
        raise TypedDecisionsError("pyarrow is not installed: pip install pyarrow") from e
    for batch in pq.ParquetFile(path).iter_batches():
        yield from batch.to_pylist()


def load_records(
    download: Callable[..., str],
    *,
    revision: str = REVISION,
    limit: int | None = None,
    workflows: Sequence[str] = WORKFLOWS,
) -> dict[str, list[Record]]:
    """All records by original split. ``limit`` keeps the first N cases per workflow and split.

    ``download(repo_id=, filename=, repo_type=, revision=)`` returns a local path, as
    ``huggingface_hub.hf_hub_download`` does; files are read at ``revision``.
    """
    result: dict[str, list[Record]] = {split: [] for split in SPLITS}
    for workflow in workflows:
        for split in SPLITS:
            path = download(
                repo_id=SOURCE,
                filename=parquet_name(workflow, split),
                repo_type="dataset",
                revision=revision,
            )
            for n, row in enumerate(read_parquet_rows(path)):
                if limit is not None and n >= limit:
                    break
                if row.get("workflow") != workflow or row.get("split") != split:
                    raise TypedDecisionsError(
                        f"{parquet_name(workflow, split)}: row {n} says workflow "
                        f"{row.get('workflow')!r}, split {row.get('split')!r}"
                    )
                result[split] += case_records(row, revision=revision)
    return result


# -- helpers ---------------------------------------------------------------------


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise TypedDecisionsError(f"a row needs a non-empty text field {key!r}, got {value!r}")
    return value


def _str(spec: Mapping[str, Any], key: str, where: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TypedDecisionsError(f"{where}: {key!r} must be non-empty text")
    return value


def _json(row: Mapping[str, Any], key: str, where: str, *, required: bool = True) -> Any:
    value = row.get(key)
    if value is None:
        if required:
            raise TypedDecisionsError(f"{where}: missing {key!r}")
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise TypedDecisionsError(f"{where}: {key!r} is not valid JSON: {e}") from e
    return value


def _json_text(row: Mapping[str, Any], key: str, where: str) -> str:
    """The state exactly as the source stores it (a JSON string); objects are serialized."""
    value = row.get(key)
    if isinstance(value, str):
        _json(row, key, where)  # must parse
        return value
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    raise TypedDecisionsError(f"{where}: {key!r} must be a JSON string or object")


# -- field statistics ------------------------------------------------------------


def field_stats(records: Sequence[Record], *, samples: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Per workflow, statistics of every text leaf of ``state``.

    Decides which fields the translation step translates and which it keeps: a
    field with few distinct values across many cases is a label or an identifier
    (keep), a long one with many distinct values is free text (translate). Each
    case counts once even though its questions share the state.
    """
    per_workflow: dict[str, dict[str, dict[str, Any]]] = {}
    seen: set[str] = set()
    for record in records:
        if record.source_id in seen:
            continue
        seen.add(record.source_id)
        stats = per_workflow.setdefault(record.extra["workflow"], {})
        for path, value in _leaves(json.loads(record.state)):
            entry = stats.setdefault(path, {"cases": 0, "values": {}, "chars": 0})
            entry["cases"] += 1
            entry["values"][value] = entry["values"].get(value, 0) + 1
            entry["chars"] += len(value)
    result: dict[str, list[dict[str, Any]]] = {}
    for workflow, stats in per_workflow.items():
        cases = sum(1 for _ in {r.source_id for r in records if r.extra["workflow"] == workflow})
        rows = []
        for path, entry in stats.items():
            total = sum(entry["values"].values())
            rows.append(
                {
                    "path": path,
                    "coverage": entry["cases"] / cases,
                    "occurrences": total,
                    "distinct": len(entry["values"]),
                    "mean_chars": entry["chars"] / total,
                    "samples": [v[:80] for v in list(entry["values"])[:samples]],
                }
            )
        result[workflow] = sorted(rows, key=lambda r: (-r["mean_chars"] * r["distinct"], r["path"]))
    return result


def _leaves(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Text leaves as (path, value); list positions collapse to ``[]``."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for value in node:
            yield from _leaves(value, f"{path}[]")
    elif isinstance(node, str):
        yield path, node
