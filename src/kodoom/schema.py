"""The one record schema for translated, native and synthetic data (plan 1.3).

A record is one decision: a state, one question and its options, and the gold
answer as a probability distribution over option IDs. Records are stored as
JSON Lines, one record per line, UTF-8, Persian kept as-is (not escaped).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

QUESTION_TYPES = ("choice", "score", "noul")
SPLITS = ("train", "validation", "calibration", "test")
ORIGINS = ("translated", "native", "synthetic")
LANGS = ("fa", "en")

# Soft labels from source datasets may be rounded (0.33 + 0.33 + 0.33).
GOLD_SUM_TOLERANCE = 1e-3


class RecordError(ValueError):
    """A record breaks the schema."""


@dataclass(frozen=True)
class Option:
    id: str
    text: str


@dataclass(frozen=True)
class Record:
    id: str
    source_id: str
    source: str
    source_revision: str | None
    license: str
    split: str
    origin: str
    state_lang: str
    question_lang: str
    state: str
    question_type: str
    question_text: str
    options: tuple[Option, ...]
    gold: dict[str, float]
    # Quality flags (plan 1.3). None: the check has not run on this record.
    checks_passed: bool | None = None
    meaning_flag: bool | None = None
    human_reviewed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate(self)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["options"] = [asdict(o) for o in self.options]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Record:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise RecordError(f"unknown fields: {sorted(unknown)}")
        try:
            options = tuple(Option(**o) for o in d["options"])
        except (KeyError, TypeError) as e:
            raise RecordError(f"bad options: {e}") from e
        try:
            return cls(**{**d, "options": options})
        except TypeError as e:  # missing required fields
            raise RecordError(str(e)) from e


def _validate(r: Record) -> None:
    def fail(msg: str) -> None:
        raise RecordError(f"record {r.id!r}: {msg}")

    for name in ("id", "source_id", "source", "license", "state", "question_text"):
        value = getattr(r, name)
        if not isinstance(value, str) or not value.strip():
            fail(f"{name} must be a non-empty string")
    for name, allowed in (
        ("split", SPLITS),
        ("origin", ORIGINS),
        ("question_type", QUESTION_TYPES),
        ("state_lang", LANGS),
        ("question_lang", LANGS),
    ):
        if getattr(r, name) not in allowed:
            fail(f"{name} must be one of {allowed}, got {getattr(r, name)!r}")

    ids = [o.id for o in r.options]
    if len(ids) < 2:
        fail("needs at least 2 options")
    if r.question_type == "noul" and len(ids) != 2:
        fail("a noul (yes/no) question needs exactly 2 options")
    if len(set(ids)) != len(ids):
        fail("option IDs must be unique")
    for o in r.options:
        if not o.id or not o.text.strip():
            fail("every option needs an ID and text")

    if not r.gold:
        fail("gold is empty")
    extra_ids = set(r.gold) - set(ids)
    if extra_ids:
        fail(f"gold names options that do not exist: {sorted(extra_ids)}")
    for option_id, p in r.gold.items():
        if isinstance(p, bool) or not isinstance(p, int | float) or not 0.0 <= p <= 1.0:
            fail(f"gold[{option_id!r}] must be a probability, got {p!r}")
    total = math.fsum(r.gold.values())
    if abs(total - 1.0) > GOLD_SUM_TOLERANCE:
        fail(f"gold must sum to 1, sums to {total:.6f}")


def one_hot(option_ids: Iterable[str], correct: str) -> dict[str, float]:
    """Gold for a source with hard labels (plan 1.3)."""
    ids = list(option_ids)
    if correct not in ids:
        raise RecordError(f"correct option {correct!r} is not among {ids}")
    return {i: 1.0 if i == correct else 0.0 for i in ids}


def read_jsonl(path: str | Path) -> Iterator[Record]:
    """Yield records from a JSONL file; errors name the file and line.

    A last line without a newline is a write cut off by a dropped session
    (see ``append_jsonl``) and is skipped; anywhere else, a bad line is an error.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            if not line.endswith("\n"):
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    return
            try:
                yield Record.from_dict(json.loads(line))
            except (json.JSONDecodeError, RecordError) as e:
                raise RecordError(f"{path}:{n}: {e}") from e


def write_jsonl(path: str | Path, records: Iterable[Record]) -> int:
    """Write all records, replacing the file only once writing succeeded."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(_dumps(r))
            count += 1
    os.replace(tmp, path)
    return count


def append_jsonl(path: str | Path, record: Record) -> None:
    """Append one record and flush it to disk, so a dropped session keeps it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _drop_partial_last_line(path)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(_dumps(record))
        f.flush()
        os.fsync(f.fileno())


def _drop_partial_last_line(path: Path) -> None:
    """Cut a half-written last line left by an interrupted append."""
    if not path.exists():
        return
    with path.open("rb+") as f:
        size = f.seek(0, os.SEEK_END)
        if size == 0:
            return
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return
        # Walk back to the previous newline and truncate after it.
        pos = size - 1
        chunk = 4096
        while pos > 0:
            start = max(0, pos - chunk)
            f.seek(start)
            newline = f.read(pos - start).rfind(b"\n")
            if newline != -1:
                f.truncate(start + newline + 1)
                return
            pos = start
        f.truncate(0)


def _dumps(r: Record) -> str:
    return json.dumps(r.to_dict(), ensure_ascii=False) + "\n"
