"""Translate typed-decisions cases into Persian records (plan 1.2).

A ``Translator`` turns English texts into Persian ones; everything around it is the
same for every translator: which texts go in (``rules``), what must come back
(``checks``), how the records are rebuilt, and resuming after a dropped session.
Gold, option ids, splits and ``source_id`` are copied from the English records and
never touched. The published translation gets ``clean_orthography`` only, never the
full normalizer (plan 1.2 step 6).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from kodoom.normalize import clean_orthography
from kodoom.schema import Option, Record, append_jsonl, read_jsonl
from kodoom.translate.checks import (
    _BACKTICKED,
    _EMAIL,
    _IDENTIFIER,
    _URL,
    Finding,
    check_state,
    check_text,
    protected_tokens,
)
from kodoom.translate.rules import FORMAL, apply, segments

STATE, QUESTION, OPTION = "state", "question", "option"


@dataclass(frozen=True)
class Item:
    """One text to translate, with what a translator may use to choose its wording."""

    text: str
    register: str  # rules.FORMAL or rules.COLLOQUIAL
    kind: str  # STATE, QUESTION or OPTION


class Translator(Protocol):
    name: str

    def translate(self, items: Sequence[Item]) -> list[str]:
        """One Persian text per item, in the same order."""


class StubTranslator:
    """Pretends to translate: every English word becomes a Persian word picked by the
    English word's spelling (so the same word always gives the same one and the text
    does not look like a loop), and code, identifiers, numbers and punctuation stay.
    It exists so the whole pipeline runs on the laptop (dev profile) and in tests; it
    must never produce published data."""

    name = "stub"
    _VOCABULARY = (
        "کار", "میز", "راه", "نام", "دست", "شهر", "باغ", "کتاب",
        "پنجره", "ستاره", "دریا", "چراغ", "خانه", "درخت", "سفر", "ماه",
    )  # fmt: skip
    _PIECE = re.compile(
        rf"(?P<keep>{_BACKTICKED.pattern}|{_URL.pattern}|{_EMAIL.pattern}|{_IDENTIFIER.pattern})"
        r"|(?P<word>[A-Za-z]+)"
    )

    def translate(self, items: Sequence[Item]) -> list[str]:
        return [self._one(item.text) for item in items]

    def _word(self, english: str) -> str:
        return self._VOCABULARY[sum(map(ord, english.lower())) % len(self._VOCABULARY)]

    def _one(self, text: str) -> str:
        def piece(m: re.Match) -> str:
            if m.group("keep"):
                token = m.group("keep")
                if token in protected_tokens(token):
                    return token
                return re.sub(r"[A-Za-z]+", lambda w: self._word(w.group()), token)
            return self._word(m.group("word"))

        return self._PIECE.sub(piece, text)


def translate_case(
    records: Sequence[Record], translator: Translator
) -> tuple[list[Record], list[Finding]]:
    """Persian records for one case (all its questions) and the findings of the checks."""
    if not records:
        raise ValueError("a case needs at least one record")
    first = records[0]
    if {r.source_id for r in records} != {first.source_id} or {r.state for r in records} != {
        first.state
    }:
        raise ValueError(f"records of case {first.source_id!r} must share one source_id and state")
    workflow = first.extra["workflow"]

    todo = [s for s in segments(workflow, first.state) if s.rule.action == "translate"]
    items = [Item(s.text, s.rule.register, STATE) for s in todo]
    for r in records:
        items.append(Item(r.question_text, FORMAL, QUESTION))
        items += [Item(o.text, FORMAL, OPTION) for o in r.options]
    out = [clean_orthography(t) for t in translator.translate(items)]
    if len(out) != len(items):
        raise ValueError(f"translator returned {len(out)} texts for {len(items)} items")

    state = apply(first.state, {s.location: out[i] for i, s in enumerate(todo)})
    state_findings = check_state(workflow, first.state, state)
    all_findings = list(state_findings)
    cursor = len(todo)
    result = []
    for r in records:
        question = out[cursor]
        options = out[cursor + 1 : cursor + 1 + len(r.options)]
        cursor += 1 + len(r.options)
        name = r.extra["question"]
        own = check_text(r.question_text, question, f"{name}.question")
        for o, text in zip(r.options, options, strict=True):
            own += check_text(o.text, text, f"{name}.option.{o.id}")
        all_findings += own
        found = state_findings + own
        result.append(
            replace(
                r,
                id=f"{r.id}:fa",
                origin="translated",
                state_lang="fa",
                question_lang="fa",
                state=state,
                question_text=question,
                options=tuple(Option(o.id, t) for o, t in zip(r.options, options, strict=True)),
                checks_passed=not found,
                extra={
                    **r.extra,
                    "translator": translator.name,
                    **({"check_findings": [vars(f) for f in found]} if found else {}),
                },
            )
        )
    return result, all_findings


def cases(records: Iterable[Record]) -> list[list[Record]]:
    """Records grouped by case, in file order (a case's questions are adjacent)."""
    grouped: dict[str, list[Record]] = {}
    for r in records:
        grouped.setdefault(r.source_id, []).append(r)
    return list(grouped.values())


def translate_file(
    source: str | Path,
    out: str | Path,
    translator: Translator,
    *,
    limit: int | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, int]:
    """Translate every case of ``source`` into ``out``, one case at a time.

    Each finished case is appended at once, so a rerun resumes after the last case that
    reached the file. Returns counts of cases done now, skipped (already there) and
    cases with findings.
    """
    out = Path(out)
    done = {r.source_id for r in read_jsonl(out)} if out.exists() else set()
    stats = {"translated": 0, "skipped": 0, "with_findings": 0}
    for n, case in enumerate(cases(read_jsonl(source))):
        if limit is not None and n >= limit:
            break
        if case[0].source_id in done:
            stats["skipped"] += 1
            continue
        translated, findings = translate_case(case, translator)
        for r in translated:
            append_jsonl(out, r)
        stats["translated"] += 1
        stats["with_findings"] += bool(findings)
        progress(case[0].source_id)
    return stats


# Translators by name for `kodoom translate`; real ones are added with the pilot.
TRANSLATORS: dict[str, Callable[[], Translator]] = {"stub": StubTranslator}
