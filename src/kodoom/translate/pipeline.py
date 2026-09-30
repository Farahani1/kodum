"""Translate typed-decisions cases into Persian records (plan 1.2).

A ``Translator`` turns English texts into Persian ones; everything around it is the
same for every translator: which texts go in (``rules``), what must come back
(``checks``), how the records are rebuilt, and resuming after a dropped session.
Gold, option ids, splits and ``source_id`` are copied from the English records and
never touched. The published translation gets ``clean_orthography`` only, never the
full normalizer (plan 1.2 step 6).
"""

from __future__ import annotations

import json
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
from kodoom.translate.glossary import Glossary
from kodoom.translate.glossary import load as load_glossary
from kodoom.translate.rules import FORMAL, apply, segments

STATE, QUESTION, OPTION = "state", "question", "option"


class TranslationError(ValueError):
    """The translator's output cannot become a record (for example an empty text)."""


@dataclass(frozen=True)
class Item:
    """One text to translate, with what a translator may use to choose its wording."""

    text: str
    register: str  # rules.FORMAL or rules.COLLOQUIAL
    kind: str  # STATE, QUESTION or OPTION
    workflow: str = ""  # which glossary terms apply ("agent" differs per workflow)


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

    def __init__(self, use_glossary: bool = True) -> None:
        self.glossary = load_glossary() if use_glossary else None

    _VOCABULARY = (
        "کار", "میز", "راه", "نام", "دست", "شهر", "باغ", "کتاب",
        "پنجره", "ستاره", "دریا", "چراغ", "خانه", "درخت", "سفر", "ماه",
    )  # fmt: skip
    _TOKEN = (
        rf"(?P<keep>{_BACKTICKED.pattern}|{_URL.pattern}|{_EMAIL.pattern}|{_IDENTIFIER.pattern})"
        r"|(?P<word>[A-Za-z]+)"
    )

    def translate(self, items: Sequence[Item]) -> list[str]:
        return [self._one(item) for item in items]

    def _word(self, english: str) -> str:
        return self._VOCABULARY[sum(map(ord, english.lower())) % len(self._VOCABULARY)]

    def _one(self, item: Item) -> str:
        terms = self.glossary.terms(item.workflow) if self.glossary else {}
        keep = self.glossary.keep if self.glossary else ()
        pieces = []
        if terms:  # longest first, so "service account" wins over "account"
            names = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
            pieces.append(rf"(?P<term>\b(?:{names})s?\b)")
        if keep:
            kept = "|".join(map(re.escape, keep))
            pieces.append(rf"(?P<kept>(?<![A-Za-z0-9])(?:{kept})(?![A-Za-z0-9]))")
        pattern = re.compile("|".join([*pieces, self._TOKEN]), re.IGNORECASE)

        def piece(m: re.Match) -> str:
            if m.groupdict().get("kept"):
                return m.group("kept")
            if m.groupdict().get("term"):
                key = m.group("term").lower()
                return terms[key if key in terms else key[:-1]]  # a plural takes the same term
            if m.group("keep"):
                token = m.group("keep")
                if token in protected_tokens(token):
                    return token
                return re.sub(r"[A-Za-z]+", lambda w: self._word(w.group()), token)
            return self._word(m.group("word"))

        return pattern.sub(piece, item.text)


def translate_case(
    records: Sequence[Record], translator: Translator, glossary: Glossary | None = None
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
    glossary = glossary if glossary is not None else load_glossary()

    todo = [s for s in segments(workflow, first.state) if s.rule.action == "translate"]
    items = [Item(s.text, s.rule.register, STATE, workflow) for s in todo]
    for r in records:
        items.append(Item(r.question_text, FORMAL, QUESTION, workflow))
        items += [Item(o.text, FORMAL, OPTION, workflow) for o in r.options]
    out = [clean_orthography(t) for t in translator.translate(items)]
    if len(out) != len(items):
        raise ValueError(f"translator returned {len(out)} texts for {len(items)} items")

    for item, text in zip(items, out, strict=True):
        if item.kind != STATE and not text.strip():
            raise TranslationError(
                f"case {first.source_id}: {translator.name} returned an empty text for the "
                f"{item.kind} {item.text!r}"
            )
    state = apply(first.state, {s.location: out[i] for i, s in enumerate(todo)})
    state_findings = check_state(workflow, first.state, state)
    for i, seg in enumerate(todo):
        state_findings += glossary.check(workflow, seg.text, out[i], "state." + seg.path)
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
            own += glossary.check(workflow, o.text, text, f"{name}.option.{o.id}")
        own += glossary.check(workflow, r.question_text, question, f"{name}.question")
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
    reached the file. A case the translator cannot translate (an empty question, say) is
    not written; it is counted as failed and logged with its reason in
    ``<out stem>.failures.jsonl`` next to ``out``. Returns counts of cases translated now,
    skipped (already there), with findings and failed.
    """
    out = Path(out)
    done = {r.source_id for r in read_jsonl(out)} if out.exists() else set()
    stats = {"translated": 0, "skipped": 0, "with_findings": 0, "failed": 0}
    for n, case in enumerate(cases(read_jsonl(source))):
        if limit is not None and n >= limit:
            break
        if case[0].source_id in done:
            stats["skipped"] += 1
            continue
        try:
            translated, findings = translate_case(case, translator)
        except TranslationError as e:
            stats["failed"] += 1
            log = out.with_name(out.stem + ".failures.jsonl")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"source_id": case[0].source_id, "error": str(e)}) + "\n")
            continue
        for r in translated:
            append_jsonl(out, r)
        stats["translated"] += 1
        stats["with_findings"] += bool(findings)
        progress(case[0].source_id)
    return stats


# Translators by name for `kodoom translate`. The model ones load lazily (torch and
# transformers, Colab only) and are looked up through `translator_factory`.
TRANSLATORS: dict[str, Callable[[], Translator]] = {"stub": StubTranslator}
MODEL_TRANSLATORS = (
    "translategemma-4b",
    "translategemma-4b-bf16",
    "translategemma-4b-bf16-terms",
    "translategemma-4b-4bit-fp32",
    "translategemma-12b-4bit",
    "qwen3-8b-4bit",
)


def translator_factory(name: str) -> Callable[[], Translator]:
    if name in TRANSLATORS:
        return TRANSLATORS[name]
    if name in MODEL_TRANSLATORS:
        from kodoom.translate import hf  # imports no torch until a model is loaded

        return {
            "translategemma-4b": hf.translategemma_4b,
            "translategemma-4b-bf16": hf.translategemma_4b_bf16,
            "translategemma-4b-bf16-terms": hf.translategemma_4b_bf16_terms,
            "translategemma-4b-4bit-fp32": hf.translategemma_4b_4bit_fp32,
            "translategemma-12b-4bit": hf.translategemma_12b_4bit,
            "qwen3-8b-4bit": hf.qwen3_8b_4bit,
        }[name]
    raise KeyError(f"unknown translator {name!r}")
