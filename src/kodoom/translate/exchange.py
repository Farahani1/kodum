"""Translate outside the pipeline, then bring the result back through the same checks.

Some translators cannot run inside ``kodoom translate``: a person, or an assistant such as
Claude Cowork working on files. For them the pipeline writes the *units* that need
translating (each distinct text once, with its register and the glossary terms that
apply) and, later, reads the filled file back through ``PrecomputedTranslator``. The
records, the automatic checks, the failure log and the review tools are then exactly the
same as for a model run.

A unit is identified by workflow and English text. The question and option texts repeat in
every case of a workflow, so they appear once and get one Persian rendering everywhere.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from kodoom.schema import Record
from kodoom.translate.glossary import Glossary
from kodoom.translate.hf import REGISTERS, RULES, SYSTEM
from kodoom.translate.pipeline import OPTION, QUESTION, STATE, Item
from kodoom.translate.rules import FORMAL, segments


class ExchangeError(ValueError):
    """A units file cannot be used."""


def units_for(grouped: Iterable[Sequence[Record]], glossary: Glossary) -> list[dict[str, object]]:
    """The distinct texts of the given cases, in order of first appearance."""
    seen: dict[tuple[str, str], dict[str, object]] = {}

    def add(workflow: str, kind: str, register: str, text: str) -> None:
        key = (workflow, text)
        if key in seen:
            return
        seen[key] = {
            "id": f"u{len(seen) + 1:05d}",
            "workflow": workflow,
            "kind": kind,
            "register": register,
            "context": glossary.context(workflow),
            "terms": glossary.relevant(workflow, text),
            "keep_in_english": glossary.kept(text),
            "text": text,
            "fa": "",
        }

    for case in grouped:
        workflow = case[0].extra["workflow"]
        for seg in segments(workflow, case[0].state):
            if seg.rule.action == "translate":
                add(workflow, STATE, seg.rule.register, seg.text)
        for r in case:
            add(workflow, QUESTION, FORMAL, r.question_text)
            for o in r.options:
                add(workflow, OPTION, FORMAL, o.text)
    return list(seen.values())


def write_units(path: str | Path, units: Sequence[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for unit in units:
            f.write(json.dumps(unit, ensure_ascii=False) + "\n")


def read_units(path: str | Path) -> list[dict[str, object]]:
    units = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                unit = json.loads(line)
            except json.JSONDecodeError as e:
                raise ExchangeError(f"{path}, line {n}: not valid JSON: {e}") from e
            for field in ("id", "workflow", "text", "fa"):
                if not isinstance(unit.get(field), str):
                    raise ExchangeError(f"{path}, line {n}: field {field!r} must be text")
            units.append(unit)
    return units


def check_filled(original: Sequence[dict], filled: Sequence[dict]) -> dict[str, list[str]]:
    """Problems of a filled file against the exported one: ids or English text changed,
    units missing or left empty."""
    problems: dict[str, list[str]] = {"changed": [], "missing": [], "empty": []}
    by_id = {u["id"]: u for u in filled}
    for unit in original:
        got = by_id.get(unit["id"])
        if got is None:
            problems["missing"].append(unit["id"])
        elif got["text"] != unit["text"] or got["workflow"] != unit["workflow"]:
            problems["changed"].append(unit["id"])
        elif not got["fa"].strip():
            problems["empty"].append(unit["id"])
    return problems


class PrecomputedTranslator:
    """Looks translations up in a filled units file; unknown or empty texts give ""."""

    def __init__(self, units: Iterable[dict], name: str = "precomputed") -> None:
        self.name = name
        self._by_text = {(u["workflow"], u["text"]): u["fa"] for u in units}

    def translate(self, items: Sequence[Item]) -> list[str]:
        return [self._by_text.get((i.workflow, i.text), "") for i in items]


def instructions(n_units: int) -> str:
    """The brief for whoever translates the units file (a person or an assistant)."""
    registers = "\n".join(f"- {name}: {text}" for name, text in REGISTERS.items())
    rules = "\n".join(f"- {rule}" for rule in RULES)
    return f"""# Translating the units file into Persian

{SYSTEM} The file `units.jsonl` has {n_units} lines; each is a JSON object with an English
`text` to translate. Fill the empty `fa` field of every line with the Persian translation and
give the file back with every other field unchanged (`id`, `workflow`, `kind`, `register`,
`text`, ...), one JSON object per line, same order.

## What each field means
- `text`: the English to translate. `kind` says where it comes from (state, question or option).
- `register`: `colloquial` or `formal`.
{registers}
- `context`: one sentence on where the text comes from. Use it to read ambiguous words the
  right way (for example, in agent traces an *agent* is an AI agent, never a person).
- `terms`: English -> Persian terms that occur in this text. Use exactly these Persian terms,
  in every unit, so the same thing is always named the same way.
- `keep_in_english`: technical terms that must stay in English, exactly as written.

## Rules
{rules}
- Translate meaning faithfully: keep negation ("never", "not"), urgency, who did what,
  and quantities. A word that flips meaning (for example "irreversible" as "reversible") makes
  the unit wrong even if the Persian reads well.
- Options of a question start with a label before a colon ("Low: ...", "High: ..."); keep the
  label, translate it, and make the labels of one question different from each other.
- Do not leave English words in the Persian text, except code, identifiers, e-mail addresses,
  URLs and the `keep_in_english` terms.
- Do not add notes or explanations. If a text is unclear, translate it literally and continue.

Work in chunks of about 50 lines if the file is long, and keep the order.
"""
