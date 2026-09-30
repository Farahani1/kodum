"""The pilot's blind review sheet and its scoring (plan 1.2, step 1).

Two translators translate the same cases; a reviewer reads each case once, with the
two translations in random order under the labels A and B, and records which is better
and how many meaning errors each has. The key that says which translator was A or B
is kept in a separate file and only used to score the filled sheet, so the review stays
blind. A case is judged as a whole (state texts, questions and options), which keeps 50
cases within a few hours.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from kodoom.schema import Record
from kodoom.translate.rules import KEEP, segments

COLUMNS = (
    "case_id",
    "workflow",
    "english",
    "translation_A",
    "translation_B",
    "better",
    "meaning_errors_A",
    "meaning_errors_B",
    "notes",
)
TIE = {"tie", "=", "0", "same"}


class PilotError(ValueError):
    """The sheet or the key cannot be used."""


def case_text(records: Sequence[Record], english_state: str) -> str:
    """One case as plain text: its translated state texts, then questions and options.

    ``english_state`` decides which state leaves are shown (the ones the rules translate),
    so the English and the Persian block list the same things in the same order.
    """
    first = records[0]
    tree = json.loads(first.state)
    lines = []
    for seg in segments(records[0].extra["workflow"], english_state):
        if seg.rule.action == KEEP:
            continue
        node = tree
        for step in seg.location:
            node = node[step]
        lines.append(f"[{seg.path}] {node}")
    for r in records:
        lines.append(f"[{r.extra['question']}] ({r.question_type}) {r.question_text}")
        lines += [f"    {o.id}: {o.text}" for o in r.options]
    return "\n".join(lines)


def build_sheet(
    english: Sequence[list[Record]],
    candidates: Mapping[str, Mapping[str, list[Record]]],
    seed: int = 1234,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    """Rows of the sheet and the key. ``candidates``: translator name -> source_id -> records.

    Only cases that both translators have are included. Which translator is A is drawn per
    case from ``seed``, so the same inputs give the same sheet.
    """
    if len(candidates) != 2:
        raise PilotError("a pilot compares exactly two translators")
    (name_1, by_id_1), (name_2, by_id_2) = candidates.items()
    rng = random.Random(seed)
    rows, key = [], {}
    for case in english:
        case_id = case[0].source_id
        if case_id not in by_id_1 or case_id not in by_id_2:
            continue
        a, b = (name_1, name_2) if rng.random() < 0.5 else (name_2, name_1)
        texts = {name_1: by_id_1[case_id], name_2: by_id_2[case_id]}
        rows.append(
            {
                "case_id": case_id,
                "workflow": case[0].extra["workflow"],
                "english": case_text(case, case[0].state),
                "translation_A": case_text(texts[a], case[0].state),
                "translation_B": case_text(texts[b], case[0].state),
                "better": "",
                "meaning_errors_A": "",
                "meaning_errors_B": "",
                "notes": "",
            }
        )
        key[case_id] = {"A": a, "B": b}
    if not rows:
        raise PilotError("the two translators have no case in common")
    return rows, key


def write_sheet(path: str | Path, rows: Sequence[Mapping[str, str]]) -> None:
    """UTF-8 with a byte-order mark, so Excel and Google Sheets read Persian correctly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_key(path: str | Path, key: Mapping[str, Mapping[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def read_sheet(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise PilotError(f"{path}: columns missing: {sorted(missing)}")
        return list(reader)


def score_sheet(rows: Sequence[Mapping[str, str]], key: Mapping[str, Mapping[str, str]]) -> dict:
    """Wins, ties and meaning errors per translator, and wins per workflow."""
    names = sorted({name for pair in key.values() for name in pair.values()})
    wins = dict.fromkeys(names, 0)
    errors: dict[str, list[int]] = {n: [] for n in names}
    by_workflow: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(names, 0))
    ties = unrated = 0
    for row in rows:
        pair = key.get(row["case_id"])
        if pair is None:
            raise PilotError(f"case {row['case_id']!r} is not in the key")
        verdict = row["better"].strip().lower()
        if verdict in ("a", "b"):
            wins[pair[verdict.upper()]] += 1
            by_workflow[row["workflow"]][pair[verdict.upper()]] += 1
        elif verdict in TIE:
            ties += 1
        else:
            unrated += 1
        for side in ("A", "B"):
            text = row[f"meaning_errors_{side}"].strip()
            if text:
                try:
                    errors[pair[side]].append(int(text))
                except ValueError:
                    raise PilotError(
                        f"case {row['case_id']}: meaning_errors_{side} must be a number, "
                        f"got {text!r}"
                    ) from None
    return {
        "cases": len(rows),
        "wins": wins,
        "ties": ties,
        "unrated": unrated,
        "mean_meaning_errors": {n: (sum(v) / len(v) if v else None) for n, v in errors.items()},
        "wins_by_workflow": {w: dict(c) for w, c in sorted(by_workflow.items())},
    }
