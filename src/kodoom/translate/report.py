"""Look at translations: counts of what the checks found, and English next to Persian.

Used after a trial or a pilot to see, without opening files, how many cases passed and
what the translator gets wrong. Metrics come from the records, so a rerun is enough
to refresh them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from kodoom.schema import Record
from kodoom.translate.pipeline import cases
from kodoom.translate.rules import KEEP, segments


def summarize(records: Sequence[Record]) -> dict:
    """Cases, cases with findings and findings by check, over Persian records."""
    grouped = cases(records)
    by_check: Counter[str] = Counter()
    failed = 0
    for case in grouped:
        found = {(f["check"], f["where"]) for r in case for f in r.extra.get("check_findings", [])}
        failed += bool(found)
        by_check.update(check for check, _ in found)
    return {
        "cases": len(grouped),
        "decisions": len(records),
        "cases_with_findings": failed,
        "findings_by_check": dict(sorted(by_check.items())),
    }


def side_by_side(english: Sequence[Record], persian: Sequence[Record]) -> str:
    """One case: every translated text of the state, then the questions and options."""
    en, fa = english[0], persian[0]
    lines = [f"== {en.source_id}  ({en.extra['workflow']})"]
    for seg in segments(en.extra["workflow"], en.state):
        if seg.rule.action == KEEP:
            continue
        translated = _leaf(fa.state, seg.location)
        lines += [f"[{seg.path}]", f"  EN: {seg.text}", f"  FA: {translated}"]
    for e, p in zip(english, persian, strict=True):
        lines += [f"[{e.extra['question']}] ({e.question_type})", f"  EN: {e.question_text}"]
        lines.append(f"  FA: {p.question_text}")
        for eo, po in zip(e.options, p.options, strict=True):
            lines.append(f"    {eo.id}: {eo.text}  ->  {po.text}")
    findings = sorted(
        {
            (f["check"], f["where"], f["message"])
            for r in persian
            for f in r.extra.get("check_findings", [])
        }
    )
    lines += [f"  ! {check} at {where}: {message}" for check, where, message in findings]
    return "\n".join(lines)


def _leaf(state: str, location):
    import json

    node = json.loads(state)
    for step in location:
        node = node[step]
    return node
