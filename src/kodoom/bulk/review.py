"""Immutable case/option review exports for each completed work unit."""

import csv
import io
import json

from kodoom.bulk.state import atomic_write


def export_unit(campaign, unit, translated):
    cases, labels = [], []
    for en, fa in zip(unit.records, translated, strict=True):
        base = {
            "record_id": en.id,
            "source_id": en.source_id,
            "dataset": unit.dataset,
            "split": en.split,
            "workflow": en.extra.get("workflow", "helmo"),
            "question_type": en.question_type,
            "english_state": en.state,
            "persian_state": fa.state,
            "english_question": en.question_text,
            "persian_question": fa.question_text,
        }
        review = dict.fromkeys(
            [
                "meaning_changed",
                "needs_edit",
                "error_category",
                "suggested_persian",
                "notes",
                "reviewer",
            ],
            "",
        )
        cases.append(
            {
                **base,
                "english_options": json.dumps([o.__dict__ for o in en.options], ensure_ascii=False),
                "persian_options": json.dumps([o.__dict__ for o in fa.options], ensure_ascii=False),
                "automatic_findings": json.dumps(
                    fa.extra.get("check_findings", []), ensure_ascii=False
                ),
                **review,
            }
        )
        for left, right in zip(en.options, fa.options, strict=True):
            labels.append(
                {
                    **base,
                    "option_id": left.id,
                    "english_label": left.text,
                    "persian_label": right.text,
                    **review,
                }
            )
    with campaign.lock:
        for kind, rows in [("cases", cases), ("labels", labels)]:
            path = campaign.root / "review" / f"{kind}-{unit.key}.csv"
            if path.exists():
                continue
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            atomic_write(path, stream.getvalue().encode("utf-8-sig"))
