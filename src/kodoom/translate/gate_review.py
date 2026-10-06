"""Label-focused human review resources; automatic findings never approve meaning."""

from __future__ import annotations

import csv
import json
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from kodoom.artifacts import checksum, export_bundle, restore_bundle, verify_bundle
from kodoom.schema import read_jsonl


def label_sheet(root: Path, recipe: dict, selected: list, translator: str) -> None:
    target = root / f"review-labels-{recipe['dataset']}.csv"
    if target.exists():
        return  # A compatible resume preserves human annotations.
    translated = {
        r.id: r for r in read_jsonl(root / recipe["output"].format(translator=translator))
    }
    rows = []
    for en in selected:
        fa = translated[en.id + ":fa"]
        for option, translated_option in zip(en.options, fa.options, strict=True):
            rows.append(
                {
                    "record_id": en.id,
                    "source_id": en.source_id,
                    "workflow": en.extra.get("workflow", "helmo"),
                    "question_type": en.question_type,
                    "english_state": en.state,
                    "persian_state": fa.state,
                    "english_question": en.question_text,
                    "persian_question": fa.question_text,
                    "option_id": option.id,
                    "english_label": option.text,
                    "persian_label": translated_option.text,
                    "meaning_changed": "",
                    "needs_edit": "",
                    "error_category": "",
                    "suggested_persian": "",
                    "notes": "",
                    "reviewer": "",
                }
            )
    if not rows:
        raise ValueError("gate has no option descriptions to review")
    fields = list(rows[0])
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    grouped = Counter(
        (
            r["workflow"],
            r["question_type"],
            r["english_question"],
            r["option_id"],
            r["english_label"],
            r["persian_label"],
        )
        for r in rows
    )
    with (root / f"label-templates-{recipe['dataset']}.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "workflow",
                "question_type",
                "english_question",
                "option_id",
                "english_label",
                "persian_label",
                "occurrences",
            ]
        )
        writer.writerows([*key, count] for key, count in grouped.items())


def review_summary(root: str | Path) -> dict:
    """Counts remain descriptive: this balanced diagnostic gate is not a random sample."""
    result = {"sampling": "balanced diagnostic sample; dataset error rate/coverage unknown"}
    for path in sorted(Path(root).glob("review-*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        reviewed = [
            r
            for r in rows
            if r.get("meaning_changed", "").lower() in ("yes", "no")
            and r.get("needs_edit", "").lower() in ("yes", "no")
        ]
        result[path.name] = {
            "total": len(rows),
            "reviewed": len(reviewed),
            "meaning_changes": sum(r["meaning_changed"].lower() == "yes" for r in reviewed),
            "needs_edit": sum(r["needs_edit"].lower() == "yes" for r in reviewed),
            "complete": len(rows) > 0 and len(reviewed) == len(rows),
        }
    return result


def compare_gate(gate: dict, baseline_bundle: str | Path) -> dict:
    """Write paired case/label CSVs only after verifying identical selected English inputs."""
    from kodoom.workflow import load_request, selected_records, validate_output

    root = Path(gate["artifact_root"])
    if gate["status"] != "awaiting-review" or gate["semantic"]["smoke"]:
        raise ValueError("comparison requires a completed real gate awaiting human review")
    baseline_bundle = Path(baseline_bundle)
    verify_bundle(baseline_bundle)
    identity = {
        "baseline_bundle_sha256": checksum(baseline_bundle),
        "candidate_outputs": gate["outputs"],
    }
    provenance = root / "paired-provenance.json"
    if provenance.exists():
        if json.loads(provenance.read_text("utf-8"))["identity"] != identity:
            raise ValueError(
                "paired review already exists for different inputs; preserve annotations"
            )
        return gate
    _, recipes = load_request()
    request = gate["request"]
    if request["translator"] != "gemma3-27b-tpu-bf16":
        raise ValueError("candidate must be the Gemma 27B TPU experiment")
    with tempfile.TemporaryDirectory(dir=root.parent) as scratch:
        baseline_root = Path(scratch) / "baseline"
        restore_bundle(baseline_bundle, baseline_root)
        baseline = json.loads((baseline_root / "execution.json").read_text("utf-8"))
        if (
            baseline.get("mode") != "gate"
            or baseline.get("status") != "awaiting-review"
            or baseline.get("model_revision") == "stub"
            or baseline.get("request", {}).get("translator") != "gemma3-4b-bf16"
        ):
            raise ValueError("attach the completed real Gemma 4B gate bundle for comparison")
        paired_cases, paired_labels = [], []
        for recipe, limit in zip(
            recipes, [request["helmo_limit"], request["typed_limit"]], strict=True
        ):
            current_en = selected_records(recipe, root, limit)
            baseline_en = selected_records(recipe, baseline_root, limit)
            if [asdict(r) for r in current_en] != [asdict(r) for r in baseline_en]:
                raise ValueError(
                    "4B and 27B selected English records differ; paired comparison refused"
                )
            validate_output(
                baseline_en,
                baseline_root / recipe["output"].format(translator="gemma3-4b-bf16"),
                "gemma3-4b-bf16",
            )
            candidate = root / recipe["output"].format(translator=request["translator"])
            if checksum(candidate) != gate["outputs"][recipe["dataset"]]["sha256"]:
                raise ValueError("candidate translation checksum changed after the gate")
            old = {
                r.id: r
                for r in read_jsonl(
                    baseline_root / recipe["output"].format(translator="gemma3-4b-bf16")
                )
            }
            new = {
                r.id: r
                for r in read_jsonl(
                    root / recipe["output"].format(translator=request["translator"])
                )
            }
            for en in current_en:
                before, after = old[en.id + ":fa"], new[en.id + ":fa"]
                context = {
                    "dataset": recipe["dataset"],
                    "source_id": en.source_id,
                    "record_id": en.id,
                    "workflow": en.extra.get("workflow", "helmo"),
                }
                paired_cases.append(
                    {
                        **context,
                        "english": json.dumps(asdict(en), ensure_ascii=False),
                        "gemma4b": json.dumps(asdict(before), ensure_ascii=False),
                        "gemma27b": json.dumps(asdict(after), ensure_ascii=False),
                        "baseline_meaning_changed": "",
                        "candidate_meaning_changed": "",
                        "assessment": "",
                        "notes": "",
                        "reviewer": "",
                    }
                )
                for en_option, old_option, new_option in zip(
                    en.options, before.options, after.options, strict=True
                ):
                    paired_labels.append(
                        {
                            **context,
                            "english_question": en.question_text,
                            "option_id": en_option.id,
                            "english_label": en_option.text,
                            "gemma4b_label": old_option.text,
                            "gemma27b_label": new_option.text,
                            "text_changed": old_option.text != new_option.text,
                            "baseline_meaning_changed": "",
                            "candidate_meaning_changed": "",
                            "assessment": "",
                            "suggested_persian": "",
                            "notes": "",
                            "reviewer": "",
                        }
                    )
        for name, rows in (("decisions", paired_cases), ("labels", paired_labels)):
            with (root / f"paired-{name}.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        provenance.write_text(
            json.dumps(
                {
                    "identity": identity,
                    "baseline": {
                        k: baseline.get(k) for k in ("git_commit", "model_revision", "semantic")
                    },
                    "note": (
                        "Identical selected train inputs. Text changes are not quality "
                        "judgments; human review required."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    gate["comparison"] = identity
    (root / "execution.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    gate["bundle"] = str(export_bundle(root, Path(gate["bundle"]).parent))
    return gate
