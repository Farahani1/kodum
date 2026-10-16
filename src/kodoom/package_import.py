"""Import the supplied bilingual package without translating or relabeling it.

The package's frozen Record is the authority for English IDs and targets.
Its bilingual Parquet layout is an input format, distinct from our versioned
Record transport. Automatic flags never certify semantic or human review.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

from kodoom.normalize import clean_orthography
from kodoom.schema import Option, Record, RecordError, read_jsonl, write_jsonl
from kodoom.sources import check_record
from kodoom.translate.checks import check_option_labels, check_state, check_text
from kodoom.translate.glossary import load as load_glossary
from kodoom.translate.rules import TRANSLATE, segments
from kodoom.transport import read_parquet, write_parquet
from kodoom.typed_decisions import read_parquet_rows

CONFIG_SOURCES = {
    "typed_decisions": "LocalLLaMA/typed-decisions",
    "helmo": "helmo/synthetic-typed-decisions",
}

_PROTECTED_TEXT = re.compile(r"```[\s\S]*?```|`[^`\n]+`|https?://\S+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def clean_translation(source: str, target: str) -> str:
    """Clean prose without reindenting code or rewriting protected spans."""
    if source == target:
        return target
    result, cursor = [], 0
    for match in _PROTECTED_TEXT.finditer(target):
        # Clean each intervening span without stripping spaces next to code.
        prose = target[cursor : match.start()]
        result.append(_clean_span(prose))
        result.append(match.group())
        cursor = match.end()
    result.append(_clean_span(target[cursor:]))
    return "".join(result).strip()


def _clean_span(text: str) -> str:
    # Sentinels preserve span boundaries; cleaner.strip() must not glue words
    # to a neighboring protected token.
    return clean_orthography("X" + text + "X")[1:-1]


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise RecordError(f"package path escapes its root: {relative!r}")
    return path


def verify_inventory(root: Path, expected_sha256: str) -> dict:
    manifest = root / "provenance/release.json"
    if file_sha(manifest) != expected_sha256:
        raise RecordError("package release manifest does not match the pinned SHA-256")
    release = json.loads(manifest.read_text(encoding="utf-8"))
    seen = set()
    for entry in release["files"]:
        path = checked_path(root, entry["path"])
        if entry["path"] in seen:
            raise RecordError(f"duplicate inventory path: {entry['path']}")
        seen.add(entry["path"])
        if path.stat().st_size != entry["bytes"] or file_sha(path) != entry["sha256"]:
            raise RecordError(f"package inventory mismatch: {entry['path']}")
    return release


def _clean_state(source, target):
    """Clean changed text leaves only; untouched code/keep fields stay byte-identical."""
    if isinstance(source, dict) and isinstance(target, dict) and source.keys() == target.keys():
        return {key: _clean_state(source[key], target[key]) for key in source}
    if isinstance(source, list) and isinstance(target, list) and len(source) == len(target):
        return [_clean_state(a, b) for a, b in zip(source, target, strict=True)]
    if isinstance(source, str) and isinstance(target, str):
        return clean_translation(source, target)
    if source != target:
        raise RecordError("translated state changed structure or a non-text value")
    return target


def row_records(row: dict) -> tuple[Record, Record]:
    original = Record.from_dict(json.loads(row["original_record_json"]))
    check_record(original)
    comparisons = {
        "question_id": original.id,
        "source_id": original.source_id,
        "source_repo": original.source,
        "source_revision": original.source_revision,
        "split": original.split,
        "source_split": original.split,
        "license": original.license,
        "question_type": original.question_type,
        "task_family": original.task_family,
        "state_en": original.state,
        "question_en": original.question_text,
    }
    for key, expected in comparisons.items():
        if row[key] != expected:
            raise RecordError(f"{original.id}: bilingual {key} differs from frozen Record")
    if json.loads(row["gold_original_json"]) != original.gold:
        raise RecordError(f"{original.id}: gold differs from frozen Record")
    if json.loads(row["source_metadata_json"]) != original.extra:
        raise RecordError(f"{original.id}: source metadata differs from frozen Record")
    options = row["options"]
    if [(o["id"], o["text_en"]) for o in options] != [(o.id, o.text) for o in original.options] or {
        o["id"]: o["gold_probability"] for o in options
    } != original.gold:
        raise RecordError(f"{original.id}: option order, text or probabilities changed")
    if row["translation"]["status"] != "complete" or row["quality"]["human_reviewed"]:
        raise RecordError(f"{original.id}: unsupported translation/review status")

    common = {**original.extra, "source_decision_id": original.id}
    english = replace(original, extra=common)
    raw_state = row["state_fa"]
    if row["state_format"] == "json":
        cleaned = _clean_state(json.loads(original.state), json.loads(raw_state))
        state = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
    elif row["state_format"] == "text":
        state = _clean_state(original.state, raw_state)
    else:
        raise RecordError(f"unsupported state format: {row['state_format']}")
    persian = replace(
        original,
        id=original.id + ":fa",
        origin="translated",
        state_lang="fa",
        question_lang="fa",
        state=state,
        question_text=clean_translation(original.question_text, row["question_fa"]),
        options=tuple(
            Option(o["id"], clean_translation(o["text_en"], o["text_fa"])) for o in options
        ),
        checks_passed=None,
        meaning_flag=None,
        human_reviewed=False,
        extra={
            **common,
            "supplied_package": {
                "row_id": row["id"],
                "case_id": row["case_id"],
                "translation": row["translation"],
                "quality": row["quality"],
            },
            "human_review_waiver": {"date": "2026-10-15", "scope": "supplied corpus"},
            "training_eligibility": "unresolved-source-teacher-translator-license-gate",
        },
    )
    return english, persian


def automatic_findings(english: Record, persian: Record) -> list[dict]:
    glossary = load_glossary()
    workflow = english.extra.get("workflow", "")
    if workflow:
        findings = check_state(workflow, english.state, persian.state, glossary.keep)
        translated = {s.path: s.text for s in segments(workflow, persian.state)}
        for seg in segments(workflow, english.state):
            if seg.rule.action == TRANSLATE:
                findings += glossary.check(workflow, seg.text, translated[seg.path], seg.path)
    else:
        findings = check_text(english.state, persian.state, "state", glossary.keep)
        findings += glossary.check("", english.state, persian.state, "state")
    texts = [("question", english.question_text, persian.question_text)] + [
        (f"option.{a.id}", a.text, b.text)
        for a, b in zip(english.options, persian.options, strict=True)
    ]
    for where, source, target in texts:
        findings += check_text(source, target, where, glossary.keep)
        findings += glossary.check(workflow, source, target, where)
    findings += check_option_labels(
        [o.text for o in english.options], [o.text for o in persian.options], "options"
    )
    return [asdict(f) for f in findings]


def import_package(root: Path, out: Path, expected_sha256: str) -> dict:
    """Verify frozen inputs, export both languages, and check every round trip.

    Output must be new: a failed build is never confused with a previous release.
    The caller updates its data_dir README after the command, including failures.
    """
    if out.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(out.resolve()):
        raise RecordError("input and output package directories must be disjoint")
    if out.exists():
        raise RecordError(f"output already exists; choose a new directory: {out}")
    release = verify_inventory(root, expected_sha256)
    grouped = {}
    seen = set()
    memberships = {}
    report = {
        "input_manifest_sha256": expected_sha256,
        "inventory_files": len(release["files"]),
        "tables": [],
        "findings_by_check": {},
        "semantic_check": "not_run",
        "human_review": "waived-2026-10-15",
        "training_eligibility": "unresolved",
    }
    inventory = {entry["path"] for entry in release["files"]}
    for table in release["configurations"]:
        config, split = table["config"], table["split"]
        if config not in CONFIG_SOURCES:
            continue
        if table["path"] not in inventory:
            raise RecordError("bilingual table is not in the verified inventory")
        count = 0
        for row in read_parquet_rows(checked_path(root, table["path"])):
            en, fa = row_records(row)
            if en.source != CONFIG_SOURCES[config] or en.split != split:
                raise RecordError("table configuration/split differs from row")
            key = (en.source, en.source_revision, en.id)
            if key in seen:
                raise RecordError(f"duplicate decision: {key}")
            seen.add(key)
            group = (en.source, en.source_revision, en.source_id)
            if memberships.setdefault(group, split) != split:
                raise RecordError(f"case crosses splits: {group}")
            findings = automatic_findings(en, fa)
            for f in findings:
                counts = report["findings_by_check"]
                counts[f["check"]] = counts.get(f["check"], 0) + 1
            fa = replace(
                fa, checks_passed=not findings, extra={**fa.extra, "automatic_findings": findings}
            )
            check_record(fa)
            for lang, record in (("en", en), ("fa", fa)):
                grouped.setdefault((config, lang, split), []).append(record)
            count += 1
        if count != table["rows"]:
            raise RecordError(f"{config}/{split}: row count differs from manifest")
    out.mkdir(parents=True)
    for (config, lang, split), records in grouped.items():
        stem = Path(config) / lang / split
        jsonl = out / "jsonl" / stem.with_suffix(".jsonl")
        parquet = out / "data" / stem.with_suffix(".parquet")
        write_jsonl(jsonl, records)
        write_parquet(parquet, records)
        if list(read_parquet(parquet)) != records or list(read_jsonl(jsonl)) != records:
            raise RecordError(f"Parquet round trip differs: {stem}")
        report["tables"].append(
            {
                "config": config,
                "language": lang,
                "split": split,
                "decisions": len(records),
                "cases": len({r.source_id for r in records}),
            }
        )
    report["status"] = "passed-structural-import"
    report["files"] = [
        {"path": path.relative_to(out).as_posix(), "sha256": file_sha(path)}
        for path in sorted(out.rglob("*"))
        if path.is_file()
    ]
    (out / "import-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
