"""Stage separate bilingual resources and unresolved quality work, privately.

This exporter does not grant translation rights or authorize publication.
Frozen input rows are retained so orthographic edits remain reconstructable.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from kodoom.package_import import CONFIG_SOURCES, checked_path, file_sha, verify_inventory
from kodoom.runs import git_commit
from kodoom.schema import RecordError, read_jsonl, write_jsonl
from kodoom.skill_release import SCORING_EXAMPLE, write_checksums
from kodoom.transport import VERSION, read_parquet, write_parquet


def dump_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_resources(source: Path, imported: Path, out: Path) -> dict:
    """Verify the import's two formats, then make standalone private resources."""
    if out.exists():
        raise RecordError(f"output already exists; choose a new directory: {out}")
    for parent in (source, imported):
        if out.resolve().is_relative_to(parent.resolve()) or parent.resolve().is_relative_to(
            out.resolve()
        ):
            raise RecordError("release input and output directories must be disjoint")
    report = json.loads((imported / "import-report.json").read_text(encoding="utf-8"))
    if report["status"] != "passed-structural-import":
        raise RecordError("a successful structural import is required")
    release = verify_inventory(source, report["input_manifest_sha256"])
    if not report.get("files"):
        raise RecordError("import report lacks output checksums; rerun the import")
    for entry in report["files"]:
        if file_sha(checked_path(imported, entry["path"])) != entry["sha256"]:
            raise RecordError(f"import checksum mismatch: {entry['path']}")
    result = {
        "status": "private-candidate",
        "resources": {},
        "publication_authorized": False,
        "training_eligibility": "unresolved",
    }
    for config, repo in CONFIG_SOURCES.items():
        root = out / config
        tables = [t for t in report["tables"] if t["config"] == config]
        if not tables:
            continue
        pairs = defaultdict(dict)
        flagged = 0
        findings = Counter()
        all_flags = []
        counts = []
        for table in tables:
            lang, split = table["language"], table["split"]
            stem = Path(config) / lang / split
            records = list(read_jsonl(imported / "jsonl" / stem.with_suffix(".jsonl")))
            if records != list(read_parquet(imported / "data" / stem.with_suffix(".parquet"))):
                raise RecordError(f"import format mismatch: {stem}")
            if (
                len(records) != table["decisions"]
                or len({r.source_id for r in records}) != table["cases"]
            ):
                raise RecordError(f"import count mismatch: {stem}")
            for r in records:
                if r.source != repo or r.question_lang != lang or r.split != split:
                    raise RecordError(f"import table identity mismatch: {r.id}")
                key = (r.source_revision, r.extra["source_decision_id"])
                if lang in pairs[key]:
                    raise RecordError(f"duplicate language key: {key}")
                pairs[key][lang] = r
                if lang == "fa" and not r.checks_passed:
                    flagged += 1
                    all_flags.append(
                        {
                            "id": r.id,
                            "source_id": r.source_id,
                            "split": r.split,
                            "findings": r.extra["automatic_findings"],
                        }
                    )
                    findings.update(f["check"] for f in r.extra["automatic_findings"])
            write_jsonl(root / f"jsonl/{lang}/{split}.jsonl", records)
            write_parquet(root / f"data/{lang}/{split}.parquet", records)
            counts.append(table)
        requests = defaultdict(lambda: {"questions": []})
        for key, views in pairs.items():
            if set(views) != {"en", "fa"}:
                raise RecordError(f"unpaired bilingual key: {key}")
            en, fa = views["en"], views["fa"]
            if (
                en.gold != fa.gold
                or en.source_id != fa.source_id
                or en.split != fa.split
                or [o.id for o in en.options] != [o.id for o in fa.options]
            ):
                raise RecordError(f"bilingual gold, source ID, split or option mismatch: {key}")
            case = requests[(en.source_revision, en.source_id)]
            case.update(
                {
                    "source": repo,
                    "source_revision": en.source_revision,
                    "source_id": en.source_id,
                    "split": en.split,
                    "state_en": en.state,
                    "state_fa": fa.state,
                }
            )
            case["questions"].append(
                {
                    "id": en.id,
                    "type": en.question_type,
                    "question_en": en.question_text,
                    "question_fa": fa.question_text,
                    "options_en": [vars(o) for o in en.options],
                    "options_fa": [vars(o) for o in fa.options],
                }
            )
        review = root / "review"
        review.mkdir(parents=True)
        # Empty manifests explicitly mean no semantic edits or exclusions have
        # been applied; proposed source holds remain in the supplied evidence.
        for name in ("corrections.jsonl", "exclusions.jsonl"):
            (review / name).write_text("", encoding="utf-8")
        for filename, values in (
            ("automatic-findings.jsonl", all_flags),
            ("meaning-check-requests.jsonl", list(requests.values())),
        ):
            with (review / filename).open("w", encoding="utf-8", newline="\n") as stream:
                for value in values:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        # Copy audit evidence without promoting suggestions into applied corrections.
        for entry in release["files"]:
            if Path(entry["path"]).parts[0] in {"quality", "provenance"}:
                dest = root / "provenance/supplied" / entry["path"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checked_path(source, entry["path"]), dest)
        dest = root / "provenance/supplied/provenance/release.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "provenance/release.json", dest)
        for entry in release["configurations"]:
            if entry["config"] == config:
                dest = root / "provenance/inputs" / Path(entry["path"]).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checked_path(source, entry["path"]), dest)
        dump_json(
            review / "status.json",
            {
                "human_review": "waived-by-owner-2026-10-15",
                "human_translation_error_rate": None,
                "meaning_check": "not_run",
                "checker_requests": len(requests),
                "automatic_flagged_decisions": flagged,
                "findings_by_check": dict(findings),
                "source_label_holds": "see provenance/supplied/quality; proposals remain unapplied",
                "applied_semantic_corrections": [],
                "canonical_gold_modified": False,
            },
        )
        dump_json(
            root / "manifest.json",
            {
                "resource": config,
                "status": "private-candidate",
                "code_revision": git_commit(),
                "record_transport_version": VERSION,
                "tables": counts,
                "input_manifest_sha256": report["input_manifest_sha256"],
                "import_report_sha256": file_sha(imported / "import-report.json"),
                "meaning_check": "not_run",
                "translation_license_grant": "not_supplied",
                "training_eligibility": "unresolved",
                "human_review": "waived-2026-10-15",
            },
        )
        for name in ("LICENSE", "NOTICE"):
            if name not in {entry["path"] for entry in release["files"]}:
                raise RecordError(f"source inventory lacks {name}")
            shutil.copyfile(source / name, root / name)
        metadata = ""
        for lang in ("en", "fa"):
            metadata += f"- config_name: {lang}\n  data_files:\n"
            for split in sorted({t["split"] for t in tables if t["language"] == lang}):
                metadata += f"  - split: {split}\n    path: data/{lang}/{split}.parquet\n"
        (root / "README.md").write_text(
            "---\nlanguage:\n- en\n- fa\nlicense: other\n"
            "license_name: component-licenses-and-translation-terms\nlicense_link: LICENSE\n"
            f"configs:\n{metadata}---\n\n# {config}: private release candidate\n\n"
            f"{len(pairs):,} paired decisions from `{repo}`. Canonical source targets,\n"
            "option IDs/order, case IDs and splits are unchanged. Helmo is train-only.\n"
            "Each English ID has a Persian `:fa` counterpart; pair by source, revision\n"
            "and `extra.source_decision_id`. Persian text receives orthographic cleanup\n"
            "only; digits remain raw. English frozen records remain the reference.\n\n"
            "Machine translations have the per-chunk provenance retained in\n"
            "`extra.supplied_package.translation`. Display names are reported names;\n"
            "unknown checkpoint IDs, prompts and settings remain unknown. Source gold\n"
            "is synthetic teacher output, not independently established truth.\n\n"
            "The owner waived human translation review on 2026-10-15. No human\n"
            "translation-error rate is available. The supplied 100-case AI audit\n"
            "sampled 50 cases per source; its evidence and proposed source-label holds\n"
            "are retained in `provenance/supplied/quality`. Proposals are unapplied.\n"
            "Automatic flags are screening findings, not confirmed semantic errors.\n"
            "Full semantic checking has not run; see `review/status.json`.\n\n"
            "A translation-specific license grant has not been supplied. LICENSE\n"
            "and NOTICE retain actual component terms and source attribution.\n"
            "Publication and downstream permissive-model training remain blocked\n"
            "pending their separate decisions. No upload is performed by this build.\n\n"
            "JSONL follows Record.to_dict(). Parquet stores gold as ordered\n"
            "{id, probability} entries and metadata as extra_json; load it with\n"
            "`kodoom.transport.read_parquet`. Original targets remain distributions.\n"
            "See `reproduce.md` for pinned inputs and the rebuild command.\n",
            encoding="utf-8",
        )
        (root / "reproduce.md").write_text(
            "# Reproduce this private candidate\n\n"
            "Use the immutable kodoom revision in manifest.json, install `.[dev]`,\n"
            "and obtain the supplied package matching input_manifest_sha256.\n"
            "Its complete inventory and source revisions are retained under\n"
            "provenance/supplied/provenance/release.json. Resource-specific bilingual\n"
            "input rows are archived under provenance/inputs. They preserve the raw\n"
            "translations, original Records, quality annotations and provenance.\n\n"
            "Run `kodoom import-package --profile dev SOURCE --manifest-sha256 PIN\n"
            "--out NEW_IMPORT`, then `kodoom release-package --profile dev\n"
            "--source SOURCE --imported NEW_IMPORT --out NEW_RELEASE`.\n"
            "This replays deterministic orthographic cleanup and automatic checks;\n"
            "it does not rerun translation or apply AI suggestions. JSONL content\n"
            "must match. Parquet bytes may vary with the writer version; verify\n"
            "Record equality rather than promising identical Parquet bytes.\n",
            encoding="utf-8",
        )
        (root / "schema.md").write_text(
            "# Record transport version 1\n\n"
            "JSONL follows kodoom.schema.Record.to_dict(). State is a string,\n"
            "including serialized JSON where structured. Options are ordered\n"
            "{id, text} objects. Gold is the original option-ID/probability map,\n"
            "including sparse maps; targets are never reduced to winning labels.\n"
            "Parquet retains scalar fields, stores options as an ordered list,\n"
            "gold as ordered {id, probability} entries, and extra as extra_json.\n"
            "Load with kodoom.transport.read_parquet to restore Record objects.\n\n"
            "checks_passed: null means unperformed; false means automatic findings\n"
            "exist; true means those automatic checks found nothing. meaning_flag:\n"
            "null means no full meaning check; true/false must only describe an\n"
            "actually performed check. human_reviewed stays false under the waiver.\n"
            "Supplied structural and AI-audit status remain separately recorded in\n"
            "extra.supplied_package. None of these statuses establish source-label\n"
            "correctness, translation rights or model-training eligibility.\n",
            encoding="utf-8",
        )
        if any(t["split"] == "test" for t in tables):
            example = root / "examples/load_and_score.py"
            example.parent.mkdir()
            example.write_text(SCORING_EXAMPLE, encoding="utf-8")
        write_checksums(root)
        result["resources"][config] = {
            "paired_decisions": len(pairs),
            "cases": len(requests),
            "automatic_flagged_decisions": flagged,
        }
    dump_json(out / "release-report.json", result)
    return result
