import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kodoom.package_import import (
    clean_translation,
    file_sha,
    import_package,
    row_records,
    verify_inventory,
)
from kodoom.schema import RecordError, read_jsonl
from kodoom.transport import read_parquet
from kodoom.typed_decisions import case_records
from tests.test_typed_decisions import make_row


def bilingual_row():
    original = case_records(
        make_row(state='{"thread": [{"text": "Refund 7 please", "role": "user"}], "n": 7}')
    )[0]
    return {
        "id": "package-row",
        "case_id": "package-case",
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
        "state_fa": original.state,
        "question_en": original.question_text,
        "question_fa": "پرسش ۱۲٫۵ چيست؟",
        "gold_original_json": json.dumps(original.gold),
        "source_metadata_json": json.dumps(original.extra),
        "original_record_json": json.dumps(original.to_dict()),
        "state_format": "json",
        "options": [
            {
                "id": o.id,
                "text_en": o.text,
                "text_fa": "پاسخ فارسی",
                "gold_probability": original.gold[o.id],
            }
            for o in original.options
        ],
        "translation": {"status": "complete", "model_id": None, "model_reported": "supplied"},
        "quality": {"human_reviewed": False, "annotation_ids": ["unresolved-finding"]},
    }


def test_orthographic_cleanup_preserves_embedded_code_and_url_bytes():
    code = "```python\nif x:\n    print('ي  ك')\n```"
    target = "مي شود " + code + " و `a  b` https://example.com/ي"
    cleaned = clean_translation("English " + code, target)
    assert code in cleaned and "`a  b`" in cleaned
    assert "https://example.com/ي" in cleaned
    assert cleaned.startswith("می\u200cشود ")
    assert clean_translation(code, code) == code


def make_package(root, rows=None):
    path = root / "data/source.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows or [bilingual_row()]), path)
    manifest = root / "provenance/release.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "data/source.parquet",
                        "bytes": path.stat().st_size,
                        "sha256": file_sha(path),
                    }
                ],
                "configurations": [
                    {
                        "config": "typed_decisions",
                        "split": "train",
                        "path": "data/source.parquet",
                        "rows": len(rows or [1]),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return file_sha(manifest)


def test_import_preserves_targets_unknown_provenance_and_raw_digits():
    row = bilingual_row()
    en, fa = row_records(row)
    assert en.gold == fa.gold == json.loads(row["gold_original_json"])
    assert en.id == row["question_id"] and fa.id == en.id + ":fa"
    assert fa.origin == "translated" and fa.meaning_flag is None and not fa.human_reviewed
    assert fa.checks_passed is None
    assert fa.question_text == "پرسش ۱۲٫۵ چیست؟"
    assert fa.extra["supplied_package"]["translation"]["model_id"] is None
    assert fa.extra["supplied_package"]["quality"]["annotation_ids"] == ["unresolved-finding"]
    assert en.extra["source_decision_id"] == fa.extra["source_decision_id"] == en.id


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "test"),
        ("gold_original_json", "{}"),
        ("source_id", "wrong"),
        ("source_metadata_json", "{}"),
    ],
)
def test_reject_changed_source_fields(field, value):
    row = bilingual_row()
    row[field] = value
    with pytest.raises(RecordError, match="differs"):
        row_records(row)


def test_reject_reordered_options_and_changed_nontext_state():
    row = bilingual_row()
    row["options"].reverse()
    with pytest.raises(RecordError, match="option order"):
        row_records(row)
    row = bilingual_row()
    row["state_fa"] = row["state_fa"].replace('"n": 7', '"n": 8')
    with pytest.raises(RecordError, match="non-text"):
        row_records(row)


def test_import_verifies_hashes_roundtrips_and_does_not_overwrite(tmp_path):
    root, out = tmp_path / "source", tmp_path / "output"
    pin = make_package(root)
    report = import_package(root, out, pin)
    assert report["status"] == "passed-structural-import"
    assert len(report["tables"]) == 2
    records = list(read_jsonl(out / "jsonl/typed_decisions/fa/train.jsonl"))
    assert records == list(read_parquet(out / "data/typed_decisions/fa/train.parquet"))
    assert not records[0].checks_passed  # deliberate missing number / repeated option text
    with pytest.raises(RecordError, match="already exists"):
        import_package(root, out, pin)
    (root / "data/source.parquet").write_bytes(b"changed")
    with pytest.raises(RecordError, match="inventory mismatch"):
        verify_inventory(root, pin)


def test_reject_unpinned_manifest_duplicate_decisions_and_path_traversal(tmp_path):
    root = tmp_path / "source"
    pin = make_package(root, [bilingual_row(), bilingual_row()])
    with pytest.raises(RecordError, match="pinned"):
        import_package(root, tmp_path / "out", "0" * 64)
    with pytest.raises(RecordError, match="duplicate decision"):
        import_package(root, tmp_path / "out", pin)
    manifest = root / "provenance/release.json"
    data = json.loads(manifest.read_text("utf-8"))
    data["files"][0]["path"] = "../outside"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RecordError, match="escapes"):
        verify_inventory(root, file_sha(manifest))
