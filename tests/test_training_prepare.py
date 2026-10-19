import json
from dataclasses import replace

import pytest

import kodoom.training.data as data
from kodoom.schema import RecordError
from tests.test_training_data import rows


def test_prepare_freezes_grouped_splits_and_context_exclusion(tmp_path, monkeypatch):
    records = rows(12)
    records = [replace(r, split="test" if i // 2 >= 10 else "train") for i, r in enumerate(records)]
    lock = {"source_root": "deployment-path", "fingerprint": "fixture"}
    monkeypatch.setattr(data, "consume_release", lambda *_: (records, lock))
    monkeypatch.setattr(data, "read_selection", lambda *_: {})
    out = tmp_path / "prepared"
    manifest = data.prepare(tmp_path, tmp_path, out)
    loaded, splits = data.load_prepared(out)
    assert loaded == manifest
    membership = {}
    for split, rs in splits.items():
        for r in rs:
            assert membership.setdefault(r.source_id, split) == split
            assert r.gold == records[0].gold
    assert len({r.source_id for r in splits["test"]}) == 2
    assert len({r.source_id for r in splits["validation"]}) == 1
    assert len({r.source_id for r in splits["calibration"]}) == 1
    assert len({r.source_id for r in splits["train"]}) == 8
    (out / "train.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RecordError, match="checksum"):
        data.load_prepared(out)
    with pytest.raises(RecordError, match="output exists"):
        data.prepare(tmp_path, tmp_path, out)


def test_reject_missing_language_and_unknown_context(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "read_selection", lambda *_: {})
    records = [r for r in rows(12) if r.question_lang == "en"]
    monkeypatch.setattr(data, "consume_release", lambda *_: (records, {}))
    with pytest.raises(RecordError, match="both selected languages"):
        data.prepare(tmp_path, tmp_path, tmp_path / "out")


def test_load_prepared_rejects_path_traversal(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"../outside": "sha"}}))
    with pytest.raises(RecordError, match="checksum/path"):
        data.load_prepared(tmp_path)
