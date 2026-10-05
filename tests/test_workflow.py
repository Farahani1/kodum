import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

import kodoom.workflow as workflow
from kodoom.artifacts import restore_bundle, verify_bundle
from kodoom.check import Check
from kodoom.config import load_profile
from kodoom.helmo import row_record
from kodoom.schema import read_jsonl, write_jsonl
from kodoom.typed_decisions import WORKFLOWS, case_records
from tests.test_helmo import make_row as helmo_row
from tests.test_typed_decisions import make_row as typed_row


def fixture_run(tmp_path, monkeypatch, mode="preflight"):
    base = replace(
        load_profile("dev"),
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        scratch_dir=tmp_path / "scratch",
        cache_dir=tmp_path / "cache",
    )
    monkeypatch.setattr(workflow, "load_profile", lambda _: base)
    root = base.data_dir / "workflows/free-text-gate" / ("smoke-" + mode)
    records = [
        row_record(helmo_row(qtype=kind), index)
        for index, kind in enumerate(["choice", "noul", "score"] * 14)
    ]
    write_jsonl(root / "helmo/en/train.jsonl", records)
    records = [
        r
        for name in WORKFLOWS
        for index in range(5)
        for r in case_records(typed_row(f"{name}-{index}", workflow=name, state="{}"))
    ]
    write_jsonl(root / "typed-decisions/en/train.jsonl", records)
    return base, root


def test_gate_end_to_end_and_bundle_can_resume_on_another_path(tmp_path, monkeypatch):
    base, root = fixture_run(tmp_path, monkeypatch, "gate")
    state = workflow.run_current(provider="generic", profile="test", mode="gate", smoke=True)
    assert state["status"] == "awaiting-review"
    assert state["outputs"]["helmo"]["cases"] == 40
    assert state["outputs"]["typed-decisions"]["cases"] == 20
    assert len(list(read_jsonl(root / "helmo/fa/stub/train.jsonl"))) == 40
    sheet = root / "review-helmo.csv"
    with sheet.open(encoding="utf-8-sig") as stream:
        assert len(list(csv.DictReader(stream))) == 40
    sheet.write_text(sheet.read_text("utf-8-sig") + "reader annotation\n", encoding="utf-8-sig")
    saved_sheet = sheet.read_bytes()
    again = workflow.run_current(provider="generic", profile="test", mode="gate", smoke=True)
    assert sheet.read_bytes() == saved_sheet
    assert again["inputs"] == state["inputs"]
    manifest = verify_bundle(Path(again["bundle"]))
    assert "execution.json" in manifest["files"]
    next_base = replace(base, data_dir=tmp_path / "new-data")
    monkeypatch.setattr(workflow, "load_profile", lambda _: next_base)
    restored = workflow.run_current(
        provider="generic", profile="test", mode="gate", smoke=True, restore=again["bundle"]
    )
    assert restored["outputs"] == again["outputs"]


def test_preflight_is_separate_and_runs_only_current_recipes(tmp_path, monkeypatch):
    _, root = fixture_run(tmp_path, monkeypatch)
    result = workflow.run_current(provider="generic", profile="test", smoke=True)
    assert result["status"] == "preflight-passed"
    assert result["outputs"]["helmo"]["cases"] == 3
    assert result["outputs"]["typed-decisions"]["cases"] == 4
    assert [c["args"][0] for c in result["commands"]] == [
        "translate",
        "translations",
        "translate",
        "translations",
    ]
    assert not (root.parent / "gate").exists()


def test_readiness_failure_stops_commands_but_exports_diagnostics(tmp_path, monkeypatch):
    _, root = fixture_run(tmp_path, monkeypatch)
    monkeypatch.setattr(workflow, "run_checks", lambda _: [Check("device", "FAIL", "no GPU")])
    monkeypatch.setattr(workflow, "command", lambda *args: pytest.fail("command ran after failure"))
    with pytest.raises(ValueError, match="readiness failed"):
        workflow.run_current(provider="generic", profile="test", smoke=True)
    assert json.loads((root / "execution.json").read_text("utf-8"))["status"] == "failed"
    assert list((tmp_path / "bundles").glob("*.zip"))


def test_changed_inputs_are_rejected_without_poisoning_the_resume_contract(tmp_path, monkeypatch):
    _, root = fixture_run(tmp_path, monkeypatch)
    workflow.run_current(provider="generic", profile="test", smoke=True)
    contract = (root / "execution.json").read_bytes()
    source = root / "helmo/en/train.jsonl"
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="input checksum"):
        workflow.run_current(provider="generic", profile="test", smoke=True)
    assert (root / "execution.json").read_bytes() == contract


def test_missing_translation_is_failure_not_a_review_gate(tmp_path, monkeypatch):
    _, root = fixture_run(tmp_path, monkeypatch)
    monkeypatch.setattr(workflow, "command", lambda *args: 1)
    with pytest.raises(ValueError, match="missing, duplicate"):
        workflow.run_current(provider="generic", profile="test", smoke=True)
    assert json.loads((root / "execution.json").read_text("utf-8"))["status"] == "failed"


def test_partial_attempt_without_second_source_can_resume(tmp_path, monkeypatch):
    _, root = fixture_run(tmp_path, monkeypatch)
    typed = root / "typed-decisions/en/train.jsonl"
    original = typed.read_bytes()
    typed.unlink()
    with pytest.raises(ValueError, match="fixture English"):
        workflow.run_current(provider="generic", profile="test", smoke=True)
    typed.write_bytes(original)
    state = workflow.run_current(provider="generic", profile="test", smoke=True)
    assert state["status"] == "preflight-passed"


def test_dry_run_has_no_side_effects(tmp_path, monkeypatch):
    base = replace(load_profile("dev"), data_dir=tmp_path / "new")
    monkeypatch.setattr(workflow, "load_profile", lambda _: base)
    result = workflow.run_current(provider="generic", profile="dev", smoke=True, dry_run=True)
    assert result["limits"] == {"helmo-gate": 3, "typed-gate": 4}
    assert not base.data_dir.exists()


def test_changed_model_semantics_cannot_resume(tmp_path, monkeypatch):
    _, root = fixture_run(tmp_path, monkeypatch)
    workflow.run_current(provider="generic", profile="test", smoke=True)
    contract = json.loads((root / "execution.json").read_text("utf-8"))
    contract["semantic"]["dtype"] = "fp16"
    (root / "execution.json").write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        workflow.run_current(provider="generic", profile="test", smoke=True)


def test_unknown_recipe_and_changed_gate_are_rejected(tmp_path):
    path = tmp_path / "current.toml"
    text = workflow.REQUEST.read_text("utf-8")
    path.write_text(text.replace('"helmo-gate"', '"training"'), encoding="utf-8")
    with pytest.raises(ValueError, match="approved free-text"):
        workflow.load_request(path)


def test_restore_refuses_existing_artifacts(tmp_path):
    with pytest.raises(ValueError, match="destination exists"):
        restore_bundle(tmp_path / "not-even-a-zip", tmp_path)
