import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from kodoom.artifacts import export_bundle, verify_bundle
from kodoom.schema import read_jsonl, write_jsonl
from kodoom.translate.gate_review import compare_gate, review_summary
from tests.test_workflow import fixture_run, run_gate
from tests.test_workflow_tpu import fake_tpu, run


def test_label_review_counts_instances_and_preserves_annotations(tmp_path, monkeypatch):
    _, root = fixture_run(tmp_path, monkeypatch)
    run_gate(provider="generic", profile="test", smoke=True)
    path = root / "review-labels-helmo.csv"
    with path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(row["meaning_changed"] == "" for row in rows)
    with (root / "label-templates-helmo.csv").open(encoding="utf-8-sig") as stream:
        assert sum(int(row["occurrences"]) for row in csv.DictReader(stream)) == len(rows)
    rows[0].update(meaning_changed="yes", needs_edit="yes", notes="negation changed")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    content = path.read_bytes()
    run_gate(provider="generic", profile="test", smoke=True)
    assert path.read_bytes() == content
    result = review_summary(root)[path.name]
    assert result["reviewed"] == result["meaning_changes"] == 1
    assert not result["complete"]


def baseline_bundle(gate, directory):
    root = Path(gate["artifact_root"])
    target = directory / "baseline"
    shutil.copytree(root, target)
    state = json.loads((target / "execution.json").read_text("utf-8"))
    state["request"]["translator"] = "gemma3-4b-bf16"
    state["model_revision"] = "b" * 40
    for dataset in ("helmo", "typed-decisions"):
        current = target / dataset / "fa/gemma3-27b-tpu-bf16/train.jsonl"
        rows = [
            replace(r, extra={**r.extra, "translator": "gemma3-4b-bf16"})
            for r in read_jsonl(current)
        ]
        write_jsonl(target / dataset / "fa/gemma3-4b-bf16/train.jsonl", rows)
    (target / "execution.json").write_text(json.dumps(state), encoding="utf-8")
    return target, export_bundle(target, directory / "baseline-bundles")


def test_paired_comparison_validates_sources_and_preserves_human_review(tmp_path, monkeypatch):
    fake_tpu(tmp_path, monkeypatch)
    run()
    gate = run("gate")
    target, bundle = baseline_bundle(gate, tmp_path)
    compared = compare_gate(gate, bundle)
    root = Path(compared["artifact_root"])
    path = root / "paired-labels.csv"
    with path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(row["assessment"] == "" for row in rows)
    assert "paired-labels.csv" in verify_bundle(Path(compared["bundle"]))["files"]
    path.write_text(path.read_text("utf-8-sig") + "reviewer note\n", encoding="utf-8-sig")
    saved = path.read_bytes()
    compare_gate(compared, bundle)
    assert path.read_bytes() == saved
    source = target / "helmo/en/train.jsonl"
    english = list(read_jsonl(source))
    write_jsonl(source, [replace(r, state=r.state + " different") for r in english])
    modified = export_bundle(target, tmp_path / "different-baseline")
    with pytest.raises(ValueError, match="different inputs"):
        compare_gate(compared, modified)
    (root / "paired-provenance.json").unlink()
    with pytest.raises(ValueError, match="English records differ"):
        compare_gate(compared, modified)


def test_smoke_does_not_count_as_a_real_model_comparison(tmp_path, monkeypatch):
    fixture_run(tmp_path, monkeypatch, "gate")
    gate = run_gate(provider="generic", profile="test", mode="gate", smoke=True)
    with pytest.raises(ValueError, match="real gate"):
        compare_gate(gate, "unused")
