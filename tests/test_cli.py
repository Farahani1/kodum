import json

import pytest

from kodoom.cli import main
from kodoom.schema import Option, Record, write_jsonl


def record(**overrides):
    base = dict(
        id="r1",
        source_id="c1",
        source="dml-qom/FarsTail",
        source_revision=None,
        license="Apache-2.0",
        split="train",
        origin="native",
        task_family="entailment",
        state_lang="fa",
        question_lang="fa",
        state="متن",
        question_type="noul",
        question_text="درست است؟",
        options=(Option("yes", "بله"), Option("no", "خیر")),
        gold={"yes": 1.0},
    )
    base.update(overrides)
    return Record(**base)


def test_info_prints_the_profile(capsys):
    assert main(["info", "--profile", "dev"]) == 0
    out = capsys.readouterr().out
    assert "device: cpu" in out
    assert "max_cases_per_source: 20" in out


def test_profile_is_required():
    with pytest.raises(SystemExit) as e:
        main(["info"])
    assert e.value.code == 2


def test_unknown_profile_is_a_clean_error(capsys):
    assert main(["info", "--profile", "laptop"]) == 1
    assert "unknown profile" in capsys.readouterr().err


def test_validate_counts_records(tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    write_jsonl(path, [record(), record(id="r2", split="test")])
    assert main(["validate", str(path)]) == 0
    out = capsys.readouterr().out
    assert "2 records OK" in out


def test_validate_rejects_test_only_source_in_training(tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    bad = record(source="facebook/belebele", license="CC-BY-SA-4.0")
    path.write_text(json.dumps(bad.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    assert main(["validate", str(path)]) == 1
    assert "test-only" in capsys.readouterr().err


def test_check_passes_on_the_dev_profile(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["check", "--profile", "dev"]) == 0
    out = capsys.readouterr().out
    assert "[  ok] runs_dir" in out
    assert out.strip().endswith("ready")


def test_runs_lists_resumable_runs(tmp_path, capsys, monkeypatch):
    from kodoom.config import load_profile
    from kodoom.runs import Run

    monkeypatch.chdir(tmp_path)
    run = Run.open(load_profile("dev"), "smoke-1", {"x": 1})
    run.save_latest(3, lambda d: (d / "w.bin").write_bytes(b"123"))
    assert main(["runs", "--profile", "dev"]) == 0
    out = capsys.readouterr().out
    assert "smoke-1" in out and "running" in out
