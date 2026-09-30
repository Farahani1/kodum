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


def test_generate_writes_records_and_a_manifest(tmp_path, capsys):
    out = tmp_path / "skills"
    assert main(["generate", "--profile", "dev", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "jalali-dates.jsonl: 160 records" in text  # dev caps pairs at 20 per kind
    assert main(["validate", str(out / "jalali-dates.jsonl")]) == 0

    import hashlib
    import json

    manifest = json.loads((out / "jalali-dates.manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 1234 and manifest["pairs_per_kind"] == 20
    assert (
        manifest["sha256"] == hashlib.sha256((out / "jalali-dates.jsonl").read_bytes()).hexdigest()
    )


def test_generate_is_reproducible_byte_for_byte(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    for out in (first, second):
        assert main(["generate", "jalali-dates", "--profile", "dev", "--out", str(out)]) == 0
    assert (first / "jalali-dates.jsonl").read_bytes() == (
        second / "jalali-dates.jsonl"
    ).read_bytes()


def test_generate_options_and_errors(tmp_path, capsys):
    out = tmp_path / "o"
    assert main(["generate", "--profile", "colab", "--pairs-per-kind", "3", "--out", str(out)]) == 0
    assert "24 records" in capsys.readouterr().out
    assert main(["generate", "nope", "--profile", "dev", "--out", str(out)]) == 1
    assert "unknown generator" in capsys.readouterr().err


def test_generate_writes_into_the_profiles_data_dir_by_default(tmp_path, capsys):
    profile = tmp_path / "mine.toml"
    profile.write_text(
        '[run]\nseed = 5\nruns_dir = "r"\n[compute]\ndevice = "cpu"\nprecision = "fp32"\n'
        '[storage]\nscratch_dir = "s"\ncache_dir = "c"\ndata_dir = "'
        + (tmp_path / "drive-data").as_posix()
        + '"\n[data]\nmax_cases_per_source = 2\n',
        encoding="utf-8",
    )
    assert main(["generate", "jalali-dates", "--profile", str(profile)]) == 0
    assert (tmp_path / "drive-data" / "skills" / "jalali-dates.jsonl").is_file()
    assert "drive-data" in capsys.readouterr().out
