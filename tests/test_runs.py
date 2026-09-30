import csv
import json
from collections import namedtuple
from pathlib import Path

import pytest

import kodoom.runs as runs_module
from kodoom.config import Profile
from kodoom.runs import NoSpaceError, Run, RunError, list_runs

CONFIG = {"model": "tiny", "lr": 1e-4, "seed": 1234}


@pytest.fixture
def profile(tmp_path):
    return Profile(
        name="test",
        seed=1234,
        runs_dir=tmp_path / "drive" / "runs",
        device="cpu",
        precision="fp32",
        threads=1,
        max_cases_per_source=2,
        scratch_dir=tmp_path / "scratch",
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
        reserve_gb=0,
    )


def writer(text: str, size: int = 10):
    def write(d: Path) -> None:
        (d / "weights.bin").write_bytes(b"w" * size)
        (d / "state.txt").write_text(text, encoding="utf-8")

    return write


def test_new_run_records_its_identity(profile):
    run = Run.open(profile, "r1", CONFIG)
    meta = json.loads((run.dir / "run.json").read_text(encoding="utf-8"))
    assert meta["config"] == CONFIG
    assert meta["status"] == "running"
    assert meta["profile"] == "test"
    assert run.latest is None and run.best is None
    assert run.read_log()[0]["event"] == "started"


def test_resume_after_a_dead_session(profile):
    run = Run.open(profile, "r1", CONFIG)
    run.save_latest(100, writer("step 100"))
    run.log(step=100, loss=0.5)
    # The session dies; a new one opens the same run with the same config.
    resumed = Run.open(profile, "r1", CONFIG)
    assert resumed.latest.step == 100
    assert (resumed.latest.path / "state.txt").read_text(encoding="utf-8") == "step 100"
    assert resumed.meta["resumed"] == 1
    events = [e["event"] for e in resumed.read_log()]
    assert events == ["started", "checkpoint", "metrics", "resumed"]


def test_resuming_with_a_changed_config_is_refused(profile):
    Run.open(profile, "r1", CONFIG)
    with pytest.raises(RunError, match="changed: lr"):
        Run.open(profile, "r1", {**CONFIG, "lr": 3e-4})


def test_latest_is_replaced_not_accumulated(profile):
    run = Run.open(profile, "r1", CONFIG)
    run.save_latest(100, writer("a"))
    run.save_latest(200, writer("b"))
    assert run.latest.step == 200
    leftovers = [p.name for p in run.dir.iterdir() if p.name.startswith(".")]
    assert leftovers == []
    assert not (profile.scratch_dir / "r1" / "latest").exists()


def test_failed_write_leaves_the_previous_checkpoint(profile):
    run = Run.open(profile, "r1", CONFIG)
    run.save_latest(100, writer("good"))

    def crash(d: Path) -> None:
        (d / "weights.bin").write_bytes(b"partial")
        raise RuntimeError("session died")

    with pytest.raises(RuntimeError):
        run.save_latest(200, crash)
    assert run.latest.step == 100


def test_no_space_is_refused_before_touching_drive(profile, monkeypatch):
    run = Run.open(profile, "r1", CONFIG)
    run.save_latest(100, writer("good"))
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(runs_module.shutil, "disk_usage", lambda _: usage(100, 95, 5))
    with pytest.raises(NoSpaceError, match="previous checkpoint is intact"):
        run.save_latest(200, writer("big", size=1000))
    assert run.latest.step == 100


def test_reserve_counts_against_free_space(profile, monkeypatch):
    run = Run.open(profile, "r1", CONFIG)
    run.reserve_bytes = 50
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(runs_module.shutil, "disk_usage", lambda _: usage(100, 40, 60))
    with pytest.raises(NoSpaceError):
        run.save_latest(1, writer("x", size=20))


def test_crash_between_renames_recovers_the_old_checkpoint(profile):
    run = Run.open(profile, "r1", CONFIG)
    run.save_latest(100, writer("good"))
    # Simulate death after "latest -> .latest.old" but before the new one was complete.
    (run.dir / "latest").rename(run.dir / ".latest.old")
    (run.dir / "latest").mkdir()
    (run.dir / "latest" / "weights.bin").write_bytes(b"half")
    (run.dir / ".latest.incoming").mkdir()

    resumed = Run.open(profile, "r1", CONFIG)
    assert resumed.latest.step == 100
    names = {p.name for p in resumed.dir.iterdir()}
    assert ".latest.incoming" not in names and ".latest.old" not in names
    assert any(n.startswith(".latest.broken-") for n in names)  # kept aside, not deleted


def test_truncated_file_invalidates_a_checkpoint(profile):
    run = Run.open(profile, "r1", CONFIG)
    run.save_latest(100, writer("good", size=10))
    (run.dir / "latest" / "weights.bin").write_bytes(b"short")  # e.g. an unsynced Drive file
    assert run.latest is None


def test_best_is_saved_only_when_it_improves(profile):
    run = Run.open(profile, "r1", CONFIG)
    assert run.save_best(1, 0.9, writer("a")).metric == 0.9
    assert run.save_best(2, 1.1, writer("b")) is None
    assert run.save_best(3, 0.7, writer("c")).step == 3
    assert run.best.metric == 0.7


def test_best_higher_is_better(profile):
    run = Run.open(profile, "r1", CONFIG)
    run.save_best(1, 0.5, writer("a"), lower_is_better=False)
    assert run.save_best(2, 0.4, writer("b"), lower_is_better=False) is None
    assert run.save_best(3, 0.6, writer("c"), lower_is_better=False).step == 3


def test_finish_keeps_best_drops_latest_and_registers(profile):
    run = Run.open(profile, "r1", CONFIG)
    run.save_latest(100, writer("a"))
    run.save_best(100, 0.3, writer("b"))
    run.finish(val_log_loss=0.3)
    assert not (run.dir / "latest").exists()
    assert run.best.metric == 0.3
    with (profile.runs_dir / "registry.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["run_id"] == "r1" and rows[0]["best_metric"] == "0.3"
    with pytest.raises(RunError, match="already finished"):
        Run.open(profile, "r1", CONFIG)


def test_log_survives_a_line_cut_by_a_dead_session(profile):
    run = Run.open(profile, "r1", CONFIG)
    run.log(step=1, loss=1.0)
    with (run.dir / "log.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"time": "2026-')
    assert [e.get("step") for e in run.read_log()] == [None, 1]


def test_list_runs(profile):
    Run.open(profile, "a", CONFIG).save_latest(5, writer("x"))
    done = Run.open(profile, "b", CONFIG)
    done.save_best(7, 0.2, writer("y"))
    done.finish()
    (profile.runs_dir / "_scratch").mkdir()
    rows = {r["run_id"]: r for r in list_runs(profile.runs_dir)}
    assert set(rows) == {"a", "b"}
    assert rows["a"]["status"] == "running" and rows["a"]["latest_step"] == 5
    assert rows["b"]["status"] == "finished" and rows["b"]["best_metric"] == 0.2


@pytest.mark.parametrize("bad", ["", "../x", "a b", "-a"])
def test_invalid_run_ids(profile, bad):
    with pytest.raises(RunError, match="run id"):
        Run.open(profile, bad, CONFIG)
