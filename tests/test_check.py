from dataclasses import replace

import pytest

import kodoom.check as check_module
from kodoom.check import FAIL, OK, WARN, run_checks
from kodoom.config import load_profile


@pytest.fixture
def dev(tmp_path):
    p = load_profile("dev")
    return replace(
        p,
        runs_dir=tmp_path / "runs",
        scratch_dir=tmp_path / "scratch",
        cache_dir=tmp_path / "cache",
    )


def by_name(checks):
    return {c.name: c for c in checks}


def test_dev_profile_is_ready(dev):
    checks = by_name(run_checks(dev))
    assert all(c.status == OK for c in checks.values()), checks
    assert checks["resumable runs"].detail == "none"


def test_unmounted_drive_is_a_clear_failure(dev, tmp_path, monkeypatch):
    fake_drive = tmp_path / "content" / "drive"
    monkeypatch.setattr(check_module, "DRIVE_ROOT", fake_drive)
    checks = by_name(run_checks(replace(dev, runs_dir=fake_drive / "MyDrive" / "kodoom")))
    assert checks["runs_dir"].status == FAIL
    assert "drive.mount" in checks["runs_dir"].detail


def test_model_cache_on_drive_fails(dev, tmp_path, monkeypatch):
    fake_drive = tmp_path / "content" / "drive"
    (fake_drive / "MyDrive").mkdir(parents=True)
    monkeypatch.setattr(check_module, "DRIVE_ROOT", fake_drive)
    profile = replace(dev, cache_dir=fake_drive / "MyDrive" / "hf")
    assert by_name(run_checks(profile))["model cache"].status == FAIL


def test_low_drive_space_warns(dev, tmp_path, monkeypatch):
    from collections import namedtuple

    fake_drive = tmp_path / "content" / "drive"
    (fake_drive / "MyDrive").mkdir(parents=True)
    monkeypatch.setattr(check_module, "DRIVE_ROOT", fake_drive)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(check_module.shutil, "disk_usage", lambda _: usage(15, 10, 5 * 1024**3))
    profile = replace(dev, runs_dir=fake_drive / "MyDrive" / "kodoom" / "runs")
    assert by_name(run_checks(profile))["free space"].status == WARN


def test_cuda_profile_without_torch_warns(dev, monkeypatch):
    monkeypatch.setattr(check_module.importlib.util, "find_spec", lambda name: None)
    checks = by_name(run_checks(replace(dev, device="cuda", precision="fp16")))
    assert checks["device"].status == WARN
