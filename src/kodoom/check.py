"""Is this environment ready to run a profile? Used by ``kodoom check``.

Run it first in every Colab session: it catches an unmounted Drive, a CPU-only
runtime, a full Drive and a model cache pointed at Drive before any time is
spent, and shows which runs can be resumed.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from kodoom import __version__
from kodoom.config import Profile
from kodoom.runs import git_commit, list_runs

OK, WARN, FAIL = "ok", "warn", "FAIL"
_GB = 1024**3
DRIVE_ROOT = Path("/content/drive")
# Below this much free space on Drive, a main mmBERT-base run may not fit
# (plan: Storage budget, "if it is under about 8 GB, clear it first").
DRIVE_FREE_WARN_GB = 8


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def apply_environment(profile: Profile) -> None:
    """Point library caches at the profile's cache_dir, never at Drive.

    Called by every command that takes a profile, before anything imports
    transformers or huggingface_hub.
    """
    os.environ.setdefault("HF_HOME", str(profile.cache_dir))
    if profile.threads is not None:
        os.environ.setdefault("OMP_NUM_THREADS", str(profile.threads))


def run_checks(profile: Profile) -> list[Check]:
    checks = [
        Check("kodoom", OK, f"{__version__}, commit {git_commit() or 'unknown'}"),
        Check("python", OK, f"{platform.python_version()} on {platform.system()}"),
    ]
    checks += _check_runs_dir(profile)
    checks.append(_check_writable("scratch_dir", profile.scratch_dir))
    checks.append(_check_cache(profile))
    checks.append(_check_device(profile))
    return checks


def _check_runs_dir(profile: Profile) -> list[Check]:
    runs_dir = profile.runs_dir
    if _is_under(runs_dir, DRIVE_ROOT) and not (DRIVE_ROOT / "MyDrive").is_dir():
        return [
            Check(
                "runs_dir",
                FAIL,
                f"{runs_dir} is on Google Drive, but Drive is not mounted. In Colab run: "
                "from google.colab import drive; drive.mount('/content/drive')",
            )
        ]
    writable = _check_writable("runs_dir", runs_dir)
    if writable.status == FAIL:
        return [writable]

    checks = [writable]
    free = shutil.disk_usage(runs_dir).free
    if free < profile.reserve_gb * _GB:
        status = FAIL
    elif _is_under(runs_dir, DRIVE_ROOT) and free < DRIVE_FREE_WARN_GB * _GB:
        status = WARN
    else:
        status = OK
    checks.append(Check("free space", status, f"{free / _GB:.1f} GB free under {runs_dir}"))

    resumable = [r for r in list_runs(runs_dir) if r["status"] == "running"]
    detail = (
        ", ".join(f"{r['run_id']} (latest step {r['latest_step']})" for r in resumable)
        if resumable
        else "none"
    )
    checks.append(Check("resumable runs", OK, detail))
    return checks


def _check_writable(name: str, path: Path) -> Check:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".kodoom-probe-{uuid.uuid4().hex}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        return Check(name, FAIL, f"{path} is not writable: {e}")
    return Check(name, OK, str(path))


def _check_cache(profile: Profile) -> Check:
    if _is_under(profile.cache_dir, DRIVE_ROOT) or _is_under(profile.cache_dir, profile.runs_dir):
        return Check(
            "model cache",
            FAIL,
            f"{profile.cache_dir} would put downloaded models on Drive; base models never "
            "go on Drive (plan: Storage budget)",
        )
    return Check("model cache", OK, f"HF_HOME={os.environ.get('HF_HOME', profile.cache_dir)}")


def _check_device(profile: Profile) -> Check:
    if profile.device == "cpu":
        return Check("device", OK, "cpu (as the profile asks)")
    if importlib.util.find_spec("torch") is None:
        return Check("device", WARN, "cuda wanted, but torch is not installed yet")
    import torch

    if not torch.cuda.is_available():
        return Check(
            "device",
            FAIL,
            "cuda wanted, but no GPU is visible. In Colab: Runtime > Change runtime type > T4 GPU",
        )
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / _GB
    return Check("device", OK, f"{name}, {memory:.0f} GB")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
