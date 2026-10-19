"""Thin Kaggle handoff: exact input pins, two workers, explicit restore action."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from kodoom.dataset_lock import read_selection
from kodoom.package_import import file_sha
from kodoom.schema import RecordError
from kodoom.training.data import load_prepared, prepare
from kodoom.training.remote import private_api, restore

DATASET_REPO = "Farahani/kodoom-en-fa"
DATASET_REVISION = "1b98bbba48043c62c6ae2c821a55b6f8a5e7a927"


def remote_input(selection_path: Path, out: Path) -> Path:
    """Download only the pinned manifest and its inventoried files; preserve bytes."""
    from huggingface_hub import snapshot_download

    selection = read_selection(selection_path)
    root = Path(
        snapshot_download(
            DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
            allow_patterns=["provenance/release.json"],
        )
    )
    path = root / "provenance/release.json"
    if file_sha(path) != selection["manifest_sha256"]:
        raise RecordError("remote dataset manifest differs from accepted baseline")
    release = json.loads(path.read_text("utf-8"))
    root = Path(
        snapshot_download(
            DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
            allow_patterns=["provenance/release.json"] + [e["path"] for e in release["files"]],
        )
    )
    if out.exists():
        m, _ = load_prepared(out)
        if m["dataset_lock"]["manifest_sha256"] != selection["manifest_sha256"]:
            raise RecordError("existing prepared directory uses a different dataset")
    else:
        prepare(root, selection_path, out)
    return out


def command(
    action: str,
    *,
    prepared: Path,
    settings: Path,
    backup_repo: str,
    run_id: str,
    runs_dir: Path | None = None,
) -> list[str]:
    if action not in ("preflight", "train"):
        raise RecordError("launch action must be preflight or train")
    result = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "kodoom.training",
        action,
        "--profile",
        "kaggle",
        "--prepared",
        str(prepared),
        "--settings",
        str(settings),
        "--run-id",
        run_id,
        "--hub-repo",
        backup_repo,
    ]
    if runs_dir:
        result += ["--runs-dir", str(runs_dir)]
    return result


def run(
    action: str,
    *,
    backup_repo: str,
    request: Path = Path("workflows/model-training.toml"),
    restore_revision: str = "",
    restore_run_id: str = "",
) -> dict:
    """Kaggle secrets/runtime setup happens in the notebook before this function."""
    if sys.platform != "linux" or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise RecordError("this handoff requires an active Kaggle Linux notebook")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", backup_repo):
        raise RecordError("set BACKUP_REPO to an existing private owner/model repository")
    if not os.environ.get("HF_TOKEN"):
        raise RecordError("enable the HF_TOKEN Kaggle Secret for this notebook")
    private_api(backup_repo)
    with request.open("rb") as stream:
        spec = tomllib.load(stream)
    prepared = remote_input(Path(spec["selection"]), Path(spec["prepared"]))
    runs_dir = None
    if action == "restore-preflight":
        if not re.fullmatch(r"[0-9a-f]{40}", restore_revision):
            raise RecordError("RESTORE_REVISION must be the complete checkpoint Hub commit")
        if restore_run_id != spec["preflight_run_id"]:
            raise RecordError("fresh-session restore currently verifies the preflight run only")
        runs_dir = Path(spec["restore_runs_dir"])
        restore(backup_repo, restore_revision, restore_run_id, runs_dir / restore_run_id)
        action, run_id = "preflight", restore_run_id
    elif action == "train":
        # Explicit operator transition after verifying preflight in a fresh session.
        run_id = spec["train_run_id"]
    elif action == "preflight":
        run_id = spec["preflight_run_id"]
    else:
        raise RecordError("ACTION must be preflight, restore-preflight or train")
    words = command(
        action,
        prepared=prepared,
        settings=Path(spec["settings"]),
        backup_repo=backup_repo,
        run_id=run_id,
        runs_dir=runs_dir,
    )
    subprocess.run(words, check=True, env=dict(os.environ, TOKENIZERS_PARALLELISM="false"))
    root = runs_dir or Path("/kaggle/working/kodoom/runs")
    return json.loads((root / run_id / "summary.json").read_text("utf-8"))
