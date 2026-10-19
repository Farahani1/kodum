"""Explicit private Hub persistence; no credentials appear in configs or logs."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from kodoom.package_import import file_sha
from kodoom.schema import RecordError


def private_api(repo_id: str):
    from huggingface_hub import HfApi

    api = HfApi()
    # Require a named, existing private repo. Creation/selection is an operator step.
    info = api.repo_info(repo_id, repo_type="model")
    if not info.private:
        raise RecordError("training backup repository must be private")
    return api


def sync(run, repo_id: str, *, logs_only: bool = False) -> str:
    api = private_api(repo_id)
    files = {}
    patterns = [
        "run.json",
        "log.jsonl",
        "training-config.json",
        "data-manifest.json",
        "encoding-stats.json",
        "summary.json",
        "durable-manifest.json",
    ]
    if not logs_only:
        patterns += ["latest/**", "best/**"]
    for path in sorted(run.dir.rglob("*")):
        if not path.is_file() or path.name == "durable-manifest.json":
            continue
        rel = path.relative_to(run.dir).as_posix()
        if rel in patterns or (not logs_only and rel.startswith(("latest/", "best/"))):
            files[rel] = {"sha256": file_sha(path), "bytes": path.stat().st_size}
    if not logs_only:
        (run.dir / "durable-manifest.json").write_text(
            json.dumps({"files": files, "run_id": run.meta["run_id"]}, indent=2), encoding="utf-8"
        )
    else:
        # Never overwrite the complete snapshot manifest with hashes of newer logs.
        patterns.remove("durable-manifest.json")
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(run.dir),
        path_in_repo=run.meta["run_id"],
        allow_patterns=patterns,
        commit_message="training logs" if logs_only else "complete checkpoint",
    )
    return result.oid


def restore(repo_id: str, revision: str, run_id: str, target: Path) -> Path:
    from huggingface_hub import snapshot_download

    from kodoom.runs import RUN_ID

    private_api(repo_id)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RecordError("restore requires an immutable complete snapshot commit")
    if not RUN_ID.fullmatch(run_id) or target.exists():
        raise RecordError("restore requires a valid run ID and a new output directory")
    root = (
        Path(
            snapshot_download(
                repo_id, revision=revision, repo_type="model", allow_patterns=[f"{run_id}/**"]
            )
        )
        / run_id
    )
    manifest = json.loads((root / "durable-manifest.json").read_text("utf-8"))
    # Log-only commits may follow the snapshot. Require the exact snapshot revision.
    for rel, entry in manifest["files"].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha(path) != entry["sha256"]:
            raise RecordError("remote snapshot mismatch; restore the complete snapshot commit")
    if "latest/state.pt" not in manifest["files"]:
        raise RecordError("snapshot has no resumable training state")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root, target)
    return target
