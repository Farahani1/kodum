"""Thin notebook orchestration with a hard child-process session deadline."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from kodoom.bulk.data import ROOT, load_request, validate_campaign_id
from kodoom.config import load_profile


def validate_settings(campaign_id, repo, operator):
    validate_campaign_id(campaign_id)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError(
            "Set HF_DATASET_REPO to your existing private organization/dataset repository"
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", operator):
        raise ValueError("OPERATOR must be a short non-secret name")


def run(
    *,
    campaign_id,
    repo,
    operator="owner",
    takeover=False,
    provider="kaggle",
    profile="kaggle-tpu",
    session_started=None,
):
    validate_settings(campaign_id, repo, operator)
    if provider != "kaggle" or profile != "kaggle-tpu":
        raise ValueError("The execution notebook bulk campaign requires Kaggle TPU")
    if not os.environ.get("HF_TOKEN"):
        raise ValueError(
            "Enable the HF_TOKEN Kaggle secret with write access to the private dataset"
        )
    request_path = ROOT / "workflows/tpu-bulk.toml"
    request = load_request(request_path)
    deadline = (session_started if session_started is not None else time.monotonic()) + request[
        "session_seconds"
    ]
    remaining = deadline - time.monotonic()
    if remaining <= request["finalize_seconds"]:
        raise RuntimeError("Insufficient session time remains after bootstrap")
    root = load_profile(profile).data_dir / "bulk" / campaign_id
    result_path = root / "result.json"
    result_path.unlink(missing_ok=True)  # A stale success must not mask a killed attempt.
    words = [
        sys.executable,
        "-u",
        "-m",
        "kodoom.bulk",
        "--campaign",
        campaign_id,
        "--repo",
        repo,
        "--root",
        str(root),
        "--provider",
        provider,
        "--profile",
        profile,
        "--operator",
        operator,
        "--deadline",
        str(deadline),
        "--request",
        str(request_path),
    ]
    if takeover:
        words.append("--takeover")
    try:
        outcome = subprocess.run(words, timeout=remaining, check=False)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Session deadline reached; resume the last verified HF checkpoint in a fresh session"
        ) from None
    if not result_path.exists():
        raise RuntimeError(
            "Worker stopped before its final report; inspect output "
            "and resume the durable checkpoint"
        )
    result = json.loads(result_path.read_bytes())
    if outcome.returncode or result["error_type"] or not result["persisted"]:
        raise RuntimeError(
            f"Bulk worker stopped: {result['error_type'] or result['status']}; "
            f"local report: {result_path}"
        )
    return result


def control(*, campaign_id, repo, pause):
    """Owner-requested pause/resume, observed by the worker at its next save."""
    from kodoom.bulk.remote import HFRemote
    from kodoom.bulk.state import encoded

    validate_settings(campaign_id, repo, "owner")
    if type(pause) is not bool:
        raise ValueError("pause must be a boolean")
    remote = HFRemote(repo, os.environ.get("HF_TOKEN", ""))
    return remote.commit(
        {f"campaigns/{campaign_id}/control.json": encoded({"pause": pause})}, remote.head()
    )


def download(*, campaign_id, repo, directory):
    """Download a verified review snapshot without acquiring the inference lease."""
    from kodoom.bulk.data import input_units
    from kodoom.bulk.remote import HFRemote, RemoteSession
    from kodoom.bulk.state import Campaign

    validate_settings(campaign_id, repo, "reviewer")
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Download into a fresh directory; keep annotations in a separate copy")
    remote = RemoteSession(
        HFRemote(repo, os.environ.get("HF_TOKEN", "")), campaign_id, "read-only", "reviewer"
    )
    if not remote.restore(root, read_only=True):
        raise ValueError("No durable campaign checkpoint exists yet")
    campaign = Campaign(
        root,
        json.loads((root / "campaign.json").read_bytes()),
        input_units((root / "inputs.json").read_bytes()),
    )
    return {
        "revision": remote.verified_revision,
        "counts": campaign.counts(),
        "directory": str(root),
    }
