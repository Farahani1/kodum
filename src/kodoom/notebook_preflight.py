"""CPU-only account, input and dependency checks before requesting Kaggle TPU time."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from kodoom.bulk.data import ROOT, load_request, prepare_units, validate_scope
from kodoom.bulk.launch import download, validate_settings
from kodoom.bulk.prompts import requests
from kodoom.bulk.remote import HFRemote
from kodoom.bulk.state import atomic_write, encoded
from kodoom.config import load_profile
from kodoom.runtime import validate_tpu_runtime
from kodoom.sources import check_record
from kodoom.tpu import host_resources
from kodoom.translate.checkpoint import describe_checkpoint
from kodoom.translate.jax import format_prompt


class CheckFailure(RuntimeError):
    """An intentionally safe, actionable message for a check failure."""


def attached_checkpoint(input_root=Path("/kaggle/input")):
    """Inspect mounted version-1 files only; never download a 54GB model as a probe."""
    suffix = ("gemma-3", "flax", "gemma3-27b-it", "1")
    candidates = [
        path.parent
        for path in input_root.rglob("tokenizer.model")
        if path.parent.parts[-4:] == suffix
    ]
    if len(candidates) != 1:
        raise CheckFailure(
            "Accept Gemma terms on Kaggle and attach google/gemma-3/flax/gemma3-27b-it/1 "
            "under Inputs/Models. Its version-1 tokenizer and checkpoint must be mounted."
        )
    directory = candidates[0]
    info = describe_checkpoint(directory)
    if not info["total_bytes"] or not (directory / "gemma3-27b-it/_METADATA").read_bytes():
        raise CheckFailure("The attached Flax checkpoint metadata is empty or unreadable")
    return directory, info


def write_probe(remote):
    """Add/read/delete a unique tiny file without acquiring or editing a campaign lease."""
    from huggingface_hub import CommitOperationDelete

    name = f"preflight-probes/{uuid.uuid4().hex}.json"
    data = encoded({"purpose": "kodoom CPU checkpoint access probe", "nonce": uuid.uuid4().hex})
    revision = None
    try:
        revision = remote.commit({name: data}, remote.head())
        if remote.read(name, revision) != data:
            raise CheckFailure("Private HF write succeeded but exact-revision readback failed")
    finally:
        if revision is not None:
            remote.api.create_commit(
                repo_id=remote.repo,
                repo_type="dataset",
                parent_commit=remote.head(),
                commit_message="Remove private CPU preflight probe",
                operations=[CommitOperationDelete(path_in_repo=name, is_folder=False)],
            )
    return "Private write and exact-revision readback passed; temporary file removed"


def resume_check(remote, campaign_id, revision, takeover):
    prefix = f"campaigns/{campaign_id}/"
    head = remote.head()
    lease = remote.read(prefix + "writer.json", head)
    if lease and json.loads(lease).get("released") is False and not takeover:
        raise CheckFailure(
            "An unreleased writer lease exists. Confirm the old worker ended before using TAKEOVER"
        )
    control = remote.read(prefix + "control.json", head)
    if control and json.loads(control).get("pause") is True:
        raise CheckFailure("This campaign is paused; clear its pause control before submitting")
    checkpoint = remote.read(prefix + "checkpoint.json", head)
    if checkpoint is None:
        return "New campaign; no previous remote checkpoint"
    with tempfile.TemporaryDirectory() as temp:
        result = download(campaign_id=campaign_id, repo=remote.repo, directory=temp)
        identity = json.loads((Path(temp) / "campaign.json").read_bytes())
        if identity["code_revision"] != revision or identity["request"] != load_request():
            raise CheckFailure(
                "Saved campaign uses another code/request identity. Keep its old pin to resume, "
                "or deliberately select a new CAMPAIGN_ID for this updated runner"
            )
        return f"Verified restore/checksums: {result['counts']}"


def dependency_resolution():
    constraints = "tpu-py313.txt" if sys.version_info[:2] == (3, 13) else "tpu.txt"
    with tempfile.TemporaryDirectory() as temp:
        report = Path(temp) / "pip-report.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--disable-pip-version-check",
                "--report",
                str(report),
                "-e",
                f"{ROOT}[tpu]",
                "-c",
                str(ROOT / "constraints" / constraints),
            ],
            capture_output=True,
            text=True,
            timeout=240,
        )
        if result.returncode or not report.exists():
            raise CheckFailure(
                "Pinned TPU dependency resolution failed. Check PyPI/network and supported Python; "
                "the CPU check did not install or import these accelerator packages"
            )
        count = len(json.loads(report.read_text("utf-8"))["install"])
    return f"{count} pinned packages resolve; TPU imports/ABI still require a real TPU session"


def prompt_audit(units, directory, request):
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(directory / "tokenizer.model"))
    maximum = 0
    oversized = []
    count = 0
    for unit in units:
        for messages in requests(unit):
            length = len(processor.encode(format_prompt(messages), add_bos=True))
            count += 1
            maximum = max(maximum, length)
            if length > request["input_tokens"]:
                oversized.append(unit.key)
    if oversized:
        raise CheckFailure(
            f"{len(oversized)} prompts exceed {request['input_tokens']} input tokens; "
            f"max={maximum}. Examples: {', '.join(dict.fromkeys(oversized))[:400]}. "
            "Do not truncate; review the scope/limits before a new campaign"
        )
    return f"CPU SentencePiece screening: {count} prompts, max={maximum}; TPU tokenizer rechecks"


def run_checks(*, repo, campaign_id, operator, revision, takeover, report_path):
    """Aggregate independent failures and skip dependent checks explicitly."""
    results = []
    values = {}

    def check(name, action, hint, needs=()):
        if any(dependency not in values for dependency in needs):
            results.append({"check": name, "status": "SKIP", "detail": "Prerequisite failed"})
            print(f"SKIP: {name} (prerequisite failed)", flush=True)
            return
        print(f"CHECK: {name}...", flush=True)
        try:
            value = action()
        except Exception as exc:
            # Never echo provider exception bodies, URLs, headers or credentials.
            detail = str(exc) if isinstance(exc, CheckFailure) else hint
            results.append({"check": name, "status": "FAIL", "detail": detail})
            print(f"FAIL: {name}: {detail}", flush=True)
        else:
            values[name] = value
            detail = value if isinstance(value, str) else "OK"
            results.append({"check": name, "status": "PASS", "detail": detail})
            print(f"PASS: {name}: {detail}", flush=True)

    check("runtime", validate_tpu_runtime, "Requires Linux Python 3.11–3.13 and glibc 2.31+")
    check("settings", lambda: validate_settings(campaign_id, repo, operator), "Invalid settings")
    check("resources", host_resources, "Insufficient host RAM or temporary disk space")

    def writable():
        profile = load_profile("kaggle-tpu")
        for path in (profile.data_dir, profile.runs_dir, profile.scratch_dir):
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=path) as handle:
                handle.write(b"kodoom probe")
                handle.flush()
        return "Data, run and scratch paths are writable"

    check("writable paths", writable, "Cannot write profile data/run/scratch paths")
    check(
        "private HF access",
        lambda: HFRemote(repo, os.environ.get("HF_TOKEN", "")),
        "Check HF_TOKEN validity, dataset privacy, ID, read permissions and organization approval",
        needs=("settings",),
    )
    check(
        "resume state",
        lambda: resume_check(values["private HF access"], campaign_id, revision, takeover),
        "Cannot verify saved snapshot, manifest or writer lease",
        needs=("private HF access",),
    )
    check(
        "private HF write/readback",
        lambda: write_probe(values["private HF access"]),
        "Check dataset WRITE permission and connectivity; probe cleanup may need attention",
        needs=("private HF access", "resume state"),
    )
    check("attached Flax model", attached_checkpoint, "Model metadata/tokenizer unreadable")

    def inputs():
        request = load_request()
        units = prepare_units(request)
        validate_scope(units, request)
        for unit in units:
            for record in unit.records:
                check_record(record)
        return units

    check(
        "pinned source inputs", inputs, "Pinned English sources could not be downloaded/validated"
    )
    check(
        "prompt limits",
        lambda: prompt_audit(
            values["pinned source inputs"], values["attached Flax model"][0], load_request()
        ),
        "Could not parse tokenizer or measure production prompts",
        needs=("attached Flax model", "pinned source inputs"),
    )
    check("TPU dependency resolution", dependency_resolution, "Resolution timed out; retry on CPU")
    report = {
        "schema_version": 1,
        "code_revision": revision,
        "passed": all(result["status"] == "PASS" for result in results),
        "checks": results,
        "remaining": [
            "TPU allocation and eight-device BF16 kernels",
            "real library imports/ABI",
            "27B weight/cache fit, compilation and production throughput",
            "physical fresh-session restore and human translation review",
        ],
    }
    atomic_write(Path(report_path), encoded(report))
    print("CPU PREFLIGHT " + ("PASSED" if report["passed"] else "FAILED"), flush=True)
    print(
        "CPU checks do not certify TPU memory fit, model quality or training-license eligibility."
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--takeover", action="store_true")
    args = parser.parse_args()
    result = run_checks(
        repo=args.repo,
        campaign_id=args.campaign,
        operator=args.operator,
        revision=args.revision,
        takeover=args.takeover,
        report_path=args.report,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
