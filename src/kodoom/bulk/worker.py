"""One model worker for a complete resumable translation campaign."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path

from kodoom import datadir
from kodoom.bulk.data import (
    input_units,
    load_request,
    prepare_units,
    prompt_identity,
    validate_scope,
)
from kodoom.bulk.prompts import VERSION, StubGenerator, rebuild, requests
from kodoom.bulk.remote import DirectoryRemote, HFRemote, RemoteSession
from kodoom.bulk.review import export_unit
from kodoom.bulk.saving import AutoSaver, StopWork
from kodoom.bulk.state import Campaign, atomic_write, digest, encoded
from kodoom.config import load_profile
from kodoom.runs import Run, git_commit
from kodoom.sources import check_record


def backend_identity(generator, dev):
    versions = {}
    if not dev:
        for name in (
            "jax",
            "jaxlib",
            "libtpu",
            "gemma",
            "kauldron",
            "flax",
            "orbax-checkpoint",
            "sentencepiece",
            "tensorflow",
        ):
            versions[name] = importlib.metadata.version(name)
    return {
        "checkpoint": generator.info["checkpoint"]["fingerprint"],
        "python": platform.python_version(),
        "libraries": versions,
        "parameter_dtypes": generator.info.get("parameter_dtypes", ["stub"]),
    }


def benchmark(generator, messages, request, *, dev=False):
    lengths = generator.lengths(messages)
    # The longest prompt is repeated to exercise the worst batch padding/cache
    # allocation. Mixed lengths then measure warm useful-work throughput.
    longest = messages[max(range(len(messages)), key=lengths.__getitem__)]
    rows = []
    examples = {}
    for size in request["batch_sizes"]:
        try:
            generator.batch([longest] * size, size)
            generator.batch(messages, size)
            before = len(generator.info["calls"])
            replies = generator.batch(messages, size)
            examples[size] = replies
            calls = generator.info["calls"][before:]
            seconds = sum(call["seconds"] for call in calls)
            devices = [device for call in calls for device in call["devices"]]
            memory_ok = dev or bool(devices)
            for device in devices:
                memory = device["memory"]
                limit = memory.get("bytes_limit")
                used = memory.get("peak_bytes_in_use", memory.get("bytes_in_use"))
                memory_ok &= (
                    limit is not None
                    and used is not None
                    and limit - used >= request["memory_reserve_bytes"]
                )
            valid = all(isinstance(reply, str) and reply.strip() for reply in replies)
            row = {
                "batch": size,
                "seconds": seconds,
                "items": len(messages),
                "requests_per_second": len(messages) / seconds if seconds else 0,
                "output_tokens": sum(call["output_tokens"] for call in calls),
                "warm": all(call["warm"] for call in calls),
                "memory_reserve_passed": bool(memory_ok),
                "complete_replies": valid,
                "devices": devices,
                "tokens_per_second": sum(call["output_tokens"] for call in calls) / seconds
                if seconds
                else 0,
                "passed": bool(
                    memory_ok and valid and seconds > 0 and all(call["warm"] for call in calls)
                ),
            }
            rows.append(row)
            print(
                f"Batch {size}: {row['requests_per_second']:.3f} requests/s; safe={row['passed']}",
                flush=True,
            )
        except Exception as exc:
            rows.append({"batch": size, "passed": False, "error_type": type(exc).__name__})
            if hasattr(generator, "recover"):
                generator.recover()
    good = [row for row in rows if row["passed"]]
    if not good:
        raise RuntimeError("No benchmark batch produced complete replies with the memory reserve")
    best = max(good, key=lambda row: row["requests_per_second"])
    return {
        "batch": best["batch"],
        "benchmark": rows,
        "benchmark_replies": examples[best["batch"]],
        "batch_one_replies": examples.get(1, []),
        "evidence": "cpu-fixture" if dev else "live-tpu",
    }


def diagnostic_order(units):
    helmo = [u for u in units if u.dataset == "helmo"][:40]
    typed = [u for u in units if u.dataset == "typed-decisions" and u.records[0].split == "train"][
        :20
    ]
    early = {u.key for u in [*helmo, *typed]}
    return [*helmo, *typed, *[u for u in units if u.key not in early]]


def _run_campaign(
    *,
    root: Path,
    campaign_id: str,
    provider="kaggle",
    profile="kaggle-tpu",
    repo="",
    local_remote=None,
    operator="operator",
    takeover=False,
    deadline=None,
    request_path=None,
    dev=False,
    max_units=None,
    generator_factory=None,
):
    request = load_request(request_path)
    profile_obj = load_profile(profile)
    if dev:
        if provider != "generic" or profile_obj.device != "cpu" or local_remote is None:
            raise ValueError("CPU smoke requires generic provider, CPU profile and local remote")
        profile_obj = replace(
            profile_obj,
            data_dir=root.parent,
            runs_dir=root.parent / "runs",
            scratch_dir=root.parent / "scratch",
        )
    elif (
        provider != "kaggle"
        or profile_obj.backend != "jax"
        or profile_obj.device != "tpu"
        or profile_obj.precision != "bf16"
        or local_remote is not None
    ):
        raise ValueError(
            "Real bulk translation requires the Kaggle TPU/BF16 profile and private HF"
        )
    if not dev and root.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("bulk artifacts must live outside the checkout")
    deadline = deadline or time.monotonic() + request["session_seconds"]
    work_deadline = deadline - request["finalize_seconds"]
    if time.monotonic() >= work_deadline:
        raise StopWork("not enough session budget remains after setup")
    profile_obj.scratch_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(profile_obj.cache_dir)
    if shutil.disk_usage(profile_obj.scratch_dir).free < (profile_obj.reserve_gb + 1) * 1024**3:
        raise RuntimeError("insufficient disk space for private snapshots")
    code = git_commit()
    if not dev and (not code or code.endswith("-dirty")):
        raise ValueError("bulk inference requires a clean pinned code checkout")
    attempt = uuid.uuid4().hex
    backend = (
        DirectoryRemote(Path(local_remote))
        if dev
        else HFRemote(repo, os.environ.get("HF_TOKEN", ""))
    )
    remote = RemoteSession(backend, campaign_id, attempt, operator)
    remote.acquire(takeover=takeover)
    campaign = saver = run = None
    status, fatal, persisted = "failed", None, False
    try:
        remote.restore(root)
        units = (
            input_units((root / "inputs.json").read_bytes())
            if (root / "campaign.json").exists()
            else prepare_units(request, dev=dev)
        )
        if not dev:
            validate_scope(units, request)
        identity = {
            "campaign_id": campaign_id,
            "code_revision": "fixture" if dev else code,
            "request": {**request, "translator": "stub" if dev else request["translator"]},
            "prompt_version": VERSION,
            "files": prompt_identity(),
            "dev": dev,
        }
        campaign = Campaign(root, identity, units)
        for unit in units:
            for record in unit.records:
                check_record(record)
            if unit.key in campaign.progress["completed"]:
                export_unit(campaign, unit, campaign.translated(unit))
        print(
            f"Restored progress: {campaign.counts()}; "
            f"last durable revision: {remote.verified_revision}",
            flush=True,
        )
        run = Run.open(
            profile_obj,
            f"bulk-{campaign_id}-{attempt[:8]}",
            {"identity": digest(encoded(identity)), "attempt": attempt},
        )
        campaign.event(
            "attempt-started", attempt=attempt, operator=operator, counts=campaign.counts()
        )
        saver = AutoSaver(campaign, remote, request, run=run)
        saver.start()
        # Read back from remote into a fresh root before loading the 27B model.
        with tempfile.TemporaryDirectory(dir=profile_obj.scratch_dir) as temp:
            restored = Path(temp) / "restore"
            if not remote.restore(restored):
                raise RuntimeError("private storage restore probe found no checkpoint")
            probe = Campaign(restored, identity, units)
            if probe.progress["completed"] != campaign.progress["completed"]:
                raise RuntimeError("private storage restore probe differs from saved cases")
        campaign.event("storage-save-restore-probe-passed")
        pending = diagnostic_order(campaign.pending())
        if pending:
            todo = {unit.key: requests(unit) for unit in pending}
            all_messages = [message for messages in todo.values() for message in messages]
            atomic_write(profile_obj.scratch_dir / "bulk-prompts.json", encoded(all_messages))
            os.environ.update(
                {
                    "KODOOM_INPUT_TOKENS": str(request["input_tokens"]),
                    "KODOOM_OUTPUT_TOKENS": str(request["output_tokens"]),
                    "KODOOM_CACHE_TOKENS": str(request["cache_tokens"]),
                    "KODOOM_PROMPTS": str(profile_obj.scratch_dir / "bulk-prompts.json"),
                    "KODOOM_TPU_WARMUP": "0",
                    "KODOOM_DEADLINE": str(work_deadline),
                    "KODOOM_TPU_METRICS": str(campaign.root / "tpu-metrics.json"),
                }
            )
            campaign.event("model-loading")
            if generator_factory:
                generator = generator_factory()
            elif dev:
                generator = StubGenerator()
            else:
                from kodoom.translate.jax import load_generator

                generator = load_generator(bulk=True)
            actual_backend = backend_identity(generator, dev)
            lengths = generator.lengths(all_messages)
            if campaign.production is None:
                indexes = {
                    min(range(len(lengths)), key=lengths.__getitem__),
                    max(range(len(lengths)), key=lengths.__getitem__),
                }
                covered, offset = set(), 0
                for unit in pending:
                    record = unit.records[0]
                    category = (unit.dataset, record.extra.get("workflow", record.question_type))
                    if category not in covered:
                        indexes.add(offset)
                        covered.add(category)
                    offset += len(todo[unit.key])
                sample = [all_messages[i] for i in sorted(indexes)]
                selected = benchmark(generator, sample, request, dev=dev)
                selected["sample_requests"] = sample
                offset = 0
                for unit in pending:
                    if unit.dataset == "helmo" and offset in indexes:
                        reply = selected["benchmark_replies"][sorted(indexes).index(offset)]
                        rebuild(unit, [reply], identity["request"]["translator"])
                    offset += len(todo[unit.key])
                selected["backend"] = actual_backend
                campaign.freeze_production(selected)
                saver.flush()
            elif campaign.production["backend"] != actual_backend:
                raise ValueError("checkpoint, interpreter or accelerator libraries changed")
            batch = campaign.production["batch"]
            print(
                f"Production batch: {batch}; model loaded once; {len(pending)} units pending",
                flush=True,
            )
            started, completed_now = time.monotonic(), 0
            for group_start in range(0, len(pending), 8):
                group = pending[group_start : group_start + 8]
                if max_units is not None:
                    group = group[: max_units - completed_now]
                    if not group:
                        raise StopWork("smoke-unit-limit")
                buffers = {unit.key: [None] * len(todo[unit.key]) for unit in group}
                jobs = [
                    (unit.key, pos, message)
                    for unit in group
                    for pos, message in enumerate(todo[unit.key])
                ]
                job_lengths = generator.lengths([job[2] for job in jobs])
                jobs = [jobs[i] for i in sorted(range(len(jobs)), key=job_lengths.__getitem__)]
                failed, finished = set(), set()
                for offset in range(0, len(jobs), batch):
                    saver.ensure_safe()
                    if time.monotonic() >= work_deadline:
                        raise StopWork("session-budget-exhausted")
                    chunk = jobs[offset : offset + batch]
                    replies = generator.batch([job[2] for job in chunk], batch)
                    if len(replies) != len(chunk):
                        raise RuntimeError("model returned an incorrect batch length")
                    for (key, pos, _), reply in zip(chunk, replies, strict=True):
                        if reply is None:
                            failed.add(key)
                        else:
                            buffers[key][pos] = reply
                    for unit in group:
                        if (
                            unit.key in finished
                            or unit.key in failed
                            or any(text is None for text in buffers[unit.key])
                        ):
                            continue
                        try:
                            translated = rebuild(
                                unit, buffers[unit.key], identity["request"]["translator"]
                            )
                            for record in translated:
                                check_record(record)
                            campaign.complete(unit, translated)
                            export_unit(campaign, unit, translated)
                            finished.add(unit.key)
                            completed_now += 1
                            campaign.event(
                                "case-completed",
                                key=unit.key,
                                records=len(translated),
                                elapsed_seconds=time.monotonic() - started,
                                completed_units_per_hour=completed_now
                                * 3600
                                / max(0.001, time.monotonic() - started),
                            )
                            print(
                                f"Completed {unit.dataset}/{unit.records[0].split} "
                                f"{unit.records[0].source_id}; {completed_now} this attempt",
                                flush=True,
                            )
                        except ValueError as exc:
                            failed.add(unit.key)
                            campaign.fail(unit, type(exc).__name__)
                            campaign.event(
                                "case-validation-failed",
                                key=unit.key,
                                error_type=type(exc).__name__,
                            )
                for unit in group:
                    if unit.key in failed:
                        campaign.fail(unit, "invalid-or-unfinished-reply")
                if max_units is not None and completed_now >= max_units and campaign.pending():
                    raise StopWork("smoke-unit-limit")
            status = "drafts-incomplete-awaiting-review" if campaign.pending() else request["stop"]
        else:
            status = request["stop"]
    except StopWork as exc:
        status = str(exc)
    except Exception as exc:
        status, fatal = "failed", type(exc).__name__
        if campaign:
            campaign.event("attempt-failed", error_type=fatal)
    finally:
        if campaign:
            with campaign.lock:
                campaign.progress["status"] = status
                campaign.save_progress()
            campaign.event("attempt-stopped", status=status, error_type=fatal)
        if saver:
            try:
                saver.stop()
                saver.flush()
                persisted = True
            except Exception as exc:
                fatal = fatal or type(exc).__name__
        try:
            remote.release()
        except Exception:
            fatal = fatal or "WriterReleaseFailed"
        if run:
            run.log("attempt-stopped", status=status, persisted=persisted, error_type=fatal)
        if campaign:
            datadir.update_readme(campaign.root.parent, "kodoom-bulk run")
    result = {
        "status": status,
        "error_type": fatal,
        "persisted": persisted,
        "attempt": attempt,
        "artifact_root": str(root),
        "last_verified_revision": remote.verified_revision,
        "counts": campaign.counts() if campaign else {},
        "hardware_evidence": bool(
            campaign and campaign.production and campaign.production.get("evidence") == "live-tpu"
        ),
    }
    atomic_write(root / "result.json", encoded(result))
    return result


def run_campaign(**kwargs):
    """Run without leaving process-local inference settings in the caller."""
    names = (
        "HF_HOME",
        "KODOOM_INPUT_TOKENS",
        "KODOOM_OUTPUT_TOKENS",
        "KODOOM_CACHE_TOKENS",
        "KODOOM_PROMPTS",
        "KODOOM_TPU_WARMUP",
        "KODOOM_DEADLINE",
        "KODOOM_TPU_METRICS",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        return _run_campaign(**kwargs)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
