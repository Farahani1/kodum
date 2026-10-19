"""Full encoder/head fine-tuning with deterministic DDP replay and private saves."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from kodoom.config import load_profile
from kodoom.package_import import file_sha
from kodoom.predictions import Prediction, write_predictions
from kodoom.runs import Run, git_commit
from kodoom.schema import RecordError
from kodoom.training import laya
from kodoom.training.data import context, load_prepared, replay_batches
from kodoom.training.settings import Settings


def rank_indices(full: list[int], *, rank: int, world: int) -> list[int]:
    return [i for start in range(rank * 2, len(full), world * 2) for i in full[start : start + 2]]


def retention_ok(metrics: dict, original: dict, settings: Settings) -> bool:
    en, base = metrics["en"], original["en"]
    return en["accuracy"] >= base["accuracy"] - settings.english_accuracy_drop and en[
        "loss"
    ] <= max(base["loss"], 1e-8) * (1 + settings.english_loss_rise)


def soft_ce(logits, targets):
    import torch

    return -(targets.float() * torch.log_softmax(logits.float(), dim=-1)).sum(-1)


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())


def evaluate_model(
    model,
    records,
    items,
    tok,
    common,
    device,
    *,
    rank=0,
    world=1,
    microbatch=2,
    amp=False,
    predictions_path=None,
):
    """Shard without padding: each validation item is scored exactly once."""
    import torch
    import torch.distributed as dist

    model.eval()
    sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    predictions = []
    indexes = list(range(rank, len(records), world))
    with torch.no_grad():
        for start in range(0, len(indexes), microbatch):
            selected = indexes[start : start + microbatch]
            batch = laya.collate([items[i] for i in selected], tok, common, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits, _ = model(**{k: batch[k] for k in laya.FORWARD_KEYS})
            loss = soft_ce(logits, batch["target"])
            probs = logits.float().softmax(-1)
            targets = batch["target"]
            # All tied maximum-gold options count as correct, consistently for all models.
            correct = (
                targets.gather(1, probs.argmax(-1)[:, None]).squeeze(1) >= targets.max(-1).values
            )
            brier = ((probs - targets) ** 2).sum(-1)
            for j, i in enumerate(selected):
                r, item = records[i], items[i]
                for key in (
                    r.question_lang,
                    f"{r.question_lang}/{r.question_type}",
                    f"{r.question_lang}/workflow/{context(r)}",
                ):
                    values = sums[key]
                    for n, value in enumerate(
                        (float(loss[j]), float(correct[j]), float(brier[j]), 1.0)
                    ):
                        values[n] += value
                if predictions_path:
                    p = probs[j, : len(item["option_ids"])].cpu().tolist()
                    predictions.append(
                        Prediction(
                            r.id,
                            dict(zip(item["option_ids"], p, strict=True)),
                            r.gold,
                            extra={
                                "language": r.question_lang,
                                "question_type": r.question_type,
                                "context": context(r),
                                "source_id": r.source_id,
                                "truncation": item["truncation"],
                            },
                        )
                    )
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, {k: v for k, v in sums.items()})
    else:
        gathered = [dict(sums)]
    merged = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for shard in gathered:
        for key, values in shard.items():
            merged[key] = [a + b for a, b in zip(merged[key], values, strict=True)]
    if predictions_path:
        if world > 1:
            raise RecordError("prediction export uses a single inference process")
        write_predictions(predictions_path, predictions)
    return {
        k: {
            "loss": v[0] / v[3],
            "accuracy": v[1] / v[3],
            "brier": v[2] / v[3],
            "records": int(v[3]),
        }
        for k, v in merged.items()
    }


def encode_rows(rows, tok, common, cfg):
    return [
        laya.encode(r, tok, common, max_len=cfg["max_len"], head_max_len=cfg["head_max_len"])
        for r in rows
    ]


def encoding_summary(rows, items):
    summary = {}
    for row, item in zip(rows, items, strict=True):
        entry = summary.setdefault(
            f"{row.question_lang}/{row.question_type}",
            {
                "records": 0,
                "state_truncated": 0,
                "state_tokens": 0,
                "state_tokens_used": 0,
                "state_tokens_dropped": 0,
            },
        )
        stats = item["truncation"]
        entry["records"] += 1
        entry["state_truncated"] += int(stats["truncated"])
        for key in ("state_tokens", "state_tokens_used", "state_tokens_dropped"):
            entry[key] += stats.get(key, 0)
    return summary


def _optimizer(model, settings):
    import torch

    groups = defaultdict(list)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr = settings.encoder_lr if name.startswith("encoder.") else settings.head_lr
        decay = 0.0 if p.ndim <= 1 or name.endswith("bias") else settings.weight_decay
        groups[(lr, decay)].append(p)
    return torch.optim.AdamW(
        [{"params": ps, "lr": lr, "weight_decay": decay} for (lr, decay), ps in groups.items()]
    )


def train(
    prepared: Path,
    settings: Settings,
    *,
    profile_name: str,
    run_id: str,
    hub_repo: str | None = None,
    smoke: bool = False,
    preflight: bool = False,
    runs_dir: Path | None = None,
) -> dict:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    profile = load_profile(profile_name, runs_dir=runs_dir)
    rank, world, local_rank = (
        int(os.environ.get(k, d))
        for k, d in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"))
    )
    if smoke:
        if profile.device != "cpu" or world != 1:
            raise RecordError("tiny smoke requires CPU profile and one process")
        settings = replace(
            settings,
            world_size=1,
            microbatch=2,
            accumulation=1,
            max_len=128,
            head_max_len=80,
            validate_every=1,
            save_every=1,
            epochs=1,
            patience=10,
        )
    elif world != settings.world_size or world != 2 or profile.device != "cuda":
        raise RecordError(
            "main training/preflight requires explicit Kaggle CUDA profile and two ranks"
        )
    torch.set_num_threads(settings.threads)
    if not smoke:
        if torch.cuda.device_count() != 2:
            raise RecordError("expected two visible T4 GPUs; no silent CPU/single-GPU fallback")
        if any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
            raise RecordError("selected recipe requires Kaggle T4 x2")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    device = torch.device("cpu" if smoke else f"cuda:{local_rank}")
    try:
        return _train(
            prepared,
            settings,
            profile,
            run_id,
            device,
            rank,
            world,
            hub_repo,
            smoke,
            preflight,
            DistributedDataParallel,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(
    prepared, settings, profile, run_id, device, rank, world, hub_repo, smoke, preflight, ddp_class
):
    import torch
    import torch.distributed as dist

    seed_all(settings.seed)
    manifest, splits = load_prepared(prepared)
    if smoke:
        # Shape/optimization test only; never interpreted as a measured real model.
        for split in ("train", "validation"):
            splits[split] = [
                r
                for lang in ("fa", "en")
                for r in [
                    x for x in splits[split] if x.question_lang == lang and len(x.options) <= 4
                ][:8]
            ]
    # Rank 0 populates shared cache before other ranks enter download/load.
    if rank:
        dist.barrier()
    model, tok, cfg, common = laya.load(profile.cache_dir, tiny=smoke)
    if world > 1 and not rank:
        dist.barrier()
    cfg.update(max_len=settings.max_len, head_max_len=settings.head_max_len)
    model.to(device)
    train_rows, dev_rows = splits["train"], splits["validation"]
    train_items = encode_rows(train_rows, tok, common, cfg)
    dev_items = encode_rows(dev_rows, tok, common, cfg)
    identity = {
        "settings": settings.to_dict(),
        "prepared_fingerprint": manifest["fingerprint"],
        "prepared_manifest_sha": file_sha(prepared / "manifest.json"),
        "model_id": laya.MODEL_ID,
        "model_revision": laya.MODEL_REVISION,
        "code_revision": laya.CODE_REVISION,
        "common_sha": laya.COMMON_SHA,
        "smoke": smoke,
        "preflight": preflight,
        "hub_repo": hub_repo,
        "code_commit": git_commit(),
    }
    run = None
    if not rank:
        run = Run.open(profile, run_id, identity)
        (run.dir / "training-config.json").write_text(
            json.dumps(identity, indent=2), encoding="utf-8"
        )
        shutil.copyfile(prepared / "manifest.json", run.dir / "data-manifest.json")
        (run.dir / "encoding-stats.json").write_text(
            json.dumps(
                {
                    "train": encoding_summary(train_rows, train_items),
                    "validation": encoding_summary(dev_rows, dev_items),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        run.log(
            "coverage",
            train=len(train_rows),
            validation=len(dev_rows),
            state_truncated=sum(i["truncation"]["truncated"] for i in train_items),
            world_size=world,
            device=str(device),
            torch=torch.__version__,
        )
        if not smoke:
            from kodoom.training.remote import private_api

            if not hub_repo:
                raise RecordError("a configured private Hugging Face backup repo is required")
            private_api(hub_repo)
    if world > 1:
        dist.barrier()
    # Distinct dropout streams, identical initial parameters and replay schedule.
    seed_all(settings.seed + rank)
    optimizer = _optimizer(model, settings)
    batches_per_epoch = len(
        replay_batches(
            train_rows,
            seed=settings.seed,
            epoch=0,
            global_batch=settings.global_batch,
            english_fraction=settings.english_fraction,
        )
    )
    total_steps = settings.epochs * batches_per_epoch
    warmup = max(1, round(total_steps * settings.warmup_fraction))

    def schedule(step):
        return min(
            (step + 1) / warmup, max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, offset, step, bad, stale, best = 0, 0, 0, 0, 0, math.inf
    original = None
    # All ranks inspect only the shared complete checkpoint after rank-0 recovery.
    latest = profile.runs_dir / run_id / "latest"
    if world > 1:
        dist.barrier()
    saved_rng = None
    if (latest / "state.pt").exists():
        # Only load trusted, checksum-verified own snapshots (torch pickle state).
        recorded_sha = json.loads((latest / "state-sha.json").read_text("utf-8"))["sha256"]
        if file_sha(latest / "state.pt") != recorded_sha:
            raise RecordError("local checkpoint checksum mismatch")
        state = torch.load(latest / "state.pt", map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise RecordError("resume identity differs from model/data/code/settings")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, offset, step = state["epoch"], state["offset"], state["step"]
        original, bad, stale, best = state["original"], state["bad"], state["stale"], state["best"]
        saved_rng = state["rng"][rank]
        del state
    if original is None:
        original = evaluate_model(
            model, dev_rows, dev_items, tok, common, device, rank=rank, world=world, amp=not smoke
        )
        if run:
            run.log("starting-development", metrics=original)
    wrapped = ddp_class(model, device_ids=[device.index]) if world > 1 else model
    if saved_rng is not None:
        restore_rng(saved_rng)
    last_save, last_sync = time.monotonic(), time.monotonic()
    stop, status = (
        False,
        "smoke_complete" if smoke else "preflight_complete" if preflight else "training_complete",
    )
    max_updates = step + 2 if smoke or preflight else total_steps

    durable_revision = None

    def save(epoch, next_offset):
        nonlocal last_save, last_sync, durable_revision
        rngs = [None] * world
        if world > 1:
            dist.all_gather_object(rngs, rng_state())
        else:
            rngs = [rng_state()]
        if run:

            def write(path):
                torch.save(
                    {
                        "identity": identity,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "offset": next_offset,
                        "step": step,
                        "original": original,
                        "bad": bad,
                        "stale": stale,
                        "best": best,
                        "rng": rngs,
                    },
                    path / "state.pt",
                )

                (path / "state-sha.json").write_text(
                    json.dumps({"sha256": file_sha(path / "state.pt")}), encoding="utf-8"
                )

            run.save_latest(step, write)
            # Verify deserialization and state identity before declaring a snapshot usable.
            restored = torch.load(
                run.latest.path / "state.pt", map_location="cpu", weights_only=False
            )
            if restored["step"] != step or restored["identity"] != identity:
                raise RecordError("checkpoint round-trip failed")
            del restored
            if hub_repo:
                try:
                    from kodoom.training.remote import sync

                    revision = sync(run, hub_repo)
                    durable_revision = revision
                    run.log("durable-checkpoint", revision=revision, step=step)
                except Exception as exc:
                    run.log("sync-failed", error_type=type(exc).__name__, step=step)
                    # Local state remains complete; preflight requires an actual durable save.
                    if preflight:
                        raise RecordError(
                            "private checkpoint upload failed; local state preserved"
                        ) from None
        last_save = last_sync = time.monotonic()
        if world > 1:
            dist.barrier()

    for epoch in range(start_epoch, settings.epochs):
        batches = replay_batches(
            train_rows,
            seed=settings.seed,
            epoch=epoch,
            global_batch=settings.global_batch,
            english_fraction=settings.english_fraction,
        )
        for batch_index in range(offset if epoch == start_epoch else 0, len(batches)):
            full = batches[batch_index]
            # Allocate bilingual pairs together even with microbatch=1.
            local = rank_indices(full, rank=rank, world=world)
            wrapped.train()
            optimizer.zero_grad(set_to_none=True)
            sums = torch.zeros(4, device=device)
            started = time.monotonic()
            for micro in range(settings.accumulation):
                indexes = local[micro * settings.microbatch : (micro + 1) * settings.microbatch]
                b = laya.collate([train_items[i] for i in indexes], tok, common, device)
                sync_context = (
                    wrapped.no_sync()
                    if world > 1 and micro < settings.accumulation - 1
                    else contextlib.nullcontext()
                )
                with sync_context:
                    with torch.autocast(device.type, dtype=torch.float16, enabled=not smoke):
                        logits, _ = wrapped(**{k: b[k] for k in laya.FORWARD_KEYS})
                        losses = soft_ce(logits, b["target"])
                        loss = losses.sum() * world / settings.global_batch
                    if not torch.isfinite(loss):
                        raise RecordError("nonfinite training loss")
                    scaler.scale(loss).backward()
                for j, i in enumerate(indexes):
                    n = 0 if train_rows[i].question_lang == "fa" else 2
                    sums[n] += losses[j].detach()
                    sums[n + 1] += 1
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            if (smoke or preflight) and not torch.isfinite(grad_norm):
                raise RecordError("preflight/smoke gradients are nonfinite")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
            if not skipped:
                scheduler.step()
            step += 1
            if world > 1:
                dist.all_reduce(sums)
            if run:
                run.log(
                    "train",
                    step=step,
                    epoch=epoch,
                    batch_index=batch_index,
                    fa_loss=float(sums[0] / sums[1]),
                    en_loss=float(sums[2] / sums[3]),
                    fa_count=int(sums[1]),
                    en_count=int(sums[3]),
                    lr=[g["lr"] for g in optimizer.param_groups],
                    grad_norm=float(grad_norm),
                    scale=scaler.get_scale(),
                    skipped=skipped,
                    examples_per_second=settings.global_batch
                    / max(time.monotonic() - started, 1e-9),
                    peak_memory=(torch.cuda.max_memory_allocated(device) if not smoke else 0),
                    types=dict(Counter(train_rows[i].question_type for i in full)),
                    workflows=dict(Counter(context(train_rows[i]) for i in full)),
                )
            at_end = batch_index == len(batches) - 1
            validate = step % settings.validate_every == 0 or at_end or step >= max_updates
            improved = False
            if validate:
                # Evaluate the unwrapped module: no uneven DDP forward collectives.
                metrics = evaluate_model(
                    model,
                    dev_rows,
                    dev_items,
                    tok,
                    common,
                    device,
                    rank=rank,
                    world=world,
                    amp=not smoke,
                )
                eligible = retention_ok(metrics, original, settings)
                bad = 0 if eligible else bad + 1
                improved = eligible and metrics["fa"]["loss"] < best
                stale = 0 if improved else stale + 1
                if improved:
                    best = metrics["fa"]["loss"]
                    if run:
                        run.save_best(
                            step,
                            best,
                            lambda path, saved_step=step, saved_metrics=metrics: laya.save_artifact(
                                path,
                                model,
                                tok,
                                cfg,
                                {**identity, "step": saved_step, "development": saved_metrics},
                            ),
                        )
                if run:
                    run.log(
                        "validation",
                        step=step,
                        metrics=metrics,
                        eligible=eligible,
                        improved=improved,
                        consecutive_retention_failures=bad,
                    )
                if bad >= 2:
                    stop, status = True, "english_retention_stop"
                elif stale >= settings.patience:
                    stop, status = True, "early_stop"
            next_epoch, next_offset = (epoch + 1, 0) if at_end else (epoch, batch_index + 1)
            # Rank 0 chooses wall-clock triggers; all ranks must enter the same collectives.
            flags = [
                step % settings.save_every == 0
                or improved
                or at_end
                or stop
                or step >= max_updates
                or time.monotonic() - last_save >= settings.snapshot_seconds,
                time.monotonic() - last_sync >= settings.log_sync_seconds,
            ]
            if world > 1:
                dist.broadcast_object_list(flags, src=0)
            if flags[0]:
                save(next_epoch, next_offset)
            elif flags[1] and run and hub_repo:
                try:
                    from kodoom.training.remote import sync

                    revision = sync(run, hub_repo, logs_only=True)
                    run.log("durable-logs", revision=revision, step=step)
                except Exception as exc:
                    run.log("sync-failed", error_type=type(exc).__name__, step=step)
                last_sync = time.monotonic()
            if step >= max_updates:
                stop = True
            if stop:
                break
        if stop:
            break
    summary = {
        "status": status,
        "step": step,
        "best_fa_loss": best if math.isfinite(best) else None,
        "smoke": smoke,
        "preflight": preflight,
        "test_evaluated": False,
        "durable_revision": durable_revision,
        "run_id": run_id,
    }
    if run:
        run.log("training-ended", **summary)
        # Retain latest for the required fresh-session restore check; Run.finish deletes it.
        run.meta["status"] = "awaiting-evaluation"
        (run.dir / "run.json").write_text(json.dumps(run.meta, indent=2), encoding="utf-8")
        (run.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if hub_repo:
            try:
                from kodoom.training.remote import sync

                sync(run, hub_repo, logs_only=True)
            except Exception as exc:
                run.log("sync-failed", error_type=type(exc).__name__, step=step)
    return summary
