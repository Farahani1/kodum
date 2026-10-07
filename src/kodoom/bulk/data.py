"""Pinned English inputs and deterministic coverage of helmo types/topics."""

from __future__ import annotations

import random
import re
import tomllib
from collections import defaultdict
from pathlib import Path

from kodoom import helmo, typed_decisions
from kodoom.bulk.state import Unit, digest, encoded
from kodoom.translate.pipeline import cases

ROOT = Path(__file__).resolve().parents[3]


def load_request(path: Path | None = None) -> dict:
    request = tomllib.loads((path or ROOT / "workflows/tpu-bulk.toml").read_text("utf-8"))
    expected = {
        "schema_version",
        "stage",
        "plan",
        "translator",
        "model",
        "backend",
        "precision",
        "batch_sizes",
        "input_tokens",
        "output_tokens",
        "cache_tokens",
        "seed",
        "helmo_limit",
        "typed_train_cases",
        "typed_test_cases",
        "session_seconds",
        "finalize_seconds",
        "upload_seconds",
        "max_unsaved_seconds",
        "memory_reserve_bytes",
        "stop",
    }
    if set(request) != expected or (
        request["schema_version"] != 1
        or request["stage"] != "bulk-translation"
        or request["model"] != "google/gemma-3/flax/gemma3-27b-it/1"
        or request["translator"] != "gemma3-27b-tpu-bf16"
        or request["backend"] != "jax"
        or request["precision"] != "bf16"
        or request["batch_sizes"] != [1, 2, 4, 8]
        or request["stop"] != "drafts-complete-awaiting-review"
    ):
        raise ValueError("unsupported bulk request/model/batch policy")
    integers = expected - {
        "stage",
        "plan",
        "translator",
        "model",
        "backend",
        "precision",
        "batch_sizes",
        "stop",
    }
    if any(type(request[k]) is not int or request[k] <= 0 for k in integers):
        raise ValueError("bulk budgets/counts must be positive integers")
    if (
        request["input_tokens"] + request["output_tokens"] > request["cache_tokens"]
        or request["finalize_seconds"] >= request["session_seconds"]
        or request["upload_seconds"] >= request["max_unsaved_seconds"]
        or request["max_unsaved_seconds"] > request["finalize_seconds"]
    ):
        raise ValueError("inconsistent token, upload or session budgets")
    if (request["typed_train_cases"], request["typed_test_cases"]) != (1200, 400):
        raise ValueError("bulk typed scope must be the complete 1200/400 cases")
    if not 1500 <= request["helmo_limit"] <= 2000:
        raise ValueError("helmo scope must remain in the planned starting range")
    return request


def validate_campaign_id(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", value):
        raise ValueError("campaign ID must be 3-64 lowercase letters/digits/hyphens")
    return value


def select_helmo(records, limit: int, seed: int):
    groups = defaultdict(list)
    for record in records:
        groups[(record.question_type, record.extra["topic"])].append(record)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    by_type = defaultdict(list)
    for kind, topic in sorted(groups):
        by_type[kind].append((kind, topic))
    for topics in by_type.values():
        rng.shuffle(topics)
    selected = []
    cursors = dict.fromkeys(by_type, 0)
    while len(selected) < limit and any(groups.values()):
        for kind, topics in sorted(by_type.items()):
            available = [topic for topic in topics if groups[topic]]
            if available and len(selected) < limit:
                slot = cursors[kind] % len(available)
                selected.append(groups[available[slot]].pop())
                cursors[kind] += 1
    if len(selected) != limit:
        raise ValueError("not enough helmo records for the frozen selection")
    return selected


def prepare_units(request: dict, *, download=None, dev: bool = False) -> list[Unit]:
    if dev:
        from kodoom.bulk.fixtures import fixture_units

        return fixture_units()
    if download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download
    typed = typed_decisions.load_records(download, limit=None)
    units = []
    for split in ("train", "test"):
        groups = cases(typed[split])
        if len(groups) != request[f"typed_{split}_cases"]:
            raise ValueError(f"typed {split} count differs from the frozen full scope")
        counts = {workflow: 0 for workflow in typed_decisions.WORKFLOWS}
        for group in groups:
            counts[group[0].extra["workflow"]] += 1
            units.append(Unit("typed-decisions", tuple(group)))
        if set(counts.values()) != {request[f"typed_{split}_cases"] // 4}:
            raise ValueError("typed workflow counts differ from source inventory")
    all_helmo = eligible_helmo(helmo.load_records(download, limit=None), typed["test"])
    selected = select_helmo(all_helmo, request["helmo_limit"], request["seed"])
    units.extend(Unit("helmo", (record,)) for record in selected)
    # Early diagnostic units cover all workflows and helmo question types.
    buckets = defaultdict(list)
    for unit in units:
        r = unit.records[0]
        buckets[(unit.dataset, r.split, r.extra.get("workflow", r.question_type))].append(unit)
    balanced = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                balanced.append(buckets[key].pop(0))
    return balanced


def eligible_helmo(records, typed_test):
    """Drop exact repeated records and exact held-out states before sampling.

    This is an exact-text guard; semantic overlap still needs the research
    leakage review before these drafts can enter a training mix.
    """

    def state_key(text):
        return " ".join(text.casefold().split())

    held_out = {state_key(record.state) for record in typed_test}
    seen, result = set(), []
    for record in records:
        state = state_key(record.state)
        key = (
            state,
            state_key(record.question_text),
            tuple((o.id, state_key(o.text)) for o in record.options),
        )
        if key not in seen and state not in held_out:
            seen.add(key)
            result.append(record)
    return result


def prompt_identity() -> dict:
    files = [
        "translate/rules.py",
        "translate/glossary.toml",
        "translate/hf.py",
        "translate/pipeline.py",
        "bulk/prompts.py",
        "translate/jax.py",
        "bulk/data.py",
    ]
    return {name: digest((ROOT / "src/kodoom" / name).read_bytes()) for name in files}


def input_units(data: bytes) -> list[Unit]:
    import json

    from kodoom.schema import Record

    return [
        Unit(row["dataset"], tuple(Record.from_dict(r) for r in row["records"]))
        for row in json.loads(data)
    ]


def validate_scope(units, request):
    counts = defaultdict(int)
    for unit in units:
        record = unit.records[0]
        expected = typed_decisions.REVISION if unit.dataset == "typed-decisions" else helmo.REVISION
        if any(r.source_revision != expected for r in unit.records):
            raise ValueError("frozen inputs use an unexpected source revision")
        counts[(unit.dataset, record.split)] += 1
    if dict(counts) != {
        ("typed-decisions", "train"): request["typed_train_cases"],
        ("typed-decisions", "test"): request["typed_test_cases"],
        ("helmo", "train"): request["helmo_limit"],
    }:
        raise ValueError("frozen input counts differ from the approved campaign scope")


def selection_hash(units: list[Unit]) -> str:
    return digest(encoded([u.to_dict() for u in units]))
