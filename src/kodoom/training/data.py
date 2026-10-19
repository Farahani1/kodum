"""Immutable model subsets and bilingual replay; never edit a source release."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from kodoom.dataset_lock import consume_release, read_selection
from kodoom.normalize import normalize
from kodoom.package_import import file_sha
from kodoom.schema import Option, Record, RecordError, read_jsonl, write_jsonl
from kodoom.sources import check_record

SPLITS = ("train", "validation", "calibration", "test")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def case_key(r: Record) -> str:
    return json.dumps([r.source, r.source_revision, r.source_id], separators=(",", ":"))


def context(r: Record) -> str:
    return r.extra.get("workflow") or r.extra.get("topic") or r.task_family


def model_record(r: Record) -> Record:
    """One normalization function at preparation and inference, leaving gold untouched."""
    return replace(
        r,
        state=normalize(r.state),
        question_text=normalize(r.question_text),
        options=tuple(Option(o.id, normalize(o.text)) for o in r.options),
    )


def grouped_cases(records: list[Record]) -> tuple[dict, list]:
    """Join IDs, identical states/requests and explicit pair groups before splitting.

    High-overlap state shingles are also joined (Jaccard >= .98). A prefix LSH
    index proposes candidates; the exact comparison decides. This is a model
    leakage check, not semantic dataset acceptance or relabeling.
    """
    parent = {case_key(r): case_key(r) for r in records}

    def root(k):
        while k != parent[k]:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def join(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    exact, states, candidates, findings = {}, {}, defaultdict(set), []
    for r in sorted(records, key=lambda x: x.id):
        check_record(r)
        k = case_key(r)
        m = model_record(r)
        signatures = [digest([r.state_lang, m.state])]
        signatures.append(
            digest([m.state, m.question_type, m.question_text, [(o.id, o.text) for o in m.options]])
        )
        for name in ("pair_id", "minimal_pair_id"):
            if r.extra.get(name):
                signatures.append(digest([r.source, name, r.extra[name]]))
        for sig in signatures:
            if sig in exact:
                join(k, exact[sig])
            exact[sig] = k
        state_key = digest([r.state_lang, m.state])
        if state_key in states:
            continue
        words = m.state.split()
        shingles = {" ".join(words[i : i + 5]) for i in range(max(0, len(words) - 4))}
        if len(shingles) >= 20:
            # Minhash bands; all comparisons use exact Jaccard afterwards.
            hashes = sorted(int(hashlib.sha256(x.encode()).hexdigest()[:16], 16) for x in shingles)
            bands = [(r.state_lang, i, h) for i, h in enumerate(hashes[:8])]
            possible = set().union(*(candidates[b] for b in bands))
            for other in sorted(possible):
                other_k, other_shingles = states[other]
                overlap = len(shingles & other_shingles) / len(shingles | other_shingles)
                if overlap >= 0.98:
                    join(k, other_k)
                    findings.append({"a": k, "b": other_k, "jaccard": overlap})
            for band in bands:
                candidates[band].add(state_key)
        states[state_key] = (k, shingles)
    groups = defaultdict(list)
    for r in records:
        groups[root(case_key(r))].append(r)
    for members in groups.values():
        if len({r.split for r in members}) != 1:
            raise RecordError(
                "related/duplicate cases cross released splits; report IDs to hf-data: "
                + ", ".join(sorted({r.source_id for r in members}))
            )
    return dict(groups), findings


def prepare(
    source: Path,
    selection: Path,
    out: Path,
    *,
    seed: int = 1234,
    validation_fraction: float = 0.1,
    calibration_fraction: float = 0.1,
    exclude_contexts: tuple[str, ...] = (),
) -> dict:
    if out.exists():
        raise RecordError(f"output exists; use a new immutable preparation directory: {out}")
    if not 0 < validation_fraction < 1 or not 0 < calibration_fraction < 1:
        raise RecordError("development/calibration fractions must be positive")
    if validation_fraction + calibration_fraction >= 1:
        raise RecordError("fractions leave no training cases")
    records, lock = consume_release(source, read_selection(selection))
    if {r.question_lang for r in records} != {"en", "fa"}:
        raise RecordError("training requires both selected languages for replay")
    groups, near = grouped_cases(records)
    strata = defaultdict(list)
    assigned = {}
    for key, members in groups.items():
        released = members[0].split
        if released == "test":
            assigned[key] = "test"
        elif released != "train":
            raise RecordError(f"unsupported released split: {released}")
        else:
            strata[tuple(sorted({(r.source, context(r)) for r in members}))].append(key)
    for stratum, keys in sorted(strata.items()):
        keys = sorted(keys, key=lambda k: digest([seed, stratum, k]))
        if len(keys) < 3:
            raise RecordError("need at least 3 independent groups per training stratum")
        nv = max(1, round(len(keys) * validation_fraction))
        nc = max(1, round(len(keys) * calibration_fraction))
        if nv + nc >= len(keys):
            raise RecordError("too few groups for requested split fractions")
        for i, key in enumerate(keys):
            assigned[key] = "validation" if i < nv else "calibration" if i < nv + nc else "train"
    output = {s: [] for s in SPLITS}
    excluded = []
    known = {context(r) for r in records}
    if set(exclude_contexts) - known:
        raise RecordError("unknown context exclusion")
    membership = []
    for key, members in sorted(groups.items()):
        split = assigned[key]
        drop = split != "test" and any(context(r) in exclude_contexts for r in members)
        for r in sorted(members, key=lambda x: x.id):
            membership.append(
                {
                    "id": r.id,
                    "case": case_key(r),
                    "group": key,
                    "split": split,
                    "excluded": drop,
                    "context": context(r),
                }
            )
            if drop:
                excluded.append(r.id)
            else:
                output[split].append(replace(r, split=split))
    for split in SPLITS:
        if not output[split]:
            raise RecordError(f"empty required split: {split}")
        for r in output[split]:
            check_record(r)
    # Each training decision must retain its English/Persian pair.
    pairs = defaultdict(set)
    for r in output["train"]:
        pairs[(case_key(r), r.extra.get("source_decision_id", r.id.removesuffix(":fa")))].add(
            r.question_lang
        )
    if any(langs != {"en", "fa"} for langs in pairs.values()):
        raise RecordError("incomplete English/Persian training pair")
    manifest = {
        "schema_version": 1,
        "dataset_lock": lock,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "calibration_fraction": calibration_fraction,
        "exclude_contexts": list(exclude_contexts),
        "memberships": membership,
        "excluded_ids": excluded,
        "near_duplicates": near,
        "normalization": "kodoom.normalize.normalize-at-model-input",
    }
    # Runtime paths do not affect portable subset identity.
    identity = {**manifest, "dataset_lock": {k: v for k, v in lock.items() if k != "source_root"}}
    manifest["fingerprint"] = digest(identity)
    out.mkdir(parents=True)
    manifest["files"] = {}
    manifest["counts"] = {}
    for split, rows in output.items():
        path = out / f"{split}.jsonl"
        write_jsonl(path, rows)
        manifest["files"][path.name] = file_sha(path)
        manifest["counts"][split] = {
            "records": len(rows),
            "cases": len({case_key(r) for r in rows}),
            "languages": dict(Counter(r.question_lang for r in rows)),
            "types": dict(Counter(r.question_type for r in rows)),
        }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_prepared(root: Path) -> tuple[dict, dict[str, list[Record]]]:
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    for name, sha in manifest["files"].items():
        if name not in {f"{s}.jsonl" for s in SPLITS} or file_sha(root / name) != sha:
            raise RecordError("prepared data checksum/path mismatch")
    return manifest, {s: list(read_jsonl(root / f"{s}.jsonl")) for s in SPLITS}


def replay_batches(
    records: list[Record],
    *,
    seed: int,
    epoch: int,
    global_batch: int = 32,
    english_fraction: float = 0.5,
) -> list[list[int]]:
    """Balanced effective batches, later sharded into rank/microstep slices.

    Cycle the smaller pool only as needed, recording repeated rows through the
    deterministic schedule. Every selected language pool is covered each epoch.
    """
    ne = round(global_batch * english_fraction)
    nf = global_batch - ne
    if ne < 1 or nf < 1:
        raise RecordError("replay batch must contain both languages")
    pools = {
        lang: [i for i, r in enumerate(records) if r.question_lang == lang] for lang in ("en", "fa")
    }
    if any(not p for p in pools.values()):
        raise RecordError("English replay requires nonempty language pools")
    rng = random.Random(seed + epoch)
    steps = max(math.ceil(len(pools["en"]) / ne), math.ceil(len(pools["fa"]) / nf))
    queues = {}
    for lang, count in (("en", ne), ("fa", nf)):
        queue = []
        while len(queue) < steps * count:
            cycle = list(pools[lang])
            rng.shuffle(cycle)
            queue.extend(cycle)
        queues[lang] = queue
    batches = []
    for step in range(steps):
        en = queues["en"][step * ne : (step + 1) * ne]
        fa = queues["fa"][step * nf : (step + 1) * nf]
        # Interleave across ranks, avoiding monolingual rank-local streams.
        batch = []
        for i in range(max(ne, nf)):
            if i < nf:
                batch.append(fa[i])
            if i < ne:
                batch.append(en[i])
        batches.append(batch)
    return batches
