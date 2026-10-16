"""Build and independently load the reproducible, CPU-only skills resource.

This is a private release candidate. Publication remains an owner decision.
The frozen record set feeds both formats; generation is repeated before export.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from importlib import resources
from pathlib import Path

from kodoom.generators import GENERATORS
from kodoom.generators.common import DEFAULT_PAIRS_PER_KIND, load_templates
from kodoom.package_import import file_sha
from kodoom.runs import git_commit
from kodoom.schema import Record, RecordError, read_jsonl, write_jsonl
from kodoom.sources import check_record
from kodoom.transport import VERSION, read_parquet, write_parquet


def check_pairs(records: list[Record]) -> int:
    grouped = defaultdict(list)
    ids = set()
    held_by_version = {
        f"{module.NAME}-v{module.VERSION}": {
            t.id
            for group in load_templates(module.NAME, module.SPEC.slots).values()
            for t in group
            if t.held_out
        }
        for module in GENERATORS.values()
    }
    for record in records:
        check_record(record)
        if record.id in ids:
            raise RecordError(f"duplicate skill decision: {record.id}")
        ids.add(record.id)
        grouped[record.source_id].append(record)
        held = held_by_version[record.extra["generator"]]
        if (record.split == "test") != (record.extra["template"] in held):
            raise RecordError(f"held-out template crossed the test boundary: {record.id}")
    for pair_id, pair in grouped.items():
        if len(pair) != 2:
            raise RecordError(f"minimal pair {pair_id} needs exactly two halves")
        a, b = pair
        if a.split != b.split or a.gold == b.gold or a.options != b.options:
            raise RecordError(f"minimal pair {pair_id} breaks split/answer/option rules")
        if {a.extra["pair_role"], b.extra["pair_role"]} != {"a", "b"}:
            raise RecordError(f"minimal pair {pair_id} has invalid roles")
    return len(grouped)


def write_checksums(root: Path) -> None:
    rows = [
        f"{file_sha(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_skills(
    out: Path, *, seed: int = 1234, pairs_per_kind: int = DEFAULT_PAIRS_PER_KIND
) -> dict:
    if out.exists():
        raise RecordError(f"output already exists; choose a new directory: {out}")
    records = []
    generators = []
    for name, module in GENERATORS.items():
        generated = module.generate(seed, pairs_per_kind)
        if generated != module.generate(seed, pairs_per_kind):
            raise RecordError(f"generator {name} is not reproducible")
        records.extend(generated)
        generators.append(
            {
                "name": name,
                "version": module.VERSION,
                "records": len(generated),
                "held_out_templates": sorted(
                    {
                        t.id
                        for group in load_templates(name, module.SPEC.slots).values()
                        for t in group
                        if t.held_out
                    }
                ),
            }
        )
    pairs = check_pairs(records)
    out.mkdir(parents=True)
    templates = out / "provenance/templates"
    templates.mkdir(parents=True)
    for name in GENERATORS:
        raw = resources.files("kodoom").joinpath("generators/templates", f"{name}.toml")
        (templates / f"{name}.toml").write_bytes(raw.read_bytes())
    splits = sorted({r.split for r in records})
    for split in splits:
        selected = [r for r in records if r.split == split]
        jsonl, parquet = out / f"jsonl/fa/{split}.jsonl", out / f"data/fa/{split}.parquet"
        write_jsonl(jsonl, selected)
        write_parquet(parquet, selected)
        if list(read_jsonl(jsonl)) != selected or list(read_parquet(parquet)) != selected:
            raise RecordError(f"skills JSONL/Parquet round trip failed: {split}")
    manifest = {
        "resource": "persian-skills",
        "release_status": "private-candidate",
        "record_transport_version": VERSION,
        "code_revision": git_commit(),
        "seed": seed,
        "pairs_per_kind": pairs_per_kind,
        "decisions": len(records),
        "minimal_pairs": pairs,
        "by_split": dict(Counter(r.split for r in records)),
        "by_type": dict(Counter(r.question_type for r in records)),
        "generators": generators,
        "template_sha256": {p.name: file_sha(p) for p in sorted(templates.glob("*.toml"))},
        "license": "CC0-1.0",
        "human_reviewed": False,
        "checks": [
            "schema and source rules",
            "exact regeneration",
            "unique IDs",
            "two distinct-answer halves per pair",
            "pair split separation",
            "held-out test templates",
            "JSONL/Parquet round trips",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    license_path = Path(__file__).resolve().parents[2] / "LICENSE-DATA"
    if license_path.exists():
        shutil.copyfile(license_path, out / "LICENSE")
    else:
        (out / "LICENSE").write_text("SPDX-License-Identifier: CC0-1.0\n", encoding="utf-8")
    metadata = "\n".join(f"  - split: {s}\n    path: data/fa/{s}.parquet" for s in splits)
    (out / "README.md").write_text(
        f"---\nlanguage: fa\nlicense: cc0-1.0\nconfigs:\n- config_name: fa\n"
        f"  data_files:\n{metadata}\n---\n\n# Persian skills suite\n\n"
        f"Private candidate: {len(records):,} decisions / {pairs:,} minimal pairs.\n"
        "Labels come from code; no translation or model labeling was used.\n"
        "Persian templates have not been human-reviewed. Raw digit forms and\n"
        "spelling variants are preserved. Two templates per question kind are\n"
        "test-only; remaining pairs split 80/10/10 by seeded source_id hashing\n"
        "into train/validation/calibration. Both halves stay together.\n\n"
        "`manifest.json` pins the seed, generator versions, template hashes and\n"
        "code revision. `SHA256SUMS` covers the candidate payload.\n\n"
        "Rebuild with the pinned kodoom checkout: `kodoom release-skills\n"
        f"--profile dev --seed {seed} --pairs-per-kind {pairs_per_kind} --out NEW_DIRECTORY`.\n"
        "No model weights or accelerator are required. JSONL is Record.to_dict();\n"
        "Parquet uses ordered options/gold entries and extra_json. Load with\n"
        "`kodoom.transport.read_parquet` to recover the exact Record objects.\n\n"
        "Run `python examples/load_and_score.py PATH_TO_THIS_RESOURCE` after\n"
        "installing kodoom's dev extra. This demonstrates uniform baseline\n"
        "scoring, not reference-model performance.\n",
        encoding="utf-8",
    )
    example = out / "examples/load_and_score.py"
    example.parent.mkdir()
    example.write_text(SCORING_EXAMPLE, encoding="utf-8")
    write_checksums(out)
    return manifest


SCORING_EXAMPLE = '''"""CPU-only static loading and scoring; no model download."""
import json
import sys
from pathlib import Path

from kodoom.baselines import uniform
from kodoom.evaluation import evaluate
from kodoom.schema import read_jsonl
from kodoom.transport import read_parquet

root = Path(sys.argv[1])
records = list(read_jsonl(root / "jsonl/fa/test.jsonl"))
assert records == list(read_parquet(root / "data/fa/test.parquet"))
predictions = {p.id: p for p in uniform(records)}
summary = evaluate(records, predictions).summary(n_boot=100, seed=1234)
print(json.dumps(summary.to_dict(), indent=2))
'''
