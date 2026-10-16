import json
import subprocess
import sys
from dataclasses import replace

import pytest

from kodoom.generators import dates
from kodoom.package_import import file_sha
from kodoom.schema import RecordError, read_jsonl
from kodoom.skill_release import build_skills, check_pairs
from kodoom.transport import read_parquet


def test_small_release_copies_templates_preserves_pairs_and_checksums(tmp_path):
    root = tmp_path / "skills"
    manifest = build_skills(root, seed=1234, pairs_per_kind=8)
    assert manifest["decisions"] == 224
    assert manifest["minimal_pairs"] == 112
    assert len(manifest["generators"]) == 5
    assert len(manifest["template_sha256"]) == 5
    assert json.loads((root / "manifest.json").read_text("utf-8")) == manifest
    records = []
    for split in manifest["by_split"]:
        selected = list(read_jsonl(root / f"jsonl/fa/{split}.jsonl"))
        assert selected == list(read_parquet(root / f"data/fa/{split}.parquet"))
        records.extend(selected)
    assert check_pairs(records) == 112
    scored = subprocess.run(
        [sys.executable, str(root / "examples/load_and_score.py"), str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(scored.stdout)["n"] == manifest["by_split"]["test"]
    for line in (root / "SHA256SUMS").read_text("utf-8").splitlines():
        digest, name = line.split("  ", 1)
        assert file_sha(root / name) == digest
    with pytest.raises(RecordError, match="already exists"):
        build_skills(root)


def test_pair_acceptance_rejects_missing_halves_and_split_leakage():
    a, b = dates.generate(1234, 1)[:2]
    with pytest.raises(RecordError, match="exactly two"):
        check_pairs([a])
    with pytest.raises(RecordError, match="split/answer/option"):
        check_pairs([a, replace(b, split="validation")])
