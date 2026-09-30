"""No dataset, prediction, checkpoint or run output may be tracked by git.

The owner keeps all data on Drive and publishes nothing until they decide to
(plan: Principles). ``.gitignore`` already covers the data folders; this test
catches a file added with ``git add -f`` or under an unexpected name, and fails CI.
"""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_SUFFIXES = {".jsonl", ".parquet", ".csv", ".tsv", ".arrow", ".safetensors", ".pt", ".bin"}
DATA_FOLDERS = {"data", "runs", "models"}


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return out.splitlines()


def test_no_data_files_are_tracked():
    bad = [
        name
        for name in tracked_files()
        if Path(name).suffix.lower() in DATA_SUFFIXES or Path(name).parts[0] in DATA_FOLDERS
    ]
    assert not bad, f"data must stay on Drive, never in git: {bad}"


def test_the_ignore_file_covers_the_data_folders():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for folder in ("/data/", "/runs/", "/models/"):
        assert folder in ignore
