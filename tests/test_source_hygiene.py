"""Invisible characters in source code are unreadable in review.

Persian text makes them easy to paste in by accident (ZWNJ, RLM, NBSP).
Write them as escapes instead, e.g. ZWNJ as a backslash-u-200c escape.
"""

import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(
    p for d in ("src", "tests") for p in (ROOT / d).rglob("*") if p.suffix in {".py", ".toml"}
)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_invisible_characters(path):
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for c in line:
            if c != " " and unicodedata.category(c) in {"Cf", "Zs", "Zl", "Zp", "Co"}:
                name = unicodedata.name(c, f"U+{ord(c):04X}")
                pytest.fail(f"{path.name}:{n}: invisible character {name}; use an escape")
