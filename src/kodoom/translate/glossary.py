"""The English -> Persian glossary and the consistency check that uses it (plan 1.2).

Terms are per workflow on top of a common part, because one English word can need
two Persian ones ("agent" is a support agent in customer service and an AI agent in
agent traces). The check is: where an English term appears in a source text, its
Persian term must appear in that text's translation. That makes a state and its
options name the same thing the same way. Inflected forms pass, because the check
looks for the term as a substring (zero-width non-joiners and spaces ignored).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib import resources

from kodoom.normalize import ZWNJ
from kodoom.translate.checks import Finding

_RESERVED = {"keep", "common"}


class GlossaryError(ValueError):
    """The glossary file is not valid."""


@dataclass(frozen=True)
class Glossary:
    common: dict[str, str]
    by_workflow: dict[str, dict[str, str]]
    keep: tuple[str, ...]
    contexts: dict[str, str]

    def terms(self, workflow: str) -> dict[str, str]:
        """The English -> Persian terms of a workflow (its own override the common ones)."""
        return {**self.common, **self.by_workflow.get(workflow, {})}

    def context(self, workflow: str) -> str:
        """A sentence describing where the texts of a workflow come from, or ""."""
        return self.contexts.get(workflow, "")

    def relevant(self, workflow: str, text: str) -> dict[str, str]:
        """The terms that occur in ``text``."""
        return {en: fa for en, fa in self.terms(workflow).items() if _occurs(en, text)}

    def kept(self, text: str) -> list[str]:
        """The keep-in-English terms that occur in ``text``."""
        return [t for t in self.keep if re.search(_whole(t), text)]

    def check(self, workflow: str, source: str, target: str, where: str = "") -> list[Finding]:
        findings = []
        flat = _squash(target)
        for en, fa in self.relevant(workflow, source).items():
            if _squash(fa) not in flat:
                findings.append(Finding("glossary", where, f"{en!r} should appear as {fa!r}"))
        for term in self.kept(source):
            if term not in target:
                findings.append(Finding("glossary", where, f"{term!r} stays in English"))
        return findings


def _whole(term: str) -> str:
    return rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"


def _squash(text: str) -> str:
    """Text without zero-width non-joiners and spaces: «تأمین کننده» equals «تأمین\u200cکننده»."""
    return re.sub(r"\s+", "", text.replace(ZWNJ, ""))


def _occurs(term: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(term)}s?\b", text, flags=re.IGNORECASE) is not None


def parse(text: str) -> Glossary:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise GlossaryError(f"glossary.toml: invalid TOML: {e}") from e
    keep = raw.get("keep", {}).get("terms", [])
    if not isinstance(keep, list) or not all(isinstance(t, str) and t for t in keep):
        raise GlossaryError("[keep] terms must be a list of non-empty strings")
    contexts = raw.get("context", {})
    if not all(isinstance(k, str) and isinstance(v, str) and v for k, v in contexts.items()):
        raise GlossaryError("[context] needs a workflow name and a sentence for each entry")
    tables = {name: table for name, table in raw.items() if name not in {"keep", "context"}}
    for name, table in tables.items():
        for en, fa in table.items():
            if not isinstance(fa, str) or not fa.strip() or en != en.strip() or not en:
                raise GlossaryError(f"[{name}] {en!r}: needs an English term and a Persian text")
            if en != en.lower():
                raise GlossaryError(f"[{name}] {en!r}: write English terms in lower case")
    common = dict(tables.pop("common", {}))
    return Glossary(common, {k: dict(v) for k, v in tables.items()}, tuple(keep), dict(contexts))


def load() -> Glossary:
    text = resources.files("kodoom").joinpath("translate", "glossary.toml")
    return parse(text.read_text(encoding="utf-8"))
