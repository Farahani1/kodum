"""Shared pieces of the code-labeled skill generators (plan 1.1).

A generator draws random facts, asks code (never a model) for the answer, and
renders them through hand-written Persian templates. Everything is a function
of ``(seed, generator version, templates)``, so anyone can regenerate the
published data exactly.
"""

from __future__ import annotations

import hashlib
import random
import string
import tomllib
import unicodedata
from dataclasses import dataclass
from importlib import resources

from kodoom.normalize import clean_orthography

REGISTERS = ("formal", "colloquial")
MIN_TEMPLATES_PER_KIND = 5
HELD_OUT_PER_KIND = 2  # test-only templates per question kind (plan 1.1)
DEFAULT_PAIRS_PER_KIND = 100
MAX_ATTEMPTS = 200  # resamples before a kind is declared exhausted

# Digit scripts a rendered number can use. Real Persian text mixes all three, and
# the benchmark keeps them as written (plan: benchmark data is stored raw).
DIGIT_SCRIPTS = {
    "latin": "0123456789",
    "fa": "".join(chr(0x06F0 + i) for i in range(10)),  # Persian digits
    "ar": "".join(chr(0x0660 + i) for i in range(10)),  # Arabic-Indic digits
}
_TRANSLATE = {s: str.maketrans("0123456789", digits) for s, digits in DIGIT_SCRIPTS.items()}


class GeneratorError(ValueError):
    """A template file or a generator request is invalid."""


@dataclass(frozen=True)
class Template:
    id: str
    kind: str
    register: str
    held_out: bool
    state: str
    question: str
    relation: str | None = None


def to_script(text: str, script: str) -> str:
    """Rewrite the Latin digits of ``text`` in another digit script."""
    return text.translate(_TRANSLATE[script])


def load_templates(
    name: str, slots: dict[str, tuple[str, ...]], *, text: str | None = None
) -> dict[str, tuple[Template, ...]]:
    """Read ``templates/<name>.toml`` and check it against the plan's rules.

    ``slots`` maps each question kind to the placeholders its templates must use.
    Texts are passed through ``clean_orthography``, so the file can be written with
    plain spaces and still produce correct zero-width non-joiners. ``text`` replaces
    the packaged file (for tests).
    """
    if text is None:
        path = resources.files("kodoom").joinpath("generators", "templates", f"{name}.toml")
        text = path.read_text(encoding="utf-8")
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise GeneratorError(f"templates/{name}.toml: invalid TOML: {e}") from e

    by_kind: dict[str, list[Template]] = {kind: [] for kind in slots}
    ids: set[str] = set()
    for entry in raw.get("template", []):
        t = _template(name, entry, slots)
        if t.id in ids:
            raise GeneratorError(f"templates/{name}.toml: duplicate template id {t.id!r}")
        ids.add(t.id)
        by_kind[t.kind].append(t)

    for kind, templates in by_kind.items():
        held = [t for t in templates if t.held_out]
        train = [t for t in templates if not t.held_out]
        where = f"templates/{name}.toml, kind {kind!r}"
        if len(templates) < MIN_TEMPLATES_PER_KIND:
            raise GeneratorError(f"{where}: needs at least {MIN_TEMPLATES_PER_KIND} templates")
        if len(held) != HELD_OUT_PER_KIND:
            raise GeneratorError(f"{where}: exactly {HELD_OUT_PER_KIND} templates must be held out")
        for group, label in ((train, "train"), (held, "held-out")):
            if {t.register for t in group} != set(REGISTERS):
                raise GeneratorError(f"{where}: {label} templates need both registers")
    return {kind: tuple(ts) for kind, ts in by_kind.items()}


def _template(name: str, entry: dict, slots: dict[str, tuple[str, ...]]) -> Template:
    where = f"templates/{name}.toml"
    missing = {"kind", "id", "register", "held_out", "state", "question"} - set(entry)
    if missing:
        raise GeneratorError(f"{where}: template {entry.get('id')!r} lacks {sorted(missing)}")
    tid, kind = entry["id"], entry["kind"]
    if kind not in slots:
        raise GeneratorError(f"{where}: template {tid!r} has unknown kind {kind!r}")
    if entry["register"] not in REGISTERS:
        raise GeneratorError(f"{where}: template {tid!r} register must be one of {REGISTERS}")
    if not isinstance(entry["held_out"], bool):
        raise GeneratorError(f"{where}: template {tid!r} held_out must be true or false")

    state, question = clean_orthography(entry["state"]), clean_orthography(entry["question"])
    if any(unicodedata.category(c) == "Nd" for c in state + question):  # any script
        raise GeneratorError(f"{where}: template {tid!r} contains digits; the generator adds them")
    used = {f for text in (state, question) for _, f, _, _ in string.Formatter().parse(text) if f}
    if used != set(slots[kind]):
        raise GeneratorError(
            f"{where}: template {tid!r} must use exactly the placeholders "
            f"{sorted(slots[kind])}, found {sorted(used)}"
        )
    if any(f"{{{s}}}" not in state for s in slots[kind]):
        raise GeneratorError(f"{where}: template {tid!r} must put every placeholder in the state")
    return Template(
        id=tid,
        kind=kind,
        register=entry["register"],
        held_out=entry["held_out"],
        state=state,
        question=question,
        relation=entry.get("relation"),
    )


def pick_template(templates: tuple[Template, ...], index: int) -> Template:
    """Every fourth item uses a held-out template; the rest cycle through the train ones."""
    train = [t for t in templates if not t.held_out]
    held = [t for t in templates if t.held_out]
    if index % 4 == 3:
        return held[(index // 4) % len(held)]
    return train[(index - index // 4) % len(train)]


def item_rng(seed: int, *parts: object) -> random.Random:
    """A generator that depends only on its arguments (never on order or hash seeds)."""
    return random.Random("|".join(str(p) for p in (seed, *parts)))


def assign_split(seed: int, source_id: str, held_out: bool) -> str:
    """Held-out templates give test items; the others split 80/10/10 by ``source_id``."""
    if held_out:
        return "test"
    bucket = int(hashlib.sha256(f"{seed}|{source_id}".encode()).hexdigest(), 16) % 10
    return {0: "validation", 1: "calibration"}.get(bucket, "train")
