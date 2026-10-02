"""Iranian format skills: mobile numbers and 10-digit postal codes.

Only rules that are certain are used as labels:

- A **mobile number** is valid in exactly three shapes: ``09`` followed by nine
  digits (national), ``+989`` followed by nine digits, or ``00989`` followed by nine
  digits (international). Spaces or dashes between digit groups do not matter.
  A ``0`` after the country code (``+98 0912 ...``) is a common mistake and invalid;
  so are numbers one digit too short or long and numbers not starting with 9.
- A **postal code** is valid when it is exactly ten digits. Stricter official rules
  (digits that may not appear in some positions) are deliberately not tested: this
  suite must never carry a label that a specialist could contest. Valid codes never
  start with 0, which keeps them clear of any such rule.

Kinds (every item is half of a minimal pair, see ``dates.py``): one half is a valid
number or code, the other differs by one digit added, removed or replaced.
"""

from __future__ import annotations

import re

from kodoom.generators.common import (
    DEFAULT_PAIRS_PER_KIND,
    GeneratorSpec,
    Half,
    PairSpec,
    Template,
    build_records,
    to_script,
)
from kodoom.schema import Option, Record

NAME = "iranian-formats"
VERSION = 2  # bump whenever templates or generation logic change
SOURCE = "kodoom/code-labeled"
TASK_FAMILY = "skill-formats"

KINDS = ("mobile", "postal")
QUESTION_TYPE = {"mobile": "noul", "postal": "noul"}
SLOTS = {"mobile": ("n",), "postal": ("n",)}

YES_NO = (Option("yes", "بله"), Option("no", "خیر"))
# Mobile operator prefixes (the three digits after the leading 09): MCI, Irancell, Rightel.
MOBILE_PREFIXES = (
    [f"91{d}" for d in range(10)]
    + [f"93{d}" for d in range(10)]
    + [f"99{d}" for d in range(10)]
    + [f"90{d}" for d in range(6)]
    + ["920", "921", "922"]
)
MOBILE = re.compile(r"(?:0|\+98|0098)9[0-9]{9}")
POSTAL = re.compile(r"[0-9]{10}")
LOOKALIKES = "OolI"  # letters that pass for a 0 or a 1 in a hurry
SEPARATORS = {"none": "", "space": " ", "dash": "-"}


def is_valid_mobile(number: str) -> bool:
    return MOBILE.fullmatch(number) is not None


def is_valid_postal(code: str) -> bool:
    return POSTAL.fullmatch(code) is not None


def generate(seed: int, pairs_per_kind: int = DEFAULT_PAIRS_PER_KIND) -> list[Record]:
    """All records: ``pairs_per_kind`` pairs (two records each) for every kind."""
    return build_records(SPEC, seed, pairs_per_kind)


# -- builders --------------------------------------------------------------------


def _mobile(rng, template: Template) -> PairSpec:
    script, separator = _script(rng), rng.choice(list(SEPARATORS))
    form = rng.choices(["national", "plus", "zeros"], weights=[60, 25, 15])[0]
    prefix = {"national": "0", "plus": "+98", "zeros": "0098"}[form]
    valid = prefix + rng.choice(MOBILE_PREFIXES) + _digits(rng, 7)
    mutation = rng.choice(
        ["short", "long", "not-nine"] + (["extra-zero"] if form == "plus" else [])
    )
    nine = len(prefix)  # index of the leading 9 of the operator code
    if mutation == "short":
        broken = valid[:-1]
    elif mutation == "long":
        broken = valid + rng.choice("0123456789")
    elif mutation == "not-nine":
        broken = valid[:nine] + rng.choice("12345678") + valid[nine + 1 :]
    else:  # a 0 after the country code, as in +98 0912 ...
        broken = valid[:nine] + "0" + valid[nine:]

    def half(number: str, changed: str | None) -> Half:
        # National numbers are grouped as a whole (0912 345 6789), international ones
        # after the country code (+98 912 345 6789).
        if form == "national":
            text = _group(number, "", (4, 3, 4), separator)
        else:
            text = _group(number[len(prefix) :], prefix, (3, 3, 4), separator)
        return Half(
            {"n": to_script(text, script)},
            "yes" if is_valid_mobile(number) else "no",
            {"number": number, "form": form, "mutation": changed},
        )

    halves = [half(valid, None), half(broken, mutation)]
    rng.shuffle(halves)
    extra = {"digits": script, "separator": separator}
    return PairSpec(f"{form}-{mutation}", (valid,), YES_NO, (halves[0], halves[1]), extra)


def _postal(rng, template: Template) -> PairSpec:
    script, separator = _script(rng), rng.choice(list(SEPARATORS))
    valid = rng.choice("123456789") + _digits(rng, 9)
    mutation = rng.choice(["short", "long", "letter"])
    at = rng.randrange(10)
    if mutation == "short":
        broken = valid[:at] + valid[at + 1 :]
    elif mutation == "long":
        broken = valid[:at] + rng.choice("0123456789") + valid[at:]
    else:
        broken = valid[:at] + rng.choice(LOOKALIKES) + valid[at + 1 :]

    def half(code: str, kind: str | None) -> Half:
        text = _group(code, "", (5, 5), separator)
        return Half(
            {"n": to_script(text, script)},
            "yes" if is_valid_postal(code) else "no",
            {"code": code, "mutation": kind},
        )

    halves = [half(valid, None), half(broken, mutation)]
    rng.shuffle(halves)
    extra = {"digits": script, "separator": separator}
    return PairSpec(mutation, (valid,), YES_NO, (halves[0], halves[1]), extra)


_BUILDERS = {"mobile": _mobile, "postal": _postal}

SPEC = GeneratorSpec(
    name=NAME,
    version=VERSION,
    source=SOURCE,
    task_family=TASK_FAMILY,
    slots=SLOTS,
    question_types=QUESTION_TYPE,
    builders=_BUILDERS,
)


# -- helpers ---------------------------------------------------------------------


def _script(rng) -> str:
    return rng.choices(["fa", "latin", "ar"], weights=[40, 40, 20])[0]


def _digits(rng, count: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(count))


def _group(digits: str, prefix: str, sizes: tuple[int, ...], separator: str) -> str:
    """Write ``digits`` in groups of ``sizes``; the last group takes whatever is left, so a
    number with a wrong length only differs in its last group. ``prefix`` goes first."""
    groups, position = [], 0
    for size in sizes[:-1]:
        groups.append(digits[position : position + size])
        position += size
    groups.append(digits[position:])
    glue = SEPARATORS[separator]
    text = glue.join(g for g in groups if g)
    return f"{prefix}{glue}{text}" if prefix else text
