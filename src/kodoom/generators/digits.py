"""Digit-form skills: the same amount written in different digit scripts and forms.

Persian text mixes Persian digits (۱۲۵۰۰۰۰), Arabic-Indic digits (١٢٥٠٠٠٠) and
Latin digits (1250000), with or without thousands separators (1,250,000 or
۱٬۲۵۰٬۰۰۰), and often as a scaled amount (۱٫۲۵ میلیون). A model that cannot
read all of these cannot compare amounts. Labels come from integers, never
from the text, so they are exact.

Kinds (every item is half of a minimal pair, see ``dates.py``):

- ``equal`` (noul): are the two amounts equal? They are always written in
  different forms. Partner: one digit of one amount changed.
- ``larger`` (noul): is the first amount more / less than the second? Partner:
  the two amounts swapped, each keeping its own written form.
- ``magnitude`` (score, 5 ordered ranges): the range of one amount. Partner: the
  amount times ten, one range up, written in the same form.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal

from kodoom.generators.common import (
    DEFAULT_PAIRS_PER_KIND,
    GeneratorError,
    GeneratorSpec,
    Half,
    PairSpec,
    Template,
    build_records,
    to_script,
)
from kodoom.schema import Option, Record

NAME = "digit-forms"
VERSION = 1  # bump whenever templates or generation logic change
SOURCE = "kodoom/code-labeled"
TASK_FAMILY = "skill-digits"

KINDS = ("equal", "larger", "magnitude")
QUESTION_TYPE = {"equal": "noul", "larger": "noul", "magnitude": "score"}
SLOTS = {"equal": ("a", "b"), "larger": ("a", "b"), "magnitude": ("d",)}

YES_NO = (Option("yes", "بله"), Option("no", "خیر"))
# Ordered ranges of an amount, written in words so they do not depend on a digit script.
LEVELS = (
    Option("l0", "کمتر از ده هزار"),
    Option("l1", "از ده هزار تا کمتر از صد هزار"),
    Option("l2", "از صد هزار تا کمتر از یک میلیون"),
    Option("l3", "از یک میلیون تا کمتر از ده میلیون"),
    Option("l4", "ده میلیون یا بیشتر"),
)
LEVEL_LIMITS = (10**4, 10**5, 10**6, 10**7)

ARABIC_THOUSANDS = chr(0x066C)  # ٬
ARABIC_DECIMAL = chr(0x066B)  # ٫
SCALES = ((10**9, "میلیارد"), (10**6, "میلیون"), (10**3, "هزار"))


@dataclass(frozen=True)
class Style:
    script: str  # latin, fa or ar
    form: str  # full (1250000) or scaled (1.25 میلیون)
    sep: str  # thousands separator of the full form: "", "," or "٬"
    dec: str  # decimal separator of the scaled form: "." or "٫"


def level(value: int) -> int:
    """The index of the range ``value`` falls in."""
    return sum(value >= limit for limit in LEVEL_LIMITS)


def render(value: int, style: Style) -> str:
    """Write ``value`` in ``style``; small values fall back to the full form."""
    scaled = next(((s, w) for s, w in SCALES if value >= s), None)
    if style.form == "scaled" and scaled:
        scale, word = scaled
        text = format((Decimal(value) / scale).normalize(), "f").replace(".", style.dec)
        return f"{to_script(text, style.script)} {word}"
    text = str(value)
    if style.sep:
        groups = []
        while text:
            groups.append(text[-3:])
            text = text[:-3]
        text = style.sep.join(reversed(groups))
    return to_script(text, style.script)


def generate(seed: int, pairs_per_kind: int = DEFAULT_PAIRS_PER_KIND) -> list[Record]:
    """All records: ``pairs_per_kind`` pairs (two records each) for every kind."""
    return build_records(SPEC, seed, pairs_per_kind)


# -- builders --------------------------------------------------------------------


def _equal(rng, template: Template) -> PairSpec:
    n, p = _mantissa(rng), rng.randint(3, 7)
    other = _change_one_digit(rng, n)
    style_a = _style(rng)
    style_b = _style(rng)
    while style_b == style_a:  # the two amounts must be written differently
        style_b = _style(rng)
    fixed = rng.choice("ab")  # the slot that keeps the true amount in both halves
    styles = {"a": style_a, "b": style_b}

    def half(mantissa: int) -> Half:
        values = {fixed: n * 10**p, ("b" if fixed == "a" else "a"): mantissa * 10**p}
        return Half(
            {s: render(values[s], styles[s]) for s in "ab"},
            "yes" if values["a"] == values["b"] else "no",
            {"a": values["a"], "b": values["b"], "styles": _styles(styles)},
        )

    halves = [half(n), half(other)]
    rng.shuffle(halves)
    variant = f"{style_a.form}-{style_b.form}"
    return PairSpec(variant, (*sorted((n, other)), p), YES_NO, (halves[0], halves[1]))


def _larger(rng, template: Template) -> PairSpec:
    n, p = _mantissa(rng), rng.randint(3, 7)
    other = _change_one_digit(rng, n)
    numbers = [(n * 10**p, _style(rng)), (other * 10**p, _style(rng))]
    rng.shuffle(numbers)
    if template.relation not in ("more", "less"):
        raise GeneratorError(f"template {template.id!r}: relation must be 'more' or 'less'")
    more = template.relation == "more"

    def half(first, second) -> Half:
        truth = first[0] > second[0] if more else first[0] < second[0]
        return Half(
            {"a": render(*first), "b": render(*second)},
            "yes" if truth else "no",
            {
                "a": first[0],
                "b": second[0],
                "relation": template.relation,
                "styles": _styles({"a": first[1], "b": second[1]}),
            },
        )

    forms = "-".join(sorted(s.form for _, s in numbers))
    return PairSpec(
        forms, (*sorted((n, other)), p), YES_NO, (half(*numbers), half(numbers[1], numbers[0]))
    )


def _magnitude(rng, template: Template) -> PairSpec:
    while True:
        n, p = _mantissa(rng), rng.randint(2, 5)
        if 10**3 <= n * 10**p < 10**7:  # so that ten times as much is still one range up
            break
    style = _style(rng)

    def half(value: int) -> Half:
        return Half(
            {"d": render(value, style)},
            LEVELS[level(value)].id,
            {"value": value, "styles": _styles({"d": style})},
        )

    halves = [half(n * 10**p), half(n * 10 ** (p + 1))]
    rng.shuffle(halves)
    return PairSpec(f"times-ten-{style.form}", (n, p), LEVELS, (halves[0], halves[1]))


_BUILDERS = {"equal": _equal, "larger": _larger, "magnitude": _magnitude}

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


def _mantissa(rng) -> int:
    return rng.randint(11, 999)


def _change_one_digit(rng, n: int) -> int:
    """``n`` with one digit replaced by a different one (same length, no leading zero)."""
    digits = list(str(n))
    position = rng.randrange(len(digits))
    choices = [d for d in "0123456789" if d != digits[position] and (position or d != "0")]
    digits[position] = rng.choice(choices)
    return int("".join(digits))


def _style(rng) -> Style:
    script = rng.choices(["fa", "latin", "ar"], weights=[40, 35, 25])[0]
    seps = {
        "latin": ["", ","],
        "fa": ["", ",", ARABIC_THOUSANDS],
        "ar": ["", ARABIC_THOUSANDS, ","],
    }
    return Style(
        script=script,
        form=rng.choices(["full", "scaled"], weights=[60, 40])[0],
        sep=rng.choice(seps[script]),
        dec="." if script == "latin" else rng.choice([".", ARABIC_DECIMAL]),
    )


def _styles(styles: dict[str, Style]) -> dict[str, dict[str, str]]:
    return {slot: asdict(style) for slot, style in styles.items()}
