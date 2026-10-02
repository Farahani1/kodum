"""Toman and Rial skills: the two Iranian currency units, one Toman being ten Rials.

Prices in Iran are quoted in both units, and a model that treats "۱٫۲ میلیون
تومان" and "۱٫۲ میلیون ریال" alike is off by a factor of ten. Every label is
computed from integers in Rial, never from the text.

Kinds (every item is half of a minimal pair, see ``dates.py``):

- ``compare`` (noul): is the first amount more / less than the second? The two
  numbers stay fixed and one unit changes, which flips the answer. The builder
  searches the four unit combinations for two that differ in one unit and in the
  answer, instead of trusting a formula.
- ``equal`` (noul): are the two amounts equal, e.g. X Toman and 10X Rial?
  Partner: one unit changed.
- ``convert`` (choice, 6 options): the amount in the other unit. Partner: the same
  number in the other unit. The options are the two correct conversions and four
  typical mistakes: an unconverted number, or the conversion in the wrong direction.

Colloquial templates sometimes spell Toman as «تومن», as people write it.
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import combinations

from kodoom.generators.common import (
    DEFAULT_PAIRS_PER_KIND,
    GeneratorError,
    GeneratorSpec,
    Half,
    PairSpec,
    Template,
    build_records,
)
from kodoom.generators.numbers import Style, draw_style, render
from kodoom.schema import Option, Record

NAME = "toman-rial"
VERSION = 2  # bump whenever templates or generation logic change
SOURCE = "kodoom/code-labeled"
TASK_FAMILY = "skill-currency"

KINDS = ("compare", "equal", "convert")
QUESTION_TYPE = {"compare": "noul", "equal": "noul", "convert": "choice"}
SLOTS = {"compare": ("a", "b"), "equal": ("a", "b"), "convert": ("d",)}

UNITS = ("toman", "rial")
RIAL_PER_UNIT = {"toman": 10, "rial": 1}
UNIT_WORD = {"toman": "تومان", "rial": "ریال"}
COLLOQUIAL_TOMAN = "تومن"
YES_NO = (Option("yes", "بله"), Option("no", "خیر"))


def rial_value(number: int, unit: str) -> int:
    return number * RIAL_PER_UNIT[unit]


def generate(seed: int, pairs_per_kind: int = DEFAULT_PAIRS_PER_KIND) -> list[Record]:
    """All records: ``pairs_per_kind`` pairs (two records each) for every kind."""
    return build_records(SPEC, seed, pairs_per_kind)


# -- builders --------------------------------------------------------------------


def _compare(rng, template: Template) -> PairSpec:
    if template.relation not in ("more", "less"):
        raise GeneratorError(f"template {template.id!r}: relation must be 'more' or 'less'")
    more = template.relation == "more"
    base, word = _base(rng), _toman_word(rng, template)
    factor = rng.randint(2, 8)
    a_number, b_number = (base, factor * base) if rng.random() < 0.5 else (factor * base, base)
    styles = {"a": draw_style(rng), "b": draw_style(rng)}
    configs = [(ua, ub) for ua in UNITS for ub in UNITS]

    def answer(config: tuple[str, str]) -> bool:
        a, b = rial_value(a_number, config[0]), rial_value(b_number, config[1])
        if a == b:  # pragma: no cover - factors 2..8 never make the amounts equal
            raise GeneratorError(f"compare: {a_number}/{b_number} {config} are equal")
        return a > b if more else a < b

    candidates = [
        (c1, c2)
        for c1, c2 in combinations(configs, 2)
        if sum(x != y for x, y in zip(c1, c2, strict=True)) == 1 and answer(c1) != answer(c2)
    ]
    first, second = rng.choice(candidates)  # never empty: see the tests

    def half(config: tuple[str, str]) -> Half:
        return Half(
            {
                "a": _amount(a_number, config[0], styles["a"], word),
                "b": _amount(b_number, config[1], styles["b"], word),
            },
            "yes" if answer(config) else "no",
            {
                "a": a_number,
                "a_unit": config[0],
                "b": b_number,
                "b_unit": config[1],
                "relation": template.relation,
                "styles": _styles(styles),
            },
        )

    halves = [half(first), half(second)]
    rng.shuffle(halves)
    changed = "a" if first[0] != second[0] else "b"
    return PairSpec(
        f"change-{changed}-unit",
        (a_number, b_number),
        YES_NO,
        (halves[0], halves[1]),
        {"toman_word": word},
    )


def _equal(rng, template: Template) -> PairSpec:
    base, word = _base(rng), _toman_word(rng, template)
    # The same money: base Toman = ten times base Rial.
    equal = [(base, "toman"), (10 * base, "rial")]
    if rng.random() < 0.5:
        equal.reverse()
    position = rng.randrange(2)
    other = list(equal)
    other[position] = (equal[position][0], "rial" if equal[position][1] == "toman" else "toman")
    styles = {"a": draw_style(rng), "b": draw_style(rng)}

    def half(amounts) -> Half:
        (a, ua), (b, ub) = amounts
        return Half(
            {"a": _amount(a, ua, styles["a"], word), "b": _amount(b, ub, styles["b"], word)},
            "yes" if rial_value(a, ua) == rial_value(b, ub) else "no",
            {"a": a, "a_unit": ua, "b": b, "b_unit": ub, "styles": _styles(styles)},
        )

    halves = [half(equal), half(other)]
    rng.shuffle(halves)
    return PairSpec(
        f"change-{'ab'[position]}-unit",
        (base,),
        YES_NO,
        (halves[0], halves[1]),
        {"toman_word": word},
    )


def _convert(rng, template: Template) -> PairSpec:
    base, word, style = _base(rng), _toman_word(rng, template), draw_style(rng)

    def option(unit: str, number: int) -> Option:
        return Option(f"{unit}:{number}", _amount(number, unit, style, word))

    options = [
        option("toman", base // 10),  # correct for an amount in Rial
        option("rial", 10 * base),  # correct for an amount in Toman
        option("toman", base),  # the number left unconverted
        option("rial", base),
        option("toman", 10 * base),  # converted in the wrong direction
        option("rial", base // 10),
    ]
    rng.shuffle(options)

    def half(unit: str) -> Half:
        correct = f"toman:{base // 10}" if unit == "rial" else f"rial:{10 * base}"
        return Half(
            {"d": _amount(base, unit, style, word)},
            correct,
            {"value": base, "unit": unit, "styles": _styles({"d": style})},
        )

    halves = [half("rial"), half("toman")]
    rng.shuffle(halves)
    return PairSpec(
        "other-unit", (base,), tuple(options), (halves[0], halves[1]), {"toman_word": word}
    )


_BUILDERS = {"compare": _compare, "equal": _equal, "convert": _convert}

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


def _base(rng) -> int:
    """A whole number of Toman, divisible by ten: n x 10^p with n in 11..999."""
    return rng.randint(11, 999) * 10 ** rng.randint(3, 6)


def _toman_word(rng, template: Template) -> str:
    if template.register == "colloquial" and rng.random() < 0.5:
        return COLLOQUIAL_TOMAN
    return UNIT_WORD["toman"]


def _amount(number: int, unit: str, style: Style, toman_word: str) -> str:
    word = toman_word if unit == "toman" else UNIT_WORD["rial"]
    return f"{render(number, style)} {word}"


def _styles(styles: dict[str, Style]) -> dict[str, dict[str, str]]:
    return {slot: asdict(style) for slot, style in styles.items()}
