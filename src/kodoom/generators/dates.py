"""Jalali-date skills: comparing, weekdays, valid dates, Gregorian equivalents.

Every item is one half of a minimal pair (plan 1.1): a partner that differs in a
single fact and has a different answer. The pair counts as solved only when both
halves are, which separates reading the fact from guessing from the wording. Both
halves share ``source_id``, template, options, digit script and date format, so
exactly one fact differs, and they always land in the same split.

Kinds:

- ``before`` (noul): is the first date before / after the second? Partner: the two
  dates swapped.
- ``weekday`` (choice, 7 options): which day of the week? Partner: the next day.
- ``valid`` (noul): is this a real Jalali date? Partner: a one-step change that
  flips validity (Shahrivar 31 vs Mehr 31; day 30 vs 31; Esfand 30 in a leap year
  vs the next or previous year).
- ``gregorian`` (choice, 5 options): which Gregorian date? Partner: the next day.
  The options are four consecutive days plus a date a year later, so an
  off-by-one answer is a listed mistake.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from kodoom import __version__
from kodoom.generators.common import (
    DEFAULT_PAIRS_PER_KIND,
    MAX_ATTEMPTS,
    GeneratorError,
    Template,
    assign_split,
    item_rng,
    load_templates,
    pick_template,
    to_script,
)
from kodoom.jalali import (
    MONTH_NAMES,
    WEEKDAY_NAMES,
    JDate,
    add_days,
    is_leap,
    is_valid,
    month_length,
    to_gregorian,
    weekday_index,
)
from kodoom.schema import Option, Record, one_hot
from kodoom.sources import get_source

NAME = "jalali-dates"
VERSION = 1  # bump whenever templates or generation logic change
SOURCE = "kodoom/code-labeled"
TASK_FAMILY = "skill-dates"

KINDS = ("before", "weekday", "valid", "gregorian")
QUESTION_TYPE = {"before": "noul", "weekday": "choice", "valid": "noul", "gregorian": "choice"}
SLOTS = {"before": ("a", "b"), "weekday": ("d",), "valid": ("d",), "gregorian": ("d",)}

YEARS = (1385, 1425)  # dates drawn from about 2006 to 2046
YES_NO = (Option("yes", "بله"), Option("no", "خیر"))
WEEKDAY_IDS = ("sat", "sun", "mon", "tue", "wed", "thu", "fri")
WEEKDAY_OPTIONS = tuple(Option(i, n) for i, n in zip(WEEKDAY_IDS, WEEKDAY_NAMES, strict=True))
GREGORIAN_MONTHS = (
    "ژانویه",
    "فوریه",
    "مارس",
    "آوریل",
    "مه",
    "ژوئن",
    "ژوئیه",
    "اوت",
    "سپتامبر",
    "اکتبر",
    "نوامبر",
    "دسامبر",
)


@dataclass(frozen=True)
class Style:
    script: str  # digit script: latin, fa or ar
    fmt: str  # numeric (1405/07/15) or named (15 مهر 1405)


@dataclass(frozen=True)
class Half:
    slots: dict[str, str]  # placeholder -> rendered text
    answer: str  # id of the correct option
    facts: dict[str, Any]  # what the answer was computed from, for audits


@dataclass(frozen=True)
class PairSpec:
    variant: str
    key: tuple  # identifies the facts, so no two pairs repeat them
    options: tuple[Option, ...]
    halves: tuple[Half, Half]


def generate(seed: int, pairs_per_kind: int = DEFAULT_PAIRS_PER_KIND) -> list[Record]:
    """All records: ``pairs_per_kind`` pairs (two records each) for every kind."""
    if pairs_per_kind < 1:
        raise GeneratorError("pairs_per_kind must be at least 1")
    templates = load_templates(NAME, SLOTS)
    records: list[Record] = []
    for kind in KINDS:
        seen: set[tuple] = set()
        for i in range(pairs_per_kind):
            template = pick_template(templates[kind], i)
            for attempt in range(MAX_ATTEMPTS):
                rng = item_rng(seed, NAME, kind, i, attempt)
                style = Style(
                    script=rng.choices(["fa", "latin", "ar"], weights=[45, 45, 10])[0],
                    fmt=rng.choice(["numeric", "named"]),
                )
                spec = _BUILDERS[kind](rng, style, template)
                if spec.key not in seen:
                    break
            else:
                raise GeneratorError(
                    f"{NAME}/{kind}: no unused facts left for item {i}; ask for fewer pairs"
                )
            seen.add(spec.key)
            records += _records(seed, kind, i, template, style, spec)
    return records


# -- builders: one per kind -----------------------------------------------------


def _before(rng, style: Style, template: Template) -> PairSpec:
    a = _random_date(rng)
    low, high = rng.choices([(1, 3), (4, 40), (41, 400)], weights=[3, 4, 3])[0]
    b = add_days(a, rng.randint(low, high))
    if rng.random() < 0.5:
        a, b = b, a
    after = template.relation == "after"

    def half(x: JDate, y: JDate) -> Half:
        truth = x > y if after else x < y
        return Half(
            {"a": _jalali(x, style), "b": _jalali(y, style)},
            "yes" if truth else "no",
            {"a": list(x), "b": list(y), "relation": template.relation},
        )

    gap = abs((to_gregorian(*a) - to_gregorian(*b)).days)
    variant = "gap-short" if gap <= 3 else "gap-medium" if gap <= 40 else "gap-long"
    return PairSpec(variant, tuple(sorted((a, b))), YES_NO, (half(a, b), half(b, a)))


def _weekday(rng, style: Style, template: Template) -> PairSpec:
    first = _random_date(rng)
    options = list(WEEKDAY_OPTIONS)
    rng.shuffle(options)

    def half(d: JDate) -> Half:
        return Half(
            {"d": _jalali(d, style)},
            WEEKDAY_IDS[weekday_index(d)],
            {"date": list(d), "gregorian": to_gregorian(*d).isoformat()},
        )

    halves = [half(first), half(add_days(first, 1))]
    rng.shuffle(halves)
    return PairSpec("next-day", (first,), tuple(options), (halves[0], halves[1]))


def _valid(rng, style: Style, template: Template) -> PairSpec:
    variant = rng.choice(["month-end", "day-31", "esfand-30"])
    year = rng.randint(*YEARS)
    if variant == "month-end":  # Shahrivar has 31 days, Mehr has 30
        pair = [(year, 6, 31), (year, 7, 31)]
    elif variant == "day-31":  # months 7-11 have 30 days
        month = rng.randint(7, 11)
        pair = [(year, month, 30), (year, month, 31)]
    else:  # Esfand has 30 days only in a leap year
        leap = rng.choice([y for y in range(YEARS[0] + 1, YEARS[1]) if is_leap(y)])
        pair = [(leap, 12, 30), (leap + rng.choice([-1, 1]), 12, 30)]
    rng.shuffle(pair)

    def half(d: JDate) -> Half:
        ok = is_valid(*d)
        return Half({"d": _jalali(d, style)}, "yes" if ok else "no", {"date": list(d), "valid": ok})

    halves = (half(pair[0]), half(pair[1]))
    if {h.answer for h in halves} != {"yes", "no"}:  # pragma: no cover - guards the variants
        raise GeneratorError(f"valid/{variant}: {pair} is not a minimal pair")
    return PairSpec(variant, (variant, *sorted(pair)), YES_NO, halves)


def _gregorian(rng, style: Style, template: Template) -> PairSpec:
    first = _random_date(rng)
    g0 = to_gregorian(*first)
    days = [g0 + timedelta(days=n) for n in (-1, 0, 1, 2)] + [g0 + timedelta(days=365)]
    options = [Option(d.isoformat(), _gregorian_text(d, style)) for d in days]
    rng.shuffle(options)

    def half(d: JDate) -> Half:
        g = to_gregorian(*d)
        return Half(
            {"d": _jalali(d, style)}, g.isoformat(), {"date": list(d), "gregorian": g.isoformat()}
        )

    halves = [half(first), half(add_days(first, 1))]
    rng.shuffle(halves)
    return PairSpec("next-day", (first,), tuple(options), (halves[0], halves[1]))


_BUILDERS = {"before": _before, "weekday": _weekday, "valid": _valid, "gregorian": _gregorian}


# -- rendering and assembly -----------------------------------------------------


def _random_date(rng) -> JDate:
    year = rng.randint(*YEARS)
    month = rng.randint(1, 12)
    return year, month, rng.randint(1, month_length(year, month))


def _jalali(d: JDate, style: Style) -> str:
    year, month, day = d
    if style.fmt == "numeric":
        text = f"{year:04d}/{month:02d}/{day:02d}"
    else:
        text = f"{day} {MONTH_NAMES[month - 1]} {year}"
    return to_script(text, style.script)


def _gregorian_text(d: date, style: Style) -> str:
    return to_script(f"{d.day} {GREGORIAN_MONTHS[d.month - 1]} {d.year}", style.script)


def _records(
    seed: int, kind: str, index: int, template: Template, style: Style, spec: PairSpec
) -> list[Record]:
    source_id = f"{NAME}-{kind}-{index:04d}"
    split = assign_split(seed, source_id, template.held_out)
    license_ = get_source(SOURCE).license
    records = []
    for role, half in zip("ab", spec.halves, strict=True):
        records.append(
            Record(
                id=f"{source_id}-{role}",
                source_id=source_id,
                source=SOURCE,
                source_revision=f"{NAME}-v{VERSION}",
                license=license_,
                split=split,
                origin="synthetic",
                task_family=TASK_FAMILY,
                state_lang="fa",
                question_lang="fa",
                state=template.state.format(**half.slots),
                question_type=QUESTION_TYPE[kind],
                question_text=template.question.format(**half.slots),
                options=spec.options,
                gold=one_hot([o.id for o in spec.options], half.answer),
                extra={
                    "kind": kind,
                    "variant": spec.variant,
                    "template": template.id,
                    "register": template.register,
                    "pair_id": source_id,
                    "pair_role": role,
                    "digits": style.script,
                    "date_format": style.fmt,
                    "generator": f"{NAME}-v{VERSION}",
                    "kodoom": __version__,
                    "facts": half.facts,
                },
            )
        )
    return records
