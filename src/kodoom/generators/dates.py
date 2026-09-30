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

from datetime import date, timedelta

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
from kodoom.generators.numbers import DateStyle, draw_date_style, render_jalali
from kodoom.jalali import (
    WEEKDAY_NAMES,
    JDate,
    add_days,
    is_leap,
    is_valid,
    month_length,
    to_gregorian,
    weekday_index,
)
from kodoom.schema import Option, Record

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


def generate(seed: int, pairs_per_kind: int = DEFAULT_PAIRS_PER_KIND) -> list[Record]:
    """All records: ``pairs_per_kind`` pairs (two records each) for every kind."""
    return build_records(SPEC, seed, pairs_per_kind)


# -- builders: one per kind -----------------------------------------------------


def _before(rng, template: Template) -> PairSpec:
    style = draw_date_style(rng)
    a = _random_date(rng)
    low, high = rng.choices([(1, 3), (4, 40), (41, 400)], weights=[3, 4, 3])[0]
    b = add_days(a, rng.randint(low, high))
    if rng.random() < 0.5:
        a, b = b, a
    if template.relation not in ("before", "after"):
        raise GeneratorError(f"template {template.id!r}: relation must be 'before' or 'after'")
    after = template.relation == "after"

    def half(x: JDate, y: JDate) -> Half:
        truth = x > y if after else x < y
        return Half(
            {"a": render_jalali(x, style), "b": render_jalali(y, style)},
            "yes" if truth else "no",
            {"a": list(x), "b": list(y), "relation": template.relation},
        )

    gap = abs((to_gregorian(*a) - to_gregorian(*b)).days)
    variant = "gap-short" if gap <= 3 else "gap-medium" if gap <= 40 else "gap-long"
    return PairSpec(variant, tuple(sorted((a, b))), YES_NO, (half(a, b), half(b, a)), _extra(style))


def _weekday(rng, template: Template) -> PairSpec:
    style = draw_date_style(rng)
    first = _random_date(rng)
    options = list(WEEKDAY_OPTIONS)
    rng.shuffle(options)

    def half(d: JDate) -> Half:
        return Half(
            {"d": render_jalali(d, style)},
            WEEKDAY_IDS[weekday_index(d)],
            {"date": list(d), "gregorian": to_gregorian(*d).isoformat()},
        )

    halves = [half(first), half(add_days(first, 1))]
    rng.shuffle(halves)
    return PairSpec("next-day", (first,), tuple(options), (halves[0], halves[1]), _extra(style))


def _valid(rng, template: Template) -> PairSpec:
    style = draw_date_style(rng)
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
        return Half(
            {"d": render_jalali(d, style)}, "yes" if ok else "no", {"date": list(d), "valid": ok}
        )

    halves = (half(pair[0]), half(pair[1]))
    if {h.answer for h in halves} != {"yes", "no"}:  # pragma: no cover - guards the variants
        raise GeneratorError(f"valid/{variant}: {pair} is not a minimal pair")
    return PairSpec(variant, (variant, *sorted(pair)), YES_NO, halves, _extra(style))


def _gregorian(rng, template: Template) -> PairSpec:
    style = draw_date_style(rng)
    first = _random_date(rng)
    g0 = to_gregorian(*first)
    days = [g0 + timedelta(days=n) for n in (-1, 0, 1, 2)] + [g0 + timedelta(days=365)]
    options = [Option(d.isoformat(), _gregorian_text(d, style)) for d in days]
    rng.shuffle(options)

    def half(d: JDate) -> Half:
        g = to_gregorian(*d)
        return Half(
            {"d": render_jalali(d, style)},
            g.isoformat(),
            {"date": list(d), "gregorian": g.isoformat()},
        )

    halves = [half(first), half(add_days(first, 1))]
    rng.shuffle(halves)
    return PairSpec("next-day", (first,), tuple(options), (halves[0], halves[1]), _extra(style))


_BUILDERS = {"before": _before, "weekday": _weekday, "valid": _valid, "gregorian": _gregorian}

SPEC = GeneratorSpec(
    name=NAME,
    version=VERSION,
    source=SOURCE,
    task_family=TASK_FAMILY,
    slots=SLOTS,
    question_types=QUESTION_TYPE,
    builders=_BUILDERS,
)


# -- rendering and assembly -----------------------------------------------------


def _random_date(rng) -> JDate:
    year = rng.randint(*YEARS)
    month = rng.randint(1, 12)
    return year, month, rng.randint(1, month_length(year, month))


def _gregorian_text(d: date, style: DateStyle) -> str:
    return to_script(f"{d.day} {GREGORIAN_MONTHS[d.month - 1]} {d.year}", style.script)


def _extra(style: DateStyle) -> dict[str, str]:
    return {"digits": style.script, "date_format": style.fmt}
