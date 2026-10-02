"""Business-hours skills: is the shop open at this time, on this weekday or date?

A weekly schedule such as «شنبه تا چهارشنبه ۹:۰۰ تا ۱۷:۰۰؛ پنجشنبه ۹:۰۰ تا ۱۳:۰۰؛
جمعه تعطیل» is read against a visit. The Iranian week starts on Saturday and Friday
is the usual day off, some shops close at noon on Thursday, and bazaar hours often
have a midday break. Labels come from the schedule as data (weekday -> open
intervals in minutes), never from the text.

A visit never falls exactly on an opening or closing minute, so nobody has to decide
whether the boundary counts.

Kinds (every item is half of a minimal pair, see ``dates.py``):

- ``time`` (noul): a weekday and a time. Partner: the same schedule and one change:
  the time moves across an opening or closing edge, or the weekday changes to one
  where the shop is closed at that time.
- ``date`` (noul): a Jalali date and a time; the weekday must be derived from the
  date. Partner: the next day, chosen so that the answer changes.
"""

from __future__ import annotations

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
from kodoom.generators.numbers import draw_date_style, render_jalali
from kodoom.jalali import WEEKDAY_NAMES, JDate, add_days, month_length, to_gregorian, weekday_index
from kodoom.schema import Option, Record

NAME = "business-hours"
VERSION = 2  # bump whenever templates or generation logic change
SOURCE = "kodoom/code-labeled"
TASK_FAMILY = "skill-hours"

KINDS = ("time", "date")
QUESTION_TYPE = {"time": "noul", "date": "noul"}
SLOTS = {"time": ("s", "v"), "date": ("s", "v")}

YES_NO = (Option("yes", "بله"), Option("no", "خیر"))
YEARS = (1400, 1425)
MAX_TRIES = 60

# weekday index (0 = Saturday .. 6 = Friday) -> open intervals in minutes since midnight,
# each [open, close): open at ``open``, closed from ``close`` on.
Schedule = dict[int, tuple[tuple[int, int], ...]]


def is_open(schedule: Schedule, weekday: int, minute: int) -> bool:
    return any(start <= minute < end for start, end in schedule[weekday])


def edges(schedule: Schedule) -> set[int]:
    return {m for intervals in schedule.values() for interval in intervals for m in interval}


def generate(seed: int, pairs_per_kind: int = DEFAULT_PAIRS_PER_KIND) -> list[Record]:
    """All records: ``pairs_per_kind`` pairs (two records each) for every kind."""
    return build_records(SPEC, seed, pairs_per_kind)


# -- builders --------------------------------------------------------------------


def _time(rng, template: Template) -> PairSpec:
    script = _script(rng)
    schedule = _draw_schedule(rng)
    variant = rng.choice(["open-edge", "close-edge", "weekday"])
    for _ in range(MAX_TRIES):
        visits = (
            _weekday_visits(rng, schedule)
            if variant == "weekday"
            else _edge_visits(rng, schedule, variant)
        )
        if visits:
            break
    else:  # pragma: no cover - the schedules always allow at least one of the three
        raise GeneratorError(f"hours/time: no {variant} pair found")

    text = render_schedule(schedule, script)

    def half(visit: tuple[int, int]) -> Half:
        weekday, minute = visit
        return Half(
            {"s": text, "v": f"{WEEKDAY_NAMES[weekday]} ساعت {_hhmm(minute, script)}"},
            "yes" if is_open(schedule, weekday, minute) else "no",
            {"schedule": _schedule_facts(schedule), "weekday": weekday, "minute": minute},
        )

    halves = [half(visits[0]), half(visits[1])]
    rng.shuffle(halves)
    key = (_schedule_key(schedule), *sorted(visits))
    return PairSpec(variant, key, YES_NO, (halves[0], halves[1]), {"digits": script})


def _date(rng, template: Template) -> PairSpec:
    style = draw_date_style(rng)
    schedule = _draw_schedule(rng)
    for _ in range(MAX_TRIES):
        minute = 15 * rng.randint(9 * 4, 21 * 4)
        if minute in edges(schedule):
            continue
        day = _random_date(rng)
        pair = next(
            (
                (d, add_days(d, 1))
                for d in (add_days(day, k) for k in range(14))
                if is_open(schedule, weekday_index(d), minute)
                != is_open(schedule, weekday_index(add_days(d, 1)), minute)
            ),
            None,
        )
        if pair:
            break
    else:  # pragma: no cover - a week always has an open and a closed day at some time
        raise GeneratorError("hours/date: no day pair found")

    text = render_schedule(schedule, style.script)

    def half(d: JDate) -> Half:
        return Half(
            {"s": text, "v": f"{render_jalali(d, style)} ساعت {_hhmm(minute, style.script)}"},
            "yes" if is_open(schedule, weekday_index(d), minute) else "no",
            {
                "schedule": _schedule_facts(schedule),
                "date": list(d),
                "gregorian": to_gregorian(*d).isoformat(),
                "minute": minute,
            },
        )

    halves = [half(pair[0]), half(pair[1])]
    rng.shuffle(halves)
    key = (_schedule_key(schedule), minute, pair[0])
    extra = {"digits": style.script, "date_format": style.fmt}
    return PairSpec("next-day", key, YES_NO, (halves[0], halves[1]), extra)


_BUILDERS = {"time": _time, "date": _date}

SPEC = GeneratorSpec(
    name=NAME,
    version=VERSION,
    source=SOURCE,
    task_family=TASK_FAMILY,
    slots=SLOTS,
    question_types=QUESTION_TYPE,
    builders=_BUILDERS,
)


# -- schedules -------------------------------------------------------------------


def _draw_schedule(rng) -> Schedule:
    start = rng.choice([480, 510, 540, 570, 600])  # 8:00 .. 10:00
    end = rng.choice([960, 1020, 1080, 1200, 1260, 1320])  # 16:00 .. 22:00
    week: tuple[tuple[int, int], ...] = ((start, end),)
    if rng.random() < 0.4:  # a midday break, as in many bazaar shops
        break_start = rng.choice([750, 780, 810])
        break_end = break_start + rng.choice([120, 150, 180])
        if break_end <= end - 90:
            week = ((start, break_start), (break_end, end))
    thursday = week if rng.random() < 0.4 else ((start, rng.choice([780, 840])),)
    friday: tuple[tuple[int, int], ...] = ()
    if rng.random() < 0.2:  # some places open on Friday evening
        friday = ((rng.choice([960, 1020, 1080]), rng.choice([1260, 1320, 1380])),)
    return {0: week, 1: week, 2: week, 3: week, 4: week, 5: thursday, 6: friday}


def render_schedule(schedule: Schedule, script: str) -> str:
    """Consecutive days with the same hours are written together."""
    groups: list[tuple[list[int], tuple[tuple[int, int], ...]]] = []
    for day in range(7):
        if groups and groups[-1][1] == schedule[day]:
            groups[-1][0].append(day)
        else:
            groups.append(([day], schedule[day]))
    parts = []
    for days, intervals in groups:
        if len(days) == 1:
            label = WEEKDAY_NAMES[days[0]]
        elif len(days) == 2:
            label = f"{WEEKDAY_NAMES[days[0]]} و {WEEKDAY_NAMES[days[1]]}"
        else:
            label = f"{WEEKDAY_NAMES[days[0]]} تا {WEEKDAY_NAMES[days[-1]]}"
        hours = " و ".join(f"{_hhmm(a, script)} تا {_hhmm(b, script)}" for a, b in intervals)
        parts.append(f"{label} {hours or 'تعطیل'}")
    return "؛ ".join(parts)


def _edge_visits(rng, schedule: Schedule, variant: str):
    """One weekday, two times on either side of an opening or closing edge."""
    weekday = rng.choice([d for d in range(7) if schedule[d]])
    start, end = rng.choice(schedule[weekday])
    delta = rng.choice([10, 15, 20, 30, 45])
    if variant == "open-edge":
        closed, opened = start - delta, start + delta
    else:
        opened, closed = end - delta, end + delta
    ok = (
        0 <= closed < 1440
        and is_open(schedule, weekday, opened)
        and not is_open(schedule, weekday, closed)
        and not {opened, closed} & edges(schedule)
    )
    return ((weekday, opened), (weekday, closed)) if ok else None


def _weekday_visits(rng, schedule: Schedule):
    """One time, two weekdays: the shop is open on one and closed on the other."""
    weekday = rng.choice([d for d in range(7) if schedule[d]])
    start, end = rng.choice(schedule[weekday])
    minute = 15 * rng.randint(-(-(start + 15) // 15), (end - 15) // 15)
    closed = [d for d in range(7) if d != weekday and not is_open(schedule, d, minute)]
    if not closed or minute in edges(schedule):
        return None
    return (weekday, minute), (rng.choice(closed), minute)


def _random_date(rng) -> JDate:
    year, month = rng.randint(*YEARS), rng.randint(1, 12)
    return year, month, rng.randint(1, month_length(year, month))


def _script(rng) -> str:
    return rng.choices(["fa", "latin", "ar"], weights=[40, 40, 20])[0]


def _hhmm(minute: int, script: str) -> str:
    return to_script(f"{minute // 60:02d}:{minute % 60:02d}", script)


def _schedule_facts(schedule: Schedule) -> dict[str, list[list[int]]]:
    return {str(d): [list(i) for i in intervals] for d, intervals in schedule.items()}


def _schedule_key(schedule: Schedule) -> tuple:
    return tuple((d, schedule[d]) for d in range(7))
