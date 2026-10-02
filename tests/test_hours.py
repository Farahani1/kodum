import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date

import pytest
from amounts import parse

from kodoom.generators import hours
from kodoom.generators.hours import edges, is_open, render_schedule

SEED = 1234
# Spelled here on purpose: the checker must not share the generator's list of names.
DAYS = ["شنبه", "یکشنبه", "دوشنبه", "سه\u200cشنبه", "چهارشنبه", "پنجشنبه", "جمعه"]
MONTHS = [
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
]  # fmt: skip
TIME = re.compile(r"([0-9٠-٩۰-۹]{2}):([0-9٠-٩۰-۹]{2})")


@pytest.fixture(scope="module")
def records():
    return hours.generate(SEED, 100)


def answer(r):
    return next(k for k, v in r.gold.items() if v == 1.0)


def schedule_of(r) -> dict[int, tuple[tuple[int, int], ...]]:
    return {int(d): tuple(map(tuple, ivs)) for d, ivs in r.extra["facts"]["schedule"].items()}


def minutes(token: str) -> int:
    h, m = TIME.fullmatch(token).groups()
    return parse(h) * 60 + parse(m)


def pairs(records, kind):
    grouped = defaultdict(list)
    for r in records:
        if r.extra["kind"] == kind:
            grouped[r.source_id].append(r)
    return list(grouped.values())


# -- the schedule and the visit in the text are the ones the label used ----------------


def parse_schedule(text: str) -> dict[int, tuple[tuple[int, int], ...]]:
    result = {}
    for part in text.split("؛ "):
        times = [minutes(t.group()) for t in TIME.finditer(part)]
        intervals = tuple(zip(times[::2], times[1::2], strict=True))
        label = (
            part[: TIME.search(part).start()].strip()
            if times
            else part.removesuffix("تعطیل").strip()
        )
        if " تا " in label:
            first, last = label.split(" تا ")
            days = range(DAYS.index(first), DAYS.index(last) + 1)
        else:
            days = [DAYS.index(name) for name in label.split(" و ")]
        for d in days:
            result[d] = intervals
    return result


def test_the_rendered_schedule_reads_back_to_the_stored_schedule(records):
    for r in records:
        text = re.search(r": (.+?)\. (?:مراجعه|مشتری|من|درخواست|ما|بیمار)", r.state).group(1)
        assert parse_schedule(text) == schedule_of(r), (r.id, text)


def test_the_visit_in_the_text_matches_the_facts(records):
    for r in records:
        f = r.extra["facts"]
        if r.extra["kind"] == "time":
            m = re.search(r"روز (\S+) ساعت (\S+)", r.state)
            assert DAYS.index(m.group(1)) == f["weekday"], r.id
            assert minutes(m.group(2)) == f["minute"], r.id
        else:
            m = re.search(r"در تاریخ (.+?) ساعت (\S+)", r.state)
            assert minutes(m.group(2)) == f["minute"], r.id
            y, mo, d = parse_jalali(m.group(1))
            assert [y, mo, d] == f["date"], r.id


def parse_jalali(text: str) -> tuple[int, int, int]:
    if "/" in text:
        y, mo, d = (parse(p) for p in text.split("/"))
        return y, mo, d
    day, month, year = text.split(" ")
    return parse(year), MONTHS.index(month) + 1, parse(day)


# -- labels, recomputed independently -------------------------------------------------


def weekday_of(r) -> int:
    f = r.extra["facts"]
    if r.extra["kind"] == "time":
        return f["weekday"]
    return (date.fromisoformat(f["gregorian"]).weekday() + 2) % 7  # Python's own weekday


def test_every_label_matches_an_independent_recomputation(records):
    for r in records:
        f = r.extra["facts"]
        opened = any(a <= f["minute"] < b for a, b in schedule_of(r)[weekday_of(r)])
        assert answer(r) == ("yes" if opened else "no"), r.id


def test_no_visit_falls_exactly_on_an_opening_or_closing_minute(records):
    for r in records:
        assert r.extra["facts"]["minute"] not in edges(schedule_of(r)), r.id


# -- minimal pairs ----------------------------------------------------------------------------


def test_time_pairs_change_only_the_time_or_only_the_weekday(records):
    variants = Counter()
    for a, b in pairs(records, "time"):
        fa, fb = a.extra["facts"], b.extra["facts"]
        assert fa["schedule"] == fb["schedule"]
        same_day, same_time = fa["weekday"] == fb["weekday"], fa["minute"] == fb["minute"]
        assert same_day != same_time, a.source_id  # exactly one of the two differs
        assert (a.extra["variant"] == "weekday") == same_time, a.source_id
        if same_day:  # the time crossed an edge
            schedule = schedule_of(a)
            low, high = sorted((fa["minute"], fb["minute"]))
            assert any(low < e < high for e in edges(schedule)), a.source_id
            assert high - low <= 90
        variants[a.extra["variant"]] += 1
    assert set(variants) == {"open-edge", "close-edge", "weekday"}


def test_date_pairs_are_consecutive_days_at_the_same_time(records):
    for a, b in pairs(records, "date"):
        fa, fb = a.extra["facts"], b.extra["facts"]
        assert fa["schedule"] == fb["schedule"] and fa["minute"] == fb["minute"]
        days = abs((date.fromisoformat(fa["gregorian"]) - date.fromisoformat(fb["gregorian"])).days)
        assert days == 1
        assert answer(a) != answer(b)


def test_answers_are_balanced_and_pairs_have_one_of_each(records):
    for a, b in list(pairs(records, "time")) + list(pairs(records, "date")):
        assert {answer(a), answer(b)} == {"yes", "no"}
    assert Counter(answer(r) for r in records) == {"yes": 200, "no": 200}


def test_schedules_are_varied(records):
    rendered = {json.dumps(r.extra["facts"]["schedule"], sort_keys=True) for r in records}
    assert len(rendered) > 150
    facts = [schedule_of(r) for r in records]
    assert any(len(s[0]) == 2 for s in facts)  # some have a midday break
    assert any(s[6] for s in facts) and any(not s[6] for s in facts)  # Friday closed or open
    assert {r.extra["digits"] for r in records} == {"fa", "latin", "ar"}
    assert {r.extra["date_format"] for r in records if r.extra["kind"] == "date"} == {
        "numeric",
        "named",
    }


# -- schedule text ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("schedule", "text"),
    [
        (
            {**dict.fromkeys(range(5), ((540, 1020),)), 5: ((540, 780),), 6: ()},
            "شنبه تا چهارشنبه 09:00 تا 17:00؛ پنجشنبه 09:00 تا 13:00؛ جمعه تعطیل",
        ),
        (
            {**dict.fromkeys(range(5), ((540, 780), (960, 1200))), 5: (), 6: ()},
            "شنبه تا چهارشنبه 09:00 تا 13:00 و 16:00 تا 20:00؛ پنجشنبه و جمعه تعطیل",
        ),
        (dict.fromkeys(range(7), ((480, 1200),)), "شنبه تا جمعه 08:00 تا 20:00"),
    ],
)
def test_render_schedule(schedule, text):
    assert render_schedule(schedule, "latin") == text


def test_is_open_is_half_open():
    schedule = dict.fromkeys(range(7), ((540, 1020),))
    assert is_open(schedule, 0, 540) and not is_open(schedule, 0, 1020)
    assert not is_open(schedule, 0, 539)


def test_fixed_generation_fingerprint():
    # Changing templates or logic changes this hash: bump hours.VERSION and update it.
    lines = [
        json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) for r in hours.generate(SEED, 5)
    ]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    assert digest == "3702bd15297e8165f68b8af021fe0b20e5ed474b2ef96c12a46d40373e79adc6", digest
