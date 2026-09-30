from datetime import date, timedelta

import pytest

from kodoom import jalali as J

# Dates checked against the official Iranian calendar and widely published conversions.
ANCHORS = [
    ((1300, 1, 1), date(1921, 3, 21)),
    ((1357, 11, 22), date(1979, 2, 11)),  # the revolution
    ((1358, 1, 1), date(1979, 3, 21)),
    ((1399, 1, 1), date(2020, 3, 20)),
    ((1400, 1, 1), date(2021, 3, 21)),
    ((1400, 7, 1), date(2021, 9, 23)),
    ((1403, 1, 1), date(2024, 3, 20)),
    ((1403, 12, 30), date(2025, 3, 20)),  # the last day of a leap year
    ((1404, 1, 1), date(2025, 3, 21)),
    ((1405, 1, 1), date(2026, 3, 21)),
]


@pytest.mark.parametrize(("jdate", "gregorian"), ANCHORS)
def test_anchor_dates(jdate, gregorian):
    assert J.to_gregorian(*jdate) == gregorian
    assert J.from_gregorian(gregorian) == jdate


def test_known_leap_years():
    leap = [y for y in range(1360, 1420) if J.is_leap(y)]
    assert leap == [
        1362,
        1366,
        1370,
        1375,
        1379,
        1383,
        1387,
        1391,
        1395,
        1399,
        1403,
        1408,
        1412,
        1416,
    ]


def test_no_two_leap_years_are_adjacent():
    # The gaps between leap years are 4 or 5: a leap year's neighbours are never leap.
    # The generator relies on this to build leap / non-leap minimal pairs.
    assert not any(J.is_leap(y) and J.is_leap(y + 1) for y in range(1300, 1500))


def test_month_lengths():
    assert [J.month_length(1405, m) for m in range(1, 13)] == [31] * 6 + [30] * 5 + [29]
    assert J.month_length(1403, 12) == 30


@pytest.mark.parametrize(
    ("date_", "valid"),
    [
        ((1405, 6, 31), True),  # Shahrivar has 31 days
        ((1405, 7, 31), False),  # Mehr has 30
        ((1405, 7, 30), True),
        ((1404, 12, 30), False),  # 1404 is not a leap year
        ((1403, 12, 30), True),
        ((1405, 13, 1), False),
        ((1405, 0, 1), False),
        ((1405, 1, 0), False),
        ((1405, 1, 32), False),
    ],
)
def test_is_valid(date_, valid):
    assert J.is_valid(*date_) is valid


def test_invalid_date_cannot_be_converted():
    with pytest.raises(ValueError, match="not a valid Jalali date"):
        J.to_gregorian(1404, 12, 30)


def test_year_outside_the_algorithm_range():
    with pytest.raises(ValueError, match="outside"):
        J.is_leap(5000)
    assert not J.is_valid(5000, 1, 1)


def test_every_day_from_1300_to_1500_round_trips_and_advances_by_one():
    # Continuity is the strongest self-check of the leap rule: across the whole range,
    # each Gregorian day maps to the next Jalali day (or to day 1 of the next month).
    d = J.to_gregorian(1300, 1, 1)
    end = J.to_gregorian(1500, 12, 29)
    previous = J.from_gregorian(d)
    assert previous == (1300, 1, 1)
    while d < end:
        d += timedelta(days=1)
        current = J.from_gregorian(d)
        y, m, day = previous
        if day < J.month_length(y, m):
            assert current == (y, m, day + 1)
        elif m < 12:
            assert current == (y, m + 1, 1)
        else:
            assert current == (y + 1, 1, 1)
        assert J.to_gregorian(*current) == d
        previous = current


def test_add_days_across_month_and_year_ends():
    assert J.add_days((1405, 6, 31), 1) == (1405, 7, 1)
    assert J.add_days((1404, 12, 29), 1) == (1405, 1, 1)
    assert J.add_days((1403, 12, 29), 1) == (1403, 12, 30)
    assert J.add_days((1405, 1, 1), -1) == (1404, 12, 29)


@pytest.mark.parametrize(
    ("jdate", "name"),
    [
        ((1405, 1, 1), "شنبه"),  # 2026-03-21 is a Saturday
        ((1357, 11, 22), "یکشنبه"),  # 1979-02-11 was a Sunday
        ((1404, 1, 1), "جمعه"),  # 2025-03-21 was a Friday
        ((1403, 12, 30), "پنجشنبه"),  # 2025-03-20 was a Thursday
    ],
)
def test_weekday_names(jdate, name):
    assert J.WEEKDAY_NAMES[J.weekday_index(jdate)] == name


def test_names_are_complete():
    assert len(J.MONTH_NAMES) == 12 and len(set(J.MONTH_NAMES)) == 12
    assert len(J.WEEKDAY_NAMES) == 7 and len(set(J.WEEKDAY_NAMES)) == 7
    assert J.WEEKDAY_NAMES[3] == "سه\u200cشنبه"
