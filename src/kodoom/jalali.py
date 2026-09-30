"""The Jalali (Solar Hijri) calendar: leap years, validity, conversion, weekdays.

Pure Python, no dependencies. Every label of the Jalali-date skills is computed
here (plan 1.1), so it is checked against known anchor dates and against the
Gregorian calendar in ``tests/test_jalali.py``.

Leap years follow the astronomical calendar, not the simple 33-year rule: the
year begins at the Tehran-noon equinox. The algorithm is the "breaks" method of
Behrooz Parhami and Kazimierz M. Borkowski (as in jalaali-js), valid for
Jalali years -61 to 3177. Dates are ``(year, month, day)`` tuples.
"""

from __future__ import annotations

from datetime import date, timedelta

JDate = tuple[int, int, int]

MONTH_NAMES = (
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
)

# Saturday first: the Iranian week starts on Saturday. Index = days since Saturday.
# سه\u200cشنبه is written with a zero-width non-joiner, as an escape so it stays visible.
WEEKDAY_NAMES = (
    "شنبه",
    "یکشنبه",
    "دوشنبه",
    "سه\u200cشنبه",
    "چهارشنبه",
    "پنجشنبه",
    "جمعه",
)

_BREAKS = (
    -61, 9, 38, 199, 426, 686, 756, 818, 1111, 1181,
    1210, 1635, 2060, 2097, 2192, 2262, 2324, 2394, 2456, 3178,
)  # fmt: skip
MIN_YEAR, MAX_YEAR = _BREAKS[0], _BREAKS[-1] - 1


def _div(a: int, b: int) -> int:
    """Integer division truncating toward zero, as the published algorithm assumes."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _mod(a: int, b: int) -> int:
    return a - _div(a, b) * b


def _year_info(jy: int) -> tuple[bool, int]:
    """Return (is_leap, day in March on which 1 Farvardin of ``jy`` falls)."""
    if not MIN_YEAR <= jy <= MAX_YEAR:
        raise ValueError(f"Jalali year {jy} is outside {MIN_YEAR}..{MAX_YEAR}")
    gy = jy + 621
    leap_j = -14
    jp = _BREAKS[0]
    jump = 0
    for jm in _BREAKS[1:]:
        jump = jm - jp
        if jy < jm:
            break
        leap_j += _div(jump, 33) * 8 + _div(_mod(jump, 33), 4)
        jp = jm
    n = jy - jp

    # Leap years since AD 621 in the Jalali calendar, and the same in the Gregorian one.
    leap_j += _div(n, 33) * 8 + _div(_mod(n, 33) + 3, 4)
    if _mod(jump, 33) == 4 and jump - n == 4:
        leap_j += 1
    leap_g = _div(gy, 4) - _div((_div(gy, 100) + 1) * 3, 4) - 150
    march = 20 + leap_j - leap_g

    # Years since the last leap year: 0 means ``jy`` itself is a leap year.
    if jump - n < 6:
        n = n - jump + _div(jump + 4, 33) * 33
    r = _mod(_mod(n + 1, 33) - 1, 4)
    return (r == 0), march


def is_leap(year: int) -> bool:
    """Whether Esfand of ``year`` has 30 days."""
    return _year_info(year)[0]


def month_length(year: int, month: int) -> int:
    """Days in a Jalali month: 31 for months 1-6, 30 for 7-11, 29 or 30 for Esfand."""
    if not 1 <= month <= 12:
        raise ValueError(f"month {month} is not in 1..12")
    if month <= 6:
        return 31
    if month <= 11:
        return 30
    return 30 if is_leap(year) else 29


def is_valid(year: int, month: int, day: int) -> bool:
    """Whether ``year/month/day`` is a real Jalali date."""
    if not MIN_YEAR <= year <= MAX_YEAR or not 1 <= month <= 12:
        return False
    return 1 <= day <= month_length(year, month)


def nowruz(year: int) -> date:
    """The Gregorian date of 1 Farvardin of ``year``."""
    return date(year + 621, 3, _year_info(year)[1])


def to_gregorian(year: int, month: int, day: int) -> date:
    if not is_valid(year, month, day):
        raise ValueError(f"{year}/{month}/{day} is not a valid Jalali date")
    days = sum(month_length(year, m) for m in range(1, month)) + day - 1
    return nowruz(year) + timedelta(days=days)


def from_gregorian(d: date) -> JDate:
    year = d.year - 621
    start = nowruz(year)
    if d < start:  # before this Jalali year's Nowruz: still the previous year
        year -= 1
        start = nowruz(year)
    days = (d - start).days
    for month in range(1, 13):
        length = month_length(year, month)
        if days < length:
            return year, month, days + 1
        days -= length
    raise AssertionError(f"{d} does not fall inside Jalali year {year}")  # pragma: no cover


def add_days(jdate: JDate, days: int) -> JDate:
    return from_gregorian(to_gregorian(*jdate) + timedelta(days=days))


def weekday_index(jdate: JDate) -> int:
    """Days since Saturday: 0 is Saturday (شنبه), 6 is Friday (جمعه)."""
    return (to_gregorian(*jdate).weekday() + 2) % 7
