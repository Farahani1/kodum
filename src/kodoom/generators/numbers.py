"""Writing numbers and Jalali dates the way Persian text does: the shared part of the skills.

Persian text mixes Persian digits (۱۲۵۰۰۰۰), Arabic-Indic digits (١٢٥٠٠٠٠) and Latin
digits (1250000), with or without thousands separators (1,250,000 or ۱٬۲۵۰٬۰۰۰),
and often as a scaled amount (۱٫۲۵ میلیون); dates as 1405/07/15 or 15 مهر 1405. Used
by the generators; labels always come from the numbers, never from the text.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from kodoom.generators.common import to_script
from kodoom.jalali import MONTH_NAMES, JDate

ARABIC_THOUSANDS = chr(0x066C)  # ٬
ARABIC_DECIMAL = chr(0x066B)  # ٫
SCALES = ((10**9, "میلیارد"), (10**6, "میلیون"), (10**3, "هزار"))


@dataclass(frozen=True)
class Style:
    script: str  # latin, fa or ar
    form: str  # full (1250000) or scaled (1.25 میلیون)
    sep: str  # thousands separator of the full form: "", "," or "٬"
    dec: str  # decimal separator of the scaled form: "." or "٫"


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


def draw_style(rng) -> Style:
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


@dataclass(frozen=True)
class DateStyle:
    script: str  # digit script: latin, fa or ar
    fmt: str  # numeric (1405/07/15) or named (15 مهر 1405)


def draw_date_style(rng) -> DateStyle:
    return DateStyle(
        script=rng.choices(["fa", "latin", "ar"], weights=[45, 45, 10])[0],
        fmt=rng.choice(["numeric", "named"]),
    )


def render_jalali(d: JDate, style: DateStyle) -> str:
    """Write a Jalali date; ``d`` may be an invalid date such as 1405/07/31 on purpose."""
    year, month, day = d
    if style.fmt == "numeric":
        text = f"{year:04d}/{month:02d}/{day:02d}"
    else:
        text = f"{day} {MONTH_NAMES[month - 1]} {year}"
    return to_script(text, style.script)
