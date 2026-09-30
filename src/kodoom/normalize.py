"""The one Persian text normalizer (plan 1.2 step 6).

The same function runs on data, at training and at inference, so the model
never sees two spellings of the same text. It is idempotent:
``normalize(normalize(t)) == normalize(t)``.

Run the automatic translation checks (plan 1.2 step 3) on the raw text first,
then normalize: keep-fields are compared byte for byte against the source.
"""

from __future__ import annotations

import re
import unicodedata

ZWNJ = "\u200c"

# Persian (U+06F0-06F9) and Arabic-Indic (U+0660-0669) digits to Latin.
_DIGITS = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIGITS.update({ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")})

_CHARS = {
    ord("ي"): "ی",  # Arabic yeh
    ord("ى"): "ی",  # Arabic alef maksura
    ord("ك"): "ک",  # Arabic kaf
    ord("٪"): "%",  # Arabic percent sign
    ord("\u0640"): None,  # tatweel (kashida), purely typographic
    **{cp: None for cp in range(0x064B, 0x0653)},  # harakat and sukun
    # Invisible direction marks and stray zero-width characters.
    **{ord(c): None for c in "\u200b\u200e\u200f\u2060\ufeff"},
    **{cp: None for cp in range(0x202A, 0x202F)},
    **{cp: None for cp in range(0x2066, 0x206A)},
    # Unusual spaces become a plain space.
    **{ord(c): " " for c in "\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006"},
    **{ord(c): " " for c in "\u2007\u2008\u2009\u200a\u202f\u205f\u3000\t"},
}

_PERSIAN_LETTER = "\u0621-\u063a\u0641-\u064a\u067e\u0686\u0698\u06a9\u06af\u06cc\u06c0"
_L = f"[{_PERSIAN_LETTER}]"

# Arabic decimal and thousands separators, only between digits.
_DECIMAL = re.compile(r"(?<=\d)٫(?=\d)")
_THOUSANDS = re.compile(r"(?<=\d)٬(?=\d)")

_SPACES = re.compile(r" {2,}")
_SPACE_AT_LINE_EDGE = re.compile(r" *\n *")
_ZWNJ_RUN = re.compile(ZWNJ + "{2,}")
# A ZWNJ next to a space, a line break or the text's edge joins nothing.
_ZWNJ_LOOSE = re.compile(rf"(?:(?<=[\s])|^){ZWNJ}+|{ZWNJ}+(?=[\s]|$)")

# Verb prefixes written with a space: "می خواهم" -> "می\u200cخواهم".
_PREFIX = re.compile(rf"(?<!{_L})(ن?می) +(?={_L})")
# Plural suffixes written with a space: "کتاب ها" -> "کتاب\u200cها".
_SUFFIX = re.compile(rf"(?<={_L}) +(ها(?:یی|یم|یت|یش|یمان|یتان|یشان|ی)?)(?!{_L})")


def to_latin_digits(text: str) -> str:
    """Only fold digits; the automatic checks compare numbers with this."""
    text = text.translate(_DIGITS)
    text = _DECIMAL.sub(".", text)
    return _THOUSANDS.sub(",", text)


def normalize(text: str) -> str:
    """Normalize Persian (and mixed) text; English text passes through."""
    text = _fold_presentation_forms(text)
    text = text.translate(_CHARS)
    text = to_latin_digits(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = _PREFIX.sub(lambda m: m.group(1) + ZWNJ, text)
    text = _SUFFIX.sub(lambda m: ZWNJ + m.group(1), text)

    text = _ZWNJ_RUN.sub(ZWNJ, text)
    text = _SPACES.sub(" ", text)
    text = _SPACE_AT_LINE_EDGE.sub("\n", text)
    text = _ZWNJ_LOOSE.sub("", text)
    return text.strip()


def _is_presentation_form(c: str) -> bool:
    return "\ufb50" <= c <= "\ufdff" or "\ufe70" <= c <= "\ufefe"


def _fold_presentation_forms(text: str) -> str:
    """Map Arabic presentation forms (old encodings, PDFs) to base letters."""
    if not any(_is_presentation_form(c) for c in text):
        return text
    return "".join(
        unicodedata.normalize("NFKC", c) if _is_presentation_form(c) else c for c in text
    )
