"""Test helper: an independent reader of the amounts a generator wrote into a state text.

It shares no code with ``kodoom.generators.numbers``, so a rendering bug cannot hide
behind the same bug in the checker.
"""

import re
from decimal import Decimal

_FA = "".join(chr(0x06F0 + i) for i in range(10))
_AR = "".join(chr(0x0660 + i) for i in range(10))
_TO_LATIN = {ord(c): str(i % 10) for i, c in enumerate(_FA + _AR)}
_DIGIT = "0-9\u0660-\u0669\u06f0-\u06f9"
_SCALE = {"هزار": 10**3, "میلیون": 10**6, "میلیارد": 10**9}

# An amount: digits with separators, optionally a scale word, optionally a unit word.
AMOUNT = re.compile(rf"[{_DIGIT}][{_DIGIT}.,\u066b\u066c]*(?: (?:هزار|میلیون|میلیارد))?")
AMOUNT_WITH_UNIT = re.compile(AMOUNT.pattern + r" (?:تومان|تومن|ریال)")
UNIT_OF = {"تومان": "toman", "تومن": "toman", "ریال": "rial"}


def parse(text: str) -> int:
    number, _, word = text.partition(" ")
    number = number.translate(_TO_LATIN).replace("\u066c", "").replace("\u066b", ".")
    if word:
        return int(Decimal(number.replace(",", "")) * _SCALE[word])
    return int(number.replace(",", ""))


def parse_with_unit(text: str) -> tuple[int, str]:
    amount, _, unit = text.rpartition(" ")
    return parse(amount), UNIT_OF[unit]
