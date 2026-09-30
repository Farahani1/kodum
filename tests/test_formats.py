import hashlib
import json
import re
from collections import Counter, defaultdict

import pytest

from kodoom.generators import formats
from kodoom.generators.formats import _group

SEED = 1234
_FA = "".join(chr(0x06F0 + i) for i in range(10))
_AR = "".join(chr(0x0660 + i) for i in range(10))
_TO_LATIN = {ord(c): str(i % 10) for i, c in enumerate(_FA + _AR)}
_D = "0-9٠-٩۰-۹"
# The number in a sentence: digits, +, lookalike letters, separators. Persian words have none.
_RUN = re.compile(rf"[{_D}+OolI][{_D}OolI +-]*")
# Owned by the test on purpose: the rules as stated in the module docstring.
MOBILE = re.compile(r"(?:09|\+989|00989)[0-9]{9}")
POSTAL = re.compile(r"[0-9]{10}")


@pytest.fixture(scope="module")
def records():
    return formats.generate(SEED, 100)


def answer(r):
    return next(k for k, v in r.gold.items() if v == 1.0)


def canonical(text: str) -> str:
    return re.sub(r"[ -]", "", _RUN.search(text).group().strip(" -")).translate(_TO_LATIN)


def pairs(records, kind):
    grouped = defaultdict(list)
    for r in records:
        if r.extra["kind"] == kind:
            grouped[r.source_id].append(r)
    return list(grouped.values())


def one_edit_apart(x: str, y: str) -> bool:
    if len(x) == len(y):
        return sum(a != b for a, b in zip(x, y, strict=True)) == 1
    short, long_ = sorted((x, y), key=len)
    return len(long_) - len(short) == 1 and any(
        long_[:i] + long_[i + 1 :] == short for i in range(len(long_))
    )


# -- text, facts and labels -------------------------------------------------------------


def test_the_number_in_the_text_is_the_number_in_the_facts(records):
    for r in records:
        f = r.extra["facts"]
        assert canonical(r.state) == f.get("number", f.get("code")), (r.id, r.state)


def test_every_label_matches_the_stated_rules(records):
    for r in records:
        f = r.extra["facts"]
        rule = MOBILE if r.extra["kind"] == "mobile" else POSTAL
        assert answer(r) == ("yes" if rule.fullmatch(f.get("number", f.get("code"))) else "no")


def test_valid_postal_codes_never_start_with_zero(records):
    for r in records:
        if r.extra["kind"] == "postal" and r.extra["facts"]["mutation"] is None:
            assert r.extra["facts"]["code"][0] != "0"


# -- minimal pairs ------------------------------------------------------------------------


def test_every_pair_is_a_valid_value_and_one_edit_of_it(records):
    for kind in formats.KINDS:
        for a, b in pairs(records, kind):
            fa, fb = a.extra["facts"], b.extra["facts"]
            key = "number" if kind == "mobile" else "code"
            valid, broken = sorted((fa, fb), key=lambda f: f["mutation"] is not None)
            assert valid["mutation"] is None and broken["mutation"] is not None
            assert one_edit_apart(valid[key], broken[key]), (a.source_id, valid, broken)
            assert {answer(a), answer(b)} == {"yes", "no"}
            assert a.extra["separator"] == b.extra["separator"]
            assert a.extra["digits"] == b.extra["digits"]


def test_mutations_and_forms_are_all_covered(records):
    mutations = Counter(
        r.extra["facts"]["mutation"] for r in records if r.extra["facts"]["mutation"]
    )
    assert set(mutations) == {"short", "long", "not-nine", "extra-zero", "letter"}
    forms = {r.extra["facts"]["form"] for r in records if r.extra["kind"] == "mobile"}
    assert forms == {"national", "plus", "zeros"}
    extra_zero = [r for r in records if r.extra["facts"].get("mutation") == "extra-zero"]
    assert extra_zero and all(r.extra["facts"]["form"] == "plus" for r in extra_zero)


def test_written_forms_are_varied_and_raw(records):
    assert {r.extra["digits"] for r in records} == {"fa", "latin", "ar"}
    assert {r.extra["separator"] for r in records} == {"none", "space", "dash"}


def test_only_real_operator_prefixes_are_used(records):
    for r in records:
        f = r.extra["facts"]
        if r.extra["kind"] == "mobile" and f["mutation"] is None:
            operator = f["number"][{"national": 1, "plus": 3, "zeros": 4}[f["form"]] :][:3]
            assert operator in formats.MOBILE_PREFIXES


# -- grouping -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("digits", "prefix", "sizes", "separator", "text"),
    [
        ("09123456789", "", (4, 3, 4), "space", "0912 345 6789"),
        ("09123456789", "", (4, 3, 4), "none", "09123456789"),
        ("9123456789", "+98", (3, 3, 4), "space", "+98 912 345 6789"),
        ("9123456789", "0098", (3, 3, 4), "dash", "0098-912-345-6789"),
        ("0912345678", "", (4, 3, 4), "space", "0912 345 678"),  # one digit short
        ("091234567890", "", (4, 3, 4), "space", "0912 345 67890"),  # one digit long
        ("1234567890", "", (5, 5), "dash", "12345-67890"),
        ("123456789", "", (5, 5), "dash", "12345-6789"),
    ],
)
def test_grouping(digits, prefix, sizes, separator, text):
    assert _group(digits, prefix, sizes, separator) == text


def test_fixed_generation_fingerprint():
    # Changing templates or logic changes this hash: bump formats.VERSION and update it.
    lines = [
        json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True)
        for r in formats.generate(SEED, 5)
    ]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    assert digest == "61ac3aeaa8bf3b2ad23b2ca421d771694f3eeda23129cc4027b1b61e5f352467", digest
