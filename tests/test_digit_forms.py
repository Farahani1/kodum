import hashlib
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal

import pytest

from kodoom.generators import digits, numbers
from kodoom.generators.common import DIGIT_SCRIPTS, GeneratorError, load_templates
from kodoom.generators.digits import level
from kodoom.generators.numbers import Style, render

SEED = 1234


@pytest.fixture(scope="module")
def records():
    return digits.generate(SEED, 100)


def answer(r):
    return next(k for k, v in r.gold.items() if v == 1.0)


def style(script="latin", form="full", sep="", dec="."):
    return Style(script, form, sep, dec)


# -- rendering -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "kwargs", "text"),
    [
        (1250000, dict(), "1250000"),
        (1250000, dict(sep=","), "1,250,000"),
        (1000, dict(sep=","), "1,000"),
        (123, dict(sep=","), "123"),
        (1250000, dict(script="fa", sep=","), "۱,۲۵۰,۰۰۰"),
        (1250000, dict(script="fa", sep="٬"), "۱٬۲۵۰٬۰۰۰"),
        (1250000, dict(script="ar", sep="٬"), "١٬٢٥٠٬٠٠٠"),
        (1250000, dict(form="scaled"), "1.25 میلیون"),
        (1250000, dict(form="scaled", script="fa", dec="٫"), "۱٫۲۵ میلیون"),
        (250000, dict(form="scaled"), "250 هزار"),
        (2500000000, dict(form="scaled"), "2.5 میلیارد"),
        (950, dict(form="scaled"), "950"),  # too small to scale: written in full
    ],
)
def test_render(value, kwargs, text):
    assert render(value, style(**kwargs)) == text


def test_level_boundaries():
    assert [level(v) for v in (0, 9999, 10**4, 99999, 10**5, 999999, 10**6, 9999999, 10**7)] == [
        0, 0, 1, 1, 2, 2, 3, 3, 4,
    ]  # fmt: skip


# -- an independent parser reads every amount back from the text -----------------

_NON_LATIN = DIGIT_SCRIPTS["fa"] + DIGIT_SCRIPTS["ar"]
_TO_LATIN = {ord(c): str(i % 10) for i, c in enumerate(_NON_LATIN)}
_DIGIT = "0-9\u0660-\u0669\u06f0-\u06f9"
_AMOUNT = re.compile(rf"[{_DIGIT}][{_DIGIT}.,\u066b\u066c]*(?: (?:هزار|میلیون|میلیارد))?")
_SCALE = {"هزار": 10**3, "میلیون": 10**6, "میلیارد": 10**9}


def parse(text: str) -> int:
    number, _, word = text.partition(" ")
    number = number.translate(_TO_LATIN).replace("٬", "").replace("٫", ".")
    if word:
        return int(Decimal(number.replace(",", "")) * _SCALE[word])
    return int(number.replace(",", ""))


def test_parser_agrees_with_render_on_random_values():
    import random

    rng = random.Random(7)
    for _ in range(500):
        st = numbers.draw_style(rng)
        value = rng.randint(11, 999) * 10 ** rng.randint(3, 7)
        assert parse(render(value, st)) == value, (value, st)


def test_every_amount_in_the_text_parses_to_the_value_the_label_used(records):
    for r in records:
        found = [parse(m.group()) for m in _AMOUNT.finditer(r.state)]
        facts = r.extra["facts"]
        expected = [facts["value"]] if r.extra["kind"] == "magnitude" else [facts["a"], facts["b"]]
        assert found == expected, (r.id, r.state)


# -- labels and pairs ---------------------------------------------------------------


def recomputed(r):
    kind, f = r.extra["kind"], r.extra["facts"]
    if kind == "equal":
        return "yes" if f["a"] == f["b"] else "no"
    if kind == "larger":
        more = f["a"] > f["b"] if f["relation"] == "more" else f["a"] < f["b"]
        return "yes" if more else "no"
    return next(o for i, o in enumerate(o.id for o in digits.LEVELS) if i == level(f["value"]))


def test_every_label_matches_an_independent_recomputation(records):
    for r in records:
        assert answer(r) == recomputed(r), r.id


def pairs(records, kind):
    grouped = defaultdict(list)
    for r in records:
        if r.extra["kind"] == kind:
            grouped[r.source_id].append(r)
    return list(grouped.values())


def test_equal_pairs_change_one_digit_of_one_amount(records):
    for a, b in pairs(records, "equal"):
        fa, fb = a.extra["facts"], b.extra["facts"]
        keys = [k for k in ("a", "b") if fa[k] != fb[k]]
        assert len(keys) == 1, a.source_id  # exactly one amount differs
        k = keys[0]
        x, y = str(fa[k]), str(fb[k])
        assert len(x) == len(y) and sum(c1 != c2 for c1, c2 in zip(x, y, strict=True)) == 1
        assert fa["styles"] == fb["styles"]
        assert {fa["a"] == fa["b"], fb["a"] == fb["b"]} == {True, False}
        # The two amounts are always written differently.
        assert fa["styles"]["a"] != fa["styles"]["b"]


def test_larger_pairs_swap_the_amounts_together_with_their_styles(records):
    for a, b in pairs(records, "larger"):
        fa, fb = a.extra["facts"], b.extra["facts"]
        assert (fa["a"], fa["b"]) == (fb["b"], fb["a"])
        assert fa["styles"]["a"] == fb["styles"]["b"] and fa["styles"]["b"] == fb["styles"]["a"]
        assert fa["a"] != fa["b"]


def test_magnitude_pairs_are_ten_times_apart_in_the_same_form(records):
    levels = Counter()
    for a, b in pairs(records, "magnitude"):
        fa, fb = a.extra["facts"], b.extra["facts"]
        low, high = sorted((fa["value"], fb["value"]))
        assert high == 10 * low
        assert fa["styles"] == fb["styles"]
        assert abs(level(fa["value"]) - level(fb["value"])) == 1
        levels.update(answer(r) for r in (a, b))
    assert set(levels) == {"l0", "l1", "l2", "l3", "l4"}


def test_magnitude_options_keep_their_order(records):
    for r in records:
        if r.extra["kind"] == "magnitude":
            assert [o.id for o in r.options] == ["l0", "l1", "l2", "l3", "l4"]
            assert r.question_type == "score"


def test_written_forms_are_varied_and_raw(records):
    styles = [s for r in records for s in r.extra["facts"]["styles"].values()]
    assert {s["script"] for s in styles} == {"fa", "latin", "ar"}
    assert {s["form"] for s in styles} == {"full", "scaled"}
    assert {s["sep"] for s in styles} == {"", ",", "٬"}
    assert {s["dec"] for s in styles} == {".", "٫"}


def test_no_facts_repeat_within_a_kind(records):
    def key(pair):
        facts = [
            {k: v for k, v in r.extra["facts"].items() if k not in ("styles", "relation")}
            for r in pair
        ]
        return tuple(sorted(json.dumps(f, sort_keys=True) for f in facts))

    for kind in digits.KINDS:
        keys = [key(p) for p in pairs(records, kind)]
        assert len(keys) == len(set(keys)) == 100, kind


def test_larger_templates_need_a_valid_relation():
    templates = load_templates(digits.NAME, digits.SLOTS)
    assert {t.relation for t in templates["larger"]} == {"more", "less"}
    bad = digits.Template("x", "larger", "formal", False, "{a} {b}", "q", relation=None)
    with pytest.raises(GeneratorError, match="relation"):
        digits._larger(__import__("random").Random(1), bad)


def test_fixed_generation_fingerprint():
    # Changing templates or logic changes this hash: bump digits.VERSION and update it.
    lines = [
        json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True)
        for r in digits.generate(SEED, 5)
    ]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    assert digest == "b9f28a9b3bc873c3bc3926218f0364243111e5d240c36443c2dcddd0dd0b079f", digest
