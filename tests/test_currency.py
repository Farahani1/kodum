import hashlib
import json
import random
from collections import Counter, defaultdict

import pytest
from amounts import AMOUNT_WITH_UNIT, parse_with_unit

from kodoom.generators import currency
from kodoom.generators.common import GeneratorError, load_templates
from kodoom.generators.currency import rial_value

SEED = 1234


@pytest.fixture(scope="module")
def records():
    return currency.generate(SEED, 100)


def answer(r):
    return next(k for k, v in r.gold.items() if v == 1.0)


def pairs(records, kind):
    grouped = defaultdict(list)
    for r in records:
        if r.extra["kind"] == kind:
            grouped[r.source_id].append(r)
    return list(grouped.values())


def test_a_toman_is_ten_rials():
    assert rial_value(1, "toman") == 10 and rial_value(1, "rial") == 1
    assert rial_value(1_200_000, "toman") == rial_value(12_000_000, "rial")


# -- the amounts in the text are the amounts the label used -----------------------


def test_every_amount_and_unit_in_the_text_matches_the_facts(records):
    for r in records:
        found = [parse_with_unit(m.group()) for m in AMOUNT_WITH_UNIT.finditer(r.state)]
        f = r.extra["facts"]
        expected = (
            [(f["value"], f["unit"])]
            if r.extra["kind"] == "convert"
            else [(f["a"], f["a_unit"]), (f["b"], f["b_unit"])]
        )
        assert found == expected, (r.id, r.state)


def test_colloquial_spelling_appears_only_in_colloquial_templates(records):
    spelled = Counter()
    for r in records:
        word = r.extra["toman_word"]
        assert word in ("تومان", "تومن")
        if word == "تومن":
            assert r.extra["register"] == "colloquial", r.id
            spelled[r.extra["kind"]] += 1
    assert set(spelled) == set(currency.KINDS)


# -- labels, recomputed independently ----------------------------------------------


def recomputed(r):
    kind, f = r.extra["kind"], r.extra["facts"]
    if kind == "convert":
        n = f["value"]
        return f"toman:{n // 10}" if f["unit"] == "rial" else f"rial:{10 * n}"
    a, b = rial_value(f["a"], f["a_unit"]), rial_value(f["b"], f["b_unit"])
    if kind == "equal":
        return "yes" if a == b else "no"
    return "yes" if (a > b if f["relation"] == "more" else a < b) else "no"


def test_every_label_matches_an_independent_recomputation(records):
    for r in records:
        assert answer(r) == recomputed(r), r.id


def test_compare_never_asks_about_equal_amounts(records):
    for r in records:
        if r.extra["kind"] == "compare":
            f = r.extra["facts"]
            assert rial_value(f["a"], f["a_unit"]) != rial_value(f["b"], f["b_unit"])


# -- minimal pairs ---------------------------------------------------------------------


def test_compare_and_equal_pairs_keep_the_numbers_and_change_one_unit(records):
    for kind in ("compare", "equal"):
        for a, b in pairs(records, kind):
            fa, fb = a.extra["facts"], b.extra["facts"]
            assert (fa["a"], fa["b"]) == (fb["a"], fb["b"]), a.source_id
            changed = [k for k in ("a_unit", "b_unit") if fa[k] != fb[k]]
            assert len(changed) == 1, a.source_id
            assert a.extra["variant"] == f"change-{changed[0][0]}-unit"
            assert fa["styles"] == fb["styles"]


def test_equal_pairs_always_have_one_equal_half(records):
    for a, b in pairs(records, "equal"):
        assert {answer(a), answer(b)} == {"yes", "no"}
        for r in (a, b):
            f = r.extra["facts"]
            if answer(r) == "yes":  # the same money written in two units
                assert {f["a_unit"], f["b_unit"]} == {"toman", "rial"}


def test_compare_pairs_include_a_mixed_unit_half(records):
    mixed = 0
    for a, b in pairs(records, "compare"):
        for r in (a, b):
            f = r.extra["facts"]
            mixed += f["a_unit"] != f["b_unit"]
    assert mixed >= 100  # at least one mixed half in every pair (100 pairs)


def test_convert_options_are_six_distinct_amounts_shared_by_the_pair(records):
    for a, b in pairs(records, "convert"):
        assert a.options == b.options and len(a.options) == 6
        assert len({o.id for o in a.options}) == 6 and len({o.text for o in a.options}) == 6
        assert {a.extra["facts"]["unit"], b.extra["facts"]["unit"]} == {"toman", "rial"}
        assert a.extra["facts"]["value"] == b.extra["facts"]["value"]
        assert answer(a) != answer(b)
        for r in (a, b):
            assert answer(r) in {o.id for o in r.options}


def test_convert_correct_option_text_is_the_conversion(records):
    for r in records:
        if r.extra["kind"] != "convert":
            continue
        f = r.extra["facts"]
        text = next(o.text for o in r.options if o.id == answer(r))
        value, unit = parse_with_unit(text)
        assert unit != f["unit"]
        assert rial_value(value, unit) == rial_value(f["value"], f["unit"])


def test_no_facts_repeat_within_a_kind(records):
    def key(pair):
        facts = [
            {k: v for k, v in r.extra["facts"].items() if k not in ("styles", "relation")}
            for r in pair
        ]
        return tuple(sorted(json.dumps(f, sort_keys=True) for f in facts))

    for kind in currency.KINDS:
        keys = [key(p) for p in pairs(records, kind)]
        assert len(keys) == len(set(keys)) == 100, kind


def test_compare_finds_a_pair_for_every_factor_and_relation():
    # The builder searches unit combinations; it must never come up empty.
    templates = load_templates(currency.NAME, currency.SLOTS)["compare"]
    for template in templates:
        rng = random.Random(3)
        for _ in range(300):
            currency._compare(rng, template)


def test_compare_needs_a_valid_relation():
    bad = currency.Template("x", "compare", "formal", False, "{a} {b}", "q", relation=None)
    with pytest.raises(GeneratorError, match="relation"):
        currency._compare(random.Random(1), bad)


def test_fixed_generation_fingerprint():
    # Changing templates or logic changes this hash: bump currency.VERSION and update it.
    lines = [
        json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True)
        for r in currency.generate(SEED, 5)
    ]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    assert digest == "fb27f7be4dde309afd96e002eedd851dcf52383928e6eb3eb27cbe87a716dded", digest
