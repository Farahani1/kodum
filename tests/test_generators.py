import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from itertools import pairwise

import pytest

from kodoom.generators import GENERATORS, common, dates
from kodoom.generators.common import GeneratorError, load_templates
from kodoom.jalali import is_valid, to_gregorian
from kodoom.normalize import clean_orthography, normalize
from kodoom.sources import check_record

SEED = 1234


@pytest.fixture(scope="module")
def records():
    return dates.generate(SEED, 100)


def by_pair(records):
    pairs = defaultdict(list)
    for r in records:
        pairs[r.source_id].append(r)
    return pairs


def answer(r):
    return next(k for k, v in r.gold.items() if v == 1.0)


# -- templates ----------------------------------------------------------------


def test_template_file_meets_the_plans_rules():
    templates = load_templates(dates.NAME, dates.SLOTS)
    assert set(templates) == set(dates.KINDS)
    for kind, group in templates.items():
        assert len(group) >= 5, kind
        assert sum(t.held_out for t in group) == 2, kind


def test_template_file_is_written_as_typed_and_has_no_invisible_characters():
    raw = (common.resources.files("kodoom") / "generators/templates/jalali-dates.toml").read_text(
        encoding="utf-8"
    )
    assert "ي" not in raw and "ك" not in raw  # Arabic letters: the cleaner would hide them
    assert "\u200c" not in raw  # ZWNJ comes from clean_orthography, not from the file


def test_templates_come_out_cleaned_and_digit_free():
    for group in load_templates(dates.NAME, dates.SLOTS).values():
        for t in group:
            assert clean_orthography(t.state) == t.state
            assert clean_orthography(t.question) == t.question
            assert not any(c.isdigit() for c in t.state + t.question)
    assert "می\u200cشود" in load_templates(dates.NAME, dates.SLOTS)["before"][0].state


GOOD = """
[[template]]
kind = "k"
id = "k-{n}"
register = "{register}"
held_out = {held}
state = "متن {{d}}"
question = "پرسش؟"
"""


def toml(specs):
    return "".join(
        GOOD.format(n=i, register=reg, held=str(held).lower())
        for i, (reg, held) in enumerate(specs)
    )


VALID_SPECS = [
    ("formal", False),
    ("colloquial", False),
    ("formal", False),
    ("colloquial", False),
    ("formal", True),
    ("colloquial", True),
]


def test_minimal_valid_template_file_loads():
    assert len(load_templates("x", {"k": ("d",)}, text=toml(VALID_SPECS))["k"]) == 6


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        (VALID_SPECS[:4], "at least 5"),
        (VALID_SPECS[:5], "exactly 2"),  # only one held out
        ([("formal", h) for _, h in VALID_SPECS], "both registers"),
        ([*VALID_SPECS, ("formal", False)], None),  # 7 templates: fine
    ],
)
def test_template_count_and_register_rules(specs, message):
    if message is None:
        load_templates("x", {"k": ("d",)}, text=toml(specs))
        return
    with pytest.raises(GeneratorError, match=message):
        load_templates("x", {"k": ("d",)}, text=toml(specs))


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ("متن ۱۲ {d}", "contains digits"),
        ("متن 12 {d}", "contains digits"),
        ("متن {e}", "placeholders"),
        ("متن بدون جانشین", "placeholders"),
    ],
)
def test_bad_template_text_is_rejected(state, message):
    text = toml(VALID_SPECS).replace("متن {d}", state, 1)
    with pytest.raises(GeneratorError, match=message):
        load_templates("x", {"k": ("d",)}, text=text)


def test_duplicate_ids_and_unknown_kinds_are_rejected():
    text = toml(VALID_SPECS).replace('id = "k-1"', 'id = "k-0"')
    with pytest.raises(GeneratorError, match="duplicate"):
        load_templates("x", {"k": ("d",)}, text=text)
    with pytest.raises(GeneratorError, match="unknown kind"):
        load_templates("x", {"other": ("d",)}, text=toml(VALID_SPECS))


# -- generation ---------------------------------------------------------------


def test_registry():
    assert {"jalali-dates": dates} == GENERATORS


def test_generation_is_reproducible_and_seed_dependent():
    assert dates.generate(SEED, 10) == dates.generate(SEED, 10)
    assert dates.generate(SEED, 10) != dates.generate(SEED + 1, 10)


def test_fixed_generation_fingerprint():
    # Changing templates or logic changes this hash. If that is intended, bump
    # dates.VERSION and update the hash; the published data is then a new version.
    # It also guards reproducibility across Python versions and operating systems.
    lines = [
        json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) for r in dates.generate(SEED, 5)
    ]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    assert digest == "37ad1bd361b14dc1622c2956e02cae2f868ec7228b21a9e78491ba45d3985f3f", digest


def test_counts_and_ids(records):
    assert len(records) == 4 * 100 * 2
    assert len({r.id for r in records}) == len(records)
    assert Counter(r.extra["kind"] for r in records) == {k: 200 for k in dates.KINDS}


def test_every_record_passes_the_schema_and_source_rules(records):
    for r in records:
        check_record(r)
        assert r.origin == "synthetic" and r.task_family == "skill-dates"
        assert r.state_lang == r.question_lang == "fa"
        assert r.source_revision == "jalali-dates-v1"


def test_a_request_for_too_many_pairs_fails_clearly():
    with pytest.raises(GeneratorError, match="no unused facts"):
        dates.generate(SEED, 400)  # the valid-date facts run out first
    with pytest.raises(GeneratorError, match="at least 1"):
        dates.generate(SEED, 0)


# -- labels are recomputed independently ----------------------------------------


def recomputed(r):
    kind, facts = r.extra["kind"], r.extra["facts"]
    if kind == "before":
        ga, gb = to_gregorian(*facts["a"]), to_gregorian(*facts["b"])
        return "yes" if (ga > gb if facts["relation"] == "after" else ga < gb) else "no"
    if kind == "weekday":  # Python's own weekday (Monday = 0), Saturday first
        return dates.WEEKDAY_IDS[(date.fromisoformat(facts["gregorian"]).weekday() + 2) % 7]
    if kind == "valid":
        return "yes" if is_valid(*facts["date"]) else "no"
    return facts["gregorian"]


def test_every_label_matches_an_independent_recomputation(records):
    for r in records:
        assert answer(r) == recomputed(r), r.id
        assert answer(r) in {o.id for o in r.options}


def test_stored_gregorian_facts_agree_with_the_calendar(records):
    for r in records:
        f = r.extra["facts"]
        if r.extra["kind"] in ("weekday", "gregorian"):
            assert f["gregorian"] == to_gregorian(*f["date"]).isoformat()


def test_weekday_option_text_matches_its_id(records):
    names = dict(zip(dates.WEEKDAY_IDS, dates.WEEKDAY_NAMES, strict=True))
    for r in records:
        if r.extra["kind"] == "weekday":
            assert {o.id: o.text for o in r.options} == names


def test_gregorian_options_are_four_consecutive_days_and_a_year_error(records):
    for r in records:
        if r.extra["kind"] != "gregorian":
            continue
        days = sorted(date.fromisoformat(o.id) for o in r.options)
        assert len(days) == 5
        assert [(b - a).days for a, b in pairwise(days[:4])] == [1, 1, 1]
        assert (days[4] - days[1]).days > 300


# -- minimal pairs --------------------------------------------------------------


def test_every_pair_has_two_records_with_different_answers(records):
    pairs = by_pair(records)
    assert len(pairs) == 400
    for source_id, (a, b) in pairs.items():
        assert (a.extra["pair_role"], b.extra["pair_role"]) == ("a", "b")
        assert answer(a) != answer(b), source_id
        # Everything except the one fact is shared.
        assert a.split == b.split and a.options == b.options
        assert a.question_type == b.question_type and a.question_text == b.question_text
        for key in ("template", "variant", "digits", "date_format", "kind", "pair_id"):
            assert a.extra[key] == b.extra[key], (source_id, key)


def test_before_pairs_swap_the_same_two_dates(records):
    for a, b in (p for p in by_pair(records).values() if p[0].extra["kind"] == "before"):
        assert a.extra["facts"]["a"] == b.extra["facts"]["b"]
        assert a.extra["facts"]["b"] == b.extra["facts"]["a"]


def test_weekday_and_gregorian_pairs_are_consecutive_days(records):
    for a, b in (
        p for p in by_pair(records).values() if p[0].extra["kind"] in ("weekday", "gregorian")
    ):
        da = to_gregorian(*a.extra["facts"]["date"])
        db = to_gregorian(*b.extra["facts"]["date"])
        assert abs((da - db).days) == 1


def test_valid_pairs_differ_in_one_step_and_cover_every_variant(records):
    variants = Counter()
    for a, b in (p for p in by_pair(records).values() if p[0].extra["kind"] == "valid"):
        (y1, m1, d1), (y2, m2, d2) = a.extra["facts"]["date"], b.extra["facts"]["date"]
        differing = (y1 != y2) + (m1 != m2) + (d1 != d2)
        assert differing == 1, (a.id, a.extra["facts"], b.extra["facts"])
        variants[a.extra["variant"]] += 1
    assert set(variants) == {"month-end", "day-31", "esfand-30"}


def test_no_facts_repeat_within_a_kind(records):
    def key(pair):
        facts = [{k: v for k, v in r.extra["facts"].items() if k != "relation"} for r in pair]
        return tuple(sorted(json.dumps(f, sort_keys=True) for f in facts))

    for kind in dates.KINDS:
        keys = [key(p) for p in by_pair(records).values() if p[0].extra["kind"] == kind]
        assert len(keys) == len(set(keys)) == 100, kind


# -- splits, balance and raw forms --------------------------------------------------


def test_held_out_templates_give_exactly_the_test_split(records):
    held = {t.id for g in load_templates(dates.NAME, dates.SLOTS).values() for t in g if t.held_out}
    for r in records:
        assert (r.split == "test") == (r.extra["template"] in held), r.id
    assert Counter(r.split for r in records).keys() == {
        "train",
        "validation",
        "calibration",
        "test",
    }


def test_a_pair_never_straddles_two_splits(records):
    assert all(a.split == b.split for a, b in by_pair(records).values())


def test_yes_and_no_are_balanced(records):
    for kind in ("before", "valid"):
        counts = Counter(answer(r) for r in records if r.extra["kind"] == kind)
        assert counts["yes"] == counts["no"] == 100


def test_all_digit_scripts_and_formats_appear_and_stay_raw(records):
    assert {r.extra["digits"] for r in records} == {"latin", "fa", "ar"}
    assert {r.extra["date_format"] for r in records} == {"numeric", "named"}
    persian_digits = set(common.DIGIT_SCRIPTS["fa"])
    assert any(persian_digits & set(r.state) for r in records)
    # Published text is not passed through the model-side normalizer (plan: stored raw).
    assert any(normalize(r.state) != r.state for r in records)


def test_registers_are_spread_across_the_data(records):
    counts = Counter(r.extra["register"] for r in records)
    assert set(counts) == {"formal", "colloquial"}
    assert min(counts.values()) > 0.4 * len(records)
