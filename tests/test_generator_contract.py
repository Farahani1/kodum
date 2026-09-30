"""What every skill generator must satisfy (plan 1.1). Runs over the registry, so a
new generator is checked the moment it is registered."""

from collections import Counter, defaultdict

import pytest

from kodoom.generators import GENERATORS
from kodoom.generators.common import GeneratorError, load_templates
from kodoom.sources import check_record

SEED = 1234
NAMES = sorted(GENERATORS)


@pytest.fixture(scope="module", params=NAMES)
def generated(request):
    module = GENERATORS[request.param]
    return module, module.generate(SEED, 100)


def pairs_of(records):
    pairs = defaultdict(list)
    for r in records:
        pairs[r.source_id].append(r)
    return pairs


def answer(r):
    return next(k for k, v in r.gold.items() if v == 1.0)


def test_generators_are_registered_under_their_own_names():
    assert all(name == module.NAME for name, module in GENERATORS.items())
    assert {
        "jalali-dates",
        "digit-forms",
        "toman-rial",
        "business-hours",
        "iranian-formats",
    } <= set(GENERATORS)


@pytest.mark.parametrize("name", NAMES)
def test_reproducible_and_seed_dependent(name):
    module = GENERATORS[name]
    assert module.generate(SEED, 8) == module.generate(SEED, 8)
    assert module.generate(SEED, 8) != module.generate(SEED + 1, 8)


@pytest.mark.parametrize("name", NAMES)
def test_zero_pairs_is_an_error(name):
    with pytest.raises(GeneratorError, match="at least 1"):
        GENERATORS[name].generate(SEED, 0)


@pytest.mark.parametrize("name", NAMES)
def test_templates_follow_the_plans_rules(name):
    spec = GENERATORS[name].SPEC
    templates = load_templates(spec.name, spec.slots)
    assert set(templates) == set(spec.slots)
    for group in templates.values():
        assert len(group) >= 5 and sum(t.held_out for t in group) == 2


def test_records_pass_the_schema_and_source_rules(generated):
    module, records = generated
    spec = module.SPEC
    assert len(records) == len(spec.slots) * 100 * 2
    assert len({r.id for r in records}) == len(records)
    for r in records:
        check_record(r)
        assert r.origin == "synthetic" and r.task_family == spec.task_family
        assert r.state_lang == r.question_lang == "fa"
        assert r.source_revision == f"{spec.name}-v{spec.version}"
        assert r.question_type == spec.question_types[r.extra["kind"]]
        assert sum(r.gold.values()) == 1.0 and answer(r) in {o.id for o in r.options}


def test_every_pair_is_two_records_that_differ_in_the_answer_only(generated):
    module, records = generated
    pairs = pairs_of(records)
    assert len(pairs) == len(module.SPEC.slots) * 100
    for source_id, group in pairs.items():
        assert len(group) == 2, source_id
        a, b = group
        assert (a.extra["pair_role"], b.extra["pair_role"]) == ("a", "b")
        assert answer(a) != answer(b), source_id
        assert a.split == b.split and a.options == b.options
        assert a.question_type == b.question_type and a.question_text == b.question_text
        for key in ("template", "variant", "kind", "pair_id", "register", "generator"):
            assert a.extra[key] == b.extra[key], (source_id, key)


def test_held_out_templates_give_exactly_the_test_split(generated):
    module, records = generated
    templates = load_templates(module.SPEC.name, module.SPEC.slots)
    held = {t.id for group in templates.values() for t in group if t.held_out}
    for r in records:
        assert (r.split == "test") == (r.extra["template"] in held), r.id
    assert set(Counter(r.split for r in records)) == {"train", "validation", "calibration", "test"}


def test_registers_are_both_used(generated):
    _, records = generated
    counts = Counter(r.extra["register"] for r in records)
    assert set(counts) == {"formal", "colloquial"}
    assert min(counts.values()) > 0.4 * len(records)


def test_noul_questions_are_balanced_by_construction(generated):
    _, records = generated
    for kind in {r.extra["kind"] for r in records if r.question_type == "noul"}:
        counts = Counter(answer(r) for r in records if r.extra["kind"] == kind)
        assert counts["yes"] == counts["no"]


def test_state_text_is_not_pre_normalized(generated):
    # Published skill data stays as people write it (plan: benchmark stored raw).
    from kodoom.normalize import normalize

    _, records = generated
    assert any(normalize(r.state) != r.state for r in records)
