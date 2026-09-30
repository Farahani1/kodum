import json

import pytest

from kodoom.translate.rules import (
    COLLOQUIAL,
    FORMAL,
    KEEP,
    RULES,
    TRANSLATE,
    RuleError,
    apply,
    segments,
    walk,
)

STATE = json.dumps(
    {
        "thread": [
            {"role": "customer", "text": "Hi, my order is late."},
            {"role": "agent", "text": "Sorry about that."},
        ],
        "orders": [{"id": "A-1", "date": "2026-01-02", "status": "settled", "total": 12.5}],
        "account": {"tier": "premium"},
    }
)


def test_segments_carry_location_path_and_rule():
    found = segments("customer_service", STATE)
    by_path = {}
    for s in found:
        by_path.setdefault(s.path, []).append(s)
    assert [s.text for s in by_path["thread[].text"]] == [
        "Hi, my order is late.",
        "Sorry about that.",
    ]
    assert by_path["thread[].text"][1].location == ("thread", 1, "text")
    assert all(
        s.rule.action == TRANSLATE and s.rule.register == COLLOQUIAL
        for s in by_path["thread[].text"]
    )
    assert by_path["orders[].id"][0].rule.action == KEEP
    assert "orders[].total" not in by_path  # numbers are not text leaves


def test_an_unknown_field_or_workflow_is_an_error_not_a_guess():
    state = json.dumps({"account": {"tier": "x", "surprise": "new text"}})
    with pytest.raises(RuleError, match=r"'account\.surprise' has no rule"):
        segments("customer_service", state)
    with pytest.raises(RuleError, match="no translation rules for workflow"):
        segments("hr", "{}")
    with pytest.raises(RuleError, match="not valid JSON"):
        segments("customer_service", "{oops")


def test_apply_replaces_only_the_given_leaves_and_keeps_order():
    new = apply(STATE, {("thread", 0, "text"): "سلام، سفارش من دیر شده است."})
    tree = json.loads(new)
    assert tree["thread"][0]["text"].startswith("سلام")
    assert tree["thread"][1] == json.loads(STATE)["thread"][1]
    assert tree["orders"] == json.loads(STATE)["orders"]
    assert list(tree) == ["thread", "orders", "account"]
    assert "سلام" in new  # written as text, not \u escapes


def test_apply_refuses_to_overwrite_a_non_text_leaf():
    with pytest.raises(RuleError, match="not a text field"):
        apply(STATE, {("orders", 0, "total"): "x"})


def test_every_rule_names_a_known_action_and_register():
    for workflow, rules in RULES.items():
        assert rules, workflow
        for rule in rules.values():
            assert rule.action in (TRANSLATE, KEEP) and rule.register in (FORMAL, COLLOQUIAL)
        assert any(r.action == TRANSLATE for r in rules.values()), workflow


def test_walk_collapses_list_positions():
    paths = [p for _, p, _ in walk({"a": [{"b": "x"}, {"b": "y"}]})]
    assert paths == ["a[].b", "a[].b"]
