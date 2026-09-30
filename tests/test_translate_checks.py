import json

import pytest

from kodoom.translate.checks import (
    check_state,
    check_text,
    foreign_letters,
    numbers,
    protected_tokens,
    repeated_phrase,
)
from kodoom.translate.rules import apply


def names(findings):
    return sorted({f.check for f in findings})


def test_protected_tokens_find_code_ids_emails_and_urls_but_not_plain_hyphenated_words():
    text = (
        "Account `svc_task_alpha` from `10.0.0.5`, host DESKTOP-882, a long-unused login "
        "e.g. alice.smith@example.com with repo:read for INV-2026-6633 and A-68034, "
        "see https://example.com/a"
    )
    tokens = protected_tokens(text)
    assert tokens == [
        "`svc_task_alpha`",
        "`10.0.0.5`",
        "https://example.com/a",
        "alice.smith@example.com",
        "DESKTOP-882",
        "repo:read",
        "INV-2026-6633",
        "A-68034",
    ]
    assert "long-unused" not in tokens and "e.g" not in tokens


def test_numbers_are_compared_across_digit_scripts_and_separators():
    assert numbers("۱۲٫۵ و 3,000 و ۱۴:۰۵") == ["12.5", "3,000", "14", "05"]


def test_a_good_translation_passes():
    source = "Account `svc_task_alpha` made 95 failed logins from host DESKTOP-882."
    target = "حساب `svc_task_alpha` تعداد ۹۵ ورود ناموفق را از میزبان DESKTOP-882 انجام داد."
    assert check_text(source, target) == []


def test_each_failure_is_named():
    source = "Account `svc_task_alpha` made 95 failed logins from host DESKTOP-882 today."
    assert names(check_text(source, "  ")) == ["empty"]
    assert "identifiers" in names(check_text(source, "حساب svc_task_alpha شماره ۹۵ ورود ناموفق"))
    assert "numbers" in names(check_text(source, "حساب `svc_task_alpha` ۹۶ ورود از DESKTOP-882"))
    english = "Account made failed logins from a host today and it was quite odd really"
    assert "script" in names(check_text("An account made failed logins.", english))
    assert "length" in names(check_text(source, "حساب"))
    assert "looping" in names(check_text("Please wait.", "لطفا صبر کنید " * 6))


def test_repeated_phrase():
    assert repeated_phrase("این است این است این است این است") == "این است"
    assert repeated_phrase("خوب خوب خوب خوب") == "خوب"
    assert repeated_phrase("a a a") is None
    assert repeated_phrase("a b c d e f") is None


STATE = json.dumps(
    {
        "task": "Rotate the expired TLS certificate on the staging load balancer.",
        "agent": {"model": "internal-agent-v1", "autonomy": "checkpointed"},
        "constraints": ["Do not exceed a $50 spend on cloud resources"],
        "trace_summary": {"steps": 11, "tool_errors": 0},
    }
)
GOOD = {
    ("task",): "گواهی TLS منقضی\u200cشده را روی متعادل\u200cکننده بار محیط آزمایشی تعویض کنید.",
    ("constraints", 0): "بیش از ۵۰ دلار برای منابع ابری خرج نکنید",
}


def test_check_state_passes_a_correct_translation():
    assert check_state("agent_trace_observability", STATE, apply(STATE, GOOD)) == []


def test_check_state_catches_changed_keep_fields_numbers_and_structure():
    w = "agent_trace_observability"
    changed = json.loads(apply(STATE, GOOD))
    changed["agent"]["model"] = "عامل داخلی"
    assert names(check_state(w, STATE, json.dumps(changed, ensure_ascii=False))) == ["keep"]

    changed = json.loads(apply(STATE, GOOD))
    changed["trace_summary"]["steps"] = 12
    assert names(check_state(w, STATE, json.dumps(changed))) == ["keep"]

    lost = apply(STATE, {**GOOD, ("constraints", 0): "بیش از حد خرج نکنید برای منابع ابری"})
    assert names(check_state(w, STATE, lost)) == ["numbers"]

    extra = json.loads(apply(STATE, GOOD))
    extra["constraints"].append("x")
    assert names(check_state(w, STATE, json.dumps(extra))) == ["structure"]
    assert names(check_state(w, STATE, "{oops")) == ["structure"]


def test_findings_say_where():
    w = "agent_trace_observability"
    bad = apply(STATE, {**GOOD, ("task",): "Rotate it"})
    assert all(f.where == "state.task" for f in check_state(w, STATE, bad))


def test_unknown_workflow_is_refused():
    with pytest.raises(KeyError):
        check_state("hr", "{}", "{}")


def test_a_look_alike_letter_from_another_script_is_flagged():
    cyrillic = (
        "هیچگاه \u043c\u043e\u043d\u0438\u0442\u043e\u0440\u0438\u043d\u0433 را غیرفعال نکنید"
    )
    findings = check_text("Never disable monitoring or alerting", cyrillic)
    assert [f.check for f in findings] == ["script"]
    assert "cyrillic" in findings[0].message
    assert foreign_letters("سلام hello ۱۲۳ 45 `x_1`") == []
    assert foreign_letters("漢字 と かな") == [
        "漢 (cjk)",
        "字 (cjk)",
        "と (hiragana)",
        "か (hiragana)",
        "な (hiragana)",
    ]
