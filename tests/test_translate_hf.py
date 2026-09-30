import sys

import pytest

from kodoom.translate import hf
from kodoom.translate.glossary import load
from kodoom.translate.hf import (
    ChatTranslator,
    TranslateGemmaTranslator,
    chat_messages,
    clean_output,
    gemma_messages,
    load_generator,
    pin_terms,
)
from kodoom.translate.pipeline import Item, translator_factory


def item(
    text="Your refund for the TLS invoice is `svc_1`.",
    register="formal",
    workflow="customer_service",
):
    return Item(text, register, "state", workflow)


def user_text(messages):
    return messages[-1]["content"]


def test_chat_prompt_carries_register_relevant_terms_and_rules():
    prompt = user_text(chat_messages(item(), load()))
    assert "formal" in prompt.split("\n")[2]
    assert "refund = بازپرداخت" in prompt and "invoice = فاکتور" in prompt
    assert "ticket" not in prompt  # only terms that occur in the text
    assert "Leave these terms in English: TLS" in prompt
    assert "backticks" in prompt and "do not convert digits" in prompt
    assert prompt.endswith("Text:\nYour refund for the TLS invoice is `svc_1`.")
    colloquial = user_text(chat_messages(item(register="colloquial"), load()))
    assert "colloquial" in colloquial


def test_glossary_terms_follow_the_workflow():
    text = "The agent stopped."
    assert "عامل" in user_text(
        chat_messages(item(text, workflow="agent_trace_observability"), load())
    )
    assert "کارشناس پشتیبانی" in user_text(chat_messages(item(text), load()))


def test_translategemma_message_is_languages_and_text_only():
    (message,) = gemma_messages(item("Hello"))
    assert message["role"] == "user"
    assert message["content"] == [
        {"type": "text", "source_lang_code": "en", "target_lang_code": "fa", "text": "Hello"}
    ]


@pytest.mark.parametrize(
    ("reply", "source", "expected"),
    [
        ("<think>hmm</think>\nسلام", "Hi", "سلام"),
        ("```\nسلام\n```", "Hi", "سلام"),
        ("Translation: سلام", "Hi", "سلام"),
        ("ترجمه: سلام", "Hi", "سلام"),
        ('"سلام"', "Hi", "سلام"),
        ('"سلام"', '"Hi"', '"سلام"'),
        ("  سلام  \n", "Hi", "سلام"),
        ("", "Hi", ""),
        ("`سلام دنیا`", "Hello world", "سلام دنیا"),
        ("`x_1` را ببین", "See `x_1`", "`x_1` را ببین"),
        ("`x_1`", "`x_1`", "`x_1`"),
    ],
)
def test_clean_output(reply, source, expected):
    assert clean_output(reply, source) == expected


def test_translators_keep_order_and_clean_replies():
    seen = []

    def generate(conversations):
        seen.extend(conversations)
        return [f"«ترجمه {i}»" for i, _ in enumerate(conversations)]

    items = [item("One"), item("Two")]
    out = ChatTranslator("chat", generate).translate(items)
    assert out == ["ترجمه 0", "ترجمه 1"] and len(seen) == 2
    assert TranslateGemmaTranslator("g", generate).translate(items) == out


def test_a_short_reply_list_is_refused():
    with pytest.raises(ValueError, match="returned 1 replies for 2 texts"):
        ChatTranslator("chat", lambda c: ["x"]).translate([item(), item()])


def test_model_translators_load_lazily_and_say_what_is_missing(monkeypatch):
    factory = translator_factory("qwen3-8b-4bit")  # importing hf loaded no torch
    assert callable(factory) and "torch" not in sys.modules
    monkeypatch.setitem(sys.modules, "torch", None)  # simulate a machine without torch
    with pytest.raises(RuntimeError, match="torch and transformers are needed"):
        load_generator("any/model")
    with pytest.raises(KeyError, match="unknown translator"):
        translator_factory("nope")


def test_the_registered_model_ids_are_named_constants():
    assert hf.TRANSLATEGEMMA_4B.startswith("google/translategemma")
    assert hf.QWEN3_8B == "Qwen/Qwen3-8B"


def test_every_registered_model_translator_has_a_factory():
    from kodoom.translate.pipeline import MODEL_TRANSLATORS

    for name in MODEL_TRANSLATORS:
        assert callable(translator_factory(name)), name
    assert {"translategemma-4b-bf16", "translategemma-4b-4bit-fp32"} <= set(MODEL_TRANSLATORS)


def test_pin_terms_writes_the_persian_term_into_the_english_text():
    glossary = load()
    ai = "agent_trace_observability"
    agent = glossary.terms(ai)["agent"]
    pinned = pin_terms(item("The agent stopped. Two agents wait.", workflow=ai), glossary)
    assert pinned.text == f"The {agent} stopped. Two {agent}\u200c\u0647\u0627 wait."
    security = glossary.terms("security_incidents")
    text = pin_terms(
        item("A service account used the account.", workflow="security_incidents"), glossary
    )
    assert text.text == f"A {security['service account']} used the {glossary.common['account']}."
    # register survives and the original item is not changed
    original = item("TLS refund", register="colloquial")
    assert pin_terms(original, glossary).register == "colloquial"
    assert original.text == "TLS refund"


def test_translategemma_sends_pinned_text_only_when_given_a_glossary():
    sent = []

    def generate(conversations):
        sent.extend(m[0]["content"][0]["text"] for m in conversations)
        return ["x"] * len(conversations)

    the_item = item("Your refund", workflow="customer_service")
    TranslateGemmaTranslator("plain", generate).translate([the_item])
    TranslateGemmaTranslator("pinned", generate, load()).translate([the_item])
    assert sent == ["Your refund", f"Your {load().common['refund']}"]


def test_the_chat_prompt_says_where_the_text_comes_from():
    prompt = user_text(
        chat_messages(item("The agent stopped.", workflow="agent_trace_observability"), load())
    )
    lines = prompt.split("\n")
    assert lines[1].startswith("Context: ") and "never a person" in lines[1]
    support = user_text(
        chat_messages(item("The agent stopped.", workflow="customer_service"), load())
    )
    assert "human support agent" in support
    assert "Context:" not in user_text(chat_messages(item("Hi", workflow="unknown"), load()))
