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
    assert "formal" in prompt.split("\n")[1]
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
