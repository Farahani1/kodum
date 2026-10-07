"""Versioned production prompts and strict text-only reconstruction (BULK-02)."""

from __future__ import annotations

import json
from dataclasses import replace

from kodoom.bulk.state import Unit
from kodoom.translate.glossary import load as load_glossary
from kodoom.translate.hf import RULES, SYSTEM, chat_messages, clean_output
from kodoom.translate.pipeline import (
    Item,
    case_items,
    translate_case,
    translate_helmo_record,
)

VERSION = "bulk-context-and-helmo-json-v1"
JSON_MARKER = "\n\nJSON:\n"


def requests(unit: Unit) -> list[list[dict]]:
    glossary = load_glossary()
    if unit.dataset == "helmo":
        record = unit.records[0]
        payload = {
            "state": record.state,
            "question": record.question_text,
            "options": {o.id: o.text for o in record.options},
        }
        text = json.dumps(payload, ensure_ascii=False)
        kept = glossary.kept(text)
        terms = glossary.relevant("", text)
        instructions = [
            "Translate every text value of this JSON from English into formal Persian.",
            "Translate state, question and all option descriptions together. Use consistent terms.",
            "Do not answer the question or select an option. Preserve negation and severity.",
            "Preserve acronyms, symbols, units, amounts and times exactly as in the source.",
            *RULES,
            "Return only valid JSON with exactly the same keys and option IDs, no extra fields.",
        ]
        if kept:
            instructions.append("Leave these terms in English: " + ", ".join(kept))
        if terms:
            instructions.append(
                "Use these terms: " + "; ".join(f"{a} = {b}" for a, b in terms.items())
            )
        return [
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n".join(instructions) + JSON_MARKER + text},
            ]
        ]
    result = []
    for item in case_items(unit.records):
        messages = chat_messages(item, glossary)
        # Context precedes the target text, so the last Text marker always names
        # exactly the one field whose translation will be reconstructed.
        context = (
            "\n\nEnglish case context (do not translate this context):\n" + unit.records[0].state
        )
        context += "\nQuestions: " + " | ".join(r.question_text for r in unit.records)
        messages[1]["content"] = messages[1]["content"].replace(
            "\n\nText:\n", context + "\n\nText:\n", 1
        )
        result.append(messages)
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in structured translation")
        result[key] = value
    return result


def rebuild(unit: Unit, replies: list[str], translator: str):
    class Fixed:
        name = translator

        def translate(self, items):
            if len(items) != len(texts):
                raise ValueError("translation item count changed")
            return texts

    if unit.dataset == "helmo":
        if len(replies) != 1:
            raise ValueError("helmo needs one complete structured reply")
        value = json.loads(clean_output(replies[0], "{"), object_pairs_hook=_unique_object)
        record = unit.records[0]
        if not isinstance(value, dict) or set(value) != {"state", "question", "options"}:
            raise ValueError("structured translation added or omitted fields")
        if not isinstance(value["options"], dict) or set(value["options"]) != {
            o.id for o in record.options
        }:
            raise ValueError("structured translation changed option IDs")
        texts = [
            value["state"],
            value["question"],
            *[value["options"][o.id] for o in record.options],
        ]
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("structured translation has empty or non-text values")
        translated, _ = translate_helmo_record(record, Fixed())
        records = [translated]
    else:
        items = case_items(unit.records)
        if len(replies) != len(items):
            raise ValueError("incomplete typed case replies")
        texts = [clean_output(reply, item.text) for reply, item in zip(replies, items, strict=True)]
        records, _ = translate_case(unit.records, Fixed())
    return [
        replace(
            r, extra={**r.extra, "prompt_version": VERSION, "review_status": "unreviewed-draft"}
        )
        for r in records
    ]


class StubGenerator:
    """CPU fixtures only; never used for a real campaign or throughput claim."""

    def __init__(self):
        self.info = {"checkpoint": {"fingerprint": "stub"}, "calls": []}

    def lengths(self, messages):
        return [max(1, len(str(message)) // 4) for message in messages]

    def batch(self, messages, size):
        from kodoom.translate.pipeline import StubTranslator

        stub = StubTranslator()
        result = []
        for conversation in messages:
            body = conversation[1]["content"]
            if JSON_MARKER in body:
                value = json.loads(body.split(JSON_MARKER)[-1])

                def translated(text):
                    return stub.translate([Item(text, "formal", "state")])[0]

                result.append(
                    json.dumps(
                        {
                            "state": translated(value["state"]),
                            "question": translated(value["question"]),
                            "options": {k: translated(v) for k, v in value["options"].items()},
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                result.append(
                    stub.translate([Item(body.split("\n\nText:\n")[-1], "formal", "question")])[0]
                )
        self.info["calls"].append(
            {
                "seconds": len(messages) / size,
                "warm": True,
                "output_tokens": len(messages) * 5,
                "devices": [],
            }
        )
        return result
