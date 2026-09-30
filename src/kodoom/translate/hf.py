"""Translators that run open models through Hugging Face transformers (plan 1.2).

Two kinds, because the plan's candidates work differently:

- ``TranslateGemmaTranslator``: a model built only for translation. Its prompt is a
  structured message with source and target language and the text, nothing else, so
  the glossary and the register cannot be put in the prompt; they are enforced by
  the checks afterwards.
- ``ChatTranslator``: any instruction-tuned chat model (Qwen3-8B is the plan's
  second candidate). Its prompt carries the register, the glossary terms that occur
  in the text and the rules for code, identifiers and numbers.

Everything that does not need a GPU (prompts, output cleaning, batching) is plain
Python and tested on the laptop. The model itself sits behind ``load_generator``, which
imports torch and transformers only when called, so ``dev`` never needs them.
The model ids and the TranslateGemma message format come from the plan and the
model card and have NOT been run yet; the first Colab trial shows whether they hold.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from kodoom.translate.glossary import Glossary
from kodoom.translate.glossary import load as load_glossary
from kodoom.translate.pipeline import Item
from kodoom.translate.rules import COLLOQUIAL

Messages = list[dict[str, Any]]
# One reply per conversation, in order.
Generate = Callable[[Sequence[Messages]], list[str]]

SYSTEM = "You are a professional English-to-Persian (Farsi) translator."
REGISTERS = {
    COLLOQUIAL: "Use a natural, colloquial Persian register, as a customer would write.",
    "formal": "Use a formal, neutral Persian register, as in business and technical writing.",
}
RULES = (
    "Keep everything inside backticks, identifiers (such as INV-2026-6633 or svc_task_1), "
    "email addresses, URLs and numbers exactly as written; do not convert digits.",
    "Write Persian letters (ی, ک) and the zero-width non-joiner (نیم\u200cفاصله) correctly.",
    "Keep the line breaks. Do not add notes, explanations or quotation marks.",
)


def chat_messages(item: Item, glossary: Glossary) -> Messages:
    """The conversation for a chat model: register, glossary terms in the text, rules."""
    lines = [
        "Translate the text below from English into Persian.",
        REGISTERS.get(item.register, REGISTERS["formal"]),
        *RULES,
    ]
    terms = glossary.relevant(item.workflow, item.text)
    if terms:
        lines.append(
            "Use these terms: " + "; ".join(f"{en} = {fa}" for en, fa in terms.items()) + "."
        )
    kept = glossary.kept(item.text)
    if kept:
        lines.append("Leave these terms in English: " + ", ".join(kept) + ".")
    lines.append("Reply with the translation only.")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(lines) + "\n\nText:\n" + item.text},
    ]


def gemma_messages(item: Item) -> Messages:
    """The TranslateGemma message: languages and text only (as in its model card)."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "en",
                    "target_lang_code": "fa",
                    "text": item.text,
                }
            ],
        }
    ]


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```[a-z]*\n(.*?)\n?```$", re.DOTALL)
_LABEL = re.compile(r"^(?:translation|ترجمه)\s*[:：]\s*", re.IGNORECASE)
_QUOTES = ('"', "'", "«", "»", "“", "”")


def clean_output(reply: str, source: str) -> str:
    """The translation from a model reply: no reasoning block, code fence or label, and
    no quotation marks the source did not have."""
    text = _THINK.sub("", reply).strip()
    fence = _FENCE.match(text)
    if fence:
        text = fence.group(1).strip()
    text = _LABEL.sub("", text)
    if len(text) > 1 and text[0] in _QUOTES and text[-1] in _QUOTES and source[:1] not in _QUOTES:
        text = text[1:-1].strip()
    return text


class ChatTranslator:
    def __init__(self, name: str, generate: Generate, glossary: Glossary | None = None) -> None:
        self.name = name
        self._generate = generate
        self._glossary = glossary if glossary is not None else load_glossary()

    def translate(self, items: Sequence[Item]) -> list[str]:
        replies = self._generate([chat_messages(i, self._glossary) for i in items])
        return _pair(items, replies)


class TranslateGemmaTranslator:
    def __init__(self, name: str, generate: Generate) -> None:
        self.name = name
        self._generate = generate

    def translate(self, items: Sequence[Item]) -> list[str]:
        return _pair(items, self._generate([gemma_messages(i) for i in items]))


def _pair(items: Sequence[Item], replies: Sequence[str]) -> list[str]:
    if len(replies) != len(items):
        raise ValueError(f"the model returned {len(replies)} replies for {len(items)} texts")
    return [clean_output(r, i.text) for i, r in zip(items, replies, strict=True)]


# -- the model (needs torch and transformers; Colab only) -----------------------------


def load_generator(
    model_id: str,
    *,
    four_bit: bool = False,
    dtype: str = "float16",
    batch_size: int = 8,
    max_new_tokens: int = 768,
) -> Generate:
    """Load a model and return a function that answers a batch of conversations.

    Greedy decoding, so a run is repeatable. Conversations are sorted by length to keep
    padding small and answered in the original order. ``four_bit`` needs bitsandbytes
    (CUDA only, the ``colab`` extra); it is how a 12B or 8B model fits a T4. ``dtype`` is
    the precision of the activations (and of the layers that are not quantized): Gemma
    models overflow in float16 and then answer with nothing, which is detected and reported;
    bfloat16 or float32 avoid it.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise RuntimeError(
            "torch and transformers are needed: pip install -e '.[colab]' on a Colab runtime"
        ) from e
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    options: dict[str, Any] = {"device_map": "auto"}
    if four_bit:
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, dtype),
        )
    options["dtype"] = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, **options)
    model.eval()

    def generate(conversations: Sequence[Messages]) -> list[str]:
        prompts = [
            tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
            for m in conversations
        ]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        replies = [""] * len(prompts)
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            batch = tokenizer(
                [prompts[i] for i in ids],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            if not torch.isfinite(out.scores[0]).all():
                raise RuntimeError(
                    f"{model_id} produced NaN or infinite scores with dtype {dtype}: the "
                    "activations overflowed (known for Gemma in float16 on a T4), so every "
                    "answer would be empty. Use another precision, e.g. the -bf16 or "
                    "-4bit-fp32 variants of the translator."
                )
            texts = tokenizer.batch_decode(
                out.sequences[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for i, text in zip(ids, texts, strict=True):
                replies[i] = text
        return replies

    return generate


# Model ids are as the plan names them; check them on Colab (a wrong id fails loudly).
TRANSLATEGEMMA_4B = "google/translategemma-4b-it"
TRANSLATEGEMMA_12B = "google/translategemma-12b-it"
QWEN3_8B = "Qwen/Qwen3-8B"


def translategemma_4b() -> TranslateGemmaTranslator:
    return TranslateGemmaTranslator("translategemma-4b", load_generator(TRANSLATEGEMMA_4B))


def translategemma_4b_bf16() -> TranslateGemmaTranslator:
    generate = load_generator(TRANSLATEGEMMA_4B, dtype="bfloat16")
    return TranslateGemmaTranslator("translategemma-4b-bf16", generate)


def translategemma_4b_4bit_fp32() -> TranslateGemmaTranslator:
    """4-bit weights with float32 activations: fits the T4 and cannot overflow."""
    generate = load_generator(TRANSLATEGEMMA_4B, four_bit=True, dtype="float32")
    return TranslateGemmaTranslator("translategemma-4b-4bit-fp32", generate)


def translategemma_12b_4bit() -> TranslateGemmaTranslator:
    generate = load_generator(TRANSLATEGEMMA_12B, four_bit=True)
    return TranslateGemmaTranslator("translategemma-12b-4bit", generate)


def qwen3_8b_4bit() -> ChatTranslator:
    return ChatTranslator("qwen3-8b-4bit", load_generator(QWEN3_8B, four_bit=True))
