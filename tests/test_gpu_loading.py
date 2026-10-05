import sys
import types

import pytest

from kodoom.translate.hf import GEMMA3_4B, load_generator


def test_gemma_multimodal_loader_pins_tokenizer_and_model_revision(monkeypatch):
    calls = []
    tokenizer = types.SimpleNamespace(pad_token="pad")
    model = types.SimpleNamespace(eval=lambda: None)
    module = types.ModuleType("transformers")
    module.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **kw: calls.append(("tokenizer", kw)) or tokenizer
    )
    module.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **kw: pytest.fail("wrong Gemma 4B loader")
    )
    module.Gemma3ForConditionalGeneration = types.SimpleNamespace(
        from_pretrained=lambda *a, **kw: calls.append(("model", kw)) or model
    )
    module.BitsAndBytesConfig = lambda **kw: kw
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setenv("KODOOM_MODEL_REVISION", "model-sha")
    monkeypatch.setenv("KODOOM_TRANSLATION_BATCH_SIZE", "1")
    load_generator(GEMMA3_4B, dtype="bfloat16")
    assert calls[0][1]["revision"] == calls[1][1]["revision"] == "model-sha"
    assert calls[1][1]["dtype"] == "bfloat16"
