from types import SimpleNamespace

import pytest

import kodoom.translate.jax as native
from kodoom.translate.checkpoint import describe_checkpoint
from kodoom.translate.glossary import load
from kodoom.translate.hf import chat_messages
from kodoom.translate.jax import decode_completed, format_prompt, parameter_summary, validate_limits
from kodoom.translate.pipeline import Item, translator_factory


def test_native_prompt_keeps_the_complete_existing_instructions():
    messages = chat_messages(
        Item("Do not approve INV-1 at 8 PM.", "formal", "state", "invoices"), load()
    )
    prompt = format_prompt(messages)
    assert prompt == (
        "<start_of_turn>user\n"
        + messages[0]["content"]
        + "\n\n"
        + messages[1]["content"]
        + "<end_of_turn>\n<start_of_turn>model\n"
    )
    assert "<bos>" not in prompt
    assert "Do not approve INV-1 at 8 PM." in prompt
    assert callable(translator_factory("gemma3-27b-tpu-bf16"))
    with pytest.raises(ValueError, match="independent"):
        format_prompt([*messages, {"role": "assistant", "content": "history"}])


def test_limits_reject_truncation_and_cache_overflow():
    validate_limits(10, 3072, 768, 4096)
    with pytest.raises(ValueError, match="not truncated"):
        validate_limits(3073, 3072, 768, 4096)
    with pytest.raises(ValueError, match="fit"):
        validate_limits(10, 4096, 768, 4096)


def test_end_token_is_required_before_saving_a_translation():
    tokenizer = SimpleNamespace(
        special_tokens=SimpleNamespace(EOS=1, END_OF_TURN=106),
        decode=lambda ids: "سلام" if ids else "",
    )
    assert decode_completed(tokenizer, [42, 106, 0]) == "سلام"
    with pytest.raises(RuntimeError, match="limit"):
        decode_completed(tokenizer, [42, 43])
    with pytest.raises(RuntimeError, match="empty"):
        decode_completed(tokenizer, [1])


def test_large_weights_must_actually_be_sharded():
    leaf = SimpleNamespace(
        dtype="bfloat16", nbytes=8 * 1024**2, sharding=SimpleNamespace(is_fully_replicated=False)
    )
    assert parameter_summary([leaf])["large_sharded_parameters"] == 1
    leaf.sharding.is_fully_replicated = True
    with pytest.raises(RuntimeError, match="replicated"):
        parameter_summary([leaf])


def test_checkpoint_fingerprint_detects_metadata_or_tokenizer_changes(tmp_path):
    (tmp_path / "gemma3-27b-it").mkdir()
    (tmp_path / "gemma3-27b-it/_METADATA").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.model").write_bytes(b"tokenizer")
    original = describe_checkpoint(tmp_path)
    (tmp_path / "tokenizer.model").write_bytes(b"changed")
    assert describe_checkpoint(tmp_path)["fingerprint"] != original["fingerprint"]


def test_generator_loads_sharded_once_and_never_forwards_history(tmp_path, monkeypatch):
    calls = []
    load_calls = []
    leaf = SimpleNamespace(
        dtype="bfloat16",
        nbytes=8 * 1024**2,
        sharding=SimpleNamespace(is_fully_replicated=False),
    )
    tokenizer = SimpleNamespace(
        special_tokens=SimpleNamespace(EOS=1, END_OF_TURN=106),
        encode=lambda text, **kwargs: list(range(len(text) // 20)),
        decode=lambda ids: str(ids[0]),
    )

    def sample(prompt, **kwargs):
        calls.append((prompt, kwargs))
        tokens = [40 + len(calls), 106, 0]
        return SimpleNamespace(
            tokens=SimpleNamespace(tolist=lambda: tokens),
            state=SimpleNamespace(cache=[leaf]),
        )

    gm = SimpleNamespace(
        nn=SimpleNamespace(Gemma3_27B=lambda **kw: kw),
        ckpts=SimpleNamespace(load_params=lambda *a, **kw: load_calls.append((a, kw)) or [leaf]),
        text=SimpleNamespace(
            Gemma3Tokenizer=lambda **kw: tokenizer,
            Greedy=lambda: "greedy",
            Sampler=lambda **kw: SimpleNamespace(sample=sample),
        ),
    )
    kd = SimpleNamespace(sharding=SimpleNamespace(FSDPSharding=lambda: "fsdp", REPLICATED="rep"))
    fake_jax = SimpleNamespace(
        tree=SimpleNamespace(leaves=lambda tree: tree),
        devices=lambda: [],
        block_until_ready=lambda x: x,
        device_get=lambda x: x,
        monitoring=SimpleNamespace(register_event_duration_secs_listener=lambda cb: None),
    )
    monkeypatch.setattr(
        native, "_libraries", lambda: (fake_jax, SimpleNamespace(bfloat16="bf16"), gm, kd)
    )
    monkeypatch.setattr(native, "probe_tpu", lambda: {})
    monkeypatch.setattr(native, "prepare_checkpoint", lambda: (tmp_path, {"fingerprint": "abc"}))
    monkeypatch.setenv("KODOOM_CHECKPOINT_FINGERPRINT", "abc")
    metrics = tmp_path / "metrics.json"
    monkeypatch.setenv("KODOOM_TPU_METRICS", str(metrics))
    generate = native.load_generator()
    messages = chat_messages(Item("Do not approve INV-1.", "formal", "option"), load())
    assert generate([messages, messages]) == ["41", "42"]
    assert generate([messages]) == ["43"]
    assert len(load_calls) == 1
    assert load_calls[0][1] == {"text_only": True, "sharding": "fsdp"}
    assert all("last_state" not in kw and kw["sharding"] == "rep" for _, kw in calls)
    assert metrics.exists()
    monkeypatch.setenv("KODOOM_CHECKPOINT_FINGERPRINT", "different")
    with pytest.raises(RuntimeError, match="identity changed"):
        native.load_generator()
    assert len(load_calls) == 1
