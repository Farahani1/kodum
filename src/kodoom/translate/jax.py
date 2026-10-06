"""Stateless, sharded Gemma generation. Heavy imports occur only on the TPU path."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from kodoom.tpu import host_resources, memory_snapshot, probe_tpu
from kodoom.translate.checkpoint import KAGGLE_MODEL, prepare_checkpoint
from kodoom.translate.hf import ChatTranslator, Messages

NAME = "gemma3-27b-tpu-bf16"


def format_prompt(messages: Messages) -> str:
    """Gemma 3 IT template without BOS (the native sampler adds it once).

    The translation adapter has exactly a system message and one user turn.
    Gemma folds the system instruction into the first user turn, separated by
    two newlines, as the HF Gemma 3 template does.
    """
    if [message.get("role") for message in messages] != ["system", "user"]:
        raise ValueError("TPU translation requires an independent system/user conversation")
    if not all(isinstance(message.get("content"), str) for message in messages):
        raise ValueError("TPU translation accepts text content only")
    text = messages[0]["content"].strip() + "\n\n" + messages[1]["content"].strip()
    return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"


def validate_limits(input_tokens: int, input_limit: int, output_limit: int, cache_length: int):
    if min(input_limit, output_limit) < 1 or input_limit + output_limit > cache_length:
        raise ValueError("Input/output budgets must fit the TPU cache")
    if input_tokens > input_limit:
        raise ValueError(
            f"Prompt has {input_tokens} tokens, exceeding {input_limit}; not truncated"
        )


def decode_completed(tokenizer, tokens: list[int]) -> str:
    stop = {int(tokenizer.special_tokens.EOS), int(tokenizer.special_tokens.END_OF_TURN)}
    end = next((index for index, token in enumerate(tokens) if token in stop), None)
    if end is None:
        raise RuntimeError("Translation reached its output limit without an end token; not saved")
    text = tokenizer.decode(tokens[:end]).strip()
    if not text:
        raise RuntimeError("Gemma returned an empty translation")
    return text


def parameter_summary(leaves: list) -> dict:
    if not leaves:
        raise RuntimeError("Checkpoint has no model parameters")
    dtypes = sorted({str(leaf.dtype) for leaf in leaves})
    if "bfloat16" not in dtypes:
        raise RuntimeError("Checkpoint does not contain BF16 model weights")
    large = [leaf for leaf in leaves if leaf.nbytes > 4 * 1024**2]
    if not large or any(leaf.sharding.is_fully_replicated for leaf in large):
        raise RuntimeError("Large Gemma parameters were replicated instead of sharded")
    if any(str(leaf.dtype) != "bfloat16" for leaf in large):
        raise RuntimeError("Large Gemma weights must be BF16")
    return {
        "parameter_dtypes": dtypes,
        "parameter_bytes": sum(leaf.nbytes for leaf in leaves),
        "large_sharded_parameters": len(large),
        "mesh": {"devices": 8},
        "sharding": "Kauldron FSDPSharding, largest divisible dimension, 4MiB threshold",
    }


def _libraries():
    try:
        import jax
        import jax.numpy as jnp
        from gemma import gm
        from kauldron import kd
    except ImportError as exc:
        raise RuntimeError(
            "Run the Kaggle bootstrap with backend='jax' to install TPU packages"
        ) from exc
    return jax, jnp, gm, kd


def load_generator():
    if int(os.environ.get("KODOOM_TRANSLATION_BATCH_SIZE", "1")) != 1:
        raise ValueError("The initial TPU translator supports batch size 1")
    input_limit = int(os.environ.get("KODOOM_INPUT_TOKENS", "3072"))
    output_limit = int(os.environ.get("KODOOM_OUTPUT_TOKENS", "768"))
    cache_length = int(os.environ.get("KODOOM_CACHE_TOKENS", "4096"))
    validate_limits(0, input_limit, output_limit, cache_length)
    host = host_resources()
    probe_tpu()
    directory, checkpoint = prepare_checkpoint()
    expected = os.environ.get("KODOOM_CHECKPOINT_FINGERPRINT")
    if expected and expected != checkpoint["fingerprint"]:
        raise RuntimeError("Checkpoint identity changed since preflight; cannot resume")
    jax, jnp, gm, kd = _libraries()
    compile_times = {}

    def compiled(event, duration, **_metadata):
        if event.startswith("/jax/core/compile/"):
            compile_times[event] = compile_times.get(event, 0.0) + duration

    jax.monitoring.register_event_duration_secs_listener(compiled)
    tokenizer = gm.text.Gemma3Tokenizer(path=directory / "tokenizer.model")
    prompt_lengths = []
    measured_conversations = []
    if path := os.environ.get("KODOOM_PROMPTS"):
        measured_conversations = json.loads(Path(path).read_text("utf-8"))
        prompt_lengths = [
            len(tokenizer.encode(format_prompt(msg), add_bos=True))
            for msg in measured_conversations
        ]
        for length in prompt_lengths:
            validate_limits(length, input_limit, output_limit, cache_length)
    started = time.perf_counter()
    print("Loading Gemma 3 27B directly with eight-device FSDP sharding...", flush=True)
    model = gm.nn.Gemma3_27B(dtype=jnp.bfloat16)
    params = gm.ckpts.load_params(
        directory / "gemma3-27b-it", text_only=True, sharding=kd.sharding.FSDPSharding()
    )
    jax.block_until_ready(params)
    info = {
        "checkpoint": checkpoint,
        "host_before_load": host,
        "model": KAGGLE_MODEL,
        "load_seconds": time.perf_counter() - started,
        **parameter_summary(jax.tree.leaves(params)),
        "devices_after_load": memory_snapshot(jax.devices()),
        "calls": [],
        "compilation_seconds": compile_times,
        "input_lengths": {
            "items": len(prompt_lengths),
            "max": max(prompt_lengths, default=0),
            "min": min(prompt_lengths, default=0),
        },
    }
    sampler = gm.text.Sampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
        sampling=gm.text.Greedy(),
        cache_length=cache_length,
        max_out_length=output_limit,
        pad_length=(256, 512, 1024, 2048, input_limit),
    )

    def save_info():
        if path := os.environ.get("KODOOM_TPU_METRICS"):
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8")

    save_info()
    print(
        f"Sharded weights ready in {info['load_seconds']:.1f}s; dtypes: {info['parameter_dtypes']}",
        flush=True,
    )
    phase = "translation"

    def generate(conversations):
        prompts = [format_prompt(messages) for messages in conversations]
        # Validate the entire batch before generating any item; do not truncate.
        lengths = [len(tokenizer.encode(prompt, add_bos=True)) for prompt in prompts]
        for length in lengths:
            validate_limits(length, input_limit, output_limit, cache_length)
        replies = []
        for prompt, length in zip(prompts, lengths, strict=True):
            if time.monotonic() >= float(os.environ.get("KODOOM_DEADLINE", "inf")):
                raise RuntimeError(
                    "TPU stage time budget exhausted; save and resume the partial bundle"
                )
            before_compile = dict(compile_times)
            started = time.perf_counter()
            try:
                result = sampler.sample(
                    prompt,
                    max_new_tokens=output_limit,
                    return_state=True,
                    sharding=kd.sharding.REPLICATED,
                    rng=0,
                )
                jax.block_until_ready(result.state)
            except BaseException as exc:
                info["failure"] = {
                    "type": type(exc).__name__,
                    "input_tokens": length,
                    "phase": phase,
                    "seconds": time.perf_counter() - started,
                    "devices": memory_snapshot(jax.devices()),
                }
                save_info()
                raise
            tokens = jax.device_get(result.tokens).tolist()
            elapsed = time.perf_counter() - started
            cache_leaves = jax.tree.leaves(result.state.cache)
            compilation = {
                key: value - before_compile.get(key, 0.0) for key, value in compile_times.items()
            }
            compile_seconds = compilation.get("/jax/core/compile/backend_compile_duration", 0.0)
            info["calls"].append(
                {
                    "input_tokens": length,
                    "seconds": elapsed,
                    "phase": phase,
                    "compilation_seconds": compilation,
                    "warm": compile_seconds == 0,
                    "output_tokens": next(
                        (i for i, t in enumerate(tokens) if t in (1, 106)), len(tokens)
                    ),
                    "cache_dtypes": sorted({str(leaf.dtype) for leaf in cache_leaves}),
                    "cache_bytes": sum(leaf.nbytes for leaf in cache_leaves),
                    "cache_replicated": all(
                        leaf.sharding.is_fully_replicated for leaf in cache_leaves
                    ),
                    "devices": memory_snapshot(jax.devices()),
                }
            )
            save_info()
            replies.append(decode_completed(tokenizer, tokens))
            # Never forward last_state: every translation starts a fresh conversation.
            del result
        return replies

    if os.environ.get("KODOOM_TPU_WARMUP") == "1" and prompt_lengths:
        phase = "preflight-warmup"
        indexes = sorted(
            {
                min(range(len(prompt_lengths)), key=prompt_lengths.__getitem__),
                max(range(len(prompt_lengths)), key=prompt_lengths.__getitem__),
            }
        )
        warmup = [measured_conversations[i] for i in indexes]
        print("Measuring short/long prompts twice for compilation and warm timing...", flush=True)
        generate(warmup)
        generate(warmup)
        phase = "translation"
    return generate


def gemma3_27b_tpu_bf16() -> ChatTranslator:
    return ChatTranslator(NAME, load_generator())
