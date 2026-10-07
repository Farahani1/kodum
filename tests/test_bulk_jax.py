from types import SimpleNamespace

import kodoom.translate.jax as native


def test_native_batching_restores_order_pads_tail_and_retains_one_model(tmp_path, monkeypatch):
    calls, loads = [], []
    leaf = SimpleNamespace(
        dtype="bfloat16", nbytes=8 * 1024**2, sharding=SimpleNamespace(is_fully_replicated=False)
    )
    tokenizer = SimpleNamespace(
        special_tokens=SimpleNamespace(EOS=1, END_OF_TURN=106),
        encode=lambda text, **kwargs: list(range(len(text))),
        decode=lambda tokens: str(tokens[0]),
    )

    def sample(prompts, **kwargs):
        calls.append((prompts, kwargs))
        rows = [
            [int(p.split("\n\n")[1].split("<end_of_turn>")[0].strip()[0]) + 40, 106, 0]
            for p in prompts
        ]
        rows = [[44, 0, 0] if row[0] == 44 else row for row in rows]
        return SimpleNamespace(
            tokens=SimpleNamespace(tolist=lambda: rows), state=SimpleNamespace(cache=[leaf])
        )

    gm = SimpleNamespace(
        nn=SimpleNamespace(Gemma3_27B=lambda **kwargs: kwargs),
        ckpts=SimpleNamespace(load_params=lambda *args, **kwargs: loads.append(True) or [leaf]),
        text=SimpleNamespace(
            Gemma3Tokenizer=lambda **kwargs: tokenizer,
            Greedy=lambda: None,
            Sampler=lambda **kwargs: SimpleNamespace(sample=sample),
        ),
    )
    jax = SimpleNamespace(
        tree=SimpleNamespace(leaves=lambda value: value),
        devices=lambda: [],
        block_until_ready=lambda value: value,
        device_get=lambda value: value,
        monitoring=SimpleNamespace(register_event_duration_secs_listener=lambda cb: None),
    )
    kd = SimpleNamespace(sharding=SimpleNamespace(FSDPSharding=lambda: "fsdp", REPLICATED="rep"))
    monkeypatch.setattr(
        native, "_libraries", lambda: (jax, SimpleNamespace(bfloat16="bf16"), gm, kd)
    )
    monkeypatch.setattr(native, "probe_tpu", lambda: {})
    monkeypatch.setattr(native, "host_resources", lambda: {})
    monkeypatch.setattr(
        native, "prepare_checkpoint", lambda: (tmp_path, {"fingerprint": "fixture"})
    )
    generate = native.load_generator(bulk=True)
    messages = [
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": f"{i} " + "x" * length},
        ]
        for i, length in [(1, 200), (2, 10), (3, 100)]
    ]
    assert generate.batch(messages, 2) == ["41", "42", "43"]
    assert len(calls) == 2 and all(len(prompt) == 2 for prompt, _ in calls)
    assert calls[-1][0][0] == calls[-1][0][1]
    assert generate.batch(messages[:1], 2) == ["41"]
    assert len(loads) == 1
    assert all("last_state" not in kwargs for _, kwargs in calls)
    assert generate.info["calls"][-1]["items"] == 1
    incomplete = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "4 unfinished"},
    ]
    assert generate.batch([incomplete, messages[0]], 2) == [None, "41"]
