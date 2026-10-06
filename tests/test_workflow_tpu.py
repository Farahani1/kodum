import json
import shutil
from dataclasses import replace

import pytest

import kodoom.cli as cli
import kodoom.workflow as workflow
from kodoom.artifacts import verify_bundle
from kodoom.check import Check
from kodoom.config import load_profile
from kodoom.schema import read_jsonl
from kodoom.translate.pipeline import StubTranslator
from tests.test_workflow import fixture_run

REQUEST = workflow.ROOT / "workflows/tpu-gate.toml"


def fake_tpu(tmp_path, monkeypatch):
    base, source = fixture_run(tmp_path, monkeypatch)
    base = replace(base, device="tpu", precision="bf16", backend="jax")
    monkeypatch.setattr(workflow, "load_profile", lambda _: base)
    monkeypatch.setattr(workflow, "check_provider_paths", lambda *args: None)
    monkeypatch.setattr(workflow, "git_commit", lambda: "a" * 40)
    monkeypatch.setattr(workflow, "run_checks", lambda _, **kwargs: [Check("files", "ok", "ready")])
    monkeypatch.setattr(
        workflow,
        "model_preflight",
        lambda *args: {
            "model_revision": "google/gemma-3/flax/gemma3-27b-it/1",
            "checkpoint": {"fingerprint": "pinned"},
            "model_dtype": "bfloat16",
            "device": "TPU v5e",
        },
    )
    request, _ = workflow.load_request(REQUEST)
    root = workflow.artifact_root(base, request, "preflight")
    for dataset in ("helmo", "typed-decisions"):
        target = root / dataset / "en/train.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / dataset / "en/train.jsonl", target)
    loaded = []

    def factory(name):
        def load():
            loaded.append(name)
            translator = StubTranslator()
            translator.name = name
            return translator

        return load

    monkeypatch.setattr(cli, "translator_factory", factory)

    def command(args, profile, log, env):
        assert env["JAX_PLATFORMS"] == "tpu"
        assert env["KODOOM_CHECKPOINT_FINGERPRINT"] == "pinned"
        return cli.main([*args, "--profile", str(profile)])

    monkeypatch.setattr(workflow, "command", command)
    return base, root, loaded


def run(mode="preflight"):
    return workflow.run_current(provider="kaggle", profile="test", mode=mode, request_path=REQUEST)


def test_tpu_gate_fixture_preserves_ids_and_resumes_without_reloading(tmp_path, monkeypatch):
    base, preflight_root, loaded = fake_tpu(tmp_path, monkeypatch)
    assert run()["status"] == "preflight-passed"
    assert len(loaded) == 2
    state = run("gate")
    assert state["status"] == "awaiting-review"
    assert len(loaded) == 4
    assert run("gate")["outputs"] == state["outputs"]
    assert len(loaded) == 4  # Completed records need no checkpoint loading.
    request, _ = workflow.load_request(REQUEST)
    gate_root = workflow.artifact_root(base, request, "gate")
    rows = list(read_jsonl(gate_root / "helmo/fa/gemma3-27b-tpu-bf16/train.jsonl"))
    assert len(rows) == len({r.id for r in rows}) == 40
    verify_bundle(state["bundle"])
    assert preflight_root.parent.name == "gemma3-27b-tpu-bf16"
    assert state["semantic"]["checkpoint"] == "pinned"
    assert load_profile(gate_root / "profile.toml").backend == "jax"


def test_tpu_gate_requires_matching_preflight(tmp_path, monkeypatch):
    _, root, loaded = fake_tpu(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="preflight before"):
        run("gate")
    assert not loaded
    run()
    path = root / "execution.json"
    state = json.loads(path.read_text("utf-8"))
    state["semantic"]["checkpoint"] = "different"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="different model"):
        run("gate")
    assert len(loaded) == 2


def test_tpu_probe_failure_has_private_diagnostics_and_no_commands(tmp_path, monkeypatch):
    _, root, loaded = fake_tpu(tmp_path, monkeypatch)

    def fail(*args):
        raise RuntimeError("no TPU")

    with monkeypatch.context() as failed:
        failed.setattr(workflow, "model_preflight", fail)
        with pytest.raises(RuntimeError, match="no TPU"):
            run()
    assert not loaded
    assert json.loads((root / "execution.json").read_text("utf-8"))["status"] == "failed"
    assert list((tmp_path / "bundles").glob("*.zip"))
    assert run()["status"] == "preflight-passed"  # A failed probe left no translations.


def test_bundle_rejects_another_translator_before_extraction(tmp_path, monkeypatch):
    _, root, _ = fake_tpu(tmp_path, monkeypatch)
    state = run()
    request, _ = workflow.load_request(REQUEST)
    request["translator"] = "gemma3-4b-bf16"
    destination = root.parent / "wrong-model"
    with pytest.raises(ValueError, match="different model"):
        workflow.restore_compatible(state["bundle"], destination, request, "preflight")
    assert not destination.exists()


def test_tpu_request_rejects_wrong_device_and_token_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "load_profile", lambda _: load_profile("dev"))
    with pytest.raises(ValueError, match="jax/tpu"):
        run()
    path = tmp_path / "current.toml"
    path.write_text(
        REQUEST.read_text("utf-8").replace("input_tokens = 3072", "input_tokens = 4096"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cache budget"):
        workflow.load_request(path)
