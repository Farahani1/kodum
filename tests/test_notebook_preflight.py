import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import kodoom.notebook_preflight as preflight
import kodoom.runtime as runtime


def test_missing_hf_secret_stops_before_checkout_install_or_accelerator(monkeypatch):
    monkeypatch.setattr(runtime, "validate_tpu_runtime", lambda: None)
    monkeypatch.setattr(runtime, "secret", lambda *args: None)
    calls = []
    monkeypatch.setattr(runtime, "bootstrap", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError, match="enable it for this notebook"):
        runtime.cpu_preflight(
            "kaggle",
            "a" * 40,
            "/tmp/code",
            campaign_id="smoke-campaign",
            repo="owner/private-data",
        )
    assert calls == []


def test_checkout_only_bootstrap_never_installs_or_sets_tpu_platform(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "validate_tpu_runtime", lambda: None)
    monkeypatch.setattr(runtime, "secret", lambda *args: None)
    monkeypatch.setattr(runtime.os, "chdir", lambda _: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)

    def run(words, **kwargs):
        calls.append(words)
        return types.SimpleNamespace(returncode=0, stdout="a" * 40)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    runtime.bootstrap("kaggle", "a" * 40, tmp_path, backend="jax", install_dependencies=False)
    assert all(words[0] == "git" for words in calls)
    assert "JAX_PLATFORMS" not in runtime.os.environ


def test_mounted_model_check_never_downloads_and_requires_exact_version(tmp_path, monkeypatch):
    wrong = tmp_path / "gemma-3/flax/gemma3-27b-it/2"
    wrong.mkdir(parents=True)
    (wrong / "tokenizer.model").write_bytes(b"tokenizer")
    with pytest.raises(preflight.CheckFailure, match="version-1"):
        preflight.attached_checkpoint(tmp_path)
    right = tmp_path / "models/google/gemma-3/flax/gemma3-27b-it/1"
    (right / "gemma3-27b-it").mkdir(parents=True)
    (right / "tokenizer.model").write_bytes(b"tokenizer")
    (right / "gemma3-27b-it/_METADATA").write_bytes(b"{}")
    directory, info = preflight.attached_checkpoint(tmp_path)
    assert directory == right and info["handle"] == "google/gemma-3/flax/gemma3-27b-it/1"


class ProbeRemote:
    repo = "owner/private-data"

    def __init__(self, mismatch=False):
        self.files = {}
        self.deleted = []
        self.api = types.SimpleNamespace(create_commit=self.cleanup)
        self.mismatch = mismatch

    def head(self):
        return "head"

    def commit(self, files, parent):
        self.files.update(files)
        return "written-revision"

    def read(self, name, revision):
        assert revision == "written-revision"
        return b"incorrect" if self.mismatch else self.files[name]

    def cleanup(self, **kwargs):
        name = kwargs["operations"][0].path_in_repo
        self.deleted.append(name)
        del self.files[name]


@pytest.mark.parametrize("mismatch", [False, True])
def test_write_probe_verifies_committed_revision_and_cleans_only_its_file(monkeypatch, mismatch):
    module = types.ModuleType("huggingface_hub")
    module.CommitOperationDelete = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    remote = ProbeRemote(mismatch=mismatch)
    if mismatch:
        with pytest.raises(preflight.CheckFailure, match="readback"):
            preflight.write_probe(remote)
    else:
        preflight.write_probe(remote)
    assert remote.files == {}
    assert len(remote.deleted) == 1 and remote.deleted[0].startswith("preflight-probes/")


@pytest.mark.parametrize("kind", ["lease", "paused", "changed-code"])
def test_resume_preflight_detects_conflicts_before_probe_writes(kind, monkeypatch):
    class Remote:
        repo = "owner/private-data"

        def head(self):
            return "head"

        def read(self, name, revision):
            if name.endswith("writer.json") and kind == "lease":
                return b'{"released":false}'
            if name.endswith("control.json") and kind == "paused":
                return b'{"pause":true}'
            if name.endswith("checkpoint.json"):
                return b"{}"
            return None

    def download(**kwargs):
        (Path(kwargs["directory"]) / "campaign.json").write_text(
            json.dumps({"code_revision": "old", "request": {}}), encoding="utf-8"
        )
        return {"counts": {}}

    monkeypatch.setattr(preflight, "download", download)
    with pytest.raises(preflight.CheckFailure):
        preflight.resume_check(Remote(), "smoke-campaign", "new", False)


def test_failed_access_is_reported_without_secret_text_and_dependent_checks_skip(
    tmp_path, monkeypatch, capsys
):
    leaked = "fixture-secret-must-not-appear"
    monkeypatch.setattr(preflight, "validate_tpu_runtime", lambda: None)
    monkeypatch.setattr(preflight, "host_resources", lambda: {})
    profile = types.SimpleNamespace(data_dir=tmp_path, runs_dir=tmp_path, scratch_dir=tmp_path)
    monkeypatch.setattr(preflight, "load_profile", lambda _: profile)

    def failed(*args):
        raise RuntimeError(leaked)

    monkeypatch.setattr(preflight, "HFRemote", failed)
    monkeypatch.setattr(preflight, "attached_checkpoint", failed)
    monkeypatch.setattr(preflight, "prepare_units", failed)
    monkeypatch.setattr(preflight, "dependency_resolution", lambda: "resolved")
    report = tmp_path / "report.json"
    result = preflight.run_checks(
        repo="owner/private-data",
        campaign_id="smoke-campaign",
        operator="owner",
        revision="a" * 40,
        takeover=False,
        report_path=report,
    )
    assert result["passed"] is False
    assert leaked not in report.read_text("utf-8") + capsys.readouterr().out
    statuses = {row["check"]: row["status"] for row in result["checks"]}
    assert statuses["private HF access"] == "FAIL"
    assert statuses["private HF write/readback"] == "SKIP"
    assert statuses["prompt limits"] == "SKIP"
    assert statuses["TPU dependency resolution"] == "PASS"


def test_cpu_dependency_probe_does_not_install_or_import_jax(tmp_path, monkeypatch):
    calls = []

    def run(words, **kwargs):
        calls.append(words)
        report = Path(words[words.index("--report") + 1])
        report.write_text('{"install":[{},{}]}', encoding="utf-8")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(preflight.subprocess, "run", run)
    assert "2 pinned packages" in preflight.dependency_resolution()
    assert "--dry-run" in calls[0] and "--ignore-installed" in calls[0]
    assert not any("import jax" in word for word in calls[0])


def test_preflight_module_import_has_no_accelerator_side_effects():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kodoom.notebook_preflight; "
            "assert not {'jax', 'gemma', 'torch', 'tensorflow'} & set(sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_all_checks_pass_only_when_every_prerequisite_succeeds(tmp_path, monkeypatch):
    profile = types.SimpleNamespace(data_dir=tmp_path, runs_dir=tmp_path, scratch_dir=tmp_path)
    monkeypatch.setattr(preflight, "validate_tpu_runtime", lambda: None)
    monkeypatch.setattr(preflight, "host_resources", lambda: {})
    monkeypatch.setattr(preflight, "load_profile", lambda _: profile)
    monkeypatch.setattr(preflight, "HFRemote", lambda *args: object())
    monkeypatch.setattr(preflight, "resume_check", lambda *args: "new")
    monkeypatch.setattr(preflight, "write_probe", lambda *args: "verified")
    monkeypatch.setattr(preflight, "attached_checkpoint", lambda: (tmp_path, {}))
    monkeypatch.setattr(preflight, "prepare_units", lambda *args: [])
    monkeypatch.setattr(preflight, "validate_scope", lambda *args: None)
    monkeypatch.setattr(preflight, "prompt_audit", lambda *args: "in budget")
    monkeypatch.setattr(preflight, "dependency_resolution", lambda: "resolved")
    result = preflight.run_checks(
        repo="owner/private-data",
        campaign_id="smoke-campaign",
        operator="owner",
        revision="a" * 40,
        takeover=False,
        report_path=tmp_path / "report.json",
    )
    assert result["passed"] and all(row["status"] == "PASS" for row in result["checks"])


def test_overlong_production_prompt_fails_on_cpu(tmp_path, monkeypatch):
    module = types.ModuleType("sentencepiece")
    module.SentencePieceProcessor = lambda **kw: types.SimpleNamespace(
        encode=lambda *a, **kw: [1] * 11
    )
    monkeypatch.setitem(sys.modules, "sentencepiece", module)
    monkeypatch.setattr(preflight, "requests", lambda unit: [[]])
    monkeypatch.setattr(preflight, "format_prompt", lambda messages: "prompt")
    units = [types.SimpleNamespace(key="typed-case")]
    with pytest.raises(preflight.CheckFailure, match="exceed 10"):
        preflight.prompt_audit(units, tmp_path, {"input_tokens": 10})
