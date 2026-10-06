import sys
import types

import pytest

import kodoom.runtime as runtime
from kodoom.runtime import secret


def test_generic_secrets_use_only_the_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fixture")
    assert secret("generic", "HF_TOKEN") == "fixture"
    monkeypatch.delenv("HF_TOKEN")
    assert secret("generic", "HF_TOKEN") is None


@pytest.mark.parametrize("provider", ["kaggle", "colab"])
def test_provider_secret_adapters_are_lazy_and_optional(provider, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    if provider == "kaggle":
        module = types.ModuleType("kaggle_secrets")
        module.UserSecretsClient = lambda: types.SimpleNamespace(get_secret=lambda _: "fixture")
        monkeypatch.setitem(sys.modules, "kaggle_secrets", module)
    else:
        module = types.ModuleType("google.colab")
        module.userdata = types.SimpleNamespace(get=lambda _: "fixture")
        monkeypatch.setitem(sys.modules, "google.colab", module)
    assert secret(provider, "HF_TOKEN") == "fixture"


def test_bootstrap_keeps_auth_out_of_commands_and_saved_remotes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        runtime, "secret", lambda _, name: "test-token" if name == "GITHUB_TOKEN" else None
    )
    monkeypatch.setattr(runtime.os, "chdir", lambda _: None)
    monkeypatch.setattr(sys, "path", list(sys.path))

    def run(words, **kwargs):
        calls.append((words, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="a" * 40 if "rev-parse" in words else "")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    runtime.bootstrap("generic", "fixed-revision", tmp_path / "code")
    assert calls
    assert all("test-token" not in " ".join(words) for words, _ in calls)
    assert all("test-token" not in word for words, _ in calls for word in words)
    assert any("extraheader" in str(kwargs.get("env")) for _, kwargs in calls)


def test_bootstrap_refuses_dirty_checkouts(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(runtime, "secret", lambda *args: None)
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *a, **kw: types.SimpleNamespace(returncode=0, stdout=" M edited.py"),
    )
    with pytest.raises(RuntimeError, match="local edits"):
        runtime.bootstrap("generic", "revision", tmp_path)


def test_tpu_bootstrap_selects_dependencies_before_imports(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.platform, "libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setattr(runtime, "secret", lambda *args: None)
    monkeypatch.setattr(runtime.os, "chdir", lambda _: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    def run(words, **kwargs):
        calls.append(words)
        return types.SimpleNamespace(returncode=0, stdout="a" * 40 if "rev-parse" in words else "")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    runtime.bootstrap("kaggle", "fixed", tmp_path / "code", backend="jax")
    install = next(words for words in calls if "pip" in words)
    assert any(word.endswith("[tpu]") for word in install)
    assert install[-1].endswith("tpu.txt")
    assert "CUDA_VISIBLE_DEVICES" not in runtime.os.environ
    assert runtime.os.environ["JAX_PLATFORMS"] == "tpu"
    assert any("probe_tpu" in " ".join(words) for words in calls)


def test_invalid_backend_stops_without_checkout(tmp_path):
    with pytest.raises(ValueError, match="backend"):
        runtime.bootstrap("generic", "fixed", tmp_path / "code", backend="unknown")
    assert not (tmp_path / "code").exists()


@pytest.mark.parametrize("python,libc", [((3, 10), "2.35"), ((3, 13), "2.35"), ((3, 11), "2.28")])
def test_tpu_runtime_rejects_incompatible_python_or_wheel_platform(monkeypatch, python, libc):
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.sys, "version_info", python)
    monkeypatch.setattr(runtime.platform, "libc_ver", lambda: ("glibc", libc))
    with pytest.raises(RuntimeError):
        runtime.validate_tpu_runtime()
