import ast
import json
import re
import sys
import tomllib
import types
import urllib.request
from pathlib import Path

import pytest

from kodoom.config import load_profile
from kodoom.workflow import ROOT, load_request


def test_operational_notebook_settings_and_summaries_match_the_active_request():
    notebook = json.loads((ROOT / "notebooks/execution.ipynb").read_text("utf-8"))
    request, recipes = load_request()
    settings = ast.parse("".join(notebook["cells"][1]["source"]))
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in settings.body
        if isinstance(node, ast.Assign)
    }
    profile = load_profile(values["PROFILE"])
    assert values["PROVIDER"] == "kaggle"
    assert values["BACKEND"] == request["backend"] == profile.backend == "jax"
    assert profile.device == "tpu" and profile.precision == request["precision"] == "bf16"
    assert re.fullmatch(r"[0-9a-f]{40}", values["CODE_REVISION"])
    assert values["TAKEOVER"] is False
    assert values["RUN_TPU"] is False
    assert notebook["metadata"]["kaggle"]["accelerator"] == "none"
    assert values["HF_DATASET_REPO"] == ""
    assert request["stage"] == recipes[0]["id"] == "bulk-translation"
    assert request["typed_train_cases"] == 1200 and request["typed_test_cases"] == 400
    assert request["helmo_limit"] == 2000 and request["batch_sizes"] == [1, 2, 4, 8]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            code = "".join(cell["source"])
            tree = ast.parse(code)
            assert not any(
                isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.For, ast.While))
                for node in ast.walk(tree)
            )
            compile(code, "execution.ipynb", "exec")
    text = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert request["translator"] in text and request["stop"] in text
    assert "v21" in text and "backend=BACKEND" in text
    assert "from kodoom.bulk.launch import run" in text
    assert "session_started=SESSION_STARTED" in text
    assert "runtime.cpu_preflight(" in text and "CPU PREFLIGHT PASSED" in text
    assert text.index("runtime.cpu_preflight(") < text.index("runtime.bootstrap(")
    reference = (ROOT / "notebooks/reference.ipynb").read_text("utf-8")
    assert "Gemma 3 27B" in reference and "v21" in reference and "v19" in reference
    assert tomllib.loads((ROOT / "workflows/tpu-bulk.toml").read_text("utf-8")) == request


def test_cpu_run_all_skips_bootstrap_and_translation(capsys):
    notebook = json.loads((ROOT / "notebooks/execution.ipynb").read_text("utf-8"))
    namespace = {"RUN_TPU": False}
    # These final cells must execute without a runtime, package import, TPU or HF writer.
    exec("".join(notebook["cells"][5]["source"]), namespace)
    exec("".join(notebook["cells"][7]["source"]), namespace)
    assert namespace["result"] is None
    assert "CPU-only mode" in capsys.readouterr().out


@pytest.mark.parametrize("cached_token", [False, True])
def test_public_bootstrap_ignores_attached_and_cached_github_tokens(
    tmp_path, monkeypatch, cached_token
):
    import io

    import kodoom.runtime as runtime

    notebook = json.loads((ROOT / "notebooks/execution.ipynb").read_text("utf-8"))
    namespace = {}
    exec("".join(notebook["cells"][1]["source"]), namespace)
    namespace["HF_DATASET_REPO"] = "owner/private-dataset"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    if cached_token:
        monkeypatch.setenv("GITHUB_TOKEN", "expired-github-fixture")
    requested_secrets = []

    def get_secret(name):
        requested_secrets.append(name)
        return "dataset-fixture" if name == "HF_TOKEN" else "expired-github-fixture"

    secrets_module = types.ModuleType("kaggle_secrets")
    secrets_module.UserSecretsClient = lambda: types.SimpleNamespace(get_secret=get_secret)
    monkeypatch.setitem(sys.modules, "kaggle_secrets", secrets_module)
    requests = []

    def urlopen(url, timeout):
        requests.append((url, timeout))
        return io.BytesIO((ROOT / "src/kodoom/runtime.py").read_bytes())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(sys, "version_info", (3, 13))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(runtime.platform, "libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setattr(runtime.os, "chdir", lambda _: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    calls = []

    def run(words, **kwargs):
        calls.append((words, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="a" * 40 if "rev-parse" in words else "")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    # Stop before remote HF writes/dependency checks; exercise both checkout calls below.
    cell = ast.parse("".join(notebook["cells"][3]["source"]))
    assert cell.body[-1].targets[0].id == "preflight"
    cell.body.pop()
    exec(compile(cell, "execution.ipynb", "exec"), namespace)
    loaded_runtime = namespace["runtime"]
    assert requests == [
        (
            "https://raw.githubusercontent.com/Farahani1/kodum/"
            + namespace["CODE_REVISION"]
            + "/src/kodoom/runtime.py",
            30,
        )
    ]
    for install in (False, True):
        loaded_runtime.bootstrap(
            "kaggle",
            namespace["CODE_REVISION"],
            tmp_path,
            backend="jax",
            install_dependencies=install,
        )
    assert requested_secrets == ["HF_TOKEN"]
    assert runtime.os.environ["HF_TOKEN"] == "dataset-fixture"
    git_calls = [kwargs for words, kwargs in calls if words[0] == "git"]
    assert git_calls
    assert all("extraheader" not in str(kwargs["env"]) for kwargs in git_calls)
    assert "expired-github-fixture" not in " ".join(request for request, _ in requests)


def test_failed_preflight_cannot_enter_tpu_bootstrap():
    import pytest

    notebook = json.loads((ROOT / "notebooks/execution.ipynb").read_text("utf-8"))
    namespace = {"RUN_TPU": True, "preflight": {"passed": False}}
    with pytest.raises(RuntimeError, match="must pass"):
        exec("".join(notebook["cells"][5]["source"]), namespace)


def test_historical_tpu_request_remains_selectable():
    request, recipes = load_request(ROOT / "workflows/tpu-gate.toml")
    assert request["stage"] == "free-text-gate" and len(recipes) == 2


def test_historical_cuda_request_remains_selectable():
    request, recipes = load_request(Path(ROOT / "workflows/gpu-gate.toml"))
    assert request["translator"] == "gemma3-4b-bf16"
    assert request["model"] == "google/gemma-3-4b-it"
    assert len(recipes) == 2
