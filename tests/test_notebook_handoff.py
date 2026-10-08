import ast
import json
import re
import tomllib
from pathlib import Path

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
