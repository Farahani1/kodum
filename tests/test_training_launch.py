import ast
import json
from pathlib import Path

import pytest

from kodoom.schema import RecordError
from kodoom.training.launch import command, run


def test_launch_uses_two_processes_and_separate_preflight_namespace():
    args = command(
        "preflight",
        prepared=Path("input"),
        settings=Path("recipe.toml"),
        backup_repo="owner/private",
        run_id="full-preflight",
    )
    assert "--nproc_per_node=2" in args
    assert args[args.index("--run-id") + 1] == "full-preflight"
    assert "kodoom.training" in args
    with pytest.raises(RecordError):
        command(
            "unknown",
            prepared=Path("input"),
            settings=Path("recipe"),
            backup_repo="owner/private",
            run_id="bad",
        )


def test_handoff_refuses_non_kaggle_before_network(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    with pytest.raises(RecordError, match="Kaggle"):
        run("preflight", backup_repo="owner/private")


def test_training_notebook_is_thin_and_has_no_credentials_or_outputs():
    path = Path("notebooks/training.ipynb")
    if not path.exists():
        pytest.skip("handoff notebook is added with the operational handoff commit")
    notebook = json.loads(path.read_text("utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            tree = ast.parse("".join(cell["source"]))
            assert not any(
                isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.For, ast.While))
                for n in ast.walk(tree)
            )
    text = "".join("".join(c["source"]) for c in notebook["cells"])
    assert 'ACTION = "preflight"' in text
    assert 'CUDA_VISIBLE_DEVICES"] = "0,1"' in text
    assert 'BACKUP_REPO = "Farahani/kodoom"' in text


def test_remote_input_prepares_from_the_full_snapshot(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    import hashlib

    from kodoom.training import launch

    manifest_root = tmp_path / "manifest"
    payload_root = tmp_path / "payload"
    (manifest_root / "provenance").mkdir(parents=True)
    release = {"files": [{"path": "data/selected.parquet"}]}
    manifest = manifest_root / "provenance/release.json"
    manifest.write_text(json.dumps(release), encoding="utf-8")
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
    selection = {
        "dataset": "hf-data",
        "release_version": "1.0.1",
        "schema_version": "1.0.0",
        "transport": "hf-data-bilingual-parquet-v1",
        "manifest_sha256": expected,
        "configurations": ["typed_decisions"],
        "languages": ["en", "fa"],
        "splits": ["train", "test"],
    }
    monkeypatch.setattr(launch, "read_selection", lambda _: selection)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hub-cache"))
    returned = iter((manifest_root, payload_root))
    calls = []

    def snapshot_download(*_args, **kwargs):
        calls.append(kwargs)
        return next(returned)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    prepared_from = []
    monkeypatch.setattr(
        launch, "prepare", lambda source, *_args, **_kwargs: prepared_from.append(source)
    )
    out = tmp_path / "prepared"
    assert launch.remote_input(tmp_path / "selection.toml", out) == out
    assert prepared_from == [payload_root]
    assert calls[1]["allow_patterns"] == [
        "provenance/release.json",
        "data/selected.parquet",
    ]
    # Both stages must materialize files rather than return cache blob links.
    local_dir = str(tmp_path / "hub-cache" / "kodoom-releases" / launch.DATASET_REVISION)
    assert [c["local_dir"] for c in calls] == [local_dir, local_dir]
