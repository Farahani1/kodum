import json
import shutil
from pathlib import Path

import pytest

from kodoom.cli import main
from kodoom.dataset_lock import consume_release, read_selection
from kodoom.package_import import file_sha
from kodoom.schema import RecordError
from tests.test_package_import import bilingual_row, make_package


def selection_for(root, rows=None):
    make_package(root, rows)
    manifest = root / "provenance/release.json"
    release = json.loads(manifest.read_text("utf-8"))
    release.update(release_version="1.0.1", schema_version="1.0.0")
    manifest.write_text(json.dumps(release), encoding="utf-8")
    selection = read_selection(Path("configs/dataset-baseline.toml"))
    return {**selection, "manifest_sha256": file_sha(manifest), "splits": ["train"]}


def test_consumer_keeps_raw_text_targets_quality_and_portable_lock(tmp_path):
    root = tmp_path / "source"
    row = bilingual_row()
    selection = selection_for(root, [row])
    en, fa = consume_release(root, selection)[0]
    assert en.id == row["question_id"] and fa.id == en.id + ":fa"
    assert en.source_id == fa.source_id == row["source_id"]
    assert fa.gold == en.gold
    assert fa.question_text == row["question_fa"]  # Arabic yeh stays raw
    assert fa.state == row["state_fa"]
    assert [o.text for o in fa.options] == [o["text_fa"] for o in row["options"]]
    assert fa.checks_passed is None and fa.human_reviewed is False
    assert fa.extra["supplied_package"]["quality"] == row["quality"]
    assert fa.extra["supplied_package"]["translation"] == row["translation"]
    assert "training_eligibility" not in fa.extra
    _, lock = consume_release(root, selection)
    assert lock["payloads"][0]["sha256"] == file_sha(root / "data/source.parquet")
    other = tmp_path / "relocated"
    shutil.copytree(root, other)
    _, relocated = consume_release(other, {**selection, "languages": ["fa", "en"]})
    assert relocated["fingerprint"] == lock["fingerprint"]
    assert relocated["source_root"] != lock["source_root"]
    _, fa_lock = consume_release(root, {**selection, "languages": ["fa"]})
    assert fa_lock["fingerprint"] != lock["fingerprint"]


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"release_version": "latest"}, "exact"),
        ({"schema_version": "2.0.0"}, "migrate"),
        ({"transport": "kodoom-record"}, "migrate"),
        ({"configurations": ["original_typed_decisions"]}, "configurations"),
        ({"configurations": ["belebele"]}, "configurations"),
        ({"languages": []}, "languages"),
        ({"languages": ["fa", "fa"]}, "languages"),
        ({"splits": ["test"]}, "unavailable"),
        ({"manifest_sha256": "0" * 64}, "pinned"),
        ({"release_version": "1.0.2"}, "differs"),
        ({"unexpected": True}, "contract keys"),
    ],
)
def test_fail_closed_selection(tmp_path, updates, match):
    selection = selection_for(tmp_path)
    with pytest.raises(RecordError, match=match):
        consume_release(tmp_path, {**selection, **updates})


def test_cli_prints_lock_without_writing_source(tmp_path, capsys):
    selection = selection_for(tmp_path)
    config = tmp_path / "selection.toml"
    config.write_text(
        "\n".join(f"{key} = {json.dumps(value)}" for key, value in selection.items()),
        encoding="utf-8",
    )
    assert main(["dataset-lock", "--source", str(tmp_path), "--selection", str(config)]) == 0
    lock = json.loads(capsys.readouterr().out)
    assert lock["records"] == 2
    (tmp_path / "data/source.parquet").write_bytes(b"tampered")
    assert main(["dataset-lock", "--source", str(tmp_path), "--selection", str(config)]) == 1
    assert "inventory mismatch" in capsys.readouterr().err


def test_reject_duplicate_decisions(tmp_path):
    selection = selection_for(tmp_path, [bilingual_row(), bilingual_row()])
    with pytest.raises(RecordError, match="duplicate decision"):
        consume_release(tmp_path, selection)


@pytest.mark.parametrize("problem", ["schema", "inventory", "table", "count", "split"])
def test_reject_rehashed_but_incompatible_release(tmp_path, problem):
    selection = selection_for(tmp_path)
    manifest = tmp_path / "provenance/release.json"
    release = json.loads(manifest.read_text("utf-8"))
    if problem == "schema":
        release["schema_version"] = "1.1.0"
    elif problem == "inventory":
        release["files"] = []
    elif problem == "table":
        release["configurations"] *= 2
    elif problem == "count":
        release["configurations"][0]["rows"] += 1
    else:
        release["configurations"][0]["split"] = "test"
        selection["splits"] = ["test"]
    manifest.write_text(json.dumps(release), encoding="utf-8")
    selection["manifest_sha256"] = file_sha(manifest)
    with pytest.raises(RecordError):
        consume_release(tmp_path, selection)
