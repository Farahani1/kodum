import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import kodoom.training.remote as remote
from kodoom.schema import RecordError


class FakeAPI:
    def upload_folder(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(oid="a" * 40)


def test_sync_never_overwrites_complete_manifest_on_log_only_update(tmp_path, monkeypatch):
    api = FakeAPI()
    monkeypatch.setattr(remote, "private_api", lambda _: api)
    run = SimpleNamespace(dir=tmp_path, meta={"run_id": "full-v1"})
    (tmp_path / "latest").mkdir()
    (tmp_path / "latest/state.pt").write_bytes(b"complete-state")
    (tmp_path / "log.jsonl").write_text("one event\n")
    assert remote.sync(run, "owner/private") == "a" * 40
    manifest = (tmp_path / "durable-manifest.json").read_bytes()
    assert "latest/state.pt" in json.loads(manifest)["files"]
    (tmp_path / "log.jsonl").write_text("newer event\n")
    remote.sync(run, "owner/private", logs_only=True)
    assert (tmp_path / "durable-manifest.json").read_bytes() == manifest
    assert "latest/**" not in api.kwargs["allow_patterns"]
    assert "durable-manifest.json" not in api.kwargs["allow_patterns"]


def test_private_repository_is_mandatory(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(
        hub,
        "HfApi",
        lambda: SimpleNamespace(repo_info=lambda *_a, **_k: SimpleNamespace(private=False)),
    )
    with pytest.raises(RecordError, match="private"):
        remote.private_api("owner/public")


def test_restore_requires_complete_snapshot_checksums(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    root = tmp_path / "cache" / "run"
    (root / "latest").mkdir(parents=True)
    (root / "latest/state.pt").write_bytes(b"corrupt")
    (root / "durable-manifest.json").write_text(
        json.dumps({"files": {"latest/state.pt": {"sha256": "wrong", "bytes": 7}}})
    )
    monkeypatch.setattr(remote, "private_api", lambda _: object())
    monkeypatch.setattr(hub, "snapshot_download", lambda *_a, **_k: str(root.parent))
    with pytest.raises(RecordError, match="snapshot mismatch"):
        remote.restore("owner/private", "a" * 40, "run", tmp_path / "restored")


def test_restore_refuses_output_overwrite(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(remote, "private_api", lambda _: object())
    with pytest.raises(RecordError, match="new output"):
        remote.restore("owner/private", "a" * 40, "run", Path(tmp_path))


def test_restore_creates_a_new_nested_destination(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    root = tmp_path / "cache" / "run"
    (root / "latest").mkdir(parents=True)
    payload = root / "latest/state.pt"
    payload.write_bytes(b"complete")
    (root / "durable-manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "latest/state.pt": {
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                        "bytes": payload.stat().st_size,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(remote, "private_api", lambda _: object())
    monkeypatch.setattr(hub, "snapshot_download", lambda *_a, **_k: str(root.parent))
    target = tmp_path / "new-parent" / "restored"
    assert remote.restore("owner/private", "a" * 40, "run", target) == target
    assert (target / "latest/state.pt").read_bytes() == b"complete"
