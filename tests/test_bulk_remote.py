import json
import sys
from types import SimpleNamespace

import pytest

from kodoom.bulk.remote import (
    DirectoryRemote,
    HFRemote,
    RemoteSession,
    StorageError,
    WriterConflict,
)
from kodoom.bulk.state import encoded
from tests.test_bulk_state import campaign, translate


def test_friend_session_restores_verified_complete_work(tmp_path):
    backend = DirectoryRemote(tmp_path / "remote")
    first = RemoteSession(backend, "test-campaign", "first", "owner")
    first.acquire()
    store = campaign(tmp_path / "first")
    store.freeze_production({"batch": 2})
    unit = store.pending()[0]
    store.complete(unit, translate(unit))
    revision = first.save(store.snapshot())
    assert revision == first.verified_revision
    first.release()
    second = RemoteSession(backend, "test-campaign", "next", "friend")
    second.acquire()
    assert second.restore(tmp_path / "next")
    restored = campaign(tmp_path / "next")
    assert len(restored.pending()) == 4
    assert restored.translated(unit) == store.translated(unit)
    second.release()
    reader = RemoteSession(backend, "test-campaign", "reader", "reviewer")
    assert reader.restore(tmp_path / "read-only", read_only=True)
    assert campaign(tmp_path / "read-only").counts() == restored.counts()
    with pytest.raises(WriterConflict):
        reader.save(restored.snapshot())


def test_active_and_stale_writers_cannot_replace_progress(tmp_path):
    backend = DirectoryRemote(tmp_path / "remote")
    first = RemoteSession(backend, "test-campaign", "first", "owner")
    second = RemoteSession(backend, "test-campaign", "second", "friend")
    first.acquire()
    with pytest.raises(WriterConflict, match="Previous writer"):
        second.acquire()
    second.acquire(takeover=True)
    with pytest.raises(WriterConflict, match="writer changed"):
        first.save(campaign(tmp_path / "run").snapshot())
    with pytest.raises(WriterConflict):
        first.release()


def test_review_download_does_not_interrupt_an_active_writer(tmp_path, monkeypatch):
    import kodoom.bulk.remote as module
    from kodoom.bulk.launch import download

    backend = DirectoryRemote(tmp_path / "remote")
    writer = RemoteSession(backend, "test-campaign", "active", "owner")
    writer.acquire()
    writer.save(campaign(tmp_path / "run").snapshot())
    head = backend.head()
    monkeypatch.setattr(module, "HFRemote", lambda repo, token: backend)
    result = download(
        campaign_id="test-campaign", repo="org/private", directory=tmp_path / "review-copy"
    )
    assert result["revision"] == head and backend.head() == head
    writer.save(campaign(tmp_path / "run").snapshot())
    with pytest.raises(ValueError, match="fresh directory"):
        download(
            campaign_id="test-campaign", repo="org/private", directory=tmp_path / "review-copy"
        )


def test_corrupt_remote_archive_is_not_restored(tmp_path):
    backend = DirectoryRemote(tmp_path / "remote")
    session = RemoteSession(backend, "test-campaign", "first", "owner")
    session.acquire()
    session.save(campaign(tmp_path / "run").snapshot())
    head = backend.head()
    manifest = json.loads(backend.read(session.prefix + "checkpoint.json", head))
    backend.commit({session.prefix + manifest["bundle"]["path"]: b"corrupted"}, head)
    with pytest.raises(ValueError, match="checksum"):
        session.restore(tmp_path / "next")


def test_local_unsaved_cases_merge_without_retranslation(tmp_path):
    backend = DirectoryRemote(tmp_path / "remote")
    session = RemoteSession(backend, "test-campaign", "first", "owner")
    session.acquire()
    store = campaign(tmp_path / "run")
    store.freeze_production({"batch": 1})
    session.save(store.snapshot())
    unit = store.pending()[0]
    store.complete(unit, translate(unit))
    assert session.restore(store.root)
    assert len(campaign(store.root).pending()) == 4


def test_successful_commit_with_failed_verification_can_retry(tmp_path):
    backend = DirectoryRemote(tmp_path / "remote")
    session = RemoteSession(backend, "test-campaign", "first", "owner")
    session.acquire()
    original = backend.read
    failures = []

    def read(name, revision):
        if name.endswith(".zip") and not failures:
            failures.append(True)
            raise StorageError("transient download failure")
        return original(name, revision)

    backend.read = read
    store = campaign(tmp_path / "run")
    with pytest.raises(StorageError):
        session.save(store.snapshot())
    assert session.save(store.snapshot()) == backend.head()


def test_pause_control_and_private_hf_enforcement(tmp_path, monkeypatch):
    backend = DirectoryRemote(tmp_path / "remote")
    session = RemoteSession(backend, "test-campaign", "first", "owner")
    session.acquire()
    backend.commit({session.prefix + "control.json": encoded({"pause": True})}, backend.head())
    assert session.paused()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda **kwargs: SimpleNamespace(
                repo_info=lambda **kwargs: SimpleNamespace(private=False)
            )
        ),
    )
    with pytest.raises(StorageError, match="private"):
        HFRemote("organization/project", "fake-token")
