import csv
import json
import threading

import pytest

from kodoom.bulk.data import load_request
from kodoom.bulk.prompts import StubGenerator
from kodoom.bulk.remote import DirectoryRemote, RemoteSession, StorageError, WriterConflict
from kodoom.bulk.saving import AutoSaver, StopWork
from kodoom.bulk.worker import benchmark, run_campaign
from tests.test_bulk_state import campaign, translate


def smoke(tmp_path, name="first", **kwargs):
    return run_campaign(
        root=tmp_path / name,
        campaign_id="fixture-campaign",
        provider="generic",
        profile="dev",
        dev=True,
        local_remote=tmp_path / "remote",
        **kwargs,
    )


def test_interrupted_campaign_resumes_only_pending_work_on_fresh_session(tmp_path):
    first = smoke(tmp_path, max_units=2)
    assert first["persisted"] and first["error_type"] is None
    assert first["status"] == "smoke-unit-limit"
    assert sum(r["completed"] for r in first["counts"].values()) == 2
    saved = {p.name: p.read_bytes() for p in (tmp_path / "first/shards").glob("*.jsonl")}
    loaded = []

    def factory():
        loaded.append(True)
        return StubGenerator()

    second = smoke(tmp_path, "next", operator="collaborator", generator_factory=factory)
    assert second["status"] == "drafts-complete-awaiting-review"
    assert second["error_type"] is None and second["persisted"]
    assert loaded == [True] and not second["hardware_evidence"]
    assert sum(r["completed"] for r in second["counts"].values()) == 5
    for name, data in saved.items():
        assert (tmp_path / "next/shards" / name).read_bytes() == data
    finished = smoke(tmp_path, "done", generator_factory=lambda: pytest.fail("model loaded"))
    assert finished["status"] == "drafts-complete-awaiting-review"
    assert len(list((tmp_path / "done/review").glob("*.csv"))) == 10
    for path in (tmp_path / "done/review").glob("labels-*.csv"):
        with path.open(encoding="utf-8-sig") as stream:
            assert all(row["meaning_changed"] == "" for row in csv.DictReader(stream))
    for path in (tmp_path / "done/shards").glob("*.jsonl"):
        for record in json.loads(path.read_bytes())["persian"]:
            assert not record["human_reviewed"]


def test_failed_whole_record_stays_pending_while_neighbors_finish(tmp_path):
    class Broken(StubGenerator):
        def batch(self, messages, size):
            values = super().batch(messages, size)
            if len(self.info["calls"]) <= 12:
                return values
            return ["{broken" if value.startswith("{") else value for value in values]

    first = smoke(tmp_path, generator_factory=Broken)
    assert first["persisted"] and first["status"] == "drafts-incomplete-awaiting-review"
    assert sum(row["failed"] for row in first["counts"].values()) == 3
    second = smoke(tmp_path, "fixed")
    assert sum(row["failed"] for row in second["counts"].values()) == 0
    assert second["status"] == "drafts-complete-awaiting-review"


def test_upload_outage_stops_new_work_at_unsaved_budget_then_recovers(tmp_path, monkeypatch):
    store = campaign(tmp_path / "run")
    store.freeze_production({"batch": 1})
    remote = RemoteSession(DirectoryRemote(tmp_path / "remote"), "test-campaign", "one", "owner")
    remote.acquire()
    now = [0]
    saver = AutoSaver(store, remote, load_request(), clock=lambda: now[0])
    saver.flush()
    unit = store.pending()[0]
    store.complete(unit, translate(unit))
    original = remote.save
    monkeypatch.setattr(remote, "save", lambda _: (_ for _ in ()).throw(StorageError("outage")))
    saver.stop_event.set()  # Retry immediately in this failure-injection test.
    with pytest.raises(StorageError):
        saver.flush()
    now[0] = 1801
    with pytest.raises(StopWork, match="remote-save-overdue"):
        saver.ensure_safe()
    monkeypatch.setattr(remote, "save", original)
    saver.flush()
    saver.ensure_safe()


def test_pause_and_writer_fencing_stop_new_inference(tmp_path, monkeypatch):
    store = campaign(tmp_path / "run")
    remote = RemoteSession(DirectoryRemote(tmp_path / "remote"), "test-campaign", "one", "owner")
    remote.acquire()
    saver = AutoSaver(store, remote, load_request())
    monkeypatch.setattr(remote, "paused", lambda: True)
    saver.flush()
    with pytest.raises(StopWork, match="paused-awaiting-review"):
        saver.ensure_safe()
    monkeypatch.setattr(remote, "save", lambda _: (_ for _ in ()).throw(WriterConflict("changed")))
    with pytest.raises(WriterConflict):
        saver.flush()
    with pytest.raises(StopWork, match="writer-conflict"):
        saver.ensure_safe()


def test_periodic_background_save_uploads_new_cases_during_work(tmp_path, monkeypatch):
    store = campaign(tmp_path / "run")
    store.freeze_production({"batch": 1})
    remote = RemoteSession(DirectoryRemote(tmp_path / "remote"), "test-campaign", "one", "owner")
    remote.acquire()
    saved = threading.Event()
    original = remote.save

    def save(snapshot):
        revision = original(snapshot)
        if any(name.startswith("shards/") for name in snapshot):
            saved.set()
        return revision

    monkeypatch.setattr(remote, "save", save)
    saver = AutoSaver(store, remote, {**load_request(), "upload_seconds": 0.01})
    saver.start()
    try:
        unit = store.pending()[0]
        store.complete(unit, translate(unit))
        assert saved.wait(3), "no verified background save"
        assert f"shards/{unit.key}.jsonl" in remote.last_manifest
    finally:
        saver.stop()


def test_benchmark_retains_safe_selection_when_larger_batch_fails():
    class Bounded(StubGenerator):
        def batch(self, messages, size):
            if size > 2:
                raise MemoryError("fixture capacity")
            return super().batch(messages, size)

    from kodoom.bulk.fixtures import fixture_units
    from kodoom.bulk.prompts import requests

    selected = benchmark(Bounded(), requests(fixture_units()[0]), load_request(), dev=True)
    assert selected["batch"] == 2
    assert len(selected["benchmark"]) == 4
    assert all(selected["benchmark_replies"])
