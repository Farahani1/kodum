import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kodoom.bulk import launch
from kodoom.config import load_profile
from kodoom.workflow import run_current


def configure(tmp_path, monkeypatch):
    profile = replace(load_profile("dev"), data_dir=tmp_path / "data")
    monkeypatch.setattr(launch, "load_profile", lambda _: profile)
    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    return profile.data_dir / "bulk/test-campaign/result.json"


def test_launcher_budgets_setup_and_passes_explicit_handoff_to_child(tmp_path, monkeypatch):
    path = configure(tmp_path, monkeypatch)
    monkeypatch.setattr(launch.time, "monotonic", lambda: 200)

    def run(words, *, timeout, check):
        assert timeout == 28700 and check is False
        assert words[words.index("--deadline") + 1] == "28900"
        assert "--takeover" in words and words[words.index("--operator") + 1] == "friend"
        assert words[words.index("--request") + 1].endswith("tpu-bulk.toml")
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {"error_type": None, "persisted": True, "status": "drafts-complete-awaiting-review"}
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch.subprocess, "run", run)
    assert launch.run(
        campaign_id="test-campaign",
        repo="org/private",
        operator="friend",
        takeover=True,
        session_started=100,
    )["persisted"]


def test_hard_deadline_cannot_report_an_old_attempt_as_success(tmp_path, monkeypatch):
    path = configure(tmp_path, monkeypatch)
    path.parent.mkdir(parents=True)
    path.write_text('{"persisted": true}', encoding="utf-8")

    def timeout(words, **kwargs):
        raise subprocess.TimeoutExpired(words, kwargs["timeout"])

    monkeypatch.setattr(launch.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="Session deadline"):
        launch.run(campaign_id="test-campaign", repo="org/private")
    assert not path.exists()


def test_active_bulk_requires_explicit_campaign_instead_of_old_gate_dispatch():
    with pytest.raises(ValueError, match="explicit campaign"):
        run_current(provider="generic", profile="dev", smoke=True)
    with pytest.raises(ValueError, match="HF_DATASET_REPO"):
        launch.validate_settings("test-campaign", "", "owner")
