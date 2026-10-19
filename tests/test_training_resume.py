"""Exercise the actual worker loop without Hub access or real model weights."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest


def test_worker_resumes_optimizer_and_rejects_changed_recipe(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import kodoom.training.worker as worker
    from kodoom.config import load_profile
    from kodoom.schema import Option
    from kodoom.training.settings import Settings
    from tests.test_training_data import rows

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Embedding(16, 8)
            self.scorer = torch.nn.Linear(8, 2)

        def forward(self, input_ids, **_):
            return self.scorer(self.encoder(input_ids).mean(1)), None

    def collate(groups, _pad):
        items = [x for group in groups for x in group]
        n = len(items)
        return {
            "input_ids": torch.ones(n, 3, dtype=torch.long),
            "attention_mask": torch.ones(n, 3),
            "marker_pos": torch.zeros(n, 2).long(),
            "marker_mask": torch.ones(n, 2).bool(),
            "qtype": torch.zeros(n).long(),
            "target": torch.tensor([[0.7, 0.3]] * n),
        }

    common = SimpleNamespace(collate_items=collate)
    tokenizer = SimpleNamespace(pad_token_id=0)
    record_rows = [
        replace(r, options=(Option("a", "A"), Option("b", "B")), gold={"a": 0.7, "b": 0.3})
        for r in rows(8)
    ]
    profile = replace(
        load_profile("dev"),
        runs_dir=tmp_path / "runs",
        scratch_dir=tmp_path / "scratch",
        reserve_gb=0,
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("fixture")
    monkeypatch.setattr(worker, "load_profile", lambda *_a, **_k: profile)
    monkeypatch.setattr(
        worker,
        "load_prepared",
        lambda _: ({"fingerprint": "fixture"}, {"train": record_rows, "validation": record_rows}),
    )
    monkeypatch.setattr(
        worker.laya,
        "load",
        lambda *_a, **_k: (Tiny(), tokenizer, {"max_len": 128, "head_max_len": 80}, common),
    )
    monkeypatch.setattr(
        worker, "encode_rows", lambda rs, *_a: [{"truncation": {"truncated": False}} for _ in rs]
    )
    monkeypatch.setattr(worker, "git_commit", lambda: "fixture")
    monkeypatch.setattr(
        worker.laya,
        "save_artifact",
        lambda path, model, *_a: torch.save(model.state_dict(), path / "weights.pt"),
    )
    s = Settings(english_accuracy_drop=1.0, english_loss_rise=1.0)
    first = worker.train(prepared, s, profile_name="dev", run_id="smoke", smoke=True)
    assert first["step"] == 2 and not first["test_evaluated"]
    second = worker.train(prepared, s, profile_name="dev", run_id="smoke", smoke=True)
    assert second["step"] == 4
    state = torch.load(profile.runs_dir / "smoke/latest/state.pt", weights_only=False)
    assert state["optimizer"]["state"] and state["offset"] == 4
    assert state["scheduler"]["last_epoch"] == 4
    assert len(state["rng"]) == 1
    events = [
        json.loads(line) for line in (profile.runs_dir / "smoke/log.jsonl").read_text().splitlines()
    ]
    assert sum(x["event"] == "resumed" for x in events) == 1
    with pytest.raises(Exception, match="different config"):
        worker.train(prepared, replace(s, seed=9), profile_name="dev", run_id="smoke", smoke=True)


def test_two_rank_replay_keeps_each_rank_bilingual_with_microbatch_one():
    from kodoom.training.data import replay_batches
    from kodoom.training.worker import rank_indices
    from tests.test_training_data import rows

    records = rows(32)
    for ratio in (0.5, 0.75):
        full = replay_batches(records, seed=1, epoch=0, english_fraction=ratio)[0]
        shards = [rank_indices(full, rank=rank, world=2) for rank in (0, 1)]
        assert sorted(shards[0] + shards[1]) == sorted(full)
        for shard in shards:
            assert {records[i].question_lang for i in shard} == {"fa", "en"}
            assert sum(records[i].question_lang == "en" for i in shard) == int(16 * ratio)
