"""The helmo/synthetic-typed-decisions loader, on invented rows in the real layout."""

import json
import sys
import types

import pytest

import kodoom.cli as cli
from kodoom.helmo import REVISION, TypedDecisionsError, load_records, row_record
from kodoom.schema import read_jsonl
from kodoom.sources import check_record


def make_row(qtype="noul", gold=None, criteria=None, **over):
    defaults = {
        "noul": ({}, {"noul": 0.15}),
        "choice": ({"A": "Sue", "B": "Settle", "C": "Wait"}, {"probabilities": {"B": 1.0}}),
        "score": (["none", "mild", "severe"], {"mean": 1.25, "variance": 0.1}),
    }
    default_criteria, default_gold = defaults[qtype]
    question = {
        "type": qtype,
        "instructions": "Is it valid?",
        "criteria": default_criteria if criteria is None else criteria,
    }
    row = {
        "topic": "law",
        "state": "A city passed an ordinance.",
        "questions": json.dumps({"q1": question}),
        "gold": json.dumps({"q1": default_gold if gold is None else gold}),
    }
    row.update(over)
    return row


def test_noul_gets_default_options_and_a_two_sided_gold():
    r = row_record(make_row("noul"), 7)
    assert r.id == r.source_id == "helmo-00007"
    assert [(o.id, o.text) for o in r.options] == [("false", "No"), ("true", "Yes")]
    assert r.gold == pytest.approx({"false": 0.85, "true": 0.15})
    assert r.extra == {"topic": "law", "row": 7, "gold_noul": 0.15}


def test_choice_keeps_option_ids_and_fills_missing_probabilities():
    r = row_record(make_row("choice"), 0)
    assert [o.id for o in r.options] == ["A", "B", "C"]
    assert r.gold == {"A": 0.0, "B": 1.0, "C": 0.0}


def test_score_mean_becomes_a_distribution_with_that_mean():
    r = row_record(make_row("score"), 0)
    assert r.gold == pytest.approx({"0": 0.0, "1": 0.75, "2": 0.25})
    assert sum(int(k) * v for k, v in r.gold.items()) == pytest.approx(1.25)
    assert r.extra["gold_variance"] == 0.1
    top = row_record(make_row("score", gold={"mean": 2.0, "variance": 0.0}), 0)
    assert top.gold == {"0": 0.0, "1": 0.0, "2": 1.0}


def test_records_are_synthetic_train_records_that_pass_the_source_rules():
    for qtype in ("noul", "choice", "score"):
        r = row_record(make_row(qtype), 3)
        assert (r.split, r.origin, r.license) == ("train", "synthetic", "MIT")
        assert r.source_revision == REVISION and r.task_family == "topics"
        check_record(r)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (make_row(state=""), "'state' must be non-empty"),
        (make_row(questions="{}"), "exactly 'q1'"),
        (make_row("noul", gold={"noul": 1.5}), "must be a number in"),
        (make_row("score", gold={"mean": 3.0}), "must be a number in"),
        (make_row("choice", gold={"probabilities": {"Z": 1.0}}), "unknown options"),
        (make_row("choice", gold={"other": 1}), "needs 'probabilities'"),
        (make_row("choice", criteria={"A": "x"}), "map option ids"),
    ],
)
def test_bad_rows_are_refused(row, message):
    with pytest.raises(TypedDecisionsError, match=message):
        row_record(row, 0)


@pytest.fixture
def fake_hub(tmp_path):
    lines = [make_row(t) for t in ("noul", "choice", "score", "noul", "choice", "score")]
    path = tmp_path / "synthetic_train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")
    calls = []

    def download(*, repo_id, filename, repo_type, revision):
        calls.append((repo_id, filename, repo_type, revision))
        return str(path)

    download.calls = calls
    return download


def test_load_records_reads_the_pinned_file_and_honours_limit(fake_hub):
    assert len(load_records(fake_hub)) == 6
    assert len(load_records(fake_hub, limit=4)) == 4
    assert fake_hub.calls[0] == (
        "helmo/synthetic-typed-decisions",
        "synthetic_train.jsonl",
        "dataset",
        REVISION,
    )


def test_a_broken_line_names_the_line(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text("{oops\n", encoding="utf-8")
    with pytest.raises(TypedDecisionsError, match="line 1"):
        load_records(lambda **_: str(path))


def test_fetch_helmo_writes_records_and_manifest(fake_hub, tmp_path, monkeypatch, capsys):
    stub = types.ModuleType("huggingface_hub")
    stub.hf_hub_download = fake_hub
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)
    out = tmp_path / "out"
    assert cli.main(["fetch", "helmo", "--profile", "dev", "--out", str(out)]) == 0
    assert len(list(read_jsonl(out / "train.jsonl"))) == 6
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["revision"] == REVISION and manifest["license"] == "MIT"
    assert manifest["by_type"] == {"noul": 2, "choice": 2, "score": 2}
    assert "6 records" in capsys.readouterr().out
