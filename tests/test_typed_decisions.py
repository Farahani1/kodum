"""The typed-decisions loader, on small synthetic rows in the real layout.

No dataset content is stored here: the rows are invented, shaped like the parquet
files read with ``kodoom inspect`` (JSON strings for state, questions and gold).
"""

import json
import sys
import types

import pytest

import kodoom.cli as cli
from kodoom.schema import read_jsonl
from kodoom.sources import check_record
from kodoom.typed_decisions import (
    REVISION,
    WORKFLOWS,
    TypedDecisionsError,
    case_records,
    field_stats,
    load_records,
    parquet_name,
)

STATE = '{"ticket": {"id": 7,  "text": "Refund please"}}'  # odd spacing must survive


def make_row(case_id="cs-1", workflow="customer_service", split="train", **over):
    questions = {
        "route": {
            "type": "choice",
            "instructions": "Where should it go?",
            "criteria": {"billing": "Billing team", "tech": "Tech team", "other": "Anyone"},
        },
        "urgent": {
            "type": "noul",
            "instructions": "Is it urgent?",
            "criteria": {"false": "No", "true": "Yes"},
        },
        "tone": {
            "type": "score",
            "instructions": "How angry?",
            "criteria": ["calm", "annoyed", "angry", "furious"],
        },
    }
    gold = {
        "route": {
            "label": "billing",
            "confidence": 0.8,
            "probabilities": {"billing": 0.8, "tech": 0.1, "other": 0.1},
        },
        "urgent": {
            "label": "false",
            "confidence": 0.6,
            "probabilities": {"false": 0.6, "true": 0.4},
        },
        "tone": {
            "label": "1",
            "confidence": 0.5,
            "score": 1.4,
            "probabilities": {"0": 0.2, "1": 0.5, "2": 0.2, "3": 0.1},
        },
    }
    row = {
        "id": case_id,
        "workflow": workflow,
        "split": split,
        "state": STATE,
        "questions": json.dumps(questions),
        "gold": json.dumps(gold),
        "factors": json.dumps({"angle": "refund"}),
        "label_agreement": json.dumps({"route": 0.9}),
    }
    row.update(over)
    return row


def test_one_record_per_question_sharing_the_case_id():
    records = case_records(make_row())
    assert [r.id for r in records] == ["cs-1:route", "cs-1:urgent", "cs-1:tone"]
    assert {r.source_id for r in records} == {"cs-1"}
    assert [r.question_type for r in records] == ["choice", "noul", "score"]
    assert {r.task_family for r in records} == {"workflow-customer-service"}


def test_state_is_kept_byte_identical():
    assert all(r.state == STATE for r in case_records(make_row()))


def test_option_ids_and_gold_are_kept_as_in_the_source():
    route, urgent, tone = case_records(make_row())
    assert [o.id for o in route.options] == ["billing", "tech", "other"]
    assert route.gold == {"billing": 0.8, "tech": 0.1, "other": 0.1}
    assert [o.id for o in urgent.options] == ["false", "true"]
    assert [o.id for o in tone.options] == ["0", "1", "2", "3"]
    assert tone.options[3].text == "furious"
    assert tone.extra["gold_score"] == 1.4
    assert route.extra["label_agreement"] == 0.9 and urgent.extra["label_agreement"] is None


def test_records_pass_the_source_rules():
    for split in ("train", "test"):
        for r in case_records(make_row(split=split)):
            assert r.split == split and r.origin == "synthetic"
            assert r.license == "Apache-2.0" and r.source_revision == REVISION
            check_record(r)


def test_extra_keeps_provenance_but_not_model_input():
    r = case_records(make_row())[0]
    assert r.extra["case_id"] == "cs-1" and r.extra["workflow"] == "customer_service"
    assert r.extra["factors"] == {"angle": "refund"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"split": "validation"}, "split must be one of"),
        ({"state": "{not json"}, "not valid JSON"),
        ({"questions": "{}"}, "non-empty object"),
        ({"gold": "{}"}, "no gold probabilities"),
        ({"id": ""}, "non-empty text field 'id'"),
    ],
)
def test_bad_rows_are_refused(change, message):
    with pytest.raises(TypedDecisionsError, match=message):
        case_records(make_row(**change))


def test_bad_question_shapes_are_refused():
    def with_question(spec):
        row = make_row()
        questions = json.loads(row["questions"])
        questions["route"] = spec
        return {**row, "questions": json.dumps(questions)}

    with pytest.raises(TypedDecisionsError, match="type must be one of"):
        case_records(with_question({"type": "essay", "instructions": "x"}))
    with pytest.raises(TypedDecisionsError, match="list of levels"):
        case_records(with_question({"type": "score", "instructions": "x", "criteria": {"a": "b"}}))
    with pytest.raises(TypedDecisionsError, match="'false' and 'true'"):
        case_records(
            with_question({"type": "noul", "instructions": "x", "criteria": {"a": "1", "b": "2"}})
        )
    with pytest.raises(TypedDecisionsError, match="map option ids"):
        case_records(with_question({"type": "choice", "instructions": "x", "criteria": ["a", "b"]}))


def _write_parquet(path, rows):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def fake_hub(tmp_path):
    """A stand-in for hf_hub_download backed by parquet files on disk."""
    calls = []
    for workflow in WORKFLOWS:
        for split in ("train", "test"):
            rows = [make_row(f"{workflow[:2]}-{split}-{i}", workflow, split) for i in range(3)]
            _write_parquet(tmp_path / "hub" / parquet_name(workflow, split), rows)

    def download(*, repo_id, filename, repo_type, revision):
        calls.append((repo_id, filename, repo_type, revision))
        return str(tmp_path / "hub" / filename)

    download.calls = calls
    return download


def test_load_records_reads_every_workflow_and_split(fake_hub):
    by_split = load_records(fake_hub)
    assert len(by_split["train"]) == 4 * 3 * 3 and len(by_split["test"]) == 4 * 3 * 3
    assert {r.split for r in by_split["test"]} == {"test"}
    assert {c[0] for c in fake_hub.calls} == {"LocalLLaMA/typed-decisions"}
    assert {c[2:] for c in fake_hub.calls} == {("dataset", REVISION)}


def test_limit_keeps_the_first_cases_per_workflow_and_split(fake_hub):
    by_split = load_records(fake_hub, limit=1)
    assert len({r.source_id for r in by_split["train"]}) == 4
    assert len(by_split["train"]) == 4 * 3


def test_a_row_in_the_wrong_file_is_refused(tmp_path):
    _write_parquet(tmp_path / "f.parquet", [make_row(split="test")])

    def download(**_):
        return str(tmp_path / "f.parquet")

    with pytest.raises(TypedDecisionsError, match="row 0 says workflow"):
        load_records(download, workflows=("customer_service",))


def test_fetch_writes_splits_and_manifest_to_the_data_dir(fake_hub, tmp_path, monkeypatch, capsys):
    # huggingface_hub is a Colab-only extra; a stand-in module keeps this on the dev path.
    stub = types.ModuleType("huggingface_hub")
    stub.hf_hub_download = fake_hub
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)
    out = tmp_path / "out"
    code = cli.main(
        ["fetch", "typed-decisions", "--profile", "dev", "--limit", "2", "--out", str(out)]
    )
    assert code == 0
    train = list(read_jsonl(out / "train.jsonl"))
    assert len(train) == 4 * 2 * 3
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["revision"] == REVISION and manifest["license"] == "Apache-2.0"
    assert manifest["splits"]["train"]["decisions"] == len(train)
    assert "decisions" in capsys.readouterr().out


def test_field_stats_counts_each_case_once_and_ranks_free_text_first():
    rows = []
    for i in range(4):
        state = json.dumps(
            {
                "status": "open",
                "notes": [f"a long free text number {i} " * 3, "short"],
                "customer": {"name": f"Name {i}"},
            }
        )
        rows.append(make_row(f"c-{i}", state=state))
    records = [r for row in rows for r in case_records(row)]
    stats = {r["path"]: r for r in field_stats(records)["customer_service"]}
    assert stats["status"]["distinct"] == 1 and stats["status"]["coverage"] == 1.0
    assert stats["notes[]"]["occurrences"] == 8  # two list items in each of 4 cases
    assert stats["customer.name"]["distinct"] == 4
    ranked = [r["path"] for r in field_stats(records)["customer_service"]]
    assert ranked[0] == "notes[]" and ranked[-1] == "status"


def test_fields_command_prints_the_paths(tmp_path, capsys):
    from kodoom.schema import write_jsonl

    write_jsonl(tmp_path / "r.jsonl", case_records(make_row()))
    assert cli.main(["fields", "--profile", "dev", str(tmp_path / "r.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "1 cases, 3 decisions" in out and "ticket.text" in out
