import dataclasses
import json

import pytest

from kodoom.schema import (
    Option,
    Record,
    RecordError,
    append_jsonl,
    one_hot,
    read_jsonl,
    write_jsonl,
)


def make(**overrides):
    base = dict(
        id="td-0001-q1-fa",
        source_id="td-0001-q1",
        source="LocalLLaMA/typed-decisions",
        source_revision="abc123",
        license="Apache-2.0",
        split="train",
        origin="translated",
        task_family="workflow",
        state_lang="fa",
        question_lang="fa",
        state="مشتری درخواست بازپرداخت برای فاکتور INV-2291 دارد.",
        question_type="choice",
        question_text="بهترین اقدام بعدی چیست؟",
        options=(Option("refund", "بازپرداخت کامل"), Option("escalate", "ارجاع به مدیر")),
        gold={"refund": 0.7, "escalate": 0.3},
    )
    base.update(overrides)
    return Record(**base)


def test_valid_record_round_trips_through_dict():
    r = make(extra={"has_money": True})
    assert Record.from_dict(r.to_dict()) == r


def test_jsonl_keeps_persian_readable_and_round_trips(tmp_path):
    path = tmp_path / "out" / "records.jsonl"
    records = [make(), make(id="td-0002-q1-fa", source_id="td-0002-q1")]
    assert write_jsonl(path, records) == 2
    text = path.read_text(encoding="utf-8")
    assert "بازپرداخت" in text  # not \u-escaped
    assert list(read_jsonl(path)) == records
    assert not (tmp_path / "out" / "records.jsonl.tmp").exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(split="dev"), "split must be one of"),
        (dict(task_family=""), "task_family"),
        (dict(task_family="Skill Dates"), "task_family"),
        (dict(origin="machine"), "origin must be one of"),
        (dict(question_type="multi"), "question_type must be one of"),
        (dict(state_lang="ar"), "state_lang must be one of"),
        (dict(state="  "), "state must be a non-empty string"),
        (dict(license=""), "license must be a non-empty string"),
        (dict(options=(Option("a", "x"),), gold={"a": 1.0}), "at least 2 options"),
        (dict(options=(Option("a", "x"), Option("a", "y")), gold={"a": 1.0}), "unique"),
        (dict(gold={"refund": 0.7, "escalate": 0.2}), "sum to 1"),
        (dict(gold={"refund": 1.2, "escalate": -0.2}), "must be a probability"),
        (dict(gold={"refund": True, "escalate": 0.0}), "must be a probability"),
        (dict(gold={"refund": 0.5, "other": 0.5}), "do not exist"),
        (dict(gold={}), "gold is empty"),
        (
            dict(
                question_type="noul",
                options=(Option("yes", "بله"), Option("no", "خیر"), Option("maybe", "شاید")),
                gold={"yes": 1.0},
            ),
            "exactly 2 options",
        ),
    ],
)
def test_invalid_records_are_rejected(overrides, message):
    with pytest.raises(RecordError, match=message):
        make(**overrides)


def test_rounded_soft_labels_are_accepted():
    opts = (Option("a", "x"), Option("b", "y"), Option("c", "z"))
    make(options=opts, gold={"a": 0.3333, "b": 0.3333, "c": 0.3334})


def test_gold_may_omit_zero_probability_options():
    make(gold={"refund": 1.0})


def test_from_dict_rejects_unknown_and_missing_fields():
    d = make().to_dict()
    with pytest.raises(RecordError, match="unknown fields"):
        Record.from_dict({**d, "labl": "x"})
    del d["license"]
    with pytest.raises(RecordError):
        Record.from_dict(d)


def test_one_hot():
    assert one_hot(["yes", "no"], "no") == {"yes": 0.0, "no": 1.0}
    with pytest.raises(RecordError):
        one_hot(["yes", "no"], "maybe")


def test_read_errors_name_file_and_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    good = json.dumps(make().to_dict(), ensure_ascii=False)
    bad = json.dumps({**make().to_dict(), "split": "dev"}, ensure_ascii=False)
    path.write_text(good + "\n" + bad + "\n", encoding="utf-8")
    with pytest.raises(RecordError, match=r"bad\.jsonl:2:"):
        list(read_jsonl(path))


def test_append_survives_an_interrupted_write(tmp_path):
    path = tmp_path / "progress.jsonl"
    first, second = make(), make(id="td-0002-q1-fa", source_id="td-0002-q1")
    append_jsonl(path, first)
    # A session dies halfway through writing the next line.
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(second.to_dict(), ensure_ascii=False)[:40])
    assert list(read_jsonl(path)) == [first]  # the partial line is skipped

    append_jsonl(path, second)  # the resumed run cuts it and carries on
    assert list(read_jsonl(path)) == [first, second]


def test_record_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make().split = "test"
