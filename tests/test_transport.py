from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kodoom.generators import dates
from kodoom.schema import RecordError, read_jsonl, write_jsonl
from kodoom.transport import from_row, read_parquet, to_row, write_parquet


def test_transport_preserves_sparse_gold_order_unicode_and_unknown_checks(tmp_path):
    record = dates.generate(1234, 1)[0]
    record = replace(
        record,
        gold={next(k for k, v in record.gold.items() if v == 1): 1.0},
        checks_passed=None,
        meaning_flag=None,
        extra={"nested": {"نام": [1, None, False]}, "empty": []},
    )
    path = tmp_path / "records.parquet"
    write_parquet(path, [record])
    restored = list(read_parquet(path))
    assert restored == [record]
    write_jsonl(tmp_path / "records.jsonl", restored)
    assert list(read_jsonl(tmp_path / "records.jsonl")) == [record]
    assert len(pq.read_table(path).to_pylist()[0]["gold"]) == 1


def test_gold_entries_follow_options_without_inventing_zeroes():
    record = dates.generate(1234, 1)[0]
    record = replace(record, gold=dict(reversed(list(record.gold.items()))))
    assert [g["id"] for g in to_row(record)["gold"]] == [o.id for o in record.options]
    row = to_row(record)
    row["gold"].append(row["gold"][0])
    with pytest.raises(RecordError, match="duplicate gold"):
        from_row(row)


def test_reject_unversioned_parquet(tmp_path):
    path = tmp_path / "other.parquet"
    pq.write_table(pa.table({"id": ["other"]}), path)
    with pytest.raises(RecordError, match="unsupported"):
        list(read_parquet(path))


def test_empty_transport_uses_the_same_schema(tmp_path):
    path = tmp_path / "empty.parquet"
    assert write_parquet(path, []) == 0
    assert list(read_parquet(path)) == []
