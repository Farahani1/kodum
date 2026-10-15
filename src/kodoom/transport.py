"""Versioned, lossless Parquet transport for Record JSONL (plan 1.3).

Dynamic gold keys and metadata do not become Arrow struct fields. Sparse
probability maps remain sparse, and option order survives the round trip.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from kodoom.schema import Record, RecordError

VERSION = "1"
TEXT_FIELDS = (
    "id",
    "source_id",
    "source",
    "source_revision",
    "license",
    "split",
    "origin",
    "task_family",
    "state_lang",
    "question_lang",
    "state",
    "question_type",
    "question_text",
)


def arrow_schema():
    import pyarrow as pa

    return pa.schema(
        [pa.field(name, pa.string(), nullable=name == "source_revision") for name in TEXT_FIELDS]
        + [
            pa.field(
                "options",
                pa.list_(
                    pa.field("element", pa.struct([("id", pa.string()), ("text", pa.string())]))
                ),
            ),
            pa.field(
                "gold",
                pa.list_(
                    pa.field(
                        "element", pa.struct([("id", pa.string()), ("probability", pa.float64())])
                    )
                ),
            ),
            pa.field("checks_passed", pa.bool_()),
            pa.field("meaning_flag", pa.bool_()),
            pa.field("human_reviewed", pa.bool_(), nullable=False),
            pa.field("extra_json", pa.string(), nullable=False),
        ],
        metadata={b"kodoom.record.transport": VERSION.encode("ascii")},
    )


def to_row(record: Record) -> dict:
    row = record.to_dict()
    row["gold"] = [
        {"id": option.id, "probability": record.gold[option.id]}
        for option in record.options
        if option.id in record.gold
    ]
    row["extra_json"] = json.dumps(row.pop("extra"), ensure_ascii=False, allow_nan=False)
    return row


def from_row(row: dict) -> Record:
    value = dict(row)
    entries = value["gold"]
    if len({entry["id"] for entry in entries}) != len(entries):
        raise RecordError("duplicate gold ID in Parquet transport")
    value["gold"] = {entry["id"]: entry["probability"] for entry in entries}
    value["extra"] = json.loads(value.pop("extra_json"))
    return Record.from_dict(value)


def write_parquet(path: str | Path, records: Iterable[Record]) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    rows = [to_row(record) for record in records]
    pq.write_table(pa.Table.from_pylist(rows, schema=arrow_schema()), tmp, compression="zstd")
    os.replace(tmp, path)
    return len(rows)


def read_parquet(path: str | Path) -> Iterator[Record]:
    import pyarrow.parquet as pq

    file = pq.ParquetFile(path)
    if not file.schema_arrow.equals(arrow_schema(), check_metadata=True):
        raise RecordError(f"{path}: unsupported Record transport schema")
    for batch in file.iter_batches():
        for row in batch.to_pylist():
            yield from_row(row)
