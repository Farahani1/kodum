from dataclasses import replace

import pytest

from kodoom import helmo, typed_decisions
from kodoom.bulk.data import eligible_helmo, load_request, prepare_units, validate_scope
from kodoom.bulk.fixtures import fixture_units


def test_full_scope_fetches_every_typed_split_before_stratified_sampling(monkeypatch):
    fixture = fixture_units()
    typed = {}
    calls = []
    for split, count in (("train", 1200), ("test", 400)):
        typed[split] = [
            replace(
                record,
                id=f"{split}-{i}:q{j}",
                source_id=f"{split}-{i}",
                split=split,
                source_revision=typed_decisions.REVISION,
                extra={**record.extra, "workflow": typed_decisions.WORKFLOWS[i % 4]},
            )
            for i in range(count)
            for j, record in enumerate(fixture[0].records)
        ]
    records = [
        replace(
            fixture[2].records[0],
            id=f"h{i}",
            source_id=f"h{i}",
            source_revision=helmo.REVISION,
            state=f"Unique state {i}",
            extra={"topic": f"topic-{i % 207}"},
        )
        for i in range(2200)
    ]

    def fetch_typed(download, *, limit):
        calls.append(("typed", limit))
        return typed

    def fetch_helmo(download, *, limit):
        calls.append(("helmo", limit))
        return records

    monkeypatch.setattr(typed_decisions, "load_records", fetch_typed)
    monkeypatch.setattr(helmo, "load_records", fetch_helmo)
    units = prepare_units(load_request(), download=lambda **kwargs: None)
    assert len(units) == 3600 and sum(len(unit.records) for unit in units) == 10000
    assert calls == [("typed", None), ("helmo", None)]
    validate_scope(units, load_request())
    with pytest.raises(ValueError, match="input counts"):
        validate_scope(units[:-1], load_request())


def test_exact_helmo_duplicates_and_held_out_states_are_excluded():
    record = fixture_units()[2].records[0]
    repeated = replace(record, id="repeat", source_id="repeat")
    other = replace(record, id="other", source_id="other", state="A held-out state.")
    test = replace(record, state="  A HELD-OUT state. ")
    assert eligible_helmo([record, repeated, other], [test]) == [record]
