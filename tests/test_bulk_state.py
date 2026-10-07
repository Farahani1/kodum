import json
from dataclasses import replace

import pytest

from kodoom.bulk.data import load_request, select_helmo
from kodoom.bulk.fixtures import fixture_units
from kodoom.bulk.prompts import StubGenerator, rebuild, requests
from kodoom.bulk.state import Campaign, Unit, encoded


def campaign(root):
    return Campaign(root, {"request": {"translator": "stub"}, "code": "fixture"}, fixture_units())


def translate(unit):
    return rebuild(unit, StubGenerator().batch(requests(unit), 2), "stub")


def test_atomic_complete_case_resume_and_new_path(tmp_path):
    store = campaign(tmp_path / "first")
    store.freeze_production({"batch": 2})
    unit = store.pending()[0]
    store.complete(unit, translate(unit))
    copied = tmp_path / "next"
    for name, data in store.snapshot().items():
        path = copied / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    restored = campaign(copied)
    assert unit.key not in {u.key for u in restored.pending()}
    assert sum(row["completed"] for row in restored.counts().values()) == 1
    assert len(restored.translated(unit)) == 5


def test_half_case_never_becomes_complete(tmp_path):
    store = campaign(tmp_path / "run")
    store.freeze_production({"batch": 1})
    unit = store.pending()[0]
    with pytest.raises(ValueError, match="incomplete"):
        store.complete(unit, translate(unit)[:1])
    assert len(store.pending()) == 5
    with pytest.raises(ValueError, match="all five"):
        Unit("typed-decisions", unit.records[:1])


def test_orphan_complete_shard_is_reconciled_but_corruption_is_refused(tmp_path):
    store = campaign(tmp_path / "run")
    store.freeze_production({"batch": 1})
    unit = store.pending()[0]
    old = (store.root / "progress.json").read_bytes()
    store.complete(unit, translate(unit))
    (store.root / "progress.json").write_bytes(old)
    restored = campaign(store.root)
    assert unit.key in restored.progress["completed"]
    path = restored.root / restored.progress["completed"][unit.key]["path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        campaign(store.root)


def test_resume_refuses_changed_inputs_settings_gold_and_batch(tmp_path):
    store = campaign(tmp_path / "run")
    store.freeze_production({"batch": 1})
    with pytest.raises(ValueError, match="production"):
        store.freeze_production({"batch": 2})
    with pytest.raises(ValueError, match="identity changed"):
        Campaign(store.root, {"request": {"translator": "other"}}, fixture_units())
    unit = store.pending()[0]
    fa = translate(unit)
    fa[0] = replace(fa[0], gold={"true": 0.0, "false": 1.0})
    with pytest.raises(ValueError, match="gold"):
        store.complete(unit, fa)
    assert store.progress["completed"] == {}


def test_structured_helmo_rejects_changed_ids_extra_keys_and_duplicate_keys():
    unit = fixture_units()[-1]
    reply = StubGenerator().batch(requests(unit), 1)[0]
    value = json.loads(reply)
    value["gold"] = {"true": 1}
    with pytest.raises(ValueError, match="fields"):
        rebuild(unit, [json.dumps(value)], "stub")
    del value["gold"]
    value["options"]["wrong"] = value["options"].pop("true")
    with pytest.raises(ValueError, match="option IDs"):
        rebuild(unit, [json.dumps(value)], "stub")
    with pytest.raises(ValueError, match="duplicate"):
        rebuild(unit, ['{"state":"a","state":"b"}'], "stub")
    translated = rebuild(unit, [reply], "stub")[0]
    assert translated.gold == unit.records[0].gold
    assert translated.human_reviewed is False


def test_selection_is_reproducible_and_spreads_over_types_and_topics():
    original = fixture_units()[-1].records[0]
    records = [
        replace(original, id=f"r-{i}", source_id=f"r-{i}", extra={"topic": f"t-{i % 5}"})
        for i in range(50)
    ]
    first = select_helmo(records, 20, 123)
    assert encoded([r.to_dict() for r in first]) == encoded(
        [r.to_dict() for r in select_helmo(records, 20, 123)]
    )
    assert len({r.extra["topic"] for r in first}) == 5
    assert len({r.id for r in first}) == 20
    assert load_request()["typed_train_cases"] == 1200
