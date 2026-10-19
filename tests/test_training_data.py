from dataclasses import replace

import pytest

from kodoom.package_import import row_records
from kodoom.schema import RecordError
from kodoom.training.data import case_key, grouped_cases, model_record, replay_batches
from tests.test_package_import import bilingual_row


def rows(n=8):
    result = []
    en, fa = row_records(bilingual_row(), preserve_text=True)
    for i in range(n):
        for r in (en, fa):
            result.append(
                replace(
                    r, id=f"{r.id}-{i}", source_id=f"case-{i}", state=r.state + f" unique case {i}"
                )
            )
    return result


def test_replay_each_update_is_balanced_and_all_rows_are_seen():
    records = rows(19)
    batches = replay_batches(records, seed=42, epoch=0, global_batch=8)
    assert all(sum(records[i].question_lang == "en" for i in b) == 4 for b in batches)
    assert set(i for b in batches for i in b) == set(range(len(records)))
    assert batches == replay_batches(records, seed=42, epoch=0, global_batch=8)
    assert batches != replay_batches(records, seed=42, epoch=1, global_batch=8)
    retry = replay_batches(records, seed=42, epoch=0, global_batch=8, english_fraction=0.75)
    assert all(sum(records[i].question_lang == "en" for i in b) == 6 for b in retry)


def test_duplicate_cases_are_grouped_before_splits_and_test_leakage_is_refused():
    records = rows(2)
    records[2] = replace(records[0], id="duplicate", source_id="different-id")
    groups, _ = grouped_cases(records)
    assert any(
        {case_key(records[0]), case_key(records[2])} <= {case_key(r) for r in g}
        for g in groups.values()
    )
    records[2] = replace(records[2], split="test")
    with pytest.raises(RecordError, match="cross released splits"):
        grouped_cases(records)


def test_normalization_preserves_gold_ids_and_raw_record():
    r = rows(1)[0]
    raw = replace(r, state="۱۲۳ كالا", question_text="مي شود؟")
    result = model_record(raw)
    assert "123" in result.state and "ک" in result.state
    assert raw.state == "۱۲۳ كالا"
    assert result.gold == raw.gold
    assert [o.id for o in result.options] == [o.id for o in raw.options]
