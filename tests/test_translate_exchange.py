import json
import shutil

import pytest

import kodoom.cli as cli
from kodoom.schema import read_jsonl, write_jsonl
from kodoom.translate.exchange import (
    ExchangeError,
    PrecomputedTranslator,
    check_filled,
    instructions,
    read_units,
    units_for,
    write_units,
)
from kodoom.translate.glossary import load
from kodoom.translate.pipeline import Item, StubTranslator, translate_case
from tests.test_translate_pipeline import case, write_profile


def test_units_are_distinct_texts_with_context_terms_and_an_empty_fa():
    units = units_for([case("a"), case("b")], load())
    texts = [(u["kind"], u["text"]) for u in units]
    assert len(texts) == len(set(texts))  # the repeated questions and options appear once
    assert [u["id"] for u in units] == [f"u{i:05d}" for i in range(1, len(units) + 1)]
    state = [u for u in units if u["kind"] == "state"]
    assert {u["register"] for u in state} == {"colloquial"}
    assert all(u["fa"] == "" and u["workflow"] == "customer_service" for u in units)
    assert "customer-service" in units[0]["context"]
    refund = [u for u in units if "refund" in u["text"].lower()]
    assert all("refund" in u["terms"] for u in refund)


def test_a_second_case_with_the_same_questions_adds_only_its_state_text():
    one = units_for([case("a")], load())
    two = units_for([case("a"), case("b")], load())
    assert len(two) == len(one)  # case b has the same state and the same questions


def test_files_round_trip_and_bad_files_are_refused(tmp_path):
    units = units_for([case("a")], load())
    write_units(tmp_path / "u.jsonl", units)
    assert read_units(tmp_path / "u.jsonl") == units
    (tmp_path / "bad.jsonl").write_text("{oops\n", encoding="utf-8")
    with pytest.raises(ExchangeError, match="line 1"):
        read_units(tmp_path / "bad.jsonl")
    (tmp_path / "nofa.jsonl").write_text(
        json.dumps({"id": "u1", "workflow": "w", "text": "t"}) + "\n"
    )
    with pytest.raises(ExchangeError, match="'fa' must be text"):
        read_units(tmp_path / "nofa.jsonl")


def test_check_filled_finds_changed_missing_and_empty_units():
    original = [
        {"id": "u1", "workflow": "w", "text": "a", "fa": ""},
        {"id": "u2", "workflow": "w", "text": "b", "fa": ""},
        {"id": "u3", "workflow": "w", "text": "c", "fa": ""},
        {"id": "u4", "workflow": "w", "text": "d", "fa": ""},
    ]
    filled = [
        {"id": "u1", "workflow": "w", "text": "a", "fa": "الف"},
        {"id": "u2", "workflow": "w", "text": "CHANGED", "fa": "ب"},
        {"id": "u3", "workflow": "w", "text": "c", "fa": "  "},
    ]
    assert check_filled(original, filled) == {"changed": ["u2"], "missing": ["u4"], "empty": ["u3"]}


def test_precomputed_translator_looks_up_by_workflow_and_text():
    units = [{"id": "u1", "workflow": "customer_service", "text": "Hi", "fa": "سلام"}]
    translator = PrecomputedTranslator(units, "manual")
    items = [Item("Hi", "formal", "state", "customer_service"), Item("Hi", "f", "state", "invoice")]
    assert translator.translate(items) == ["سلام", ""]
    assert translator.name == "manual"


def test_instructions_carry_the_rules_the_models_get():
    text = instructions(120)
    assert "120 lines" in text and "`fa`" in text and "keep_in_english" in text
    assert "never a person" in text and "backticks" in text and "Low:" in text


def test_translations_from_a_filled_file_go_through_the_same_checks():
    english = case("a")
    stub = StubTranslator()
    units = units_for([english], load())
    for unit in units:
        unit["fa"] = stub.translate(
            [Item(unit["text"], unit["register"], unit["kind"], unit["workflow"])]
        )[0]
    persian, findings = translate_case(english, PrecomputedTranslator(units, "manual"))
    assert findings == [] and persian[0].extra["translator"] == "manual"
    units[0]["fa"] = "سفارش A-68034 با 3 روز late"  # one English word left behind
    _, findings = translate_case(english, PrecomputedTranslator(units, "manual"))
    assert "english" in {f.check for f in findings}


def test_commands_round_trip(tmp_path, capsys):
    profile = str(write_profile(tmp_path))
    data = tmp_path / "data" / "typed-decisions"
    write_jsonl(data / "en" / "train.jsonl", case("a") + case("b"))
    assert cli.main(["export-units", "typed-decisions", "--profile", profile]) == 0
    exchange = data / "exchange"
    assert (exchange / "INSTRUCTIONS.md").exists()
    assert "distinct texts from 2 cases" in capsys.readouterr().out
    # the unfilled file itself: every unit is empty, so the import reports it and builds nothing
    shutil.copy(exchange / "units.jsonl", exchange / "units-filled.jsonl")
    base = ["import-units", "typed-decisions", "--profile", profile, "--name", "manual"]
    assert cli.main(base) == 1
    out = capsys.readouterr().out
    assert "units empty" in out and "2 cases could not be built" in out
    # a "translator" fills the file
    stub = StubTranslator()
    units = read_units(exchange / "units.jsonl")
    for u in units:
        u["fa"] = stub.translate([Item(u["text"], u["register"], u["kind"], u["workflow"])])[0]
    write_units(exchange / "units-filled.jsonl", units)
    failures = data / "fa" / "manual" / "train.failures.jsonl"
    failures.unlink(missing_ok=True)
    assert cli.main(base) == 0
    assert len(list(read_jsonl(data / "fa" / "manual" / "train.jsonl"))) == 6
    assert cli.main(base) == 0  # a second run skips what is done
    assert "already there" in capsys.readouterr().out
