import json

import pytest

import kodoom.cli as cli
from kodoom.schema import write_jsonl
from kodoom.translate.pilot import (
    COLUMNS,
    PilotError,
    build_sheet,
    case_text,
    read_sheet,
    score_sheet,
    write_sheet,
)
from kodoom.translate.pipeline import StubTranslator, translate_case
from tests.test_translate_pipeline import case, write_profile


def persian(english_case, translator=None):
    records, _ = translate_case(english_case, translator or StubTranslator())
    return records


def test_case_text_lists_translated_state_texts_questions_and_options():
    english = case("a")
    text = case_text(english, english[0].state)
    assert "[thread[].text] Hi, order A-68034 is 3 days late." in text
    assert "[route] (choice) Where should it go?" in text and "    billing: Billing team" in text
    assert "orders" not in text  # kept fields are not shown
    fa = case_text(persian(english), english[0].state)
    assert "A-68034" in fa and "Hi," not in fa and fa.count("\n") == text.count("\n")


def test_the_sheet_is_blind_and_reproducible():
    english = [case("a"), case("b"), case("c")]
    one = {c[0].source_id: persian(c, StubTranslator()) for c in english}
    two = {c[0].source_id: persian(c, StubTranslator(use_glossary=False)) for c in english}
    candidates = {"first": one, "second": two}
    rows, key = build_sheet(english, candidates, seed=7)
    again_rows, again_key = build_sheet(english, candidates, seed=7)
    assert key == again_key and rows == again_rows
    assert [r["case_id"] for r in rows] == ["a", "b", "c"]
    for row in rows:
        assert set(row) == set(COLUMNS) and row["better"] == ""
        assert "first" not in json.dumps(row) and "second" not in json.dumps(row)
        assert set(key[row["case_id"]]) == {"A", "B"}
        assert set(key[row["case_id"]].values()) == {"first", "second"}
    other = build_sheet(english, candidates, seed=8)[1]
    assert other != key  # a different seed draws differently (for 3 cases, with seeds 7 and 8)


def test_only_cases_both_translators_have_are_included():
    english = [case("a"), case("b")]
    one = {"a": persian(english[0]), "b": persian(english[1])}
    two = {"a": persian(english[0])}
    rows, _ = build_sheet(english, {"x": one, "y": two})
    assert [r["case_id"] for r in rows] == ["a"]
    with pytest.raises(PilotError, match="no case in common"):
        build_sheet(english, {"x": one, "y": {}})
    with pytest.raises(PilotError, match="exactly two"):
        build_sheet(english, {"x": one})


def test_the_sheet_round_trips_with_persian_and_multiline_cells(tmp_path):
    rows = [
        dict.fromkeys(COLUMNS, "")
        | {"case_id": "a", "english": "line 1\nline 2", "translation_A": "سلام\nدنیا"}
    ]
    write_sheet(tmp_path / "s.csv", rows)
    assert (tmp_path / "s.csv").read_bytes().startswith(b"\xef\xbb\xbf")  # BOM for Excel
    assert read_sheet(tmp_path / "s.csv") == rows
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(PilotError, match="columns missing"):
        read_sheet(tmp_path / "bad.csv")


def test_scoring_unblinds_wins_ties_errors_and_workflows():
    key = {
        "c1": {"A": "first", "B": "second"},
        "c2": {"A": "second", "B": "first"},
        "c3": {"A": "first", "B": "second"},
        "c4": {"A": "first", "B": "second"},
    }

    def row(case_id, better, a="", b="", workflow="customer_service"):
        return dict.fromkeys(COLUMNS, "") | {
            "case_id": case_id,
            "workflow": workflow,
            "better": better,
            "meaning_errors_A": a,
            "meaning_errors_B": b,
        }

    rows = [
        row("c1", "A", "0", "2"),
        row("c2", "a", "1", "0", "invoice_processing"),
        row("c3", "tie", "1", "1"),
        row("c4", ""),
    ]
    result = score_sheet(rows, key)
    assert result["wins"] == {"first": 1, "second": 1}  # c1: first; c2 A is second
    assert (result["ties"], result["unrated"], result["cases"]) == (1, 1, 4)
    # first: c1 A=0, c2 B=0, c3 A=1; second: c1 B=2, c2 A=1, c3 B=1
    assert result["mean_meaning_errors"] == {
        "first": pytest.approx(1 / 3),
        "second": pytest.approx(4 / 3),
    }
    assert result["wins_by_workflow"]["invoice_processing"] == {"first": 0, "second": 1}
    with pytest.raises(PilotError, match="not in the key"):
        score_sheet([row("zzz", "A")], key)
    with pytest.raises(PilotError, match="must be a number"):
        score_sheet([row("c1", "A", "many", "")], key)


def test_commands_end_to_end(tmp_path, capsys, monkeypatch):
    import kodoom.translate.pipeline as pipeline

    profile = str(write_profile(tmp_path))
    data = tmp_path / "data" / "typed-decisions"
    write_jsonl(data / "en" / "train.jsonl", case("a") + case("b"))
    monkeypatch.setattr(pipeline.StubTranslator, "name", "stub", raising=False)
    translate = ["translate", "typed-decisions", "--profile", profile, "--split", "train"]
    assert cli.main([*translate, "--translator", "stub"]) == 0
    second = data / "fa" / "stub2"
    second.mkdir()
    text = (data / "fa" / "stub" / "train.jsonl").read_text(encoding="utf-8")
    (second / "train.jsonl").write_text(text, encoding="utf-8")
    capsys.readouterr()
    sheet = ["pilot-sheet", "typed-decisions", "--profile", profile, "--a", "stub", "--b", "stub2"]
    assert cli.main(sheet) == 0
    assert "2 cases, two translations each" in capsys.readouterr().out
    pilot = data / "pilot"
    rows = read_sheet(pilot / "sheet.csv")
    for r in rows:
        r["better"], r["meaning_errors_A"], r["meaning_errors_B"] = "tie", "1", "0"
    write_sheet(pilot / "sheet-filled.csv", rows)
    score = ["pilot-score", "typed-decisions", "--profile", profile]
    assert cli.main(score) == 0
    out = capsys.readouterr().out
    assert "2 cases: 2 ties, 0 not rated" in out and "mean meaning errors per case" in out
    assert cli.main([*score, "--sheet", "nope.csv"]) != 0
