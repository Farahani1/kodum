import json
import re

import pytest

from kodoom.schema import read_jsonl, write_jsonl
from kodoom.translate.pipeline import Item, StubTranslator, cases, translate_case, translate_file
from kodoom.typed_decisions import case_records
from tests.test_typed_decisions import make_row

STATE = json.dumps(
    {
        "thread": [
            {"role": "customer", "text": "Hi, order A-68034 is 3 days late."},
            {"role": "agent", "text": "Sorry! I see it on `svc_track_1`."},
        ],
        "orders": [{"id": "A-68034", "date": "2026-01-23", "status": "settled"}],
        "account": {"tier": "premium"},
    }
)


def case(case_id="cs-1", state=STATE):
    return case_records(make_row(case_id, state=state))


def test_stub_keeps_code_ids_numbers_and_punctuation():
    stub = StubTranslator()
    text = "Hi, order A-68034 is 3 days late. See `x_1` long-unused."
    out = stub.translate([Item(text, "f", "state"), Item(text, "f", "state")])
    assert out[0] == out[1]  # deterministic
    assert "A-68034" in out[0] and "`x_1`" in out[0] and " 3 " in out[0]
    assert not re.search(r"[A-Za-z]{3,}", out[0].replace("A-68034", "").replace("x_1", ""))
    assert out[0].endswith(".") and "," in out[0]


def test_a_translated_case_keeps_gold_ids_and_structure():
    english = case()
    persian, findings = translate_case(english, StubTranslator())
    assert findings == []
    assert [r.id for r in persian] == [f"{r.id}:fa" for r in english]
    for en, fa in zip(english, persian, strict=True):
        assert (fa.source_id, fa.split, fa.gold, fa.question_type) == (
            en.source_id,
            en.split,
            en.gold,
            en.question_type,
        )
        assert [o.id for o in fa.options] == [o.id for o in en.options]
        assert (fa.origin, fa.state_lang, fa.question_lang) == ("translated", "fa", "fa")
        assert fa.checks_passed is True and fa.extra["translator"] == "stub"
        assert "check_findings" not in fa.extra
    tree = json.loads(persian[0].state)
    assert tree["orders"] == json.loads(STATE)["orders"]  # kept fields are identical
    assert tree["thread"][0]["role"] == "customer"
    assert "A-68034" in tree["thread"][0]["text"] and "Hi" not in tree["thread"][0]["text"]
    assert persian[0].extra["gold_label"] == english[0].extra["gold_label"]


class Sloppy:
    name = "sloppy"

    def translate(self, items):
        # drops the identifier and one number, and leaves English in a question
        out = StubTranslator().translate(items)
        return [
            t.replace("A-68034", "").replace("3", "") if i.kind == "state" else t
            for t, i in zip(out, items, strict=True)
        ]


def test_problems_are_recorded_on_the_records_not_hidden():
    persian, findings = translate_case(case(), Sloppy())
    assert {f.check for f in findings} == {"identifiers", "numbers"}
    assert all(r.checks_passed is False for r in persian)
    assert {f["check"] for f in persian[0].extra["check_findings"]} == {"identifiers", "numbers"}


def test_a_translator_that_returns_the_wrong_count_is_refused():
    class Short:
        name = "short"

        def translate(self, items):
            return ["x"]

    with pytest.raises(ValueError, match="returned 1 texts"):
        translate_case(case(), Short())


def test_records_of_one_case_must_share_state_and_id():
    mixed = case("a") + case("b")
    with pytest.raises(ValueError, match="share one source_id"):
        translate_case(mixed, StubTranslator())


def test_translate_file_resumes_and_honours_limit(tmp_path):
    src, out = tmp_path / "en.jsonl", tmp_path / "fa" / "train.jsonl"
    write_jsonl(src, [r for cid in ("a", "b", "c") for r in case(cid)])
    stats = translate_file(src, out, StubTranslator(), limit=2)
    assert stats == {"translated": 2, "skipped": 0, "with_findings": 0}
    assert len(list(read_jsonl(out))) == 6
    stats = translate_file(src, out, StubTranslator())
    assert stats == {"translated": 1, "skipped": 2, "with_findings": 0}
    assert [c[0].source_id for c in cases(read_jsonl(out))] == ["a", "b", "c"]


def test_resume_survives_a_line_cut_by_a_dead_session(tmp_path):
    src, out = tmp_path / "en.jsonl", tmp_path / "fa.jsonl"
    write_jsonl(src, [r for cid in ("a", "b") for r in case(cid)])
    translate_file(src, out, StubTranslator(), limit=1)
    with out.open("a", encoding="utf-8") as f:
        f.write('{"id": "b:route:fa", "sour')  # cut mid-write
    # the cut line is dropped; case "b" is translated in full
    stats = translate_file(src, out, StubTranslator())
    assert stats["translated"] == 1
    assert len(list(read_jsonl(out))) == 6


def write_profile(tmp_path):
    def toml_path(name):
        return str(tmp_path / name).replace("\\", "/")

    path = tmp_path / "p.toml"
    path.write_text(
        f"""[run]
seed = 1
runs_dir = "{toml_path("runs")}"
[compute]
device = "cpu"
precision = "fp32"
[data]
max_cases_per_source = 5
[storage]
scratch_dir = "{toml_path("scratch")}"
cache_dir = "{toml_path("cache")}"
data_dir = "{toml_path("data")}"
reserve_gb = 0
""",
        encoding="utf-8",
    )
    return path


def test_translate_command_end_to_end_on_a_dev_style_profile(tmp_path, capsys):
    import kodoom.cli as cli

    profile = write_profile(tmp_path)
    en = tmp_path / "data" / "typed-decisions" / "en"
    write_jsonl(en / "train.jsonl", case("a") + case("b"))
    write_jsonl(en / "test.jsonl", case("c"))
    args = ["translate", "typed-decisions", "--profile", str(profile), "--translator", "stub"]
    assert cli.main(args) == 0
    out = tmp_path / "data" / "typed-decisions" / "fa" / "stub"
    assert len(list(read_jsonl(out / "train.jsonl"))) == 6
    assert len(list(read_jsonl(out / "test.jsonl"))) == 3
    assert cli.main(args) == 0  # a second run skips everything
    assert "already done" in capsys.readouterr().out
    assert cli.main([*args, "--split", "train", "--limit", "1"]) == 0


def test_translate_command_says_what_to_run_first(tmp_path, capsys):
    import kodoom.cli as cli

    args = ["translate", "typed-decisions", "--profile", str(write_profile(tmp_path))]
    assert cli.main([*args, "--translator", "stub"]) != 0
    assert "kodoom fetch typed-decisions" in capsys.readouterr().err
