import csv

from kodoom.cli import main
from kodoom.generators import GENERATORS
from kodoom.review import TEMPLATE_COLUMNS, glossary_rows, template_rows, write_review_pack
from kodoom.translate.glossary import parse


def test_glossary_rows_list_common_then_workflow_terms():
    glossary = parse('[common]\n"refund" = "بازپرداخت"\n[customer_service]\n"agent" = "کارشناس"\n')
    assert glossary_rows(glossary) == [
        {"section": "common", "english": "refund", "persian": "بازپرداخت"},
        {"section": "customer_service", "english": "agent", "persian": "کارشناس"},
    ]


def test_every_template_appears_once_with_a_generated_example():
    rows = template_rows(seed=7)
    assert {r["generator"] for r in rows} == set(GENERATORS)
    ids = [(r["generator"], r["template"]) for r in rows]
    assert len(ids) == len(set(ids))
    assert all(r["example_state"] and r["example_answer"] for r in rows)
    assert all("{" not in r["example_state"] for r in rows)
    assert {r["use"] for r in rows} == {"train", "test only"}


def test_review_pack_writes_html_and_csv_with_feedback_columns(tmp_path):
    counts = write_review_pack(tmp_path, seed=7)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "glossary.csv",
        "glossary.html",
        "templates.csv",
        "templates.html",
    ]
    raw = (tmp_path / "templates.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # a BOM, so Excel reads the Persian as UTF-8
    with (tmp_path / "templates.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == TEMPLATE_COLUMNS
    assert len(rows) == counts["templates"]
    page = (tmp_path / "templates.html").read_text(encoding="utf-8")
    assert 'dir="rtl"' in page and "jalali-dates" in page
    glossary = (tmp_path / "glossary.html").read_text(encoding="utf-8")
    assert "عامل" in glossary and "agent_trace_observability" in glossary


def test_review_pack_command_writes_under_data_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["review-pack", "--profile", "dev"]) == 0
    assert (tmp_path / "data" / "review" / "templates.html").exists()
    readme = (tmp_path / "data" / "README.md").read_text(encoding="utf-8")
    assert "review/  — glossary and templates for a native reader" in readme
