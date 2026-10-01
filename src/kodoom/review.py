"""Review pack: the glossary and generator templates for a native reader (plan 1.2 step 5).

The glossary (``translate/glossary.toml``) and the Persian templates
(``generators/templates/*.toml``) are drafts that a native speaker must read
before anything is published. TOML is hard to read and harder to comment on,
so ``kodoom review-pack`` writes them to ``<data_dir>/review/`` as:

- ``glossary.html`` and ``templates.html``: right-to-left pages to read in a
  browser; every template is shown next to an item generated from it;
- ``glossary.csv`` and ``templates.csv``: the same rows with empty ``ok``,
  ``suggestion`` and ``note`` columns for the reviewer (UTF-8 with a BOM, so
  Excel shows the Persian).

The files are derived from the repository and can be rewritten at any time;
the filled CSVs are the reviewer's, so save them under another name.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path

from kodoom.generators import GENERATORS
from kodoom.generators.common import load_templates
from kodoom.translate.glossary import Glossary
from kodoom.translate.glossary import load as load_glossary

FEEDBACK = ("ok", "suggestion", "note")
EXAMPLE_PAIRS = 8  # enough items per kind that every template is used at least once

GLOSSARY_COLUMNS = ("section", "english", "persian", *FEEDBACK)
TEMPLATE_COLUMNS = (
    "generator",
    "kind",
    "question_type",
    "template",
    "register",
    "use",
    "state_template",
    "question_template",
    "example_state",
    "example_question",
    "example_options",
    "example_answer",
    *FEEDBACK,
)


def glossary_rows(glossary: Glossary) -> list[dict[str, str]]:
    rows = [
        {"section": "common", "english": en, "persian": fa} for en, fa in glossary.common.items()
    ]
    for workflow, terms in glossary.by_workflow.items():
        rows += [{"section": workflow, "english": en, "persian": fa} for en, fa in terms.items()]
    return rows


def template_rows(seed: int) -> list[dict[str, str]]:
    rows = []
    for name, module in GENERATORS.items():
        examples = {}
        for record in module.generate(seed, EXAMPLE_PAIRS):
            examples.setdefault(record.extra["template"], record)
        for kind, templates in load_templates(name, module.SLOTS).items():
            for t in templates:
                example = examples.get(t.id)
                row = {
                    "generator": name,
                    "kind": kind,
                    "question_type": module.QUESTION_TYPE[kind],
                    "template": t.id,
                    "register": t.register,
                    "use": "test only" if t.held_out else "train",
                    "state_template": t.state,
                    "question_template": t.question,
                    "example_state": "",
                    "example_question": "",
                    "example_options": "",
                    "example_answer": "",
                }
                if example is not None:
                    answer = max(example.gold, key=example.gold.get)
                    row |= {
                        "example_state": example.state,
                        "example_question": example.question_text,
                        "example_options": " | ".join(o.text for o in example.options),
                        "example_answer": next(o.text for o in example.options if o.id == answer),
                    }
                rows.append(row)
    return rows


def write_review_pack(out_dir: Path, seed: int) -> dict[str, int]:
    """Write the four review files; returns the row counts."""
    glossary = load_glossary()
    terms, templates = glossary_rows(glossary), template_rows(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "glossary.csv", GLOSSARY_COLUMNS, terms)
    _write_csv(out_dir / "templates.csv", TEMPLATE_COLUMNS, templates)
    _write_text(out_dir / "glossary.html", _glossary_html(glossary, terms))
    _write_text(out_dir / "templates.html", _templates_html(templates))
    return {"terms": len(terms), "templates": len(templates)}


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


_STYLE = """
body { font-family: Vazirmatn, Tahoma, "Segoe UI", sans-serif; margin: 2rem auto;
       max-width: 72rem; padding: 0 1rem; line-height: 1.7; color: #1d1d1f; background: #fff; }
h1, h2, h3 { line-height: 1.3; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 1.5rem; }
th, td { border: 1px solid #d0d0d5; padding: 0.4rem 0.6rem; vertical-align: top;
         text-align: start; }
th { background: #f2f2f5; }
.fa { direction: rtl; text-align: right; font-size: 1.1rem; }
.meta { color: #555; font-size: 0.9rem; }
.test { background: #fff6e0; }
code { background: #f2f2f5; padding: 0 0.25rem; border-radius: 3px; }
"""


def _page(title: str, body: list[str]) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body>",
            *body,
            "</body></html>",
            "",
        ]
    )


def _fa(text: str) -> str:
    return f'<td class="fa" lang="fa" dir="rtl">{html.escape(text)}</td>'


def _glossary_html(glossary: Glossary, rows: list[dict[str, str]]) -> str:
    body = [
        "<h1>Glossary: English → Persian</h1>",
        '<p class="meta">Draft from <code>src/kodoom/translate/glossary.toml</code>. One Persian '
        "term per English term, so a case and its options name the same thing the same way. "
        "Workflow sections override the common one. Mark each term in <code>glossary.csv</code> "
        "(ok, suggestion, note).</p>",
        "<h2>Terms that stay in English</h2>",
        f"<p>{html.escape(', '.join(glossary.keep))}</p>",
    ]
    section = None
    for row in rows:
        if row["section"] != section:
            if section is not None:
                body.append("</table>")
            section = row["section"]
            body.append(f"<h2>{html.escape(section)}</h2>")
            context = glossary.context(section)
            if context:
                body.append(f'<p class="meta">{html.escape(context)}</p>')
            body.append("<table><tr><th>English</th><th>Persian</th></tr>")
        body.append(f"<tr><td>{html.escape(row['english'])}</td>{_fa(row['persian'])}</tr>")
    body.append("</table>")
    return _page("kodoom glossary review", body)


def _templates_html(rows: list[dict[str, str]]) -> str:
    body = [
        "<h1>Persian skill templates</h1>",
        '<p class="meta">Drafts from <code>src/kodoom/generators/templates/</code>. Each '
        "template is shown with an item generated from it; placeholders such as <code>{a}</code> "
        "and <code>{d}</code> are filled by code with dates, amounts or numbers in Persian, "
        "Arabic-Indic or Latin digits, on purpose. Highlighted rows are held out for the test "
        "split. Read each one aloud: a template that sounds unnatural is a bug. Mark each in "
        "<code>templates.csv</code> (ok, suggestion, note).</p>",
    ]
    group = None
    for row in rows:
        if (row["generator"], row["kind"]) != group:
            if group is not None:
                body.append("</table>")
            if group is None or row["generator"] != group[0]:
                body.append(f"<h2>{html.escape(row['generator'])}</h2>")
            group = (row["generator"], row["kind"])
            body.append(
                f"<h3>{html.escape(row['kind'])} "
                f'<span class="meta">({html.escape(row["question_type"])})</span></h3>'
            )
            body.append(
                "<table><tr><th>Template</th><th>State and question</th>"
                "<th>Generated example</th></tr>"
            )
        cls = ' class="test"' if row["use"] == "test only" else ""
        example = row["example_state"]
        if example:
            options = row["example_options"].replace(" | ", "، ")
            example += (
                f"\n{row['example_question']}\nپاسخ درست: {row['example_answer']}"
                f" (گزینه\u200cها: {options})"
            )
        body.append(
            f"<tr{cls}><td>{html.escape(row['template'])}<br>"
            f'<span class="meta">{html.escape(row["register"])}, '
            f"{html.escape(row['use'])}</span></td>"
            + _fa(f"{row['state_template']}\n{row['question_template']}").replace("\n", "<br>")
            + _fa(example).replace("\n", "<br>")
            + "</tr>"
        )
    body.append("</table>")
    return _page("kodoom template review", body)
