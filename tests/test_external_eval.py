from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kodoom.external_eval import (
    ExternalEvalError,
    _eval_split,
    _flatten_persianqa,
    convert_parsinlu_row,
    convert_persianqa_row,
    predict_diba_x1,
    validate_external_source,
)


def test_parsinlu_entailment_uses_pinned_labels_and_keeps_text():
    record = convert_parsinlu_row(
        {
            "id": "case-1",
            "premise": "باران می\u200cبارد.",
            "hypothesis": "هوا خشک است.",
            "label": 2,
        },
        task="entailment",
        revision="a" * 40,
        split="test",
        label_names=["entailment", "neutral", "contradiction"],
    )

    assert record.question_type == "choice"
    assert record.state == "باران می\u200cبارد."
    assert [option.text for option in record.options] == [
        "نتیجه می\u200cشود",
        "نامشخص",
        "تناقض دارد",
    ]
    assert record.source_revision == "a" * 40
    assert record.gold == {"label-0": 0.0, "label-1": 0.0, "label-2": 1.0}
    validate_external_source(record)


def test_parsinlu_mcq_accepts_letter_gold_and_uses_all_options():
    record = convert_parsinlu_row(
        {"id": "q1", "question": "پایتخت کجاست؟", "choices": ["تهران", "شیراز"], "answer": "A"},
        task="multiple-choice-qa",
        revision="b" * 40,
        split="test",
    )

    assert [option.text for option in record.options] == ["تهران", "شیراز"]
    assert record.gold == {"option-0": 1.0, "option-1": 0.0}
    assert record.extra["protocol"] == "native-multiple-choice"


@pytest.mark.parametrize(
    ("row", "gold", "answerable"),
    [
        (
            {
                "id": "yes",
                "context": "پاسخ اینجاست",
                "question": "چی؟",
                "answers": {"text": ["پاسخ"]},
            },
            "yes",
            True,
        ),
        ({"id": "no", "context": "متن", "question": "چی؟", "is_impossible": True}, "no", False),
    ],
)
def test_persianqa_becomes_declared_two_choice_answerability(row, gold, answerable):
    record = convert_persianqa_row(row, revision="c" * 40, split="test")

    assert record.gold[gold] == 1.0
    assert record.extra["source_answerable"] is answerable
    assert record.extra["protocol"] == "answerability-to-two-choice"
    validate_external_source(record)


def test_external_benchmark_rejects_training_split():
    with pytest.raises(ExternalEvalError, match="held-out split"):
        convert_persianqa_row(
            {"id": "train", "context": "متن", "question": "چی؟", "is_impossible": False},
            revision="d" * 40,
            split="train",
        )


def test_parsinlu_integer_labels_require_names_from_dataset_features():
    with pytest.raises(ExternalEvalError, match="verified ClassLabel names"):
        convert_parsinlu_row(
            {"id": "x", "premise": "الف", "hypothesis": "ب", "label": 0},
            task="entailment",
            revision="e" * 40,
            split="test",
        )


def test_squad_style_persianqa_is_flattened_with_context_and_answerability():
    rows = _flatten_persianqa(
        {
            "data": [
                {
                    "paragraphs": [
                        {
                            "context": "پاراگراف",
                            "qas": [
                                {
                                    "id": "q1",
                                    "question": "چی؟",
                                    "answers": {"text": ["پاسخ"]},
                                }
                            ],
                        }
                    ]
                }
            ]
        }
    )

    assert rows == [
        {
            "id": "q1",
            "question": "چی؟",
            "answers": {"text": ["پاسخ"]},
            "context": "پاراگراف",
        }
    ]


def test_dataset_split_selection_never_falls_back_to_train():
    with pytest.raises(ExternalEvalError, match="refusing to score train data"):
        _eval_split({"train": []})


def test_diba_adapter_records_probability_scores_and_mode():
    class FakeDiba:
        def decide(self, **request):
            assert request["mode"] == "best"
            assert request["options"] == ["A", "B"]
            return {"scores": {"A": 3.0, "B": 1.0}}

    record = convert_parsinlu_row(
        {"id": "q1", "question": "چی؟", "choices": ["A", "B"], "answer": "A"},
        task="multiple-choice-qa",
        revision="f" * 40,
        split="test",
    )
    prediction = predict_diba_x1([record], FakeDiba())[0]

    assert not prediction.error
    assert prediction.probs["option-0"] > prediction.probs["option-1"]
    assert sum(prediction.probs.values()) == pytest.approx(1.0)


def test_kaggle_notebook_code_is_valid_and_has_no_saved_outputs():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "external-evaluation.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code_cells
    for cell in code_cells:
        ast.parse("".join(cell["source"]))
        assert not cell["outputs"]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    source = source.replace('"', "'")
    assert "f315b5da129e2b1df55d971ddf2053c3cbfe5282" in source
    assert "pip', 'install', '-q', 'dibaone'" not in source
    assert "run_comparison" in source
    adapter = (path.parents[1] / "src/kodoom/model_comparison.py").read_text("utf-8")
    assert 'backend="torch", device=device' in adapter
    assert "revision=revision" in adapter


def test_bootstrap_imports_checkout_in_kernel_without_site_packages():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/external-evaluation.ipynb").read_text("utf-8"))
    bootstrap = "".join(notebook["cells"][3]["source"])
    # Model an existing kernel that has not processed the editable install's
    # .pth file. Stub installation and Git only; exercise real package imports.
    script = """
import json, subprocess, sys
from pathlib import Path
settings = json.loads(sys.stdin.read())
CHECKOUT = Path(settings['checkout'])
subprocess.run = lambda *args, **kwargs: None
subprocess.check_output = lambda *args, **kwargs: 'fake-code-revision'
exec(settings['bootstrap'])
from kodoom.package_import import file_sha
assert callable(file_sha)
assert Path(kodoom.__file__).resolve().parent == CHECKOUT / 'src' / 'kodoom'
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        input=json.dumps({"checkout": str(root), "bootstrap": bootstrap}),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_official_parsinlu_files_convert_without_dataset_scripts(monkeypatch):
    from kodoom import external_eval as module

    fixtures = {
        "entailment/test.csv": [
            {"sent1": "premise", "sent2": "hypothesis", "label": "e"},
            {"sent1": "premise", "sent2": "hypothesis", "label": "-"},
        ],
        "qqp/test.jsonl": [{"q1": "first", "q2": "second", "label": "1"}],
        "sentiment-analysis/food_test.jsonl": [
            {"review": "review", "review_id": "1", "question": "taste?", "label": "-3"},
            {"review": "review", "review_id": "1", "question": "value?", "label": "3"},
        ],
        "sentiment-analysis/movie_test.jsonl": [
            {"review": "review", "review_id": "1", "question": "music?", "label": "2"},
        ],
        "multiple-choice/test.jsonl": [
            {"question": "which?", "candidates": ["A", "B", "C", "D"], "answer": "1"},
            {"question": "which?", "candidates": ["A", "B", "C", "D"], "answer": "4"},
        ],
    }
    calls = []

    def read_file(relative, revision):
        assert revision == module.PARSINLU_REVISION
        assert "train" not in relative and "dev" not in relative
        calls.append(relative)
        return fixtures[relative]

    monkeypatch.setattr(module, "_read_parsinlu_file", read_file)
    records = {}
    for task in module.PARSINLU_FILES:
        rows, revision, split = module.load_parsinlu_task(module.PARSINLU_REPOSITORY, task)
        assert revision == module.PARSINLU_REVISION
        assert split == "test"
        for row in rows:
            validate_external_source(row)
            assert row.extra["source_repository"] == "https://github.com/persiannlp/parsinlu"
        records[task] = rows

    assert len(calls) == 5
    entailment = records["entailment"]
    assert len(entailment) == 1
    assert entailment[0].extra["excluded_unlabelled_rows"] == 1
    assert entailment[0].extra["label_name"] == "entailment"
    assert records["paraphrase"][0].extra["label_name"] == "paraphrase"
    sentiment = records["sentiment"]
    assert [r.question_text for r in sentiment] == ["taste?", "value?", "music?"]
    assert len(sentiment[0].options) == 7
    assert sentiment[0].source_id == sentiment[1].source_id
    assert sentiment[0].source_id != sentiment[2].source_id
    assert len({r.id for r in sentiment}) == 3
    mcq = records["multiple-choice-qa"]
    assert mcq[0].gold["option-0"] == 1.0
    assert mcq[1].gold["option-3"] == 1.0


@pytest.mark.parametrize("label", ["unknown", "", "4"])
def test_official_parsinlu_rejects_unknown_labels(monkeypatch, label):
    from kodoom import external_eval as module

    monkeypatch.setattr(module, "_read_parsinlu_file", lambda *args: [{"label": label}])
    with pytest.raises(ExternalEvalError, match="unknown label"):
        module.load_parsinlu_task(module.PARSINLU_REPOSITORY, "entailment")


@pytest.mark.parametrize("answer", ["0", "5", "A", "-1", ""])
def test_official_mcq_rejects_invalid_one_based_answers(monkeypatch, answer):
    from kodoom import external_eval as module

    monkeypatch.setattr(
        module,
        "_read_parsinlu_file",
        lambda *args: [
            {"question": "which?", "candidates": ["A", "B", "C", "D"], "answer": answer}
        ],
    )
    with pytest.raises(ExternalEvalError, match="one-based answer"):
        module.load_parsinlu_task(module.PARSINLU_REPOSITORY, "multiple-choice-qa")


def test_official_parsinlu_requires_immutable_revision():
    from kodoom.external_eval import load_parsinlu_task

    with pytest.raises(ExternalEvalError, match="immutable"):
        load_parsinlu_task("persiannlp/parsinlu", "entailment", revision="master")


@pytest.mark.parametrize(
    "relative, content, expected",
    [
        (
            "entailment/test.csv",
            "sent1,sent2,label\nfirst,second,e\n",
            [{"sent1": "first", "sent2": "second", "label": "e"}],
        ),
        (
            "qqp/test.jsonl",
            '{"q1":"first","q2":"second","label":"1"}\n',
            [{"q1": "first", "q2": "second", "label": "1"}],
        ),
    ],
)
def test_official_reader_uses_pinned_url(monkeypatch, relative, content, expected):
    import io

    from kodoom import external_eval as module

    def open_url(url, timeout):
        assert f"/{module.PARSINLU_REVISION}/data/{relative}" in url
        assert timeout == 60
        return io.BytesIO(content.encode("utf-8-sig"))

    monkeypatch.setattr(module, "urlopen", open_url)
    assert module._read_parsinlu_file(relative, module.PARSINLU_REVISION) == expected


def test_official_mcq_trims_only_trailing_padding(monkeypatch):
    from kodoom import external_eval as module

    rows = [{"question": "which?", "candidates": ["A", "B", "", " "], "answer": "2"}]
    monkeypatch.setattr(module, "_read_parsinlu_file", lambda *args: rows)
    records, _, _ = module.load_parsinlu_task(module.PARSINLU_REPOSITORY, "multiple-choice-qa")
    assert [o.text for o in records[0].options] == ["A", "B"]
    assert records[0].gold["option-1"] == 1.0
    rows[0]["answer"] = "4"
    with pytest.raises(ExternalEvalError, match="points to padding"):
        module.load_parsinlu_task(module.PARSINLU_REPOSITORY, "multiple-choice-qa")
    rows[0].update(candidates=["A", "", "C"], answer="1")
    with pytest.raises(ExternalEvalError, match="malformed"):
        module.load_parsinlu_task(module.PARSINLU_REPOSITORY, "multiple-choice-qa")
