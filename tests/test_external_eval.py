from __future__ import annotations

import ast
import json
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
        {"id": "case-1", "premise": "باران می‌بارد.", "hypothesis": "هوا خشک است.", "label": 2},
        task="entailment", revision="a" * 40, split="test",
        label_names=["entailment", "neutral", "contradiction"],
    )

    assert record.question_type == "choice"
    assert record.state == "باران می‌بارد."
    assert [option.text for option in record.options] == [
        "نتیجه می‌شود",
        "نامشخص",
        "تناقض دارد",
    ]
    assert record.source_revision == "a" * 40
    assert record.gold == {"label-0": 0.0, "label-1": 0.0, "label-2": 1.0}
    validate_external_source(record)


def test_parsinlu_mcq_accepts_letter_gold_and_uses_all_options():
    record = convert_parsinlu_row(
        {"id": "q1", "question": "پایتخت کجاست؟", "choices": ["تهران", "شیراز"], "answer": "A"},
        task="multiple-choice-qa", revision="b" * 40, split="test",
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
            revision="d" * 40, split="train",
        )


def test_parsinlu_integer_labels_require_names_from_dataset_features():
    with pytest.raises(ExternalEvalError, match="verified ClassLabel names"):
        convert_parsinlu_row(
            {"id": "x", "premise": "الف", "hypothesis": "ب", "label": 0},
            task="entailment", revision="e" * 40, split="test",
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
        task="multiple-choice-qa", revision="f" * 40, split="test",
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
    assert "f315b5da129e2b1df55d971ddf2053c3cbfe5282" in source
    assert "pip', 'install', '-q', 'dibaone'" not in source
    assert "from dibaone_x1 import DibaOneX1" in source
    assert "backend='torch', device=device" in source
    assert "revision=diba_revision" in source

