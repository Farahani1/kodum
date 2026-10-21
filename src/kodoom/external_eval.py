"""Strict adapters for the test-only ParsiNLU and PersianQA evaluations.

These converters create evaluation Records only. They never write source data
back to a dataset repository, and they reject unknown schemas rather than
silently dropping or guessing labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kodoom.schema import Option, Record, one_hot
from kodoom.sources import check_record, get_source

PARSINLU_SOURCE = "persiannlp/parsinlu"
PERSIANQA_SOURCE = "sajjjadayobi/PersianQA"
PARSINLU_LICENSE = "CC-BY-NC-SA-4.0"
PERSIANQA_LICENSE = "GPL-3.0"


class ExternalEvalError(ValueError):
    """An external benchmark row cannot safely be converted."""


def _text(row: Mapping[str, Any], keys: Sequence[str], *, where: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ExternalEvalError(f"{where}: expected a non-empty text field from {tuple(keys)}")


def _stable_id(source: str, revision: str, task: str, raw_id: Any, row: Mapping[str, Any]) -> str:
    raw = str(raw_id) if raw_id is not None else json.dumps(row, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(f"{source}:{revision}:{task}:{raw}".encode("utf-8")).hexdigest()[:20]
    return f"{task}-{digest}"


def _label_name(label: Any, label_names: Sequence[str] | None, *, where: str) -> str:
    if isinstance(label, bool):
        return str(label).lower()
    if isinstance(label, int):
        if label_names is None or label < 0 or label >= len(label_names):
            raise ExternalEvalError(
                f"{where}: integer label {label} has no verified ClassLabel names"
            )
        return str(label_names[label])
    if isinstance(label, str) and label.strip():
        return label.strip()
    raise ExternalEvalError(f"{where}: unsupported label value {label!r}")


def _display_label(name: str) -> str:
    key = re.sub(r"[\s-]+", "_", name.strip().lower())
    return {
        "entailment": "نتیجه می‌شود",
        "entails": "نتیجه می‌شود",
        "contradiction": "تناقض دارد",
        "contradicts": "تناقض دارد",
        "neutral": "نامشخص",
        "paraphrase": "هم‌معنی",
        "paraphrases": "هم‌معنی",
        "not_paraphrase": "ناهم‌معنی",
        "non_paraphrase": "ناهم‌معنی",
        "positive": "مثبت",
        "negative": "منفی",
    }.get(key, name.strip())


def _label_options(names: Sequence[str], *, where: str) -> tuple[Option, ...]:
    cleaned = [str(name).strip() for name in names]
    if len(cleaned) < 2 or any(not value for value in cleaned):
        raise ExternalEvalError(f"{where}: classification task needs at least two named labels")
    if len({v.casefold() for v in cleaned}) != len(cleaned):
        raise ExternalEvalError(f"{where}: label names are not unique: {cleaned!r}")
    options = tuple(Option(f"label-{i}", _display_label(name)) for i, name in enumerate(cleaned))
    if len({option.text.casefold() for option in options}) != len(options):
        raise ExternalEvalError(f"{where}: translated display labels are not unique")
    return options


def _classification_record(
    row: Mapping[str, Any],
    *,
    task: str,
    revision: str,
    row_id: Any,
    split: str,
    label: Any,
    label_names: Sequence[str] | None,
    state: str,
    question: str,
) -> Record:
    where = f"{task}/{row_id}"
    if label_names is None:
        if isinstance(label, str):
            label_names = sorted({label})
        else:
            raise ExternalEvalError(f"{where}: missing verified class names")
    names = list(label_names)
    gold_name = _label_name(label, names if isinstance(label, int) else None, where=where)
    if gold_name not in names:
        raise ExternalEvalError(f"{where}: gold label {gold_name!r} is outside {names!r}")
    options = _label_options(names, where=where)
    option_ids = {name: f"label-{index}" for index, name in enumerate(names)}
    record_id = _stable_id(PARSINLU_SOURCE, revision, f"{task}-{split}", row_id, row)
    source_case_id = str(row_id) if row_id is not None else record_id
    record = Record(
        id=record_id,
        source_id=f"parsinlu:{task}:{split}:{source_case_id}",
        source=PARSINLU_SOURCE,
        source_revision=revision,
        license=PARSINLU_LICENSE,
        split="test",
        origin="native",
        task_family=f"parsinlu-{task}",
        state_lang="fa",
        question_lang="fa",
        state=state,
        question_type="choice",
        question_text=question,
        options=options,
        gold=one_hot(option_ids.values(), option_ids[gold_name]),
        extra={
            "benchmark_split": split,
            "source_row_id": str(row_id),
            "label_name": gold_name,
            "protocol": "label-classification-as-choice",
        },
    )
    check_record(record)
    return record


def convert_parsinlu_row(
    row: Mapping[str, Any],
    *,
    task: str,
    revision: str,
    split: str,
    label_names: Sequence[str] | None = None,
) -> Record:
    """Convert one ParsiNLU classification or multiple-choice QA example.

    ``label_names`` must come from the pinned dataset's ClassLabel feature for
    integer-labeled tasks. For label classification tasks, callers should pass
    those names explicitly so option order is stable and auditable.
    """
    if split not in {"test", "validation", "validation_matched", "validation_mismatched"}:
        raise ExternalEvalError(f"ParsiNLU evaluation requires a held-out split, got {split!r}")
    row_id = row.get("id", row.get("idx", row.get("index")))
    label_key = next(
        (k for k in ("label", "labels", "gold", "answer_idx", "answer") if k in row), None
    )
    if label_key is None:
        raise ExternalEvalError(f"{task}/{row_id}: no label field found")
    label = row[label_key]
    task_key = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")
    where = f"{task_key}/{row_id}"

    choice_values = next(
        (row[k] for k in ("choices", "options", "answer_choices", "candidates") if k in row),
        None,
    )
    if isinstance(choice_values, Mapping):
        choice_values = choice_values.get("text", choice_values.get("options"))
    if isinstance(choice_values, Sequence) and not isinstance(choice_values, str):
        texts = [str(v).strip() for v in choice_values]
        if len(texts) < 2 or any(not value for value in texts):
            raise ExternalEvalError(f"{where}: malformed multiple-choice options")
        options = tuple(Option(f"option-{i}", value) for i, value in enumerate(texts))
        if isinstance(label, int):
            correct_id = f"option-{label}"
        elif isinstance(label, str) and label.strip() in texts:
            correct_id = f"option-{texts.index(label.strip())}"
        elif isinstance(label, str) and re.fullmatch(r"[A-Za-z]", label.strip()):
            correct_id = f"option-{ord(label.strip().upper()) - ord('A')}"
        else:
            raise ExternalEvalError(f"{where}: answer {label!r} does not identify an option")
        ids = {option.id for option in options}
        if correct_id not in ids:
            raise ExternalEvalError(f"{where}: answer index is outside the option list")
        question = _text(row, ("question", "query", "prompt"), where=where)
        state = next(
            (
                row[k]
                for k in ("context", "passage", "paragraph", "premise")
                if isinstance(row.get(k), str)
            ),
            "",
        )
        record_id = _stable_id(PARSINLU_SOURCE, revision, f"{task_key}-{split}", row_id, row)
        source_case_id = str(row_id) if row_id is not None else record_id
        record = Record(
            id=record_id,
            source_id=f"parsinlu:{task_key}:{split}:{source_case_id}",
            source=PARSINLU_SOURCE,
            source_revision=revision, license=PARSINLU_LICENSE, split="test", origin="native",
            task_family=f"parsinlu-{task_key}", state_lang="fa", question_lang="fa",
            state=state.strip() or question, question_type="choice", question_text=question,
            options=options, gold=one_hot(ids, correct_id),
            extra={
                "benchmark_split": split,
                "source_row_id": str(row_id),
                "protocol": "native-multiple-choice",
            },
        )
        check_record(record)
        return record

    if isinstance(label, Sequence) and not isinstance(label, str):
        raise ExternalEvalError(f"{where}: label field is not a scalar class label")
    if task_key in {"entailment", "textual-entailment", "parsinlu-entailment"}:
        premise = _text(row, ("premise", "sentence1", "text1", "text_a"), where=where)
        hypothesis = _text(row, ("hypothesis", "sentence2", "text2", "text_b"), where=where)
        state, question = premise, f"آیا این فرضیه از متن نتیجه می‌شود؟\nفرضیه: {hypothesis}"
    elif task_key in {"paraphrase", "paraphrase-detection", "parsinlu-paraphrase"}:
        first = _text(row, ("sentence1", "text1", "text_a", "question1"), where=where)
        second = _text(row, ("sentence2", "text2", "text_b", "question2"), where=where)
        state, question = first, f"آیا دو عبارت هم‌معنی هستند؟\nعبارت دوم: {second}"
    elif task_key in {"sentiment", "sentiment-analysis", "parsinlu-sentiment"}:
        state = _text(row, ("text", "sentence", "review", "content"), where=where)
        question = "احساس بیان‌شده در متن چیست؟"
    else:
        raise ExternalEvalError(
            f"{where}: unsupported ParsiNLU task; configure an explicit adapter"
        )
    return _classification_record(
        row, task=task_key, revision=revision, row_id=row_id, split=split,
        label=label, label_names=label_names, state=state, question=question,
    )


def _answerable(row: Mapping[str, Any], *, where: str) -> bool:
    for key in ("is_impossible", "unanswerable", "is_unanswerable"):
        if key in row:
            return not bool(row[key])
    for key in ("is_answerable", "answerable", "has_answer"):
        if key in row:
            return bool(row[key])
    answers = row.get("answers")
    if isinstance(answers, Mapping):
        texts = answers.get("text", answers.get("texts"))
        if isinstance(texts, Sequence) and not isinstance(texts, str):
            return any(isinstance(text, str) and text.strip() for text in texts)
        if isinstance(texts, str):
            return bool(texts.strip())
        return bool(answers)
    if isinstance(answers, Sequence) and not isinstance(answers, str):
        return bool(answers)
    if isinstance(answers, str):
        return bool(answers.strip())
    raise ExternalEvalError(f"{where}: cannot determine answerability from the row")


def convert_persianqa_row(row: Mapping[str, Any], *, revision: str, split: str) -> Record:
    """Convert PersianQA span-QA metadata to an explicit yes/no decision.

    Diba X1 receives the same two options as a choice question; the run manifest
    labels this as a task conversion, not native X1 support for ``noul``.
    """
    if split not in {"test", "validation", "validation_matched", "validation_mismatched"}:
        raise ExternalEvalError(f"PersianQA evaluation requires a held-out split, got {split!r}")
    row_id = row.get("id", row.get("idx", row.get("index")))
    where = f"PersianQA/{row_id}"
    question = _text(row, ("question", "query"), where=where)
    passage = _text(row, ("context", "passage", "paragraph", "text"), where=where)
    answerable = _answerable(row, where=where)
    options = (Option("yes", "بله"), Option("no", "خیر"))
    correct = "yes" if answerable else "no"
    record_id = _stable_id(
        PERSIANQA_SOURCE, revision, f"persianqa-answerability-{split}", row_id, row
    )
    source_case_id = str(row_id) if row_id is not None else record_id
    record = Record(
        id=record_id,
        source_id=f"persianqa:{split}:{source_case_id}", source=PERSIANQA_SOURCE,
        source_revision=revision, license=PERSIANQA_LICENSE, split="test", origin="native",
        task_family="persianqa-answerability", state_lang="fa", question_lang="fa",
        state=passage,
        question_type="choice",
        question_text=f"آیا متن به پرسش پاسخ می‌دهد؟\nپرسش: {question}",
        options=options, gold=one_hot((option.id for option in options), correct),
        extra={
            "benchmark_split": split,
            "source_row_id": str(row_id),
            "protocol": "answerability-to-two-choice",
            "source_answerable": answerable,
        },
    )
    check_record(record)
    return record


def validate_external_source(record: Record) -> None:
    """Extra guard used by notebook loaders before any model sees a row."""
    source = get_source(record.source)
    if source.role != "test_only" or source.publish_derived_data:
        raise ExternalEvalError(f"{record.id}: benchmark source policy is not test-only/private")
    if record.split != "test":
        raise ExternalEvalError(f"{record.id}: external benchmark records must be marked test")
    check_record(record)


def predict_laya(records, model, tokenizer, common, cfg, *, device: str, microbatch: int = 8):
    """Run the selected Laya checkpoint and return standard Predictions."""
    import torch

    from kodoom.predictions import Prediction
    from kodoom.training import laya

    if microbatch < 1:
        raise ExternalEvalError("microbatch must be positive")
    records = list(records)
    if any(record.question_type != "choice" for record in records):
        raise ExternalEvalError("the external benchmark adapter currently requires choice records")
    encoded: list[dict | None] = []
    predictions_by_id = {}
    for record in records:
        try:
            encoded.append(
                laya.encode(
                    record,
                    tokenizer,
                    common,
                    max_len=int(cfg["max_len"]),
                    head_max_len=int(cfg["head_max_len"]),
                )
            )
        except Exception as exc:
            encoded.append(None)
            predictions_by_id[record.id] = Prediction(
                record.id,
                {},
                gold=dict(record.gold),
                error=f"{type(exc).__name__}: {exc}",
                extra={"adapter": "kodoom-laya"},
            )
    model.to(device)
    model.eval()
    with torch.inference_mode():
        valid_indexes = [index for index, item in enumerate(encoded) if item is not None]
        for start in range(0, len(valid_indexes), microbatch):
            selected = valid_indexes[start : start + microbatch]
            batch_records = [records[index] for index in selected]
            batch_items = [encoded[index] for index in selected]
            assert all(item is not None for item in batch_items)
            batch = laya.collate(batch_items, tokenizer, common, torch.device(device))
            started = time.perf_counter()
            try:
                logits, _ = model(**{key: batch[key] for key in laya.FORWARD_KEYS})
                probs = logits.float().softmax(-1).cpu()
            except Exception as exc:
                for record in batch_records:
                    predictions_by_id[record.id] = Prediction(
                        record.id,
                        {},
                        gold=dict(record.gold),
                        error=f"{type(exc).__name__}: {exc}",
                        extra={"adapter": "kodoom-laya"},
                    )
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000 / len(batch_records)
            for index, (record, item) in enumerate(
                zip(batch_records, batch_items, strict=True)
            ):
                count = len(item["option_ids"])
                values = probs[index, :count].tolist()
                predictions_by_id[record.id] = Prediction(
                    record.id,
                    dict(zip(item["option_ids"], values, strict=True)),
                    gold=dict(record.gold),
                    latency_ms=elapsed_ms,
                    extra={"adapter": "kodoom-laya", "max_len": int(cfg["max_len"])},
                )
    return [predictions_by_id[record.id] for record in records]


def _decision_probabilities(result: Any, record: Record) -> dict[str, float]:
    """Read a probability/score vector from X1 without inventing a hard label."""
    candidates = ("probabilities", "option_probs", "probs", "scores", "option_scores")
    raw = None
    field = None
    if isinstance(result, Mapping):
        field = next((key for key in candidates if key in result), None)
        raw = result.get(field) if field else None
    else:
        field = next((key for key in candidates if hasattr(result, key)), None)
        raw = getattr(result, field) if field else None
    if raw is None:
        raise ExternalEvalError(
            "DibaOne X1 response has no full probability/score vector; "
            "cannot compute comparable probabilistic metrics"
        )
    option_ids = [option.id for option in record.options]
    option_texts = [option.text for option in record.options]
    if isinstance(raw, Mapping):
        values = []
        for option_id, option_text in zip(option_ids, option_texts, strict=True):
            if option_id in raw:
                values.append(float(raw[option_id]))
            elif option_text in raw:
                values.append(float(raw[option_text]))
            else:
                raise ExternalEvalError(f"DibaOne X1 response omits option {option_id!r}")
    elif isinstance(raw, Sequence) and not isinstance(raw, str):
        values = [float(value) for value in raw]
        if len(values) != len(option_ids):
            raise ExternalEvalError("DibaOne X1 score vector length does not match option count")
    else:
        raise ExternalEvalError("DibaOne X1 returned a malformed probability/score vector")
    if any(not math.isfinite(value) for value in values):
        raise ExternalEvalError("DibaOne X1 returned a non-finite probability/score")
    if (
        field in {"scores", "option_scores"}
        or any(value < 0 for value in values)
        or sum(values) <= 0
    ):
        # Scores are logits or arbitrary rank scores. Convert with stable softmax.
        maximum = max(values)
        exps = [math.exp(value - maximum) for value in values]
        total = sum(exps)
        values = [value / total for value in exps]
    else:
        total = sum(values)
        values = [value / total for value in values]
    return dict(zip(option_ids, values, strict=True))


def predict_diba_x1(records, model, *, mode: str = "best"):
    """Evaluate X1 through its choice API, retaining per-item errors and timing."""
    from kodoom.predictions import Prediction

    predictions = []
    for record in records:
        if record.question_type != "choice":
            predictions.append(
                Prediction(record.id, {}, gold=dict(record.gold), error="unsupported-question-type")
            )
            continue
        started = time.perf_counter()
        try:
            result = model.decide(
                state=record.state,
                question=record.question_text,
                options=[option.text for option in record.options],
                mode=mode,
            )
            probs = _decision_probabilities(result, record)
            latency_ms = (time.perf_counter() - started) * 1000
            predictions.append(
                Prediction(
                    record.id, probs, gold=dict(record.gold), latency_ms=latency_ms,
                    extra={
                        "adapter": "dibaone-x1",
                        "mode": mode,
                        "protocol": record.extra.get("protocol", "native-choice"),
                    },
                )
            )
        except Exception as exc:
            predictions.append(
                Prediction(
                    record.id, {}, gold=dict(record.gold),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                    extra={"adapter": "dibaone-x1", "mode": mode},
                )
            )
    return predictions


def _eval_split(dataset_dict):
    for name in ("test", "validation", "dev"):
        if name in dataset_dict:
            return name, dataset_dict[name]
    raise ExternalEvalError(
        "dataset has no test/validation/dev split; refusing to score train data"
    )


def load_parsinlu_task(repo_id: str, task: str):
    """Load a held-out ParsiNLU split and return (records, resolved revision, split)."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(repo_id).sha
    loaded = load_dataset(repo_id, revision=revision)
    split, rows = _eval_split(loaded)
    label_column = next(
        (
            key
            for key in ("label", "labels", "gold", "answer_idx", "answer")
            if key in rows.column_names
        ),
        None,
    )
    names = None
    if label_column:
        feature = rows.features.get(label_column)
        names = getattr(feature, "names", None)
    converted = [
        convert_parsinlu_row(
            row,
            task=task,
            revision=revision,
            split=split,
            label_names=names,
        )
        for row in rows
    ]
    return converted, revision, split


def _flatten_persianqa(value: Any) -> list[dict[str, Any]]:
    """Flatten common SQuAD JSON structures while retaining answer metadata."""
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_flatten_persianqa(item))
        return rows
    if not isinstance(value, Mapping):
        return []
    if "question" in value and any(
        key in value for key in ("context", "passage", "paragraph", "text")
    ):
        return [dict(value)]
    if isinstance(value.get("data"), list):
        return _flatten_persianqa(value["data"])
    rows = []
    context = value.get("context", value.get("paragraph", value.get("passage")))
    if isinstance(value.get("paragraphs"), list):
        for paragraph in value["paragraphs"]:
            if not isinstance(paragraph, Mapping):
                continue
            para_context = paragraph.get("context", paragraph.get("paragraph", context))
            for qa in paragraph.get("qas", paragraph.get("questions", [])):
                if isinstance(qa, Mapping):
                    rows.append({**qa, "context": para_context})
        return rows
    if isinstance(value.get("qas"), list):
        for qa in value["qas"]:
            if isinstance(qa, Mapping):
                rows.append({**qa, "context": context})
        return rows
    for key in ("examples", "questions", "items", "records"):
        if isinstance(value.get(key), list):
            rows.extend(_flatten_persianqa(value[key]))
    return rows


def load_persianqa_repository(repo_url: str, checkout: str | Path):
    """Clone the read-only PersianQA source and use only files named test/dev/valid."""
    checkout = Path(checkout)
    if checkout.exists():
        raise ExternalEvalError(f"PersianQA checkout path already exists: {checkout}")
    checkout.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(checkout)], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    files = [
        path for path in checkout.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl", ".csv"}
        and re.search(r"(^|[-_.])(test|dev|valid|validation)([-_.]|$)", path.stem.lower())
        and not re.search(r"(^|[-_.])train([-_.]|$)", path.stem.lower())
    ]
    if not files:
        raise ExternalEvalError("PersianQA repository has no explicitly named held-out data file")
    raw_rows: list[tuple[dict[str, Any], str]] = []
    for path in sorted(files):
        split = "test" if "test" in path.stem.lower() else "validation"
        if path.suffix.lower() == ".json":
            rows = _flatten_persianqa(json.loads(path.read_text(encoding="utf-8")))
        elif path.suffix.lower() == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        raw_rows.extend((row, split) for row in rows)
    if not raw_rows:
        raise ExternalEvalError("PersianQA held-out files contain no rows")
    converted = [
        convert_persianqa_row(row, revision=revision, split=split)
        for row, split in raw_rows
    ]
    return converted, revision, sorted({split for _, split in raw_rows})
