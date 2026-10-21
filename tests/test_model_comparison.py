import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from kodoom import model_comparison as comparison
from kodoom.external_eval import ExternalEvalError, convert_parsinlu_row
from kodoom.predictions import Prediction, read_predictions
from kodoom.schema import Option


def record(case="a", question="1", *, group="customer", kind="choice", split="test"):
    base = convert_parsinlu_row(
        {"id": case, "question": "پرسش؟", "choices": ["الف", "ب"], "answer": "A"},
        task="multiple-choice-qa",
        revision="a" * 40,
        split="test",
    )
    return replace(
        base,
        id=f"{case}:{question}",
        split=split,
        question_type=kind,
        extra={**base.extra, "workflow": group},
    )


def test_sampling_keeps_whole_cases_every_group_and_is_independent_of_row_order():
    records = [
        record(f"{group}-{index}", question, group=group)
        for group in ("customer", "security")
        for index in range(12)
        for question in ("a", "b", "c")
    ]
    selected, manifest = comparison.select_cases(
        records, cases_per_group=3, seed=17, translated=True
    )
    reversed_selected, reversed_manifest = comparison.select_cases(
        records[::-1], cases_per_group=3, seed=17, translated=True
    )
    assert selected == reversed_selected and manifest == reversed_manifest
    assert len(selected) == 18
    assert {r.extra["workflow"] for r in selected} == {"customer", "security"}
    assert all(
        sum(r.source_id == case for r in selected) == 3 for case in {r.source_id for r in selected}
    )
    other, _ = comparison.select_cases(records, cases_per_group=3, seed=18, translated=True)
    assert {r.id for r in selected} != {r.id for r in other}


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_sample_limit_rejected(count):
    with pytest.raises(ExternalEvalError, match="positive integer"):
        comparison.select_cases([record()], cases_per_group=count, seed=17)


def test_duplicate_ids_do_not_silently_change_sample():
    with pytest.raises(ExternalEvalError, match="duplicate"):
        comparison.select_cases([record(), record()], cases_per_group=1, seed=17)


@pytest.mark.parametrize("kind", ["choice", "noul", "score"])
def test_shared_choice_protocol_preserves_soft_gold_order_split_and_original_type(kind):
    original = record(kind=kind, split="train")
    original = replace(
        original,
        options=(Option("false", "خیر"), Option("true", "بله")),
        gold={"false": 0.2, "true": 0.8},
    )
    converted = comparison.choice_record(original)
    assert converted.question_type == "choice"
    assert converted.options == original.options and converted.gold == original.gold
    assert converted.split == "train" and original.question_type == kind
    assert converted.extra["original_question_type"] == kind
    if kind != "choice":
        assert converted.extra["protocol"] == f"{kind}-to-choice-for-all-models"


def test_translated_input_does_not_relabel_helmo_train_as_test(monkeypatch, tmp_path):
    selections = []

    def consume(root, selection):
        selections.append(selection)
        return [record(split=selection["splits"][0])], {"selection": selection}

    monkeypatch.setattr(comparison, "consume_release", consume)
    benchmarks, locks = comparison.translated_benchmarks(tmp_path, {}, include_helmo=True)
    assert benchmarks["translated-typed-decisions-test"][0].split == "test"
    assert benchmarks["translated-helmo-train-diagnostic"][0].split == "train"
    assert [s["configurations"] for s in selections] == [["typed_decisions"], ["helmo"]]
    assert all(s["languages"] == ["fa"] for s in selections)
    assert set(locks) == set(benchmarks)


def test_three_way_report_pairs_on_common_successes_and_reports_all_failures():
    records = [comparison.choice_record(record(case)) for case in ("a", "b", "c")]
    predictions = {
        "laya-multilingual": {r.id: Prediction(r.id, r.gold) for r in records},
        "kodoom": {r.id: Prediction(r.id, r.gold) for r in records[:2]},
        "dibaone-x1": {r.id: Prediction(r.id, {}, error="failed") for r in records},
    }
    report = comparison.comparison_report(records, predictions, n_boot=10)
    assert report["models"]["kodoom"]["missing"] == 1
    assert report["models"]["dibaone-x1"]["failed"] == 3
    assert report["common_items"]["n"] == 0
    assert len(report["pairwise"]) == 3
    assert report["pairwise"]["laya-multilingual-minus-kodoom"]["n"] == 2
    assert report["pairwise"]["kodoom-minus-dibaone-x1"]["accuracy_difference_ci95"] == [None] * 3
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("fail_diba", [False, True])
def test_runner_saves_three_models_same_requests_and_keeps_model_failure(
    monkeypatch, tmp_path, fail_diba
):
    torch = pytest.importorskip("torch")
    from kodoom.training import laya

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "fake-t4")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    cfg = {"max_len": 1024, "head_max_len": 256}
    monkeypatch.setattr(laya, "load", lambda *args, **kwargs: (object(), None, cfg, None))
    monkeypatch.setattr(
        comparison, "load_kodoom", lambda *args, **kwargs: (object(), None, cfg, None)
    )

    def diba_loader(*args):
        if fail_diba:
            raise RuntimeError("fixture download failure")
        return object()

    monkeypatch.setattr(comparison, "load_diba", diba_loader)
    requests = []

    def predict(records, *args, **kwargs):
        requests.append([(r.id, r.question_type, r.options, r.gold) for r in records])
        return [Prediction(r.id, dict(r.gold)) for r in records]

    monkeypatch.setattr(comparison, "predict_laya", predict)
    monkeypatch.setattr(comparison, "predict_diba_x1", predict)
    benchmarks = {"fixture": [comparison.choice_record(record(kind="score"))]}
    samples = {"fixture": {"interpretation": "held-out"}}
    args = dict(
        work=tmp_path,
        output=tmp_path / "results",
        model_repo="owner/model",
        model_revision="a" * 40,
        model_run_id="full",
        code_revision="b" * 40,
        diba_revision="c" * 40,
    )
    if fail_diba:
        with pytest.raises(ExternalEvalError, match="model failure"):
            comparison.run_comparison(benchmarks, samples, **args)
    else:
        comparison.run_comparison(benchmarks, samples, **args)
    assert len(requests) == (2 if fail_diba else 3)
    assert all(request == requests[0] for request in requests)
    output = args["output"]
    manifest = json.loads((output / "manifest.json").read_text("utf-8"))
    assert manifest["status"] == ("failed" if fail_diba else "completed")
    for name in comparison.MODEL_NAMES:
        path = output / "fixture" / f"{name}.jsonl"
        predictions = read_predictions(path)
        assert set(predictions) == {benchmarks["fixture"][0].id}
        assert manifest["output_sha256"][str(path.relative_to(output))] == comparison.file_sha(path)
    with (output / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["original_question_type"] == "score"
    assert bool(rows[0]["dibaone-x1_error"]) == fail_diba
    assert (output / "summary.csv").exists()


def test_notebook_is_reproducible_and_keeps_computation_in_package():
    import runpy

    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks/external-evaluation.ipynb"
    before = notebook.read_bytes()
    runpy.run_path(str(root / "scripts/build_comparison_notebook.py"), run_name="__main__")
    assert notebook.read_bytes() == before
    source = notebook.read_text("utf-8")
    assert "NATIVE_CASES_PER_GROUP" in source and "TRANSLATED_CASES_PER_GROUP" in source
    assert "run_comparison" in source and "paired_difference_ci" not in source


def test_diba_preserves_real_shipped_response_calibration_and_shortlist():
    from kodoom.external_eval import predict_diba_x1

    class Diba:
        max_length = 256

        def temperature(self, goal, count):
            assert goal is None and count == 2
            return 1.7

        def decide(self, **request):
            return {
                "probs": [0.75, 0.25],
                "considered": [0, 1],
                "tier": "x1",
                "abstain": False,
                "ms": {"total": 5},
            }

    request = comparison.choice_record(record())
    prediction = predict_diba_x1([request], Diba())[0]
    assert not prediction.error and prediction.temperature == 1.7
    assert prediction.extra["considered_option_ids"] == [o.id for o in request.options]
    assert prediction.extra["shortlist_gold_mass"] == 1
    assert prediction.extra["max_length"] == 256


def test_persianqa_can_replay_exact_revision_and_ignores_train_files(monkeypatch, tmp_path):
    from kodoom import external_eval

    checkout = tmp_path / "PersianQA"
    calls = []
    pin = "d" * 40

    def run(args, **kwargs):
        calls.append(args)
        if "clone" in args:
            checkout.mkdir()
            row = {"id": "dev", "context": "متن", "question": "چی؟", "is_impossible": True}
            (checkout / "dev.json").write_text(json.dumps([row]), encoding="utf-8")
            (checkout / "train.json").write_text("invalid training data", encoding="utf-8")

    monkeypatch.setattr(external_eval.subprocess, "run", run)
    monkeypatch.setattr(external_eval.subprocess, "check_output", lambda *a, **k: pin)
    records, revision, splits = external_eval.load_persianqa_repository(
        "https://github.com/sajjjadayobi/PersianQA.git", checkout, revision=pin
    )
    assert len(records) == 1 and revision == pin and splits == ["validation"]
    assert all(r.source_revision == pin and r.split == "test" for r in records)
    assert ["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", pin] in calls
    assert ["git", "-C", str(checkout), "checkout", "--detach", pin] in calls
