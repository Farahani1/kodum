import json
import math

import pytest

from kodoom import baselines
from kodoom.calibration import (
    Calibration,
    CalibrationError,
    fit_calibration,
    read_calibration,
    write_calibration,
)
from kodoom.cli import main
from kodoom.evaluation import evaluate, format_table
from kodoom.metrics import MetricError
from kodoom.predictions import (
    Prediction,
    PredictionError,
    read_predictions,
    write_predictions,
)
from kodoom.schema import Option, Record


def record(
    id_, gold, *, qtype="choice", family="workflow", split="test", extra=None, source_id=None
):
    options = tuple(Option(k, k.upper()) for k in ("a", "b", "c")[: 2 if qtype == "noul" else 3])
    return Record(
        id=id_,
        source_id=source_id or id_,
        source="kodoom/code-labeled",
        source_revision=None,
        license="Apache-2.0",
        split=split,
        origin="synthetic",
        task_family=family,
        state_lang="fa",
        question_lang="fa",
        state="متن",
        question_type=qtype,
        question_text="پرسش؟",
        options=options,
        gold=gold,
        extra=extra or {},
    )


# -- predictions ---------------------------------------------------------------------------


def test_predictions_round_trip(tmp_path):
    path = tmp_path / "p.jsonl"
    original = [
        Prediction("x", {"a": 0.7, "b": 0.3}, {"a": 1.0}, 1.5, 12.5),
        Prediction("y", {}, None, error="timeout"),
    ]
    assert write_predictions(path, original) == 2
    assert list(read_predictions(path).values()) == original


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"id": "x", "probs": {"a": 1}, "typo": 1}', "unknown fields"),
        ('{"probs": {"a": 1}}', "needs an id"),
        ('{"id": "x", "probs": []}', "probs must be an object"),
        ("{not json", r"p\.jsonl:1"),
    ],
)
def test_bad_prediction_lines_are_rejected(tmp_path, line, message):
    path = tmp_path / "p.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(PredictionError, match=message):
        read_predictions(path)


def test_duplicate_prediction_ids_are_rejected(tmp_path):
    path = tmp_path / "p.jsonl"
    line = json.dumps({"id": "x", "probs": {"a": 1.0}})
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(PredictionError, match="duplicate id"):
        read_predictions(path)


# -- evaluation, hand-computed ----------------------------------------------------------------


def test_evaluate_scores_join_and_reports_what_is_missing():
    records = [
        record("r1", {"a": 1.0}),
        record("r2", {"b": 1.0}),
        record("r3", {"c": 1.0}),
        record("r4", {"a": 1.0}),
    ]
    predictions = {
        "r1": Prediction("r1", {"a": 0.8, "b": 0.1, "c": 0.1}),
        "r2": Prediction("r2", {"a": 0.6, "b": 0.3, "c": 0.1}),  # wrong
        "r3": Prediction("r3", {}, error="timeout"),
        "zz": Prediction("zz", {"a": 1.0}),
    }
    result = evaluate(records, predictions)
    assert (result.missing, result.failed, result.unexpected) == (["r4"], ["r3"], ["zz"])
    s = result.summary()
    assert s.n == 2 and s.accuracy == 0.5
    assert s.nll == pytest.approx((-math.log(0.8) - math.log(0.3)) / 2)
    assert s.soft_accuracy == pytest.approx((0.8 + 0.3) / 2)
    assert s.confidence == pytest.approx((0.8 + 0.6) / 2)


def test_breakdowns_by_field_and_by_extra():
    records = [
        record("r1", {"a": 1.0}, qtype="noul", family="skill-dates", extra={"kind": "before"}),
        record("r2", {"a": 1.0}, qtype="noul", family="skill-dates", extra={"kind": "valid"}),
        record("r3", {"a": 1.0}, family="intent"),
    ]
    predictions = {
        "r1": Prediction("r1", {"a": 0.9, "b": 0.1}),
        "r2": Prediction("r2", {"a": 0.2, "b": 0.8}),
        "r3": Prediction("r3", {"a": 0.5, "b": 0.3, "c": 0.2}),
    }
    result = evaluate(records, predictions, seen_families={"intent"})
    assert {k: v.n for k, v in result.by("question_type").items()} == {"choice": 1, "noul": 2}
    assert result.by("extra.kind")["before"].accuracy == 1.0
    assert result.by("extra.kind")["(none)"].n == 1
    assert {k: v.n for k, v in result.by("family_status").items()} == {"seen": 1, "unseen": 2}
    assert result.by("task_family")["skill-dates"].accuracy == 0.5


def test_score_questions_get_mae_and_within_one():
    r = record("s1", {"c": 1.0}, qtype="score")
    result = evaluate([r], {"s1": Prediction("s1", {"a": 1.0})})
    s = result.summary()
    assert s.score_n == 1 and s.score_mae == pytest.approx(2) and s.score_within_one == 0
    assert math.isnan(
        evaluate([record("x", {"a": 1.0})], {"x": Prediction("x", {"a": 1.0})}).summary().score_mae
    )


def test_an_empty_evaluation_is_all_nan_not_an_error():
    s = evaluate([record("r1", {"a": 1.0})], {}).summary()
    assert s.n == 0 and math.isnan(s.accuracy) and s.to_dict()["accuracy"] is None


def test_calibration_lowers_nll_and_brier_against_soft_gold():
    records = [record(f"r{i}", {"a": 0.7, "b": 0.3}, qtype="noul") for i in range(20)]
    predictions = {r.id: Prediction(r.id, {"a": 0.95, "b": 0.05}) for r in records}
    raw = evaluate(records, predictions).summary()
    fitted = fit_calibration(
        [("noul", p.probs, r.gold) for r, p in zip(records, predictions.values(), strict=True)]
    )
    calibrated = evaluate(records, predictions, calibration=fitted).summary()
    assert calibrated.nll < raw.nll and calibrated.brier < raw.brier and calibrated.kl < raw.kl
    assert calibrated.accuracy == raw.accuracy  # the choice never changes
    # ECE compares confidence with hard correctness, which is always right here, so it
    # gets worse as the confidence falls to the soft gold's 0.7: read KL and Brier with it.
    assert calibrated.ece_15 > raw.ece_15


def test_calibration_lowers_ece_when_the_gold_is_hard():
    records = [
        record(f"r{i}", {"a": 1.0 if i < 14 else 0.0, "b": 0.0 if i < 14 else 1.0}, qtype="noul")
        for i in range(20)
    ]
    predictions = {r.id: Prediction(r.id, {"a": 0.95, "b": 0.05}) for r in records}  # right 70%
    raw = evaluate(records, predictions).summary()
    fitted = fit_calibration(
        [("noul", p.probs, r.gold) for r, p in zip(records, predictions.values(), strict=True)]
    )
    calibrated = evaluate(records, predictions, calibration=fitted).summary()
    assert raw.ece_15 == pytest.approx(0.25) and calibrated.ece_15 < 0.02
    assert calibrated.nll < raw.nll and calibrated.accuracy == raw.accuracy


def test_minimal_pair_report_catches_a_model_that_ignores_the_fact():
    records = []
    for i in range(10):  # each pair: one yes, one no
        for role, answer in (("a", "a"), ("b", "b")):
            records.append(
                record(
                    f"p{i}-{role}",
                    {answer: 1.0},
                    qtype="noul",
                    source_id=f"p{i}",
                    extra={"pair_id": f"p{i}"},
                )
            )
    always_a = {r.id: Prediction(r.id, {"a": 0.9, "b": 0.1}) for r in records}
    report = evaluate(records, always_a).pairs()
    assert report.pairs == 10 and report.pair_accuracy == 0 and report.item_accuracy == 0.5
    assert report.one_right == 10
    assert evaluate([record("x", {"a": 1.0})], {"x": Prediction("x", {"a": 1.0})}).pairs() is None


def test_coverage_curve_of_a_useful_confidence():
    records = [record(f"r{i}", {"a": 1.0}, qtype="noul") for i in range(4)]
    predictions = {
        "r0": Prediction("r0", {"a": 0.95, "b": 0.05}),  # right and sure
        "r1": Prediction("r1", {"a": 0.9, "b": 0.1}),
        "r2": Prediction("r2", {"a": 0.4, "b": 0.6}),  # wrong and unsure
        "r3": Prediction("r3", {"a": 0.45, "b": 0.55}),
    }
    curve = evaluate(records, predictions).coverage(points=4)
    assert curve[0][:2] == (0.25, 1.0) and curve[-1][:2] == (1.0, 0.5)


def test_format_table_has_a_row_per_group():
    records = [record("r1", {"a": 1.0}, qtype="noul"), record("r2", {"a": 1.0})]
    predictions = {
        "r1": Prediction("r1", {"a": 0.9, "b": 0.1}),
        "r2": Prediction("r2", {"a": 0.9, "b": 0.05, "c": 0.05}),
    }
    text = format_table(evaluate(records, predictions).by("question_type"), "question_type")
    assert text.splitlines()[0].startswith("question_type") and len(text.splitlines()) == 3


def test_a_prediction_naming_an_option_the_question_lacks_is_an_error_not_a_score():
    records = [record("r1", {"a": 1.0}, qtype="noul")]
    with pytest.raises(MetricError, match="unknown options"):
        evaluate(records, {"r1": Prediction("r1", {"a": 0.9, "b": 0.05, "c": 0.05})})


# -- calibration artifact ---------------------------------------------------------------------


def test_a_type_needs_enough_examples_for_its_own_temperature():
    over = {"a": 0.95, "b": 0.05}
    gold = {"a": 0.7, "b": 0.3}
    examples = [("noul", over, gold)] * 100 + [("choice", over, gold)] * 99
    c = fit_calibration(examples)
    assert set(c.temperatures) == {"noul"} and c.fitted_on == {
        "all": 199,
        "choice": 99,
        "noul": 100,
    }
    assert c.for_type("noul") == c.temperatures["noul"] and c.for_type("choice") == c.temperature
    assert c.temperature > 1


def test_calibration_file_round_trip_and_validation(tmp_path):
    path = tmp_path / "c" / "calibration.json"
    original = Calibration(1.4, {"noul": 1.2}, {"all": 300})
    write_calibration(path, original)
    assert read_calibration(path) == original
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2
    for bad in (
        '{"temperatures": {}}',
        '{"temperature": 0}',
        '{"temperature": 1, "temperatures": {"x": -1}}',
    ):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(CalibrationError):
            read_calibration(path)
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(CalibrationError, match="invalid JSON"):
        read_calibration(path)


def test_fitting_on_nothing_is_an_error():
    with pytest.raises(MetricError):
        fit_calibration([])


# -- baselines ---------------------------------------------------------------------------------


def test_oracle_and_uniform_baselines():
    records = [record("r1", {"a": 1.0}, qtype="noul"), record("r2", {"b": 1.0}, qtype="noul")]
    oracle = evaluate(records, {p.id: p for p in baselines.oracle(records)}).summary()
    assert (oracle.accuracy, oracle.nll, oracle.brier, oracle.kl) == (1.0, 0, 0, 0)
    uniform = evaluate(records, {p.id: p for p in baselines.uniform(records)}).summary()
    assert uniform.accuracy == 0.5 and uniform.nll == pytest.approx(math.log(2))
    assert uniform.confidence == 0.5


def test_prior_follows_the_training_labels_and_ignores_the_input():
    training = [record(f"t{i}", {"a": 1.0}, qtype="noul", split="train") for i in range(9)]
    training.append(record("t9", {"b": 1.0}, qtype="noul", split="train"))
    target = [record("x", {"b": 1.0}, qtype="noul")]
    p = baselines.prior(target, training)[0].probs
    assert p["a"] == pytest.approx(0.9, abs=0.01) and p["b"] == pytest.approx(0.1, abs=0.01)
    assert sum(p.values()) == pytest.approx(1)
    unseen = baselines.prior([record("y", {"a": 1.0}, family="other")], [])[0].probs
    assert all(v == pytest.approx(1 / 3) for v in unseen.values())  # no training: uniform


# -- the commands, end to end on the generated skills -------------------------------------------


@pytest.fixture
def skills(tmp_path):
    assert (
        main(["generate", "--profile", "colab", "--pairs-per-kind", "30", "--out", str(tmp_path)])
        == 0
    )
    return tmp_path / "jalali-dates.jsonl"


def test_score_the_oracle_and_the_uniform_baseline(skills, tmp_path, capsys):
    preds = tmp_path / "oracle.jsonl"
    assert main(["baseline", "oracle", "--data", str(skills), "--out", str(preds)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "score",
                "--gold",
                str(skills),
                "--pred",
                str(preds),
                "--json",
                str(tmp_path / "r.json"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "minimal pairs: 120 pairs, both right 1.000" in out and "missing 0, failed 0" in out
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["summary"]["accuracy"] == 1.0 and report["summary"]["nll"] == 0

    uniform = tmp_path / "uniform.jsonl"
    assert main(["baseline", "uniform", "--data", str(skills), "--out", str(uniform)]) == 0
    capsys.readouterr()
    assert main(["score", "--gold", str(skills), "--pred", str(uniform), "--by", "extra.kind"]) == 0
    out = capsys.readouterr().out
    assert "weekday" in out and "before" in out


def test_score_exits_nonzero_when_predictions_are_missing(skills, tmp_path, capsys):
    preds = tmp_path / "partial.jsonl"
    main(["baseline", "oracle", "--data", str(skills), "--out", str(preds)])
    lines = preds.read_text(encoding="utf-8").splitlines()
    preds.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")
    capsys.readouterr()
    assert main(["score", "--gold", str(skills), "--pred", str(preds)]) == 1
    assert "missing 3" in capsys.readouterr().out


def test_seen_from_reports_seen_and_unseen_families(skills, tmp_path, capsys):
    preds = tmp_path / "u.jsonl"
    main(["baseline", "uniform", "--data", str(tmp_path / "toman-rial.jsonl"), "--out", str(preds)])
    capsys.readouterr()
    code = main(
        [
            "score",
            "--gold",
            str(tmp_path / "toman-rial.jsonl"),
            "--pred",
            str(preds),
            "--seen-from",
            str(skills),
        ]
    )
    assert (
        code == 0 and "unseen" in capsys.readouterr().out
    )  # skill-currency is not in the dates file


def test_calibrate_refuses_the_test_split_and_writes_an_artifact(skills, tmp_path, capsys):
    preds = tmp_path / "u.jsonl"
    main(["baseline", "uniform", "--data", str(skills), "--out", str(preds)])
    out = tmp_path / "calibration.json"
    capsys.readouterr()
    assert (
        main(
            [
                "calibrate",
                "--gold",
                str(skills),
                "--pred",
                str(preds),
                "--split",
                "test",
                "--out",
                str(out),
            ]
        )
        == 1
    )
    assert "never fit a calibration on the test split" in capsys.readouterr().err
    assert main(["calibrate", "--gold", str(skills), "--pred", str(preds), "--out", str(out)]) == 0
    assert read_calibration(out).temperature > 0


def test_prior_needs_training_records(skills, tmp_path, capsys):
    assert (
        main(["baseline", "prior", "--data", str(skills), "--out", str(tmp_path / "p.jsonl")]) == 1
    )
    assert "needs --train" in capsys.readouterr().err
    assert (
        main(
            [
                "baseline",
                "prior",
                "--data",
                str(skills),
                "--train",
                str(skills),
                "--split",
                "test",
                "--out",
                str(tmp_path / "p.jsonl"),
            ]
        )
        == 0
    )
