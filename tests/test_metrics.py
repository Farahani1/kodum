import math
import random

import pytest

from kodoom import metrics as M

OPTIONS = ["a", "b", "c"]


def test_item_metrics_hand_computed():
    m = M.item_metrics(OPTIONS, {"a": 0.7, "b": 0.3}, {"a": 0.5, "b": 0.25, "c": 0.25})
    assert m.correct
    assert m.soft_accuracy == pytest.approx(0.425)
    assert m.nll == pytest.approx(0.7 * math.log(2) + 0.3 * math.log(4))
    assert m.brier == pytest.approx(0.105)
    assert m.kl == pytest.approx(m.nll - (-(0.7 * math.log(0.7) + 0.3 * math.log(0.3))))
    assert m.confidence == 0.5
    assert m.score_error is None


def test_a_perfect_prediction():
    m = M.item_metrics(OPTIONS, {"b": 1.0}, {"b": 1.0})
    assert (m.correct, m.soft_accuracy, m.nll, m.brier, m.kl, m.js) == (True, 1.0, 0, 0, 0, 0)


def test_gold_ties_all_count_as_correct():
    assert M.item_metrics(["a", "b"], {"a": 0.5, "b": 0.5}, {"a": 0.2, "b": 0.8}).correct


def test_prediction_ties_go_to_the_first_option():
    # The order of the options decides, never the order of the dict.
    assert not M.item_metrics(["a", "b"], {"b": 1.0}, {"b": 0.5, "a": 0.5}).correct
    assert M.item_metrics(["a", "b"], {"a": 1.0}, {"b": 0.5, "a": 0.5}).correct


def test_wrong_and_confident_is_punished_hard():
    m = M.item_metrics(["a", "b"], {"a": 1.0}, {"a": 0.01, "b": 0.99})
    assert not m.correct and m.nll > 4 and m.brier > 1.9


def test_predictions_are_renormalized_and_missing_options_are_zero():
    m = M.item_metrics(OPTIONS, {"a": 1.0}, {"a": 2.0, "b": 2.0})
    assert m.soft_accuracy == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("probs", "message"),
    [
        ({"a": 1.0, "z": 0.5}, "unknown options"),
        ({"a": -0.1, "b": 1.1}, "negative"),
        ({"a": 0.0, "b": 0.0}, "sum to zero"),
        ({"a": float("nan")}, "negative or not a number"),
    ],
)
def test_bad_predictions_are_rejected(probs, message):
    with pytest.raises(M.MetricError, match=message):
        M.item_metrics(OPTIONS, {"a": 1.0}, probs)


def test_js_is_symmetric_and_bounded():
    p, q = {"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9}
    ab = M.item_metrics(["a", "b"], p, q).js
    ba = M.item_metrics(["a", "b"], q, p).js
    assert ab == pytest.approx(ba) and 0 < ab < math.log(2)


def test_score_error_uses_the_level_order():
    levels = ["l0", "l1", "l2"]
    far = M.item_metrics(levels, {"l2": 1.0}, {"l0": 1.0}, ordered=True)
    near = M.item_metrics(levels, {"l2": 1.0}, {"l1": 1.0}, ordered=True)
    assert far.score_error == pytest.approx(2) and near.score_error == pytest.approx(1)
    half = M.item_metrics(levels, {"l2": 1.0}, {"l0": 0.5, "l2": 0.5}, ordered=True)
    assert half.score_error == pytest.approx(1)


def test_ece_hand_computed():
    # Two bins: 0.9 confident and right half the time, 0.6 confident and always right.
    assert M.ece([0.9, 0.9, 0.6, 0.6], [True, False, True, True]) == pytest.approx(0.4)
    assert M.ece([1.0, 1.0], [True, True]) == 0
    assert math.isnan(M.ece([], []))


def test_ece_of_a_calibrated_source_is_small():
    rng = random.Random(3)
    conf = [rng.uniform(0.5, 1) for _ in range(20000)]
    ok = [rng.random() < c for c in conf]
    assert M.ece(conf, ok) < 0.02


def test_coverage_curve():
    curve = M.coverage_curve([0.9, 0.8, 0.7, 0.6], [True, True, False, True], points=4)
    assert curve == [
        (0.25, 1.0, 0.9),
        (0.5, 1.0, 0.8),
        (0.75, pytest.approx(2 / 3), 0.7),
        (1.0, 0.75, 0.6),
    ]
    assert M.coverage_curve([], []) == []


def test_quantile():
    assert M.quantile([1, 2, 3, 4], 0.5) == 2.5
    assert M.quantile([5], 0.95) == 5
    assert math.isnan(M.quantile([], 0.5))


# -- bootstrap ---------------------------------------------------------------------------------


def test_bootstrap_of_a_constant_is_that_constant():
    assert M.bootstrap_ci([0.7] * 50) == pytest.approx((0.7, 0.7, 0.7))


def test_bootstrap_interval_contains_the_mean_and_is_reproducible():
    values = [random.Random(1).random() for _ in range(1)] + [i % 2 for i in range(100)]
    mean, low, high = M.bootstrap_ci(values, seed=5)
    assert low <= mean <= high
    assert M.bootstrap_ci(values, seed=5) == (mean, low, high)


def test_grouped_bootstrap_is_wider_when_items_within_a_group_agree():
    rng = random.Random(2)
    per_group = [1.0 if rng.random() < 0.5 else 0.0 for _ in range(100)]
    values = [v for v in per_group for _ in range(2)]  # two identical items per group
    groups = [g for g in range(100) for _ in range(2)]
    _, low_iid, high_iid = M.bootstrap_ci(values, seed=1)
    _, low_grp, high_grp = M.bootstrap_ci(values, groups, seed=1)
    assert (high_grp - low_grp) > 1.2 * (high_iid - low_iid)


def test_paired_difference():
    a = [0.9, 0.8, 0.7, 0.6] * 25
    b = [x - 0.1 for x in a]
    mean, low, high = M.paired_difference_ci(a, b)
    assert mean == pytest.approx(0.1) and low == pytest.approx(0.1) and high == pytest.approx(0.1)
    with pytest.raises(M.MetricError, match="same items"):
        M.paired_difference_ci([1.0], [1.0, 2.0])


def test_a_real_difference_excludes_zero_and_noise_includes_it():
    rng = random.Random(4)
    a = [1.0 if rng.random() < 0.8 else 0.0 for _ in range(400)]
    worse = [1.0 if rng.random() < 0.6 else 0.0 for _ in range(400)]
    _, low, _ = M.paired_difference_ci(a, worse)
    assert low > 0
    # Noise: the same answers with ten wins and ten losses swapped, so the true difference is 0.
    same = list(a)
    ones = [i for i, v in enumerate(a) if v == 1.0][:10]
    zeros = [i for i, v in enumerate(a) if v == 0.0][:10]
    for i in ones:
        same[i] = 0.0
    for i in zeros:
        same[i] = 1.0
    diff, low, high = M.paired_difference_ci(a, same)
    assert diff == 0 and low < 0 < high


def test_bootstrap_of_nothing_is_nan():
    assert all(math.isnan(x) for x in M.bootstrap_ci([]))


# -- temperature scaling -----------------------------------------------------------------------


def test_temperature_one_changes_nothing_and_never_changes_the_choice():
    p = {"a": 0.7, "b": 0.2, "c": 0.1}
    assert M.apply_temperature(p, 1.0) == pytest.approx(p)
    for t in (0.3, 2.0, 7.0):
        scaled = M.apply_temperature(p, t)
        assert max(scaled, key=scaled.get) == "a" and sum(scaled.values()) == pytest.approx(1)


def test_high_temperature_flattens_and_low_sharpens():
    p = {"a": 0.7, "b": 0.3}
    assert M.apply_temperature(p, 50)["a"] < 0.52
    assert M.apply_temperature(p, 0.2)["a"] > 0.95
    with pytest.raises(M.MetricError):
        M.apply_temperature(p, 0)


def test_fit_recovers_the_temperature_that_matches_soft_gold():
    # Predictions say 0.9/0.1 but the gold is 0.7/0.3: the fix is T = ln(9) / ln(7/3).
    examples = [({"a": 0.9, "b": 0.1}, {"a": 0.7, "b": 0.3})] * 20
    assert M.fit_temperature(examples) == pytest.approx(math.log(9) / math.log(7 / 3), rel=1e-3)


def test_fit_leaves_a_calibrated_model_alone():
    examples = [({"a": 0.7, "b": 0.3}, {"a": 0.7, "b": 0.3})] * 20
    assert M.fit_temperature(examples) == pytest.approx(1.0, abs=1e-3)


def test_fit_sharpens_an_underconfident_model():
    examples = [({"a": 0.6, "b": 0.4}, {"a": 1.0})] * 20
    assert M.fit_temperature(examples) < 1


def test_fit_on_nothing_is_an_error():
    with pytest.raises(M.MetricError, match="no examples"):
        M.fit_temperature([])
