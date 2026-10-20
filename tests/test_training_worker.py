from dataclasses import replace

import pytest

from kodoom.training.settings import Settings
from kodoom.training.worker import retention_ok


def test_settings_reject_invalid_recipe_and_keep_global_batch():
    s = Settings()
    assert s.global_batch == 32
    assert replace(s, microbatch=1, accumulation=16).global_batch == 32
    for changes in (
        {"epochs": 5},
        {"microbatch": 0},
        {"encoder_lr": float("nan")},
        {"english_fraction": 1.0},
        {"world_size": 3},
        {"max_len": 128},
    ):
        with pytest.raises(ValueError):
            replace(s, **changes)


def test_retention_guard_requires_both_accuracy_and_raw_loss():
    original = {"en": {"accuracy": 0.6, "loss": 1.0}}
    assert retention_ok({"en": {"accuracy": 0.59, "loss": 1.04}}, original, Settings())
    assert not retention_ok({"en": {"accuracy": 0.57, "loss": 0.9}}, original, Settings())
    assert not retention_ok({"en": {"accuracy": 0.7, "loss": 1.06}}, original, Settings())


def test_soft_ce_preserves_soft_targets_and_masks_padding():
    torch = pytest.importorskip("torch")
    from kodoom.training.worker import soft_ce

    logits = torch.tensor([[1.0, 0.0, -10000.0]], requires_grad=True)
    y = torch.tensor([[0.7, 0.3, 0.0]])
    loss = soft_ce(logits, y).mean()
    loss.backward()
    assert torch.allclose(logits.grad, logits.detach().softmax(-1) - y, atol=1e-6)
    assert torch.isfinite(loss)


def test_noul_semantic_order_and_score_order_are_explicit():
    from kodoom.schema import Option, RecordError
    from kodoom.training.laya import question
    from tests.test_training_data import rows

    r = replace(
        rows(1)[0],
        question_type="noul",
        options=(Option("true", "Yes"), Option("false", "No")),
        gold={"true": 0.8, "false": 0.2},
    )
    q, ids = question(r)
    assert ids == ["false", "true"] and q["crit"]["true"] == "Yes"
    with pytest.raises(RecordError, match="zero-based"):
        question(replace(r, question_type="score"))


def test_amp_overflow_replays_batch_before_one_finite_optimizer_update():
    torch = pytest.importorskip("torch")
    from kodoom.training.worker import optimizer_update

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    draws, overflows = [], []

    def backward():
        draws.append(float(torch.rand(())))
        # Finite forward and true gradient, but the initial scaled fp16
        # backward signal (1024 * 100) overflows. Half that scale is finite.
        loss = model.weight.to(torch.float16).float().sum() * 100
        scaler.scale(loss).backward()
        return "same-batch"

    result, norm, retries = optimizer_update(
        model,
        opt,
        scaler,
        backward,
        device=torch.device("cpu"),
        world=1,
        max_grad_norm=1.0,
        max_retries=3,
        on_overflow=lambda *args: overflows.append(args),
    )
    assert result == "same-batch" and retries == 1
    assert draws[0] == draws[1] and len(draws) == 2
    assert overflows == [(1, 1024.0, 512.0)]
    assert torch.isfinite(norm) and float(norm) == pytest.approx(100.0)
    # Applying an optimizer update on the overflow would corrupt this weight.
    assert float(model.weight.detach()) == pytest.approx(0.9)


def test_amp_persistent_nonfinite_gradients_fail_without_changing_weights():
    torch = pytest.importorskip("torch")
    from kodoom.schema import RecordError
    from kodoom.training.worker import optimizer_update

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
    attempts = []

    def backward():
        attempts.append(1)
        scaler.scale(model.weight.sqrt().sum()).backward()

    with pytest.raises(RecordError, match="persist after bounded"):
        optimizer_update(
            model,
            opt,
            scaler,
            backward,
            device=torch.device("cpu"),
            world=1,
            max_grad_norm=1.0,
            max_retries=2,
        )
    assert len(attempts) == 3
    assert float(model.weight.detach()) == 0.0
    assert scaler.get_scale() == 2.0
