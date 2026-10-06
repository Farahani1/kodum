from dataclasses import replace
from types import SimpleNamespace

import pytest

from kodoom.check import FAIL, OK, _check_device
from kodoom.config import ProfileError, _check_values, load_profile
from kodoom.tpu import memory_snapshot, validate_devices


def devices(kind="TPU v5 lite", count=8):
    return [
        SimpleNamespace(id=i, platform="tpu", device_kind=kind, memory_stats=lambda: {})
        for i in range(count)
    ]


def test_tpu_profiles_are_explicit_and_round_trip():
    for name in ("kaggle-tpu", "kaggle-tpu-preflight"):
        profile = load_profile(name)
        assert (profile.device, profile.backend, profile.precision) == ("tpu", "jax", "bf16")
    assert load_profile("dev").backend == "torch"


@pytest.mark.parametrize(
    "device,backend,precision",
    [("tpu", "torch", "bf16"), ("cuda", "jax", "bf16"), ("tpu", "jax", "fp16")],
)
def test_invalid_accelerator_backend_combinations(device, backend, precision):
    with pytest.raises(ProfileError):
        _check_values(
            replace(load_profile("dev"), device=device, backend=backend, precision=precision)
        )


@pytest.mark.parametrize(
    "kind,count,hosts", [("TPU v4", 8, 1), ("TPU v5e", 4, 1), ("TPU v5e", 8, 2)]
)
def test_wrong_tpu_topology_stops_before_loading(kind, count, hosts):
    with pytest.raises(RuntimeError, match="v5e"):
        validate_devices(devices(kind, count), hosts)


def test_device_check_dispatches_without_torch(monkeypatch):
    monkeypatch.setattr("kodoom.tpu.probe_tpu", lambda: {"device": "TPU v5e", "visible_devices": 8})
    assert _check_device(load_profile("kaggle-tpu")).status == OK

    def missing():
        raise RuntimeError("missing runtime")

    monkeypatch.setattr("kodoom.tpu.probe_tpu", missing)
    assert _check_device(load_profile("kaggle-tpu")).status == FAIL
    assert _check_device(load_profile("dev")).status == OK
    assert len(memory_snapshot(devices())) == 8
