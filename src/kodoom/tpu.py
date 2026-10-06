"""Lazy TPU readiness probes; importing this module never initializes JAX."""

from __future__ import annotations

import importlib


def validate_devices(devices: list, process_count: int) -> None:
    if process_count != 1 or len(devices) != 8:
        raise RuntimeError("Select Kaggle TPU v5e-8: eight devices on one host are required")
    for device in devices:
        kind = device.device_kind.lower().replace(" ", "").replace("_", "")
        if device.platform != "tpu" or not any(name in kind for name in ("v5e", "v5lite")):
            raise RuntimeError(f"Expected TPU v5e-8, found {device.device_kind}")


def memory_snapshot(devices: list) -> list[dict]:
    result = []
    for device in devices:
        try:
            stats = device.memory_stats() or {}
        except (RuntimeError, NotImplementedError):
            stats = {}
        result.append({"id": device.id, "kind": device.device_kind, "memory": stats})
    return result


def probe_tpu() -> dict:
    try:
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
    except ImportError as exc:
        raise RuntimeError("TPU dependencies missing; bootstrap with backend='jax'") from exc
    devices = jax.devices()
    validate_devices(devices, jax.process_count())
    with jax.default_device(devices[0]):
        probe = jnp.ones((8, 8), dtype=jnp.bfloat16)
        product = (probe @ probe).block_until_ready()
        if not bool(jnp.isfinite(product).all()):
            raise RuntimeError("TPU BF16 probe produced non-finite values")
    return {
        "device": devices[0].device_kind,
        "visible_devices": len(devices),
        "model_dtype": "bfloat16",
        "backend": "jax",
        "devices": memory_snapshot(devices),
    }
