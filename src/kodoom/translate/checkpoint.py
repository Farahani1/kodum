"""Versioned Kaggle Flax checkpoint access without initializing an accelerator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

KAGGLE_MODEL = "google/gemma-3/flax/gemma3-27b-it/1"


def describe_checkpoint(directory: Path) -> dict:
    tokenizer = directory / "tokenizer.model"
    checkpoint = directory / "gemma3-27b-it"
    if not tokenizer.is_file() or not (checkpoint / "_METADATA").is_file():
        raise RuntimeError("Expected the official Gemma 3 27B Flax checkpoint and tokenizer.model")
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            entry = {"name": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size}
            # Versioned Kaggle assets identify weights; hash small metadata and
            # tokenizer files without rereading tens of GB at every preflight.
            metadata = path.name in {"_METADATA", "_CHECKPOINT_METADATA", "_sharding", ".zarray"}
            if path == tokenizer or (
                entry["bytes"] <= 2 * 1024**2 and (metadata or path.suffix == ".json")
            ):
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append(entry)
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {
        "handle": KAGGLE_MODEL,
        "fingerprint": fingerprint,
        "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        "files": files,
        "total_bytes": sum(entry["bytes"] for entry in files),
        "weight_identity": "immutable Kaggle version plus file sizes; small files SHA256",
    }


def prepare_checkpoint() -> tuple[Path, dict]:
    import kagglehub
    from kagglehub.env import is_in_kaggle_notebook

    # On Kaggle, Hub attaches the asset read-only under /kaggle/input. Do not
    # accidentally download a 54GB archive into the saved working directory.
    if not is_in_kaggle_notebook():
        raise RuntimeError("The TPU checkpoint adapter requires a Kaggle notebook session")
    try:
        directory = Path(kagglehub.model_download(KAGGLE_MODEL))
    except Exception as exc:
        raise RuntimeError(
            "Kaggle model access failed. Accept Gemma terms for google/gemma-3/flax/"
            "gemma3-27b-it and attach version 1 in the notebook Inputs."
        ) from exc
    if not directory.resolve().is_relative_to(Path("/kaggle/input")):
        raise RuntimeError("KaggleHub must mount model weights under /kaggle/input")
    return directory, describe_checkpoint(directory)
