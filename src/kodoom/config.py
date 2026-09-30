"""Run profiles: one codebase, the profile decides where and how big.

A profile is a TOML file. The built-in ones (``dev``, ``colab-preflight``,
``colab``) ship inside the package; any other path to a TOML file works too.
The code never guesses where it runs: the profile is always named explicitly.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

BUILTIN_PROFILES = ("dev", "colab-preflight", "colab")
DEVICES = ("cpu", "cuda")
PRECISIONS = ("fp32", "fp16")

# Every key a profile may contain, by section. Unknown keys are an error,
# so a typo cannot silently fall back to a default.
_SCHEMA: dict[str, dict[str, type | tuple[type, ...]]] = {
    "run": {"seed": int, "runs_dir": str},
    "compute": {"device": str, "precision": str, "threads": int},
    "data": {"max_cases_per_source": int},
    "storage": {"scratch_dir": str, "cache_dir": str, "reserve_gb": (int, float)},
}


class ProfileError(ValueError):
    """A profile is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class Profile:
    name: str
    seed: int
    runs_dir: Path
    device: str
    precision: str
    threads: int | None  # None: leave the library default
    max_cases_per_source: int | None  # None: use every case
    # Fast local disk: checkpoints are written here, then copied to runs_dir.
    scratch_dir: Path
    # Where downloaded base models go (HF_HOME). Never on Drive (plan: Storage budget).
    cache_dir: Path
    # Space to leave free on the runs_dir disk after any checkpoint write.
    reserve_gb: float


def load_profile(name_or_path: str | Path, *, runs_dir: str | Path | None = None) -> Profile:
    """Load a built-in profile by name, or a profile TOML file by path.

    ``runs_dir`` overrides the profile's output directory.
    """
    name, raw = _read(name_or_path)
    _check_keys(raw)

    run, compute, data = raw.get("run", {}), raw.get("compute", {}), raw.get("data", {})
    storage = raw.get("storage", {})
    for section, key in (("run", "seed"), ("run", "runs_dir"), ("compute", "device"),
                         ("compute", "precision"), ("storage", "scratch_dir"),
                         ("storage", "cache_dir")):  # fmt: skip
        if key not in raw.get(section, {}):
            raise ProfileError(f"profile {name!r}: missing [{section}] {key}")

    profile = Profile(
        name=name,
        seed=run["seed"],
        runs_dir=Path(runs_dir if runs_dir is not None else run["runs_dir"]),
        device=compute["device"],
        precision=compute["precision"],
        threads=compute.get("threads"),
        max_cases_per_source=data.get("max_cases_per_source"),
        scratch_dir=Path(storage["scratch_dir"]),
        cache_dir=Path(storage["cache_dir"]),
        reserve_gb=float(storage.get("reserve_gb", 1.0)),
    )
    _check_values(profile)
    return profile


def _read(name_or_path: str | Path) -> tuple[str, dict[str, Any]]:
    if str(name_or_path) in BUILTIN_PROFILES:
        name = str(name_or_path)
        text = resources.files("kodoom").joinpath("profiles", f"{name}.toml").read_text("utf-8")
    else:
        path = Path(name_or_path)
        if not path.is_file():
            raise ProfileError(
                f"unknown profile {str(name_or_path)!r}: not one of "
                f"{', '.join(BUILTIN_PROFILES)} and not a file"
            )
        name, text = path.stem, path.read_text("utf-8")
    try:
        return name, tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ProfileError(f"profile {name!r}: invalid TOML: {e}") from e


def _check_keys(raw: dict[str, Any]) -> None:
    for section, values in raw.items():
        if section not in _SCHEMA:
            raise ProfileError(f"unknown section [{section}]")
        if not isinstance(values, dict):
            raise ProfileError(f"[{section}] must be a table")
        for key, value in values.items():
            expected = _SCHEMA[section].get(key)
            if expected is None:
                raise ProfileError(f"unknown key [{section}] {key}")
            # bool is a subclass of int; reject it where an int is expected.
            if isinstance(value, bool) or not isinstance(value, expected):
                raise ProfileError(f"[{section}] {key} has the wrong type: {value!r}")


def _check_values(p: Profile) -> None:
    if p.device not in DEVICES:
        raise ProfileError(f"profile {p.name!r}: device must be one of {DEVICES}")
    if p.precision not in PRECISIONS:
        raise ProfileError(f"profile {p.name!r}: precision must be one of {PRECISIONS}")
    if p.device == "cpu" and p.precision == "fp16":
        raise ProfileError(f"profile {p.name!r}: fp16 needs a GPU; use fp32 on cpu")
    if p.threads is not None and p.threads < 1:
        raise ProfileError(f"profile {p.name!r}: threads must be at least 1")
    if p.reserve_gb < 0:
        raise ProfileError(f"profile {p.name!r}: reserve_gb cannot be negative")
    if p.max_cases_per_source is not None and p.max_cases_per_source < 1:
        raise ProfileError(f"profile {p.name!r}: max_cases_per_source must be at least 1")
