"""Explicit notebook bootstrap; no provider imports on the local CPU path."""

from __future__ import annotations

import base64
import os
import platform
import subprocess
import sys
from pathlib import Path

PROVIDERS = ("kaggle", "colab", "generic")
REPOSITORY = "https://github.com/Farahani1/kodum.git"


def validate_tpu_runtime() -> None:
    if sys.platform != "linux" or not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError("Kaggle TPU bootstrap supports Linux with Python 3.11 through 3.13")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(int(x) for x in version.split(".")[:2]) < (2, 31):
        raise RuntimeError("The pinned libtpu wheel requires glibc 2.31 or newer")


def secret(provider: str, name: str) -> str | None:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")
    if os.environ.get(name):
        return os.environ[name]
    try:
        if provider == "kaggle":
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret(name)
        if provider == "colab":
            from google.colab import userdata

            return userdata.get(name)
    except Exception:
        # Public repositories need no GitHub token. Gated access is checked later.
        return None
    return None


def bootstrap(
    provider: str,
    revision: str,
    directory: str | Path,
    *,
    backend: str = "torch",
    install_dependencies: bool = True,
) -> Path:
    """Fetch clean code and install the explicitly selected accelerator dependencies."""
    if provider not in PROVIDERS or not revision or revision.startswith("-"):
        raise ValueError("select a valid provider and code revision")
    if backend not in ("torch", "jax"):
        raise ValueError("select backend='torch' or backend='jax' explicitly")
    if backend == "jax":
        validate_tpu_runtime()
    if provider == "colab":
        from google.colab import drive

        drive.mount("/content/drive")
    token = secret(provider, "GITHUB_TOKEN")
    hf_token = secret(provider, "HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    if install_dependencies and backend == "torch":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    elif install_dependencies:
        os.environ["JAX_PLATFORMS"] = "tpu"
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    checkout = Path(directory).resolve()
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    if token:
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        # Authentication belongs in the child environment, never the command/remote URL.
        count = int(env.get("GIT_CONFIG_COUNT", "0"))
        env.update(
            {
                f"GIT_CONFIG_KEY_{count}": "http.https://github.com/.extraheader",
                f"GIT_CONFIG_VALUE_{count}": f"Authorization: Basic {auth}",
                "GIT_CONFIG_COUNT": str(count + 1),
            }
        )

    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=90,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "GitHub checkout timed out; check Internet and retry on CPU"
            ) from None
        if result.returncode:
            # Do not repeat remote/credential diagnostics into notebook output.
            raise RuntimeError("Git bootstrap failed; check revision and repository access")
        return result.stdout.strip()

    if not (checkout / ".git").exists():
        if checkout.exists() and any(checkout.iterdir()):
            raise RuntimeError(f"checkout directory is not empty: {checkout}")
        git("init", str(checkout))
    elif git("-C", str(checkout), "status", "--porcelain"):
        raise RuntimeError("checkout has local edits; use a fresh directory")
    git("-C", str(checkout), "fetch", "--depth", "1", REPOSITORY, revision)
    git("-C", str(checkout), "checkout", "--detach", "FETCH_HEAD")
    sha = git("-C", str(checkout), "rev-parse", "HEAD")
    if provider != "generic" and install_dependencies:
        extra = "tpu" if backend == "jax" else "gpu"
        # TensorFlow 2.19 has no Python 3.13 wheel. Keep older kernels on their
        # existing lock and select the separately resolved 2.20 set for 3.13.
        constraints = "tpu-py313" if backend == "jax" and sys.version_info[:2] == (3, 13) else extra
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                f"{checkout}[{extra}]",
                "-c",
                str(checkout / "constraints" / f"{constraints}.txt"),
            ],
            check=True,
        )
        if backend == "jax":
            # Probe in a child: the notebook kernel must not own TPU devices also
            # needed by the translation process. A failed install/probe stops Run all.
            subprocess.run(
                [sys.executable, "-c", "from kodoom.tpu import probe_tpu; print(probe_tpu())"],
                env=dict(os.environ, PYTHONPATH=str(checkout / "src")),
                check=True,
            )
    os.chdir(checkout)
    sys.path.insert(0, str(checkout / "src"))
    print(f"Code revision: {sha}")
    return checkout


def cpu_preflight(
    provider,
    revision,
    directory,
    *,
    campaign_id,
    repo,
    operator="owner",
    takeover=False,
    profile="kaggle-tpu",
    backend="jax",
):
    """Check cheap failures before a TPU queue, without changing the notebook environment."""
    import json
    import re
    import shutil

    validate_tpu_runtime()
    if (profile, backend) != ("kaggle-tpu", "jax"):
        raise ValueError("Keep PROFILE='kaggle-tpu' and BACKEND='jax' for this campaign")
    if provider != "kaggle" or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("CPU preflight requires Kaggle and the notebook's full code SHA")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Set HF_DATASET_REPO to your private owner/dataset repository")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", campaign_id):
        raise ValueError("CAMPAIGN_ID must be 3-64 lowercase letters/digits/hyphens")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", operator) or type(takeover) is not bool:
        raise ValueError("Set a short OPERATOR name and a boolean TAKEOVER")
    if not shutil.which("git"):
        raise RuntimeError("Git is missing from this runtime")
    token = secret(provider, "HF_TOKEN")
    if not token or not token.strip():
        raise ValueError(
            "HF_TOKEN is missing or inaccessible. Open Kaggle Add-ons > Secrets, "
            "create HF_TOKEN with private dataset write access, and enable it for this notebook. "
            "Creating a secret without enabling notebook access is insufficient."
        )
    os.environ["HF_TOKEN"] = token
    print("PASS: Python/platform, settings and HF_TOKEN availability (token hidden)", flush=True)
    checkout = bootstrap(provider, revision, directory, backend="jax", install_dependencies=False)
    # An isolated CPU environment avoids downgrading Kaggle's preinstalled ML stack.
    environment = Path(f"/tmp/kodoom-cpu-preflight-py{sys.version_info[0]}{sys.version_info[1]}")
    python = environment / "bin/python"
    child_env = dict(os.environ, PYTHONPATH=str(checkout / "src"), JAX_PLATFORMS="cpu")

    def checked(words, timeout, message):
        try:
            result = subprocess.run(
                words, env=child_env, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(message + " (timed out; retry on CPU)") from None
        if result.returncode:
            raise RuntimeError(message) from None

    if not python.exists():
        checked(
            [sys.executable, "-m", "venv", str(environment)],
            90,
            "Could not create the isolated CPU preflight environment",
        )
    constraints = "tpu-py313.txt" if sys.version_info[:2] == (3, 13) else "tpu.txt"
    print("Installing lightweight CPU checks in /tmp (no JAX or model weights)...", flush=True)
    checked(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-e",
            f"{checkout}[preflight]",
            "-c",
            str(checkout / "constraints" / constraints),
            "--disable-pip-version-check",
        ],
        240,
        "CPU tools installation failed; check Internet/PyPI access and retry before queuing a TPU",
    )
    report_path = Path("/kaggle/working/kodoom/cpu-preflight.json")
    report_path.unlink(missing_ok=True)
    words = [
        str(python),
        "-u",
        "-m",
        "kodoom.notebook_preflight",
        "--repo",
        repo,
        "--campaign",
        campaign_id,
        "--operator",
        operator,
        "--revision",
        revision,
        "--report",
        str(report_path),
    ]
    if takeover:
        words.append("--takeover")
    # The child emits only explicitly safe check results, never provider exception text.
    try:
        outcome = subprocess.run(words, env=child_env, timeout=900, check=False)
    except subprocess.TimeoutExpired:
        raise RuntimeError("CPU preflight timed out; retry it before requesting a TPU") from None
    if outcome.returncode or not report_path.exists():
        raise RuntimeError("CPU preflight failed. Fix the FAIL items above before requesting a TPU")
    return json.loads(report_path.read_text("utf-8"))
