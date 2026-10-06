"""Explicit notebook bootstrap; no provider imports on the local CPU path."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

PROVIDERS = ("kaggle", "colab", "generic")
REPOSITORY = "https://github.com/Farahani1/kodum.git"


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
    provider: str, revision: str, directory: str | Path, *, backend: str = "torch"
) -> Path:
    """Fetch clean code, install the GPU extra, and leave caches outside saved outputs."""
    if provider not in PROVIDERS or not revision or revision.startswith("-"):
        raise ValueError("select a valid provider and code revision")
    if backend not in ("torch", "jax"):
        raise ValueError("select backend='torch' or backend='jax' explicitly")
    if backend == "jax" and (sys.platform != "linux" or sys.version_info < (3, 11)):
        raise RuntimeError("Kaggle TPU bootstrap requires Linux and Python 3.11 or newer")
    if provider == "colab":
        from google.colab import drive

        drive.mount("/content/drive")
    token = secret(provider, "GITHUB_TOKEN")
    hf_token = secret(provider, "HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    if backend == "torch":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    else:
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
        result = subprocess.run(
            ["git", *args], env=env, capture_output=True, text=True, encoding="utf-8"
        )
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
    if provider != "generic":
        extra = "tpu" if backend == "jax" else "gpu"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                f"{checkout}[{extra}]",
                "-c",
                str(checkout / "constraints" / f"{extra}.txt"),
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
