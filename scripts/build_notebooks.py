"""Build empty-output execution/reference notebooks from the current request."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind: str, source: str) -> dict:
    item = {
        "cell_type": kind,
        "id": hashlib.sha256(source.encode()).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def document(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(revision: str) -> dict[str, dict]:
    request = tomllib.loads((ROOT / "workflows/current.toml").read_text("utf-8"))
    settings = f"""PROVIDER = "kaggle"  # kaggle, colab, generic
PROFILE = "kaggle"  # colab on Colab; custom TOML path for generic
CODE_REVISION = {revision!r}
CHECKOUT = "/tmp/kodoom-code"  # use a new path when changing code revisions
RESTORE_PREFLIGHT = ""  # optional attached preflight ZIP file
RESTORE_GATE = ""  # optional attached gate ZIP file
"""
    bootstrap = """import os
import types
import urllib.parse
import urllib.request

if PROVIDER == "kaggle":
    from kaggle_secrets import UserSecretsClient
    try:
        os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        pass  # optional for a public repository
elif PROVIDER == "colab":
    from google.colab import userdata
    try:
        os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
    except Exception:
        pass
url = ("https://api.github.com/repos/Farahani1/kodum/contents/src/kodoom/runtime.py"
       + "?ref=" + urllib.parse.quote(CODE_REVISION, safe=""))
headers = {"Accept": "application/vnd.github.raw+json"}
if os.environ.get("GITHUB_TOKEN"):
    headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
    source = response.read().decode("utf-8")
runtime = types.ModuleType("kodoom_bootstrap")
exec(compile(source, "runtime.py", "exec"), runtime.__dict__)
checkout = runtime.bootstrap(PROVIDER, CODE_REVISION, CHECKOUT)
"""
    restore = """from kodoom.workflow import restore_stage

if RESTORE_PREFLIGHT:
    restore_stage(PROVIDER, PROFILE, "preflight", RESTORE_PREFLIGHT)
if RESTORE_GATE:
    restore_stage(PROVIDER, PROFILE, "gate", RESTORE_GATE)
"""
    preflight = """from kodoom.workflow import run_current

preflight = run_current(provider=PROVIDER, profile=PROFILE,
                        mode="preflight")
"""
    gate = """gate = run_current(provider=PROVIDER, profile=PROFILE,
                   mode="gate",
                   model_revision=preflight["model_revision"])
"""
    end = """print("Gate status:", gate["status"])
print("Save this private output bundle:", gate["bundle"])
print("Review sheets are in the bundle. Stop here for the main-plan review decision.")
"""
    title = f"""# kodoom: current execution

Active stage: **{request["stage"]}** ({request["plan"]}).
Run all performs a 3-record/4-case GPU preflight, then the 40-record helmo and
20-case typed-decisions training gate, using **{request["translator"]}**.
It ends at **{request["stop"]}**, not at full translation.

Before starting: create a private Kaggle notebook, enable Internet and a GPU,
accept `google/gemma-3-4b-it` on Hugging Face, and enable the `HF_TOKEN` secret.
See `docs/execution-workflow.md` for saving/restoring and provider settings.
"""
    execution = document(
        [
            cell("markdown", title),
            cell("code", settings),
            cell("markdown", "## Bootstrap the pinned code\n"),
            cell("code", bootstrap),
            cell("markdown", "## Restore selected private artifacts\n"),
            cell("code", restore),
            cell("markdown", "## GPU preflight\n"),
            cell("code", preflight),
            cell("markdown", "## Free-text gate\n"),
            cell("code", gate),
            cell(
                "markdown",
                "## Private output and review\n\n"
                + request["review"]
                + "\n\n"
                + "Save a private version with outputs, then attach that output in a new session "
                + "and restore its ZIP to verify persistence. "
                + "Source-only saves do not prove durability. "
                + "The live session may lose all progress since its last saved output.\n",
            ),
            cell("code", end),
        ]
    )
    catalog = tomllib.loads((ROOT / "workflows/recipes.toml").read_text("utf-8"))
    reference = [
        cell(
            "markdown",
            "# kodoom: artifact recipe reference\n\n"
            "Read this notebook; use execution.ipynb for current work. The recipes below "
            "describe outputs and decisions, not a Run all workload. Main-plan future "
            "steps are marked blocked or unimplemented. No data or model outputs are committed.\n",
        ),
        cell(
            "markdown",
            "## Shared recipe catalog\n\n"
            + "\n\n".join(
                f"### {r['id']} ({r['status']})\n\nPlan: {r['plan']}.\n\n"
                + (
                    "Command: `kodoom " + " ".join(r["args"]) + " --profile PROFILE`.\n\n"
                    if "args" in r
                    else ""
                )
                + (f"Input: `{r['input']}`. Output: `{r['output']}`.\n\n" if "input" in r else "")
                + r.get("note", "")
                for r in catalog["recipe"]
            ),
        ),
        cell(
            "markdown",
            "## Historical Colab recipes\n\n"
            "The following preserves every legacy cell in its original order. Commands "
            "are shown as text, so reading/rendering this reference cannot run obsolete "
            "experiments or delete caches. Original decisions may be superseded; "
            "main-plan v15 selected Gemma 4B with no 12B fallback.\n",
        ),
    ]
    old = json.loads((ROOT / "notebooks/colab.ipynb").read_text("utf-8"))
    for index, original in enumerate(old["cells"]):
        source = "".join(original["source"])
        if original["cell_type"] == "code":
            source = f"### Legacy cell {index}\n\n````python\n{source}\n````\n"
        reference.append(cell("markdown", source))
    return {"execution.ipynb": execution, "reference.ipynb": document(reference)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, notebook in build(args.revision).items():
        path = ROOT / "notebooks" / name
        if args.check:
            if not path.exists() or json.loads(path.read_text("utf-8")) != notebook:
                print(f"notebook drift: {name}", file=sys.stderr)
                return 1
        else:
            path.write_text(
                json.dumps(notebook, ensure_ascii=True, indent=1) + "\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
