"""Build the thin, three-model Kaggle evaluation notebook without saved outputs."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, text):
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build():
    cells = [
        cell(
            "markdown",
            """# Kaggle comparison: multilingual Laya, Kodoom, DibaOne X1

Run all three models on the **same selected samples**, chosen once by whole
source case with a fixed seed. Includes all four ParsiNLU tasks, PersianQA,
and Persian translations grouped by typed-decisions workflow and Helmo topic.

Enable **Internet**, **GPU T4** (one GPU is sufficient), and the **HF_TOKEN**
Kaggle Secret with read access to `Farahani/kodoom` and `Farahani/kodoom-en-fa`.
Keep the notebook/output private. Save a Kaggle version with output when done.
Use a fresh session and new WORK directory for each changed configuration.

X1 supports choices only, so all three receive the same choice-form requests.
Converted noul/score questions retain gold, option order and original type in
the reports. These are converted-choice results, not native score/noul metrics.
Helmo has only a source training split and is reported separately as a diagnostic.
No test-set calibration or checkpoint selection runs here.
""",
        ),
        cell(
            "code",
            """from pathlib import Path

CODE_BRANCH = "codex/evaluate-parsinlu-persianqa"
MODEL_REPO = "Farahani/kodoom"
MODEL_RUN_ID = "laya-full-v1"
MODEL_REVISION = "f315b5da129e2b1df55d971ddf2053c3cbfe5282"
DATASET_REPO = "Farahani/kodoom-en-fa"
DATASET_REVISION = "1b98bbba48043c62c6ae2c821a55b6f8a5e7a927"
PARSINLU_REVISION = "a3b08d0b93ca84e0017e2f490a2aa3ab36c79885"
DIBA_REVISION = None  # optional exact Hub SHA; resolved once and saved if None
PERSIANQA_REVISION = None  # optional exact Git SHA from a previous samples.json

SEED = 17
NATIVE_CASES_PER_GROUP = 100  # per task; sentiment food/movie sampled separately
TRANSLATED_CASES_PER_GROUP = 3  # per workflow/topic; retain all questions in each case
INCLUDE_HELMO = True  # train-only source diagnostic; never merged into held-out scores
MICROBATCH = 4  # lower to 1 if the selected T4 runs out of memory

WORK = Path("/kaggle/working/kodoom-three-model-eval")
CHECKOUT = WORK / "kodum"
OUTPUT = WORK / "results"
WORK.mkdir(parents=True, exist_ok=True)
""",
        ),
        cell(
            "markdown",
            """## Bootstrap the evaluation code

The evaluation branch must be pushed before running this cell. Its resolved
commit is recorded. The installer keeps Kaggle's existing CUDA PyTorch.
If reusing the checkout after a code update, start a fresh Kaggle session.
""",
        ),
        cell(
            "code",
            """import importlib
import os
import subprocess
import sys
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if not CHECKOUT.exists():
    subprocess.run([
        "git", "clone", "--branch", CODE_BRANCH, "--single-branch",
        "https://github.com/Farahani1/kodum.git", str(CHECKOUT)
    ], check=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "-e", f"{CHECKOUT}[train,eval]",
    "-c", str(CHECKOUT / "constraints/gpu.txt")
], check=True)
# Editable-install .pth files are not reprocessed by an already running kernel.
source_root = (CHECKOUT / "src").resolve()
sys.path.insert(0, str(source_root))
importlib.invalidate_caches()
import kodoom  # noqa: E402

if Path(kodoom.__file__).resolve().parent != source_root / "kodoom":
    raise RuntimeError("A different kodoom checkout is already imported; restart the kernel.")
print("kodoom imported from:", kodoom.__file__)
CODE_REVISION = subprocess.check_output(
    ["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True
).strip()
print("kodum code revision:", CODE_REVISION)
""",
        ),
        cell(
            "markdown",
            """## Resolve datasets and select samples once

ParsiNLU uses pinned official test files. PersianQA resolves its Git revision
and reads explicit test/dev/validation files. Translations come from the exact
accepted release with inventory/hash verification. Typed-decisions uses test
cases only; Helmo keeps its source `train` split and diagnostic label.

Sampling uses a stable seeded hash, retains complete cases and does not use
gold labels. `samples.json` records IDs, group counts, pins and content hashes.
Changing the sample limits creates a different comparison.
""",
        ),
        cell(
            "code",
            """from kaggle_secrets import UserSecretsClient
from kodoom.model_comparison import prepare_benchmarks, save_json

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
records_by_benchmark, samples = prepare_benchmarks(
    work=WORK,
    selection_path=CHECKOUT / "configs/dataset-baseline.toml",
    parsinlu_revision=PARSINLU_REVISION,
    dataset_repo=DATASET_REPO,
    dataset_revision=DATASET_REVISION,
    native_cases=NATIVE_CASES_PER_GROUP,
    translated_cases=TRANSLATED_CASES_PER_GROUP,
    seed=SEED,
    include_helmo=INCLUDE_HELMO,
    persianqa_revision=PERSIANQA_REVISION,
)
save_json(WORK / "selected-samples.json", samples)
""",
        ),
        cell(
            "markdown",
            """## Run all three models and save comparisons

Models load sequentially to fit one T4. Each benchmark's predictions are saved
immediately. All three use the project's input normalizer and identical option
order. Model-specific context/option budgets and truncation remain visible in
prediction metadata; model inference latency excludes source download and load.

The report includes per-model metrics and failures, pairwise source-case
bootstrap intervals, three-way common-item results, workflow/topic breakdowns
and original question types. X1 uses its shipped calibration; both Laya models
use temperature 1. No calibration is fitted on these samples.
""",
        ),
        cell(
            "code",
            """from kodoom.model_comparison import run_comparison

report = run_comparison(
    records_by_benchmark, samples,
    work=WORK, output=OUTPUT,
    model_repo=MODEL_REPO, model_revision=MODEL_REVISION,
    model_run_id=MODEL_RUN_ID, code_revision=CODE_REVISION,
    seed=SEED, microbatch=MICROBATCH, diba_revision=DIBA_REVISION,
)
print("Results saved to", OUTPUT)
""",
        ),
        cell(
            "markdown",
            """## Inspect and preserve results

`comparison.csv` places every model's answer/probabilities/error beside gold
on the same row. `summary.csv` contains aggregate metrics; `report.json` adds
group/type breakdowns and all paired comparisons. Each benchmark has three
prediction JSONL files. `manifest.json` records models, datasets, sample hashes,
dependencies, failures and output hashes. Outputs omit source passages.

Save a **private Kaggle version with outputs** to preserve these files.
For another run, use a fresh session and a new WORK path; set DIBA_REVISION
from the previous manifest and PERSIANQA_REVISION from samples.json to retain
the same input pins. Full restart and
protocol details: `docs/external-evaluation-handoff.md`.
""",
        ),
        cell(
            "code",
            """import csv
from IPython.display import Markdown, display

with (OUTPUT / "summary.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
columns = ["benchmark", "model", "n", "accuracy", "coverage", "failed", "common_accuracy"]
table = "| " + " | ".join(columns) + " |\\n"
table += "| " + " | ".join("---" for _ in columns) + " |\\n"
table += "\\n".join("| " + " | ".join(row[key] for key in columns) + " |" for row in rows)
display(Markdown(table))
print("Save private Kaggle outputs from:", OUTPUT)
""",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "gpu", "isGpuEnabled": True, "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for index, item in enumerate(cells):
        item["id"] = f"comparison-{index:02d}"
    destination = ROOT / "notebooks/external-evaluation.ipynb"
    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    # Generation is a dev task: use the same Ruff notebook rules as CI.
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", str(destination), "--select", "I", "--fix"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", str(destination)],
        check=True,
        capture_output=True,
    )


if __name__ == "__main__":
    build()
