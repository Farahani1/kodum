# kodoom

Persian Typed Decisions: an open bilingual dataset, a Persian skills suite and an evaluation harness for typed-decision models (choice, score and yes/no questions answered with calibrated probabilities), with CPU-friendly reference models.

The big picture of the whole project is in [docs/project-plan.md](docs/project-plan.md).
The separate [workflow change plan](docs/workflow-change-plan.md) covers the
planned reference/execution notebook split and Colab/Kaggle portability.
[agent.md](agent.md) defines that pipeline and how agents update each stage.
Ideas beyond the current plan are collected in [docs/future-work.md](docs/future-work.md). How our data compares with earlier Persian decision models: [docs/data-comparison.md](docs/data-comparison.md).

## Setup (laptop)

Python 3.11 or newer. CPU only; no GPU or model weights are needed for development.

```
git clone https://github.com/Farahani1/kodum.git
cd kodum
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Usage

Every command that depends on the environment takes an explicit profile: `dev` (laptop), `colab-preflight` or `colab` (T4).

```
kodoom info --profile dev                 # show the resolved settings
kodoom validate data/some-records.jsonl   # check records against the schema and source rules
kodoom generate --profile dev             # the five code-labeled Persian skill sets -> <data_dir>/skills (Drive on Colab)
kodoom baseline uniform --data FILE.jsonl --out preds.jsonl   # trivial baselines: uniform, prior, oracle
kodoom score --gold FILE.jsonl --pred preds.jsonl --by task_family   # metrics, calibration, minimal pairs
kodoom calibrate --gold FILE.jsonl --pred preds.jsonl --out calibration.json   # a temperature per question type
kodoom inspect LocalLLaMA/typed-decisions --profile colab-preflight   # structure of a Hugging Face dataset (Colab)
kodoom check --profile dev                # is this machine ready for the profile?
kodoom runs --profile dev                 # runs on disk and what can be resumed
```

## Colab

The existing [`notebooks/colab.ipynb`](notebooks/colab.ipynb) mixes historical trials
with current work; run setup and only the cells for your active stage, not Run all.
The documented next stage is the free-text gate (main plan 1.2 step 1).
Updating, resuming and Drive storage: [docs/colab.md](docs/colab.md).
The current Kaggle TPU v5e-8 / Gemma 3 27B experiment runner is [`notebooks/execution.ipynb`](notebooks/execution.ipynb);
the complete recipe record is [`notebooks/reference.ipynb`](notebooks/reference.ipynb).
Kaggle setup, private output saving and review are in
[docs/execution-workflow.md](docs/execution-workflow.md). Training stages and
live Kaggle validation remain pending.

## Layout

| Path | What |
| --- | --- |
| `src/kodoom/config.py`, `src/kodoom/profiles/` | Run profiles (plan: Environments) |
| `src/kodoom/schema.py` | The one decision record, JSONL I/O (plan 1.3) |
| `src/kodoom/sources.py` | Data sources, licenses, test-only and excluded rules (plan 2.1) |
| `src/kodoom/normalize.py` | The Persian normalizer (plan 1.2 step 6) |
| `src/kodoom/jalali.py` | The Jalali calendar: leap years, validity, conversion, weekdays |
| `src/kodoom/generators/` | Code-labeled skill generators with minimal pairs; `templates/*.toml` are the Persian templates (plan 1.1) |
| `src/kodoom/metrics.py`, `evaluation.py`, `predictions.py`, `calibration.py`, `baselines.py` | The scoring harness: metrics, prediction files, `calibration.json`, baselines (plan 3.2, 3.6) |
| `src/kodoom/inspect_hf.py` | `kodoom inspect`: the exact structure of a Hugging Face dataset |
| `src/kodoom/runs.py` | Run directories: crash-safe logs and checkpoints on Drive, resume, registry |
| `src/kodoom/check.py` | `kodoom check`: Drive mounted, free space, GPU, model cache |
| `src/kodoom/cli.py` | The `kodoom` command |
| `notebooks/colab.ipynb` | The thin Colab notebook ([docs/colab.md](docs/colab.md)) |
| `notebooks/execution.ipynb`, `notebooks/reference.ipynb` | Active gate and complete recipe history ([docs/execution-workflow.md](docs/execution-workflow.md)) |
| `src/kodoom/workflow.py`, `src/kodoom/artifacts.py`, `workflows/` | Gate orchestration, provenance and private output bundles |
| `tests/` | Tests; `pytest` runs them all in about a second |

Datasets, checkpoints and run outputs go in `data/`, `runs/` or Drive and are never committed.

## Licenses

- **Code: [0BSD](LICENSE)**, the least restrictive OSI-approved license: use it for any purpose, with no attribution required.
- **This project's own data, [CC0-1.0](LICENSE-DATA)** (public-domain dedication): the generated Persian skill data and the templates that produce it.
- **Data derived from other people's work keeps their license.** For example typed-decisions-fa is a translation of an Apache-2.0 dataset, so it stays Apache-2.0 with attribution. `kodoom.sources` records every source's license and the code enforces which ones may enter training.
