# kodoom

Persian Typed Decisions: an open bilingual dataset, a Persian skills suite and an evaluation harness for typed-decision models (choice, score and yes/no questions answered with calibrated probabilities), with CPU-friendly reference models.

The big picture of the whole project is in [docs/project-plan.md](docs/project-plan.md).
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
kodoom generate --profile dev             # the five code-labeled Persian skill sets -> data/skills/
kodoom check --profile dev                # is this machine ready for the profile?
kodoom runs --profile dev                 # runs on disk and what can be resumed
```

## Colab

Open [`notebooks/colab.ipynb`](notebooks/colab.ipynb) in Colab and Run all. Updating, resuming after a dead session and what goes on Drive: [docs/colab.md](docs/colab.md).

## Layout

| Path | What |
| --- | --- |
| `src/kodoom/config.py`, `src/kodoom/profiles/` | Run profiles (plan: Environments) |
| `src/kodoom/schema.py` | The one decision record, JSONL I/O (plan 1.3) |
| `src/kodoom/sources.py` | Data sources, licenses, test-only and excluded rules (plan 2.1) |
| `src/kodoom/normalize.py` | The Persian normalizer (plan 1.2 step 6) |
| `src/kodoom/jalali.py` | The Jalali calendar: leap years, validity, conversion, weekdays |
| `src/kodoom/generators/` | Code-labeled skill generators with minimal pairs; `templates/*.toml` are the Persian templates (plan 1.1) |
| `src/kodoom/runs.py` | Run directories: crash-safe logs and checkpoints on Drive, resume, registry |
| `src/kodoom/check.py` | `kodoom check`: Drive mounted, free space, GPU, model cache |
| `src/kodoom/cli.py` | The `kodoom` command |
| `notebooks/colab.ipynb` | The thin Colab notebook ([docs/colab.md](docs/colab.md)) |
| `tests/` | Tests; `pytest` runs them all in about a second |

Datasets, checkpoints and run outputs go in `data/`, `runs/` or Drive and are never committed.
