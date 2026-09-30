# kodoom

An open, Persian-capable decision model based on Jev's interface: choice, score and yes/no questions answered with calibrated probabilities.

The big picture of the whole project is in [docs/project-plan.md](docs/project-plan.md).

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
```

## Layout

| Path | What |
| --- | --- |
| `src/kodoom/config.py`, `src/kodoom/profiles/` | Run profiles (plan: Environments) |
| `src/kodoom/schema.py` | The one decision record, JSONL I/O (plan 1.3) |
| `src/kodoom/sources.py` | Data sources, licenses, test-only and excluded rules (plan 2.1) |
| `src/kodoom/normalize.py` | The Persian normalizer (plan 1.2 step 6) |
| `src/kodoom/cli.py` | The `kodoom` command |
| `tests/` | Tests; `pytest` runs them all in about a second |

Datasets, checkpoints and run outputs go in `data/`, `runs/` or Drive and are never committed.
