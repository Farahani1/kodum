# Agent instructions for kodoom

These instructions are for any coding agent (Claude Code, Codex, Cursor, …) working in this repository.

## The project

kodoom builds a Persian typed-decision data and evaluation layer (dataset, skills suite, harness) with small reference models on top. `docs/project-plan.md` is the big picture and the source of truth for scope, data rules, licenses and milestones. Read the parts relevant to your task before changing code, and do not work against it. If a change needs the plan to change, update the plan in the same piece of work, bump its version and add a row to its version history.

## Commits: one insightful commit for every change

Commit every change you make. Do not leave work uncommitted at the end of a task, and do not pile unrelated changes into one commit.

**One commit = one logical change.** A new module with its tests is one commit. A bug fix is one commit. A rename across files is one commit. If you need "and" to describe a commit, it is probably two.

**The message must teach the reader something the diff does not.** The diff already shows *what* changed; the message explains *why*, and what someone needs to know later.

Format:

```
<area>: <imperative summary, at most 72 characters>

<Why this change is needed: the problem, the plan section, or the
failure it prevents.>

<Decisions and trade-offs: what was chosen, what was rejected and why,
any assumption a future reader could trip over.>

<How it was verified: the command run and its result, e.g.
"pytest tests/test_normalize.py: 14 passed", or "not verified: needs a
T4, covered by colab-preflight".>

Plan: <section, e.g. 1.2 step 6>   (when it relates to the plan)
```

- `<area>` is the part touched: `schema`, `normalize`, `config`, `cli`, `generators`, `translate`, `train`, `eval`, `docs`, `agents`, `build`, …
- Summary in the imperative ("Add Jalali date generator", not "Added" or "Adds").
- Wrap the body at about 72 characters. Skip a paragraph only when it would say nothing.
- Be honest about verification. Never claim a test passed that you did not run.
- Good: `normalize: map Arabic ي/ك to Persian ی/ک before digit folding`, with a body explaining that digit folding compares strings and would otherwise miss mixed-script matches.
- Bad: `update files`, `fix bug`, `wip`, `changes`.

Before committing, run the fast checks (below) and commit only when they pass. If a check fails for a reason outside your change, say so in the message.

## Where things are

- `src/kodoom/`: the package. `config.py` + `profiles/` (run profiles), `schema.py` (the record and JSONL I/O), `sources.py` (source registry and license/split rules), `normalize.py` (the one Persian normalizer), `jalali.py` (the Jalali calendar), `generators/` (code-labeled skill generators: `common.py` pair-building loop, `numbers.py` written forms of numbers and dates, one module per generator, Persian templates in `templates/*.toml`; five so far: jalali-dates, digit-forms, toman-rial, business-hours, iranian-formats), `runs.py` (run directories: logs, checkpoints, resume), `metrics.py` (accuracy, ECE, KL, bootstrap, temperature scaling; pure Python), `predictions.py` + `calibration.py` (the prediction file and `calibration.json`), `evaluation.py` + `baselines.py` (the scoring harness), `typed_decisions.py` + `helmo.py` (the two English sources as records; `kodoom fetch`, `kodoom fields`), `translate/` (`rules.py`: translate/keep table per workflow; `checks.py`: automatic checks; `glossary.py` + `glossary.toml`: the term glossary and its check; `pipeline.py`: `Translator`, the stub, resumable `translate_file`; `hf.py`: TranslateGemma and chat-model translators (torch imported lazily); `report.py`: `kodoom translations`; `kodoom translate`), `inspect_hf.py` (`kodoom inspect`), `check.py` (`kodoom check`), `cli.py` (the `kodoom` command).
- `notebooks/colab.ipynb`: the only notebook; thin, tested by `tests/test_notebook.py`. `docs/colab.md` explains the Colab workflow.
- `tests/`: one test file per module. New code comes with tests in the same commit.
- `docs/project-plan.md`: the plan.
- Reuse these instead of re-implementing them: build records with `kodoom.schema.Record`, check them with `kodoom.sources.check_record`, normalize with `kodoom.normalize.normalize`, write with `write_jsonl` / `append_jsonl` (the latter for resumable steps). Every training or long-running step opens a `kodoom.runs.Run` and saves through `save_latest` / `save_best`, never by writing checkpoints to Drive directly, so resume and the storage budget hold.
- Write invisible characters (ZWNJ, RLM, NBSP) as `\u` escapes in source; a test enforces it. Persian templates are written with plain spaces and cleaned on load.
- A new skill generator: labels come from code, every item is one half of a minimal pair, digits stay raw, held-out templates give exactly the test split, and it registers in `generators/__init__.py`. Changing templates or logic means bumping the generator's `VERSION` and the fingerprint test.

## Development and Colab are separate

Read "Environments" in the plan. In short:

- One codebase, one pipeline. Profiles (`dev`, `colab-preflight`, `colab`) decide models, data sizes, precision and paths. The code never detects where it is running; the profile is always explicit.
- Everything must run end to end on the `dev` profile on a modest laptop (i3, 4 threads, about 12 GB RAM, CPU only) in under 5 minutes. Never make the `dev` path need a GPU, real model weights or CUDA-only packages.
- No logic in notebooks. Colab notebooks only mount Drive, clone the repo, install, and call the same command.
- CUDA-only dependencies (e.g. `bitsandbytes`) live in the `colab` extra and are imported lazily.
- Code must run on Windows and Linux: use `pathlib`, write text files as UTF-8 explicitly, no shell-specific steps.

## Data rules that code must enforce

- Every record carries `source`, `license`, `split` and `origin`. Never drop them.
- Test-only sources (ParsiNLU, PersianQA, Belebele, own STT data) never enter training. Khayyam / PersianMMLU is excluded entirely. Only permissively usable data enters training.
- Split by `source_id`, so a case and its translation always land on the same side.
- The full normalizer (`normalize`) runs when building the training mix and at inference, the same function both times. Published data is never fully normalized: translations get `clean_orthography` only, and the skills suite keeps its raw digit forms and spellings, because the benchmark must test other models on them.
- Never commit datasets, model weights, checkpoints or run outputs. They go to `data/`, `runs/` or Drive, all git-ignored.

## Fast checks

```
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```
