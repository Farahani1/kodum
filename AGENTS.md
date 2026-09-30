# Agent instructions for kodoom

These instructions are for any coding agent (Claude Code, Codex, Cursor, …) working in this repository.

## The project

kodoom builds an open, Persian-capable typed-decision model. `docs/project-plan.md` is the big picture and the source of truth for scope, data rules, licenses and milestones. Read the parts relevant to your task before changing code, and do not work against it. If a change needs the plan to change, update the plan in the same piece of work, bump its version and add a row to its version history.

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

## Development and Colab are separate

Read "Environments" in the plan. In short:

- One codebase, one pipeline. Profiles (`dev`, `colab-preflight`, `colab`) decide models, data sizes, precision and paths. The code never detects where it is running; the profile is always explicit.
- Everything must run end to end on the `dev` profile on a modest laptop (i3, 4 threads, about 12 GB RAM, CPU only) in under 5 minutes. Never make the `dev` path need a GPU, real model weights or CUDA-only packages.
- No logic in notebooks. Colab notebooks only mount Drive, clone the repo, install, and call the same command.
- CUDA-only dependencies (e.g. `bitsandbytes`) live in the `colab` extra and are imported lazily.
- Code must run on Windows and Linux: use `pathlib`, write text files as UTF-8 explicitly, no shell-specific steps.

## Data rules that code must enforce

- Every record carries `source`, `license`, `split` and `origin`. Never drop them.
- Test-only sources (ParsiNLU, Belebele, own STT data) never enter training. Khayyam / PersianMMLU is excluded entirely.
- Split by `source_id`, so a case and its translation always land on the same side.
- The same normalizer runs on data, at training and at inference.
- Never commit datasets, model weights, checkpoints or run outputs. They go to `data/`, `runs/` or Drive, all git-ignored.

## Fast checks

```
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```
