# Run the Gemma 3 27B experiment on Kaggle TPU

The active request implements project-plan v18, section 1.2 step 1. It compares
Gemma 3 27B with the saved 4B baseline before any translator adoption. Local
checks and Linux dependency resolution pass. **Real Kaggle TPU execution,
resource fit, fresh-session persistence and Persian human review are pending.**

## Kaggle setup

1. Import `notebooks/execution.ipynb` from branch `codex/kaggle-tpu-v5e-8` into
   a new **private** Kaggle notebook. The notebook pins the implementation SHA;
   keep `CODE_REVISION` unchanged for compatible resume.
2. Enable **Internet** and select **TPU v5e-8**. Start with a fresh kernel.
   The pinned runtime supports Linux x86_64, Python 3.11/3.12 and glibc 2.31+.
3. Accept Gemma's model terms on Kaggle and add the official
   [Gemma 3 27B Flax version 1](https://www.kaggle.com/models/google/gemma-3/flax/gemma3-27b-it/1)
   under **Inputs / Models**. Its `gemma3-27b-it` directory and `tokenizer.model`
   are mounted read-only under `/kaggle/input`. KaggleHub uses the notebook's
   model access. An HF token does not authenticate this Flax asset.
4. If the GitHub repository is private, enable a read-only `GITHUB_TOKEN`
   secret for this notebook. A public repository needs no secret. Never paste
   tokens into notebook cells.
5. Keep the defaults `PROVIDER="kaggle"`, `PROFILE="kaggle-tpu"`, `BACKEND="jax"`.
   Leave restore and baseline paths blank for the first run. Use `RUN_GATE=False`
   if you want to inspect and save the measured preflight before running the gate.

## What Run all does

The thin notebook fetches pinned clean code, installs only the TPU extra using
`constraints/tpu.txt` and the 190-package Linux lock, then probes eight v5e chips
and real BF16 arithmetic in a child process. JAX inference and probes stay out
of the notebook kernel so that it does not retain TPU device ownership.

The runner checks writable storage, a minimum 8 GiB available host memory and
4 GiB temporary disk space, plus model access. It measures the token length of
every item selected for the gate before loading weights. No input is truncated.
Weights load directly with eight-device FSDP sharding. Large weights must be
BF16 and non-replicated. The single-item cache is deliberately replicated;
actual cache dtypes, sizes and per-device memory are recorded.

Preflight generates the shortest and longest selected prompts twice to measure
compilation and warm timing, then completes 3 balanced helmo records and 4 typed
cases. A successful matching preflight permits the gate: **40 helmo training
records and 20 typed training cases**, balanced across four workflows. The
request uses greedy decoding, batch 1, 3072 input tokens, 768 output tokens,
a 4096-token cache and a **7200-second budget per stage**. Time checks happen
between items; an in-flight load or generation may exceed the budget before
returning. No precision reduction, source shortening or model substitution occurs.

The model loads once per recipe process. Repeated completed cases skip loading.
Generation uses independent conversations, the existing instructions/glossary,
and the checkpoint's bundled tokenizer. Replies must terminate explicitly;
an output-limit failure does not save a partial case. Each complete case is
appended before its progress line prints. An unfinished case may need rework.

The gate ends at `awaiting-review`; full translation, training and publication
are outside the notebook. Source IDs, option IDs, gold, licenses, keep-fields and
train splits are preserved. This trial does not replace the 4B production decision.

## Files and private saving

Artifacts live under:

```
/kaggle/working/kodoom/data/workflows/free-text-gate/gemma3-27b-tpu-bf16/{preflight,gate}/
```

The stage contains English inputs, translated JSONL, `execution.json`, attempt
logs, reports, case-review CSVs, label-review CSVs and label-template counts.
`tpu-*-<attempt>.json` records checkpoint/tokenizer identity, input lengths,
actual dtypes, memory, load time, XLA compilation durations and generation call
times. Calls with no backend compilation are marked warm. Do not add trace,
MLIR and backend-compile durations together: compiler phases can overlap.

The printed ZIP path under `/kaggle/working/kodoom/bundles` contains only this
stage's private artifacts, with a checksummed manifest. It excludes weights,
compilation caches, checkout and credentials. A finally path exports diagnostics
on execution failures too. A ZIP ready on session disk is not proof of durability.

Save a private notebook version **with outputs**. For interactive results use
Quick Save with output saving enabled; **Save & Run All** executes in a new
session. Check the current [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)
for saving controls and output limits. Confirm the ZIP appears in the saved
output, then attach that notebook output to a fresh session.

## Restart and verify persistence

Set `RESTORE_PREFLIGHT` and/or `RESTORE_GATE` to the attached **27B** ZIP paths.
Use a fresh working directory. Restore verifies checksums and model/stage/mode
identity before extraction and refuses existing destinations. The runner also
checks code, library versions, prompt/glossary, checkpoint fingerprint, allocator,
input selection and generation settings before resuming.

Keep `CODE_REVISION` and the request unchanged. Completed IDs skip translation;
an interrupted case is regenerated. Save the resumed output and verify its
completed IDs, counts and checksums. A failed hardware probe that saved no
translations can retry after correcting the accelerator setup. A corrupted or
incompatible run needs a separate artifact root, not overwritten identities.
Session failure can lose all progress since the last saved output version.

Do not put a 4B bundle into a 27B restore setting. Attach it only as
`BASELINE_GATE_BUNDLE` for comparison. Old CUDA code remains reproducible through
git; `workflows/gpu-gate.toml` preserves its request. The CUDA CLI can select that
request explicitly with `--request workflows/gpu-gate.toml --profile kaggle`.

## Review and the adoption decision

- `review-helmo.csv` and `review-typed-decisions.csv`: complete case meaning and edits.
- `review-labels-*.csv`: every option description with English/Persian state and
  question context. Check negation, severity ordering, actor/action, conditions,
  technical terms and distinctions between options. Fill `meaning_changed` and
  `needs_edit` with `yes`/`no`, plus error category, suggested wording and reviewer.
- `label-templates-*.csv`: recurring variants and occurrence counts. Repeated
  labels are not independent evidence; review them in their case context.

An optional real 4B gate ZIP produces `paired-decisions.csv`, `paired-labels.csv`
and `paired-provenance.json` only after identical selected English records are
verified. The 4B checkpoint/code identities are retained separately. Fill paired
assessments as fixed, regressed, unchanged or uncertain after reading both outputs;
automatic text changes are not quality judgments. Comparison creates a new
private ZIP. Compatible reruns preserve human annotations.

For this experiment the bar is fixed before review: at most **1 meaning change
in 40 helmo records**, at most **10% of helmo records needing edits**, and at most
**10% of typed cases needing edits**. Consider label fidelity separately and
record the owner's adoption/rejection decision with measured resource costs.
Empty review cells remain unreviewed. `review_summary` reports descriptive
counts and never approves a gate. The balanced diagnostic sample cannot estimate
whole-dataset error rates or noisy-label detector coverage. A representative
random sample and the wider human-review plan remain necessary before release.

## References and handoff

Research: `docs/project-plan.md` v18. Active configuration:
`workflows/current.toml`; shared recipes: `workflows/recipes.toml`.
Implementation and pending live checks: `docs/kaggle-tpu-change-plan.md`.
Historical recipes: `notebooks/reference.ipynb`. Agent rules: `AGENTS.md` and
`agent.md`. No provider run or research gate is complete without recorded evidence.
