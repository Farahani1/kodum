# Kaggle TPU v5e-8: change plan

Version 3 | Oct 7, 2026 | Local implementation; TPU execution unverified

Branch: `codex/kaggle-tpu-v5e-8`. Baseline: `81f93a5` from
`codex/portable-notebook-workflow`, whose tracked tree was clean and whose push
was confirmed up to date before this branch was created.

This plan extends the [portable workflow plan](workflow-change-plan.md), using
`TPU-*` task IDs. The [project plan](project-plan.md) remains the research source
of truth, and [agent.md](../agent.md) defines execution and storage rules. This
document tracks the implementation and operator handoff. The active request
runs a bounded 27B experiment; it does not adopt a production translator
or pass a research gate.

## Objective and evidence

Run instruction-tuned **Gemma 3 27B** on Kaggle **TPU v5e-8**, initially for a
bounded English-to-Persian comparison with the existing Gemma 3 4B outputs.
Reuse the record pipeline, glossary, translation rules, output cleaning,
automatic checks, review sheets and private bundle contract.

The [TranslateGemma report](https://arxiv.org/pdf/2601.09012), Table 4, reports
English-to-Persian MetricX scores of 4.77 for Gemma 3 4B and 2.98 for Gemma 3 27B
(lower is better). This supports a trial; it neither predicts a label error
percentage nor proves that the project's errors will be repaired. Its Persian
results are automatic evaluations, not a Persian human review. TranslateGemma
12B/27B also score well, but adding them is a separate experiment after this path
works.

Google's [Gemma sharding example](https://gemma-llm.readthedocs.io/en/latest/colab_sharding.html)
loads `gm.nn.Gemma3_27B()` with
`gm.ckpts.CheckpointPath.GEMMA3_27B_IT` and
`sharding=kd.sharding.FSDPSharding()` across eight TPU devices. Use it as the
starting implementation, with a compatible pinned version of Gemma/JAX/Kauldron.
It demonstrates an eight-device TPU setup, not validation of our Kaggle runtime.

[v5e specifications](https://cloud.google.com/tpu/docs/v5e) give 16 GB HBM per
chip. Eight chips provide 128 GB aggregate. Roughly 54 GB for 27B BF16 weights
suggests feasibility, but embeddings, caches, activations, collectives, loading
buffers and compilation increase actual usage. Neither 128 GB nor a successful
device enumeration guarantees that a particular sharding scheme fits.

## Inspected gaps before implementation

| Component | Current behavior | Required change |
| --- | --- | --- |
| `src/kodoom/config.py` | Devices are `cpu`/`cuda`; precisions are `fp32`/`fp16` | Explicit TPU/BF16 support and compatible backend validation |
| `src/kodoom/check.py` | Non-CPU readiness assumes CUDA and torch | Dispatch checks for the explicitly selected backend |
| `src/kodoom/runtime.py` | Provider bootstrap installs `[gpu]`; defaults CUDA visibility to device 0 | Separate TPU dependencies and accelerator bootstrap |
| `src/kodoom/workflow.py` | Hardcodes Gemma 3 4B, CUDA-only gate, BF16 torch probe and GPU 0 visibility | Validated model/backend configurations, TPU preflight and accurate provenance |
| `src/kodoom/translate/hf.py` | PyTorch/Transformers generation with `device_map="auto"` | Retain CUDA path; add a separate JAX generator behind the existing interface |
| `src/kodoom/translate/pipeline.py` | Lazy named translator factories | Register a TPU-specific Gemma 3 27B translator |
| Workflow artifact roots | Stage/mode roots do not distinguish translator experiments | Isolate 4B and 27B artifacts and enforce resume compatibility |
| Operational notebook | Bootstraps the current GPU request | Thin, explicit TPU setup and summaries when the TPU experiment is activated |

`device_map="auto"` and bitsandbytes are not the proposed TPU distribution
mechanism. Moving all weights onto one TPU first, or replicating the 27B model
on every chip, would exceed the per-chip budget. Load directly with sharding.

## Design decisions

- Add a lazy JAX/Gemma generator in proposed `src/kodoom/translate/jax.py`.
  Adapt the existing `Generate` contract: conversations in, one reply per
  conversation in the original order out. Keep `ChatTranslator` responsible for
  prompts and `_pair`/`clean_output` responsible for cleaning.
- Preserve the existing chat instructions, workflow context, glossary and
  deterministic decoding. Verify exact chat-template/tokenizer behavior rather
  than assuming `ChatSampler` reproduces the Transformers conversation. Ensure
  system instructions are included and do not accidentally carry chat history
  between independent translation items.
- Request BF16 weights/compute explicitly where the pinned library supports
  them, and record actual parameter/cache dtypes. Do not infer them from model
  names or import a CUDA quantization configuration into this backend.
- Start with batch size 1 and a conservative explicit input/cache budget.
  Measure the source lengths before selecting limits. Reject oversized inputs
  clearly; do not silently truncate states, questions or option descriptions.
- Reuse the official sharded loading strategy, then verify per-chip allocation
  and generation cache placement. Fixed shape buckets and cache sizing should
  control repeated XLA compilation. Increasing batch size is a measured follow-up.
- Keep `dev` CPU-only and lightweight, with no JAX, TPU runtime, CUDA or large
  weights required. Retain the existing GPU/Colab profiles and implementations.
- Treat hardware/provider selection and model selection as explicit configuration.
  A failed TPU check must stop execution, without a silent CPU/GPU fallback.

## Python 3.13 compatibility correction

The first live notebook attempt reached bootstrap on Python 3.13 and was
rejected by the earlier 3.11/3.12 guard. The new interpreter-specific lock
updates only TensorFlow and TensorBoard to 2.20, omitting the no-longer-required
tensorflow-io-gcs-filesystem dependency. Gemma/JAX/Kauldron and all other
pins remain unchanged. Its 189 packages resolve for Linux Python 3.13; wheel
availability is checked for every package except source-only, pure-Python
promise 2.3. Python 3.11/3.12 retain the original 190-package lock.
This corrects bootstrap compatibility; live imports and TPU execution remain
acceptance checks. Restart the Kaggle session after updating CODE_REVISION.

## Implementation tasks

Software deliverables for TPU-01 through TPU-07 are implemented and locally
checked: explicit profiles, a separate resolved Linux dependency lock, lazy
sharded inference, backend-aware execution, pinned notebook handoff, measured
preflight/restore hooks and label-focused paired review resources.

**Pending after the handoff:** real Kaggle imports and eight-device kernels,
27B memory fit and throughput, interruption and saved-output restore across
provider sessions, Persian human review and the translator adoption decision.
Local fixtures and package resolution do not satisfy those acceptance checks.
Each logical change has a separate commit and verification record.

Local handoff verification (Oct 6): `ruff check .` and `ruff format --check .`
pass, and the full suite passes 668 tests. Notebook cells compile with empty
outputs; defaults and summaries match the active request. Linux dependency
metadata resolves for Python 3.11 and 3.12 with glibc 2.31+. The Kaggle TPU
dry-run selects only the approved preflight/gate recipes and 27B artifact roots.
No weights or datasets were committed, and no live provider result is claimed.

| ID | Deliverable | Dependencies | Acceptance |
| --- | --- | --- | --- |
| TPU-00 | This branch plan and link from the workflow plan | None | Documentation committed; implementation and live validation remain pending |
| TPU-01 | Explicit TPU backend/profile schema and readiness dispatch | TPU-00 | CPU/CUDA behavior retained; invalid backend/device/dtype combinations rejected; CPU fixtures need no JAX |
| TPU-02 | Kaggle TPU bootstrap, `[tpu]` extra and proposed `constraints/tpu.txt` | TPU-01 | Fresh TPU session reports compatible pinned libraries and eight v5e devices; no CUDA extra installation or unintended runtime replacement |
| TPU-03 | Sharded Gemma 3 27B JAX generator and translator registration | TPU-02 | Real checkpoint loads without full replication; greedy generation returns complete Persian replies in input order; prompt/cleaning parity checked |
| TPU-04 | Backend-aware recipe runner, experiment roots and provenance | TPU-01, TPU-03 | TPU request passes schema checks; readiness failures start no translation; 4B outputs and incompatible resume bundles cannot mix |
| TPU-05 | Thin notebook handoff and operator/reference documentation | TPU-04 | Explicit TPU settings, pinned revision and current-stage summaries agree; empty cell outputs; Run all invokes package logic only |
| TPU-06 | Live memory, timing, failure and restore validation | TPU-05 | Fresh-session preflight and interrupted/resumed translation work; completed IDs are not duplicated; durable private output restores with checksums |
| TPU-07 | Paired Persian label/meaning evaluation and translator decision | TPU-06 | Same English inputs compared; label errors reported separately; human review and measured resource costs support a recorded decision |

### TPU-01 and TPU-02: configuration and bootstrap

Add proposed `kaggle-tpu-preflight.toml` and `kaggle-tpu.toml` profiles. Document
whether `device="tpu"` identifies the accelerator while a separate backend field
selects JAX; avoid treating CUDA-style `xla` tensor device names as a JAX API.
The request should declare model family, backend, precision and limits through a
validated configuration registry, replacing the current 4B-only equality checks.

Select dependency extras based on the explicit backend before accelerator imports.
Probe the Kaggle Python version, available devices, effective memory and runtime
libraries. Establish a tested version set on the real provider rather than
inventing pins locally. Generic CPU bootstrap must still avoid provider imports.
Do not assume CUDA `HF_TOKEN` access automatically authenticates the JAX checkpoint
source: validate the selected download source and its credential mechanism.

Check disk and host-memory capacity before loading. Account for checkpoint
downloads, conversion copies if needed, compiler buffers, output exports and the
working/cache filesystem's actual limits. Cache model weights outside saved
artifacts, and never copy them to Drive or an output bundle. The 15 GB Drive
budget cannot hold a BF16 27B checkpoint.

### TPU-03 and TPU-04: generation and reproducibility

Record the exact model checkpoint source/revision, all shard/config/tokenizer
identifiers or checksums, JAX/Gemma/Kauldron versions, prompt/glossary checksum,
decoding settings, input/output limits, mesh/sharding policy and actual dtypes.
If checkpoint conversion is necessary, record the source and converted identities
and validate tokenizer/config/weight correspondence. A JAX checkpoint is not
automatically the identical artifact identified by a Hugging Face commit.

Record compilation time separately from warmed generation throughput, total wall
time, load time and observed per-device memory. Include failures and truncated or
missing outputs in structured execution results. Greedy decoding is reproducible
within the tested configuration; do not promise identical floating-point results
across frameworks or accelerators.

Namespace runs by experiment identity as well as stage/mode. Include backend,
checkpoint identity, dtypes, prompt, input selection and generation settings in the
resume contract. Reuse compatible pinned English inputs, never prior translations
from another model. Initialize one model per translation process and avoid repeated
loading/compilation for each record. Consider reuse across source recipes only
after the basic subprocess contract is validated.

### TPU-05 and TPU-06: handoff, tests and persistence

When activating the TPU experiment, update `workflows/current.toml`, both notebook
summaries, `docs/execution-workflow.md`, `agent.md`, and the research plan's model
and environment assumptions in the same logical change. Bump the research plan
version then; retain the 4B decision and historical pilot as history. The active request and both notebook summaries now select the bounded 27B
experiment. `workflows/gpu-gate.toml` retains the historical 4B request.

Local checks use fixtures/fakes to cover explicit profile selection, missing TPU
dependencies/devices, wrong device count, prompt conversion, reply count/order,
failure propagation and incompatible resume identities. Retain existing CLI,
schema, keep-field, split and private-bundle tests. Run repository fast checks;
local tests do not establish TPU kernels, real memory fit or translation quality.

Live preflight first loads the checkpoint and translates a few short and long
representative examples. Set measured session/time budgets before expanding to
the 40-helmo/20-typed gate. Confirm actual eight-chip sharding and record resource
use. Interrupt after completed records, save a private bundle, attach it in a new
session, and verify checksums and exactly-once completed case IDs after resume.
If a partial case cannot resume, document the bounded rework explicitly.

Keep existing per-case persistence and finally-path diagnostics. Exports exclude
weights, compilation caches and credentials. Saved notebook source alone is not
proof of durable outputs. Report the last verified durable bundle and progress
since it; use the same owner-controlled storage contract as the current runner.

### TPU-07: assess the result before adopting the model

Compare identical source revisions and selected source IDs with the existing 4B
translations. Separate this paired diagnostic set, including known failures, from
a fresh representative sample used to estimate error prevalence. AI review drafts
are supporting annotations; they do not satisfy the plan's human review requirement.

Review translated **option/label descriptions** separately from states and
questions, and also judge the complete case for preserved decision meaning.
Prioritize negation, severity ordering, actor/action roles, conditions, technical
terms and distinctions between options. Keep option IDs and gold byte-identical;
a bad option description does not itself mean the stored gold was modified.

Report both unique recurring labels/templates and case-level results, counting
repeated labels explicitly. Report sample sizes, paired fixes and regressions,
meaning changes, edit rates and coverage by workflow/question type. Use a random
sample for prevalence estimates and uncertainty intervals; a selected failure set
cannot establish detector coverage or whole-dataset error rates. Test-time examples
must not be used to tune prompts/glossaries or select the translator.

Retain the existing gate criteria (at most 1 meaning change in 40 helmo records;
at most 10-15% needing edits, with the workflow criterion specified before review).
End at `awaiting-review`; do not launch full translation or publication automatically.
Choose or reject 27B based on human-reviewed fidelity plus measured runtime and
review effort, not model size or public benchmark scores alone.

## Completion and fallback

TPU support is complete only after local checks, live sharded generation, private
fresh-session restore and an accurate operator handoff pass. Adopting 27B is a
separate research decision following TPU-07. If it does not fit or fails the
session budget, preserve diagnostics and keep the existing CUDA route available;
do not silently lower precision, shorten source text, change datasets or substitute
another model. Any alternative becomes an explicit new experiment.

## Version history

| Version | Date | Change |
| --- | --- | --- |
| 3 | Oct 7, 2026 | Add interpreter-specific Python 3.13 constraints after the observed Kaggle bootstrap failure; retain all model/backend pins and the older kernel lock. Live TPU validation remains pending. |
| 2 | Oct 6, 2026 | Implement the software path and Kaggle handoff, including a 190-package Linux lock, sharded native generation, model-specific resume identities, short/long measured preflight and paired option review. Live TPU validation and human adoption remain pending. |
| 1 | Oct 6, 2026 | Plan Gemma 3 27B JAX inference on Kaggle v5e-8, with explicit sharding, dependencies, provenance, restore validation and label-focused evaluation. |
