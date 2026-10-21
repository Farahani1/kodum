# Kodum: Persian typed-decision models and evaluation

Version 40 · Oct 21, 2026 · @Shah

## Current status and ownership decision

**The dataset is accepted as ready for consumption by kodum.** Its authoritative
home is `D:\Coding\hf-data`, whose dataset project owns ongoing data development,
review/checks, quality decisions, releases and dataset publication. Kodum owns
model development and model-performance evaluation using explicitly selected
versions of that data. This owner decision replaces the earlier requirement
to finish dataset acceptance/publication in kodum before starting model work.

The current dataset baseline is package **1.0.1**, schema **1.0.0**, with this
`provenance/release.json` SHA-256:

```text
89859f966e8af06abe965dbe0cb2a966edd6b9b4488a10ada5d6b0e3831f8419
```

It supplies 6,000 typed-decisions train questions / 1,200 cases, 2,000 test
questions / 400 cases, and the separate 2,000-question Helmo train selection.
That is 10,000 bilingual question rows across 3,600 cases. Original workflow
configurations overlap the aggregate original dataset; do not concatenate them
as independent examples. The other 7,879 original Helmo items are outside the
completed translation scope. Helmo is optional and never a benchmark source.

The package has validated local exports, provenance and the owner-approved
Apache-2.0 translation contribution grant, preserving upstream Apache-2.0/MIT
terms. Its actual quality status remains as declared: machine translations,
a 100-case AI audit, and the Oct 15 human-review waiver. Acceptance for kodum
does not assert that unperformed semantic checks passed. Remaining dataset
checks/flag decisions and future improvements belong to hf-data; they are not
the active kodum backlog or a prerequisite to beginning model implementation.

The 4,200-item skills candidate and legacy dataset tooling currently remain
in this checkout. The data project owns their future migration and release.
They are not silently added to package 1.0.1. Native-source candidates in 2.1
are delivery requests to hf-data; availability must be checked per release.

**Owner decision, Oct 19: prioritize a finished portfolio release.** Train one
Laya-multilingual adaptation on the accepted data, compare it with original Laya
and DibaOne X1, and deliver a reproducible report and small working demo. The
former M2 A/B/C research gate is retired, not passed. Broad native-benchmark
coverage, multiple architectures and causal ablations are deferred. The full-data
model is stage 1; a partial-context model is the planned second training stage,
not a prerequisite to the first fine-tune or its portfolio release.

**Next work, in order:**

1. KD-01 is complete: the baseline selection pins release 1.0.1/schema 1.0.0
   and its manifest hash. The verified selection contains 8,000 bilingual
   typed-decisions rows (16,000 language-specific Records). Helmo is opt-in.
2. KD-02 complete: deterministic preparation preserves the 400 released test
   cases and divides the 1,200 training cases into 960 training, 120 development
   and 120 calibration cases (9,600 / 1,200 / 1,200 language-specific Records).
   IDs, translations and duplicate groups remain together. Verified local
   fingerprint: `fb7b3ef929756ce9a30a2bdd2d23414efe64a161088a662ecf13ede997c02b5e`.
3. KD-03 complete Oct 20: the tiny Laya-compatible CPU path passed four
   optimizer updates across an initial run and a resumed run, including
   validation, checkpoint save/restore, and split/token-truncation manifests.
   This does not validate real Laya weights or a GPU.
4. KD-05 GPU preflight passed Oct 20: the owner-supplied saved notebook
   reports `preflight_complete`, two optimizer updates, a best checkpoint and
   private durable revision `de6c6cc0fd75f51b404c93aebfddad632e74a177` in
   `Farahani/kodoom`. Code pin: `0fa1a580758da9e020075b0537ace4166a03b328`.
   Next verify restore in a fresh saved session with that exact code/revision;
   KD-05 remains open until this succeeds. Final test was not evaluated.
5. KD-04/KD-06: evaluate original Laya and the fine-tuned model on identical
   test examples, plus DibaOne X1 on supported choice questions. Baseline
   adapters can be completed alongside training. Freeze the evaluation protocol
   before final test scoring; use development results for recipe selection.
6. Deliver one checkpoint, inference/calibration code, saved predictions, a
   concise model card/report and a small demo. Report measured improvements
   only; a competitor win or broad-generalization result is not a release gate.
7. Next train the partial-context model from the same original Laya revision,
   independently of the full-data checkpoint, using the protocol in 3.4.

Training and real baseline measurements remain open. This plan update does not
claim execution, change the pinned dataset, or repoint the legacy translation
notebook. Missing native resources do not block this release. Dataset work
remains governed by `D:\Coding\hf-data\docs\project-plan.md`; the earlier
combined data plan remains in [the v30 archive](project-plan-v30-archive.md).

## Project scope and principles

Kodum builds a model-agnostic decision evaluation harness and CPU-friendly
Persian reference models for choice, score and yes/no (noul) questions answered
with calibrated probabilities. Dataset creation and dataset release operations
are a separate project. Public model reports may reference private or public
versioned data under its declared terms; model publication has its own decision.

| hf-data owns | Kodum owns |
| --- | --- |
| Canonical source conversion, translation, augmentation and skill generation | Release loaders, model request formatting and training-time augmentation protocols |
| Dataset semantics checks, review, flags, corrections and source-label decisions | Model errors, predictive metrics, calibration and model robustness experiments |
| Source/benchmark split policy and immutable release content | Leakage-safe model subset selection, training mixtures and train-derived calibration |
| Dataset licenses/provenance/eligibility declarations and publication | Reading/enforcing declared use, attribution, model runs and model publication |
| Argilla and dataset release compatibility tests | Consumer schema/hash checks and pinned run reproducibility |

Input-contract checks remain required here. Check a selected release's identity,
checksums, supported schema and required fields, preserve IDs/options/gold, and
enforce source roles/test isolation. This is consumption safety, not a new data
quality campaign. Report suspected dataset defects to hf-data with release and
record IDs; never modify its release or silently substitute corrected labels.

The same `kodoom.normalize.normalize` runs for training inputs and inference.
Canonical benchmark text stays raw. Record every model-specific transformation,
option remapping, exclusion and subset in a run manifest without changing data
release files. Evaluate native Persian, human-localized, translated and
procedural data separately. Shared backbones do not alone establish causal
claims about model recipes or training data.

Existing data-development commands/modules remain available for historical
reproduction during migration. Do not delete them or move production data as
part of this plan update, or start a new dataset campaign in kodum by default.
The Python package and command remain named `kodoom`; the project is `kodum`.

## Environments and model execution

Choose provider and profile explicitly. `dev` is CPU-only on the modest laptop
(i3, four threads, about 12 GB RAM; the MX110 is unused). It must not download
real model weights or need CUDA. Use tiny random model configs with the real
tokenizers to exercise shapes, save/load and resume. The intended end-to-end
dev smoke path is under five minutes: consume a small frozen fixture/release
slice, build a model mix, train, calibrate, predict and report. Dataset
translation/review is no longer part of this model smoke path.

**Planned training runtime: Kaggle GPU T4 x2.** Select that accelerator
explicitly and verify two visible T4 devices and their actual free memory in
preflight. This replaces Colab as the default model-training target. Kaggle
model training and cross-session restore remain unvalidated until measured;
historical TPU translation recipes are not this model run.

Use PyTorch DistributedDataParallel (DDP), one process/model replica per GPU,
launched by a package training entrypoint through `torchrun --standalone
--nproc_per_node=2`. The notebook invokes the launcher; it does not implement
the training loop. Two GPUs provide separate memory budgets, not one pooled
32 GB allocation: model, optimizer and one microbatch must fit on each device.
Use fp16 autocast with GradScaler, supported SDPA/eager attention, and no
BF16/FlashAttention-2 assumption. Keep softmax/log-loss computations in float32.
Use the installed compatible provider PyTorch and validated lazy GPU extras.

Start with microbatch 2 per GPU and 8 accumulation steps: effective batch
2 GPUs x 2 examples x 8 = 32 language-specific questions per optimizer update.
If memory requires microbatch 1, use 16 accumulation steps to retain that batch.
Measure speed; do not promise a 2x speedup or double learning rate for two GPUs.
DDP sampling must shard a deterministic language-balanced schedule, use equal
step counts on both ranks, and advance its epoch seed. Avoid duplicate padded
validation rows or deduplicate by ID before scoring. Freeze unused heads or
configure unused-parameter handling so DDP does not hang on inactive outputs.

Scale accumulated losses consistently; synchronize gradients on update steps.
Unscale before gradient clipping. Start AMP loss scaling at 1024 and allow at
most 12 backoff retries of the same effective batch when gradients overflow.
Synchronize the retry decision across ranks and replay each rank's RNG; failed
attempts update neither weights nor scheduler/sampler progress. Persistent
nonfinite gradients or nonfinite forward losses remain fatal. Count only
successful finite optimizer updates toward the preflight's two-update target. Aggregate validation statistics across ranks
and broadcast checkpoint/early-stop decisions. Rank 0 writes shared logs and
checkpoints; preserve per-rank RNG and sampler state for resume. Checkpoint
model, optimizer, scheduler, scaler, global step and run/split identities.

Short preflight: verify both GPUs participate, finite loss/gradients, memory,
one optimizer update, evaluation, and save/restore with both ranks. This is a
bounded implementation check, not a research gate. A single-GPU fallback must
be explicitly recorded with effective batch preserved, never silently used.
See [PyTorch DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
and [AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

Code remains portable to Windows/Linux; laptop development remains CPU-only.
Start with num_workers=0 per rank and limit CPU threads to avoid oversubscription.
Use `kodoom.runs.Run`, `save_latest` and `save_best` with rank-aware writers.
Keep model inputs/checkpoint staging under `/kaggle/working`, source attachments
read-only under `/kaggle/input`, and base-model caches outside saved outputs.
Retain latest full resume state plus best inference checkpoint within measured
storage capacity. Use a private Hugging Face repository as the planned durable
backup, with Kaggle as the local write/cache layer. Record its exact repo ID
and verify private visibility before syncing; use Kaggle Secrets for the token.
Verify restore in a fresh session; local Kaggle disk alone is not durable. Record the last
verified durable step and potential lost work. Never commit datasets or weights,
and never upload private artifacts publicly as part of automatic training.

The existing execution notebook and `workflows/current.toml` still describe
the frozen translation campaign. They are not a model-development launch path.
Prepare and validate a separate model execution request/notebook handoff before
running it. This documentation change does not silently repoint notebooks,
advance `WF-*` tasks, or claim a training preflight passed. See
[agent.md](../agent.md) and [workflow-change-plan.md](workflow-change-plan.md).

## Part 1 — Consume versioned dataset releases

### 1.0 Accepted external input

The former data-augmentation phase is closed for kodum by owner acceptance and
transfer of responsibility. Its remaining review, semantic checking, Argilla,
source cleanup, skill/native-source preparation and dataset-publication tasks
belong to hf-data. This is an ownership/milestone change, not a claim that the
old dataset-publication gate passed. Public Hub/Viewer verification is separate.

Package 1.0.1 is the available baseline, not a floating directory called latest.
The dataset root is currently a mutable development location. Consume only files
named in the pinned manifest or a verified immutable release snapshot. Snapshot
layout adoption is owned by hf-data; do not treat newly added root documentation
as an in-place modification of an old release's frozen payload.

### 1.1 Consumer contract and experiment pins

Every run records a dataset lock with dataset identity, exact release version,
schema/transport version, source root or repository plus immutable revision,
manifest SHA-256, selected configurations/languages/splits and payload hashes.
Record code/model/tokenizer revisions, seeds and model-derived selection/mix
hashes alongside it. Local paths are deployment settings, not release identity.

The currently published input layout is bilingual Parquet with nested options,
quality and translation metadata, not identical to kodum Record transport.
The adapter must map that supported layout explicitly, preserve source/question
identity and soft gold, and keep machine/human quality statuses distinct. Handle
optional fields according to the schema contract. Unknown major versions or
unsupported transport identifiers fail with a migration message.

The current bilingual JSON schema is closed (`additionalProperties: false`),
so optional new fields are not automatically compatible with old validators.
Accept extensions only through a supported contract/view; compatibility tests
must cover older consumers rather than trusting a version label alone.

hf-data's releases use MAJOR.MINOR.PATCH and record schema_version separately.
An adapter can declare a compatible schema range, but a run always pins one
exact release and its content. No automatic upgrade is allowed during a run.
Same-major compatibility means the documented loader/semantic contract is
preserved; it does not promise unchanged samples, benchmark scores or model
behavior. Any upgrade creates a new lock/experiment identity and a common-item
or matched-release comparison where appropriate. Benchmark fingerprints belong
in results as well as release numbers.

Dataset component licenses, permitted-use and eligibility declarations are
provided by hf-data and read/enforced by kodum. Do not reopen source/teacher/
translator investigations here, fabricate missing declarations or reinterpret
campaign names as actual translator identities. Report any required missing
contract information to the dataset owner. Test-only resources cannot enter
training; Khayyam/PersianMMLU is excluded. Model release compatibility and model
attribution remain kodum responsibilities.

### 1.2 Model-development tasks

| ID | Work | Acceptance |
| --- | --- | --- |
| KD-01 | Dataset lock and release loader — complete Oct 17 | Exact baseline selection, schema and payload hashes; raw bilingual Records and source roles preserved. |
| KD-02 | Mixture and split manifests — complete Oct 19 | Use accepted training resources; group whole cases and language variants; preserve source test membership; derive development/calibration from train; validate targets and option mappings. |
| KD-03 | Laya CPU smoke — complete Oct 20 | Tiny compatible model with pinned tokenizer completed four updates across save/resume; finite losses/gradients, validation and manifests recorded. Real checkpoint/GPU remain untested. |
| KD-04 | Two baseline adapters and evaluation | Original Laya on all supported types, Diba X1 on choice; identical examples/options, raw and equally recalibrated probabilities, coverage and saved predictions. No broad baseline matrix or A/B/C decision. |
| KD-05 | GPU preflight passed Oct 20; fresh-session restore pending | Kaggle T4 x2, DDP and balanced replay, real checkpoint/dtype, one train/eval step, memory and distributed checkpoint restore across fresh sessions. No research performance threshold before training. |
| KD-06 | Stage 1 full-data fine-tune and portfolio release — training complete Oct 21; evaluation/release pending | The owner's Kaggle run `laya-full-v1` completed 600 updates; best Persian development loss 0.953416 and durable snapshot `f315b5da129e2b1df55d971ddf2053c3cbfe5282`. Final test evaluation, inference/calibration artifacts, model card and small demo remain required. Stage 2 does not block this release. |
| KD-07 | Stage 2 independent partial-context fine-tune | Original Laya initialization; frozen context exclusions across languages and training/development/calibration; same evaluation cases for original/full/partial models, with exposure disclosed. |

## Part 2 — Implementation

Part 2 first fine-tunes a full-data Laya-multilingual checkpoint on a pinned hf-data selection and records enough predictions and configuration to reproduce its evaluation. Canonical data development and review remain owned by hf-data.

### 2.1 Data preparation (training mix)

**Native Persian source candidates: external dataset delivery requests**

The preparation column below belongs to hf-data. It is a record of resources
requested for future model experiments, not a kodum conversion/review backlog.
Only the resources actually present in a pinned release are available now.
Missing configurations are reported and requested from hf-data; do not invent
results or silently manufacture replacement benchmarks in kodum.

| Dataset | Converts to | Size | License | Role | Why | Dataset-project delivery request |
| --- | --- | --- | --- | --- | --- | --- |
| [MASSIVE](https://github.com/alexa/massive) (fa-IR) | choice: 60 intents or 18 scenarios | \~11.5k train, 2k dev, 3k test | CC BY 4.0 | Train and test | Voice-assistant commands, close to the STT→intent project. Many options per question teach the model to read option lists. | Write a short Persian description for each intent and scenario (labels are English codes such as `alarm_set`). Sample option subsets. |
| [FarsTail](https://github.com/dml-qom/FarsTail) | choice (3-way entailment) or noul | 10,367 | Apache-2.0 | Train and test | Human-made reasoning over two texts, cleanly licensed. | Download from GitHub, map labels, write templates. |
| [PersianQA](https://github.com/sajjjadayobi/PersianQA) | noul: does the passage answer the question? | 9,000+ | GPL-3.0 (the repository's LICENSE; DibaOne M3's card cites CC BY-NC-SA 4.0 from a GitHub release) | **Test only** | Its unanswerable questions give natural "no" cases. Either license would put a permissive model license in question, so it stays out of training and becomes one more held-out task family. | Pair questions with wrong passages for extra negatives. Truncate passages without cutting the answer. Publish the converter, not converted data. |
| [ParsiNLU](https://huggingface.co/persiannlp) (entailment, paraphrase, sentiment, multiple-choice QA) | all three types | 1.3k–17.5k per task | CC BY-NC-SA 4.0 | **Test only** | Tasks the model never trains on. Keeping them out of training also keeps the model's license clean. | Convert to records; never mix into training. |
| [Belebele](https://huggingface.co/datasets/facebook/belebele) (Persian) | choice (4-way reading comprehension) | 900 | CC BY-SA 4.0 | **Test only** | Parallel across 115 languages, so results compare with published ones. | Convert; check passage lengths. |
| Own STT→intent transcripts | choice | yours | yours | **Test only** | Noisy speech-recognition text, the closest thing to real use. | Remove anything private; label with intent options. |
| [Khayyam / PersianMMLU](https://github.com/raia-center/khayyam-challenge) | — | — | CC BY-ND, academic only | **Excluded** | Its license forbids derivative benchmarks. | — |

**Consumer roles.** MASSIVE, FarsTail, workflow translations and skills are
candidate training resources when their release declares the appropriate use.
ParsiNLU, PersianQA, Belebele and own STT data remain test-only; Khayyam /
PersianMMLU remains excluded. Kodum reads and enforces the dataset project's
component licenses, attribution, provenance and declared eligibility. Source,
teacher or translator questions are resolved and recorded by hf-data; kodum
does not repeat dataset legal/quality review. A missing required declaration is
an input-contract issue reported to the data project, not a reopened dataset
acceptance campaign. Never infer Gemma authorship from a campaign name.

**Accepted dataset input.** Use the chosen release of `D:\Coding\hf-data`.
Workflow translations are accepted for consumption by the owner. Helmo is an
optional train-only component, selected explicitly by the experiment; dataset
source-label audits and hold decisions belong to hf-data. Canonical quality
flags remain available as experimental metadata. Any model-specific subset
selection is an immutable run manifest, not a canonical dataset edit. Skills
are consumed when an explicitly versioned skills resource is delivered; the
old local candidate is not silently treated as part of package 1.0.1.

**First-release mix:** use the selected Persian and English typed-decisions training rows across all four workflows. "Full data" means all selected training contexts, not the released test rows or train-derived development/calibration cases. Helmo remains optional train-only augmentation; new native sources and mixed-language request augmentation are deferred.

**Model preparation steps**

1. Load an exact release/configuration through a tested consumer adapter;
   preserve stable IDs, ordered options, soft gold, source roles and provenance.
2. Render model requests and sample option subsets under a frozen protocol.
   Shuffle/rename option IDs only with explicit reversible target remapping.
   New canonical task templates, source-label decisions and converters belong
   to hf-data. Do not add synthetic "none" labels without an explicit protocol
   that establishes the resulting targets.
3. Apply the same full `kodoom.normalize.normalize` at training and inference.
   Write transformed inputs to run artifacts; keep the dataset release raw.
4. Measure lengths with the pinned Laya tokenizer and request formatter.
   Start from checkpoint-compatible limits; record any T4-driven truncation,
   head/context limits and unsupported inputs consistently across comparisons.
5. Use the available selected training rows without waiting for more sources
   or a source-balancing quota. Record language/source counts and any weights.
6. Select early-stopping validation and temperature-only calibration from
   released train data by source_id. Respect any existing released validation
   or calibration roles. All questions, language variants and minimal-pair
   halves stay grouped. Source test membership is never reassigned.
7. Enforce model-side leakage protection across selected splits, including
   source cases, pairs and normalized/near-duplicate inputs. Report suspect
   source defects to hf-data; do not rewrite canonical splits or targets.
8. Freeze mixture/selection manifests and hashes before training. No larger
   dataset target is required for the first portfolio release.

**Request format**

Retain Laya's checkpoint-compatible all-options-per-question input, typed
question representation, option markers and scoring head. Preserve gold
probabilities when mapping options. Do not replace it with the former
per-option mmBERT architecture. Use the same formatter for original and
fine-tuned Laya; record the adapter and option budget for Diba X1 separately.

### 2.2 Training

**One main model:** initialize from
[`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual),
pinning model, tokenizer and implementation revisions. Fine-tune its encoder
and decision head (full fine-tuning, not head-only) with soft-target cross-entropy on the accepted bilingual
training mix. Retaining the architecture does not require retaining the
upstream policy-gradient objective; no reward model is required here.

| Run | Scope | Purpose |
| --- | --- | --- |
| Smoke | Tiny compatible config, then short real-checkpoint preflight | Catch input, target, loss, memory and checkpoint failures cheaply |
| Stage 1: full-data | Original Laya-multilingual, all selected training workflows | First trained model and portfolio deliverable |
| Stage 2: partial-context | Same original Laya revision, selected contexts excluded | Measure transfer to excluded contexts after stage 1 |
| Deferred | English-only or translation ablations; fresh mmBERT; decoder challenger; additional seeds | Optional follow-up after the portfolio release |

**Evaluation baselines, not training gates:**

- Original pinned Laya-multilingual: primary before/after comparison for choice,
  noul and score. This measures the adaptation's effect on the selected workload.
- [DibaOne X1](https://huggingface.co/Dibachain/DibaOne-X1): additional external
  competitor on supported single-choice questions. Pin its revision and mode
  before test scoring. Do not count unsupported noul/score calls as errors.
- Uniform and train-derived prior baselines are inexpensive sanity checks.
- DibaOne M3, other Laya variants, Decima, prompted Qwen and API models are
  optional. M3 trained on ParsiNLU/PersianQA; label that exposure whenever these
  resources are evaluated, rather than treating them as unseen tasks for M3.

Compare raw probabilities and give each evaluated model the same temperature
fitting opportunity on the train-derived calibration subset. No baseline test
score is required before training. A failed external adapter is disclosed as
missing evidence and never replaced with a published score on another dataset.

Both stages initialize independently from the identical original Laya-multilingual
encoder and head. Do not initialize the full-data model from a partial-data
checkpoint, or the partial-context model from the full-data checkpoint: the
latter would carry prior fine-tuning exposure to its nominally held-out contexts.
Resume only a checkpoint from the same stage/run with matching split manifests.

**Initial training recipe (starting defaults, not tuned results)**

- AdamW full fine-tune: encoder LR 5e-6, existing decision-head LR 2e-5,
  weight decay 0.01 (exclude bias/norm parameters), 10% linear warmup followed
  by linear decay, max gradient norm 1.0. Keep pretrained weights/head intact
  at initialization. Use soft-target cross-entropy; no reward-model stage.
- Start with 2 epochs over the bilingual training schedule; extend to at most
  4 only if development results justify it. Effective batch 32 on Kaggle T4 x2
  as above. Keep the global batch and learning rate fixed when changing ranks.
- Freeze a seed and record all settings/IDs. One seed is sufficient for this
  release, but it does not establish stability across seeds.
- Evaluate every 50 optimizer updates and at each epoch end; use the English
  retention policy below when selecting best weights. Stop after 3 evaluations
  without an improvement in eligible Persian development log loss.
- Fit temperature only after checkpoint selection, on the separate calibration
  subset. All retention monitoring uses raw probabilities; calibration cannot
  conceal loss of answer-selection performance.
- Target a 1–2 hour training session as a planning budget, not a measured runtime.
  Preserve resumability and adjust memory settings using preflight/development
  evidence, never final test performance.

**English replay and forgetting prevention — required in the main run**

1. Split cases before creating language examples. English replay comes only
   from accepted training cases; their Persian counterparts share the same
   partition. Never replay development, calibration or test examples.
2. Use an explicit 50% Persian / 50% English schedule throughout training,
   including the last epoch. Every effective optimizer batch contains 16
   Persian and 16 English questions. Shard the balanced schedule across ranks;
   with microbatch 2, aim for one example of each language per rank per
   microstep. With microbatch 1, alternate languages through accumulation.
   Do not run long Persian-only blocks followed by occasional English epochs.
3. Shuffle within language each epoch and cover all selected workflows and
   question types in both languages. Keep source proportions comparable across
   languages. If Helmo is selected, include its eligible English counterparts
   within the same replay ratio. Log actual language/type/workflow counts per
   update and epoch so a sampler bug cannot silently remove English replay.
4. Average losses within each language and combine as
   `L = 0.5 * mean(CE_fa) + 0.5 * mean(CE_en)` over the effective batch.
   Preserve soft targets; do not sharpen labels or add entropy maximization
   as a supposed retention fix. Use correct weighting when batches are uneven.
5. Before the first update, evaluate original Laya on the fixed Persian and
   English development subsets. This small retention reference is not the
   retired broad baseline gate. Track language-specific raw log loss, top-1
   accuracy, and question-type breakdowns at every validation interval.
6. Initial checkpoint guard: English overall development accuracy may decline
   at most 2 percentage points, and English mean soft-target log loss may rise
   at most 5% relative to the starting checkpoint (numerical floor 1e-8).
   Among checkpoints satisfying both, choose lowest Persian development log
   loss. Freeze these operational tolerances before training; they are not
   statistical guarantees or universal definitions of forgetting. Inspect
   per-type regressions even when the overall guard passes.
7. If both consecutive validation checks fail the guard, stop and retain the
   last eligible checkpoint. If no trained checkpoint qualifies, report that
   explicitly. For a bounded retry, restart from an eligible/original checkpoint,
   halve learning rates and increase English replay to 75% (24 English / 8
   Persian per effective batch, loss weights 0.75/0.25). Record a new run and
   select using development only; do not launch an open-ended ablation grid.
8. Report final English as well as Persian test performance versus original
   Laya, overall and by type/workflow. English replay reduces forgetting risk;
   it cannot guarantee retention of other languages, unrelated tasks or the
   base model's entire capability set. Claims are limited to measured tasks.

Distillation from the frozen original model, partial encoder freezing or LoRA
are optional remedies if the bounded replay/LR adjustment is insufficient.
They are not additional mandatory runs or a requirement to acquire new data.

A workflow may be selected for specialization using training/development
evidence. Freeze its scope before final testing and report all tested workflow
results. Do not select winning test examples or claim generalization to unseen
contexts from a within-workflow split. The initial default remains all four
typed-decisions workflows.

### 2.3 Collecting logs and training data

Keep enough that any number in the final report can be recomputed without retraining.

- **Run registry:** one row per run with run ID, date, model and revision, dataset version, key settings, headline metrics, and a one-line decision (keep, or drop and why).
- **Per-run config:** model revision hash, dataset version hash, seeds, hyperparameters, max length, option-sampling settings, library versions, GPU type.
- **Curves:** train loss, validation loss, validation accuracy and ECE per step or epoch; runtime and peak GPU memory.
- **Where:** write locally in the Kaggle run directory (CSV/JSON or local TensorBoard); periodically sync to the configured private Hugging Face repository. Rank 0 owns shared writes; public publication is a separate action.
- **Logging cadence:** log every optimizer update: step, epoch, per-language loss and counts, learning rates, gradient norm, scaler/skipped steps, throughput and memory. Save per-language/type validation results at each evaluation. Sync compact logs/config/manifests about every 5 minutes.
- **Snapshot cadence:** save a resumable latest checkpoint every 50 optimizer updates or 15 minutes, whichever comes first, plus epoch end and graceful shutdown; save best weights when the retention-aware validation criterion improves. Sync completed latest/best snapshots to private Hugging Face after saving. Use complete manifests/checksums, atomic local writes and a remote commit before marking a snapshot durable; never upload partially written files.
- **Storage and restart:** retain latest full resume state and best inference weights locally; avoid duplicate base caches in uploads. Track remote revision, durable step and sync failures. Ordinary Hub commits can retain old blobs despite overwriting a path: monitor remote storage/history rather than assuming only two snapshots consume space. Resume from a verified complete private snapshot; failed uploads leave the last durable copy intact. No exact private repo name is selected by this plan update.
- **Predictions, not just metrics:** for every evaluation, save per item the ID, gold distribution, predicted distribution, temperature used and latency. Every later metric, chart and error analysis comes from these files, and they can be published as raw results.
- **Error log:** inspect a small development-error sample for input and modeling issues. Final test error analysis is descriptive; do not tune on it and reuse that test as untouched evidence.
- **Artifacts kept:** best checkpoint, tokenizer, calibration, exact mix/split manifests and model card. Prepare a reviewable release; external publication follows the owner's publication decision. Do not delete the only checkpoint copy.

### 2.4 Direct-to-training scope decision

The owner retired the former A/B/C research gate on Oct 19 to reduce completion
risk. Do not wait for language-gap thresholds, native benchmarks, Persian skills
scores or evidence that a new general-purpose model is needed. Proceed after
the existing input-contract checks, grouped splits and short training smoke.
These implementation checks are not a replacement research campaign.

The deliverable is a Persian workflow specialization, not a proof of universal
Persian reasoning superiority. Whole-context holdouts belong to the planned
second partial-context training stage and do not block stage 1. No performance win is guaranteed or required
to document and release the completed engineering work.

## Part 3 — Testing and evaluation

### 3.1 First-release test set

Use the released typed-decisions test split: 400 cases / 2,000 bilingual
questions, with Persian as the primary endpoint. These are new cases from
workflows represented in training, not unseen contexts. Preserve the original
test membership and keep all associated questions/translations together.
English results are a required retention check on the same cases; they are not used for checkpoint selection.
Helmo is train-only and never supplies this benchmark.

Before scoring final test predictions, freeze checkpoint-selection rules,
metrics, adapter revisions, Diba mode, input budgets and common-item coverage.
Development data can guide specialization and debugging; final test results
cannot guide model selection while retaining an untouched-test claim.

ParsiNLU, PersianQA and Belebele remain test-only optional later evaluations.
MASSIVE/FarsTail, skills, STT, held-out templates and whole-context evaluation
are follow-ups when released resources exist; missing resources do not block
training or the first portfolio release. Do not make broad transfer claims
without those experiments. Diba's custom benchmark examples are not assumed
available, and public source datasets do not reproduce its custom test sets.

### 3.2 Minimum metrics

- Top-1 accuracy with the gold/tie rule stated, overall and by workflow and
  question type; Diba X1 comparisons use the same supported choice subset.
- Soft-target cross-entropy/log loss and Brier score for distribution quality;
  optional ECE with fixed bins. Report raw and recalibrated probabilities.
- For score questions, report error of the predicted expected ordinal score
  against the gold expected score, with the scale defined.
- Paired 95% confidence intervals for model differences, resampling whole
  cases and keeping language/model predictions paired. If the difference's
  interval includes zero, call it inconclusive, not statistically established.
- Save IDs, gold/predicted distributions, calibration, failures and coverage.
  Report machine-translated Persian as such: source-label agreement is not an
  independent measurement of human judgment or real-world uncertainty.
- Record inference hardware, precision, input lengths and option counts with
  a small latency measurement. No voice-assistant latency threshold blocks release.

### 3.3 Required comparisons and claim boundaries

| Comparison | Same evaluation items | Supported interpretation |
| --- | --- | --- |
| Original Laya vs fine-tuned Laya | Persian test questions across supported types | Adaptation improvement on held-out cases of represented workflows |
| Diba X1 vs fine-tuned Laya | Identical supported choice questions/options | Competitive performance on this target workload, with task-specific training disclosed |
| Raw vs calibrated | Same predictions/test cases; calibration fitted outside test | Probability calibration improvement, separately from answer selection |

Record model training exposure as known, suspected or unknown. Our model is a
typed-decisions specialist; external baselines receive no task fine-tuning here.
This is a practical workload comparison, not an equal-data/equal-budget trial or
proof that our architecture is superior. Sharing a backbone alone does not
control the complete architecture. The before/after comparison does not isolate
Persian data from task adaptation; an English-only control is deferred.

Use the same semantic requests and complete option sets where supported. Keep
truncation, option remapping, unsupported inputs and failures visible. Report
common-item counts rather than silently dropping difficult examples. Preserve
test isolation when inspecting duplicates or suspected source defects.

### 3.4 Stage 2: partial-context model, after full-data training

Train the full-data model first and preserve its checkpoint/report. Then run
one partial-context fine-tune from the same original Laya revision, using the
same full encoder/head adaptation, Kaggle T4 x2 and English replay policy.
Stage 1 can be released before stage 2 finishes.

Freeze one or two context exclusions and their case IDs before stage-2 training.
Prefer metadata and training/development evidence to choose feasible contexts
with adequate independent test cases and manageable training loss. Preferably
record the intended exclusions before stage-1 test scoring; if prior test
results informed selection, label stage 2 exploratory rather than an untouched
confirmatory test. Do not select individual test items for favorable outcomes.

Remove excluded-context cases from stage-2 training, development and calibration
in both languages, including duplicate/related cases and optional augmentation
that reintroduces the context. Select checkpoints using retained-context
development only. Preserve original released test membership; evaluate the
excluded-context test cases without moving released training rows into test.
Typed-decisions workflows have test coverage; Helmo topics alone cannot supply
this experiment because Helmo remains train-only. Record overlapping contexts
and limit transfer claims to the actual exclusion boundary.

Compare original Laya, partial-context and full-data predictions on the same
excluded-context test cases. The full-data model is a context-exposed reference,
not a context-held-out model. Also report retained-context performance. Record
both training-set sizes and update counts; do not call this a causal data-only
comparison if budgets differ. Claims concern the measured excluded contexts,
not general dataset superiority. Other ablations, architectures, extra seeds
and native benchmarks remain optional follow-ups.

### 3.5 Portfolio completion and performance claims

- [ ] One fine-tuned Laya checkpoint with reproducible data/code/model pins,
  grouped splits, calibration and working inference.
- [ ] Original-Laya comparison and Diba X1 choice comparison, with saved
  predictions, counts, workflow/type breakdowns and paired uncertainty.
- [ ] A concise model card/report and a small local demo with sample decisions.
- [ ] Actual quality/provenance, exposure, limitations and attribution disclosed.

Completion is distinct from winning. Publish measured improvements only and
report regressions or inconclusive differences. Do not require superiority to
every baseline, a new architecture, or an unseen-context result for release.

Claim template, only when measured: "Fine-tuned Laya-multilingual for Persian
[workflows], achieving X% accuracy on held-out workflow cases, a Y percentage-
point improvement over the original checkpoint." If supported, add: "On the
choice subset, it exceeded DibaOne X1 by Z percentage points." Include sample
counts, uncertainty and specialist-vs-external-baseline disclosure in the report.
Avoid claims of general Persian superiority or isolated dataset generalization.

### 3.6 Release artifacts

Keep the best checkpoint, tokenizer/request adapter, calibration, split manifests,
run config and per-item predictions sufficient to reproduce the headline table.
Provide a small demo and documented inference command, plus the exact consumed
dataset release and its actual machine-translation/review status. Prepare public
artifacts under applicable terms; publication remains a separate owner action.

ONNX/int8 export, an extensive model-agnostic harness, community leaderboard
integration, reliability/coverage dashboards and a standalone recalibration
tutorial are optional follow-ups. A working standard checkpoint and inference
path suffice for the first release; do not block it on export optimization.

## Failure modes and responsibility

| Failure | Kodum response |
| --- | --- |
| Floating data changes an experiment | Require exact release and manifest/payload hashes; keep an immutable local cache or snapshot. |
| A compatible loader is mistaken for an unchanged benchmark | Record content fingerprints, IDs and exact releases; rerun comparisons or use common-item protocols after upgrades. |
| Test data or its language variant leaks into training/calibration | Group cases/pairs and enforce source roles and immutable subset manifests in the model pipeline. |
| Within-workflow results are mistaken for broad transfer | Label the test as unseen cases from represented workflows; defer unseen-context claims until separately evaluated. |
| Small language gaps or calibration claims are noise | Use paired case/pair bootstrap, common coverage and fixed endpoints; publish raw and recalibrated baselines. |
| Dataset defects appear during model error analysis | Record release/record IDs and report them to hf-data. Use explicit run-specific subset manifests; do not silently correct the source or require a new full review campaign. |
| A baseline was trained on a nominally held-out task | Record known/suspected/unknown model training exposure and keep in-domain scores out of held-out claims. |
| Code works on CPU but fails on the real provider | Run the selected model preflight with real dtype, memory, context limits and checkpoint restore before long execution. |
| A fine-tune adds little or loses to Diba | Report the measured result and development lessons; keep one-model scope rather than reopening the retired research gate. |

## Milestones

| Milestone | Current status / exit condition |
| --- | --- |
| M1: accepted dataset input | Closed by owner acceptance; KD-01 complete. Dataset follow-up remains in hf-data. |
| M2: minimal implementation readiness | Open: grouped split manifests, valid model inputs/targets and short CPU/provider smoke. Former broad baseline A/B/C gate retired; baseline scoring may follow training. |
| M3: stage 1 full-data Laya adaptation | Open: train on selected training cases, select on Persian development loss subject to English retention, save resumable/best checkpoint and fit calibration outside test. Stage 2 is not a prerequisite. |
| M4: portfolio report and release | Open: original Laya and Diba choice comparisons, reproducible artifacts, model card and small demo; publish after owner decision. No mandatory winning threshold. |
| M5: stage 2 partial-context adaptation | Planned after stage 1: independent original-Laya initialization, frozen exclusions and matched-case transfer report. Does not delay the first portfolio release. |

This active plan supersedes historical broad model gates and multi-model run
requirements. The separate WF-* tooling plan and frozen translation handoff are
not advanced by this edit. Dataset work remains in
`D:\Coding\hf-data\docs\project-plan.md`; keep the v30 archive and Git history
for previous decisions. References and earlier version-history entries below
are historical context, not additional first-release requirements.

## References

Licenses marked \* are from memory; confirm them on the page before use.

| Resource | Kind | License | Used in |
| --- | --- | --- | --- |
| [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | English decision dataset | Apache-2.0 | 1.1, 2.1, 3.1 |
| [helmo/synthetic-typed-decisions](https://huggingface.co/datasets/helmo/synthetic-typed-decisions) | English decision dataset, 9,879 records, 207 topics | MIT | 1.1 |
| [typed-decisions-ru](https://huggingface.co/datasets/yyhlm/typed-decisions-ru) | Russian translation, card model | Apache-2.0 | 1.3 |
| [typed-decisions-ja](https://huggingface.co/datasets/GeneLab/typed-decisions-ja) | Japanese translation, card model | Apache-2.0 | 1.3 |
| [MASSIVE](https://github.com/alexa/massive) | Intent dataset, fa-IR locale (to confirm) | CC BY 4.0 (NOTICE.md) | 2.1, 3.1 |
| [FarsTail](https://github.com/dml-qom/FarsTail) | Persian entailment | Apache-2.0 | 2.1, 3.1 |
| [PersianQA](https://github.com/sajjjadayobi/PersianQA) | Persian reading QA | GPL-3.0 (LICENSE; a release is cited as CC BY-NC-SA 4.0) | 2.1, 3.1 (test only) |
| [ParsiNLU](https://huggingface.co/persiannlp) | Persian NLU suite | CC BY-NC-SA 4.0 | 3.1 (test only) |
| [Belebele](https://huggingface.co/datasets/facebook/belebele) | Multilingual reading comprehension, includes `pes_Arab` | CC BY-SA 4.0 | 3.1 (test only) |
| [Khayyam / PersianMMLU](https://github.com/raia-center/khayyam-challenge) | Persian exam questions | CC BY-ND, academic only | Excluded |
| [mmBERT](https://github.com/jhu-clsp/mmBERT) | Multilingual encoder (small, base) | see repo | 2.2 |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Small multilingual decoder | Apache-2.0 | 2.2 |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Fallback decoder | Apache-2.0\* | 2.2 |
| [Persian-ModernBERT-base](https://huggingface.co/myrkur/Persian-ModernBert-base) | Persian encoder | see card | 2.2 |
| [Laya](https://github.com/NandhaKishorM/laya) ([laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual), [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions); a copy at [he-jev/laya](https://github.com/he-jev/laya)) | Open decision models, baselines | Apache-2.0 | 2.2, 2.4, 3.3 |
| [DibaOne X1](https://huggingface.co/Dibachain/DibaOne-X1), [DibaOne M3](https://huggingface.co/Dibachain/DibaOne-M3) | Persian-first open decision models, baselines | X1 Apache-2.0; M3 CC BY-NC-SA 4.0 | 2.2, 2.4 |
| [laya-persian-benchmark](https://github.com/alipyth/laya-persian-benchmark) | 64-case Persian diagnostic for Laya (routing, 8 Persian families) | MIT | Related work |
| [Dohnuts-0.1.0-0.8B](https://huggingface.co/PsiACE/Dohnuts-0.1.0-0.8B) | Decision model on Qwen3.5-0.8B, evidence | see card | 2.2 |
| [Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev) | Decision model on Qwen3-0.6B, evidence | see card | 2.2 |
| [TranslateGemma](https://arxiv.org/pdf/2601.09012) ([vLLM guide](https://docs.vllm.ai/projects/recipes/en/latest/Google/TranslateGemma.html)) | Translation model | Gemma terms | 1.2 |
| [Gemma 3](https://huggingface.co/google/gemma-3-12b-it) | 4B historical adoption baseline, 27B TPU draft candidate; 12B retired | Gemma terms | 1.2 |
| [Decima System One Tasks](https://huggingface.co/datasets/amyrmahdy/decima-system-one-tasks), [jabr-v2-persian](https://huggingface.co/datasets/amyrmahdy/jabr-v2-persian/commit/ebd9d7aab6d0cd7be834258903343e8ccf37f901) | New Persian augmentation / evaluation audit candidates, not adopted | CC BY 4.0 / CC0 per cards | 1.0 |
| [TypeSafe Jev models page](https://docs.typesafe.ai/models) | Jev customization and language support | — | Overview |
| [typed-decision-bench](https://github.com/kyr0/typed-decision-bench) | Community benchmark and format | see repo | 3.6 |
| [open-system-one](https://github.com/zhlei07/open-system-one) | Community benchmark vs Jev | see repo | 3.6 |

## Version history

| Version | Date | Changes |
| --- | --- | --- |
| 40 | Oct 21, 2026 | Record the owner's saved Kaggle full-data Laya training result: 600 updates, best Persian development loss 0.953416 and durable private checkpoint revision. Keep KD-06 open for held-out evaluation and portfolio artifacts; document the intended Hub tag in the training handoff. |
| 39 | Oct 20, 2026 | Record successful two-update Kaggle T4 x2 preflight and private checkpoint commit from the owner's saved notebook. Preserve code pin 0fa1a58 for the next fresh-session restore; KD-05 and main training remain open. |
| 38 | Oct 20, 2026 | Record the second Kaggle attempt reaching both GPUs and stopping on nonfinite gradients before its first optimizer update. Add a conservative initial AMP scale and bounded synchronized batch replay on overflow, requiring actual finite updates and retaining fatal persistent-failure checks. GPU preflight/restore remain pending rerun. |
| 37 | Oct 20, 2026 | Record the first Kaggle preflight failure during dataset loading, before GPU checks or training. Materialize Hub downloads for strict inventory/restore path checks; preserve pins and checksum checks. Real-model preflight and fresh-session restore remain pending rerun. |
| 36 | Oct 20, 2026 | Complete the tiny Laya-compatible CPU smoke, including optimizer resume and tokenization/truncation accounting. Add the Kaggle T4 x2 training handoff; real-model preflight and cross-session private restore are the next operator action. |
| 35 | Oct 19, 2026 | Complete KD-02 on codex/laya-training: immutable case-grouped train/development/calibration/test preparation, duplicate isolation, source roles, target preservation and deterministic balanced English replay. Verify the accepted local release as 960/120/120/400 cases; accelerator training remains pending. |
| 34 | Oct 19, 2026 | Explicitly retain full encoder/head fine-tuning. Train the full-data model first, then an independently initialized partial-context model from original Laya. Specify frequent Kaggle-local logs and resumable snapshots with periodic private Hugging Face sync, durable revisions and restore checks. Stage 2 does not block the first release. |
| 33 | Oct 19, 2026 | Prioritize one portfolio Laya-multilingual fine-tune. Retire the A/B/C research gate and defer partial-context training, fresh encoders, decoder challengers and broad benchmarks. Keep scenario-grouped train/dev/calibration/test isolation, original-Laya and Diba X1 choice comparisons, bounded claims, saved predictions and a small demo. Set Kaggle T4 x2/DDP as the target with per-update English replay, conservative learning rates, retention-aware checkpoint selection and private durable resume. No training or evaluation is marked complete. |
| 32 | Oct 17, 2026 | Complete KD-01 with an exact baseline selection, read-only release consumer and portable dataset lock. Verify the local release's 8,000 typed-decisions rows as 16,000 raw language-specific Records, retaining quality/provenance and source roles. KD-02 mixture/split manifests are next; model measurements and training remain open. |
| 31 | Oct 16, 2026 | Separate kodum model development from hf-data dataset development. Accept hf-data 1.0.1 as the baseline input; replace active Part 1 review/publication gates with versioned consumption and model milestones. Preserve v30 as an archive, move dataset follow-up ownership to the new hf-data plan, and require exact release/schema/hash pins without automatic upgrades. |
| 30 | Oct 16, 2026 | Record the owner-approved Apache-2.0 translation contribution grant in hf-data v1.0.1 and mirror it in kodum. Preserve upstream Apache-2.0/MIT terms, full license text, attribution and source-license fields. Teach exporters to carry only explicit inventoried grants; retain separate training and publication gates. |
| 29 | Oct 16, 2026 | Record the pinned package validation and full kodoom Record import, lossless JSONL/Parquet transport, and private bilingual/4,200-item skills candidate tooling. Preserve screening flags and AI-audit evidence without treating them as semantic acceptance. Keep meaning-check execution, translation permissions and publication open; document the local acceptance handoff. |
| 28 | Oct 15, 2026 | Record the owner decision to accept supplied translations and skip human translation review for the current corpus. Remove that review as an M1 prerequisite, preserve actual quality flags and AI-audit evidence, require honest release disclosure, and record the successful owner-run validator. Next: kodoom schema compatibility/import. |
| 27 | Oct 15, 2026 | Record completion of the planned translation scope and local Hugging Face packaging in `D:\Coding\hf-data`: 1,600 typed-decisions cases and 2,000 selected Helmo items. Distinguish untranslated Helmo originals, actual Codex provenance and AI audit from historical Gemma recipes and human review. Shift next work to import/validation, review, skills, release terms and baselines; retain M1 and training gates. |
| 26 | Oct 9, 2026 | At the owner's request, configure one stable published notebook with the confirmed private dataset and tested runtime pin, defaulting to TPU translation without code edits. Retain short session/production checks and optional CPU diagnosis; Kaggle secrets, consent and model attachment remain prerequisites. Preserve the frozen v21 campaign and research gates. |
| 25 | Oct 9, 2026 | Narrow default CPU preparation to access, storage, resume and attached-model setup; keep source/prompt/dependency audits as optional diagnostics. Skip the CPU environment in allocated TPU sessions while repeating short session checks and retaining production validation. Record a successful private HF access probe and unresolved model attachment; preserve the v21 recipe and research gates. |
| 24 | Oct 8, 2026 | Add an explicit CPU-first notebook preflight after the observed missing-HF_TOKEN failure. Check access, private checkpoint writes, mounted Flax inputs, source/prompt validity, storage, resume conflicts and dependency resolution before TPU setup. Default to CPU-only; require an explicit TPU opt-in and a new published runtime pin. Preserve the v21 data recipe and scientific gates; reject incompatible old campaign identities. |
| 23 | Oct 8, 2026 | Define the published data contract: separate resource licenses, paired decision rows, Parquet/JSONL exports, review/provenance and split manifests, pinned reproduction code and release checks. Distinguish rebuilding reviewed records from rerunning translation; exporters and release verification remain pending. Preserve the frozen v21 production campaign. |
| 22 | Oct 8, 2026 | Audit community value and newer Persian alternatives before computation. Separate teacher agreement, source-label validity and translation fidelity; make helmo optional and enforce training-only use. Hold Gemma 3 drafts out of permissive-model training pending a license route; clarify review sampling, controlled ablations, context limits and acceptance gates. Preserve the frozen v21 production request and notebook pin. |
| 21 | Oct 8, 2026 | Implement the bulk draft campaign: frozen complete scope, structured helmo text, native BF16 batches, one managed worker, atomic cases, verified private HF snapshot saves, deadlines, review downloads/pause and portable operator resume. Switch the notebook/active request; local checks pass, while live TPU/HF evidence and human adoption/release decisions remain pending. |
| 20 | Oct 7, 2026 | Plan a resumable 27B bulk draft campaign to reduce repeated TPU queue waits: all typed cases plus 2,000 helmo records, BF16 batching before quantization, verified private remote checkpoints and portable session/collaborator handoffs. Permit draft generation before human gate acceptance while retaining review/adoption requirements; implementation remains pending. |
| 19 | Oct 7, 2026 | Fix the observed Kaggle Python 3.13 bootstrap failure with a separately resolved TensorFlow 2.20 dependency lock and interpreter-specific selection. Preserve older kernel pins and model versions; live TPU imports, inference and review remain pending. |
| 18 | Oct 6, 2026 | Prepare a bounded Gemma 3 27B BF16/JAX experiment on Kaggle TPU v5e-8, with pinned Flax assets, sharded loading, local workflow tests, private restore contracts and label-focused paired review. Retain the 4B baseline; live TPU execution, persistence, Persian review and adoption remain pending. |
| 17 | Oct 5, 2026 | Add the reference/execution notebook workflow and prepare the private Kaggle free-text gate with pinned code, resumable bundles and review outputs. Kaggle GPU/persistence validation and full training remain pending; no research gate is advanced. |
| 16 | Oct 5, 2026 | Link the separate portable notebook workflow change plan and agent handoff policy. Reference/current-stage notebook separation and Kaggle support are planned; existing Colab/Drive implementation, research scope and gates remain the baseline until migration validation. |
| 15 | Oct 2, 2026 | Gemma 3 12B dropped as a translator and as the free-text-gate fallback: too slow on a free Colab T4 for this project's session budget. Gemma 3 4B is the translator for the whole dataset, including helmo, decided outright rather than conditionally on the free-text gate. The gate (1.2 step 1) now sizes 4B's risk on free text instead of choosing between 4B and 12B; if it fails, the plan shrinks or drops helmo and leans on the automatic checks, meaning check and a wider human-review sample, instead of switching models. Failure-mode mitigation updated to match. |
| 14 | Oct 2, 2026 | Gemma terms read (AI-assisted review, not legal advice): outputs are not Model Derivatives, so a Gemma-translated typed-decisions-fa appears publishable under Apache-2.0; but a model trained on Gemma synthetic data is a Model Derivative, so whether Gemma-made translations may enter the training mix is an open decision before M3 (Licenses, 1.2). |
| 13 | Oct 2, 2026 | Gemma 3 4B (bf16) is the translator for the full run, for speed; 12B is the fallback. Because 4B was weaker on free text and the pilot covered only agent traces, a free-text gate before the full run: about 40 helmo records plus customer_service and security_incidents cases, with a pass bar fixed in advance and outcomes for passing and failing (1.2 step 1). Helmo: whole-record translation, a Latin-token check, an answer comparison in the meaning check, its own review sample of about 100 with a separate error rate, a 1.5–2k start, and a separate train-only config. A failure mode for 4B on helmo. |
| 12 | Oct 1, 2026 | A principle that `data_dir` describes itself (a kodoom-written `README.md` refreshed by every data-writing command, and a notebook that ends with the tree of files the run generated or updated). |
| 11 | Sep 30, 2026 | Translation pilot done (1.2 step 1). Claude Cowork (Claude Opus 5.5) rated five candidates blind on 50 records, and Gemma 3 12B (4-bit) was chosen, with backticks stripped. Gemma 3 added as a candidate and a reference. The dataset card must say the choice came from an LLM rating. Details in `docs/translation-eval-plan.md`. |
| 10 | Sep 30, 2026 | From the first Colab translation trial: TranslateGemma is gated (license and token needed) and its terms on generated outputs must be read before publishing; the API candidate is unavailable (no budget); Qwen3-8B (4-bit) ran on the T4 but changed meaning in several places, so it stays a checker only. |
| 9 | Sep 30, 2026 | Licenses set to the least restrictive: code 0BSD, the project's own data CC0-1.0; derived data keeps its source's license (typed-decisions-fa: Apache-2.0); the model license stays open. Two principles added: licenses, and data stays on the owner's Drive (`data_dir`, a git guard test, nothing published until the owner decides). |
| 8 | Sep 30, 2026 | From the data comparison (`docs/data-comparison.md`): a provenance statement on every dataset card (which data involved a model), metrics reported by question language (3.2), and a code-mixing robustness check (3.4). Taarof moved to `future-work.md`. |
| 7 | Sep 30, 2026 | Added the Google Drive storage constraint (free 15 GB plan, shared with Gmail and Photos): a storage budget in Environments with estimated sizes and rules (no base models on Drive, latest checkpoint plus best as fp16 weights only, Hugging Face as the archive, free-space check before saving, optional frozen embeddings tested in run 1). Updated 2.2 checkpointing, 2.3 artifacts, Constraints, and a failure mode. |
| 6 | Sep 30, 2026 | Updated from the model and dataset cards of Laya-multilingual, DibaOne X1 and M3, and typed-decisions. Gate: language gap moved to parallel MASSIVE and Belebele items, because Laya is near chance on typed-decisions zero-shot even in English; M3 cannot be the gate reference (trained on ParsiNLU and PersianQA); X1 is choice-only. Baselines: Laya as the primary controlled comparison (same backbone), laya-typed-decisions, M3 as an evaluation-only baseline, every baseline reported raw and recalibrated. Run 7 (Laya start, no Persian data). Metrics: KL from gold, parallel language gap; stated mode (specialist or general) and request shape. Robustness: score position bias, noul vs two-option choice. typed-decisions case structure in 1.1. Licenses confirmed for MASSIVE and Belebele; PersianQA's conflicting licenses noted. A failure mode for baselines trained on test sets. Laya links corrected. Success criteria use the parallel language gap and recalibrated baselines. |
| 5 | Sep 30, 2026 | Reframed after a peer review: the project is a Persian typed-decision data and evaluation layer with reference models, not a model first. Deliverables reordered; new principles (resource first, clean training data, benchmark stored raw). PersianQA moved to test-only (GPL-3.0). Full human review of the 400-case test split. Minimal pairs, raw forms and task families in 1.1. Orthographic cleanup only for published translations; the full normalizer moved to the model's input pipeline. DibaOne X1 added as a baseline and gate reference. Resource success criteria. `calibration.json` output, a model-agnostic harness, latency context, seen vs unseen family and minimal-pair metrics. Two failure-mode updates. Fixed the synthetic-typed-decisions and Laya links. |
| 4 | Sep 30, 2026 | Added the decision gate after the M2 baselines (2.4): three signals, thresholds fixed before the baselines (placeholders for now), outcomes A/B/C deciding M3's scope. Added run 6 (Laya-multilingual warm start), a fallback reference if Laya cannot run, adoption items in 3.6 (recalibration recipe, CPU export, request/response adapter), a failure mode, and notes on Gate 2 and M3. Ideas outside this plan moved to `docs/future-work.md`. |
| 3 | Sep 30, 2026 | Added "Environments": laptop development separated from Colab runs, with `dev`, `colab-preflight` and `colab` profiles, tiny random models, a 5-minute smoke run and the laptop's limits. Added the laptop to Constraints, a principle, and a failure mode. |
| 2 | Sep 30, 2026 | Plan as first added to the repository. |
