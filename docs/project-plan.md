# Kodum: Persian typed-decision models and evaluation

Version 31 · Oct 16, 2026 · @Shah

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

**Next work, in order:**

1. Implement/test exact release selection and a consumer adapter for the chosen
   hf-data schema/transport; record the baseline version and checksum in config.
2. Build model-specific mixture and train/validation/calibration ID manifests
   from available released resources; enforce test isolation and declared roles.
3. Complete the CPU-only model smoke path, baseline adapters and raw/recalibrated
   evaluation on the selected resources. Native data absent from a release is
   requested from hf-data and reported unavailable until delivered.
4. Freeze the M2 endpoints and decision thresholds before real baseline runs;
   measure the available baseline matrix, record the gate, then select M3 scope.
5. Run an explicitly selected provider's training preflight, train the selected
   reference models, and deliver reproducible evaluation and CPU artifacts.

Real M2 baseline measurements and M3 training are not marked complete. Existing
uniform-baseline CPU examples are loading/harness checks, not model evidence.
The old dataset-first plan is preserved in
[the v30 archive](project-plan-v30-archive.md). Dataset work is governed by
`D:\Coding\hf-data\docs\project-plan.md`; historical data specifications and
handoffs in this repository are migration/reference material.

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

Real training targets a free Colab T4: fp16 rather than bf16, SDPA/eager
attention, no FlashAttention 2. Validate real model loading, memory, numerical
stability, one train/eval step and interrupted/resumed checkpoints in the
selected preflight before a long run. CUDA dependencies stay in lazy GPU extras
and use `constraints/gpu.txt` without replacing provider PyTorch. Kaggle model
training remains unvalidated; TPU translation recipes are historical data work.

Code must run on Windows and Linux. Use pathlib and explicit UTF-8, roughly
four CPU threads and main-process data loading. Every training/long model run
uses `kodoom.runs.Run` and `save_latest` / `save_best`. Notebooks bootstrap pinned
code and invoke package commands; no model computation logic belongs in cells.

Drive's free 15 GB includes mail/photos. Measure actual free space before a
training run; keep at most latest and best for the active run, best weights in
fp16, base-model caches on provider scratch, and small predictions/logs under
the run directory. Archive model artifacts only to an explicitly approved
destination. Kaggle output is temporary until privately saved and restored in
a fresh session. Never commit datasets, weights, caches or run outputs.

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
| KD-01 | Dataset lock and release loader | Pin baseline 1.0.1/schema 1.0.0 and manifest hash; select resources explicitly; verify checksums and stable bilingual keys through a supported adapter. Reuse existing import/schema/transport helpers without triggering dataset review. |
| KD-02 | Model mixture and split manifests | Only declared training resources; source_id/language/pair grouping; source test isolation; reproducible train-derived validation/calibration; one normalization function; unchanged canonical data. |
| KD-03 | CPU model smoke and resume | Tiny/random encoder and decoder paths, fixture-sized input, prediction/calibration/report schemas and interrupted/resumed checkpoints; no real weights or GPU. |
| KD-04 | Baseline adapters and M2 measurements | Freeze endpoints/thresholds first; score available released benchmarks raw and recalibrated; save predictions, coverage, exposure and latency/context details. Request missing native/skills resources from hf-data. |
| KD-05 | Gate and selected training preflight | Record A/B/C decision, actual required benchmark coverage, chosen run scope, provider and measured runtime/numerical/resume readiness. No implicit use of the old translation notebook. |
| KD-06 | Reference training and report | Execute chosen runs, preserve matched ablations and exact data locks, produce calibration/CPU artifacts and reproducible model report. |


## Part 2 — Implementation

Part 2 builds model-specific mixtures from pinned hf-data releases, trains an encoder and a small decoder on free Colab, and keeps every log and prediction needed to judge them. Canonical data development and review are owned by hf-data.

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

**English slice:** the original English typed-decisions train split, plus some records with an English question over a Persian state. This keeps the model usable the way a foreign client would use it.

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
4. Measure token lengths with each candidate tokenizer. Default model budgets
   remain 256 tokens, or 384 for QA; disclose truncation and unsupported inputs.
5. Balance the available training mix, targeting no source above about 30%.
   Record weights, repetitions and source counts rather than changing releases.
6. Select early-stopping validation and temperature-only calibration from
   released train data by source_id. Respect any existing released validation
   or calibration roles. All questions, language variants and minimal-pair
   halves stay grouped. Source test membership is never reassigned.
7. Enforce model-side leakage protection across selected splits, including
   source cases, pairs and normalized/near-duplicate inputs. Report suspect
   source defects to hf-data; do not rewrite canonical splits or targets.
8. Freeze mixture/selection manifests and hashes before training. An eventual
   30–60k-decision mix is a target conditional on available data releases, not
   a claim about the current package's size.

**Format per model family**

- **Encoder:** one input per state + question + one option, scored, then a softmax across that question's options. Questions with more than 8 options train on the correct one plus 5–8 sampled wrong ones.
- **Decoder:** one prompt with lettered options; the answer is the next-token probabilities of the option letters. At most 20 options per prompt (sample subsets for MASSIVE's 60 intents). One question per prompt in v0.

### 2.2 Training

**Base model candidates**

| Model | Size | Role | Why | Watch out for |
| --- | --- | --- | --- | --- |
| [mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) | 140M (42M non-embedding) | Pipeline debugging | Fast, same family and tokenizer as base, runs easily on the laptop CPU. | Lower ceiling. |
| [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) | 307M (110M non-embedding) | **Main model** | Modern multilingual encoder covering 1,800+ languages. Backbone of Laya's multilingual checkpoint. Trains fully on a T4 and answers fast on CPU, which suits an offline voice assistant. | Weaker at following free-form instructions. Many-option questions multiply compute. |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 0.8B | **Challenger** (LoRA) | Apache-2.0, 201 languages. Reads instructions and all options in one pass. [Dohnuts](https://huggingface.co/PsiACE/Dohnuts-0.1.0-0.8B), built on it, beat Laya-multilingual on 51-language MASSIVE intent, about 60% vs 37%. | Hybrid Gated DeltaNet layers may not train cleanly on a T4 in fp16. Slower on CPU. |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B | Fallback challenger | Standard transformer, safe on a T4; the base [Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev) used. | Older and less multilingual than Qwen3.5. |
| [Persian-ModernBERT-base](https://huggingface.co/myrkur/Persian-ModernBert-base) | base size | Optional baseline | Custom Persian tokenizer, 2.5B Persian training tokens. Tests whether a Persian tokenizer beats a multilingual one. | Weak on English questions. |

Not chosen: ParsBERT and XLM-R (superseded by mmBERT), and anything 2B or larger (too tight on a free T4).

**Baselines first**, with no training, so every gain is measured against something:

- Random and majority-class.
- Untrained mmBERT-base with a fresh head (expected near chance).
- [Laya](https://github.com/NandhaKishorM/laya)-multilingual (`convaiinnovations/laya-multilingual`, Apache-2.0), zero-shot: the open multilingual reference, and the **primary controlled comparison**, because it starts from the same mmBERT-base as the main model (see below). What its card says matters here: it ships uncalibrated (temperature 1.0; refitting moved its mean ECE from 0.314 to 0.106), it is near chance on typed-decisions zero-shot (0.342, against 0.318 random and 0.461 majority), it wants choice questions under about 20 options (256 tokens for a question and its options), it rarely picks the first level of a score question, and noul can under-report "true".
- [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) (ModernBERT-large, English), optional: an English specialist fitted on the typed-decisions workflows. On the Persian test it shows what an English-only specialist loses in Persian.
- [Decima-base](https://huggingface.co/amyrmahdy/decima-base), a newer mmBERT decision baseline candidate: audit its exact checkpoint, context budget and training exposure before reporting held-out results. Its published results are author reports until independently reproduced here.
- [DibaOne X1](https://huggingface.co/Dibachain/DibaOne-X1) (Apache-2.0), zero-shot: the Persian-first open reference. A 278M XLM-R-family cross-encoder behind a 118M retriever; **choice questions only** (no noul or score). Trained on its own synthetic data (CC0) and Wikipedia, so no overlap with this project's test sets is expected. Slow on CPU (0.7–1.8 s per decision at 2 threads in `best` mode), so it is evaluated on a T4.
- [DibaOne M3](https://huggingface.co/Dibachain/DibaOne-M3) (CC BY-NC-SA 4.0), evaluation only: a Persian-first bi-encoder answering all three question types. **It was trained on ParsiNLU and PersianQA**, so its scores on those sets are in-domain and excluded from every held-out comparison; its full source list (`NOTICE`) is checked against the other test sets before its scores are reported.
- Every baseline is reported **raw and recalibrated**: a temperature per question type is fitted on this project's calibration split for every model alike, so calibration differences are not just "one model shipped a temperature and another did not".
- If neither reference runs on the laptop or a T4, prompted Qwen3.5-0.8B takes their place, including at the decision gate (2.4).
- Qwen3.5-0.8B prompted, no training.
- Jev through its API, only if access works. Otherwise cite published third-party numbers and label them so.

**Run plan**

| Run | Model | Data | Purpose |
| --- | --- | --- | --- |
| 0 | mmBERT-small | 1k records | Smoke test: loss falls, and it can overfit a tiny batch |
| 1 | mmBERT-small | Full mix | First full result, cheap |
| 2 | mmBERT-base | Full mix | Main model |
| 3 | mmBERT-base | Mix without translated data | Ablation: did translation help or hurt? |
| 4 | Qwen3.5-0.8B with LoRA (or Qwen3-0.6B) | Full mix | Challenger, head-to-head with run 2 |
| 5 (optional) | Best of runs 2 and 4 | Mix without the English slice | Does English data help Persian? |
| 6 | Laya-multilingual as the starting point (warm start) | Full mix | Does starting from a decision model beat starting from plain mmBERT-base? Compare with run 2. |
| 7 | Laya-multilingual as the starting point | English slice only (no Persian data) | With run 6, the cleanest measure of what the Persian data adds: same start, same procedure, only the Persian data differs. |

Which runs actually happen depends on the decision gate (2.4).

**Controlled and uncontrolled comparisons.** A shared backbone controls architecture only: comparing the main model with Laya does not isolate training data, objective or recipe. Runs 6 and 7 isolate the Persian-data intervention only when starting checkpoint, update budget, seeds and calibration policy match. Other model comparisons are empirical scores, not causal claims about backbone suitability. Add a separate with/without-helmo comparison if helmo is eligible; do not blend its effect with typed-decisions-fa. One seed is exploratory; repeat promising matched comparisons with at least three seeds when affordable.

**Settings on the Colab T4**

- fp16 mixed precision; SDPA or eager attention (a T4 has no FlashAttention 2).
- Max length 256 (384 for QA); gradient accumulation for a larger effective batch.
- Encoder: full fine-tune, small learning rate with warmup, 2–4 epochs. Decoder: LoRA, rank 16–32.
- Loss: cross-entropy against the gold distribution. It is a proper scoring rule, so it rewards honest confidence.
- Early stopping on validation log-loss, not accuracy.
- After training: temperature scaling on the calibration split, one value per model (optionally per question type).
- Fixed seeds. Checkpoint to Drive every few hundred steps, within the storage budget (Environments): the latest checkpoint overwritten in place, the best one as fp16 weights only. Every run must resume after a dropped session.
- Budget: one run fits one Colab session (1–2 hours). If it doesn't, shrink the data or model rather than stretch the session.

### 2.3 Collecting logs and training data

Keep enough that any number in the final report can be recomputed without retraining.

- **Run registry:** one row per run with run ID, date, model and revision, dataset version, key settings, headline metrics, and a one-line decision (keep, or drop and why).
- **Per-run config:** model revision hash, dataset version hash, seeds, hyperparameters, max length, option-sampling settings, library versions, GPU type.
- **Curves:** train loss, validation loss, validation accuracy and ECE per step or epoch; runtime and peak GPU memory.
- **Where:** plain files on Google Drive (CSV or JSON, or TensorBoard / MLflow file logs). Avoid depending on a hosted tracker that may not be reachable.
- **Predictions, not just metrics:** for every evaluation, save per item the ID, gold distribution, predicted distribution, temperature used and latency. Every later metric, chart and error analysis comes from these files, and they can be published as raw results.
- **Error log:** after each main run, read 50 wrong answers and tag them: translation artifact, ambiguous options, truncated context, number or date reasoning, colloquial text, label noise.
- **Artifacts kept:** the best weights in fp16, its tokenizer, its temperature value, the exact training-mix version, and a draft model card; pushed to Hugging Face, then removed from Drive.

### 2.4 Decision gate: does Persian need a new model?

After the M2 baselines and before M3's training runs, one planned decision: does Persian need a new decision model, or does it mostly need data and a benchmark? Every outcome still publishes something; the gate only decides where the time goes and what the release leads with.

**Reference model.** The stronger on held-out Persian tasks of Laya-multilingual (same mmBERT-base as the main model) and DibaOne X1 (Persian-first; choice questions only, so noul and score signals come from Laya). DibaOne M3 cannot be the reference: it was trained on ParsiNLU and PersianQA, two of the held-out sets. Their weaknesses in Persian are the model's reason to exist; if a Persian-first model is already strong, the gate should find that out. If neither can be run, prompted Qwen3.5-0.8B is the reference.

**Three signals**, all measured with no training:

1. **Language gap.** The reference's accuracy on parallel native sets, English minus Persian on the same items: MASSIVE (`en-US` vs `fa-IR`, same utterance IDs, 20-option questions as in Laya's own MASSIVE evaluation) and Belebele (`eng_Latn` vs `pes_Arab`). Not on typed-decisions: Laya-multilingual is near chance there zero-shot even in English (0.342 against 0.318 random), so any gap would be noise at the floor. A reference's gap on a set counts only if its English score there is clearly above chance. The typed-decisions gap remains a metric for trained models (3.2).
2. **Held-out native tasks.** The reference on ParsiNLU and Belebele-fa, against the random and majority baselines.
3. **Persian skills.** The reference on the code-labeled test templates (Jalali dates, digit forms, Toman vs Rial, Iranian formats).

Calibration in Persian and the STT transcripts are measured and reported too, but they do not decide the outcome.

**Thresholds, written down before any baseline runs.** Freeze primary endpoints, weights across suites and the owner's thresholds before M2 starts. The values below are placeholders until recorded in that decision. Baseline results can set practical M3 targets, but must not select a favorable primary metric after seeing the scores.

| Signal | Large gap (counts toward A) | Small gap (counts toward C) |
| --- | --- | --- |
| Language gap (MASSIVE, Belebele) | ≥ _8_ points | < _3_ points |
| Held-out native tasks | Within _5_ points of majority-class on most tasks | Clearly above chance on every task |
| Persian skills | Below _60_% accuracy | At or above _80_% accuracy |

**Rules**

- A signal counts only if its 95% confidence interval clears the threshold, using the paired bootstrap from 3.3. With 2,000 decisions, a 2-point gap can be noise.
- The skills signal counts on its own: a large skills gap alone means outcome B at least, because the skills data alone justifies a small model or a Laya fine-tune.
- A result on a threshold: spend one session on run 1 (mmBERT-small, full mix) as a probe. Its gain over its own untrained baseline shows whether training pays off.

**Outcomes**

| Outcome | What the baselines show | What M3 does | What the release leads with |
| --- | --- | --- | --- |
| **A. Large gap** | Large language gap, weak held-out scores, skills near chance | The full run plan (runs 0–6) | The model and report, referencing the consumed data release |
| **B. Middle** | A moderate gap, or a large gap in only some signals (often the skills) | Runs 0–3 and 6; drop the challenger (4) and run 5 | The model comparison and measured areas of gain |
| **C. Small gap** | The reference loses little in Persian and does reasonably on held-out tasks | At most a light Persian fine-tune of Laya-multilingual (run 6), mostly on the skills data | A report comparing open decision models on pinned Persian data, with at most a light reference-model adaptation |

Outcome C is not a failure: a report on how well open decision models handle Persian is useful to every model author in this space, and it still ships a model.

## Part 3 — Testing and evaluation

The model is judged on Persian it never trained on, on how honest its confidence is, and on how much it loses against English, always against baselines on the same items.

### 3.1 Test sets

| Test set | Question it answers | Source |
| --- | --- | --- |
| In-task native | Did it learn the trained tasks? | MASSIVE-fa test (20-option questions, fixed seed), FarsTail test |
| Held-out native tasks | Is it a decision model, or a classifier of the tasks it saw? | ParsiNLU (4 tasks), Belebele-fa, PersianQA (in-domain for DibaOne M3, which trained on ParsiNLU and PersianQA) |
| Parallel language gap | How much does the same native item lose in Persian? | MASSIVE `en-US`/`fa-IR` and Belebele `eng_Latn`/`pes_Arab` test items |
| Held-out templates | Did it learn the task, or memorize the phrasing? | Held-out template IDs from the selected data release and frozen model protocols |
| typed-decisions-fa test + English original | How much does the same case lose in Persian? | 400 cases, 2,000 decisions, in both languages |
| Persian skills | Dates, digits, currency, Iranian formats | Code-labeled data, test templates only, raw forms, with minimal pairs |
| Real use | Does it hold up on noisy speech transcripts? | Own STT→intent data |

### 3.2 Metrics

- **Accuracy** (top-1), plus macro-F1 where classes are imbalanced.
- **Calibration:** ECE with fixed bins, Brier score, log-loss, and a reliability diagram.
- **Accuracy vs coverage:** if only answers above a confidence threshold are auto-accepted, what share is automated and how accurate is it? This is the number that matters for Jev-style gating.
- **Distance from the gold distribution** on typed-decisions: KL divergence from gold, the typed-decisions leaderboard's headline next to accuracy and Brier, so results can be compared with it (Jensen–Shannon as a bounded companion). ECE alone is not read as quality: on that leaderboard the prior, which ignores the input, has the best ECE.
- **Score questions:** within-one accuracy and mean absolute error.
- **Language gap:** English minus Persian accuracy on the same typed-decisions cases, and on the parallel MASSIVE and Belebele items.
- **By question language:** every metric reported separately for Persian and English questions over the same Persian state (`question_lang`). laya-persian-benchmark found English instructions scored higher than Persian ones on identical Persian content (41 vs 36 of 64 for Laya-multilingual), so instruction language is a variable, not noise.
- **Seen vs unseen task families:** every metric reported separately for task families in training and those held out, so "decision model, not classifier" is a number rather than a claim.
- **Minimal-pair consistency:** share of skill pairs where both halves are right.
- **Cost:** CPU latency per question (p50 and p95) and peak memory (RSS) on the laptop, no GPU. Every latency figure is published with its context, so it can be compared elsewhere: CPU model, threads, RAM, model, precision (and export format), sequence length and options per question.

### 3.3 Comparisons

- Every fine-tuned run against every baseline from 2.2, on the same items.
- **Matched intervention comparison:** run 6 against run 7, same start with and without the selected eligible Persian data and matched training budget. Run 2 against Laya-multilingual is an architecture-matched external baseline, not an isolated data intervention.
- **Mode is stated with every typed-decisions result**, as the leaderboard asks: *specialist* (trained on typed-decisions `train`, as this project's models are) or *general* (zero-shot, as Laya, DibaOne and prompted Qwen are). The gap between the modes is the price of generality, not a quality ranking; held-out tasks measure generality.
- **Request shape is recorded:** whole case in one request (the leaderboard's shape) or one question per request. It changes results (Jev's yes/no accuracy was 0.843 per question and 0.788 alongside the others), so compared rows use the same shape.
- Run 2 vs run 3 (with vs without translated data) and run 2 vs run 4 (encoder vs decoder).
- Before vs after temperature scaling.
- 95% confidence intervals by bootstrap, with paired comparisons on the same items. A difference inside the interval is reported as a tie.
- Resample whole source cases (all five questions together) or minimal pairs; language/model comparisons retain paired IDs. Report missing/failed coverage and common-item comparisons. Log loss and Brier complement teacher agreement; ECE alone is insufficient. Known, suspected and unknown baseline exposure are separate statuses.
- Report originally authored Persian, human-localized parallel data, machine translations and procedural skills separately. Audit tokenizer/context limits for every adapter; unsupported inputs remain visible rather than silently truncated or dropped. Option subsampling and full-option evaluation are distinct protocols, and removal of an option requires explicit remapping of soft target mass.

### 3.4 Robustness checks

- Shuffle option order: the answer should not change.
- Rename options, or add a plausible wrong option.
- Rewrite a formal state colloquially, and the reverse.
- Swap Persian and Latin digits.
- Ask in English about a Persian state.
- Code-mixing: English technical terms inside a Persian state ("اکانتم لاک شده", "invoice رو ریجکت کردن"), as Persian users write them. The answer should not change against the all-Persian wording.
- Position bias on score questions: how often each level position is chosen, since Laya reports rarely picking the first level.
- noul against the same question as a two-option choice (neutral keys, yes/no descriptions): the probabilities should agree.

### 3.5 Success criteria

**Harness and consumption criteria:**

- [ ] Every experiment pins an exact dataset release, schema/transport version, manifest checksum, selected configurations and model-derived split IDs. Reports cite the release's actual quality status and waiver without repeating dataset review.
- [ ] The harness has scored at least three models this project did not train (for example Laya-multilingual, DibaOne X1 and prompted Qwen3.5-0.8B), with their raw predictions published.
- [ ] Skills evaluation consumes the released raw forms, held-out template IDs and minimal-pair grouping; regeneration acceptance is owned by hf-data.
- [ ] The model pipeline enforces released training/test-only roles and permitted-use decisions, and the model card attributes every consumed component.

**Model criteria.** Set exact thresholds after the baselines and before runs 1–4, so results can't move the goalposts. (The decision gate's thresholds in 2.4 are different: they are fixed before the baselines.)

- [ ] Fine-tuned mmBERT-base beats Laya-multilingual on held-out native Persian tasks, outside the confidence interval.
- [ ] ECE after temperature scaling is lower than every baseline's, each baseline also recalibrated on the same calibration split.
- [ ] The Persian–English gap on the parallel MASSIVE and Belebele items is smaller than Laya-multilingual's. (Not on typed-decisions, where Laya is near chance zero-shot in both languages, so its gap there says nothing.)
- [ ] CPU latency is low enough for a voice assistant (threshold set after measuring baselines).

### 3.6 Reporting

- One headline table, a reliability diagram, an accuracy-vs-coverage curve and a language-gap chart.
- Raw prediction files published with the report.
- A model-agnostic evaluation harness (for example `kodoom evaluate --model <adapter> --benchmark typed-decisions-fa --out results/<model>/`) whose input is the System One request (a state plus typed questions, as in typed-decisions and Laya), with adapters for per-question models such as DibaOne, and that reads the same request and response format as the community benchmarks ([typed-decision-bench](https://github.com/kyr0/typed-decision-bench), [open-system-one](https://github.com/zhlei07/open-system-one)), so others can run their models on the Persian sets. Its outputs are raw predictions (ID, gold, predicted distribution, temperature, latency), from which every metric is recomputed.
- Model card: exact consumed data releases and hashes, component licenses, intended use, known limits and the dataset project's actual quality statement. No human translation-error rate is invented; the human-review waiver remains disclosed.

**Making adoption easy.** For downloads, easy use matters more than a few accuracy points.

- **Recalibration recipe.** A documented step, with a script, that fits a new temperature on 100–300 of a user's own labels, and writes it as a `calibration.json` in typed-decision-bench's format (one temperature per question type, a global fallback), so engines that read that format pick it up. Temperature scaling already exists from 2.2; this makes it usable on someone else's data, where the shipped calibration will not hold. The model card says plainly that the top answer transfers better than the probability.
- **CPU-friendly export.** An exported model (for example ONNX, optionally int8) with the latency measured on it, so the CPU latency in 3.2 is reproducible by users.
- **Community request/response format.** The evaluation harness above already reads it; a small adapter lets the model answer in the same format.

## Failure modes and responsibility

| Failure | Kodum response |
| --- | --- |
| Floating data changes an experiment | Require exact release and manifest/payload hashes; keep an immutable local cache or snapshot. |
| A compatible loader is mistaken for an unchanged benchmark | Record content fingerprints, IDs and exact releases; rerun comparisons or use common-item protocols after upgrades. |
| Test data or its language variant leaks into training/calibration | Group cases/pairs and enforce source roles and immutable subset manifests in the model pipeline. |
| The model memorizes task labels/options | Use held-out released task families/templates, option perturbations and matched intervention comparisons. |
| Small language gaps or calibration claims are noise | Use paired case/pair bootstrap, common coverage and fixed endpoints; publish raw and recalibrated baselines. |
| Dataset defects appear during model error analysis | Record release/record IDs and report them to hf-data. Use explicit run-specific subset manifests; do not silently correct the source or require a new full review campaign. |
| A baseline was trained on a nominally held-out task | Record known/suspected/unknown model training exposure and keep in-domain scores out of held-out claims. |
| Code works on CPU but fails on the real provider | Run the selected model preflight with real dtype, memory, context limits and checkpoint restore before long execution. |
| A calibrated model is too slow or a full fine-tune adds little | Measure CPU cost and follow the A/B/C gate rather than expanding model scope automatically. |

## Milestones

| Milestone | Current status / exit condition |
| --- | --- |
| M1: accepted external dataset input | Closed for kodum by the Oct 16 owner decision. Baseline hf-data 1.0.1 is available. Dataset review, future improvement and public release are owned by hf-data; they are not marked complete by this transfer. Operational lock/loader work is KD-01. |
| M2: model baseline evidence and decision | Open. KD-01 through KD-04, actual benchmark availability, frozen endpoints/thresholds and measured raw/recalibrated baselines. Record the gate before choosing substantial training. |
| M3: selected reference-model development | Open. KD-05/KD-06 at the gate's chosen scope, provider preflight, checkpoints, calibration and controlled comparisons. |
| M4: model evaluation and release | Open. Reproducible report/predictions and model card, CPU export, stated limits/attribution and a separate model-publication decision. |

Stopping at a model gate can produce a useful evaluation report without a new
large training campaign. Dataset publication is managed separately and does not
need to coincide with a model release. Existing embedded diagrams and historical
documents reflect the old combined project unless explicitly updated; this
table and the active ownership/consumer contract above govern current work.

Remaining data work is tracked only in `D:\Coding\hf-data\docs\project-plan.md`.
Keep [the v30 archive](project-plan-v30-archive.md) for evidence and past decisions.
The following reference table/version history retains earlier source context;
old Part 1 references point to that archive rather than assigning data tasks to
kodum again. Published model specifications/numbers in the retained recipe are
historical candidate evidence; verify exact revisions/context/runtime before
using them in a new experiment.


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
