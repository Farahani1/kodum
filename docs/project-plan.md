# Persian Typed Decisions: Project Plan

Version 11 · Sep 30, 2026 · @Shah

## Overview

**Persian Typed Decisions: an open bilingual dataset, a Persian skills suite and an evaluation harness, with CPU-friendly reference models.**

Typed decisions are choice, score and yes/no (noul) questions answered with calibrated probabilities, the interface Jev popularized. Open decision models already exist, including multilingual and Persian-first ones, so another Persian model alone is not a distinctive contribution. What is missing is an independent, reproducible Persian data and evaluation layer that any decision model can be trained or measured on. This project builds that layer, and trains small reference models on top of it.

The test for every deliverable: could a developer drop this project's model entirely and still benefit? For the first three, the answer is yes.

**Deliverables**, each publishable on its own, in order of importance:

1. **typed-decisions-fa**: a Persian translation of LocalLLaMA/typed-decisions, same case IDs, splits and gold distributions as the English original, so the same case can be scored in both languages. Every test case is human-reviewed (Part 1).
2. **Persian skills suite**: executable generators for Jalali dates, digit forms, Toman and Rial, business hours and Iranian formats, with labels computed by code and minimal pairs, regenerable by anyone from published seeds and templates (Part 1).
3. **Evaluation harness and report**: a model-agnostic harness that scores any decision model on the Persian sets, raw predictions for every model evaluated, and a report on accuracy, calibration, the English–Persian gap and CPU latency (Part 3).
4. **Reference models**: a fine-tuned mmBERT-base (main) and Qwen3.5-0.8B (challenger), trained only on permissively licensed data, each with a model card, a CPU export and a `calibration.json` (Part 2). How much of this is trained depends on the decision gate (2.4).
5. **Reproducible tooling**: the record schema, converters, split and leakage tools, and the Persian normalizer (Parts 1–2).

**Constraints**

- Training runs on the free Colab tier: a T4 GPU (16 GB), no bf16, no FlashAttention 2, sessions that drop.
- Google Drive storage is the free 15 GB plan, shared with Gmail and Photos, so less than 15 GB is actually free. Drive is scratch space, not an archive (see Environments: Storage budget).
- Data preparation, generation and small-model evaluation run on the laptop CPU.
- The laptop is modest: Intel i3-1005G1 (2 cores, 4 threads), about 12 GB RAM, and an NVIDIA MX110 (2 GB) that is not used. It proves the code works; it does not train the real models (see Environments).
- The Jev API may not be reachable; nothing in the plan depends on it.

**Principles**

- The data is the project. Laya's authors report their base checkpoints sit near chance on typed decisions until fine-tuned.
- Build a decision model, not a classifier. Options arrive in the request, and whole tasks stay out of training to prove it.
- Report native Persian results separately from translated data, always.
- Publish early. Part 1's dataset is the first public output.
- Develop on the laptop, run on Colab. Nothing reaches Colab until it has run end to end on the laptop.
- The resource is the headline; the model is a reference implementation. The project is worthwhile even if the model turns out average.
- Clean training data only. A source enters training only if its license allows a permissively licensed model; everything else is test-only or converter-only.
- Licenses: the least restrictive. The code is 0BSD and the project's own data (the generated skills and their templates) is CC0-1.0. Data derived from other people's work keeps their license: typed-decisions-fa stays Apache-2.0 with attribution, and a source's license is enforced in code (`kodoom.sources`). The choice of license for the trained models is still open, and must respect the attribution terms of what they were trained on (MASSIVE is CC BY 4.0).
- Data stays on the owner's Drive. Generated, converted and translated datasets are written to the profile's `data_dir` (Drive on Colab), never into the repository checkout and never into git (a test fails if a data file is tracked). Nothing is published to Hugging Face or anywhere else until the owner decides to; the publication steps in Part 1 wait for that decision.
- Benchmark data is stored as people write it. Normalization is part of a model's input pipeline, not of the published data, so the benchmark still tests how other models handle digit forms and spelling variants.

&#91;embedded content: pipeline · sources through Parts 1–3\]

Translated and native data meet only in the training mix, and only their train splits; every test set reaches Part 3 untouched.

## Environments: laptop development, Colab runs

Development and Colab are separate. The code is written and proven error-free on the laptop, then run for real on Colab. There is one codebase and one pipeline; only a config profile decides which models, data sizes and paths are used.

**Three profiles**, chosen explicitly (for example `--profile dev`); the code never guesses where it is running.

| Setting | `dev` (laptop CPU) | `colab-preflight` (T4) | `colab` (T4) |
| --- | --- | --- | --- |
| Purpose | Prove the whole pipeline runs, error-free | Catch T4-only failures before a long run | The real run |
| Data | 20–50 cases per source, fixed seed | Small slice | Full sets |
| Translator | Stub (echo or tagged text) or a tiny model | Real translator | TranslateGemma or an API |
| Checker | Stub with fixed or random flags | Real checker | Qwen3-8B, 4-bit |
| Encoder / decoder | Tiny random copies (below) | Real models | mmBERT-small / base; Qwen3.5-0.8B with LoRA |
| Precision | fp32 | fp16 | fp16 |
| Training length | 10–20 steps | About 20 steps | Full epochs |
| Output paths | `./runs/` | Drive | Drive |

**Tiny random models.** For the laptop, build a copy of each real model from its own config, shrunk to about 2 layers and hidden size 64, with random weights and the real tokenizer. Only the config and tokenizer are downloaded, never the real weights. Most of the size is the vocabulary table (about 250k tokens × 64, roughly 60 MB), and a training step on a small batch should take well under a second on the laptop (an estimate, to confirm once the code exists). The copies use the same classes, input format, tokenizer and save/load code as the real models, so they catch shape, schema and pipeline bugs.

**The smoke run.** One command runs the whole pipeline on the `dev` profile: prepare → translate → automatic checks → pack → training mix → train → temperature scaling → evaluate → report. It passes when every step finishes and every output file has the expected schema. It also tests resuming: stop it during translation and during training, rerun, and confirm it continues where it stopped. **Budget: under 5 minutes on the laptop.** If it gets slower, shrink the dev data or the tiny models; a slow smoke run stops being run.

**What runs where**

| Task | Laptop | Colab |
| --- | --- | --- |
| `dev` smoke run | Yes | — |
| Data preparation, converters, generators, automatic checks, metrics | Yes | Yes |
| mmBERT-small / base inference, including the CPU latency test (3.2) | Yes, slow but fine | — |
| Qwen3.5-0.8B inference | A few examples only (about 3–4 GB RAM) | Yes |
| mmBERT-small training | Possible but not worth it | Yes |
| mmBERT-base and Qwen training, TranslateGemma, Qwen3-8B checker | No | Yes |

**Laptop settings**

- CPU only. The MX110 is ignored: 2 GB, an old GPU generation that recent PyTorch builds may no longer support, and unlike a T4 in any case.
- About 4 threads (`torch.set_num_threads(4)`) and data loading in the main process (`num_workers=0`).
- Close the browser during the few real-model inference checks; about 12 GB RAM is the limit.
- The same code runs on Windows and Colab: `pathlib` for paths, no shell-specific steps.

**Colab stays thin.** A notebook only mounts Drive, clones the repo at a fixed commit or tag, installs the pinned requirements, and calls the same command with `--profile colab-preflight` or `--profile colab`. No logic lives in notebooks, so nothing runs on Colab that the laptop has not already run.

**Dependencies.** One pinned requirements set shared by both, plus a Colab-only extra for CUDA packages (such as `bitsandbytes` for 4-bit loading) that the laptop never installs.

**What the laptop cannot catch.** fp16 overflow (the Gemma 3 / TranslateGemma issue), Qwen3.5's Gated DeltaNet layers in fp16, GPU memory at the real batch size and length, and real run times. The `colab-preflight` profile (real models, about 20 steps, about 5 minutes) catches these before a full session is spent.

**Workflow:** laptop `dev` smoke run → Colab `colab-preflight` → Colab full run.

### Storage budget: Google Drive, free 15 GB plan

Drive holds only what must survive a dropped session. Everything else lives on the Colab runtime's own disk (tens of GB, wiped at the end of each session) or on Hugging Face. The sizes below are estimates from parameter counts, not measurements; measure them in the first preflight run.

| What | Size (estimate) | On Drive? |
| --- | --- | --- |
| Datasets, translations, training mix | about 1 GB | Yes |
| Predictions, logs, run registry | under 0.5 GB | Yes |
| mmBERT-base run: latest checkpoint with optimizer state (for resuming) | about 3.7 GB | Only during the run |
| mmBERT-base run: best checkpoint, weights only, fp16 (for early stopping) | about 0.6 GB | Only during the run |
| Qwen3.5 LoRA run: adapter weights and their optimizer state | about 0.5 GB | Only during the run |
| Final models (mmBERT-small, mmBERT-base variants, Qwen3.5 adapter), fp16 | about 0.3–0.6 GB each | Until pushed to Hugging Face |
| CPU exports (ONNX) | about 1.5 GB | Until pushed to Hugging Face |
| Base models, translators, checker (TranslateGemma 4B about 8.6 GB, Qwen3-8B about 16 GB) | 25+ GB | **Never** |

Rules:

- **Never put base models, translators or the checker on Drive.** Colab downloads them from Hugging Face into the runtime disk (the Hugging Face cache stays under `/content`, not on Drive).
- **Two checkpoints at most per run:** the latest, with optimizer state, overwritten in place so a dropped session can resume; and the best, as fp16 weights only. That keeps an mmBERT-base run's peak at about 4.3 GB instead of about 7.5 GB for two full checkpoints.
- **When a run ends,** delete the latest checkpoint and keep only the final fp16 weights, its tokenizer and its temperature.
- **Decoder runs save only the LoRA adapter,** never the full 0.8B model at every checkpoint.
- **Hugging Face is the archive.** Each finished dataset, model or export is pushed there (privately until it is released), and only then deleted from Drive. Predictions and logs stay on Drive; they are small.
- **Check free space before writing a checkpoint,** and stop the run with a clear message if the next write would not fit, instead of failing halfway through a save.
- **One run at a time on Drive.** Runs share the space, so a run's scratch files are cleaned up before the next one starts.
- **Optional:** freeze mmBERT's embedding matrix. Its 256k-token vocabulary holds about two-thirds of the 307M parameters, so freezing it removes most of the optimizer state (a resumable checkpoint drops to about 2 GB) and speeds up training. Run 1 checks that it costs no accuracy before any main run relies on it.

Measure the space actually free on Drive before M3; if it is under about 8 GB, clear it first. With these rules the project's own peak stays around 6–7 GB.

## Part 1 — Data augmentation

Part 1 turns English typed-decision data and code-generated examples into Persian data, and publishes typed-decisions-fa as the project's first public output.

### 1.1 Data preparation

**Sources to translate**

| Dataset | Size | License | Why this one | Preparation needed |
| --- | --- | --- | --- | --- |
| [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | 1,200 train cases (6,000 decisions); 400 test cases (2,000) | Apache-2.0 | Already in Jev's shape. Gold labels are probability distributions, which suits calibration. Covers four business workflows Persian data lacks (agent traces, customer service, invoices, security incidents). Russian and Japanese versions exist, so a Persian one joins a set people already run. | Pin the dataset revision. Map the schema. Mark every text field as translate or keep. Flag cases with money, dates or numeric thresholds. Keep the original splits and case IDs. |
| [helmo/synthetic-typed-decisions](https://huggingface.co/datasets/helmo/synthetic-typed-decisions) | 9,879 single-question records, 207 topic domains | MIT | Topic breadth the four workflows lack. One question per record makes translation simpler. | Deduplicate. Sample 3–5k, balanced across choice, score and yes/no. Confirm no overlap with the typed-decisions test split. |

**What a typed-decisions case looks like** (from its dataset card): one `state` and five `questions`, which together are exactly the body of a System One request (`POST /v1/systemone`), plus `gold` with the full distribution for every question. Every option carries a written description in `criteria`, and those descriptions are model input, so they are translated. A state is free text where the artefact is textual and structured (JSON) where it is structured; only text values are translated. `factors` and `label_agreement` describe how a case was built, are not model input, and are not translated. Parquet files, one config per workflow plus `all`.

**Field rules, decided before any translation**

- Translate: natural-language text in states, question text, option descriptions.
- Keep byte-identical: JSON keys, option IDs the code uses, invoice and ticket numbers, emails, URLs, log lines, tool names, error messages, code.
- Numbers, currencies and dates stay as in the source in v1. Localization (Toman, Jalali) is a separate v2 with labels recomputed by code.

**Glossary and register**

- A two-column English→Persian glossary for recurring workflow terms (refund, invoice, ticket, incident, escalation). It also records which technical terms stay in English.
- A register rule per field type: customer messages colloquial; invoices, incident reports and system text formal.
- Both go into the translation prompt and the dataset card.

**Code-labeled Persian data (generated, not translated)**

These teach skills multilingual models are weak at. Code computes every label, so labels are exact, and no GPU is needed.

| Generator | Question types | Example question |
| --- | --- | --- |
| Jalali dates | noul, choice | Is 1405/07/15 before 1405/08/02? Which weekday is it? |
| Digit forms | noul, score | The same amount written in Persian, Arabic-Indic and Latin digits |
| Toman and Rial | noul, choice | Is 12 million Rial more than 1 million Toman? |
| Business hours | noul | Is the shop open at 21:30 on Friday? |
| Iranian formats | noul | Is this a valid mobile number? A valid 10-digit postal code? |

Preparation: write 5–10 Persian templates per generator, colloquial and formal. Keep 2 templates per generator for testing only. Fix the random seed and target 3–5k items in total.

**Minimal pairs.** Every generated item comes with a partner that differs in one fact and flips or changes the answer: the two dates swapped, Rial written instead of Toman, one digit of a postal code changed. Code computes both answers. A pair counts as correct only when both halves are, which separates a model that reads the fact from one that guesses from the wording.

**Raw forms.** Generated text keeps its digit forms and spelling variants as written (Persian, Arabic-Indic and Latin digits; ٫ and ٬ separators). It is never passed through the normalizer before publishing, or the suite would stop testing other models on exactly these variations.

**Task families.** Every record names its task family (for example `workflow`, `topics`, `intent`, `entailment`, `reading`, `sentiment`, `paraphrase`, `skill-dates`, `skill-currency`), so results can be reported for families seen in training vs unseen ones (3.2).

**Pilot first.** Take 50 typed-decisions cases through all of 1.2 before the full run.

### 1.2 Translation

| Candidate | Runs on | Why | Watch out for |
| --- | --- | --- | --- |
| [TranslateGemma](https://arxiv.org/pdf/2601.09012) 12B (4-bit) or 4B (fp16) | Colab T4 | Free, open, built for translation; Persian is among its 55 evaluated languages. | Translation-only prompt, so fields go one at a time. Tuned for inputs of about 2K tokens. Gemma 3 models have had fp16 overflow problems on T4-class GPUs. **Gated on Hugging Face** (found on Colab): accept the Gemma license on the model page and give Colab a read token (secret `HF_TOKEN`). Before publishing translations, read the Gemma terms on generated outputs (a dataset made with Gemma may count as its derivative) and confirm that the Apache-2.0 plan for typed-decisions-fa still holds. |
| [Gemma 3](https://huggingface.co/google/gemma-3-12b-it) 12B (4-bit) or 4B (bf16), prompted to translate | Colab T4 | **Chosen in the pilot (Sep 30).** 12B had the best free-text accuracy and no meaning changes; 4B is effectively tied and is the fallback. See [translation-eval-plan.md](translation-eval-plan.md). | Wraps most output in stray backticks, so strip them after translating. 12B mistranslates the risk-scale labels (Benign, Low, Moderate, High), so fix those by hand. Gemma terms apply, the same as for TranslateGemma. |
| A frontier model through an API | Cloud | Best Persian quality. Takes a whole case as JSON with rules (keep keys, register). The dataset is small, so cost is low. | Payment and access. Record the exact model for the card. **Not available**: the owner has no budget for an API. |
| Qwen3-8B (4-bit) | Colab T4 | The checker, in a separate pass: meaning comparison and consistency. Standard architecture, safe on a T4. | Weaker translator than the two above; use it to judge, not to translate. **Trial on the T4 (Sep 30):** it runs in 4-bit and follows the glossary, but its Persian changed meaning ("expired" became "valid", "irreversible" became "reversible", "benign" became nonsense) and it put Cyrillic letters inside a Persian word. Confirmed as a checker and glossary-following draft at most, not as the translator. |
| NLLB-200 | — | Not recommended. | Non-commercial weights cloud the license of a dataset meant to be Apache-2.0. |

1. **Pilot, 50 cases.** Translate with two candidates, review both blind, and keep the translator and prompt that win. **Done (Sep 30):** five candidates were rated blind on 50 test records from one workflow. Claude Cowork running Claude Opus 5.5 did the rating, as a single LLM rater; no human rated. Gemma 3 12B (4-bit) won. Before the full run, spot-check its free text from the other workflows (customer threads, security alerts), because the pilot sample did not cover them. Method, scores and caveats: [translation-eval-plan.md](translation-eval-plan.md).
2. **Full run.** All 1,600 typed-decisions cases plus the synthetic-typed-decisions sample. Save each finished case to Drive at once, so a dropped session resumes where it stopped.
3. **Automatic checks on every case:**
   - Keep-fields are byte-identical to the source.
   - Every number, ID and amount survives (convert Persian digits to Latin, then compare).
   - Output is Persian, not empty, not looping, with no extreme length ratio.
   - Glossary terms match between a state and its options.
4. **Meaning check.** The checker compares English and Persian for each case. It flags any change in meaning, urgency, negation, or who did what.
5. **Human review.**
   - **Test split: all 400 cases**, English and Persian side by side, for meaning, negation, urgency and who did what. The dataset card can then say every benchmark case was reviewed, and reports how many needed a fix. At 2–4 minutes a case this is roughly 15–25 hours, the largest single cost of M1.
   - **Train split:** every flagged case plus 100–150 random unflagged ones. The error rate in the random sample goes into the dataset card.
   - Fix or drop cases; never change the gold.
6. **Orthographic cleanup only.** Fix translator noise: Persian ی/ک instead of Arabic ي/ك, the zero-width non-joiner (نیم‌فاصله) where it belongs, invisible marks and stray spaces. Digits stay as in the source. The full normalizer, including one digit policy, is part of the reference model's input pipeline at training and inference; it is not applied to the published data.

Output: typed-decisions-fa (train and test, original gold unchanged) and a Persian sample of synthetic-typed-decisions.

### 1.3 Packing

**One record schema** for translated, native and synthetic data alike:

- `id`, plus `source_id` (the original case, identical across languages)
- `source` dataset with pinned revision, and `license`
- `split`
- `origin`: translated, native or synthetic
- `state_lang` and `question_lang`
- `state`, `question_type` (choice, score or noul), `question_text`, `options` (ID and text)
- `gold` as a probability distribution (one-hot where the source has hard labels)
- `task_family` (1.1)
- Quality flags: automatic checks passed, meaning-check flag, human-reviewed

**Publish typed-decisions-fa on Hugging Face**

- Same splits, labels and case IDs as the original.
- Dataset card: source revision, translator and checker models, how the translator was chosen (a blind LLM rating by Claude Opus 5.5 via Cowork, not a human review), prompt summary, glossary, the full test-split review and its fix count, the train review sample size and measured error rate, known limits (translationese, unlocalized numbers, labels from a teacher model), Apache-2.0 attribution. Use the [Russian](https://huggingface.co/datasets/yyhlm/typed-decisions-ru) and [Japanese](https://huggingface.co/datasets/GeneLab/typed-decisions-ja) cards as models.
- **Provenance statement** on every published dataset card: which parts involved a model and which did not. Translations come from a translator model and were checked by a checker model; typed-decisions gold comes from its source's teacher model (roughly 4B-class); the synthetic-typed-decisions sample is as generated by its source; the skills suite involves no model at all (code and templates). Readers comparing with DibaOne X1, which uses no third-party model in its data, can then see the difference exactly.
- Versions: v0.1 is the pilot, v1.0 the full reviewed set.

**Keep generators reproducible.** Store seeds and template files with the synthetic data, so anyone can regenerate it.

## Part 2 — Implementation

Part 2 merges native Persian data with Part 1's output into one training mix, trains an encoder and a small decoder on free Colab, and keeps every log and prediction needed to judge them.

### 2.1 Data preparation (training mix)

**Native Persian sources**

| Dataset | Converts to | Size | License | Role | Why | Preparation needed |
| --- | --- | --- | --- | --- | --- | --- |
| [MASSIVE](https://github.com/alexa/massive) (fa-IR) | choice: 60 intents or 18 scenarios | \~11.5k train, 2k dev, 3k test | CC BY 4.0 | Train and test | Voice-assistant commands, close to the STT→intent project. Many options per question teach the model to read option lists. | Write a short Persian description for each intent and scenario (labels are English codes such as `alarm_set`). Sample option subsets. |
| [FarsTail](https://github.com/dml-qom/FarsTail) | choice (3-way entailment) or noul | 10,367 | Apache-2.0 | Train and test | Human-made reasoning over two texts, cleanly licensed. | Download from GitHub, map labels, write templates. |
| [PersianQA](https://github.com/sajjjadayobi/PersianQA) | noul: does the passage answer the question? | 9,000+ | GPL-3.0 (the repository's LICENSE; DibaOne M3's card cites CC BY-NC-SA 4.0 from a GitHub release) | **Test only** | Its unanswerable questions give natural "no" cases. Either license would put a permissive model license in question, so it stays out of training and becomes one more held-out task family. | Pair questions with wrong passages for extra negatives. Truncate passages without cutting the answer. Publish the converter, not converted data. |
| [ParsiNLU](https://huggingface.co/persiannlp) (entailment, paraphrase, sentiment, multiple-choice QA) | all three types | 1.3k–17.5k per task | CC BY-NC-SA 4.0 | **Test only** | Tasks the model never trains on. Keeping them out of training also keeps the model's license clean. | Convert to records; never mix into training. |
| [Belebele](https://huggingface.co/datasets/facebook/belebele) (Persian) | choice (4-way reading comprehension) | 900 | CC BY-SA 4.0 | **Test only** | Parallel across 115 languages, so results compare with published ones. | Convert; check passage lengths. |
| Own STT→intent transcripts | choice | yours | yours | **Test only** | Noisy speech-recognition text, the closest thing to real use. | Remove anything private; label with intent options. |
| [Khayyam / PersianMMLU](https://github.com/raia-center/khayyam-challenge) | — | — | CC BY-ND, academic only | **Excluded** | Its license forbids derivative benchmarks. | — |

Licenses checked against each repository's LICENSE or NOTICE file (September 2026). Still to confirm: that MASSIVE's release includes the `fa-IR` locale.

**Training sources**, all compatible with a permissively licensed model: MASSIVE (CC BY 4.0, attribution in the model card), FarsTail (Apache-2.0), and Part 1's data (Apache-2.0 and MIT sources, and the project's own CC0-1.0 skills data). Everything else is test-only or excluded.

**From Part 1:** typed-decisions-fa train split (soft labels), the Persian synthetic-typed-decisions sample, and the code-labeled Persian data (training templates only).

**English slice:** the original English typed-decisions train split, plus some records with an English question over a Persian state. This keeps the model usable the way a foreign client would use it.

**Conversion steps**

1. **Templates.** 5–10 question phrasings per task, mostly Persian, some English. Hold 2 per task back for testing.
2. **Options.** Human-readable option text, shuffled every time, sometimes renamed. Sometimes add "none of these" and remove the correct option, so "none" becomes right.
3. **Gold.** Soft distributions where the source has them, one-hot elsewhere.
4. **Normalize** with the reference model's full normalizer (orthography and digits). This happens when building the training mix and at inference, never in published test data.
5. **Length check.** Measure token lengths with each candidate's tokenizer; Persian often costs more tokens. Max length 256, or 384 for QA passages.
6. **Balance.** No single source above about 30% of training decisions; upsample small sources.
7. **Splits.** Train, validation (early stopping), calibration (temperature scaling only), test. Split by `source_id`, so a case and its translation always land on the same side.
8. **Leakage check.** Remove near-duplicates across splits (MASSIVE has many paraphrases). Nothing from any test split, in any language, enters training.

Target size: 30–60k training decisions.

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

**Controlled and uncontrolled comparisons.** Laya-multilingual and the main model share the mmBERT-base backbone, so comparing them (and runs 6 and 7) isolates decision training and Persian data. DibaOne (XLM-R family), Qwen and the others differ in backbone, data, objective, option handling and calibration at once. Their comparisons are empirical ("model A scored X, model B scored Y on this benchmark") and are never read as evidence that one backbone suits Persian better.

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

**Thresholds, written down before any baseline runs.** Once the numbers are visible it is easy to argue for the outcome already wanted. Unlike the success criteria in 3.5, which are set after the baselines, these decide how time is spent, so they are fixed first. The values below are placeholders to replace with the project owner's own before M2 starts.

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
| **A. Large gap** | Large language gap, weak held-out scores, skills near chance | The full run plan (runs 0–6) | The model, with the dataset as evidence |
| **B. Middle** | A moderate gap, or a large gap in only some signals (often the skills) | Runs 0–3 and 6; drop the challenger (4) and run 5 | The dataset and the model together, framed around where the gains are |
| **C. Small gap** | The reference loses little in Persian and does reasonably on held-out tasks | At most a light Persian fine-tune of Laya-multilingual (run 6), mostly on the skills data | typed-decisions-fa, the Persian benchmark, the skills data, and a report comparing open decision models on Persian |

Outcome C is not a failure: a report on how well open decision models handle Persian is useful to every model author in this space, and it still ships a model.

## Part 3 — Testing and evaluation

The model is judged on Persian it never trained on, on how honest its confidence is, and on how much it loses against English, always against baselines on the same items.

### 3.1 Test sets

| Test set | Question it answers | Source |
| --- | --- | --- |
| In-task native | Did it learn the trained tasks? | MASSIVE-fa test (20-option questions, fixed seed), FarsTail test |
| Held-out native tasks | Is it a decision model, or a classifier of the tasks it saw? | ParsiNLU (4 tasks), Belebele-fa, PersianQA (in-domain for DibaOne M3, which trained on ParsiNLU and PersianQA) |
| Parallel language gap | How much does the same native item lose in Persian? | MASSIVE `en-US`/`fa-IR` and Belebele `eng_Latn`/`pes_Arab` test items |
| Held-out templates | Did it learn the task, or memorize the phrasing? | Test-only templates from 1.1 and 2.1 |
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
- **The primary controlled comparison:** the main model (run 2) against Laya-multilingual, same backbone; and run 6 against run 7, same start with and without Persian data.
- **Mode is stated with every typed-decisions result**, as the leaderboard asks: *specialist* (trained on typed-decisions `train`, as this project's models are) or *general* (zero-shot, as Laya, DibaOne and prompted Qwen are). The gap between the modes is the price of generality, not a quality ranking; held-out tasks measure generality.
- **Request shape is recorded:** whole case in one request (the leaderboard's shape) or one question per request. It changes results (Jev's yes/no accuracy was 0.843 per question and 0.788 alongside the others), so compared rows use the same shape.
- Run 2 vs run 3 (with vs without translated data) and run 2 vs run 4 (encoder vs decoder).
- Before vs after temperature scaling.
- 95% confidence intervals by bootstrap, with paired comparisons on the same items. A difference inside the interval is reported as a tie.

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

**Resource criteria**, which do not depend on the model:

- [ ] Every one of the 400 typed-decisions-fa test cases has been human-reviewed; the number of fixed and dropped cases is published.
- [ ] The harness has scored at least three models this project did not train (for example Laya-multilingual, DibaOne X1 and prompted Qwen3.5-0.8B), with their raw predictions published.
- [ ] The skills suite regenerates identically from the published seeds and templates.
- [ ] Every training source's license allows a permissively licensed model, and the model card lists them.

**Model criteria.** Set exact thresholds after the baselines and before runs 1–4, so results can't move the goalposts. (The decision gate's thresholds in 2.4 are different: they are fixed before the baselines.)

- [ ] Fine-tuned mmBERT-base beats Laya-multilingual on held-out native Persian tasks, outside the confidence interval.
- [ ] ECE after temperature scaling is lower than every baseline's, each baseline also recalibrated on the same calibration split.
- [ ] The Persian–English gap on the parallel MASSIVE and Belebele items is smaller than Laya-multilingual's. (Not on typed-decisions, where Laya is near chance zero-shot in both languages, so its gap there says nothing.)
- [ ] CPU latency is low enough for a voice assistant (threshold set after measuring baselines).

### 3.6 Reporting

- One headline table, a reliability diagram, an accuracy-vs-coverage curve and a language-gap chart.
- Raw prediction files published with the report.
- A model-agnostic evaluation harness (for example `kodoom evaluate --model <adapter> --benchmark typed-decisions-fa --out results/<model>/`) whose input is the System One request (a state plus typed questions, as in typed-decisions and Laya), with adapters for per-question models such as DibaOne, and that reads the same request and response format as the community benchmarks ([typed-decision-bench](https://github.com/kyr0/typed-decision-bench), [open-system-one](https://github.com/zhlei07/open-system-one)), so others can run their models on the Persian sets. Its outputs are raw predictions (ID, gold, predicted distribution, temperature, latency), from which every metric is recomputed.
- Model card: training data with licenses, intended use, known limits, and the translation error rate from Part 1.

**Making adoption easy.** For downloads, easy use matters more than a few accuracy points.

- **Recalibration recipe.** A documented step, with a script, that fits a new temperature on 100–300 of a user's own labels, and writes it as a `calibration.json` in typed-decision-bench's format (one temperature per question type, a global fallback), so engines that read that format pick it up. Temperature scaling already exists from 2.2; this makes it usable on someone else's data, where the shipped calibration will not hold. The model card says plainly that the top answer transfers better than the probability.
- **CPU-friendly export.** An exported model (for example ONNX, optionally int8) with the latency measured on it, so the CPU latency in 3.2 is reproducible by users.
- **Community request/response format.** The evaluation harness above already reads it; a small adapter lets the model answer in the same format.

## Failure modes

The likeliest failure is a classifier in disguise; the costliest are silent label errors from translation and leakage between splits. Checks already in Parts 1–3 catch all three early.

| Failure mode | Likelihood | Impact | Early signal | Mitigation |
| --- | --- | --- | --- | --- |
| A classifier in disguise: the model learns templates and trained tasks, not decisions | High | High | Strong in-task scores, near chance on held-out tasks and templates | Template variety, option shuffling and sampling, held-out tasks and templates from day one |
| Google Drive fills up during a run (free 15 GB, shared with Gmail and Photos) | Medium | High | Save fails; a half-written checkpoint; no space to resume | Storage budget rules: no base models on Drive, at most two checkpoints, best as fp16 weights only, finished artifacts pushed to Hugging Face; free-space check before every checkpoint |
| Free Colab GPU unavailable, capped, or sessions drop | High | Medium | Runs cut off; no GPU assigned | Runs sized to one session, checkpoints on Drive, resumable training; Kaggle notebooks as a backup if reachable |
| Label ceiling: typed-decisions labels come from an unnamed \~4B teacher model | High | Medium | Scores plateau; the typed-decisions card reports a teacher self-agreement ceiling of 0.735 (and warns that scores well above it mean learning the teacher's quirks) | Report results against the ceiling; weigh human-labeled native sets more |
| Scope creep on a solo side project | High | Medium | A milestone slips twice | Each milestone is publishable alone; M1 by itself is a valid finish |
| Existing open models already handle Persian well, so a new model adds little | Medium | Medium | Small language gap and reasonable held-out scores in the M2 baselines | Decision gate (2.4): lead with the dataset, benchmark and skills data; train little |
| Jev API not reachable | High | Low | Waitlist or payment fails | Nothing depends on it; cite published third-party numbers, labeled as such |
| Translation changes meaning, so gold labels become silently wrong | Medium | High | Meaning-check flags; errors in the review sample | Automatic checks, checker model, human sample, published error rate, bad cases dropped |
| Leakage between splits (a case or its translation on both sides) | Medium | High | Test scores far above baselines, too good to be true | Split by source\_id across languages, deduplicate, freeze test sets before training |
| Translationese: the model fits translated Persian, not native text | Medium | Medium | Run 3 (no translation) beats run 2 on native tests | Colloquial register rule, native results reported separately, the run 2 vs 3 ablation |
| Overconfident probabilities | Medium | Medium | High ECE; wrong answers at 0.99 | Soft labels, log-loss early stopping, temperature scaling |
| TranslateGemma misbehaves in fp16 on the T4 | Medium | Medium | Empty, garbled or looping output in the pilot | The pilot catches it; patched loader, the 4B model, or an API translator |
| Encoder weak on many-option questions (MASSIVE's 60 intents) | Medium | Medium | Intent accuracy far below scenario accuracy | Option sampling, scenario then intent in two steps, compare with the decoder |
| Persian tokenization: long inputs truncated, answers cut, slow steps | Medium | Medium | Many records hit max length | Measure lengths in 2.1, 384 tokens for QA, truncate around the answer |
| Checker model unreliable (too many or too few flags) | Medium | Medium | Human review disagrees with its flags | Tune it on the pilot's human review; flags prioritize review, never filter alone |
| Code passes on the laptop but fails on Colab (fp16, GPU memory, CUDA-only packages) | Medium | Medium | Errors or NaN loss early in a Colab run | `colab-preflight` profile before every long run; Colab-only extras kept separate; same code path in all profiles |
| Qwen3.5-0.8B won't train on a T4 in fp16 | Medium | Low | Kernel errors, NaN loss, very slow steps | Switch to Qwen3-0.6B |
| Someone publishes a Persian version first, or the Jev trend fades | Medium | Low | New Persian decision datasets appear | Publish typed-decisions-fa early; the evaluation work stays useful either way |
| License contamination (non-commercial or GPL data in a release) | Low | High | NC or GPL licenses show up in training records | License field on every record, checked by code; ParsiNLU and PersianQA test-only; Khayyam excluded; converters published instead of GPL-derived data |
| A baseline was trained on a test set (DibaOne M3 on ParsiNLU and PersianQA), inflating its scores | Medium | High | A baseline far ahead on one set only | Check every baseline's published training sources; mark in-domain scores and exclude them from held-out comparisons |
| Normalized benchmark data hides the skills it should test | Medium | High | Published test files contain only Latin digits and one spelling | Benchmark stored raw (Principles); orthographic cleanup only for translations; the full normalizer lives in the model's input pipeline |

Likelihood and impact are judgment calls for a solo project on free Colab; revisit them after the M1 pilot.

## Milestones

&#91;embedded content: milestones · 4 phases, 3 gates\]

Stopping at any gate still leaves something published; M1 alone is a complete, useful result.

Changes since the milestone diagram was drawn (the diagram itself still needs updating):

- **Gate 2** now also applies the decision gate (2.4): its thresholds are written down before the baselines run, and its outcome (A, B or C) is recorded.
- **M3's scope depends on the gate outcome**: the full run plan, a reduced one, or a light fine-tune of Laya-multilingual.
- **M1 now includes the full review of the 400-case test split** (1.2 step 5), and the skills suite with minimal pairs.

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
| [Gemma 3](https://huggingface.co/google/gemma-3-12b-it) | Prompted translator (chosen in the pilot) | Gemma terms | 1.2 |
| [TypeSafe Jev models page](https://docs.typesafe.ai/models) | Jev customization and language support | — | Overview |
| [typed-decision-bench](https://github.com/kyr0/typed-decision-bench) | Community benchmark and format | see repo | 3.6 |
| [open-system-one](https://github.com/zhlei07/open-system-one) | Community benchmark vs Jev | see repo | 3.6 |

## Version history

| Version | Date | Changes |
| --- | --- | --- |
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
