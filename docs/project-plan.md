# Persian Typed-Decision Model: Project Plan

Version 4 · Sep 30, 2026 · @Shah

## Overview

The goal is an open, Persian-capable decision model that speaks Jev's interface: choice, score and yes/no (noul) questions answered with calibrated probabilities. Jev itself can't be fine-tuned (closed weights; TypeSafe customizes it only through the request), so this project trains an open model and builds the Persian data and tests that prove it works.

**Deliverables**, each publishable on its own:

1. **typed-decisions-fa**: a reviewed Persian translation of LocalLLaMA/typed-decisions, same splits and labels (Part 1).
2. **Persian training mix**: converters and templates that turn native Persian datasets and synthetic data into decision records (Part 2).
3. **Models**: a fine-tuned mmBERT-base (main) and Qwen3.5-0.8B (challenger), each with a model card (Part 2).
4. **Evaluation report**: accuracy, calibration, English–Persian gap and CPU latency, with raw predictions (Part 3).

**Constraints**

- Training runs on the free Colab tier: a T4 GPU (16 GB), no bf16, no FlashAttention 2, sessions that drop.
- Data preparation, generation and small-model evaluation run on the laptop CPU.
- The laptop is modest: Intel i3-1005G1 (2 cores, 4 threads), about 12 GB RAM, and an NVIDIA MX110 (2 GB) that is not used. It proves the code works; it does not train the real models (see Environments).
- The Jev API may not be reachable; nothing in the plan depends on it.

**Principles**

- The data is the project. Laya's authors report their base checkpoints sit near chance on typed decisions until fine-tuned.
- Build a decision model, not a classifier. Options arrive in the request, and whole tasks stay out of training to prove it.
- Report native Persian results separately from translated data, always.
- Publish early. Part 1's dataset is the first public output.
- Develop on the laptop, run on Colab. Nothing reaches Colab until it has run end to end on the laptop.

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

## Part 1 — Data augmentation

Part 1 turns English typed-decision data and code-generated examples into Persian data, and publishes typed-decisions-fa as the project's first public output.

### 1.1 Data preparation

**Sources to translate**

| Dataset | Size | License | Why this one | Preparation needed |
| --- | --- | --- | --- | --- |
| [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | 1,200 train cases (6,000 decisions); 400 test cases (2,000) | Apache-2.0 | Already in Jev's shape. Gold labels are probability distributions, which suits calibration. Covers four business workflows Persian data lacks (agent traces, customer service, invoices, security incidents). Russian and Japanese versions exist, so a Persian one joins a set people already run. | Pin the dataset revision. Map the schema. Mark every text field as translate or keep. Flag cases with money, dates or numeric thresholds. Keep the original splits and case IDs. |
| [helmo/synthetic-typed-decisions](https://huggingface.co/datasets/helmo/synthetic-typed-decisions) | 9,879 single-question records, 207 topic domains | MIT | Topic breadth the four workflows lack. One question per record makes translation simpler. | Deduplicate. Sample 3–5k, balanced across choice, score and yes/no. Confirm no overlap with the typed-decisions test split. |

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

**Pilot first.** Take 50 typed-decisions cases through all of 1.2 before the full run.

### 1.2 Translation

| Candidate | Runs on | Why | Watch out for |
| --- | --- | --- | --- |
| [TranslateGemma](https://arxiv.org/pdf/2601.09012) 12B (4-bit) or 4B (fp16) | Colab T4 | Free, open, built for translation; Persian is among its 55 evaluated languages. | Translation-only prompt, so fields go one at a time. Tuned for inputs of about 2K tokens. Gemma 3 models have had fp16 overflow problems on T4-class GPUs. |
| A frontier model through an API | Cloud | Best Persian quality. Takes a whole case as JSON with rules (keep keys, register). The dataset is small, so cost is low. | Payment and access. Record the exact model for the card. |
| Qwen3-8B (4-bit) | Colab T4 | The checker, in a separate pass: meaning comparison and consistency. Standard architecture, safe on a T4. | Weaker translator than the two above; use it to judge, not to translate. |
| NLLB-200 | — | Not recommended. | Non-commercial weights cloud the license of a dataset meant to be Apache-2.0. |

1. **Pilot, 50 cases.** Translate with two candidates, review both blind, and keep the translator and prompt that win.
2. **Full run.** All 1,600 typed-decisions cases plus the synthetic-typed-decisions sample. Save each finished case to Drive at once, so a dropped session resumes where it stopped.
3. **Automatic checks on every case:**
   - Keep-fields are byte-identical to the source.
   - Every number, ID and amount survives (convert Persian digits to Latin, then compare).
   - Output is Persian, not empty, not looping, with no extreme length ratio.
   - Glossary terms match between a state and its options.
4. **Meaning check.** The checker compares English and Persian for each case. It flags any change in meaning, urgency, negation, or who did what.
5. **Human review.** Read every flagged case plus 100–150 random unflagged ones. Fix or drop them. The error rate in the random sample goes into the dataset card.
6. **Normalize.** Persian ی/ک instead of Arabic ي/ك, the zero-width non-joiner (نیم‌فاصله) where it belongs, one digit policy. The same normalizer runs later at training and inference.

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
- Quality flags: automatic checks passed, meaning-check flag, human-reviewed

**Publish typed-decisions-fa on Hugging Face**

- Same splits, labels and case IDs as the original.
- Dataset card: source revision, translator and checker models, prompt summary, glossary, review sample size and measured error rate, known limits (translationese, unlocalized numbers, labels from a teacher model), Apache-2.0 attribution. Use the [Russian](https://huggingface.co/datasets/yyhlm/typed-decisions-ru) and [Japanese](https://huggingface.co/datasets/GeneLab/typed-decisions-ja) cards as models.
- Versions: v0.1 is the pilot, v1.0 the full reviewed set.

**Keep generators reproducible.** Store seeds and template files with the synthetic data, so anyone can regenerate it.

## Part 2 — Implementation

Part 2 merges native Persian data with Part 1's output into one training mix, trains an encoder and a small decoder on free Colab, and keeps every log and prediction needed to judge them.

### 2.1 Data preparation (training mix)

**Native Persian sources**

| Dataset | Converts to | Size | License | Role | Why | Preparation needed |
| --- | --- | --- | --- | --- | --- | --- |
| [MASSIVE](https://github.com/alexa/massive) (fa-IR) | choice: 60 intents or 18 scenarios | \~11.5k train, 2k dev, 3k test | CC BY 4.0\* | Train and test | Voice-assistant commands, close to the STT→intent project. Many options per question teach the model to read option lists. | Write a short Persian description for each intent and scenario (labels are English codes such as `alarm_set`). Sample option subsets. |
| [FarsTail](https://github.com/dml-qom/FarsTail) | choice (3-way entailment) or noul | 10,367 | Apache-2.0 | Train and test | Human-made reasoning over two texts, cleanly licensed. | Download from GitHub, map labels, write templates. |
| [PersianQA](https://github.com/sajjjadayobi/PersianQA) | noul: does the passage answer the question? | 9,000+ | GPL-3.0 | Train and test | Its unanswerable questions give natural "no" cases. | Pair questions with wrong passages for extra negatives. Truncate passages without cutting the answer. Publish the converter, not converted data. |
| [ParsiNLU](https://huggingface.co/persiannlp) (entailment, paraphrase, sentiment, multiple-choice QA) | all three types | 1.3k–17.5k per task | CC BY-NC-SA 4.0 | **Test only** | Tasks the model never trains on. Keeping them out of training also keeps the model's license clean. | Convert to records; never mix into training. |
| [Belebele](https://huggingface.co/datasets/facebook/belebele) (Persian) | choice (4-way reading comprehension) | 900 | CC BY-SA 4.0\* | **Test only** | Parallel across 115 languages, so results compare with published ones. | Convert; check passage lengths. |
| Own STT→intent transcripts | choice | yours | yours | **Test only** | Noisy speech-recognition text, the closest thing to real use. | Remove anything private; label with intent options. |
| [Khayyam / PersianMMLU](https://github.com/raia-center/khayyam-challenge) | — | — | CC BY-ND, academic only | **Excluded** | Its license forbids derivative benchmarks. | — |

\* License from memory; confirm on the dataset page.

**From Part 1:** typed-decisions-fa train split (soft labels), the Persian synthetic-typed-decisions sample, and the code-labeled Persian data (training templates only).

**English slice:** the original English typed-decisions train split, plus some records with an English question over a Persian state. This keeps the model usable the way a foreign client would use it.

**Conversion steps**

1. **Templates.** 5–10 question phrasings per task, mostly Persian, some English. Hold 2 per task back for testing.
2. **Options.** Human-readable option text, shuffled every time, sometimes renamed. Sometimes add "none of these" and remove the correct option, so "none" becomes right.
3. **Gold.** Soft distributions where the source has them, one-hot elsewhere.
4. **Normalize** exactly as in Part 1.
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
- [Laya](https://github.com/NandhaKishorM/laya)-multilingual, zero-shot: the open reference. If it does not load or run on the laptop or a T4, prompted Qwen3.5-0.8B takes its place as the reference, including at the decision gate (2.4).
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
| 6 | Laya-multilingual as the starting point (warm start) | Full mix | Does starting from a decision model beat starting from plain mmBERT-base? Compare with run 2. Confirm Laya's license on its repo first. |

Which runs actually happen depends on the decision gate (2.4).

**Settings on the Colab T4**

- fp16 mixed precision; SDPA or eager attention (a T4 has no FlashAttention 2).
- Max length 256 (384 for QA); gradient accumulation for a larger effective batch.
- Encoder: full fine-tune, small learning rate with warmup, 2–4 epochs. Decoder: LoRA, rank 16–32.
- Loss: cross-entropy against the gold distribution. It is a proper scoring rule, so it rewards honest confidence.
- Early stopping on validation log-loss, not accuracy.
- After training: temperature scaling on the calibration split, one value per model (optionally per question type).
- Fixed seeds. Checkpoint to Drive every few hundred steps; every run must resume after a dropped session.
- Budget: one run fits one Colab session (1–2 hours). If it doesn't, shrink the data or model rather than stretch the session.

### 2.3 Collecting logs and training data

Keep enough that any number in the final report can be recomputed without retraining.

- **Run registry:** one row per run with run ID, date, model and revision, dataset version, key settings, headline metrics, and a one-line decision (keep, or drop and why).
- **Per-run config:** model revision hash, dataset version hash, seeds, hyperparameters, max length, option-sampling settings, library versions, GPU type.
- **Curves:** train loss, validation loss, validation accuracy and ECE per step or epoch; runtime and peak GPU memory.
- **Where:** plain files on Google Drive (CSV or JSON, or TensorBoard / MLflow file logs). Avoid depending on a hosted tracker that may not be reachable.
- **Predictions, not just metrics:** for every evaluation, save per item the ID, gold distribution, predicted distribution, temperature used and latency. Every later metric, chart and error analysis comes from these files, and they can be published as raw results.
- **Error log:** after each main run, read 50 wrong answers and tag them: translation artifact, ambiguous options, truncated context, number or date reasoning, colloquial text, label noise.
- **Artifacts kept:** best checkpoint, its tokenizer, its temperature value, the exact training-mix version, and a draft model card.

### 2.4 Decision gate: does Persian need a new model?

After the M2 baselines and before M3's training runs, one planned decision: does Persian need a new decision model, or does it mostly need data and a benchmark? Every outcome still publishes something; the gate only decides where the time goes and what the release leads with.

**Reference model.** Laya-multilingual, zero-shot. It shares mmBERT-base with the main model, so its weaknesses in Persian are this model's reason to exist. If it cannot be run, prompted Qwen3.5-0.8B is the reference.

**Three signals**, all measured with no training:

1. **Language gap.** The reference's accuracy on the English typed-decisions test minus its accuracy on the same 2,000 decisions in Persian. Same items in both languages, so the difference is the cost of Persian alone.
2. **Held-out native tasks.** The reference on ParsiNLU and Belebele-fa, against the random and majority baselines.
3. **Persian skills.** The reference on the code-labeled test templates (Jalali dates, digit forms, Toman vs Rial, Iranian formats).

Calibration in Persian and the STT transcripts are measured and reported too, but they do not decide the outcome.

**Thresholds, written down before any baseline runs.** Once the numbers are visible it is easy to argue for the outcome already wanted. Unlike the success criteria in 3.5, which are set after the baselines, these decide how time is spent, so they are fixed first. The values below are placeholders to replace with the project owner's own before M2 starts.

| Signal | Large gap (counts toward A) | Small gap (counts toward C) |
| --- | --- | --- |
| Language gap | ≥ _8_ points | < _3_ points |
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
| In-task native | Did it learn the trained tasks? | MASSIVE-fa test, FarsTail test, PersianQA held-out part |
| Held-out native tasks | Is it a decision model, or a classifier of the tasks it saw? | ParsiNLU (4 tasks), Belebele-fa |
| Held-out templates | Did it learn the task, or memorize the phrasing? | Test-only templates from 1.1 and 2.1 |
| typed-decisions-fa test + English original | How much does the same case lose in Persian? | 400 cases, 2,000 decisions, in both languages |
| Persian skills | Dates, digits, currency, Iranian formats | Code-labeled data, test templates only |
| Real use | Does it hold up on noisy speech transcripts? | Own STT→intent data |

### 3.2 Metrics

- **Accuracy** (top-1), plus macro-F1 where classes are imbalanced.
- **Calibration:** ECE with fixed bins, Brier score, log-loss, and a reliability diagram.
- **Accuracy vs coverage:** if only answers above a confidence threshold are auto-accepted, what share is automated and how accurate is it? This is the number that matters for Jev-style gating.
- **Soft agreement** on typed-decisions: distance between predicted and gold distributions (total variation or Jensen–Shannon).
- **Score questions:** within-one accuracy and mean absolute error.
- **Language gap:** English minus Persian accuracy on the same typed-decisions cases.
- **Cost:** CPU latency per question (median and 95th percentile) and memory on the laptop, no GPU.

### 3.3 Comparisons

- Every fine-tuned run against every baseline from 2.2, on the same items.
- Run 2 vs run 3 (with vs without translated data) and run 2 vs run 4 (encoder vs decoder).
- Before vs after temperature scaling.
- 95% confidence intervals by bootstrap, with paired comparisons on the same items. A difference inside the interval is reported as a tie.

### 3.4 Robustness checks

- Shuffle option order: the answer should not change.
- Rename options, or add a plausible wrong option.
- Rewrite a formal state colloquially, and the reverse.
- Swap Persian and Latin digits.
- Ask in English about a Persian state.

### 3.5 Success criteria

Set exact thresholds after the baselines and before runs 1–4, so results can't move the goalposts. (The decision gate's thresholds in 2.4 are different: they are fixed before the baselines.)

- [ ] Fine-tuned mmBERT-base beats Laya-multilingual on held-out native Persian tasks, outside the confidence interval.
- [ ] ECE after temperature scaling is lower than every baseline's.
- [ ] The Persian–English gap on typed-decisions is smaller than Laya-multilingual's.
- [ ] CPU latency is low enough for a voice assistant (threshold set after measuring baselines).

### 3.6 Reporting

- One headline table, a reliability diagram, an accuracy-vs-coverage curve and a language-gap chart.
- Raw prediction files published with the report.
- An evaluation harness that reads the same request and response format as the community benchmarks ([typed-decision-bench](https://github.com/kyr0/typed-decision-bench), [open-system-one](https://github.com/zhlei07/open-system-one)), so others can run their models on the Persian sets.
- Model card: training data with licenses, intended use, known limits, and the translation error rate from Part 1.

**Making adoption easy.** For downloads, easy use matters more than a few accuracy points.

- **Recalibration recipe.** A documented step, with a script, that fits a new temperature on 100–300 of a user's own labels. Temperature scaling already exists from 2.2; this makes it usable on someone else's data, where the shipped calibration will not hold. The model card says plainly that the top answer transfers better than the probability.
- **CPU-friendly export.** An exported model (for example ONNX, optionally int8) with the latency measured on it, so the CPU latency in 3.2 is reproducible by users.
- **Community request/response format.** The evaluation harness above already reads it; a small adapter lets the model answer in the same format.

## Failure modes

The likeliest failure is a classifier in disguise; the costliest are silent label errors from translation and leakage between splits. Checks already in Parts 1–3 catch all three early.

| Failure mode | Likelihood | Impact | Early signal | Mitigation |
| --- | --- | --- | --- | --- |
| A classifier in disguise: the model learns templates and trained tasks, not decisions | High | High | Strong in-task scores, near chance on held-out tasks and templates | Template variety, option shuffling and sampling, held-out tasks and templates from day one |
| Free Colab GPU unavailable, capped, or sessions drop | High | Medium | Runs cut off; no GPU assigned | Runs sized to one session, checkpoints on Drive, resumable training; Kaggle notebooks as a backup if reachable |
| Label ceiling: typed-decisions labels come from an unnamed \~4B teacher model | High | Medium | Scores plateau; Laya reports a teacher self-agreement ceiling of about 0.735 | Report results against the ceiling; weigh human-labeled native sets more |
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
| License contamination (non-commercial or GPL data in a release) | Low | High | NC or GPL licenses show up in training records | License field on every record, ParsiNLU test-only, Khayyam excluded, converters published instead of GPL-derived data |

Likelihood and impact are judgment calls for a solo project on free Colab; revisit them after the M1 pilot.

## Milestones

&#91;embedded content: milestones · 4 phases, 3 gates\]

Stopping at any gate still leaves something published; M1 alone is a complete, useful result.

Changes since the milestone diagram was drawn (the diagram itself still needs updating):

- **Gate 2** now also applies the decision gate (2.4): its thresholds are written down before the baselines run, and its outcome (A, B or C) is recorded.
- **M3's scope depends on the gate outcome**: the full run plan, a reduced one, or a light fine-tune of Laya-multilingual.

## References

Licenses marked \* are from memory; confirm them on the page before use.

| Resource | Kind | License | Used in |
| --- | --- | --- | --- |
| [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | English decision dataset | Apache-2.0 | 1.1, 2.1, 3.1 |
| [helmo/synthetic-typed-decisions](https://huggingface.co/helmo/DecidaBERT-large) (described on the DecidaBERT card) | English decision dataset | MIT | 1.1 |
| [typed-decisions-ru](https://huggingface.co/datasets/yyhlm/typed-decisions-ru) | Russian translation, card model | Apache-2.0 | 1.3 |
| [typed-decisions-ja](https://huggingface.co/datasets/GeneLab/typed-decisions-ja) | Japanese translation, card model | Apache-2.0 | 1.3 |
| [MASSIVE](https://github.com/alexa/massive) | Intent dataset, fa-IR locale | CC BY 4.0\* | 2.1, 3.1 |
| [FarsTail](https://github.com/dml-qom/FarsTail) | Persian entailment | Apache-2.0 | 2.1, 3.1 |
| [PersianQA](https://github.com/sajjjadayobi/PersianQA) | Persian reading QA | GPL-3.0 | 2.1, 3.1 |
| [ParsiNLU](https://huggingface.co/persiannlp) | Persian NLU suite | CC BY-NC-SA 4.0 | 3.1 (test only) |
| [Belebele](https://huggingface.co/datasets/facebook/belebele) | Multilingual reading comprehension | CC BY-SA 4.0\* | 3.1 (test only) |
| [Khayyam / PersianMMLU](https://github.com/raia-center/khayyam-challenge) | Persian exam questions | CC BY-ND, academic only | Excluded |
| [mmBERT](https://github.com/jhu-clsp/mmBERT) | Multilingual encoder (small, base) | see repo | 2.2 |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Small multilingual decoder | Apache-2.0 | 2.2 |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Fallback decoder | Apache-2.0\* | 2.2 |
| [Persian-ModernBERT-base](https://huggingface.co/myrkur/Persian-ModernBert-base) | Persian encoder | see card | 2.2 |
| [Laya](https://github.com/NandhaKishorM/laya) | Open decision model, baseline | see repo | 2.2, 3.3 |
| [Dohnuts-0.1.0-0.8B](https://huggingface.co/PsiACE/Dohnuts-0.1.0-0.8B) | Decision model on Qwen3.5-0.8B, evidence | see card | 2.2 |
| [Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev) | Decision model on Qwen3-0.6B, evidence | see card | 2.2 |
| [TranslateGemma](https://arxiv.org/pdf/2601.09012) ([vLLM guide](https://docs.vllm.ai/projects/recipes/en/latest/Google/TranslateGemma.html)) | Translation model | Gemma terms | 1.2 |
| [TypeSafe Jev models page](https://docs.typesafe.ai/models) | Jev customization and language support | — | Overview |
| [typed-decision-bench](https://github.com/kyr0/typed-decision-bench) | Community benchmark and format | see repo | 3.6 |
| [open-system-one](https://github.com/zhlei07/open-system-one) | Community benchmark vs Jev | see repo | 3.6 |

## Version history

| Version | Date | Changes |
| --- | --- | --- |
| 4 | Sep 30, 2026 | Added the decision gate after the M2 baselines (2.4): three signals, thresholds fixed before the baselines (placeholders for now), outcomes A/B/C deciding M3's scope. Added run 6 (Laya-multilingual warm start), a fallback reference if Laya cannot run, adoption items in 3.6 (recalibration recipe, CPU export, request/response adapter), a failure mode, and notes on Gate 2 and M3. Ideas outside this plan moved to `docs/future-work.md`. |
| 3 | Sep 30, 2026 | Added "Environments": laptop development separated from Colab runs, with `dev`, `colab-preflight` and `colab` profiles, tiny random models, a 5-minute smoke run and the laptop's limits. Added the laptop to Constraints, a principle, and a failure mode. |
| 2 | Sep 30, 2026 | Plan as first added to the repository. |
