# Data Behind Persian Typed-Decision Work: Theirs and Ours

Sep 30, 2026 · compares the data used by earlier Persian-capable decision models with the data this project plans to use ([project plan](project-plan.md), v7).

**Historical comparison.** Its "ours only" claims describe that Sep 30 source
list, not the current community. The [Oct 8 review](precompute-project-review.md)
adds Decima's Persian resources, updates the contribution claim and separates
source licenses from translator-derived training eligibility. Use project-plan
v22 for present decisions.

**Where the facts come from.** Each earlier work is described only from its own published card or README:

- **DibaOne X1:** its model card, pasted by the project owner.
- **DibaOne M3:** its model card, printed to PDF by the owner.
- **Laya-multilingual:** its model card, printed to PDF by the owner.
- **laya-persian-benchmark:** its GitHub README.
- **LocalLLaMA/typed-decisions:** its dataset card, pasted by the owner.

"Not stated" means the card does not say, not that the answer is no. M3's card lists only its *main* non-commercial sources and points to a `NOTICE` file for the full list, which has not been read yet. So an M3 cell marked "?" is unknown, not absent.

## 1. The works at a glance

| | DibaOne M3 | DibaOne X1 | Laya-multilingual | laya-persian-benchmark | **Ours (planned)** |
| --- | --- | --- | --- | --- | --- |
| What it is | Persian-first decision model | Persian-first decision model | Multilingual decision model (100+ languages) | Persian diagnostic benchmark (evaluation only) | Persian dataset, skills suite, harness + reference models |
| Training size | 1,183,895 examples | 27,843 decisions | Not stated (card gives only "15,987 updates, 4 epochs") | — (no training) | 30–60k decisions (target) |
| Persian : English | 558,103 : 625,792 (47% fa) | "Split evenly" | Not stated | Persian content; instructions in fa and en | Mostly Persian, plus an English slice |
| Task families | 78 | 3 (dev path, tool call, Wiki Race) | Not stated | 1 (support routing), 8 Persian phenomena | About 10 in training (workflow, topics, intent, entailment, skill families), more held out |
| Question types trained | choice, noul, score | choice only | choice, noul, score | choice only (4 labels) | choice, noul, score |
| Where the data comes from | Public datasets converted + own synthetic generators | Own synthetic data + Wikipedia | Not stated | Hand-built synthetic messages | Translated typed-decisions, native Persian datasets converted, code-generated skills |
| LLM-generated or LLM-labeled data | Not stated | "No third-party language model was used to generate training data" | Not stated | Not stated | Yes: translation by a model, typed-decisions gold from a ~4B teacher, synthetic-typed-decisions |
| Training-data license | Mixed; 41% of rows non-commercial or unstated, so weights are CC BY-NC-SA 4.0 | CC0 (own) + CC BY-SA 4.0 (Wikipedia); weights Apache-2.0 | Not stated; weights Apache-2.0 | MIT (benchmark) | Only Apache-2.0, MIT, CC BY 4.0 in training (enforced by code); weights Apache-2.0 target |
| Training data published? | Not stated (sources listed in NOTICE) | Not stated | Not stated | Yes (the benchmark) | Yes: typed-decisions-fa and the skills generators |

## 2. Source by source

T = used for training, E = used for evaluation only, ? = not in the card's partial list (M3's `NOTICE` not yet read), — = not used / not stated.

| Data source | M3 | X1 | Laya-ml | laya-fa-bench | **Ours** | Shared? |
| --- | --- | --- | --- | --- | --- | --- |
| **ParsiNLU** (entailment, sentiment, paraphrase, multiple choice) | T | — | — | — | E (held out) | **M3 and ours**: M3 trains on it, we test on it |
| **PersianQA** | T | — | — | — | E (held out) | **M3 and ours**: same conflict |
| ANLI, RACE, Yelp, SciFact, customer-support tickets | T | — | — | — | — | M3 only |
| M3's own synthetic generators | T | — | — | — | — | M3 only |
| **Wikipedia** (fa and en) | ? | T (Wiki Race, 2,101) | — | — | — | X1 only |
| X1's own synthetic tool-call and dev-path requests | — | T (25,742) | — | — | — | X1 only |
| **MASSIVE** (intent, 51 languages) | ? | — | E (20-option, all 51 languages) | — | T (fa-IR) + E (fa/en parallel gap) | **Laya and ours** evaluate on it; we also train on it |
| XNLI (15 languages) | ? | — | E | — | — | Laya only |
| **typed-decisions** (English) | — | — | E (zero-shot: 0.342) | — | T + E (English slice, paired test) | **Laya and ours** |
| typed-decisions **translated to Persian** | — | — | — | — | T + E (typed-decisions-fa) | **Ours only** |
| helmo/synthetic-typed-decisions (translated sample) | — | — | — | — | T | Ours only |
| FarsTail | ? | — | — | — | T + E | Ours (M3 unknown) |
| Belebele (fa and en) | ? | — | — | — | E (held out, parallel gap) | Ours (M3 unknown) |
| Code-labeled Persian skills (Jalali dates, digits, Toman/Rial, hours, Iranian formats) | — | — | — | Partly: one family of "Persian vs Latin digits and dates" | T (train templates) + E (test templates, minimal pairs) | **Only partly shared** with laya-persian-benchmark |
| Own speech-to-text transcripts | — | — | — | — | E (held out) | Ours only |
| Native Persian evaluation sets | E: persian_eval_v1 (10,766), product_eval_v1 (8,822) | E: 3 frozen benchmarks (4,358, half fa) | — | E: 64 messages | E: native held-out sets above | Everyone builds their own; none is shared |

## 3. What is common across several works

| Common practice | Who does it | Notes |
| --- | --- | --- |
| Persian **and** English together | M3, X1, ours (Laya across 100+ languages) | Nobody trains Persian-only. M3 and X1 are about half and half; ours is mostly Persian. |
| **Synthetic data** as a main ingredient | M3, X1, laya-fa-bench, ours | Differs in *how*: M3 and X1 use their own generators; X1 rules out third-party LLMs; ours uses code (skills), a teacher model (typed-decisions gold) and a translator. |
| **Converting existing datasets** into decision questions | M3 (many), ours (MASSIVE, FarsTail, ParsiNLU, PersianQA, Belebele) | M3 converts sources into 78 families, all for training; we convert fewer and keep several for testing only. |
| **ParsiNLU and PersianQA** as the Persian NLU sources | M3, ours | M3 trains on them (hence its non-commercial license); we hold them out for testing. |
| **Held-out / frozen evaluation data**, checked against training for overlap | M3 (families disjoint from its rows), X1 (built by a separate team, frozen first, mechanical overlap check), ours (split by `source_id`, leakage check, test-only sources enforced) | All three check overlap. X1's and ours are the strictest; M3's evaluation sets are in-domain by task type (its card says so). |
| **Temperature calibration on a dev split** | M3, X1, ours (Laya ships uncalibrated) | M3 and X1 fit per task and option count; we fit per question type, as typed-decision-bench's `calibration.json` does. |
| **Wikipedia-style or reading data** for comprehension | X1 (Wikipedia), ours (Belebele, PersianQA, test only) | — |
| **Persian-specific phenomena** named as test targets | laya-fa-bench (8 families), ours (skills suite, robustness checks) | Overlap: digits and dates; formal vs colloquial register; orthographic variation (ours via the normalizer tests). |

## 4. What is different

**Only in our plan**

| Ours only | Why it matters |
| --- | --- |
| A **Persian translation of typed-decisions**, same case IDs as the English | The only paired English–Persian decision data among these works, so the cost of Persian can be measured on identical cases. |
| **Code-labeled skills with minimal pairs** at scale (3–5k items, five generators) | Exact labels and no LLM involved; laya-fa-bench covers digits and dates in only 8 of its 64 messages (one of its 8 families), without minimal pairs. |
| **Several native tasks deliberately test-only** (ParsiNLU, PersianQA, Belebele, STT) | The others train on what they have (M3) or test only their own families (X1). Ours can test "decision model, not classifier" on task families never trained on. |
| **License rule enforced in code** | M3 mixed in non-commercial data and inherited CC BY-NC-SA; X1 avoided it by building its own data; ours enforces it per record. |
| **Business workflows** (agent traces, customer service, invoices, security incidents) in Persian | None of the others covers them in Persian. |
| **Speech-to-text transcripts** as a real-use test | None of the others. |

**Only in theirs**

| Theirs only | Who | What it means for us |
| --- | --- | --- |
| **Scale**: 1.18M training examples, 78 families | M3 | Ours is 20–40× smaller. We compete on test design and data quality, not volume (as the plan already says). |
| **Agent tool calling and web navigation** | X1 | Out of our scope; X1 is a reference for choice questions only. |
| **Non-commercial English sources** (ANLI, RACE, Yelp, SciFact) | M3 | Excluded from ours by the license rule. |
| **No third-party LLM anywhere in the training data** | X1 | Ours uses a translator and a teacher-labeled source; our dataset card must state this plainly. |
| **Finglish, taarof (polite deflection), sarcasm, code-mixing** as test families | laya-fa-bench | Not in our plan. Finglish is in `future-work.md`; taarof and code-mixing are not yet anywhere. |
| **Instruction language as a variable** (same content, fa vs en instructions) | laya-fa-bench | Found English instructions scored higher (41 vs 36 of 64). Our templates already mix Persian and English questions; reporting results by question language would make this comparable. |

## 5. What this means for the plan

Most points were already handled in plan v6/v7; the three new ones were added in plan v8.

| Finding | Status in the plan |
| --- | --- |
| M3 trained on ParsiNLU and PersianQA, our held-out sets | Handled (v6): M3's scores there are in-domain and excluded from held-out comparisons. |
| M3's full source list is unknown | Open: read M3's `NOTICE` before reporting M3 on MASSIVE, FarsTail or Belebele. |
| X1 is choice-only and built without third-party LLMs | Handled (v6) for evaluation. **Added (plan v8):** every dataset card states which data involved a model (translator, teacher labels) and which did not (the skills suite). |
| Nobody else has paired English–Persian decision data or code-labeled minimal pairs | These are the project's clearest data contributions; the plan already leads with them (v5). |
| laya-fa-bench found instruction language changes results | **Added (plan v8, 3.2):** every metric reported by question language (`question_lang`, already in the schema). |
| laya-fa-bench tests Finglish, taarof, sarcasm and code-mixing | **Added (plan v8):** code-mixing as a robustness check (3.4); taarof and sarcasm in `future-work.md` (they need native writers); Finglish already there. |
