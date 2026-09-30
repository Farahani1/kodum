# Persian Decision Model: How Our Method Compares

Sep 30, 2026 · @Shah

## Summary

Our method's advantage is in how we test, not in how we build. None of the open models compared here (Laya-multilingual, DibaOne M3, DibaOne X1) has shown that it works on task types it wasn't trained on; Jev has the most independent evidence of that.

Our architecture is not new: the encoder reads one option at a time like DibaOne X1, and the decoder reads all options together like Laya. What is new is the evaluation: held-out tasks and templates, option renaming and reordering, native and translated results kept apart, and Persian-specific skills. It can show that our model is a decision model, or show honestly that it isn't. Companion to the [project plan](https://claude.ai/code/artifact/8a5e6024-53ad-4ec1-9082-a33430eedffa).

## The models compared

Jev is the reference, Laya-multilingual is the closest open model (same mmBERT-base backbone as our main model), DibaOne M3 and X1 are the existing Persian-first decision models, and Dohnuts is the evidence for our Qwen3.5 challenger.

| Model | Maker, release | Base model and size | Question types | Languages | Weights licence |
| --- | --- | --- | --- | --- | --- |
| [Jev](https://typesafe.ai/) | TypeSafe AI, Sep 15, 2026 | Unpublished; hosted API only | Choice, score, yes/no | [Per TypeSafe docs](https://docs.typesafe.ai/models) | Closed |
| [Laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | Convai Innovations; Laya family released Sep 18, 2026 | mmBERT-base + 2-layer decision head, 322M | Choice, score, yes/no | 100+ | Apache-2.0 |
| [DibaOne M3](https://huggingface.co/Dibachain/DibaOne-M3) | Dibachain, Sep 2026 | XLM-R-family bi-encoder, 278M | Choice, score, yes/no | Persian, English | CC BY-NC-SA 4.0 |
| [DibaOne X1](https://huggingface.co/Dibachain/DibaOne-X1) | Dibachain, Sep 2026 | XLM-R-family cross-encoder, 278M, plus an untrained 118M retriever | Choice only | Persian, English | Apache-2.0 |
| [Dohnuts 0.1.0](https://huggingface.co/PsiACE/Dohnuts-0.1.0-0.8B) | PsiACE | Qwen3.5-0.8B + LoRA + candidate scorer | Choice, score, yes/no; text and images | Multilingual | CC BY-NC-SA 4.0 |
| Ours, main | This project (planned) | mmBERT-base, 307M | Choice, score, yes/no | Persian first, English | Apache-2.0 (target) |
| Ours, challenger | This project (planned) | Qwen3.5-0.8B + LoRA | Choice, score, yes/no | Persian first, English | Apache-2.0 (target) |

## How each model scores options

The design sets a ceiling, not a verdict: a bi-encoder can only match meanings, while the other two designs can reason over the context and an option together. Every design ends the same way, with a softmax over the options the caller supplied.

| Design | What the model reads | Options see the context? | Options see each other? | Used by |
| --- | --- | --- | --- | --- |
| Bi-encoder | Context and each option embedded separately; score = cosine similarity | No, only through one similarity number | No | DibaOne M3 |
| Per-option cross-encoder | Context + question + one option, one pass per option | Yes | No | DibaOne X1, our encoder |
| All options in one pass | Context + question + every option in one input, each scored at its own marker | Yes | Yes | Laya-multilingual, Dohnuts, our decoder |

Jev's design is unpublished; it returns every option's probability from one call. Two practical limits follow from these designs: M3 is weak on long option lists because it never reads them against the context, and Laya-multilingual gives all options a shared 256-token budget, so its card advises keeping choices under about 20 options. MASSIVE's 60 intents exceed that.

## Classifier or decision model

Every model here returns probabilities over supplied options, so the difference is behavioural, and it is a spectrum. Jev is the only model with third-party evidence on the first test, and no model has published evidence on the third.

| Test | Jev | Laya-multilingual | DibaOne M3 | DibaOne X1 | Ours (planned test) |
| --- | --- | --- | --- | --- | --- |
| 1. Works on unseen task types | Yes, by third parties: radiology AUROC 0.977, crash coding F1 0.908, used as-is | Weak: 0.342 on typed-decisions zero-shot, below the 0.461 majority baseline | No: 0.351 / 0.179 / 0.188 on three new families (random 0.135 / 0.097 / 0.029) | Not claimed; tested only on its three trained families | Held-out tasks: ParsiNLU (4 tasks), Belebele-fa |
| 2. Chooses by meaning, not name or position | Studied in arXiv 2609.26758 | Studied in 2609.26758; position bias on score questions | Not reported | Not reported | Option rename, reorder and distractor tests (3.4) |
| 3. Same state, different question, different answer | Not reported | Not reported | Not reported | Not reported | Several questions per state in typed-decisions; explicit swap test to add |
| 4. Calibration holds on new tasks | Partly: overstated prevalence on crash data until recalibrated | No: ships uncalibrated; ECE 0.247–0.394 on DibaOne's benchmarks | In-domain only: ECE 0.010 on Persian, 0.088–0.172 on new families | ECE 0.070–0.130 on its test, above its own 0.05 target | ECE, Brier and log-loss on held-out sets, before and after temperature scaling |

## Published results

Trained models win on the tasks they trained on and fall to near chance on new ones; Jev is the exception with strong zero-shot numbers. All figures are as reported by each model's maker or by the benchmark named.

| Benchmark | What it measures | Results as reported |
| --- | --- | --- |
| typed-decisions (2,000 English decisions) | Agreement with a teacher LLM; ceiling 0.735 | Jev 0.727 zero-shot · Laya-multilingual 0.342 zero-shot · Laya English checkpoint, fine-tuned on its train split, 0.766 |
| Banking77 (77 options) | Intent with many options | Jev 0.870 · Laya 0.425 |
| MASSIVE, 51 languages, 20 options | Multilingual intent | Laya-multilingual 0.366 macro (random 0.050) |
| persian\_eval\_v1 (10,766 native Persian questions, DibaOne's own set) | Task types M3 trained on | DibaOne M3 0.738 · Laya-multilingual zero-shot 0.470 |
| DibaOne X1 benchmarks: dev path / tool call / Wiki Race (4,358) | New families for M3 and Laya; trained families for X1 | X1 0.767 / 0.751 / 0.279 · M3 0.351 / 0.179 / 0.188 · Laya-multilingual 0.336 / 0.176 / 0.288 · random 0.135 / 0.097 / 0.029 |
| Latency, one decision | Speed | Laya-multilingual 32.8 ms (T4) · X1 74–165 ms (T4) · M3 about 340 ms (CPU, 2 threads) · Jev 236–276 ms p50 (hosted, network included) |

Zero-shot versus trained rows measure what training buys, not general rank, and each maker ran its own comparisons. Our model has no results yet; section 3.1 of the plan scores it against these models on shared test sets.

## Where our method has an advantage

Our advantage is in what we can prove, not in what we build. It holds only if the held-out results come out well.

| Dimension | Ours compared with the others | Advantage? |
| --- | --- | --- |
| Evaluation design | Held-out tasks and templates, option rename and reorder tests, native and translated results reported apart, Persian skills test; none of the others is tested this way | Yes |
| Persian-specific data | Jalali dates, digit forms, Toman and Rial, Iranian formats, plus a public typed-decisions-fa; DibaOne has its own Persian set, not known to be public | Yes |
| Licence | Apache-2.0 target; M3 and Dohnuts weights are non-commercial | Yes, over M3 and Dohnuts |
| Training breadth | Many task families, against X1's three and M3's 78 | Over X1 only |
| Architecture | Encoder matches X1's design; decoder matches Laya's and Dohnuts' | No |
| Speed | Laya-multilingual scores all options in one pass; our encoder needs one pass per option | No |

If the held-out results come out weak, the honest finding is that our model is also a classifier in disguise. Shown clearly, that is still a useful result.

## What this changes in the plan

Eight changes follow from the comparison; none alters the plan's structure.

- [ ] Add DibaOne M3 and X1 as baselines (2.2) and to the references.
- [ ] Treat ParsiNLU and PersianQA as in-domain for M3, which trained on both; compare with it on Belebele-fa, typed-decisions-fa and the skills set.
- [ ] Refit Laya-multilingual's temperature on the calibration split before comparing calibration; it ships uncalibrated.
- [ ] Add a controlled run that warm-starts from Laya-multilingual and keeps its head, so only the training data differs.
- [ ] Make the M2 decision gate compare against the best existing Persian model, not Laya-multilingual alone.
- [ ] Add a question-swap test (same state, different question) and report an option-order flip rate.
- [ ] Never claim "beats Jev" from typed-decisions scores; lead with held-out Persian results.
- [ ] Check PersianQA's licence: M3's card lists CC BY-NC-SA 4.0, the plan lists GPL-3.0.

## Sources

- [Laya-multilingual model card](https://huggingface.co/convaiinnovations/laya-multilingual)
- [DibaOne M3 model card](https://huggingface.co/Dibachain/DibaOne-M3)
- [DibaOne X1 model card](https://huggingface.co/Dibachain/DibaOne-X1)
- [Dohnuts 0.1.0 model card](https://huggingface.co/PsiACE/Dohnuts-0.1.0-0.8B)
- [TypeSafe AI](https://typesafe.ai/) and [Jev models page](https://docs.typesafe.ai/models)
- [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
- [Luni/laya-jev-benchmark](https://huggingface.co/datasets/Luni/laya-jev-benchmark)
- [Type-Safe Is Not Error-Free (arXiv 2609.26758)](https://arxiv.org/pdf/2609.26758)
- [Calibrated decision models for pentest harnesses (arXiv 2609.28940)](https://arxiv.org/pdf/2609.28940)
- [Can Jev Judge Radiology Reports? (arXiv 2609.27607)](https://arxiv.org/pdf/2609.27607)
- [Calibrated Decisions at Scale (arXiv 2609.24052)](https://arxiv.org/pdf/2609.24052)
- [AGTP post on Laya and Jev latency](https://x.com/AGTPinsights/status/2102269496304267510)
