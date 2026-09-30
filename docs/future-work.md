# kodoom: Future Work

Ideas that could improve the project **after** the current plan. None of them is part of [the project plan](project-plan.md), and nothing in the plan depends on them. They are recorded here so they are not lost and do not creep into the current scope.

To move an idea into the plan, add it to the plan in its own change, bump the plan's version, and remove it here.

Effort is a rough guess for a solo developer: **S** is days, **M** is weeks, **L** is a month or more. Some ideas need more than free Colab; they say so.

## Adoption and serving

| Idea | Why | Effort | Needs first |
| --- | --- | --- | --- |
| **Jev-compatible HTTP server** | Developers could swap the model in for a hosted decision API without code changes. Several community models already do this. Left out of the plan because it is a new component to maintain, and demand is unknown. | M | A released model; signs that people want it (issues, downloads) |
| **Loading through Laya's package** | Users of Laya get the Persian model with one line. | S | Run 6 (Laya warm start) winning, so the model shares Laya's format |
| **Mobile and browser runtimes** (quantized ONNX on Android, WebAssembly) | Offline decisions on a phone fit the voice-assistant use case better than a laptop. | M | The CPU export from plan 3.6 |
| **Framework adapters** (e.g. an agent framework's "router" or "classifier" component) | Meets developers where they already build. | S each | A stable request/response format |
| **Hosted demo** (e.g. a Hugging Face Space) | People try before they download. | S | A released model |

## Data

| Idea | Why | Effort | Needs first |
| --- | --- | --- | --- |
| **Localized typed-decisions-fa (v2)** | Amounts in Toman/Rial and dates in Jalali, with labels recomputed by code. The plan already names this as a separate v2 and keeps v1 unlocalized. | M | typed-decisions-fa v1.0; a way to recompute labels that depend on numbers |
| **Native, human-labeled Persian decision set** | Persian customer messages, invoices and tickets written by Persians and labeled by people. Removes translationese and the ceiling of labels from an unnamed teacher model, the two biggest limits of typed-decisions-fa. | L | Annotators, a labeling guide, a license for the source texts |
| **Second reviewer for the test split** | A second person reviews a sample of the 400 test cases, so agreement between reviewers can be published alongside the fix count. | S–M | A Persian-speaking volunteer; the reviewed v1.0 test split |
| **Better soft labels** | Relabel typed-decisions (English and Persian) with several strong models and average them, or add human labels on a sample, to raise the label ceiling. | M | API budget or annotators |
| **Finglish (Persian in Latin script)** | Many users type Persian in Latin letters on phones and in chats. No current data covers it. | M | A transliteration source or a generator; tests on real Finglish |
| **Colloquial and dialect coverage** (Tehrani colloquial, other regional varieties) | The plan covers formal vs colloquial register only broadly. | M | Native speakers for review |
| **Dari and Tajik** | Close relatives of Persian, with little data of this kind. Tajik is written in Cyrillic, so it also tests the tokenizer. | L | Native speakers; sources with clean licenses |
| **More code-labeled skills** | Iranian national ID checksums, IBAN (Sheba) validation, card numbers, Persian addresses, Persian holidays and business days. Exact labels, no GPU. | S each | The generator framework from plan 1.1 |
| **Speech-noise augmentation** | Simulate speech-recognition errors on training text, so the model holds up on noisy transcripts without needing real ones. | M | An error model from the own STT data |

## Modeling

| Idea | Why | Effort | Needs first |
| --- | --- | --- | --- |
| **Distillation from a larger model** | Train the small model on soft labels from a large, strong multilingual model, for accuracy without a larger deployed model. | M | API budget or a bigger GPU than a free T4 |
| **Models of 2B parameters and larger** | Left out of the plan because they are too tight on a free T4. Might close the gap on many-option questions. | M | Paid GPU time |
| **Several questions per prompt** | The plan uses one question per decoder prompt in v0. Answering all of a case's questions at once is faster and closer to how the interface is used. | M | The decoder pipeline from the plan |
| **Learned abstention** ("none of these", "not enough information") as a first-class answer | Safer automation than a confidence threshold alone. | M | Training data with real abstention cases |
| **Multi-label and ranking questions** | Some decisions pick several options or order them; the plan covers choice, score and yes/no only. | M | Schema and metric extensions |
| **Long states** (beyond 384 tokens) | Long documents, email threads, logs. | M | A long-context encoder or chunking strategy |
| **Few-label adaptation** (for example SetFit-style fine-tuning on 50–200 labels) | Goes beyond the plan's recalibration recipe: adapts the answers, not only the confidence, to a user's own task. | M | A released model |

## Calibration and trust

| Idea | Why | Effort | Needs first |
| --- | --- | --- | --- |
| **Conformal prediction** | Returns a set of options with a guaranteed error rate, a stronger promise than a calibrated probability, and one that holds on a user's own data. | S–M | A calibration set from the user |
| **Drift detection** | Warns when inputs look unlike the training data, the situation in which calibration fails. | M | Logged predictions from real use |
| **Fairness and robustness audit** across register, dialect and name origin | Checks that answers do not change with who is writing, not only with digit forms or option order (plan 3.4). | M | Paired test items |

## Community and benchmark

| Idea | Why | Effort | Needs first |
| --- | --- | --- | --- |
| **Public Persian leaderboard** | Makes the Persian test suite a standing benchmark others submit to, which is what gets a dataset cited. | S–M | The evaluation harness from plan 3.6 |
| **Contributing the Persian sets to community benchmarks** (e.g. typed-decision-bench, which had no Persian cases as of September 2026) | Persian results appear wherever other languages are compared. | S | typed-decisions-fa v1.0 and the harness |
| **Feedback loop from real use** | Users flag wrong answers; these become new test items and, after review, training data. | L | A deployed model with users; a privacy policy |
| **Write-up** (blog post or workshop paper) on how decision models handle Persian | Reaches people who never browse model hubs. The evaluation report is most of the material. | M | The evaluation report |
