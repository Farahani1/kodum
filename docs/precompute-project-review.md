# Project review before the Kaggle campaign

Oct 8, 2026 · research audit for project-plan v22

**Recommendation: conditional go for private, resumable drafts. Keep publication,
translator adoption and training on separate gates.** The existing 1,600-case
typed-decisions scope remains a useful bilingual anchor. The 2,000 helmo records
are an optional augmentation experiment, not a prerequisite for the first useful
release. No TPU execution or complete raw-data audit was performed in this review.

## Dataset choices and alternatives

| Resource | Judgment | Next action |
| --- | --- | --- |
| LocalLLaMA/typed-decisions | Keep as the bilingual workflow anchor. Its labels measure agreement with a synthetic teacher, not independently established correctness. | Preserve the original IDs, splits and targets; review translation fidelity separately from source-label validity. |
| helmo/synthetic-typed-decisions | Useful topic breadth, weaker evidence of correctness. It is optional training augmentation. | Audit English labels before adopting Persian translations; exclude unsupported questions and keep its results/config separate. |
| Existing MASSIVE-fa and FarsTail | Higher priority for testing useful Persian transfer than acquiring another large translated corpus. | Implement the planned converters and audit task descriptions, provenance and splits. MASSIVE is human-localized parallel data; label it accordingly rather than calling it originally authored Persian. |
| Code-labeled Persian skills | Strongest independent contribution: exact labels, local phenomena, reproducible generators. | Publish seeds, template holdouts, pair IDs and generator versions; review the Persian templates. |
| amyrmahdy/decima-system-one-tasks, `s2` Persian subset | A plausible alternative to part of helmo's augmentation; already includes Persian and grouped tasks. | Audit a pinned Persian sample for correctness, license/provenance, group splits and overlap before adding it. Existing synthetic Persian can save translation compute, but still needs review. |
| amyrmahdy/jabr-v2-persian | A complementary benchmark already available in Persian. | Audit and adapt it for evaluation only; keep it out of training and tuning. Do not translate it again. |
| CLINC150 (`clinc/clinc_oos`) | Useful future test of out-of-scope routing and abstention. | Consider a small reviewed slice only after M1; it cannot replace the full three-type workflow corpus. |
| BANKING77 (`PolyAI/banking77`) | Useful fine-grained customer-support intent data. | Lower priority than existing Persian intent data; translating it adds review work and retains foreign banking conventions. |

Source evidence checked on Oct 8: the [typed-decisions card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
describes synthetic teacher targets. The [helmo card](https://huggingface.co/datasets/helmo/synthetic-typed-decisions)
discloses unchecked model labels and says it is not a benchmark. The
[MASSIVE documentation](https://github.com/alexa/massive) describes localization
and parallel IDs; [FarsTail](https://github.com/dml-qom/FarsTail) supplies Persian NLI.

The [Decima task card](https://huggingface.co/datasets/amyrmahdy/decima-system-one-tasks)
describes a CC BY 4.0 multilingual synthetic corpus, Persian included, with soft
targets and whole-task holdouts. This makes it worth a small comparative audit,
not an automatic replacement. The [Persian jabr release card](https://huggingface.co/datasets/amyrmahdy/jabr-v2-persian/commit/ebd9d7aab6d0cd7be834258903343e8ccf37f901)
describes 845 parallel cases across 48 translated tasks, all-Persian and
Persian-state/English-question configurations, and casewise review; it cautions
that review was not by a native annotator panel. Those are author reports, not
verification by this project. [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos)
lists CC BY 3.0; [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) lists
CC BY 4.0. None of these new sources has been downloaded, registered or adopted.

## DibaOne and Laya: existing baselines considered

All three were already in project-plan sections 2.2 and 2.4 and the historical
data comparison. The first audit inherited those comparisons; their live cards
were freshly checked following the owner's question on Oct 8. This is a
document review, not independent reproduction of their measurements.

| Work | What its current card establishes | Consequence for this project |
| --- | --- | --- |
| [DibaOne M3](https://huggingface.co/Dibachain/DibaOne-M3) | Persian-first model supporting choice, yes/no and score; weights CC BY-NC-SA 4.0. Its training sources include ParsiNLU and PersianQA. Its own evaluation is explicitly described as in-domain by task type. | Include as an evaluation baseline, especially for Persian classification and score tasks. Exclude those exposed task families from claims of held-out task generalization. Source exposure alone does not prove exact test-row leakage; verify splits and overlap separately. |
| [DibaOne X1](https://huggingface.co/Dibachain/DibaOne-X1) | Apache-2.0, choice-only in v1; retriever plus cross-encoder with fast, balanced and best modes. Training focuses on dev-path, tool-call and Wiki Race tasks. Its card says comparisons against zero-shot Laya demonstrate task-specific training, not general model superiority. | Keep it as a Persian-first choice baseline. Record mode, shortlist coverage, precision and end-to-end latency. Do not use it as native support for score or yes/no without declaring a conversion protocol. |
| [Laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | Apache-2.0, mmBERT-base decision model supporting all three types. Its card discloses uncalibrated shipped temperatures, near-chance zero-shot typed-decisions performance and score-position bias. It defaults to 1,024 context tokens; an explicit long-context setting supports up to 8,192, while questions/options retain their own budget. | Keep it as the multilingual baseline and possible warm start. Evaluate raw and equally recalibrated results, test option/score bias, and explicitly configure context limits. Shared backbone alone does not make comparison a controlled training-data experiment. |

These works already provide Persian-capable decision models. Our useful claim
is an independent, reusable data/evaluation layer that can assess and improve
them. M3's classification strength and X1's agent focus also argue for separate
workflow/task results rather than one undifferentiated leaderboard. Their full
training inventories, pinned revisions and exact benchmark exposure still need
checking before our model evaluations; current card statements are not that audit.

## What the project can credibly contribute

The defensible contribution is an independently auditable Persian evaluation
resource: parallel workflow cases, human-reviewed translations, native or
human-localized tasks, exact-label local skills, public prediction files and
consistent adapters. Avoid claiming the first Persian typed-decision dataset or
benchmark. Existing Persian resources make that claim untenable without a much
narrower, verified definition.

Translation volume alone is easy to reproduce. A reviewer can instead inspect
what changed, which cases remain questionable, how splits were protected and
which behaviors transfer across languages. Users benefit even if our reference
model never wins a leaderboard. We should be optimistic about that bounded
contribution, while measuring demand through reproducible external model runs
and independent reuse rather than promising downloads or adoption.

## Source quality is a separate review

Use distinct findings for `translation_error`, `source_label_issue`,
`underspecified_state`, `external_knowledge_required` and `ambiguous_rubric`.
Answer agreement between English and Persian does not validate the English
answer. A translator and checker can both repeat the same source mistake.

Preserve upstream gold in the canonical bilingual config, including questionable
cases. Publish exclusions and a clean-subset manifest for primary analysis. Any
new adjudicated targets belong in a separately named, versioned config, with
their rationale; never overwrite upstream targets to make a translation fit.
Show complete-set results beside subset results so removing hard cases cannot
quietly improve the headline.

Before helmo is used for training, independently review at least 100 English
records from the frozen selection, stratified by type and topic, including
long states. Source correctness and translation fidelity get separate error
counts. The current converter interpolates a score mean onto adjacent levels;
it preserves the reported variance as metadata but does not reproduce it in the
constructed distribution. Identify those targets as mean-preserving converted
targets, not recovered teacher probabilities. Report choice renormalization too.

## Review effort and acceptance

The diagnostic 40 helmo / 20 typed cases is deliberately diverse, not a random
prevalence sample. Keep the existing pilot thresholds as screening criteria;
they cannot certify the whole dataset. Even zero errors in a genuinely random
40-item sample gives an approximately 7.2% one-sided 95% binomial upper bound;
zero in 100 gives about 3.0%. A purposive diagnostic sample cannot use that
binomial calculation as its population error bound.

Review all 400 test cases, including all five decisions and option descriptions.
The plan's 2–4 minutes per case implies about 13–27 hours before corrections,
adjudication and bookkeeping. Time ten real cases first and revise the budget.
Get a second native reader for at least 40 cases where feasible, plus disputed
cases; record disagreements and adjudication. If that is unavailable, describe
the release as single-reviewer and keep the limitation visible.

Train and helmo each need separate random unflagged samples as well as all
flagged records. Preserve the sampling frame, seed, inclusion probabilities,
pre-edit severity counts and confidence intervals. A topic-balanced rate is
not automatically the population-weighted rate. Keep the original draft,
correction diff, reviewer, date and review decision together. An LLM judge
prioritizes work; it cannot set `human_reviewed`.

The proposed meaning checker is still research work to validate. Test its flags
against human findings; count both missed errors and false flags. Automatic
number checks do not prove that quantities retain their actors, units, signs,
comparisons or conditions. Protected Latin tokens are acronyms, identifiers,
code and technical terms; ordinary English words must be translated.

## Licensing and the open-model objective

The [Gemma 3 terms](https://ai.google.dev/gemma/terms), sections 1.1(e) and 3.3,
distinguish outputs from models built through transfer of its outputs. Outputs
are not themselves Model Derivatives, and Google claims no rights in them.
Some models trained on synthetic outputs can fall within the derivative
definition. Whether this specific translation-training use does so requires a
documented interpretation; human correction is not an assumed exemption.

Default for the planned permissive reference models: hold Gemma 3 translations
out of training until the owner records a compatible licensing route. Retaining
a source's Apache/MIT label does not establish downstream model eligibility.
Record source terms, source teacher where disclosed, translator/version and
review provenance separately. The current registry enforces source/split rules;
it does not implement a full teacher/translator-derived training license gate.
That gate is required before M3.

The official [Gemma terms page](https://ai.google.dev/gemma/terms) routes Gemma 4
to its separate [Apache 2.0 license](https://ai.google.dev/gemma/apache_2).
Gemma 4 is therefore a licensing candidate for a separate train-translation
trial. This does not establish Persian quality, available Flax assets or
compatibility with our pinned Gemma 3/JAX runner. Do not swap it into a frozen
campaign. Source licenses and attribution still apply to translations.

## Evaluation that demonstrates value

Before training, freeze primary endpoints and the M2 gate thresholds. Use native
held-out accuracy/macro-F1 and log loss/Brier as primary transfer evidence;
teacher agreement, calibration, pair consistency and latency are complementary.
ECE alone can favor an uninformative prior. Report native-authored,
human-localized, machine-translated and procedural sets separately.

Use matched ablations with the same backbone, seed, update budget and calibration
policy: native/skills plus eligible English; add licensed reviewed
typed-decisions-fa; add helmo only if eligible; test an audited existing-Persian
augmentation as a later alternative. Keep helmo's marginal effect separate from
typed-decisions'. One seed is exploratory; repeat a promising comparison with
at least three seeds when affordable. Case-grouped paired intervals cannot
substitute for training-seed variation.

Track model pretraining/fine-tuning exposure as known, suspected or unknown.
The newer [Decima-base card](https://huggingface.co/amyrmahdy/decima-base) makes it
a relevant baseline candidate, but its scores and exposure need independent
checks. Sharing a backbone alone does not isolate a training recipe; only
matched starting checkpoints and interventions support that causal claim.

Split source cases, translations, paraphrases and minimal pairs together;
scan near-duplicates across sources and tasks before training. Never use test
answers to choose prompts or calibration. Publish missing/failed coverage and
paired common-item comparisons, not just accuracy on successful requests.
Measure token lengths for each evaluated model; reject or explicitly mark
unsupported long inputs instead of silently discarding the decisive evidence.
Subsampled options and full option sets are different protocols. Any option
removal requires a documented remapping of all soft probability mass.

## Campaign and release gates

| Gate | Evidence needed | Current assessment |
| --- | --- | --- |
| Private draft generation | Pinned inputs/runtime; permitted model access; live private save/readback/restore; measured token lengths; safe real batch throughput; deadline and final-save evidence | Local software exists; TPU and private HF evidence pending. |
| Translator adoption | Native human review of production prompts across workflows/types, long states and option descriptions; comparison with retained 4B baseline | Pending; model size and the earlier single-workflow LLM pilot do not establish it. |
| Data release | Full test review; separate source-label audit; correction/exclusion manifests; source/model provenance; license/attribution card; independent regeneration/use example | Pending. |
| Reference-model training | Source and translator license gate; frozen train/dev/calibration/test and overlap report; native baselines; chosen M2 outcome; context-length policy | Pending; private drafts alone do not satisfy it. |

Keep the current queued-session strategy. The notebook already tests private
storage and benchmarks BF16 batches inside the allocation; a second physical
session still has to prove restoration. Use actual completed cases and decisions
per hour, rejection rate and accepted-work review effort for the budget. Raw
generated tokens/second is insufficient. The frozen 3,072 input / 1,536 output
limits may reject long or expanded structured records; report affected IDs and
change a campaign deliberately if necessary. No silent truncation or scope loss.

The current request and notebook pin remain the v21 production recipe.
This v22 audit changes research interpretation and the local helmo split guard;
it does not silently migrate an existing campaign. A runtime change needs a new
published pin and compatible campaign handling under the bulk identity rules.

First useful release: reviewed typed-decisions-fa plus the Persian skills suite,
cards and runnable scoring examples, packaged under the
[published dataset specification](dataset-release-spec.md) added in plan v23.
Helmo can follow separately. A small
100–200-case Persian-authored everyday decision collection, with documented
rules, negation, abstention and near-miss pairs, is a higher-value next extension
than enlarging helmo blindly; it is optional after M1 and outside this run.
Continue training only when the data shows an unresolved gap that our effort can
plausibly improve.
