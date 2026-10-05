# Changelog and project status

This file records what changed in kodoom and where the project stands against
`docs/project-plan.md` (currently version 16, the source of truth). Format follows
[Keep a Changelog](https://keepachangelog.com/): newest first, grouped by kind of
change. The project has no releases yet; everything is under "Unreleased" until the
first published dataset (typed-decisions-fa v0.1, plan 1.3).

Update this file in the same piece of work as any change that moves the status
table below or adds a user-visible command, module or decision.

## Status against the plan

Last updated: Sep 30, 2026, at commit `3505090`. Checks: `ruff check`, `ruff format
--check` and `pytest` (567 tests) pass; CI runs Windows and Ubuntu on Python 3.11 to
3.13.

| Plan | Item | State |
| --- | --- | --- |
| Environments | `dev`, `colab-preflight`, `colab` profiles, storage keys, crash-safe run directories, `kodoom check`, thin notebook, Colab guide | Done. Preflight ran on a real T4 (Python 3.13): check, generate, fetch and fields all ran. |
| 1.1 | Code-labeled skills: Jalali dates, digit forms, Toman/Rial, business hours, Iranian formats, with minimal pairs, held-out test templates, raw forms | Done: 4,200 items at the default 150 pairs per kind. Templates reviewed on Oct 2, 2026 (an assistant review accepted by the owner, `data/review/templates-reviewed.csv`): 21 of 84 reworded, all five generators now at version 2. Every label in the version-1 data was also re-checked against two independent Jalali libraries: no errors. |
| 1.1 | typed-decisions loader, pinned revision `e135720c…`, field statistics | Done. Real data loads: 1,200 train cases (6,000 decisions) and 400 test cases (2,000), matching the dataset card. Preflight caps train at 200 cases per workflow. |
| 1.1 | Translate/keep rules for each field | Done (`translate/rules.py`), decided from all 1,200 cases; a text field with no rule is an error. |
| 1.1 | helmo/synthetic-typed-decisions: dedupe, sample 3-5k, no overlap with the test split | Loader written (`kodoom fetch helmo`) from the real structure: one JSONL file, 9,879 records, MIT, pinned `1827dc0d…`; score gold is a mean, turned into a two-level distribution. Ran on Colab (preflight, 200 records, 132 topics: 71 yes/no, 71 choice, 58 score); 7 of 200 had gold not summing to 1 and were rescaled and flagged. Dedupe, sampling and the overlap check are not started. |
| 1.2 | Automatic checks, pipeline, orthographic cleanup | Done and tested with a stub translator (`kodoom translate --translator stub`): keep-fields, identifiers, numbers, script, length, looping; resumable per case. Glossary (reviewed Oct 2, 2026: refund → استرداد وجه, freight → هزینه حمل) and its consistency check done. Model translators written. **Trial 1 (Colab, T4):** Qwen3-8B (4-bit) ran, and all automatic checks passed, yet its translations changed meaning ("expired" became "valid", "irreversible" became "reversible") and one word contained Cyrillic letters (now caught by a new check). TranslateGemma failed to load: the repo is gated (license and `HF_TOKEN` needed). **Trial 2:** access granted; plain fp16 gave empty output (fp16 overflow suspected), bf16 works. **Trial 3 (TranslateGemma 4B, bf16, 10 test cases):** it ran on the T4 and its Persian is clearly better than Qwen's ("expired" and "irreversible" correct; identifiers, numbers and script checks all passed), but it ignores the glossary (132 findings; "agent" came out as نماینده, کارگزار and «this person» in one case), turned "halt the agent" into halting a person, lost "queue" and garbled one phrase. **Trial 4 (glossary terms written into the input, `-bf16-terms`):** 132 → 110 glossary findings only; the model turns "agent" back into "person" (فرد, شخص, کارگر) and pinning leaves artifacts (عامل "Halt"). A translation-only model cannot be steered this way. New: `gemma3-4b-bf16` and `gemma3-12b-4bit`, instruction-tuned Gemma 3 with the glossary, the register and a per-workflow context sentence in the prompt **Trial 5 (`gemma3-4b-bf16`, bf16, 10 test cases):** best so far and zero automatic findings: "agent" is «عامل» and "trace" «ردگیری» everywhere, "irreversible" and "expired" right, and no person-for-agent drift. Flaws: it wrapped many answers in backticks (now unwrapped and forbidden in the prompt), left English words such as "indefinitely" and "(load balancer)" (new `english` check), and some words are simply wrong ("load balancer" as «بار متعبیری», "observability" as «پایشدهایی», "job" as «شغل»), which only the meaning check and human review can catch. The 12B model is the next trial (24 GB download; the Colab disk was at 79 of 112.6 GB, so free the Qwen cache first). **Trial 6 (`gemma3-12b-4bit`, 10 test cases, old code):** more fluent prose than the 4B but it garbled the score levels (Low/Moderate/High became «سطح»/«شدت»/«شدت», Benign a nonsense word) and some terms («مراقبتی» for monitoring); no clear winner, so the 50-case blind pilot (`kodoom pilot-sheet`, `pilot-score`, `translate --balanced`; new `labels` check) decided it. **Pilot done (Sep 30):** Gemma 3 12B won narrowly, 4B effectively tied. **Decided (Oct 2, plan v13/v15): Gemma 3 4B is the translator for the whole dataset**, including helmo; 12B is dropped (too slow on a free T4). Gemma's terms on generated outputs were read (plan v14): outputs are not Model Derivatives, so publishing under Apache-2.0 looks fine, but a model trained on Gemma-made translations would be. `kodoom translate helmo`/`translations helmo` now exist (plan v15), so the free-text gate (step 1) can run on Colab. **Not done:** the gate itself, the meaning check, human review. The TPU trial is optional. |
| 1.3 | One record schema, source rules | Done (`schema.py`, `sources.py`). |
| 1.3 | Publish typed-decisions-fa | Not started, and deliberately on hold: owner decision that no data is pushed anywhere yet, everything stays on the owner's Drive. |
| 2.1-2.3 | Training mix, training, logs | Not started. Run manager, profiles and checkpoint rules are ready. |
| 2.2 | Baselines (uniform, prior, oracle) and harness | Done for these three. Real model baselines (Laya-multilingual, DibaOne, prompted Qwen) need Colab. |
| 2.4 | Decision gate | Not started. **Open:** the owner must set the gate thresholds, which are placeholders in the plan, before any baseline runs. |
| 3.x | Metrics, grouped bootstrap, temperature scaling, `calibration.json` v2, scoring harness | Done and tested on generated skills. Not yet run on any real model. |
| 3.4-3.6 | Robustness checks, success criteria, reporting, CPU export, adapter | Not started. |

Milestones: M1 (data and skills suite) is partly done: the skills suite is built and
the English source loads; translation, review and publishing remain. M2 (baselines
and gate) has its harness and metrics but no model has been scored. M3 (training)
has not started.

### Open items for the owner

- Decide before M3 whether Gemma-made translations may enter the training mix: Gemma's terms count a model trained on Gemma synthetic data as a Model Derivative (plan v14, Licenses). Publishing the dataset under Apache-2.0 appears fine.
- Choose the gate thresholds in plan 2.4 before the M2 baselines run.
- Rename the GitHub repository to `kodoom` (Settings).
- Copyright-holder name in `LICENSE` (currently "the kodoom authors").
- Confirm: MASSIVE fa-IR locale, the ParsiNLU and mmBERT-base licenses, DibaOne M3's NOTICE file.
- Re-run the *Get the code* cell and the new *Fetch helmo* cell on Colab, and paste the output.

### Next

1. **Owner, on Colab:** run the notebook's *Free-text gate* cells (plan 1.2 step 1, v15):
   translate ~40 helmo records and 20 typed-decisions training cases with `gemma3-4b-bf16`,
   paste both `translations` printouts back for rating against the pass bar (at most 1 meaning
   change in 40, at most 10-15% needing an edit).
2. The meaning check (plan 1.2 step 4): the checks cannot see "agent" turning into "person", so
   a checker model must compare English and Persian per case. Not started.
3. If the gate passes: the full run (plan 1.2 step 2) on all typed-decisions cases and the
   helmo sample, still `gemma3-4b-bf16`. If it fails: shrink or drop helmo per the plan's
   updated failure-mode mitigation (v15), rather than falling back to 12B.
4. Optional: a v5e TPU trial (TranslateGemma 4B in bf16, ten sentences) and, if it works, a `colab-tpu` profile.
5. Optional: `kodoom fetch helmo` with the `colab` profile for the full counts.

### What the existing Russian and Japanese versions show

Read from their cards and first rows (`kodoom inspect`, Colab): both keep case ids,
splits, option ids and gold identical to the English original, and translate the
state text, question text and option descriptions. Both are Apache-2.0. The Japanese
version adds a back-translation verifier (`back_translation_match`,
`mismatch_category`, `verifier_reason`). Their first rows show what to guard
against: the Japanese score levels keep English labels (" benign:", "Low:") and the
Japanese `factors` were left in English while the Russian ones were translated.
Our checks and glossary are meant to catch this kind of inconsistency.

## Unreleased

### Added

- Portable notebook workflow design in `agent.md` and the separate
  `docs/workflow-change-plan.md` (WF-00 complete; WF-01 through WF-06 pending).
  Defines reference and active execution notebooks, explicit Colab/Kaggle/generic
  environments, artifact provenance and verified private persistence/restore.
  Main plan v16 links the migration without advancing any research gate. Existing
  notebook guidance now warns against Run all across historical experiments.
  New notebooks, Kaggle support and data migration are not implemented yet.

- `kodoom translate helmo` and `kodoom translations helmo` (`pipeline.py`, `report.py`): helmo
  has no workflows and its `state` is one free-text paragraph rather than a JSON tree, so each
  record is translated whole, in one call (state, question and options together), and checked
  with the same text-level checks as a typed-decisions question or option. `--balanced` takes
  records evenly across question type (choice, score, noul) instead of workflow. Closes the gap
  that blocked plan 1.2's free-text gate, which needs a helmo sample translated before the full
  run; the notebook gained a *Free-text gate* section that runs it (plan v15).
- `kodoom review-pack` (`review.py`): the glossary and all 84 generator templates, each with a
  generated example, as right-to-left HTML pages and CSV sheets with feedback columns, in
  `<data_dir>/review/`, for the native review. The notebook also gained a `pilot-score` cell.
- `<data_dir>/README.md`, kept current by every command that writes data (`datadir.py`): the
  file tree with sizes, record counts and what each folder holds, what the last update made
  or changed, and the last 20 updates with their command lines. `kodoom tree --start` and
  `kodoom tree` bracket a notebook run; the last notebook cell prints the tree with the files
  that run generated or updated marked.
- `kodoom export-units` and `import-units` (`translate/exchange.py`): a translator that cannot run
  inside kodoom (Claude Cowork, a person) gets each distinct English text once with its
  register, context and glossary terms and a brief; the filled file comes back through the
  same checks, failure log and review tools as a model run.
- Model translators (`translate/hf.py`): TranslateGemma (translation-only prompt) and a chat
  translator for Qwen3-8B (register, glossary and rules in the prompt); `kodoom translations`
  shows check counts and English next to Persian. The English-Persian glossary
  (`translate/glossary.toml`, draft) and its consistency check.
- `kodoom translate typed-decisions`: the translation pipeline with a stub translator.
  Rules (`translate/rules.py`), automatic checks (`translate/checks.py`) and
  resumable per-case output; Persian records copy gold, option ids, split and
  `source_id` from the English ones and carry `checks_passed` and any findings.
- `kodoom fetch typed-decisions` and `kodoom fields`: the English cases as one
  record per question (state kept byte-identical, option ids and gold as in the
  source), and per-field statistics that decide what gets translated. Fetched data
  goes to the profile's `data_dir`, never into the repository.
- `kodoom fetch helmo`: helmo/synthetic-typed-decisions as records (see the status
  table for the score-gold conversion).
- `kodoom inspect`: prints the real structure, license and card of a Hugging Face
  dataset (parquet or JSON lines), used once per dataset on Colab.
- Evaluation harness: `kodoom baseline`, `score`, `calibrate`; prediction files,
  `calibration.json` v2, metrics (accuracy, soft accuracy, NLL, Brier, ECE, KL,
  score MAE), grouped bootstrap intervals, temperature scaling, minimal-pair
  accuracy, seen vs unseen task families.
- Five code-labeled skill generators and `kodoom generate`, with a contract test
  suite every generator must pass and pinned fingerprints.
- The Jalali calendar, checked against reference libraries for 1280-1520.
- Run directories with crash-safe checkpoints, resume, and the Drive storage
  budget (`kodoom runs`, `kodoom check`); the thin Colab notebook and `docs/colab.md`.
- One record schema with a required `task_family`, the source registry with license
  and role rules, and the single Persian normalizer (`normalize`, with
  `clean_orthography` for published data).
- Documents: `docs/project-plan.md` (v1 to v9), `docs/future-work.md`,
  `docs/data-comparison.md`; `AGENTS.md` with the commit convention.
- CI on Windows and Ubuntu, Python 3.11 to 3.13.

### Changed

- Skill templates and glossary after review (Oct 2, 2026). Templates: mobile and postal
  questions now ask about the format ("از نظر قالب", "ده رقم") instead of «معتبر»/«درست»,
  which a reader could take as real-world validity; contexts whose plausible amounts are
  narrow (bus ticket, taxi fare, phone, salary) replaced by ones that fit any amount
  (purchase, transfer, quote, transaction); and wording fixes in before-4, valid-5,
  magnitude-2/3, time-1 and date-1. Generators bumped to version 2, so regenerated skills
  data is a new version. Glossary: refund «بازپرداخت» (loan repayment) → «استرداد وجه»;
  freight «حمل» → «هزینه حمل».
- Plan v3 to v9: laptop/Colab separation (v3); the M2 decision gate (v4); reframed
  as a data and evaluation layer, PersianQA test-only, full review of the 400-case
  test split (v5); baselines and gate grounded in the published model cards (v6);
  the 15 GB Drive budget (v7); provenance statement, question-language metrics,
  code-mixing check (v8); least restrictive licenses and data kept on Drive (v9).
- Plan v13: Gemma 3 4B (bf16) is the translator for the full run, for speed, with
  12B as the fallback, not dropped. The pilot showed 4B tied 12B overall (4.62 vs
  4.47) but weaker on free text (4.23 vs 4.57: "backfill" became "return", English
  words left in), and it covered only `agent_trace_observability`. Because helmo is
  all free text and its gold rests on one fact per record, a free-text gate now runs
  before the full run (1.2 step 1): about 40 helmo records plus 15-20
  customer_service/security_incidents cases, translated by 4B with the full-run
  prompt, against a pass bar fixed in advance (at most 1 meaning change in 40, at
  most 10-15% needing an edit); 12B is the fallback for whatever fails it, with
  helmo's affected topics or all of helmo dropped from v1 if 12B also fails. Helmo
  translates state, question and options in one call, keeps acronyms and units
  unchanged (checked automatically), gets an answer-comparison meaning check, its
  own ~100-record review sample with a separately reported error rate, and starts
  at 1,500-2,000 records as a train-only config. A new failure-mode row covers 4B
  on helmo.
- Licenses: code 0BSD, the project's own data CC0-1.0; data derived from others
  keeps the upstream license (typed-decisions-fa stays Apache-2.0).
- The default is 150 pairs per kind (about 4,200 skill items), inside the plan's
  3-5k target.

### Fixed

- An empty translation no longer surfaces as a schema error: it raises a `TranslationError`
  naming the text, `kodoom translate` logs the case in `<split>.failures.jsonl`, skips it
  and exits 1. Found in trial 2, where TranslateGemma 4B answered every text with nothing.
  The model loader now checks the first-step scores for NaN/inf (fp16 overflow) and says so;
  `translategemma-4b-bf16` and `translategemma-4b-4bit-fp32` avoid it.
- The script check now flags letters from another alphabet (found in trial 1: Cyrillic
  letters inside a Persian word passed every check).
- The helmo loader no longer stops at rows whose choice gold does not sum to 1 (the
  first real fetch failed at row 33, sum 0.95): such gold is scaled to 1, the original
  sum is kept in `extra.gold_sum_in_source`, and `fetch` reports how many there were.
- The typed-decisions loader accepts yes/no questions that have no `criteria`
  (found on the first real fetch, invoice_processing "duplicate"); they get the
  options "No" and "Yes" with the ids the gold already uses.
- The notebook token cell says why a private clone failed instead of hiding the
  cause; the owner's Colab save was reconciled with the repository notebook
  (saving back to GitHub from Colab is not part of the workflow).

### Decisions worth remembering

- Hugging Face is not reachable from the development environment, so every step
  that reads a Hugging Face dataset or model runs on Colab; the loader is tested on
  invented rows in the real layout, and no dataset content is committed (a test
  fails if a data file is tracked).
- A typed-decisions case becomes one record per question and its questions share a
  `source_id`, so a case never straddles two splits.
- Published data is never fully normalized: translations get `clean_orthography`
  only, and the skills suite keeps raw digit forms; the model's input pipeline
  applies the full normalizer.
- The typed-decisions gate signal for the language gap was moved to parallel MASSIVE
  and Belebele items, because Laya-multilingual is near chance on typed-decisions
  even in English.
