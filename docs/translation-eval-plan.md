# Translation pilot: choosing the Persian translator

This file records how the translator for typed-decisions-fa was chosen
(plan 1.2 step 1). It has two parts:

- **The brief** below, as it was given to Claude Cowork.
- **[How it was run](#how-it-was-run)** and **[Results and decision](#results-and-decision)**,
  which were added afterwards.

The full outputs (`ratings.csv`, `integrity.csv`, `report.md`) are in
`data/translation-eval/` on the owner's machine. Like all data, they are not
committed.

## The brief

Goal: decide which model gives the best English→Persian translation of the
`typed-decisions` test set, so we can use it for the full translation run.
You have read access to `data/`. Do not change anything in it. Write all output to a new `data/translation-eval/` folder (called `eval/` below).

### Inputs

- English source: `data/typed-decisions/en/test.jsonl` (2000 records; only 50 are used here).
- Five candidates, each `data/typed-decisions/fa/<model>/test.jsonl` with the same 50 records:
  `translategemma-4b-bf16`, `translategemma-4b-bf16-terms`, `gemma3-4b-bf16`,
  `gemma3-12b-4bit`, `qwen3-8b-4bit`.
- Match records by id: a Persian id is the English id plus `:fa`
  (`agent_trace_observability_000000:action:fa` ↔ `agent_trace_observability_000000:action`).
- Every record is one JSON object per line. Translate-relevant fields: `state` (a JSON string;
  only `task` and `constraints` are translated text), `question_text` and `options[].text`.
  `gold`, option `id`s and `extra.factors` must stay the same as in the English record.

What we already know (confirm it, don't just repeat it):

- All 50 are from one workflow (`agent_trace_observability`): 20 choice, 20 score, 10 yes/no (`noul`).
- `gemma3-*` wrap almost every text in stray backticks (`` `...` ``). This is a formatting defect.
- Both `translategemma-*` have `checks_passed: false`. The reasons are in
  `extra.check_findings`, which are mostly glossary misses (e.g. "trace" should be "ردگیری").
  `qwen3` and `gemma3` passed the automatic checks.

### Steps

1. **Integrity check (automatic).** For each candidate and record, check: the record exists,
   option ids and order match English, `gold` is unchanged, `state` is still valid JSON with the
   same keys and the same non-text values, and no text is empty or left in English. Count stray
   backticks, Latin-script words other than names/product names, and mixed Persian/Arabic or
   Latin digits.
2. **Human-style quality rating.** Rate every translated text unit (the task, each constraint,
   the question and each option) against the English source, 1–5 on each of:
   - **Accuracy:** same meaning, nothing added or dropped. This matters most, because a
     meaning change can flip the correct answer.
   - **Fluency:** natural, grammatical Persian with correct half-spaces (ZWNJ).
   - **Terminology:** consistent domain terms (agent=عامل, trace=ردگیری, run=اجرا, …).
     Use the terms in `extra.check_findings` as the glossary.
   Mark any unit whose meaning changed badly enough that the gold answer might no longer hold
   (`meaning_flag`). Rate candidates side by side for each record and don't show the model
   names while rating, so you stay consistent and unbiased.
3. **Aggregate.** For each model: mean scores per dimension and per question type, number of
   meaning flags, and integrity defect counts. Treat backticks as fixable by post-processing:
   report scores with and without that penalty.
4. **Recommend.** Rank the models and pick one. Give the main reason and the risks. Say whether
   the glossary (`-terms`) variant helped, and whether a cheap post-fix (strip backticks,
   glossary replace) would change the ranking.

### Deliverables (in `eval/`)

- `ratings.csv`: one row per model × record × text unit, with the columns
  `model, id, field, en, fa, accuracy, fluency, terminology, meaning_flag, note`.
- `integrity.csv`: one row per model × record, listing the defects found.
- `report.md`: no more than one page. It has a summary table (models × metrics), the ranking
  and recommendation, 3–5 short example translations that show the key differences, and any
  caveats (sample is only 50 records from one workflow, and it was rated by an LLM).

Keep it proportional: this is a quick choice between models, not a benchmark.

## How it was run

- **Who rated:** Claude Cowork (Anthropic's desktop agent that works on local files),
  running **Claude Opus 5.5** (`claude-opus-5-5`). It was the only rater. No human
  rated translations in this pilot.
- **Input:** Cowork got this brief and read access to `data/`. It wrote only to
  `data/translation-eval/` and confirmed that the input files were unchanged.
- **Integrity checks:** Cowork wrote and ran a script for step 1. These checks come
  from code, not from the model's judgment.
- **Quality ratings:** Cowork rated each translation blind, without the model names.
  It compared all five candidates side by side for each English text, then unblinded
  the labels. Only 34 English strings and 161 distinct translations exist across the
  1,900 rated units. So each distinct translation was rated once, and that rating was
  copied to every record where the same translation appears.
- **Who decided:** Opus 5.5 made the ranking and recommendation. The project owner
  accepted it, and it is recorded below.

What this means for trust in the result:

- The scores come from one LLM rater in one pass, and no human rater checked them.
  They are consistent with each other, but no human has checked that they are right.
- The sample is 50 records from one workflow (`agent_trace_observability`). The
  free-text ranking rests on only 16 distinct strings.
- The dataset card should say that the translator was chosen with an LLM rating
  (Opus 5.5 via Cowork), not with human review. This pilot does not replace the
  human review in plan 1.2 step 5.

## Results and decision

Scores are means on a 1–5 scale. "Overall" is the score after the backtick post-fix,
with the score before it in brackets. Free-text accuracy covers the `task` and
`constraints` units only.

| Model | Accuracy | Fluency (after fix) | Terminology | Overall | Free-text accuracy | Meaning flags (records) |
| --- | --- | --- | --- | --- | --- | --- |
| **gemma3-12b-4bit** | 4.47 | 4.33 | 4.59 | **4.46** (4.14) | **4.57** | 0 |
| gemma3-4b-bf16 | 4.62 | 3.99 | 4.71 | 4.44 (4.26) | 4.23 | 0 |
| translategemma-4b-bf16 | 4.03 | 4.24 | 3.33 | 3.87 | 3.80 | 0 |
| translategemma-4b-bf16-terms | 4.00 | 4.08 | 3.54 | 3.87 | 4.00 | 20 |
| qwen3-8b-4bit | 3.84 | 3.35 | 4.18 | 3.79 | 3.27 | 38 |

**Decision (Sep 30, this pilot): Gemma 3 12B (4-bit) is the translator for the full run.**
Its output is post-processed to strip the stray backticks. **Superseded Oct 2** (see
`project-plan.md` v15, 1.2): 12B is too slow on a free Colab T4 for this project's
session budget, so Gemma 3 4B is the translator for the whole dataset instead, with
no 12B fallback. This page keeps the pilot's own numbers and reasoning as the record
of what was measured.

- **Why Gemma 3 12B:** it translated free text most accurately, and it is the text
  in `state` that varies from case to case. None of its outputs changed meaning.
- **Gemma 3 4B:** effectively tied overall, and within the noise of this sample.
  It is the fallback. It is weaker on free text ("backfill" became "return") and
  leaves English words in the Persian.
- **TranslateGemma:** it uses the wrong domain terms (agent became "person", trace
  became "path"). The glossary variant did not fix this and added two meaning changes.
- **Qwen3-8B:** stays a checker only, as plan version 10 already decided. In 38 of
  50 records the meaning changed badly enough that the gold answer may no longer hold.

### Known problems to fix before the full run

- **Backticks:** Gemma 3 12B wraps 97% of units in backticks, and Gemma 3 4B wraps
  54%. The brief said "almost every" for both, which was wrong for 4B.
- **Risk-scale labels:** Gemma 3 12B translates the four labels wrongly. "Benign"
  became "sound sleeper", and both "Moderate" and "High" became "severity". The
  descriptions after the labels are correct. Fix the labels by hand.
- **Repeated text:** the whole English test set has only 94 distinct question,
  option, task and constraint strings. Translate each one once, review them by hand,
  and reuse them.
- **Other workflows:** they carry much more free text than this sample covered: 412
  distinct customer-thread messages, plus alert descriptions and evidence in the
  security workflow. Spot-check about 20 of those strings from Gemma 3 12B before
  starting the full run.
- **License:** Gemma 3 is under the Gemma terms, like TranslateGemma. The license
  check in plan 1.2 (whether Gemma output counts as a derivative) applies unchanged.
