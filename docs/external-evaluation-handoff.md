# Kaggle external benchmark evaluation

This handoff compares the selected full-data Laya checkpoint with DibaOne X1
on held-out ParsiNLU and PersianQA data. The operational notebook is
`notebooks/external-evaluation.ipynb`; the selected request is recorded in
`workflows/model-evaluation.toml`.

## Frozen inputs and scope

- Laya checkpoint: private `Farahani/kodoom`, complete snapshot revision
  `f315b5da129e2b1df55d971ddf2053c3cbfe5282`, inference files under
  `laya-full-v1/best/`. The notebook verifies downloaded files against the
  snapshot manifest before loading.
- Diba baseline: `Dibachain/DibaOne-X1`; its resolved Hub revision is saved
  before inference. The adapter uses `mode="best"` and preserves request errors.
- ParsiNLU tasks: entailment, paraphrase, sentiment and multiple-choice QA.
  Each dataset revision is resolved once and used to load its held-out split.
- PersianQA: the notebook records the cloned Git commit and reads only files
  explicitly named test/dev/validation. Train files are never selected.
- No source is added to model training, no final-test tuning is done, and no
  raw source text is saved in prediction outputs.

The converters mark each row as project split `test`, retain the source and
revision, and enforce the source registry's test-only role. Unknown schemas or
integer labels without their dataset's ClassLabel names stop the conversion.
Converted benchmark records remain private because their source licenses do not
authorize publishing derived data.

## Comparison protocol

ParsiNLU classification tasks are represented as labelled choices; its MCQ task
retains its original options. PersianQA is an answerability task: answerable
examples map to `بله`, unanswerable examples to `خیر`. This is converted to a
two-choice request for Diba X1 and is explicitly tagged in records and reports.
It is not evidence of native X1 yes/no support. Do not compare metrics across
different protocols as if they were the same task.

Both models receive the same converted records and option order. The report
contains per-model accuracy, soft accuracy, NLL, Brier, KL, ECE, latency,
coverage, failures, and the same-item model comparison with a paired
source-case bootstrap interval. Diba errors remain visible in the prediction
file and are excluded only from the scored count; the coverage denominator is
the complete benchmark. Probabilities are reported raw; the evaluation test set
does not fit a temperature or guide checkpoint selection.

## Run and restart

1. Push branch `codex/evaluate-parsinlu-persianqa`; upload the notebook to Kaggle
   as private, enable Internet and a T4, and add `HF_TOKEN` as a Kaggle Secret
   with read access to the private model repository.
2. Run all cells. The code checkout, model revision and all dataset commits are
   resolved and recorded in `/kaggle/working/kodoom-eval/results/manifest.json`.
3. Save a private Kaggle notebook version/output to persist results. A live
   Kaggle run has not yet verified the external repository layouts or Diba
   runtime installation. The converters fail closed if a source layout differs.
4. Preserve `results/report.json`, `results/<benchmark>/*.jsonl` and
   `results/manifest.json` together. A fresh run should use the same manifest
   revisions to reproduce the comparison.

The notebook deliberately does not upload benchmark rows, predictions or
reports to Hugging Face. Make any later sharing decision separately after
checking the source terms and removing any restricted derived content.
