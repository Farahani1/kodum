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
  before downloading both the model assets and `dibaone_x1.py` from that same
  revision. The notebook uses the model card's PyTorch backend; there is no
  `dibaone` PyPI distribution. The adapter uses `mode="best"` and preserves
  request errors.
- ParsiNLU tasks: entailment, paraphrase, sentiment and multiple-choice QA.
  The loader reads only explicit test CSV/JSONL files from the official
  `persiannlp/parsinlu` Git commit
  `a3b08d0b93ca84e0017e2f490a2aa3ab36c79885`, bypassing legacy Hub scripts.
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
3. Save a private Kaggle notebook version/output to persist results. The
   initial run failed in bootstrap because it attempted to install a nonexistent
   `dibaone` PyPI package. The notebook now loads `dibaone_x1.py` and model files
   from the pinned public Hub revision using the documented PyTorch backend. A
   complete live rerun is still needed to validate Kaggle model loading and the
   external dataset layouts; converters fail closed if a source layout differs.
4. Preserve `results/report.json`, `results/<benchmark>/*.jsonl` and
   `results/manifest.json` together. A fresh run should use the same manifest
   revisions to reproduce the comparison.

The notebook deliberately does not upload benchmark rows, predictions or
reports to Hugging Face. Make any later sharing decision separately after
checking the source terms and removing any restricted derived content.

The second supplied Kaggle run reached code revision `1cd8a38` but stopped
before model download with `ModuleNotFoundError: No module named 'kodoom'`.
An editable install in a subprocess does not activate its new `.pth` file in
the running notebook kernel. Bootstrap now adds the checkout's `src` directory
to the kernel import path and checks that `kodoom` comes from that checkout.
Re-upload the corrected notebook and run from bootstrap in a fresh session.
Laya still loads `laya-full-v1/best/model.safetensors`, its tokenizer and
configuration from the frozen model revision above; this failure did not
reach or invalidate those artifacts.

The third supplied run stopped in dataset loading with `RuntimeError: Dataset
scripts are no longer supported`. The official entailment Hub dataset uses a
legacy Python loader rejected by datasets 4.x. Downgrading alone would still
leave invalid paraphrase/MCQ repository IDs and incompatible schemas.

The corrected loader reads the official immutable upstream files directly:
entailment `sent1`/`sent2` with e/c/n labels; query paraphrasing `q1`/`q2` with
0/1 labels; food and movie sentiment with the original aspect question and all
seven labels (including absent aspect and mixed sentiment); and MCQ candidates
with one-based answers. Empty trailing MCQ padding is trimmed without
changing answer indices; an empty interior or gold option remains an error. Entailment's explicit unlabelled `-` rows are excluded
and counted in the manifest. Unexpected labels still stop evaluation. Source
repository, file, original label and Git revision remain auditable; sentiment
aspect questions from a review share a source-case bootstrap group.

Push the corrected code branch before uploading the notebook. Use a fresh
Kaggle session because bootstrap reuses an existing checkout without fetching
updates. CPU validation covers the real held-out file conversion; GPU inference
and a complete Kaggle rerun still need validation.

Real-source CPU validation at the pinned commit converted 1,673 entailment
records (two unlabelled rows excluded), 1,916 paraphrase, 2,160 sentiment
(1,344 food and 816 movie), and 1,050 MCQ records. All converted IDs were unique
within each benchmark and all records retained the test-only source role.
