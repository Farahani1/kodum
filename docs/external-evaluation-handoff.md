# Kaggle three-model sample comparison

Run `notebooks/external-evaluation.ipynb` to compare **Laya multilingual**,
**Kodoom** (our full-data Laya best checkpoint), and **DibaOne X1** on identical
selected Persian samples. The request is `workflows/model-evaluation.toml`.
Sampling, consumption, inference and reporting live in `kodoom.model_comparison`;
the notebook configures and calls the package.

## Frozen inputs

| Input | Revision / selection |
| --- | --- |
| Multilingual Laya | `convaiinnovations/laya-multilingual`, `1720e3e3357cfe1e281542e223f8273b0890ca34` |
| Kodoom | private `Farahani/kodoom`, `f315b5da129e2b1df55d971ddf2053c3cbfe5282`, `laya-full-v1/best/` |
| DibaOne X1 | `Dibachain/DibaOne-X1`; resolve once before inference, record exact SHA for code and assets |
| ParsiNLU | official `persiannlp/parsinlu` Git commit `a3b08d0b93ca84e0017e2f490a2aa3ab36c79885`; explicit test files |
| PersianQA | official `sajjjadayobi/PersianQA`; record resolved Git commit; explicit test/dev/validation files |
| Translations | private `Farahani/kodoom-en-fa`, `1b98bbba48043c62c6ae2c821a55b6f8a5e7a927`; accepted release 1.0.1 and manifest hash from `configs/dataset-baseline.toml` |

Kodoom inference files must match the durable checkpoint manifest. The bilingual
release uses existing inventory, schema, source-role and hash checks. No new
translations, dataset corrections or training occur here.

## Sample selection

Settings expose `SEED=17`, `NATIVE_CASES_PER_GROUP=100`,
`TRANSLATED_CASES_PER_GROUP=3`, `INCLUDE_HELMO=True` and `MICROBATCH=4`.
Limits count **source cases**, so question counts can exceed the limit.
A stable seeded hash selects cases independent of row order before model loading.
All questions in each case are retained; every model receives identical records
and option order. Small groups use all available cases.

- ParsiNLU: entailment, paraphrase, sentiment and MCQ; food/movie sentiment
  domains are sampled separately, retaining every selected review's aspect
  questions. Official unlabelled entailment rows are excluded and counted.
- PersianQA: held-out answerability decisions, preserving source split.
- Translated typed-decisions: **test only**, sampled per workflow, retaining
  choice, noul and score questions. These are represented-workflow held-out
  cases for Kodoom, which trained on typed-decisions train data.
- Translated Helmo: sampled per topic from **source train**. This is a separate
  train-only-source diagnostic, never a held-out score. The recorded full-data
  recipe selected typed-decisions only, but the source split must still not be
  renamed test. Set `INCLUDE_HELMO=False` to omit it.

`samples.json` records selected IDs, group counts, locks, seeds and content hashes.
`selected-samples.json` is also saved before inference. Outputs omit passages.
Preserve exact input revisions, settings and manifests to reproduce a comparison.

## Shared inference protocol

X1 is choice-only. All three models receive the **same choice conversion**,
preserving gold distributions, option IDs/order/text and source split. Original
types and `noul-to-choice-for-all-models` / `score-to-choice-for-all-models`
protocols are retained in reports. These results do not measure native noul/score
heads or ordinal score metrics. ParsiNLU classification uses labelled choices;
MCQ retains native choices; PersianQA is `answerability-to-two-choice`.

All three use `kodoom.normalize.normalize` at model input; released text remains
unchanged. Laya/Kodoom use temperature 1; X1 uses its shipped calibration.
Nothing is fitted on benchmark samples. Laya prediction metadata retains token
budgets/truncation. GPU inference timing excludes download/load; Laya batch time
is amortized per question and X1 timing covers its decision API. These are not
matched end-to-end application latency measurements. Models load sequentially.

Reports separate benchmarks, workflow/topic and original question type. Each
model reports accuracy, soft accuracy, NLL, Brier, KL, ECE, latency, coverage,
missing requests and failures. Each pair has a common-item accuracy difference
with source-case bootstrap intervals; three-way common-item metrics are included.
There is no pooled native/translated/Helmo ranking.

## Kaggle run and restart

1. Push `codex/evaluate-parsinlu-persianqa` before uploading the updated notebook.
   Open it as **private**, enable **Internet** and a **GPU T4**. One GPU suffices.
2. Enable the `HF_TOKEN` Kaggle Secret with read access to both private inputs.
   Tokens are never saved in notebook source.
3. Choose sample limits and Run all in a fresh session. The notebook records
   the exact code SHA and resolved model/data revisions.
4. Inspect `/kaggle/working/kodoom-three-model-eval/results/`:
   `comparison.csv` puts all three answers/probabilities/errors beside gold;
   `summary.csv` contains aggregate metrics; `report.json` contains breakdowns
   and paired results. Each benchmark has `laya-multilingual.jsonl`,
   `kodoom.jsonl` and `dibaone-x1.jsonl`.
5. Preserve `samples.json` and `manifest.json` with the outputs. Model-load
   errors become failed predictions, allow other models to finish, and produce
   a failed manifest plus final exception. Item errors reduce coverage.
6. The final notebook cell creates `WORK/results.zip`, containing the entire
   `results/` directory, and displays a download link. Save a **private Kaggle
   version with outputs** to preserve the archive; local disk is temporary.
   For another run use a fresh session and new `WORK` path; existing results
   are refused. Set `DIBA_REVISION` from the prior manifest and
   `PERSIANQA_REVISION` from samples.json to retain those pins.
   There is no automatic partial inference resume.

The notebook does not upload datasets, predictions, weights or reports.
ParsiNLU/PersianQA retain their test-only/private source policy. Historical TPU
and training notebooks keep their recipes. Rebuild this notebook with
`python scripts/build_comparison_notebook.py`.

## Validation boundary and retained fixes

CPU tests cover case sampling, conversion, split preservation, three-model
dispatch, empty/common coverage, model-load failures, hashes and bootstrap.
The accepted local release verified 2,000 typed-decisions test questions and
2,000 Helmo source-train questions. Default sampling selects 60 questions from
four typed-decisions workflows and 621 questions from 207 Helmo topics.
A live Kaggle GPU rerun is still required; fixtures do not validate real-model
loading or throughput.

Earlier fixes remain: there is no `dibaone` PyPI package (use the pinned Hub
module); an editable install's `.pth` does not activate in a running kernel
(add checkout/src); datasets 4.x rejects ParsiNLU loader scripts (read pinned
official test files). Real-source conversion previously yielded 1,673 entailment
rows after excluding two unlabelled rows, 1,916 paraphrase, 2,160 sentiment and
1,050 MCQ rows. Those totals describe full pools, not selected sample sizes.
