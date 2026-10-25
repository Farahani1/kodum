# Kodoom model documentation on Hugging Face

The model repository is [Farahani/kodoom](https://huggingface.co/Farahani/kodoom).
The owner authorized the card, examples and suitable evaluation-result uploads
on October 25, 2026. Its existing visibility is **private** and its root model
card already declares **Apache-2.0**. Preserve those settings when updating
documentation. A documentation upload does not make the model public.

The seven-file upload is recorded in Hub commit
[`2bae4f3754825d216a62c795ec325c0284e5f6b8`](https://huggingface.co/Farahani/kodoom/commit/2bae4f3754825d216a62c795ec325c0284e5f6b8),
based on `88bffaacf58e336d5a10cfc0b1a3ab75a174e542`. The repository's file
inventory and private visibility were rechecked after that commit. The initial
upload response timed out and the service temporarily rate-limited reads;
checking the remote commit established success without repeating the write.
All seven files were then downloaded at that exact revision; their SHA-256
hashes matched the staged copies, including the unmodified CSV and JSON report.

## Layout and source mapping

| Local source | Path in the model repository | Purpose |
| --- | --- | --- |
| `MODEL_CARD.md` | `README.md` | Rendered Hugging Face model card. |
| `MODEL_CARD_EXAMPLES.md` | `docs/examples.md` | Original bilingual interface examples. |
| Supplied `summary.csv` | `eval/persian-shared-choice-v2/summary.csv` | Complete aggregate model/task metrics. |
| Supplied `report.json` | `eval/persian-shared-choice-v2/report.json` | Aggregate workflow/type/topic breakdowns and paired intervals. |
| Publication notes | `eval/persian-shared-choice-v2/README.md` | File index, protocol, full accuracy table and limitations. |
| Sanitized input manifest/sample metadata | `eval/persian-shared-choice-v2/provenance.json` | Model/data/code pins, environment, sampling counts and fingerprints. |
| Publication hashes | `eval/persian-shared-choice-v2/checksums.json` | SHA-256 hashes of the other six uploaded files. |

The root code-repository `README.md` remains the package guide. Copy the model
card to the Hub root without replacing that local package guide. Rewrite the
card's examples link to `docs/examples.md`, and the examples' return link to
`../README.md`, when staging the Hub layout.

Keep the existing `laya-full-v1/` and `laya-full-v1-preflight/` checkpoint
directories at their original paths. They are referenced by durable manifests
and pinned evaluation identities. Organizing documentation requires no checkpoint
movement, root weight copies, or deletion of historical files.

## Evaluation publication boundary

The supplied results come from the completed `shared-choice-persian-v2` run.
Before staging, verify all 25 original outputs against their manifest hashes
and match the 21 summary rows to the aggregate report. Publish `summary.csv`
and `report.json` byte-for-byte so metric values and original report identities
remain unchanged. These files contain aggregate metrics, not benchmark text.

Retain **all evaluated tasks** in this evidence bundle, including regressions
and inconclusive differences. The model card remains a selected capability
summary; the complete result files provide the wider context. Keep the Helmo
source-`train` diagnostic separate from held-out evaluations. The supplied
numeric evidence covers Persian only. The owner's report of similar English
improvements remains attributed, with English test details forthcoming.

The sanitized provenance retains source repositories and immutable revisions,
model and code pins, sampling limits and counts, dataset-lock fingerprints,
selected-record content hashes, environment and inference settings. Exclude
sample ID arrays, local filesystem paths, the full nested private dataset locks,
benchmark contexts/questions/options/gold, per-item prediction files, raw
`comparison.csv`, and the original `samples.json` and `manifest.json`. This
is an aggregate evidence release, not a replacement for private per-item
reproduction artifacts or a benchmark dataset release.

Stage generated results under git-ignored `runs/hub-publication/`, outside the
source commit. Upload an explicit seven-file allowlist in one Hub commit with
the inspected parent revision. Verify every uploaded file's bytes against local
SHA-256 and check the final file inventory and unchanged repository visibility.
Never persist an access token in staged payloads, scripts, local Git, or logs.

Hugging Face documents [root README model cards](https://huggingface.co/docs/hub/model-cards)
and [file uploads with explicit paths](https://huggingface.co/docs/huggingface_hub/guides/upload).
The aggregate evaluation files are supplementary artifacts alongside the card;
they do not claim submission to an external leaderboard.
