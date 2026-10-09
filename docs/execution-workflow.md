# Run the resumable Gemma 3 27B campaign on Kaggle TPU

For account creation, tokens, model attachment, notebook settings and restart
steps, follow the [Hugging Face and Kaggle setup guide](kaggle-huggingface-setup-guide.md).

The execution notebook retains the project-plan **v21** data recipe, section 1.2, and the
[bulk plan](tpu-bulk-translation-plan.md). It generates unreviewed drafts of all
**1,600 typed cases (1,200 train / 400 test, 8,000 decisions)** and a deterministic
**2,000-record helmo selection**. Local checks pass. Real TPU fit, throughput,
private HF access and a second live session still need runtime evidence.

Before running, read the [project review](precompute-project-review.md) and
project-plan v26. The v21 request is preserved; the notebook's new runtime pin
adds CPU checks before requesting TPU time. The scientific interpretation is narrower:
typed-decisions measures teacher agreement, helmo is optional training
augmentation, and Gemma 3 drafts are held out of permissive-model training until
a compatible licensing route is recorded. A private completed draft campaign
does not establish release approval, source correctness or training readiness.

Record the actual input/output-limit failures, completed cases per hour,
automatic findings and last durable save in the first allocation. Validate
storage and batching before continuing; review the diagnostic outputs locally
as they arrive. The 40/20 diagnostic set is screening, not a random estimate of
the corpus error rate. All 400 test cases still need human review.

## Set up the notebook

Use one stable [configured notebook download](https://raw.githubusercontent.com/Farahani1/kodum/codex/tpu-bulk-resume-batching/notebooks/execution.ipynb).
It already contains the owner's confirmed dataset, tested runtime pin,
CAMPAIGN_ID, provider/backend/profile and RUN_TPU=True. No code edits are needed
for ordinary imports or clean resumes. Import it into a private Kaggle notebook,
enable Internet and HF_TOKEN, select TPU v5e-8, and accept/attach the exact model.
Then use **Save Version > Save & Run All**.

The early cell skips isolated CPU preparation. After installing the production
packages and probing devices, short checks verify access, storage, model mounts
and resume state before translation. The worker retains input/token/restore,
real library and model validation. A tiny private HF probe is added, read back
and removed; its two commits remain in repository history.

Optional CPU-only diagnosis uses RUN_TPU=False and Accelerator=None. That path
installs only the HF client into an isolated environment. Source/prompt/dependency
audits require explicit deep=True and do not run by default. CPU setup uses
venv without ensurepip and starts its checks without provider sitecustomize.
It does not prove TPU memory fit, ABI, kernels or throughput. A changed code pin
must remain compatible with saved campaign identity; never migrate it silently.

1. Import the [configured execution.ipynb download](https://raw.githubusercontent.com/Farahani1/kodum/codex/tpu-bulk-resume-batching/notebooks/execution.ipynb) from
   `codex/tpu-bulk-resume-batching` into a **private** Kaggle notebook. Keep its
   prefilled dataset ID and `CODE_REVISION` unchanged: it pins the implemented runtime and its frozen
   `workflows/tpu-bulk.toml` request. `workflows/current.toml` records the same
   active request in the handoff branch.
2. Enable **Internet**, select **TPU v5e-8**, and use a fresh kernel. The pinned
   runtime supports Linux x86_64, Python 3.11–3.13 and glibc 2.31+. Python 3.13
   selects `constraints/tpu-py313.txt`; older kernels use `constraints/tpu.txt`.
3. Accept Gemma's Kaggle terms and add the official
   [Gemma 3 27B Flax version 1](https://www.kaggle.com/models/google/gemma-3/flax/gemma3-27b-it/1)
   under **Inputs / Models**. Weights and the bundled tokenizer remain read-only
   under `/kaggle/input`. They are never included in checkpoints or saved output.
4. Create an existing **private Hugging Face dataset repository**. For another
   authorized operator to save progress, use an owner-administered organization
   repository and grant that person write access. Set `HF_DATASET_REPO` to
   `organization/dataset-name`. The code checks privacy and write access before
   loading weights; it does not create a repository or publish data.
5. Enable the **HF_TOKEN** Kaggle secret with write access to that dataset repo.
   Every operator uses their own token. A private GitHub checkout also needs a
   read-only **GITHUB_TOKEN** secret. Never put tokens in cells or share logins.
   An HF token does not authenticate the Kaggle Flax model; each operator needs
   their own Kaggle model access.
6. Choose a stable `CAMPAIGN_ID`, such as `gemma27b-bulk-v1`, and a short
   non-secret `OPERATOR` name. Keep `TAKEOVER=False` for ordinary starts/resumes.
   Keep `PROVIDER="kaggle"`, `PROFILE="kaggle-tpu"`, `BACKEND="jax"`.
7. `RUN_TPU=True` is prefilled. Use **Save Version → Save & Run All** for a server-side run. Bootstrap,
   secret lookup and restore all run in that new session. The laptop can sleep
   after submission; this does not extend Kaggle's limits. Inspect execution
   logs and the latest verified HF checkpoint while it runs.

The deadline is eight hours from the bootstrap cell, including setup, loading,
compilation and inference, with a 30-minute finalization reserve. Weekly quota
is separate from a session limit; use the limits shown for the actual account.
No automatic account rotation or quota workaround is implemented.

The attached Oct 8 failure occurred after a successful eight-device BF16 probe,
when `HF_TOKEN` was missing/inaccessible. It establishes neither model loading
nor a checkpoint save. CPU preparation now catches that error before TPU setup.
Its exported notebook also lists no attached input sources; confirm the exact
Flax model is attached. New code pins remain incompatible with older saved
campaign identities unless explicitly migrated; preflight reports this conflict.

## What Run all does

The thin notebook installs the pinned TPU extra and probes eight v5e devices
and BF16 arithmetic in a child. The notebook kernel never imports JAX. The
managed inference child has a hard deadline enforced by its parent launcher.

The worker acquires one campaign writer lease, restores a verified snapshot,
and prints completed/pending/failed counts before model work. A new campaign
fetches the complete pinned English sources, checks all typed split/workflow
counts, removes exact helmo duplicates and exact held-out-state matches, then
selects 2,000 records across question types/topics with seed 1234. Semantic
overlap still needs the research leakage review before training.

English IDs, revisions, splits, licenses, gold distributions and selection
hashes are frozen. Resume checks code, prompt/glossary/rules hashes, request,
checkpoint/tokenizer fingerprint, interpreter and accelerator library versions.
Changing paths or an authorized operator alone is compatible. Changing data,
model, precision, prompts, code or production batch requires a new campaign.

Before weights load, a tiny private commit is read back and restored into a
fresh directory, then its campaign and complete-case references are checked.
Every pending prompt is measured with the model tokenizer before weights load.
The request uses **3,072 input / 1,536 output / 6,144 cache tokens**. Inputs are
never truncated; outputs need an explicit end token.

One worker loads BF16 weights with eight-device FSDP sharding once. A first
session benchmarks **1, 2, 4, 8** on representative inputs, including the longest
prompt repeated at the candidate batch size. It records load/compilation time,
warm requests and tokens per second, per-device memory and a **512 MiB reserve**.
It freezes the fastest passing batch and comparison replies. A resume reuses
that selection; OOM never silently changes the batch or precision. If nothing
passes, fix the configuration in a new campaign after inspecting evidence.

Requests are bucketed by input length and mapped back to their work units.
Helmo state, question and descriptions go together as strict JSON; only text
values are accepted. Typed cases share one translated state and retain workflow
context. IDs, gold and metadata come from English inputs. Each complete typed
case needs all five decisions. Unfinished or malformed cases remain pending;
valid neighbors can be saved. Automatic check findings remain review flags,
and all translations remain `unreviewed-draft` with `human_reviewed=False`.

The first 40 helmo records and 20 typed training cases form the diagnostic
review set. Generation continues through the approved draft scope without
waiting for human review. Completed-unit rates appear in events. Quantization
is deferred. A restored campaign with no pending work skips loading entirely.

## Files and durable checkpoints

Local artifacts:

```text
/kaggle/working/kodoom/data/bulk/<campaign-id>/
  campaign.json       frozen request, source/input identity and code hashes
  inputs.json         complete frozen English work units
  production.json     selected batch, backend identity and benchmark evidence
  shards/<key>.jsonl  one atomic unit with English and Persian decision arrays
  progress.json       complete references/checksums, failed work, stop status
  events.jsonl        UTC attempts, completions, failures, saves and rates
  tpu-metrics.json    current attempt's load/compile/cache/memory measurements
  review/cases-*.csv  one row per decision with whole-case context
  review/labels-*.csv one row per option description with context
  result.json         final status, counts and last verified remote revision
```

These case shards are checkpoint envelopes, not the final released dataset
format. Research curation/release steps remain separate. The parent data
directory gets the standard generated README; per-attempt Run logs/checkpoints
are under `/kaggle/working/kodoom/runs`.

The private HF repository contains `campaigns/<campaign-id>/writer.json`,
`checkpoint.json`, optional `control.json`, and immutable `snapshots/<sha>.zip`.
A snapshot compresses all complete shards, English inputs, configuration,
events, metrics and review CSVs. One atomic commit updates its manifest with the
archive. Every artifact hash and the committed archive are verified on readback.
This avoids downloading thousands of tiny files on restart. Snapshots contain
no model weights, caches, credentials, checkout or unrelated Drive data.
Earlier snapshots remain available; storage grows with checkpoint history.

A background thread snapshots under a lock approximately **every ten minutes**
while inference runs. Cases are sealed locally immediately. Transient saves
retry with bounded backoff. After **30 minutes of unsaved completed work**,
the worker stops new inference and attempts final persistence. Normal stop and
failure also attempt a verified final save. The status report shows whether
that succeeded. An abrupt platform kill can lose work since the last verified
remote snapshot; local disk alone is not durable. Save private Kaggle output
as an additional copy. Live persistence is first checked inside the allocated
session; verify a separate real session restore at the next allocation.

## Resume in a fresh session

Import the same pinned notebook, attach the same model, enable your own secrets,
and select the same repo/campaign. Set the operator name for the person running
it. The package restores inputs and verified complete cases and computes only
pending work. It does not trust a completion log as proof of a translated case.

A normal stop releases its writer lease. An abruptly killed session leaves an
active lease: **confirm that worker ended**, then explicitly set `TAKEOVER=True`
for the next attempt. Set it back to False afterwards. Revision checks fence a
stale writer from replacing a newer checkpoint. Do not run simultaneous workers
for one campaign. Kaggle collaboration uses each person's own account and model
access; borrowing credentials is not a restart mechanism.

Corruption, missing shards, changed configuration or conflicting local/remote
content stops restore. Keep evidence and resolve the cause instead of deleting
or overwriting the contract. Failed cases are retried on the next compatible
attempt and never count as completed.

## Review during generation and pause

On a CPU machine, install the package and `huggingface_hub>=0.34,<1`, configure
`HF_TOKEN` securely in the environment, then use a **fresh directory**:

```python
from kodoom.bulk.launch import download

snapshot = download(
    campaign_id="gemma27b-bulk-v1",
    repo="organization/private-dataset",
    directory="data/bulk-review/snapshot-001",
)
print(snapshot)
```

This downloads/validates durable artifacts without acquiring the inference
lease. Copy the review CSVs elsewhere before editing; later downloads should
use another fresh directory. Corrections remain separate from original drafts.
The running worker does not import or approve human annotations automatically.

Read state/question context when judging labels: negation, severity ordering,
actors/actions, conditions and technical terms. Fill `meaning_changed` and
`needs_edit` with yes/no, plus category, suggested wording, notes and reviewer.
Count distinct typed source IDs for case-level rates; each case has five rows.
All 400 test cases require human review, alongside the train/helmo samples in
the main plan. The balanced diagnostic set cannot estimate dataset-wide noisy
label coverage. Whole-record prompts require fresh quality review; they are
not directly equivalent to the historical fieldwise gate prompt.

To pause a live campaign explicitly (observed at the next periodic save):

```python
from kodoom.bulk.launch import control

control(campaign_id="gemma27b-bulk-v1", repo="organization/private-dataset", pause=True)
```

Use `pause=False` before resuming. The worker finalizes verified progress when
it observes a pause. Neither completion nor an automatic check adopts 27B,
approves releases, changes gold, starts training or publishes data.

## Local validation and historical recipes

The CPU fixture run needs no GPU, model download or cloud access:

```text
python -m kodoom.bulk --dev --provider generic --profile dev --campaign smoke-campaign --root data/bulk-smoke/first --local-remote data/bulk-smoke/remote --max-units 2
python -m kodoom.bulk --dev --provider generic --profile dev --campaign smoke-campaign --root data/bulk-smoke/next --local-remote data/bulk-smoke/remote --operator collaborator
```

This local directory adapter exercises the artifact contract, not cloud
durability or model quality. The automated suite also injects upload outages,
corruption, writer conflicts, pause requests and unfinished replies.

`workflows/tpu-gate.toml` retains the bounded 27B experiment and
`workflows/gpu-gate.toml` the 4B CUDA gate. Select either explicitly with
`kodoom-workflow --request PATH` and its matching profile. The full recipe
history is in [reference.ipynb](../notebooks/reference.ipynb).

Live batch fit, timing, representative Persian output inspection and a physical
fresh-session handoff remain unverified until recorded. They do not block
importing the prepared notebook; runtime readiness checks block generation on
failure. Record those results in the bulk plan before claiming hardware success.
