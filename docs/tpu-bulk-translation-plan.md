# Resumable bulk translation on Kaggle TPU

Version 1 | Oct 7, 2026 | Proposed; implementation and live validation pending

Branch: `codex/tpu-bulk-resume-batching`.
Baseline: `bc0c679` on `codex/kaggle-tpu-v5e-8`, including the correction for
Kaggle's observed Python 3.13 kernel.

This extends [project-plan.md](project-plan.md), section 1.2 (v20), and the
[TPU implementation plan](kaggle-tpu-change-plan.md). Tasks use `BULK-*` IDs.
This document records a development direction, not a completed implementation.
The existing execution notebook still runs the bounded 27B experiment.

## Why the execution approach changes

The owner reports long waits for Kaggle to allocate a TPU. Repeatedly acquiring
a session, translating a small gate, stopping for review and acquiring another
session spends too much calendar time waiting. Local progress alone is also
insufficient when the next run starts on a new machine or with a collaborator.

On Oct 7 the owner chose to plan one bulk translation campaign, continuing for
as much of each allocated session as practical. Results are saved remotely
throughout the run and reviewed on the laptop as they arrive. Restarting means
restoring the same campaign and doing its unfinished work, rather than repeating
completed translations. Generating these drafts before human gate acceptance
is an explicit change in sequencing; acceptance into a release or training mix
still requires the existing quality and licensing decisions.

Gemma 3 27B is the chosen candidate for this campaign because it is the strongest
translator currently being prepared for the available TPU. Its Persian label
accuracy and live throughput are not yet established. The owner also chose
**batching before quantization**: improve throughput while retaining BF16 and
measure the result before introducing another numerical configuration.

A campaign may require multiple sessions. Kaggle documents 20 TPU hours per
week and 9 hours per session, also repeated in its Tunix competition guidance.
Use the limits shown for the actual account, with an initial total run budget
of at most 8 hours from session start, including a 30-minute finalization reserve.
Queue waits and weekly allowance resets affect calendar time. The previous
rough batch-one time estimate is a planning assumption, not a TPU benchmark
or evidence that the campaign fits one session.

Sources: [Kaggle TPU limits](https://www.kaggle.com/docs/tpu),
[Tunix guidance](https://www.kaggle.com/competitions/google-tunix-hackathon/overview/description),
[Kaggle notebook execution](https://www.kaggle.com/docs/notebooks).

## Workload and research boundaries

| Source | Campaign scope | Required treatment |
| --- | --- | --- |
| LocalLLaMA/typed-decisions | All 1,600 cases: 1,200 train and 400 test; 8,000 decisions | Preserve source case IDs, all five decisions per case, source splits, option IDs and gold distributions. |
| helmo/synthetic-typed-decisions | Deterministic 2,000-record selection within the planned 1,500-2,000 starting range | Freeze a representative selection across question types and topics, with its source revision, selection method and seed; retain the source registry's training-only treatment. |
| Existing Persian sources and generated skills | Outside this translation campaign | Do not translate them or dispatch training/evaluation. |

Freeze the exact input revisions, IDs, counts and hashes before the bulk phase.
Fetching a capped gate sample cannot satisfy the full typed-decisions scope.
The full 9,879-record helmo corpus and later 3,000-5,000 expansion are separate
scope decisions after review. No silent expansion occurs on resume.

Generate the 40-record/20-case diagnostic review set with the campaign's actual
production prompt and settings near the start. Save it for paired comparison
with 4B and continue producing **unreviewed drafts** after technical readiness
checks. Human review can take place outside the TPU session. Review failure
blocks adoption and can pause further work through an explicit campaign control;
it does not make previously generated drafts approved.

Retain source/license/split/origin metadata, keep-fields, numeric and Latin-token
checks, glossary checks and orthographic cleaning. Never change a gold answer
to accommodate a changed translation. All 400 test cases still require human
review; train and helmo samples retain the requirements in section 1.2 step 5.
Human-readable option descriptions receive their own review with state/question
context, including negation, severity, actors, actions and conditions.

## Batching and model lifetime

- Retain the versioned Gemma 3 27B Flax checkpoint, bundled tokenizer, native
  JAX sharding and BF16 weights/compute. The pinned CPU development path remains
  independent of accelerator imports.
- Load weights once in one managed inference worker per session, covering
  both datasets and splits. Probes and generation remain outside the notebook
  kernel. Skip loading entirely if restored work is already complete.
- Benchmark batch sizes 1, 2, 4 and 8 on fixed representative short and long
  inputs, recording actual peak per-device memory, compilation time, warm
  generated tokens/second and completed cases/records per hour. Select a batch
  size that fits with a memory reserve and improves completed-work throughput.
  Larger batches are candidates, not promised capacity or speedups.
- Bucket requests by measured input length; bound padding and cache allocation.
  Preserve independent conversations and map replies back to explicit unit IDs
  before rebuilding cases. Batched decoding must handle different end times,
  output limits and the final partial batch without saving padding or unfinished
  replies. Never truncate input to make a batch fit.
- Resolve the existing helmo granularity gap before bulk activation: section
  1.2 requires state, question and descriptions together, while the current
  diagnostic adapter translates fields independently. Implement a versioned,
  validated whole-record structured helmo request for production. Only text
  fields are accepted from the model; IDs, gold and metadata come from the
  source. Batching groups these independent requests. Preserve the existing
  field-based experiment as a historical baseline and review the new prompt.
- Keep typed-decisions' shared state and workflow context when batching its
  translation items. Deduplicating repeated labels solely by English text is
  outside scope because their correct wording can depend on context.
- Verify singleton/batched ordering and content with deterministic fixtures,
  then inspect representative real outputs for batch-dependent changes. Freeze
  the production batch, prompt and decoding configuration before bulk work.
  An OOM or configuration change must not silently alter a resumed campaign.

INT8 or other quantization is deferred. Consider it only if measured BF16
batching cannot meet a useful throughput/memory budget. A later quantized trial
needs compatible JAX operations, a separate configuration identity, measured
speed and paired Persian label review. Smaller weights do not establish faster
execution or unchanged translation quality.

## Durable progress, logs and resume

Human-readable logs explain progress. Verified output shards and their
machine-readable manifest determine whether work is complete.

| Artifact | Contents and purpose |
| --- | --- |
| `campaign.json` | Immutable campaign ID, schema version, exact code/model/tokenizer/input revisions, selection, prompt/glossary/rules hashes, decoding settings and frozen production batch. |
| `shards/` | Immutable JSONL batches of completed cases/records, retaining English-to-Persian correspondence, checksums and review status. |
| `progress.json` | Versioned checkpoint: referenced shards and hashes, complete/pending/failed counts by dataset and split, last verified remote revision and next work. |
| `events.jsonl` | UTC attempt events with stable work IDs, completion/failure reasons, upload results, timing and stop reasons. Never credentials. |
| `review/` | Case and option review exports, findings and separate human decisions; append corrections without replacing original drafts. |

1. A typed-decisions case is complete only when all expected decision IDs have
   valid translated outputs. The existing per-record append loop cannot provide
   that atomicity: a crash after the first of five records must not mark the
   whole case done. Stage a full case, seal a shard with an atomic local rename,
   and publish its completed IDs only after validation.
2. Save each completed case/record locally immediately. Seal and upload pending
   shards approximately every 10 minutes, and at normal stop/failure. Snapshot
   only complete units; concurrent upload must not read a changing output file.
3. Upload immutable shards and their checkpoint in a consistent remote revision.
   Verify the committed contents/hashes before reporting durable completion.
   Retry transient upload failures with bounded backoff. Expose upload failures,
   the last successful save time and the count of locally completed but unsaved
   units; pause generation after a configured maximum unsaved-work interval
   (initial default: 30 minutes).
4. On startup, download a selected verified checkpoint and its referenced shards
   into a fresh writable root. Check campaign identity, input hashes, shard
   checksums, complete case membership and duplicate/conflicting IDs. Derive
   pending work from the frozen inputs minus verified complete units.
5. Lost or unfinished units may be regenerated. Valid committed cases must never
   be retransmitted to the model. Neither a completion line in a log nor a lone
   case record is sufficient proof of completion.
6. One inference writer owns a campaign at a time. Use remote revision checks
   and an explicit handoff to prevent a stale session overwriting newer progress.
   Concurrent collaborative workers and automatic account switching are outside
   scope; their conflict/shard assignment protocol is not implemented here.

Default proposed destination: an **owner-controlled private Hugging Face dataset
repository**, supporting incremental file commits and restoration independently
of the Kaggle notebook owner. For collaborator uploads, use an owner-administered
HF organization repository with appropriate write access and each collaborator's
own token: a personal repository does not provide another user direct write
access. A private Kaggle dataset can be a later adapter.
Keep final private notebook output as an additional copy. Local `/kaggle/working`
alone does not establish persistence. Never upload weights, checkout caches,
credentials or unrelated data.

The owner must select the destination repository and authorized collaborators
before enabling uploads. This planning change creates no external repository,
transfers no existing Drive data and grants no publication authorization.
Keep per-operator credentials in provider secrets; do not embed or transfer them
in bundles. The shared artifact contract, not the original username or absolute
path, identifies the campaign.

Sources: [Hugging Face uploads](https://huggingface.co/docs/huggingface_hub/guides/upload),
[private repository access](https://huggingface.co/docs/hub/repositories-settings),
[organization roles](https://huggingface.co/docs/hub/organizations-security),
[Kaggle dataset versioning](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).

## Fresh-session and collaborator handoff

The next session needs the pinned notebook/code, the frozen campaign request,
the private repository/checkpoint revision, permitted access to its inputs and
the Gemma model, and its own secrets. It should print completed/pending/failed
counts and the last durable save before allocating work to the model.
Every attempt records a separate attempt ID and operator/provider details;
changing an authorized operator or a storage path alone must not discard work.

Kaggle's [terms](https://www.kaggle.com/terms) prohibit one person having,
controlling or operating multiple active accounts, and multiple people using
one login. A friend participates by operating their own account and an explicitly
authorized private collaboration workflow. Do not document borrowing credentials,
automated account rotation or quota bypass as restart mechanisms. Each operator
remains responsible for platform limits and model access.

Use **Save Version -> Save & Run All** for the background run; that creates a
fresh server-side session independent of the laptop. Include all bootstrap,
restore and secret lookup steps in the thin notebook. Sleep or browser closure
does not extend platform deadlines or remove the need for periodic saving.
Count setup, model loading, compilation and translation in the session budget.
Stop launching new work before the finalization reserve, finish or abandon the
in-flight unit safely, verify the final remote checkpoint and exit normally.
Hard session termination can still lose work since the last durable checkpoint.

## Implementation tasks and acceptance

All tasks below are **pending**. Complete them through package modules/CLI and
thin notebook calls; do not place computation or persistence logic in cells.

| ID | Work | Acceptance evidence |
| --- | --- | --- |
| BULK-01 | Add the frozen bulk request, full source fetching and campaign identity | Exact 1,200/400 typed case counts, all decision IDs and 2,000 selected helmo IDs; revisions and hashes recorded; dev fixture has small caps. |
| BULK-02 | Add the structured helmo production prompt and batch-aware JAX adapter | Validated complete replies with untouched keep-fields; independent conversations, ordering, EOS/output limits and final partial batches covered. |
| BULK-03 | Add one inference worker, length bucketing and batch benchmark | Batch 1/2/4/8 measurements, load/compile/memory separation, safe chosen batch and representative output comparison on the real TPU. |
| BULK-04 | Add atomic case shards and reconstructible progress/events | Kill mid-case and mid-shard; no incomplete case skips, missing decisions, duplicated IDs or false completion on restart. |
| BULK-05 | Add private HF save/restore adapter and verification | Test upload retry, inconsistent checkpoint, corruption, stale revision and missing access; live tiny save/restore proves the selected storage. |
| BULK-06 | Add timed checkpointing, unsaved-work limit and session deadline | Uploads occur during inference; failures are visible; fake-clock tests cover graceful stop and finalization reserve. |
| BULK-07 | Add portable fresh-session/collaborator handoff | A fresh root with another authorized operator restores the same campaign; credentials/paths stay outside semantic identity; conflicting writers fail clearly. |
| BULK-08 | Add incremental review exports and pause control | Local CPU review sees case and label context from durable drafts; findings and human decisions remain distinct; no automatic adoption or training. |
| BULK-09 | Update execution/reference notebooks, active request and handoff docs | One Run all bootstraps, restores, benchmarks/selects the frozen configuration, translates pending work, periodically saves and finalizes; historical gate remains selectable. |
| BULK-10 | Validate locally, on the TPU and across a real restart | Fast checks pass; interrupted/resumed fixtures equal an uninterrupted run; live 27B batch fit, remote persistence and new-session restoration are recorded before a long run. |

For BULK-09, reuse recorded matching benchmark evidence on resume rather than
changing the batch each session. A new benchmark decision/configuration needs
a new campaign identity or an explicitly tested compatibility migration.

Completion means the selected scope is durably translated and reviewable, with
all expected units accounted for. Cases that fail generation/validation remain
listed as failed or pending; they do not count as translated. Operational status
is `drafts-complete-awaiting-review`, not translator adoption, release approval
or training readiness. Hardware throughput and persistence remain unverified
until the live evidence above exists.

## Version history

| Version | Date | Change |
| --- | --- | --- |
| 1 | Oct 7, 2026 | Plan bulk draft generation to reduce repeated TPU queue waits, with BF16 batching before quantization, atomic cases, verified remote saves and portable collaborator/session resume. |
