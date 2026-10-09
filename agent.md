# Notebook and accelerator execution workflow

Read this together with `AGENTS.md`. This file defines the shared execution
workflow; `docs/workflow-change-plan.md` tracks remaining migration work. The
current Kaggle runner covers the bulk draft campaign; a live Kaggle run and
fresh-session restore remain unverified.

Research acceptance follows project-plan v26 and
[the precompute audit](docs/precompute-project-review.md). The active request
retains the frozen v21 data recipe. The CPU-first notebook uses a newly published
runtime pin. At the owner's request, the canonical notebook now defaults to
RUN_TPU=True with the confirmed private HF dataset prefilled. Optional CPU
diagnosis uses RUN_TPU=False and Accelerator=None; normal imports launch the
approved draft translation after session setup checks. Do not silently change
a campaign identity to follow document revisions. Helmo is training-only and
optional; Gemma 3 translations remain ineligible for the planned permissive
training mix until a compatible licensing route is recorded. The source registry
alone does not enforce translator-derived model licensing.

The owner-selected active direction is the implemented
[bulk translation plan](docs/tpu-bulk-translation-plan.md), project-plan v21.
Generate unreviewed drafts to reduce queue waits, with BF16 batching and verified
private HF snapshots. BULK-01 through BULK-09 are implemented; live TPU, private
storage and a physical new-session handoff remain unverified. Human adoption,
release and training decisions have not advanced.

## Three separate sources of truth

- `docs/project-plan.md`: research scope, datasets, licenses, milestones and
  scientific decision gates. These do not advance merely because a workflow
  task is complete.
- `docs/workflow-change-plan.md`: the environment and notebook migration, with
  independent `WF-*` task IDs, dependencies and acceptance checks.
- `workflows/current.toml`: the active execution request, updated with
  each project stage. It records a main-plan reference, recipe IDs, parameters,
  prerequisites and a review stop. It is configuration, not another research plan.

Do not infer the active stage from the last notebook cell or from the existence
of an output file. The active step is bulk draft generation in main-plan section 1.2 (v21), using Gemma 3 27B BF16/JAX on Kaggle TPU v5e-8. Historical gate requests remain explicit recipes. The 4B adoption baseline and human review requirements remain in force.

## Two notebooks, one package

`notebooks/reference.ipynb` will be the complete, ordered recipe reference for
how artifacts are created. Include prerequisites, commands, inputs, output
locations, provenance and links to the main plan. Retain historical trials as
historical recipes, marking decisions that superseded them. Mark future steps
as unavailable until their package commands exist. The reference is for reading
and deliberately selecting recipes, not for running every experiment at once.

`notebooks/execution.ipynb` will be the short operational notebook used on Colab,
Kaggle and other explicitly configured Jupyter runtimes. Every stage update must
replace its active request and summaries, so Run all performs only the approved
current work. Completed pilots and optional experiments belong in the reference.
The operational sequence is settings, bootstrap, restore, readiness check,
current-stage execution, artifact report and persistence confirmation.

CPU preparation checks secrets before dependency installation, checks out the
exact revision without accelerator setup and runs lightweight checks in an
isolated CPU environment. It performs a tiny private HF write/read/delete probe,
validates attached Flax assets, writable paths and resume state. It does not claim legal eligibility,
load model weights, initialize JAX or acquire a campaign writer lease. Source/prompt/dependency audits are optional deep diagnostics. The saved
TPU session skips the isolated CPU environment and repeats short setup checks
because secrets and mounts are session-specific. Production validation remains
in the worker.

Both notebooks call the same tested `kodoom` package. Dataset processing,
translation, training, evaluation, recipe dispatch and artifact handling belong
in modules and CLI commands, not notebook cells. A shared recipe catalog prevents
the reference and execution notebooks from defining different commands.

## Session contract

1. Select the provider, explicit compute profile, exact code revision, active
   stage, run ID and storage paths. Never choose a workload by guessing the host.
2. Mount or attach storage and obtain credentials from provider secrets or
   environment variables. Never write tokens into notebooks, URLs saved in git,
   logs, artifact manifests or outputs.
3. Check out the requested revision and record the resolved full commit SHA.
   Stop on incompatible or dirty code instead of silently discarding edits.
   Install validated dependencies; keep GPU-only dependencies out of `dev`.
4. Restore explicitly selected input/checkpoint bundles into writable locations.
   Validate their manifests, checksums and semantic run configuration.
5. Validate prerequisites, writable paths, storage capacity, model access,
   actual accelerator capabilities and the selected precision. A failed check
   must stop Run all before any expensive stage starts.
6. Execute only the active recipe group through the package. Preserve the
   existing record, license, split and normalization rules.
7. In a finalization path, report new/changed files, failures and review findings,
   and update the existing data-directory README. Produce a run manifest with
   code revision, input revisions, parameters, model/prompt versions, dependency
   versions, output checksums and stage status. Do this on failure too.
8. Save or export approved private artifacts and verify the durable copy. Mark a
   stage complete only after execution and persistence succeed. A translation
   gate ends at `awaiting-review`, never by starting the full run automatically.

## Provider and storage boundaries

Keep provider bootstrap separate from computation. Colab mounts the owner's
Drive; Kaggle restores verified campaign snapshots into writable working space and saves periodically to its explicitly configured private HF dataset repo; private notebook output is an additional copy; generic platforms provide paths,
secrets and a persistence method explicitly. Local caches and checkpoint staging
are disposable, distinct from the artifacts selected for durable storage.

Writing a checkpoint to a Kaggle session disk is not proof that it will survive
session loss. Report the last verified durable checkpoint and the possible work
lost since it. Test saving and restoring across new sessions before a full run.
When moving between providers, preserve semantic configuration and input hashes;
changes to model, precision or data require a new run unless compatibility is
explicitly validated. Path changes alone should not invalidate a compatible run.

Existing data remains on the owner's Drive until a transfer is explicitly
authorized. Publishing this branch or a reference notebook does not authorize
uploading datasets, checkpoints, credentials or private logs to Kaggle or any
public service. Code and sanitized documentation go to git; generated artifacts
stay outside git. No dataset/model publication runs in the operational notebook.

## Agent handoff at every stage

Update the active request, execution notebook, recipe reference and workflow
documentation in the same logical change. State the main-plan step, required
inputs, expected files, review criteria, stop condition and restart instructions.
Keep prior versions reproducible through git and artifact manifests. Never claim
a stage, provider preflight or scientific gate passed without recorded evidence.
Follow `AGENTS.md` for tests and commits; use the separate `WF-*` task IDs when
reporting workflow implementation progress.

## Kaggle TPU handoff

Follow `docs/kaggle-tpu-change-plan.md` (TPU-* tasks). JAX imports and device
probes must run in children so the notebook kernel does not retain TPU devices.
Use the official versioned Flax asset and bundled tokenizer mounted under
`/kaggle/input`; never copy 27B weights to saved output or Drive. Namespace 27B
runs, enforce complete token budgets, and reject incompatible resume identities.
Keep backend packages lazy and the CPU dev path independent. The operational
notebook stops at human review and cannot approve the research gate or adopt 27B.
