# Portable notebook workflow: change plan

Version 1 | Oct 5, 2026 | Status: design recorded; implementation pending

Baseline inspected: commit `02a69bc` on `claude/sweet-franklin-pm0tdm` in
`D:/Coding/kodum`. Design branch: `codex/portable-notebook-workflow`.

This is a tooling migration plan, separate from `project-plan.md` (research,
milestones and data rules) and `translation-eval-plan.md` (the historical pilot).
Tasks here use `WF-*` IDs, not M1/M2/M3 or research section numbers. Completing
them does not complete a research milestone. `../agent.md` defines the workflow
agents must maintain.

## Problem and scope

`notebooks/colab.ipynb` currently has 55 cells. Setup, source fetching, alternative
translator trials, a completed pilot, cache removal, the current free-text gate
and optional external translation are mixed together. Run all can execute obsolete
experiments and an import that requires a manually filled file. The user has to
find the appropriate cells at every stage.

The target is one complete reference notebook plus one compact execution notebook
whose active stage is updated as the project advances. Both use the same package
and recipes on Colab, Kaggle and explicitly configured Jupyter environments.
Moving to Kaggle must not silently change datasets, model choices or research
gates. This branch records the design; it does not yet create the new notebooks,
add Kaggle profiles, move data or run a GPU job.

## Existing foundations and gaps

| Existing code | Reuse | Change needed |
| --- | --- | --- |
| `config.py`, `profiles/*.toml` | Explicit profiles; custom TOML paths already work | Kaggle profiles and portable storage configuration; retain current names |
| `cli.py` | Data, translation, review and metric commands | Active-stage dispatch and explicit prerequisite checks |
| `check.py` | Writable paths, GPU and space checks | Provider-neutral errors, secret/access checks and persistence readiness |
| `runs.py` | Resume IDs, latest/best checkpoints and manifests | Durable export/restore and compatibility checks across provider paths |
| `datadir.py`, `kodoom tree` | Artifact README and before/after reports | Run-scoped provenance and failure finalization, without a second tree writer |
| `translate/pipeline.py` | Per-case resumable translation | Stage-specific artifact namespaces and explicit review status |
| `tests/test_notebook.py` | Thin cells, explicit profiles, empty outputs | Both notebooks, shared recipes and stopped/failed execution |
| `pyproject.toml` | Lightweight CPU package and optional dependencies | Provider-neutral GPU extra and validated dependency constraints |

Important gaps: no complete smoke-pipeline command or training implementation
exists yet. Do not invent executable recipes for these future steps. Profiles
currently allow fp32/fp16, while the chosen translator is named
`gemma3-4b-bf16`; make model-specific dtype validation explicit instead of treating
the profile precision as proof that the translator is compatible with a GPU.

## Target layout

All paths below, except existing files, are proposed until their task is complete.

| Path | Responsibility |
| --- | --- |
| `notebooks/reference.ipynb` | Full artifact recipes, dependencies, main-plan links, historical decisions and future placeholders |
| `notebooks/execution.ipynb` | Short Run all entry point for the active request; provider settings and summaries |
| `workflows/recipes.toml` | Ordered, named CLI recipes, declared inputs/outputs, parameters and status |
| `workflows/current.toml` | One active stage, main-plan reference, selected recipe IDs, review stop and expected artifacts |
| `src/kodoom/workflow.py` | Recipe validation, execution, status and finalization using existing modules |
| `src/kodoom/runtime.py` | Provider bootstrap/storage contract; explicit Colab, Kaggle and generic adapters |
| `src/kodoom/profiles/kaggle-preflight.toml`, `kaggle.toml` | GPU workload limits and paths for the first supported Kaggle configuration |
| `docs/execution-workflow.md` | Operator instructions, provider setup, restart and stage handoff |
| `<artifact_root>/.kodoom/executions/<execution_id>/` | Private resolved request, provenance, status and artifact manifest; never tracked |

Start with TOML, stdlib subprocess/pathlib and existing CLI commands; do not add a
workflow framework. The proposed workflow CLI must support inspection/dry-run
before execution. CLI names and schema are finalized in WF-02, not promises of
currently available commands.

### Notebook responsibilities

The reference covers setup, generators, source acquisition, translation trials,
pilot, free-text gate, full translation, checking/review, packing/publication,
training mix, models, calibration, evaluation and reports. Distinguish available,
historical, blocked and not-yet-implemented recipes. Historical trials remain
reproducible at their code/model revisions and retain the measured decisions.
They are never dependencies of the current gate merely because they appear first.
Do not automatically execute the whole reference; rendering it must not download
models or publish private artifacts. Commit empty cell outputs.

The execution notebook should have roughly 6-8 code cells, with a small settings
cell and one package invocation for the active stage. Generated summaries may be
maintained by a small rendering/check script. Every stage change updates
`current.toml` and these summaries together; CI checks for drift. The active
notebook must not grow into a second recipe catalog. No interactive input or
manual cell selection should be required inside Run all; unmet review or input
requirements stop before the stage runs.

### Configuration and reproducibility

Keep three concepts distinct: the research stage, compute workload profile and
provider storage/bootstrap. The user selects all explicitly; environment probes
validate that choice but never silently select a workload. Preserve `dev`,
`colab-preflight`, `colab` and custom TOML profile support during migration.

Resolve the requested branch/tag to a full SHA once per execution and record it.
Use that SHA throughout a session; production requests should pin it. Record the
recipe/request revision separately so an exported notebook is attributable even
when it loads a later package. A bootstrap must reject incompatible recipes or
dirty checkouts rather than force-checkout edits. Document how to update the
package revision at a stage handoff without manually maintaining duplicate hashes.

Each execution records input dataset revisions/checksums, sampling seed, source
IDs/splits, translator/model revision, prompt/glossary version, actual dtype and
device, package versions, commands, output paths/checksums and status. Credentials
are excluded. `data_dir/README.md` still describes the folder; the new manifest
describes one execution. Scope limited gates and full translations separately so
a partial gate output cannot be mistaken for the completed dataset.

### Storage and restart contract

| Environment | Inputs and writable artifacts | Disposable files | Durable confirmation |
| --- | --- | --- | --- |
| Local / generic Jupyter | Explicit user-owned input, data and runs paths | Explicit local cache/scratch | Verify writes on persistent storage, or a configured export backend |
| Colab | Owner's mounted Drive for data/runs | Runtime-local checkout, HF cache and staging | Verify checkpoint manifests on Drive; sync at normal session end |
| Kaggle | Attached inputs; copy selected resume/data files to writable `/kaggle/working/kodoom` | Runtime-local checkout/cache/scratch outside saved outputs | Explicit private version/output save or owner-approved external export; verify restore in a new session |

Kaggle's [notebook documentation](https://www.kaggle.com/docs/notebooks) describes
saved outputs and reuse as inputs, and a fresh session for Save & Run All. Design
against that fresh-session behavior, not hidden interactive state. Saving notebook
source alone is not the durability test. Exact quotas and available hardware are
checked at runtime rather than hard-coded from Colab assumptions.

Do not promise recovery of every locally written Kaggle checkpoint after a killed
session. Without an approved continuous remote store, only the last verified
export is durable. Report its time/step and the unexported progress. A normal-end
save and a fresh-session restore are required before full workloads; document the
remaining loss window. Use private storage only after the owner authorizes data
transfer; this code-branch publication is not that authorization.

Restore the selected bundle version, validate checksums, stage atomically into a
writable data/runs root and preserve the latest/best policy. Never resume from an
attached input directory in place. Separate execution IDs (attempts) from semantic
run IDs (resumable work). Reject changed inputs/model/precision; provider/path
changes may resume only when semantic and checkpoint compatibility checks pass.
Moving checkpoints between CPU/GPU frameworks may require device remapping.
Repeated exports use versioned manifests and do not overwrite an unrelated run.

### Failure and review contract

Use checked subprocess calls, not IPython `!` lines whose nonzero status can let
Run all continue. A failed readiness check prevents every data/model command.
Finalization reports files and attempts permitted persistence even after a stage
fails, without hiding the original error if finalization also fails.

Current `translate` exit code 1 conflates review findings, failed cases and other
CLI errors. WF-02 must distinguish these through structured status or explicit
output inspection; do not blanket-accept code 1. Report partial failures and
missing expected IDs. Review findings may produce `awaiting-review`; missing or
failed cases produce incomplete/failed status and do not pass a scientific gate.
Statuses should include pending, running, awaiting-review, failed and complete,
with persistence tracked separately. Completion requires validated artifacts and
verified durable storage, not simply reaching the last notebook cell.

## Implementation tasks

| ID | Change | Depends on | Status |
| --- | --- | --- | --- |
| WF-00 | Record pipeline policy, migration plan and documentation links | None | Complete in this branch |
| WF-01 | Inventory cells and specify artifact recipes/active request | WF-00 | Pending |
| WF-02 | Build the validated recipe runner, dry-run and provenance | WF-01 | Pending |
| WF-03 | Add explicit runtime adapters, Kaggle profiles and dependencies | WF-02 | Pending |
| WF-04 | Implement private persistence/export/restore and resume checks | WF-03 | Pending; data transfer needs owner-selected private destination |
| WF-05 | Create reference/execution notebooks and update operator docs | WF-02, WF-03, WF-04 | Pending |
| WF-06 | Prove fresh-session operation and retire the legacy entry point | WF-05 | Pending |

### WF-01: Inventory and freeze the current request

- Map all 55 cells to active, historical, optional or bootstrap recipes. Preserve
  commands and decision history before removing any legacy cell.
- Define input/output contracts and separate preflight, gate and full namespaces.
  Test that a gate cannot satisfy the full-translation artifact contract.
- Initialize the active request to main-plan 1.2 step 1's free-text gate. The pilot
  is complete; the gate remains pending unless new evidence is supplied.
- Declare required English source files and counts; fetch only missing compatible
  sources as explicit prerequisites. Do not regenerate all skills/review packs.

Acceptance: a dry inventory makes the current commands and expected files clear
without searching notebook cells; every historical recipe has a plan/decision link.

### WF-02: Package-owned execution and provenance

- Validate recipe/request schema, typed arguments, dependency order and outputs;
  construct subprocess argument lists without shell interpolation.
- Add inspect/dry-run and stage execution to `cli.py`. Integrate `datadir.py` and
  existing run metadata; keep notebook orchestration minimal.
- Add structured translation outcomes and review stops, finally-path artifact
  reporting and compatible rerun behavior. Include attempts that fail bootstrap
  where a local report is possible.

Acceptance: CPU fixtures run only selected recipes; a failed check starts none;
missing inputs fail clearly; findings and failed translations differ; interrupted
and repeated executions preserve provenance and do not rerun completed pilots.

### WF-03: Explicit environments

- Implement Colab Drive/secrets, Kaggle input/working/secrets and generic
  path/environment-variable adapters. Import provider APIs only in that adapter.
- Add Kaggle preflight/full profiles. Keep the single-GPU implementation initially;
  do not assume multiple GPUs combine memory or require distributed training.
- Introduce a neutral GPU dependency extra with the old `colab` extra retained
  for compatibility. Validate constraints against provider Python/torch versions;
  do not reinstall CUDA blindly or claim current lower bounds are pinned.
- Check actual GPU/dtype support, gated model access, writable storage, effective
  HF cache location and capacity. No provider-specific import on the CPU path.

Acceptance: existing profile tests pass, generic/custom profiles work, and adapter
tests cover missing secrets/storage/GPU. GPU preflight records actual precision,
memory and dependencies; model names do not substitute for capability checks.

### WF-04: Durable artifacts

- Add manifest-based export/restore with explicit bundle version selection,
  checksums, atomic staging and semantic resume validation using `runs.py`.
- Implement the Kaggle private save/restore instructions or approved store;
  exclude checkout, credentials and model caches from output bundles.
- Integrate storage budget checks before writes/exports; retain at most latest
  and best checkpoints per run. Report durable and local steps separately.

Acceptance: corrupted/partial bundles fail validation; path-only migration can
resume a compatible CPU fixture; changed precision/inputs cannot. A saved Kaggle
bundle restores in a new session. Abrupt termination documents the measured loss
window rather than assuming Drive-like persistence.

### WF-05: Split the notebooks

- Build the complete reference from shared recipes, including clearly unavailable
  future training/publication sections; strip execution outputs and secrets.
- Build the short operational notebook and active summaries from the current
  request. Include persistence completion as an explicit final operation.
- Update README, `docs/colab.md`, new operator instructions and `AGENTS.md` with
  direct provider opening/import instructions and stage-update rules.
- Keep `colab.ipynb` available during validation; label it legacy and remove its
  Run all recommendation. Git history preserves earlier execution requests.

Acceptance: notebook JSON/Python are valid, no processing logic is in cells,
profiles are explicit, summaries match the request and no historical experiment
or publication command runs from the active notebook. Imports work without Colab.

### WF-06: Validate and switch entry points

- Run the repository fast checks on Windows and Linux. Add meaningful focused
  workflow/runtime tests and execute both notebook bootstrap paths with mocks.
- Execute the operational notebook on a fresh local CPU fixture, then real Colab
  and Kaggle preflight sessions. Compare artifact schemas, selected IDs, seeds and
  provenance; do not require identical stochastic GPU-generated translations.
- Interrupt translation, restore and rerun; confirm completed cases are skipped.
  Test checkpoint restore with existing fixtures; real training validation waits
  for training code rather than claiming it exists now.
- After provider/storage acceptance, make `execution.ipynb` the recommended entry
  point, archive or replace the legacy notebook with a migration notice, and update
  only the main plan's environment/storage assumptions and version history.

Acceptance: the active stage runs from top to bottom with no manual cell selection
on both providers, survives the documented restore path, and stops for review.
Research gates and publication still need their own recorded decisions.

## First execution after migration

Run only the free-text gate: `gemma3-4b-bf16` on about 40 helmo records balanced
across question types and 20 typed-decisions training cases balanced across
workflows, followed by the two translation reports. Keep this gate in its own
artifact namespace and record the exact selected IDs and source revisions.
The review criteria are the main plan's (at most one meaning change in 40 helmo
records, at most 10-15% requiring edits); automatic checks do not prove them.
End at awaiting-review. A recorded owner/reader decision selects the next stage:
full translation if the gate passes, or shrink/drop helmo if it fails. Do not
retry Gemma 12B or repeat the completed pilot as part of Run all.

## Version history

| Version | Date | Change |
| --- | --- | --- |
| 1 | Oct 5, 2026 | Separate workflow migration, two notebook roles, explicit providers, provenance, durable artifact contract and WF-00 through WF-06 acceptance checks. |
