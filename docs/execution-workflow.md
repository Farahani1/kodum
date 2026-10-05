# Running kodoom on Kaggle

This prepares the next main-plan step, the free-text gate in 1.2 step 1. The
notebook runs a small GPU preflight, then translates 40 balanced helmo records
and 20 balanced typed-decisions training cases with the selected Gemma 3 4B
translator. It ends for review. It does not run the full translation, training
or publication. Kaggle GPU execution and cross-session restore still need a live
provider run.

## One-time Kaggle setup

1. Open [execution.ipynb](../notebooks/execution.ipynb) from the
   `codex/portable-notebook-workflow` branch in Kaggle with **Import Notebook**.
   Create a new private notebook.
2. Enable notebook **Internet** and select a GPU accelerator. The runner exposes
   one GPU, so two Kaggle T4 cards are not treated as one 32 GB card.
3. Accept the Gemma terms using the Hugging Face account you will use for Kaggle.
   In Kaggle notebook Settings, add a secret named `HF_TOKEN` with a read token
   and attach it to this notebook. A public GitHub repo needs no secret. For a
   private repo, also add `GITHUB_TOKEN` with read access to `Farahani1/kodum`.
4. Leave the notebook provider/profile set to `kaggle` / `kaggle`. The runner
   saves data, reports and ZIP bundles under `/kaggle/working/kodoom`; model
   cache and temporary files go under `/tmp` and are excluded from saved output.

## Run the gate

Run all cells in the interactive session. The bootstrap checks out the committed revision, installs
the GPU extra, checks storage and the visible GPU, verifies bf16 arithmetic and
Gemma access, then runs a 3-helmo / 4-typed-case preflight followed by the
40-helmo / 20-typed-case gate. After it finishes, use **Quick Save** with output
saving enabled to save the current results without running the GPU work again.
Alternatively, start with **Save & Run All** and let that fresh session finish
once. Kaggle can save outputs from `/kaggle/working` for later use as notebook
inputs. See the
[Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks) for
Quick Save, Save & Run All and output limits.

At the end, note the printed ZIP path. It contains only that stage's data,
manifests, reports and review sheets. It contains no GitHub or Hugging Face
token, checkout, base model or model cache. The ZIP is ready to save; this alone
does not prove Kaggle retained it. Enable **Save output files on Quick Save**
before saving interactive results. After the output is available, attach that
notebook output under Kaggle's Input pane.

For a new Kaggle session, copy the attached ZIP path into `RESTORE_PREFLIGHT`
for a preflight bundle or `RESTORE_GATE` for a gate bundle in the settings cell.
Use an empty `/kaggle/working` directory so restore cannot overwrite live data.
Run the notebook again, save another private version with outputs, and check
that the restored ZIP, translation IDs and checksum match. The current gate
artifacts remain on Kaggle; no Drive-to-Kaggle data transfer is required.

## Review the result

Open `review-helmo.csv` and `review-typed-decisions.csv` from the output ZIP.
Review each sample against its English text. Mark `meaning_changed` and
`needs_edit` as `yes` or `no`, and add notes where useful. The automatic checks
help find formatting and data-integrity issues; they cannot decide whether a
translation changes meaning. Preserve the reviewed CSVs and save a new private
bundle/version.

The acceptance bar is the one in the main plan: at most 1 meaning change in the
40 helmo records and at most 10-15% of sampled records needing edits. Record the
owner's decision before setting the next active request. A passing gate permits
planning full translation; a failed gate means shrink or drop helmo. The next
request is not dispatched automatically. Kaggle output is private working data,
not a dataset or model publication.

## Where to continue

- Research stage and decisions: [project plan](project-plan.md), section 1.2.
- Active recipe and accepted limits: `workflows/current.toml` and
  `workflows/recipes.toml`.
- Code and agent handoff rules: [`agent.md`](../agent.md).
- Independent tooling migration and acceptance checks:
  [workflow change plan](workflow-change-plan.md).
