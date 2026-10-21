# Kaggle T4 x2 model training

The model path is `notebooks/training.ipynb` and
`workflows/model-training.toml`. The older `execution.ipynb` / `current.toml`
pair remains a historical TPU translation recipe; do not use it for training.
The new path implements project-plan KD-03/KD-05/KD-06 without waiting for the
retired research gate or republishing the dataset.

## First operator action

1. Upload/open `notebooks/training.ipynb` as a **private** Kaggle notebook.
2. Enable Internet, choose **GPU T4 x2**, and add a Hugging Face access token
   as a Kaggle Secret named exactly `HF_TOKEN`. Give it read access to
   `Farahani/kodoom-en-fa` and write access to the private model repository
   `Farahani/kodoom`. Never paste the token into a notebook cell or commit it.
   The code branch must be pushed before its pinned revision can be fetched
   from GitHub.
3. The notebook is configured with `BACKUP_REPO = "Farahani/kodoom"`. This
   existing private model repository stores checkpoints and logs; it is
   separate from the dataset repository.
4. Leave `ACTION = "preflight"`, then choose **Save Version -> Save & Run All
   (Commit)**. The preflight uses real Laya on both GPUs for two updates,
   validates losses/gradients and checkpoint writes, and must successfully save
   a complete private Hub snapshot. Success means the final summary contains
   `status: preflight_complete`, `step: 2`, and a nonempty `durable_revision`.
   Reaching a final cell that prints "Done" does not establish success if an
   earlier cell contains a Python error.

The first Kaggle attempt with code revision `e728b4e` failed during dataset
inventory checking because Hub cache symlinks resolve outside their snapshot
root. No GPU training or resumable checkpoint was produced. The fixed downloader
materializes regular local files and retains the same pinned hashes and path
containment checks. Use the updated notebook code pin and rerun `preflight`;
keep GPU T4 x2 selected.

The second attempt with code `4ebfac9` passed dataset loading and reached both
GPU workers, then stopped on nonfinite gradients before the first optimizer
update. The updated trainer starts with AMP scale 1024 and retries the same
batch at lower scales (up to 12 times), advancing weights/scheduler/data only
when gradients are finite. Persistent failures remain fatal. Use the latest
notebook code pin in a fresh **Save & Run All** session; do not restore or start
main training from either failed attempt.

The local dataset revision is pinned at
`1b98bbba48043c62c6ae2c821a55b6f8a5e7a927`, release 1.0.1. The handoff verifies
its accepted manifest hash before consumption. Remote availability must pass
the Kaggle preflight; a local Git revision alone does not prove publication.
If access/revision fails, report the missing exact revision; do not substitute
the floating dataset main branch or weaken the release pin.

## Restore, then train

**Current result (Oct 20):** the third saved run passed preflight at step 2
using code `0fa1a580758da9e020075b0537ace4166a03b328`. Its complete private
checkpoint commit is `de6c6cc0fd75f51b404c93aebfddad632e74a177` in
`Farahani/kodoom`. Keep that code pin unchanged and run a new saved version with:

```python
ACTION = "restore-preflight"
RESTORE_REVISION = "de6c6cc0fd75f51b404c93aebfddad632e74a177"
RESTORE_RUN_ID = "laya-full-v1-preflight"
```

The expected successful restore summary is `preflight_complete`, step 4 and a
new nonempty `durable_revision`. Use the final notebook result dictionary;
the other GPU worker may print a summary with a null durable revision.


Read `durable_revision` from the preflight summary or the
`durable-checkpoint` log event. It is the commit for a **complete snapshot**;
a later logs-only commit is not interchangeable with it.

In a fresh Kaggle session, keep the same pinned code/settings, set
`ACTION = "restore-preflight"`, `RESTORE_RUN_ID` to the preflight run ID, and
`RESTORE_REVISION` to that complete snapshot commit. Use **Save & Run All**
again to verify hashes,
restore optimizer/scheduler/scaler/RNG/sampler progress and continue the short
preflight. It restores into a separate run root so no existing run is overwritten.

After this succeeds, set `ACTION = "train"`. This starts the full-data run
independently from original Laya, not from preflight weights. The run is two
epochs, global batch 32, 50/50 English/Persian replay, full encoder/head tuning
with lower encoder learning rate, and development-only English retention checks.
It does not evaluate the final test set automatically. The partial-context
model is the next stage and starts independently from original Laya too.

## Full-data training result

**Training completed (Oct 21, 2026):** the owner's attached Kaggle notebook
`laya-training.ipynb` ran `ACTION = "train"` on T4 x2 with code
`0fa1a580758da9e020075b0537ace4166a03b328`. Run `laya-full-v1` finished at
step 600 with `status: training_complete` and best Persian development loss
`0.9534161526337266`. The final result reports `test_evaluated: false`; this is
training completion, not completion of KD-06 evaluation or portfolio release.

The notebook records complete private checkpoint revision
`f315b5da129e2b1df55d971ddf2053c3cbfe5282` in `Farahani/kodoom`, under
`laya-full-v1/`. Use `best/` for the selected inference weights and `latest/`
for resumable optimizer/training state. Pin this full commit hash when fetching
the snapshot; later logs-only commits are not model checkpoint revisions.
The notebook does not record a fresh-session restore of this full-run checkpoint
or the final held-out evaluation. Those remain follow-up checks.

Intended Hub tag for the complete training stage (not yet applied):
`laya-full-v1-training-complete-awaiting-evaluation`. The tag must point to the
complete snapshot revision above, not to a later logs-only commit.

To resume an interrupted main run, use the restore CLI with the main run ID,
exact complete snapshot commit, and an empty output directory, then invoke the
same `train` command with `--runs-dir` pointing to its parent. Keep identical
code, data and settings; changed recipes require a new run ID. Example:

```bash
python -m kodoom.training restore --hub-repo Farahani/kodoom \
  --revision COMPLETE_SNAPSHOT_COMMIT --run-id laya-full-v1 \
  --out /kaggle/working/kodoom/runs/laya-full-v1
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m kodoom.training train --profile kaggle \
  --prepared /kaggle/working/kodoom/data/training/full-v1 \
  --settings configs/laya-full.toml --run-id laya-full-v1 \
  --hub-repo Farahani/kodoom
```

## Persistence and evidence

- Rank 0 writes per-update language losses/counts, learning rates, gradient norm,
  scaler skips, throughput and GPU memory locally under `/kaggle/working/kodoom`.
  Compact logs sync about every five minutes.
- Latest full resume state is saved every 50 updates or 15 minutes, plus epoch
  end and orderly completion; best inference weights follow the retention-aware
  validation rule. Complete snapshots sync after local atomic writes.
- Preserve latest and best. A local Kaggle file is not durable until the private
  upload succeeds. Sync failures are logged; preflight fails rather than
  claiming readiness when its private checkpoint upload fails.
- Hub history can retain older weight blobs. Monitor repository storage rather
  than assuming overwriting latest automatically frees remote space.
- Pinned upstream Laya code is checksum verified before import. The upstream
  implementation and checkpoint are Apache-2.0; preserve their attribution in
  the model card and follow the source notice for any redistribution.

Local tests do not establish GPU memory, mixed-precision numerics, two-T4
throughput or fresh-session persistence. Those require the real Kaggle preflight.
