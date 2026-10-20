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
4. Leave `ACTION = "preflight"`, then Run all. The preflight uses real Laya on
   both GPUs for two updates, validates losses/gradients and checkpoint writes,
   and must successfully save a complete private Hub snapshot.

The local dataset revision is pinned at
`1b98bbba48043c62c6ae2c821a55b6f8a5e7a927`, release 1.0.1. The handoff verifies
its accepted manifest hash before consumption. Remote availability must pass
the Kaggle preflight; a local Git revision alone does not prove publication.
If access/revision fails, report the missing exact revision; do not substitute
the floating dataset main branch or weaken the release pin.

## Restore, then train

Read `durable_revision` from the preflight summary or the
`durable-checkpoint` log event. It is the commit for a **complete snapshot**;
a later logs-only commit is not interchangeable with it.

In a fresh Kaggle session, keep the same pinned code/settings, set
`ACTION = "restore-preflight"`, `RESTORE_RUN_ID` to the preflight run ID, and
`RESTORE_REVISION` to that complete snapshot commit. Run all to verify hashes,
restore optimizer/scheduler/scaler/RNG/sampler progress and continue the short
preflight. It restores into a separate run root so no existing run is overwritten.

After this succeeds, set `ACTION = "train"`. This starts the full-data run
independently from original Laya, not from preflight weights. The run is two
epochs, global batch 32, 50/50 English/Persian replay, full encoder/head tuning
with lower encoder learning rate, and development-only English retention checks.
It does not evaluate the final test set automatically. The partial-context
model is the next stage and starts independently from original Laya too.

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
