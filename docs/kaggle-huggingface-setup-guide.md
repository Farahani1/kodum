# Hugging Face and Kaggle setup: resumable translation campaign

Prepared October 8, 2026. This guide uses the current
[execution notebook](../notebooks/execution.ipynb), the frozen v21 draft recipe,
and the research/release boundaries in [project-plan v26](project-plan.md).
It sets up private translation drafts; human review, training and publication
remain separate decisions. Live TPU performance and a physical fresh-session
restore still need to be demonstrated.

Follow steps 1–11 once. Use steps 12–14 after a run or when resuming.
The Kaggle menu names below are the usual labels; placement can vary. Its
documentation pages did not expose readable UI instructions to the web checker,
so these are not a claim that your signed-in screen was inspected.

## 1. Choose the Hugging Face account that will own the checkpoints

Create an account if needed and sign in to [Hugging Face](https://huggingface.co/).
Complete email verification if the account setup requests it.

- If you alone will run the campaign, your personal account is sufficient.
- If another authorized person will resume it, create or use an organization
  that you administer, and put the repository there from the start. Each person
  uses their own account and token. Invite them in the organization's member
  settings and grant the access needed to write the checkpoint repository.
  Standard `write` membership applies across the organization's repositories;
  `contributor` alone does not permit editing a repository you created.

See [Hugging Face organization access control](https://huggingface.co/docs/hub/organizations-security).
Choose the owner now and keep the repository ID stable throughout the campaign.

## 2. Create a private dataset repository

1. Open [New Dataset](https://huggingface.co/new-dataset).
2. Set the owner to your username or the organization chosen above.
3. Use a name such as `kodoom-kaggle-checkpoints`.
4. Select **Private** and create the dataset.
5. Confirm the resulting repository shows as private.

Record its ID in this form:

```text
your-hf-username/kodoom-kaggle-checkpoints
```

For an organization, replace `your-hf-username` with its exact name. The notebook
needs this ID, without `https://huggingface.co/datasets/` or a trailing slash.

Create a **dataset** repository, rather than a model, Space or storage bucket.
No manual upload of source data, model weights or checkpoint ZIPs is needed;
the runner creates its campaign files. A brief private README can say this is
an unreviewed checkpoint store. Leave any blanket license metadata unset for
this mixed checkpoint store; the final resources retain their individual terms.
Official instructions: [create a dataset repository](https://huggingface.co/docs/hub/datasets-adding).

## 3. Create a Hugging Face access token

1. Open [Settings → Access Tokens](https://huggingface.co/settings/tokens).
2. Choose **New token** and name it, for example, `kodoom-kaggle-checkpoints`.
3. Prefer a **fine-grained** token scoped to the dataset from step 2.
4. Grant read access and permission to write its files/contents. A read-only
   token cannot save campaign progress. No inference-provider permission is
   needed for this runner.
5. Create the token and copy it into your password manager temporarily so you
   can enter it into Kaggle Secrets in step 7.

If you choose a general **Write** token instead, it has broader access to
repositories your account can write. Token permission does not replace account
or organization membership. Where an organization requires token approval,
ensure the token is approved before running.
[Official token documentation](https://huggingface.co/docs/hub/security-tokens).

The token belongs in Kaggle Secrets. Do not place it in the notebook settings,
an output cell, a repository README or a chat message.

For this TPU notebook, `HF_TOKEN` only needs private dataset read/write access.
Gemma weights and the tokenizer come from the attached Kaggle Flax model; accept
the model terms on Kaggle as described in step 8. An HF model grant is not used.

For other notebooks that download Gemma from Hugging Face, one token can serve
both purposes if its owner has the model access grant and its permissions cover
both dataset writes and gated-model reads. A fine-grained token scoped only to
the dataset does not automatically cover Gemma. Separate tokens are optional;
acceptance/access is attached to the HF user account, not to an individual token.
[Gated-model access documentation](https://huggingface.co/docs/hub/models-gated).

## 4. Check available Hugging Face storage

Check the owner's storage usage against its current allowance. Hugging Face
currently lists 100 GB of private storage for free users and organizations;
confirm the applicable allowance in your account. The campaign saves repeated
compressed text snapshots, whose history consumes storage, and excludes model
weights. [Current storage documentation](https://huggingface.co/docs/hub/storage-limits).

Keep the repository private and retain its checkpoint files/history during the
campaign. A dataset preview may be absent because this store contains ZIPs and
manifests; use **Files and versions** to inspect it.

## 5. Import the correct notebook into Kaggle

1. Create an account if needed and sign in to [Kaggle Code](https://www.kaggle.com/code).
2. Create a new Python notebook and give it a recognizable title, such as
   `kodoom Gemma 27B bulk translation`.
3. Use **File → Import Notebook → Upload** and choose the project's
   `notebooks/execution.ipynb` from your local checkout.
4. Confirm the notebook is **Private** in its sharing/visibility controls.
5. Confirm the title inside it says **resumable Gemma 3 27B translation** and
   the active stage is **bulk-translation**.

If downloading the notebook from GitHub, use the
[published CPU-first bulk handoff notebook](https://github.com/Farahani1/kodum/blob/codex/tpu-bulk-resume-batching/notebooks/execution.ipynb).
Download its `.ipynb` file using the raw/download control, then import that file.
Do not import `reference.ipynb` or the legacy `colab.ipynb` for this run.

The handoff notebook and its runtime have different revisions. The notebook's
`CODE_REVISION` below deliberately points to the tested runtime. Downloading
the notebook itself from that runtime revision would retrieve an older gate
notebook. Preserve the settings shown in step 9.

The code pin must actually exist on GitHub before running. The Oct 9 missing-pin
HTTP 404 was resolved by publishing the CPU runtime. The current pin below also
repairs isolated environment setup and narrows default preparation to access/setup
after observed Kaggle Python 3.13 failures. Deep audits are optional.
Do not substitute an older revision without `cpu_preflight`. An HTTP 404 for the
pin is separate from invalid-token HTTP 401.

## 6. Configure Kaggle Internet and accelerator

In the notebook's session/settings controls:

1. Enable **Internet**.
2. Select **TPU v5e-8** for the configured translation notebook.
3. Keep Python as the notebook language and start from a fresh session/kernel.
4. Check that your account has TPU time available. Complete any account
   verification Kaggle requests to enable these features.

For optional CPU-only diagnosis, select **None** and set RUN_TPU=False.
Normal imports already have RUN_TPU=True and use **TPU v5e-8** for translation. The runner requires
eight v5e devices on one host. If that accelerator is
unavailable, wait for an allocation or resolve account access; CPU, GPU and
another TPU type will not satisfy this campaign's readiness check.

The pinned bootstrap supports Linux Python 3.11–3.13 and chooses its dependency
lock automatically. Let it install the dependencies. Keep extra JAX/device
experiments out of the notebook kernel, because TPU probes and generation use
child processes. Use the account limits displayed by Kaggle rather than
assuming the whole campaign fits in one allocation.

## 7. Add the Hugging Face token to Kaggle Secrets

1. Open **Add-ons → Secrets** in the notebook.
2. Choose **Add a new secret**.
3. Set the label to exactly `HF_TOKEN`.
4. Paste the token from step 3 into the secret's value field and save it.
5. Enable/attach that secret for **this notebook**. Creating it without
   enabling notebook access is insufficient.

The bootstrap reads this secret and passes it to the checkpoint worker.
Kaggle's [official secret-client implementation](https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_secrets.py)
requires a secret to be attached to the kernel before it can be retrieved.
You do not need a separate Kaggle API token for the notebook's attached model.

For the currently public project code, a `GITHUB_TOKEN` is unnecessary. If an
old notebook reports HTTP 401, disable its `GITHUB_TOKEN` secret and restart the
session to clear any token already copied into the environment. GitHub rejects
invalid credentials even for public files. The updated bulk notebook fetches
the pinned public source without authentication and ignores `GITHUB_TOKEN` for
both CPU and TPU Git checkouts, including an attached or cached stale secret.

If the GitHub repository becomes private, create a fine-grained GitHub token
for `Farahani1/kodum` with repository **Contents: Read-only**, and add/enable it
under the exact Kaggle secret label `GITHUB_TOKEN`. Confirm any organization
approval requirements and use a bootstrap configured for private code; the bulk
notebook described here explicitly uses public code access.
[GitHub token instructions](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

## 8. Accept Gemma access and attach the exact Flax model

1. While signed in to the same Kaggle account, open
   [Google Gemma 3 27B IT, Flax, version 1](https://www.kaggle.com/models/google/gemma-3/flax/gemma3-27b-it/1).
2. Complete the access/consent form or accept Gemma's terms if prompted.
3. Return to your notebook and use **Add Input / Add Models** (the label may
   appear under the input or data panel).
4. Select the model with this exact handle:

   ```text
   google/gemma-3/flax/gemma3-27b-it/1
   ```

5. Confirm it appears among the notebook's attached inputs before starting.

Match **Gemma 3**, **27B**, **instruction-tuned**, **Flax**, and **version 1**.
Keep the checkpoint and bundled tokenizer mounted under `/kaggle/input`.
An HF token does not grant Kaggle model access, and the runner does not use an
HF Transformers checkpoint in place of this Flax asset. No manual model download
to your laptop, Google Drive or the checkpoint repository is required.

## 9. Use the configured notebook settings

The stable [configured notebook download](https://raw.githubusercontent.com/Farahani1/kodum/codex/tpu-bulk-resume-batching/notebooks/execution.ipynb)
already contains the following settings for the owner's confirmed dataset.
Import that file each time instead of reusing an old download or editing the pin.
No settings edits are required for the configured TPU translation run. Other operators using a
different checkpoint repository must deliberately update that destination:

```python
PROVIDER = "kaggle"
PROFILE = "kaggle-tpu"
BACKEND = "jax"
CODE_REVISION = "74e992e439d7bfb5f26e5d19ea8eea704769df6e"
CHECKOUT = "/tmp/kodoom-tpu-code"
CAMPAIGN_ID = "gemma27b-bulk-v1"
HF_DATASET_REPO = "Farahani/kodoom-kaggle-checkpoints"
OPERATOR = "owner"
RUN_TPU = True
TAKEOVER = False
```

Keep `CODE_REVISION` unchanged. Choose `CAMPAIGN_ID` once; ordinary resumes use
the same ID. `OPERATOR` is a short non-secret name, not a token. Leave
`TAKEOVER=False` for the first run and for resumes after a clean stop.

This frozen recipe generates 1,600 typed cases (1,200 train / 400 test; 8,000
decisions) plus 2,000 helmo records. Helmo remains in this run even though it is
optional for the eventual release. The proposed test-first order and typed-only
campaign are not implemented in this notebook. Changing code, model, prompts,
precision or scope requires deliberate campaign handling, not an ordinary resume.

## 10. Submit one server-side run

Import the [configured notebook](https://raw.githubusercontent.com/Farahani1/kodum/codex/tpu-bulk-resume-batching/notebooks/execution.ipynb).
It defaults to RUN_TPU=True for the owner-requested translation run, with the
confirmed dataset and tested runtime pin already filled. Keep the notebook
private, enable Internet, choose TPU v5e-8, enable HF_TOKEN, and accept/attach the
exact version-1 Flax model. Then submit Save Version > Save & Run All.

The early cell skips the isolated CPU environment and dependency dry-run. After
production installation, short access/storage/model/resume checks must pass
before translation. The worker retains its real source/token/library/TPU checks.
This runner cannot enable a Kaggle secret or accept model terms on your behalf.

Optional CPU-only diagnosis uses Accelerator=None and RUN_TPU=False. It checks
GitHub, HF read/write/readback, writable paths, attached model and resume state.
The private HF probe adds and removes a tiny file, leaving two history commits.
Deep source/prompt/dependency audits require deep=True in runtime.cpu_preflight;
they are not part of ordinary imports or TPU runs.

1. Recheck private visibility, Internet, TPU v5e-8, attached Flax model,
   enabled `HF_TOKEN`, repository ID and campaign ID.
2. Confirm `RUN_TPU=True` and another worker is not already running this campaign. Do not run
   an interactive bulk worker and a saved background version simultaneously.
3. Click **Save Version → Save & Run All** and submit the version.
4. Open that version's execution logs/status to follow the run.

Use the server-side run rather than only saving notebook source. Once submitted,
the laptop can sleep; closing its browser does not make local interactive progress
durable. The runner's eight-hour budget starts at bootstrap and includes setup,
loading, compilation and generation, with 30 minutes reserved for finalization.

## 11. Confirm setup, generation and saving in the logs

Expect these stages, allowing time for installation, TPU allocation, model
loading and compilation:

1. CPU checks print PASS/FAIL/SKIP items and a resolved code revision. After PASS,
   the opted-in TPU setup prints an eight-device BF16 probe.
2. The worker checks the private HF repository, acquires a writer lease and
   prints **Restored progress**. A new campaign normally starts at zero.
3. It saves and reads back a snapshot and restores it into a fresh directory
   before loading model weights. This validates storage within this allocation.
4. It measures prompt lengths and benchmarks BF16 batch sizes 1, 2, 4 and 8,
   then freezes the fastest passing batch. A compatible resume reuses it.
5. It generates complete units, exports review CSVs and periodically saves
   verified snapshots. All translations remain unreviewed drafts.

The first 40 helmo records and 20 typed training cases are diagnostic review
material. Generation continues without waiting for review. Inputs are never
silently truncated; oversized prompts and incomplete replies need inspection
and do not count as completed translations. Automatic check findings remain
review flags on saved drafts; completion does not certify their meaning.

The live fit/speed checks may fail even though local software tests pass.
Capture the failure and relevant logs; do not change the code pin or batch
configuration simply to bypass an incompatible resume check.

## 12. Check Hugging Face checkpoints and the final result

Open your private dataset's **Files and versions** tab. Under
`campaigns/gemma27b-bulk-v1/`, expect:

```text
writer.json
checkpoint.json
snapshots/<hash>.zip
```

`control.json` appears only if a pause/resume control has been written. A tiny
initial checkpoint can exist before any translations finish. Later snapshots
contain English inputs, translated shards, progress, settings, metrics and review
CSVs. The ZIP path in `checkpoint.json` identifies the current checkpoint archive.
Download that ZIP if you want to inspect `progress.json` and the review CSVs;
keep human edits in separate copies.

Snapshots are normally saved about every ten minutes and at finalization.
The worker stops new inference after 30 minutes of unsaved completed work and
attempts final saving. An abrupt provider kill can lose work since the last
verified remote save.

At a normal end, the notebook prints:

- **Status**: inspect whether drafts are complete, incomplete or paused.
- **Completed / failed / total cases**: these count typed cases and helmo records,
  rather than individual decisions; full frozen scope is 3,600 work units.
- **Last verified remote revision**: a concrete revision for saved progress.
- **Final remote save verified: True**: persistence succeeded for that attempt.

`True` does not imply the entire campaign is complete. Check status and counts
too. A failure may stop before the reporting cell; inspect the worker logs and
local `result.json`, if present, under:

```text
/kaggle/working/kodoom/data/bulk/gemma27b-bulk-v1/
```

Keep the saved Kaggle version and outputs private as an additional copy. HF
snapshots are the normal cross-session restore path. Review completion, source
correctness, translator adoption and release approval do not follow automatically
from a successful save.

## 13. Resume after a clean stop or session timeout

1. Confirm the previous Kaggle execution has ended.
2. Start a fresh session with the same bulk notebook and exact Flax model.
3. Keep the same `CODE_REVISION`, `CAMPAIGN_ID` and `HF_DATASET_REPO` as the saved
   campaign. This newly pinned runner cannot silently resume an older code identity;
   preflight reports that conflict. The failed missing-token run stopped before
   starting its HF worker, so that attempt did not create a campaign checkpoint.
4. Recheck Internet, TPU availability and that the token is enabled.
5. For a cleanly finalized run, leave `TAKEOVER=False`.
6. If the previous session was abruptly killed and a writer-conflict error
   reports an unreleased lease, confirm no prior worker is alive, then use
   `TAKEOVER=True` for the replacement attempt. Return it to `False` afterwards.
7. If you explicitly paused the campaign, clear the pause control before
   resuming; see [pause/resume instructions](execution-workflow.md#review-during-generation-and-pause).
8. Run the CPU checks again first if setup changed, then select the TPU, set
   `RUN_TPU=True` and submit **Save Version → Save & Run All** again.
9. Check that **Restored progress** reflects the previously verified completed
   units. Only pending/failed work should be processed. This second physical
   session is the live restoration evidence the plan still needs.

A different Python/library version may fail the production compatibility check
even with the same pin. Preserve the evidence and resolve the environment change;
do not disable validation, delete checkpoints or silently start the same identity
with different settings.

## 14. Let an authorized collaborator resume

Use the organization repository chosen in step 1. The collaborator needs their
own organization write access, HF token, Kaggle account, TPU allowance and Gemma
terms acceptance. They import the same handoff notebook, attach the same model,
enable their token and use the same repository/campaign/code pin. They change
`OPERATOR` to a short name identifying themselves.

Only one person runs the campaign at a time. Apply the clean-stop/takeover rules
from step 13. Share repository IDs and notebook settings; each person keeps their
own credentials. A change of operator alone is compatible with resume.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `Set HF_DATASET_REPO...` | Replace the empty value with `owner/dataset-name`, not a URL. |
| `Enable the HF_TOKEN Kaggle secret...` | Exact label, saved value and notebook access checkbox/toggle. |
| `Cannot access HF dataset...` / HF 401 or 403 | Dataset exists; ID is correct; token is valid; account membership and read/write permissions are sufficient; organization token approval is complete. |
| `Bulk checkpoints require a private HF dataset repository` | Check the dataset's visibility. A public repository fails this runner's storage check. |
| GitHub bootstrap HTTP 401/403 | Disable an unnecessary expired/invalid `GITHUB_TOKEN` for public code, or enable a valid read token for private code. |
| GitHub bootstrap HTTP 404 | Check the exact repository and runtime pin; private repositories also require authorized access. |
| `Could not create the isolated CPU preflight environment` | Use the current runtime pin above and restart the session. It creates venv without `ensurepip`, then uses notebook pip to install pip into the isolated interpreter; it also repairs a partial environment. Check writable `/tmp`, Internet/PyPI and notebook pip 22.3+ if setup still fails. |
| `Error in sitecustomize ... wrapt` | This provider startup warning did not stop the reported checks. The current pin runs isolated checks without provider sitecustomize; it does not install wrapt into the notebook. |
| TPU probe/device error | Select v5e-8, use a fresh session, and let bootstrap install its pinned dependencies. |
| Missing Flax model/tokenizer or access denied | Accept terms using this Kaggle account and attach the exact version-1 model from step 8. |
| Writer conflict | Confirm the previous run has ended. Use takeover only for an abandoned lease; never start a second live writer. |
| Changed identity/environment or corrupted checkpoint | Keep the checkpoint and logs. Resolve compatibility/corruption rather than editing the manifest or bypassing the checks. |
| Final save unverified or worker exits early | Inspect the last verified HF checkpoint and `result.json`/logs. Retain available local outputs; resume from verified progress after resolving the failure. |

For the underlying execution/storage contract, see
[execution-workflow.md](execution-workflow.md). No Google Drive mount,
Hugging Face inference endpoint, or manual training setup is needed for this
Kaggle draft stage.
