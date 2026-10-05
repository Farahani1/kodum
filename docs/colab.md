# Running kodoom on Colab

The laptop is where code is written and proven (`dev` profile). Colab only runs it. This page covers moving to Colab, updating the code there, and what happens when a session dies.

The existing notebook contains historical trials and optional manual-input steps.
Select setup and the active stage; do not Run all. The current documented next
stage is the free-text gate (main plan 1.2 step 1), not the completed pilot.
A separate [workflow change plan](workflow-change-plan.md) tracks the planned
reference/execution notebooks and Kaggle support; they are not yet implemented.

## One-time setup

1. **Open the notebook from GitHub.** In Colab: File → Open notebook → GitHub, then pick `notebooks/colab.ipynb` on your branch. Or open this link directly:
   `https://colab.research.google.com/github/Farahani1/kodum/blob/claude/sweet-franklin-pm0tdm/notebooks/colab.ipynb`
   (replace the branch name if you work on another one). Keep it in Drive with File → Save a copy in Drive if you want it in your Colab list.
2. **Private repository only:** create a GitHub fine-grained token with read access to this repository, then in Colab open 🔑 Secrets (left sidebar), add a secret named `github-kodoom` (or whatever `TOKEN_SECRET` says in the notebook's Settings cell) with the token as its value, and turn on notebook access.
3. **Check free space on Drive:** the free plan has 15 GB shared with Gmail and Photos. `kodoom check` warns under 8 GB free; clear space before the main training runs (plan: Storage budget).

## Every session

1. Runtime → Change runtime type → **T4 GPU**.
2. In the *Settings* cell, set `REF` (the branch, tag or commit) and `PROFILE` (`colab-preflight` before any long run, then `colab`).
3. Run Settings, Mount Google Drive, Get the code, install, Check the environment,
   runs, Record the data, and Hugging Face access if needed. A failed check must
   stop you here; the existing `!` commands do not enforce that for Run all.
4. Run only the cells for your current stage. For the free-text gate, ensure the
   English sources exist (fetch missing sources), then run its translation/report
   cells. Do not rerun old translator trials, the pilot, cache deletion or optional
   import cells. Finish with Files generated or updated, then optional Drive sync.

`kodoom check` reports failure if Drive is not mounted, no GPU is attached, or
Drive is too full, and lists runs that can be resumed. With the existing shell
cell, stop manually on failure; automatic stopping is part of the migration.

## Updating the code

1. Change the code on the laptop, run the fast checks and the `dev` smoke run, commit, push.
2. On Colab, re-run the **Get the code** cell. It moves to the newest commit of `REF`. The install is editable, and commands run as fresh `kodoom` processes. Then run setup checks and only the active stage.
3. Re-run the install cell only if `pyproject.toml` dependencies changed.

Edits made directly on Colab are discarded by the next update, on purpose: the code on Colab must always be a commit that already ran on the laptop. The commit is recorded in every run's `run.json` (with `-dirty` if it was not clean).

## Native review of the glossary and templates

The *Review pack* cell (`kodoom review-pack`) writes `MyDrive/kodoom/data/.../review/`: `glossary.html` and `templates.html` show the glossary and every Persian template next to an item generated from it, right to left, ready to read in a browser (download them from Drive and open them). `glossary.csv` and `templates.csv` hold the same rows with empty `ok`, `suggestion` and `note` columns; they open in Excel or Google Sheets with the Persian intact. Save the filled sheets under a new name, because the cell rewrites the originals, and bring the suggestions back to the assistant; the TOML files in the repository stay the source of truth.

## Translation trial

The *Translation trial* cells run `kodoom translate typed-decisions --translator NAME --split test --limit 10` for TranslateGemma 4B (bf16, then 4-bit weights with 32-bit activations), then `kodoom translations`, which prints how many cases the automatic checks flagged and shows English next to Persian for the first cases. Models download into `/content/hf-cache` (never Drive, see the storage budget); results go to `data_dir/typed-decisions/fa/NAME` on Drive. The command resumes: run it again after a dropped session and finished cases are skipped. It exits with status 1 when some case has check findings; that is information, not a crash. Qwen3-8B (4-bit) ran on the T4 in an earlier trial (a 16 GB download, about three minutes). TranslateGemma 4B loads on the T4 (8.6 GB), but in plain fp16 Gemma overflows and answers with nothing, which `kodoom translate` now reports as a NaN/inf error; use the `-bf16` or `-4bit-fp32` translators. A case the model cannot translate is skipped and its reason goes to `<split>.failures.jsonl` next to the output. TranslateGemma is **gated**: open its Hugging Face page, accept the license, create a read token and store it as the Colab secret `HF_TOKEN` (the *Hugging Face access* cell passes it on); without it the trial stops with a 401 `GatedRepoError`. Whether the TranslateGemma message format is right is still untested: copy any error back.

## Translating outside kodoom (Claude Cowork, or a person)

A translator that cannot run inside `kodoom translate` still goes through the same checks:

1. Run the *Export units* cell. It writes `MyDrive/kodoom/data/.../typed-decisions/exchange/units.jsonl` (each distinct English text once, with its register, context and glossary terms, and an empty `fa`) and `INSTRUCTIONS.md` (the brief).
2. Give both files to the translator. Claude Cowork works on local files: download them from Drive, ask it to follow `INSTRUCTIONS.md` and fill `fa` in every line, in chunks of about 50 lines. Keep all other fields unchanged.
3. Save the filled file as `units-filled.jsonl` in the same Drive folder.
4. Run the *Import units* cell (`kodoom import-units --name cowork`). It reports units that are missing, empty or changed, builds the Persian records in `fa/cowork/`, and runs the usual automatic checks. The result can go into the blind pilot sheet like any model.

Because the question and option texts repeat in every case of a workflow, the units file holds each once, so a sentence gets one Persian rendering everywhere. Check the terms of the tool you use before publishing data it produced, and say on the dataset card which model translated.

## Disk space during trials

The Colab local disk (about 112 GB) holds the model cache at `/content/hf-cache`, never Drive. Each model stays there until the session ends: Qwen3-8B 16 GB, TranslateGemma 4B 8.6 GB, Gemma 3 4B 7.75 GB, Gemma 3 12B about 24 GB. The *Free disk space* cell removes the Qwen download; run `!du -sh /content/hf-cache/hub/models--*` to see what is left and delete folders you no longer need.

## Fetching typed-decisions and reading its fields

The *Fetch typed-decisions* cells run `kodoom fetch typed-decisions`, which downloads the English cases at the pinned commit and writes one record per question to `data_dir/typed-decisions/en` on your Drive (with a manifest and checksums), and `kodoom fields`, which reports every text field of the case state (coverage, distinct values, length, samples). The translate/keep rules are decided from that output, so copy it back to the assistant. Nothing is pushed anywhere.

The *Fetch helmo* cell (`kodoom fetch helmo`) does the same for `helmo/synthetic-typed-decisions`: 9,879 single-question records, converted to `data_dir/helmo/en`. If a dataset is not stored as parquet or JSON lines, `kodoom inspect` lists its files instead; use it for any new source.

`kodoom translate helmo --translator NAME` translates its records the same way
(`data_dir/helmo/fa/NAME/train.jsonl`, resumable, same exit-1-on-findings behaviour). helmo
has no splits or workflows: `--split` does not apply, and `--balanced` with `--limit` takes
records evenly from each question type (choice, score, noul) instead of from a workflow,
since helmo has no workflow field. helmo's `state` is one free-text paragraph rather than a
JSON tree, so the whole record (state, question, options) is translated in a single call and
checked with the same text-level checks as a typed-decisions question or option.
`kodoom translations helmo --translator NAME --show N` prints its counts and English next to
Persian the same way `kodoom translations typed-decisions` does.

## When a session dies

Nothing to rescue: logs and checkpoints are written to Drive while the run goes.

1. Reconnect (or open a new session), T4 GPU, then rerun setup and only the active
   stage as described above. Reuse the same data paths and translation settings.
2. `kodoom runs` shows each unfinished run and its latest checkpoint step.
3. Start the same step again with the **same run id and the same settings**. It resumes from the latest checkpoint. With different settings it refuses and names what changed; use a new run id instead.

At most one training step since the last checkpoint is lost, plus one log line.

## What is on Drive

```
MyDrive/kodoom/runs/
├── registry.csv          one row per finished run (plan 2.3)
└── <run id>/
    ├── run.json          config, git commit, profile, status
    ├── log.jsonl         one line per event, written as it happens
    ├── latest/           resumable checkpoint (weights + optimizer); deleted when the run finishes
    └── best/             best weights so far (fp16, weights only)
```

```
MyDrive/kodoom/data/      (data/preflight/ for the colab-preflight profile)
├── README.md             what each file is, sizes, record counts, history of updates
├── .kodoom/              bookkeeping for README.md and `kodoom tree` (hidden)
├── skills/               kodoom generate
├── review/               kodoom review-pack: glossary and templates for a native reader
├── typed-decisions/      en/ (fetch), fa/<translator>/ (translate, import-units), pilot/, exchange/
└── helmo/en/             kodoom fetch helmo
```

`README.md` is rewritten by every command that writes data (`generate`, `fetch`, `translate`, `pilot-sheet`, `export-units`, `import-units`), and only when a file changed. It lists the files the last update made or changed and the last 20 updates with their command lines; files you add by hand (a filled units file) show up at the next update. The *Record the data* cell near the top runs `kodoom tree --start`, and the *Files generated or updated in this run* cell at the end runs `kodoom tree`, which prints the tree and marks what this run made (`[new]`) or changed (`[updated]`).

Datasets (`data/`, generated by `kodoom generate` and later the converters and translations) are written to `MyDrive/kodoom/data` and nowhere else: not into the repository checkout on Colab, and never into git (`data/` is git-ignored and a test fails if a data file is ever tracked). Nothing is published to Hugging Face or anywhere else until the owner decides to.

Never on Drive: base models, translators and the checker (downloaded to `/content/hf-cache` each session) and checkpoint staging (`/content/kodoom-scratch`). Finished models and datasets go to Hugging Face, then leave Drive.

How a checkpoint is saved, so a dead session never leaves a broken one: it is written to local disk, free space on Drive is checked, it is copied to Drive under a temporary name with a manifest written last, and then swapped in. On the next start, anything half-done is repaired or set aside (`.latest.broken-…`), never deleted.

## Troubleshooting

| Message | Fix |
| --- | --- |
| `[FAIL] runs_dir: ... Drive is not mounted` | Run the *Mount Google Drive* cell and accept the prompt. |
| `[FAIL] device: ... no GPU is visible` | Runtime → Change runtime type → T4 GPU, then Run all. If no GPU is available, try later; the free tier has limits. |
| `[warn] free space: ... GB free` | Under 8 GB on Drive: delete old runs (`kodoom runs` shows sizes) or push finished models to Hugging Face and delete them. |
| `NoSpaceError ... the previous checkpoint is intact` | Free space on Drive, then run the same step again; it resumes. |
| `run ... already exists with a different config` | You changed a setting. Revert it to resume, or use a new run id. |
| The *Get the code* cell fails with an authentication error | The repository is private: add the token secret named in `TOKEN_SECRET` (one-time setup, step 2). |
