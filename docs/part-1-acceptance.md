# Part 1 acceptance handoff

This is the local acceptance workflow for project-plan Part 1. It consumes the
completed translations; it does not rerun the historical TPU campaign.
All generated files below stay in git-ignored `data/part1/`.

## Frozen input and import

The supplied package is `D:\Coding\hf-data`. Its original
`provenance/release.json` SHA-256 is:

```text
05758f7e10d7e58cb90f7ece4058e2597b1535b2937acf2a26567eb1b752ba05
```

Its validation script passed on a private copy: 8 configurations, 14 splits,
10,000 bilingual questions, 3,600 cases and 307 inventoried files. The original
package was not modified. That validator rewrites its validation report and
release manifest, so preserve the original input manifest and its pin separately
from any subsequently validated copy.

From an installed kodoom checkout, import into a new directory:

```powershell
kodoom import-package --profile dev D:/Coding/hf-data --manifest-sha256 05758f7e10d7e58cb90f7ece4058e2597b1535b2937acf2a26567eb1b752ba05 --out data/part1/import-v3
```

`import-v3/import-report.json` records all 10,000 paired decisions: 6,000
typed-decisions train, 2,000 typed-decisions test, and 2,000 Helmo train.
Both languages are exported as Record JSONL and versioned Parquet. The import
checks source fields against frozen Records, keeps original gold and split
membership, retains raw digits, and applies orthographic cleanup only.
File checksums protect subsequent staging from changed import outputs.

## Private release candidates

Use fresh output directories; commands refuse to overwrite an existing build.
The full skills command explicitly requests 150 pairs per question kind, rather
than using the smaller development-profile generation cap.

```powershell
kodoom release-skills --profile dev --seed 1234 --pairs-per-kind 150 --out data/part1/skills-candidate
kodoom release-package --profile dev --source D:/Coding/hf-data --imported data/part1/import-v3 --out data/part1/bilingual-candidate
```

The bilingual candidate keeps typed-decisions and optional Helmo separate.
Each resource has Record JSONL, Parquet, a card, source attribution, checksums,
input provenance and a review status. Archived bilingual inputs preserve the
pre-cleanup translations and frozen English Records. No AI suggestion is
silently applied, and original targets remain unchanged. The supplied package's
license limitations are copied, not replaced with a new translation grant.

The skills candidate records generator versions, seed, held-out templates and
template hashes. Generation is repeated and compared before export. Pair
membership, different answers, split boundaries, schema and both file formats
are checked. Existing generator tests independently exercise label calculations.

Run the static CPU loading and uniform-baseline scoring examples:

```powershell
python data/part1/skills-candidate/examples/load_and_score.py data/part1/skills-candidate
python data/part1/bilingual-candidate/typed_decisions/examples/load_and_score.py data/part1/bilingual-candidate/typed_decisions
```

These are harness checks, not trained-model results or M2 measurements.
Candidate manifests record the code revision used for the build; the revision
must be clean and committed for a reproducible handoff.

## Observed verification — Oct 16, 2026

Final candidates were built from clean commit
`94ccdf75cde0205fdad76c3c0d241fd997367560`. Fresh isolated Python processes
verified all exported JSONL/Parquet records and every payload checksum:

| Candidate | Decision rows across language views | Payload checksums |
| --- | ---: | ---: |
| `skills-candidate` | 4,200 Persian rows / 2,100 pairs | 17 |
| `bilingual-candidate/typed_decisions` | 16,000 rows / 8,000 bilingual decisions | 160 |
| `bilingual-candidate/helmo` | 4,000 rows / 2,000 bilingual decisions | 154 |

All four skills split JSONL hashes exactly match the earlier independent local
build. Static scoring examples loaded and scored all 1,036 skills test decisions
and 2,000 Persian workflow test decisions. The bilingual checker queues contain
1,600 workflow cases and 2,000 Helmo cases. These queues have not been executed.
External reproduction by another person and Hub/Viewer checks remain unverified.

Repository verification: Ruff lint/format checks passed; the full test suite
passed 771 tests. Updated CLI/import/release checks passed 24 focused tests,
including the 9 protected-text importer tests. An earlier restricted run failed
in two pre-existing temporary-directory permission checks; the unrestricted
full run passed. No dataset files, model outputs or credentials are committed.

## Quality status and remaining decisions

The broader automatic import checks flagged 4,398 Persian decisions across
1,842 source cases. Findings overlap and repeated case states contribute to
multiple decisions. This is not a measured translation-error rate:

| Check | Finding occurrences |
| --- | ---: |
| Glossary wording | 4,310 |
| English retained | 3,512 |
| Script | 249 |
| Protected identifiers | 119 |
| Repetition | 120 |
| Option labels | 2 |
| Number loss | 0 |

Some flags concern code, mathematical notation, or wording already present in
English. They require triage; no claim is made that every flagged item is wrong,
or that unflagged items are semantically correct. The supplied AI audit covered
100 cases / 300 decisions, not the whole corpus. Its proposed Helmo source-label
holds remain unapplied and require a separate inclusion/exclusion decision.
The final import protects embedded code indentation, inline code, and URLs
during orthographic cleanup. This removed 16 spurious identifier findings
from the earlier import; the earlier 4,406-decision screening count is superseded.
Helmo adoption additionally requires the plan's 100-record English-source audit;
the supplied audit covers 50 Helmo records. Helmo remains optional for M1.

Each bilingual resource's `review/meaning-check-requests.jsonl` contains whole
cases with English/Persian state, questions and options, without gold targets.
These are prepared requests, not completed checker results. The planned meaning
check (1.2 step 4), including separate English/Persian answering for Helmo, has
not run. An accelerator execution route and checker implementation/validation
still need to be prepared before dispatch. The active execution notebook remains
the historical translation campaign and must not be run for this new stage.

Human translation review is waived by the owner on Oct 15, 2026. Keep
`human_reviewed=false` and disclose that no human translation-error rate exists.
The waiver does not waive automatic checks, semantic acceptance, source-label
validity, distribution permissions, or the downstream training-license gate.

The owner must supply/confirm the translation-specific license grant and decide
the quality-check execution route before public release can proceed. Source
Apache-2.0/MIT labels alone do not supply that grant. Preserve unknown historical
model IDs and settings as unknown. Training eligibility needs its own documented
source/teacher/translator decision and enforcement before Part 2 training.

M1 remains open until technical acceptance and permissions are resolved and the
owner explicitly authorizes publication. Hub loading/Viewer verification follows
that publication decision. No dataset upload is part of these local commands.
