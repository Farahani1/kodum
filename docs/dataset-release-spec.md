# Published dataset specification

Oct 8, 2026 · release contract for project-plan v23, section 1.3

The final data deliverable is a versioned, ready-to-use dataset with an audit
trail and runnable reproduction recipes. People should be able to use it for
evaluation or permitted training without running our translator or adopting our
reference model. This document specifies the target release; it does not claim
that packaging, human review or publication has already been completed.

## What to publish

Prefer separate Hugging Face dataset repositories for these resources, linked
from one project page or collection:

| Resource | Contents | Release boundary |
| --- | --- | --- |
| `typed-decisions-fa` | Paired English/Persian workflow decisions; original case IDs, options, splits and teacher targets | Apache-2.0 source attribution; translator provenance and downstream-use qualifications on the card |
| Persian skills suite | Generated Persian decisions, minimal pairs, seeds, templates and exact code labels | Our CC0-1.0 data/templates; linked 0BSD generator code |
| Optional helmo-fa augmentation | Selected translated training examples, selection manifest and conversion details | Separate MIT-derived resource; never included in benchmark test splits |

Avoid putting all resources under one blanket data license. Native-source
converters and evaluation recipes belong in the code repository. Redistribute
converted source text only after its permissions and attribution are checked;
otherwise provide instructions to obtain it from the original source. Restricted
or private evaluation texts are not part of the public bundle.

Use **Parquet for convenient loading and browsing**, plus **UTF-8 JSONL as an
auditable interchange copy** of the same records. Hugging Face recommends
Parquet for most datasets and supports JSONL for nested data. Both copies must
come from the same frozen record set, with a documented, lossless conversion.
Do not flatten probability targets into only a winning label. See the official
[upload and file-format guidance](https://huggingface.co/docs/hub/datasets-adding).

## Rows, configurations and splits

The primary unit is **one decision**, using the existing
[`Record` schema](../src/kodoom/schema.py). A case with five questions therefore
has five decision rows sharing a `source_id`. For the full typed-decisions source,
report both units: 1,600 cases / 8,000 decisions per language, with 1,200 train
and 400 test cases. These are target counts, not evidence of completed review.

Use explicit `fa` and `en` configurations with matching upstream decision keys.
Preserve current record IDs: the translator adds `:fa` to the English decision
ID. The exporter must record the unmodified upstream decision ID in
`extra.source_decision_id` in both views. Pair by source, pinned source revision
and that key, rather than row position or assuming identical row IDs. Record
IDs are unique within a configuration. Additional mixed-language views are separately named and
documented. An optional case view groups the five decisions and shared state;
it must be derived from the same records and describe any differences from the
original source's layout.

Keep upstream train/test membership in the published translated resource.
Model-development and calibration selections come only from upstream train
and are distributed as separate immutable ID manifests. For the skills suite,
publish its actual split policy and held-out template IDs. A source case, its
language variants and both halves of a minimal pair stay together. Never move
test templates into training or silently replace official splits with a model's
training mix. Configure the Hub's files and splits explicitly in card metadata;
see [data-files configuration](https://huggingface.co/docs/hub/datasets-data-files-configuration).

## Record contents

| Fields | Meaning |
| --- | --- |
| `id`, `source_id` | Stable decision ID and original case/pair grouping ID |
| `source`, `source_revision`, `license` | Source identity, immutable upstream revision and source license |
| `split`, `origin`, `task_family` | Partition, translated/native/synthetic origin and workflow or skill family |
| `state_lang`, `question_lang` | Language of the context and question |
| `state`, `question_type`, `question_text` | Complete context, choice/score/noul type and question |
| `options` | Ordered option IDs and texts; preserve the source's IDs and score-scale order |
| `gold` | Probabilities keyed by option ID; one-hot only for genuinely hard labels |
| `checks_passed`, `meaning_flag`, `human_reviewed` | Separate automatic, meaning-check and human-review status |
| `extra` | Source details, generator metadata and translation/review provenance |

JSONL follows `Record.to_dict()` exactly: `state` is a string (serialized JSON
when structured), `options` is a list of `{id, text}` objects, and `gold` is an
ID-to-probability object. Keep additional metadata inside `extra`, because the
current parser rejects unknown top-level fields. Identify the schema version
in the release manifest and document null/false/true meanings; an unperformed
check must not look like a successful check.

For the Parquet export, use an explicitly versioned transport schema:
`options` remains an ordered list; `gold` becomes an ordered list of
`{id, probability}` entries following option order, without altering values;
`extra_json` stores the UTF-8 JSON representation of `extra`. Other fields retain
their logical types. The export/import adapter must recover the original
`Record`, including sparse gold maps if present. This avoids an expanding column
schema for different option IDs and metadata. The card must show this small
conversion and a working loading example for both formats. The adapter and
round-trip verification are release work still to implement.

Preserve full context, protected tokens, raw digit forms and spelling variants.
Published translations receive only the planned orthographic cleanup; model
input normalization is separate. Translation does not change source gold.
Document questionable source targets and provide a clean-subset exclusion
manifest. If adjudicated labels are later offered, version them as a separate
view with explicit reasons, while retaining the original-target view.

## Files accompanying the data

The following tree is a target layout for `typed-decisions-fa`, not a current
public repository. The skills release uses the same pattern with generator
metadata and its actual splits.

```text
README.md                       # dataset card and quickstart
LICENSE                         # applicable data license
NOTICE                          # source attribution/notices where applicable
schema.md                       # logical and transport schemas, examples
manifest.json                   # release version, code/source revisions, counts
SHA256SUMS                      # checksums of release payload files
data/fa/train.parquet
data/fa/test.parquet
data/en/train.parquet
data/en/test.parquet
jsonl/fa/train.jsonl             # the same decisions in Record format
jsonl/fa/test.jsonl
jsonl/en/train.jsonl
jsonl/en/test.jsonl
provenance/                     # prompts, glossary, settings, source selection
review/                         # quality report, correction/exclusion manifests
splits/                         # optional train-derived dev/calibration ID lists
reproduce.md                    # pinned code link and tested commands
```

The manifest records release/schema versions, code commit, upstream revisions
and hashes, actual case/decision counts by split/language/family/type, generator
versions/seeds/template hashes, and translator/checker checkpoints and revisions.
Record prompts, glossary, decoding settings, truncation policy, dependency
locks, relevant runtime/hardware and any selection or rejected IDs. The review
report records reviewed denominators, sampling method, defects, corrections and
unresolved source-label issues separately from translation errors. Link the
approved translation drafts and correction patches needed for rebuilding,
subject to their redistribution permissions; omit credentials and private logs.

The card explains intended uses, provenance, license and attribution, label
semantics, full-test versus sampled-train review, limitations, citation, version
history and loading/scoring examples. Do not present synthetic teacher targets
as human-verified truth, or LLM ratings as human review. State applicable
downstream-model restrictions separately from source licensing; Gemma 3 output
training eligibility remains subject to the plan's license gate. See the
[Hugging Face dataset-card guidance](https://huggingface.co/docs/hub/datasets-cards).

## Should reproduction code be included?

**Yes, for this project.** Releasing data with preparation/generation code is good
research practice and supports our promised reproducibility. It is not a
universal prerequisite for uploading a dataset: the Hub supports direct file
uploads. Our project makes it a release requirement, especially because the
skills generators are themselves a contribution. Users should still be able
to load the static data without executing custom remote loading code.

Keep implementation in the public code repository and link an immutable commit
or release tag from each data release. Include converters, source-fetch recipes,
generators/templates/seeds, translation prompts and settings, review-patch
application, split/overlap checks, export/import verification, dependency locks
and a minimal scoring example. Archive the code release when feasible. Copying
the entire codebase into every dataset repository is unnecessary. Generated
data remains outside this code checkout and is never committed here.

Document three different reproduction promises:

1. **Rebuild the published release:** start from pinned, redistributable source
   material, archived approved drafts and review patches, then rebuild records
   and verify their canonical content hashes. Freeze serialization details if
   byte-for-byte JSONL is promised. Retain original Parquet files and checksums;
   Parquet bytes can differ across writer versions despite identical records.
2. **Regenerate skills:** pinned code, templates, versions, seeds and settings
   reproduce the records exactly in the documented environment. Verify labels,
   pair membership and split/template boundaries independently.
3. **Rerun translation:** provide the recipe and access/hardware requirements,
   but disclose that model inference can differ across software or hardware.
   A seed cannot regenerate a person's editorial choices; retained drafts and
   patches are what make the reviewed release reconstructable.

## Acceptance before publication

Require schema validation and JSONL/Parquet round trips; matching bilingual keys,
source option IDs and unchanged gold; count/hash checks; split and minimal-pair
leakage checks; complete review of all 400 test cases; documented training-sample
review and unresolved flags; license/attribution/provenance review; and a fresh
CPU-only loading, rebuilding and scoring example by someone other than the
release author where possible. A pilot must clearly state its smaller scope
and incomplete coverage. A public upload still waits for the owner's release
decision under the project plan.

Existing schema, converters, generators and validators provide part of this
foundation. The final exporter, manifests, correction replay and end-to-end
release verification must be implemented and checked before claiming a
reproducible v1.0. This specification does not advance that gate.
