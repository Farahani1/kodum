# Proposed additions to the plan

Changes suggested for `docs/project-plan.md` that the owner has not yet accepted. The plan
itself is not edited until they are; each entry says where it would go and why.

## data_dir describes itself (Environments / Storage budget)

**Proposed text,** after the principle "Data stays on the owner's Drive":

> `data_dir` carries a `README.md` written by kodoom: the file tree with sizes, record
> counts and what each folder holds, the files the last update made or changed, and the
> history of updates with their command lines. Every command that writes data refreshes
> it. The Colab notebook records the files when a run starts and ends by printing the tree
> with the files that run generated or updated marked.

**Why:** data stays on Drive and is not in git, so nothing else records which translator
or command produced a file, or when. The README makes the folder readable on its own when
it is shared or reviewed, and the end-of-run tree shows at a glance whether a notebook run
wrote what it was meant to.

**Status:** implemented (`src/kodoom/datadir.py`, `kodoom tree`, the first and last data
cells of `notebooks/colab.ipynb`); only the plan text waits for the owner.
