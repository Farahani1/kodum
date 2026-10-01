"""What is in ``data_dir``: its README, its change history and the file tree.

Every command that writes data refreshes ``<data_dir>/README.md`` (see
``update_readme``), so whoever opens the folder on Drive sees what each file
is, how big it is, and what the last commands changed. Bookkeeping lives in
the hidden ``.kodoom/`` folder next to it; hidden files and folders are never
listed.

``kodoom tree --start`` records the files at the start of a notebook run, and
``kodoom tree`` then prints the tree with the files that run made or changed
marked.
"""

from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

README = "README.md"
STATE_DIR = ".kodoom"
HISTORY = "history.jsonl"
README_SNAPSHOT = "readme-snapshot.json"
START_SNAPSHOT = "tree-start.json"
HISTORY_SHOWN = 20

# What each folder holds, keyed by its path relative to data_dir (fnmatch patterns).
DESCRIPTIONS = {
    "skills": "code-labeled Persian skill data (`kodoom generate`)",
    "typed-decisions": "LocalLLaMA/typed-decisions and its Persian versions",
    "typed-decisions/en": "English originals at the pinned commit (`kodoom fetch typed-decisions`)",
    "typed-decisions/fa": "Persian translations, one folder per translator",
    "typed-decisions/fa/*": "one translator's output (`kodoom translate` or `kodoom import-units`)",
    "typed-decisions/pilot": "blind review sheet and its key (`kodoom pilot-sheet`)",
    "typed-decisions/exchange": "units for a translator outside kodoom (`kodoom export-units`)",
    "helmo": "helmo/synthetic-typed-decisions",
    "helmo/en": "English records at the pinned commit (`kodoom fetch helmo`)",
}

Snapshot = dict[str, tuple[int, int]]  # relative posix path -> (size, mtime_ns)


@dataclass
class Changes:
    new: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.new or self.updated or self.removed)

    def marks(self) -> dict[str, str]:
        return {**{p: "new" for p in self.new}, **{p: "updated" for p in self.updated}}


def snapshot(root: Path) -> Snapshot:
    """Size and modification time of every visible file under ``root``."""
    files: Snapshot = {}
    if not root.is_dir():
        return files
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in filenames:
            if name.startswith(".") or name.endswith(".tmp"):
                continue
            path = Path(dirpath) / name
            st = path.stat()
            files[path.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return files


def diff(before: Snapshot, after: Snapshot, *, ignore: tuple[str, ...] = ()) -> Changes:
    keys = (set(before) | set(after)) - set(ignore)
    return Changes(
        new=sorted(k for k in keys if k in after and k not in before),
        updated=sorted(k for k in keys if k in after and k in before and after[k] != before[k]),
        removed=sorted(k for k in keys if k in before and k not in after),
    )


def save_start(root: Path) -> int:
    """Record the files now, for ``changes_since_start``; returns how many."""
    files = snapshot(root)
    _write_json(root / STATE_DIR / START_SNAPSHOT, {"time": _now(), "files": files})
    return len(files)


def changes_since_start(root: Path) -> tuple[str, Changes] | None:
    """When the start was recorded and what changed since, or None if it never was."""
    path = root / STATE_DIR / START_SNAPSHOT
    if not path.exists():
        return None
    start = json.loads(path.read_text(encoding="utf-8"))
    files = {k: tuple(v) for k, v in start["files"].items()}
    return start["time"], diff(files, snapshot(root))


def update_readme(root: Path, command: str, *, now: str | None = None) -> Changes | None:
    """Rewrite ``README.md`` if any file changed since it was last written.

    Changes are measured against the files recorded at the previous update,
    so files added by hand (a filled units file) are noticed too. Returns
    the changes, or None when nothing changed and the README was left alone.
    """
    if not root.is_dir():
        return None
    state = root / STATE_DIR
    previous_path = state / README_SNAPSHOT
    previous: Snapshot = {}
    if previous_path.exists():
        raw = json.loads(previous_path.read_text(encoding="utf-8"))
        previous = {k: tuple(v) for k, v in raw.items()}
    files = snapshot(root)
    changes = diff(previous, files, ignore=(README,))
    if not changes and ((root / README).exists() or not files):
        return None

    entry = {"time": now or _now(), "command": command, **vars(changes)}
    state.mkdir(parents=True, exist_ok=True)
    with (state / HISTORY).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    history = [
        json.loads(line)
        for line in (state / HISTORY).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    text = render_readme(root, files, changes, history, entry["time"])
    tmp = root / (README + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, root / README)
    files = snapshot(root)  # README's own size and time, so it never counts as a change
    _write_json(previous_path, files)
    return changes


def render_readme(
    root: Path, files: Snapshot, changes: Changes, history: list[dict], time: str
) -> str:
    lines = [
        "# kodoom data",
        "",
        f"Written by kodoom; do not edit, it is rewritten after every command that writes "
        f"data here. Last updated {time}.",
        "",
        "Generated, fetched and translated datasets for this profile. None of it is in git "
        "or published anywhere. Each `manifest.json` holds the source revision, the seed "
        "and a checksum.",
        "",
        "## Files",
        "",
        "`[new]` and `[updated]` mark what the last update changed.",
        "",
        "```",
        *tree_lines(
            root, {k: v for k, v in files.items() if k != README}, changes.marks(), records=True
        ),
        "```",
    ]
    if changes.removed:
        lines += ["", "Removed by the last update: " + ", ".join(f"`{p}`" for p in changes.removed)]
    lines += ["", "## History", "", "Newest first; changes since the update before.", ""]
    for entry in reversed(history[-HISTORY_SHOWN:]):
        parts = [
            f"{len(entry[kind])} {kind}" for kind in ("new", "updated", "removed") if entry[kind]
        ]
        lines.append(f"- {entry['time']} `{entry['command']}`: {', '.join(parts) or 'no files'}")
        for kind in ("new", "updated", "removed"):
            shown = entry[kind][:10]
            if shown:
                more = f" and {len(entry[kind]) - 10} more" if len(entry[kind]) > 10 else ""
                lines.append(f"  - {kind}: {', '.join(f'`{p}`' for p in shown)}{more}")
    return "\n".join(lines) + "\n"


def tree_lines(
    root: Path, files: Snapshot, marks: dict[str, str] | None = None, *, records: bool = False
) -> list[str]:
    """The files as an indented tree with sizes, folder descriptions and marks."""
    marks = marks or {}
    tree: dict = {}
    for path in files:
        node = tree
        for part in path.split("/"):
            node = node.setdefault(part, {})
    out = [f"{root.as_posix()}/"]

    def walk(node: dict, prefix: str, rel: str) -> None:
        names = sorted(node, key=lambda n: (not node[n], n))  # folders first
        for i, name in enumerate(names):
            last = i == len(names) - 1
            branch, child_prefix = ("└── ", "    ") if last else ("├── ", "│   ")
            path = f"{rel}{name}"
            if node[name]:
                note = _description(path)
                out.append(f"{prefix}{branch}{name}/" + (f"  — {note}" if note else ""))
                walk(node[name], prefix + child_prefix, path + "/")
            else:
                size = _size(files[path][0])
                if records and name.endswith(".jsonl"):
                    size += f", {_count_lines(root / path)} records"
                mark = f"  [{marks[path]}]" if path in marks else ""
                out.append(f"{prefix}{branch}{name}  {size}{mark}")

    walk(tree, "", "")
    if not files:
        out.append("(empty)")
    return out


def _description(path: str) -> str | None:
    for pattern, text in DESCRIPTIONS.items():
        if fnmatch.fnmatchcase(path, pattern) and path.count("/") == pattern.count("/"):
            return text
    return None


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)
