"""Run directories: logs and checkpoints that survive a dead Colab session.

Layout under ``<runs_dir>/<run_id>/`` (on Colab, ``runs_dir`` is on Drive):

- ``run.json``: what the run is (its config, git commit, profile) and its status.
- ``log.jsonl``: one line per event, flushed and fsynced as it is written, so a
  session that dies loses at most the line being written.
- ``latest/``: the newest resumable checkpoint (weights and optimizer state).
- ``best/``: the best weights so far, for early stopping (weights only).

A checkpoint is written to the local scratch disk first, then copied to
``runs_dir`` under a temporary name, then swapped in by renaming. Its
``MANIFEST.json`` is written last and lists every file with its size, so a
directory without a matching manifest is incomplete and is never resumed from.
At any moment there is at least one complete copy of each checkpoint on Drive.

Finished runs get a row in ``<runs_dir>/registry.csv`` (plan 2.3).
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kodoom.config import Profile

MANIFEST = "MANIFEST.json"
KINDS = ("latest", "best")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_GB = 1024**3


class RunError(RuntimeError):
    """A run cannot be opened, resumed or saved as asked."""


class NoSpaceError(RunError):
    """Writing a checkpoint would leave less than the reserve free."""


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    kind: str
    step: int
    metric: float | None
    size_bytes: int
    saved_at: str


class Run:
    """One training or evaluation run; open it with ``Run.open``."""

    def __init__(self, directory: Path, scratch: Path, reserve_bytes: int, meta: dict) -> None:
        self.dir = directory
        self.scratch = scratch
        self.reserve_bytes = reserve_bytes
        self.meta = meta

    # -- opening and resuming ------------------------------------------------

    @classmethod
    def open(cls, profile: Profile, run_id: str, config: dict[str, Any]) -> Run:
        """Start a new run, or resume one with the same ID and the same config.

        Resuming with a different config is refused: continuing a checkpoint
        under changed settings would produce a model nobody can reproduce.
        """
        if not RUN_ID.fullmatch(run_id):
            raise RunError(f"run id {run_id!r}: use letters, digits, '.', '_' and '-'")
        config = json.loads(json.dumps(config))  # normalize (tuples to lists, etc.)
        directory = profile.runs_dir / run_id
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory / "run.json"

        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["config"] != config:
                changed = sorted(
                    k
                    for k in set(meta["config"]) | set(config)
                    if meta["config"].get(k) != config.get(k)
                )
                raise RunError(
                    f"run {run_id!r} already exists with a different config "
                    f"(changed: {', '.join(changed)}); resume with the same settings "
                    f"or start a new run id"
                )
            if meta["status"] == "finished":
                raise RunError(f"run {run_id!r} is already finished; start a new run id")
            meta["resumed"] = meta.get("resumed", 0) + 1
        else:
            meta = {
                "run_id": run_id,
                "profile": profile.name,
                "config": config,
                "git_commit": git_commit(),
                "status": "running",
                "created": _now(),
                "resumed": 0,
            }
        meta["updated"] = _now()
        _write_json(meta_path, meta)

        run = cls(
            directory,
            profile.scratch_dir / run_id,
            int(profile.reserve_gb * _GB),
            meta,
        )
        for kind in KINDS:
            run._recover(kind)
        run.log("resumed" if meta["resumed"] else "started", git_commit=meta["git_commit"])
        return run

    @property
    def run_id(self) -> str:
        return self.meta["run_id"]

    @property
    def latest(self) -> Checkpoint | None:
        """The checkpoint to resume from, or None for a fresh start."""
        return _read_checkpoint(self.dir / "latest", "latest")

    @property
    def best(self) -> Checkpoint | None:
        return _read_checkpoint(self.dir / "best", "best")

    # -- logging -------------------------------------------------------------

    def log(self, event: str = "metrics", **fields: Any) -> None:
        """Append one event to log.jsonl and force it to disk."""
        line = json.dumps({"time": _now(), "event": event, **fields}, ensure_ascii=False)
        with (self.dir / "log.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_log(self) -> list[dict[str, Any]]:
        path = self.dir / "log.jsonl"
        if not path.exists():
            return []
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(json.JSONDecodeError):  # a line cut by a dead session
                events.append(json.loads(line))
        return events

    # -- checkpoints ---------------------------------------------------------

    def save_latest(self, step: int, write: Callable[[Path], None]) -> Checkpoint:
        """Save a resumable checkpoint; ``write(dir)`` puts the files into ``dir``."""
        return self._save("latest", step, None, write)

    def save_best(
        self,
        step: int,
        metric: float,
        write: Callable[[Path], None],
        *,
        lower_is_better: bool = True,
    ) -> Checkpoint | None:
        """Save weights only if ``metric`` beats the best so far; else return None."""
        current = self.best
        if current is not None and current.metric is not None:
            better = metric < current.metric if lower_is_better else metric > current.metric
            if not better:
                return None
        return self._save("best", step, metric, write)

    def finish(self, **summary: Any) -> None:
        """Close the run: drop the resumable checkpoint, keep the best weights.

        The latest checkpoint (with optimizer state) is the largest file of the
        run and is only needed to resume, so it goes (plan: Storage budget).
        """
        shutil.rmtree(self.dir / "latest", ignore_errors=True)
        shutil.rmtree(self.scratch, ignore_errors=True)
        best = self.best
        self.meta.update(status="finished", finished=_now(), updated=_now(), summary=summary)
        _write_json(self.dir / "run.json", self.meta)
        self.log("finished", **summary)
        _append_registry(self.dir.parent / "registry.csv", self.meta, best)

    def _save(
        self, kind: str, step: int, metric: float | None, write: Callable[[Path], None]
    ) -> Checkpoint:
        # 1. Write on the fast local disk. If this fails, Drive is untouched.
        stage = self.scratch / kind
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        write(stage)
        files = {p.relative_to(stage).as_posix(): p.stat().st_size for p in _files(stage)}
        if not files:
            raise RunError(f"{kind} checkpoint at step {step}: write() produced no files")
        size = sum(files.values())

        # 2. Refuse before copying if it would not fit with the reserve left free.
        free = shutil.disk_usage(self.dir).free
        if size + self.reserve_bytes > free:
            raise NoSpaceError(
                f"{kind} checkpoint at step {step} needs {size / _GB:.2f} GB plus a "
                f"{self.reserve_bytes / _GB:.2f} GB reserve, but only {free / _GB:.2f} GB "
                f"is free under {self.dir}. Free space on Drive and resume; the previous "
                f"checkpoint is intact."
            )

        # 3. Copy to Drive under a temporary name; the manifest goes last.
        incoming = self.dir / f".{kind}.incoming"
        shutil.rmtree(incoming, ignore_errors=True)
        shutil.copytree(stage, incoming)
        for rel in files:
            _fsync_file(incoming / rel)
        manifest = {"kind": kind, "step": step, "metric": metric, "files": files}
        manifest["saved_at"] = _now()
        _write_json(incoming / MANIFEST, manifest)

        # 4. Swap it in. A crash in between leaves .old complete; _recover() uses it.
        target, old = self.dir / kind, self.dir / f".{kind}.old"
        shutil.rmtree(old, ignore_errors=True)
        if target.exists():
            target.rename(old)
        incoming.rename(target)
        shutil.rmtree(old, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)

        self.log("checkpoint", kind=kind, step=step, metric=metric, size_bytes=size)
        return Checkpoint(target, kind, step, metric, size, manifest["saved_at"])

    def _recover(self, kind: str) -> None:
        """Repair what a session that died during a save left behind."""
        target, old = self.dir / kind, self.dir / f".{kind}.old"
        incoming = self.dir / f".{kind}.incoming"
        shutil.rmtree(incoming, ignore_errors=True)  # never complete: the swap had not begun
        if _read_checkpoint(target, kind) is not None:
            shutil.rmtree(old, ignore_errors=True)
            return
        if target.exists():  # present but incomplete or damaged: keep it aside, never delete
            broken = self.dir / f".{kind}.broken-{datetime.now(UTC):%Y%m%dT%H%M%S}"
            target.rename(broken)
            self.log("warning", message=f"{kind} checkpoint was incomplete; moved to {broken.name}")
        if _read_checkpoint(old, kind) is not None:
            old.rename(target)
            self.log("recovered", kind=kind, step=_read_checkpoint(target, kind).step)


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Summaries of every run under ``runs_dir``, for ``kodoom runs``."""
    if not runs_dir.is_dir():
        return []
    rows = []
    for d in sorted(runs_dir.iterdir()):
        meta_path = d / "run.json"
        if d.name.startswith(("_", ".")) or not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        latest = _read_checkpoint(d / "latest", "latest")
        best = _read_checkpoint(d / "best", "best")
        rows.append(
            {
                "run_id": meta["run_id"],
                "status": meta["status"],
                "profile": meta["profile"],
                "updated": meta.get("updated", ""),
                "latest_step": latest.step if latest else None,
                "best_step": best.step if best else None,
                "best_metric": best.metric if best else None,
                "size_bytes": sum(p.stat().st_size for p in _files(d)),
            }
        )
    return rows


def git_commit() -> str | None:
    """The commit the code runs from, with '-dirty' if there are local edits."""
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None  # not installed from a git checkout
    return commit + ("-dirty" if dirty else "")


def _read_checkpoint(path: Path, kind: str) -> Checkpoint | None:
    """A checkpoint only counts if its manifest exists and every file matches it."""
    try:
        manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for rel, size in manifest["files"].items():
        f = path / rel
        if not f.is_file() or f.stat().st_size != size:
            return None
    return Checkpoint(
        path,
        kind,
        manifest["step"],
        manifest["metric"],
        sum(manifest["files"].values()),
        manifest["saved_at"],
    )


def _append_registry(path: Path, meta: dict, best: Checkpoint | None) -> None:
    row = {
        "run_id": meta["run_id"],
        "profile": meta["profile"],
        "created": meta["created"],
        "finished": meta["finished"],
        "git_commit": meta["git_commit"],
        "resumed": meta["resumed"],
        "best_step": best.step if best else "",
        "best_metric": best.metric if best else "",
        "summary": json.dumps(meta.get("summary", {}), ensure_ascii=False),
    }
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _fsync_file(path: Path) -> None:
    with path.open("rb+") as f:
        os.fsync(f.fileno())


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
