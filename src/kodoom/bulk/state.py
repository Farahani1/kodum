"""Atomic complete-case shards and verified campaign identity (BULK-01/04)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kodoom.schema import Record


def encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
        or path.as_posix() != name
    ):
        raise ValueError("unsafe campaign artifact path")
    return name


@dataclass(frozen=True)
class Unit:
    dataset: str
    records: tuple[Record, ...]

    def __post_init__(self):
        if not self.records:
            raise ValueError("empty work unit")
        first = self.records[0]
        if self.dataset not in ("typed-decisions", "helmo") or any(
            r.source_id != first.source_id or r.split != first.split for r in self.records
        ):
            raise ValueError("a work unit must hold one complete source case")
        if len({r.id for r in self.records}) != len(self.records):
            raise ValueError("duplicate decision IDs")
        if len(self.records) != (5 if self.dataset == "typed-decisions" else 1):
            raise ValueError("a typed case needs all five decisions; helmo needs one")
        source = (
            "LocalLLaMA/typed-decisions"
            if self.dataset == "typed-decisions"
            else "helmo/synthetic-typed-decisions"
        )
        if first.source != source or any(
            (r.source, r.source_revision, r.state, r.extra.get("workflow"))
            != (first.source, first.source_revision, first.state, first.extra.get("workflow"))
            for r in self.records
        ):
            raise ValueError("case source, revision, workflow or shared state changed")

    @property
    def key(self) -> str:
        r = self.records[0]
        return digest(encoded([self.dataset, r.split, r.source_id]))

    def to_dict(self) -> dict:
        return {"dataset": self.dataset, "records": [r.to_dict() for r in self.records]}


def validate_pair(unit: Unit, translated: list[Record]) -> None:
    if len(translated) != len(unit.records):
        raise ValueError("incomplete translated case")
    mutable = {
        "id",
        "origin",
        "state_lang",
        "question_lang",
        "state",
        "question_text",
        "options",
        "checks_passed",
        "extra",
    }
    for en, fa in zip(unit.records, translated, strict=True):
        left, right = en.to_dict(), fa.to_dict()
        if {k: v for k, v in left.items() if k not in mutable} != {
            k: v for k, v in right.items() if k not in mutable
        }:
            raise ValueError("translation changed source metadata, gold or review flags")
        if (
            fa.id != en.id + ":fa"
            or fa.origin != "translated"
            or fa.state_lang != "fa"
            or fa.question_lang != "fa"
            or [o.id for o in fa.options] != [o.id for o in en.options]
            or any(fa.extra.get(k) != v for k, v in en.extra.items())
            or fa.extra.get("review_status") != "unreviewed-draft"
        ):
            raise ValueError("translation changed identity or original extra fields")
    if len({r.state for r in translated}) != 1:
        raise ValueError("case decisions no longer share one state")


class Campaign:
    def __init__(self, root: Path, identity: dict, units: list[Unit]):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.units = {unit.key: unit for unit in units}
        if len(self.units) != len(units):
            raise ValueError("duplicate source case IDs")
        inputs = encoded([u.to_dict() for u in units])
        self.identity = {"schema_version": 1, **identity, "inputs_sha256": digest(inputs)}
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "campaign.json"
        if manifest.exists():
            if manifest.read_bytes() != encoded(self.identity):
                raise ValueError(
                    "campaign configuration/input identity changed; use a new campaign"
                )
            if (self.root / "inputs.json").read_bytes() != inputs:
                raise ValueError("frozen English inputs changed")
        else:
            if any(self.root.iterdir()):
                raise ValueError("new campaign root must be empty")
            atomic_write(self.root / "inputs.json", inputs)
            atomic_write(manifest, encoded(self.identity))
        path = self.root / "progress.json"
        self.progress = (
            json.loads(path.read_bytes())
            if path.exists()
            else {"schema_version": 1, "completed": {}, "failed": {}, "status": "pending"}
        )
        if self.progress.get("schema_version") != 1:
            raise ValueError("unsupported progress schema")
        self.production = None
        if (path := self.root / "production.json").exists():
            self.production = json.loads(path.read_bytes())
        self.verify()
        # Reconcile an atomic case committed just before a local process died.
        for path in sorted((self.root / "shards").glob("*.jsonl")):
            key = path.stem
            if key not in self.progress["completed"]:
                self._read_shard(key, path.read_bytes())
                self.progress["completed"][key] = self._reference(path)
                self.progress["failed"].pop(key, None)
        self.save_progress()
        self.verify()

    def _reference(self, path: Path) -> dict:
        return {
            "path": path.relative_to(self.root).as_posix(),
            "sha256": digest(path.read_bytes()),
            "records": len(self.units[path.stem].records),
        }

    def _read_shard(self, key: str, data: bytes) -> list[Record]:
        if key not in self.units or not data.endswith(b"\n"):
            raise ValueError("unexpected or unfinished case shard")
        value = json.loads(data)
        if value.get("key") != key or value.get("english") != self.units[key].to_dict():
            raise ValueError("shard English case identity changed")
        fa = [Record.from_dict(r) for r in value["persian"]]
        validate_pair(self.units[key], fa)
        if any(r.extra.get("translator") != self.identity["request"]["translator"] for r in fa):
            raise ValueError("shard translator changed")
        return fa

    def verify(self) -> None:
        for key, ref in self.progress["completed"].items():
            path = self.root / safe_name(ref["path"])
            if ref["path"] != f"shards/{key}.jsonl":
                raise ValueError("shard path does not match case key")
            data = path.read_bytes()
            if digest(data) != ref["sha256"]:
                raise ValueError("case shard checksum changed")
            self._read_shard(key, data)
            if ref["records"] != len(self.units[key].records):
                raise ValueError("shard decision count changed")
        if not set(self.progress["failed"]) <= self.units.keys():
            raise ValueError("failed progress names unknown work")
        if self.progress["completed"] and self.production is None:
            raise ValueError("translated campaign has no frozen production configuration")

    def freeze_production(self, production: dict) -> None:
        with self.lock:
            if self.production is not None and self.production != production:
                raise ValueError("production batch or backend identity changed")
            if self.production is None:
                atomic_write(self.root / "production.json", encoded(production))
                self.production = production

    def save_progress(self) -> None:
        atomic_write(self.root / "progress.json", encoded(self.progress))

    def complete(self, unit: Unit, translated: list[Record]) -> None:
        with self.lock:
            if self.production is None:
                raise ValueError("freeze production before saving translations")
            validate_pair(unit, translated)
            if unit.key in self.progress["completed"]:
                raise ValueError("case already completed")
            data = encoded(
                {
                    "key": unit.key,
                    "english": unit.to_dict(),
                    "persian": [r.to_dict() for r in translated],
                }
            )
            self._read_shard(unit.key, data)
            path = self.root / "shards" / f"{unit.key}.jsonl"
            atomic_write(path, data)
            self.progress["completed"][unit.key] = self._reference(path)
            self.progress["failed"].pop(unit.key, None)
            self.save_progress()

    def fail(self, unit: Unit, reason: str) -> None:
        with self.lock:
            self.progress["failed"][unit.key] = reason
            self.save_progress()

    def pending(self) -> list[Unit]:
        return [u for key, u in self.units.items() if key not in self.progress["completed"]]

    def counts(self) -> dict:
        result = {}
        with self.lock:
            for key, unit in self.units.items():
                name = f"{unit.dataset}/{unit.records[0].split}"
                row = result.setdefault(name, {"total": 0, "completed": 0, "failed": 0})
                row["total"] += 1
                row["completed"] += key in self.progress["completed"]
                row["failed"] += key in self.progress["failed"]
            return result

    def snapshot(self) -> dict[str, bytes]:
        with self.lock:
            files = {"campaign.json", "inputs.json", "progress.json"}
            if self.production is not None:
                files.add("production.json")
            if (self.root / "events.jsonl").exists():
                files.add("events.jsonl")
            if (self.root / "tpu-metrics.json").exists():
                files.add("tpu-metrics.json")
            files.update(ref["path"] for ref in self.progress["completed"].values())
            files.update(
                p.relative_to(self.root).as_posix() for p in (self.root / "review").glob("*.csv")
            )
            return {name: (self.root / name).read_bytes() for name in sorted(files)}

    def event(self, kind: str, **details) -> None:
        from datetime import UTC, datetime

        with self.lock, (self.root / "events.jsonl").open("ab") as out:
            out.write(encoded({"time": datetime.now(UTC).isoformat(), "event": kind, **details}))
            out.flush()
            os.fsync(out.fileno())

    def translated(self, unit: Unit) -> list[Record]:
        with self.lock:
            return self._read_shard(unit.key, (self.root / f"shards/{unit.key}.jsonl").read_bytes())
