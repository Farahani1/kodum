"""Private storage, optimistic commits and explicit writer handoff (BULK-05/07)."""

from __future__ import annotations

import io
import json
import re
import threading
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from kodoom.bulk.state import atomic_write, digest, encoded, safe_name


class StorageError(RuntimeError):
    """Safe diagnostics that never include provider exceptions or credentials."""


class WriterConflict(StorageError):
    pass


class HFRemote:
    def __init__(self, repo: str, token: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not token:
            raise ValueError("select a private HF dataset repo and enable the HF_TOKEN secret")
        from huggingface_hub import HfApi

        self.repo, self.token = repo, token
        self.api = HfApi(token=token)
        try:
            info = self.api.repo_info(repo_id=repo, repo_type="dataset")
        except Exception:
            raise StorageError(
                "Cannot access HF dataset; check repository and token permissions"
            ) from None
        if info.private is not True:
            raise StorageError("Bulk checkpoints require a private HF dataset repository")

    def head(self) -> str:
        try:
            info = self.api.repo_info(repo_id=self.repo, repo_type="dataset")
            if info.private is not True:
                raise StorageError("HF dataset is no longer private; saving stopped")
            return info.sha
        except StorageError:
            raise
        except Exception:
            raise StorageError("Cannot read private storage revision") from None

    def read(self, name: str, revision: str) -> bytes | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            path = hf_hub_download(
                repo_id=self.repo,
                repo_type="dataset",
                filename=safe_name(name),
                revision=revision,
                token=self.token,
            )
            return Path(path).read_bytes()
        except EntryNotFoundError:
            return None
        except Exception:
            raise StorageError("Cannot download private checkpoint file") from None

    def commit(self, files: dict[str, bytes], parent: str) -> str:
        from huggingface_hub import CommitOperationAdd

        try:
            result = self.api.create_commit(
                repo_id=self.repo,
                repo_type="dataset",
                parent_commit=parent,
                commit_message="Save private kodoom translation progress",
                operations=[
                    CommitOperationAdd(path_in_repo=safe_name(name), path_or_fileobj=data)
                    for name, data in files.items()
                ],
            )
            return result.oid
        except Exception:
            raise StorageError(
                "Checkpoint commit failed; retry or check for another writer"
            ) from None


class DirectoryRemote:
    """Versioned local storage for CPU smoke tests; never a cloud persistence claim."""

    _lock = threading.RLock()

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def head(self) -> str:
        path = self.root / "HEAD"
        return path.read_text("utf-8").strip() if path.exists() else "empty"

    def read(self, name: str, revision: str) -> bytes | None:
        if revision == "empty":
            return None
        if not re.fullmatch(r"[0-9a-f]{64}", revision):
            raise ValueError("invalid local revision")
        path = self.root / "versions" / revision / safe_name(name)
        return path.read_bytes() if path.exists() else None

    def commit(self, files: dict[str, bytes], parent: str) -> str:
        with self._lock:
            if self.head() != parent:
                raise WriterConflict("storage changed while committing")
            combined = {}
            if parent != "empty":
                root = self.root / "versions" / parent
                combined = {
                    p.relative_to(root).as_posix(): p.read_bytes()
                    for p in root.rglob("*")
                    if p.is_file()
                }
            combined.update({safe_name(name): data for name, data in files.items()})
            revision = digest(
                encoded({name: digest(data) for name, data in sorted(combined.items())})
            )
            for name, data in combined.items():
                atomic_write(self.root / "versions" / revision / name, data)
            atomic_write(self.root / "HEAD", (revision + "\n").encode())
            return revision


class RemoteSession:
    def __init__(self, backend, campaign_id: str, attempt: str, operator: str):
        from kodoom.bulk.data import validate_campaign_id

        self.backend = backend
        self.prefix = "campaigns/" + validate_campaign_id(campaign_id) + "/"
        self.attempt, self.operator = attempt, operator
        self.verified_revision = None
        self.last_manifest = {}
        self.pending_manifest = None
        self.lock = threading.RLock()

    def _read_json(self, name: str, revision: str):
        data = self.backend.read(self.prefix + name, revision)
        return json.loads(data) if data is not None else None

    def _assert_owner(self, revision: str):
        lease = self._read_json("writer.json", revision)
        if not lease or lease.get("attempt") != self.attempt or lease.get("released") is not False:
            raise WriterConflict("campaign writer changed; stop this attempt")

    def acquire(self, *, takeover: bool = False):
        with self.lock:
            parent = self.backend.head()
            lease = self._read_json("writer.json", parent)
            if (
                lease
                and not lease.get("released")
                and lease.get("attempt") != self.attempt
                and not takeover
            ):
                raise WriterConflict(
                    "Previous writer is active; confirm it ended before explicit takeover"
                )
            value = {"attempt": self.attempt, "operator": self.operator, "released": False}
            revision = self.backend.commit({self.prefix + "writer.json": encoded(value)}, parent)
            self._assert_owner(revision)

    def restore(self, root: Path, *, read_only: bool = False) -> bool:
        with self.lock:
            revision = self.backend.head()
            if not read_only:
                self._assert_owner(revision)
            manifest = self._read_json("checkpoint.json", revision)
            if manifest is None:
                return False
            if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
                raise ValueError("unsupported remote checkpoint")
            downloaded = {}
            bundle = manifest.get("bundle", {})
            if not re.fullmatch(r"snapshots/[0-9a-f]{64}\.zip", bundle.get("path", "")):
                raise ValueError("invalid snapshot archive path")
            packed = self.backend.read(self.prefix + bundle["path"], revision)
            if packed is None or digest(packed) != bundle["sha256"]:
                raise ValueError("remote snapshot missing or checksum changed")
            with ZipFile(io.BytesIO(packed)) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != set(manifest["files"]):
                    raise ValueError("snapshot has duplicate, missing or unlisted artifacts")
                if sum(item.file_size for item in archive.infolist()) > 512 * 1024**2:
                    raise ValueError("snapshot exceeds the campaign storage budget")
                downloaded = {name: archive.read(name) for name in names}
            for name, sha in manifest["files"].items():
                safe_name(name)
                if name not in {
                    "campaign.json",
                    "inputs.json",
                    "progress.json",
                    "production.json",
                    "events.jsonl",
                    "tpu-metrics.json",
                } and not re.fullmatch(
                    r"(?:shards/[0-9a-f]{64}\.jsonl|review/[a-z-]+-[0-9a-f]{64}\.csv)", name
                ):
                    raise ValueError("remote checkpoint includes an unexpected artifact")
                data = downloaded[name]
                if data is None or digest(data) != sha:
                    raise ValueError("remote artifact missing or checksum changed")
                downloaded[name] = data
            if not {"campaign.json", "inputs.json", "progress.json"} <= downloaded.keys():
                raise ValueError("checkpoint missing campaign inputs or progress")
            # A local rerun may have more unsaved complete cases. Merge only
            # after proving the immutable identity and every overlapping shard.
            for name in ("campaign.json", "inputs.json", "production.json"):
                path = root / name
                if name in downloaded and path.exists() and path.read_bytes() != downloaded[name]:
                    raise ValueError("local and remote campaign identities differ")
            local_progress = root / "progress.json"
            if local_progress.exists():
                local = json.loads(local_progress.read_bytes())
                remote = json.loads(downloaded["progress.json"])
                for key, ref in remote["completed"].items():
                    if key in local["completed"] and local["completed"][key] != ref:
                        raise ValueError("local/remote completed cases conflict")
                remote["completed"].update(local["completed"])
                remote["failed"].update(local["failed"])
                for key in remote["completed"]:
                    remote["failed"].pop(key, None)
                downloaded["progress.json"] = encoded(remote)
                if (root / "events.jsonl").exists():
                    # Local attempt history is already a superset on a same-root rerun.
                    downloaded.pop("events.jsonl", None)
            for name, data in downloaded.items():
                path = root / name
                if (
                    name.startswith(("shards/", "review/"))
                    and path.exists()
                    and path.read_bytes() != data
                ):
                    raise ValueError("local/remote immutable artifacts conflict")
                atomic_write(path, data)
            self.last_manifest = manifest["files"]
            self.verified_revision = revision
            return True

    def save(self, snapshot: dict[str, bytes]) -> str:
        with self.lock:
            parent = self.backend.head()
            self._assert_owner(parent)
            current = self._read_json("checkpoint.json", parent)
            if current and self.last_manifest and current["files"] != self.last_manifest:
                if (
                    current.get("attempt") != self.attempt
                    or current["files"] != self.pending_manifest
                ):
                    raise WriterConflict("durable campaign changed since restore/save")
                # A commit may have succeeded while its verification request
                # failed. Retry verification of that exact sealed snapshot.
                bundle = current["bundle"]
                data = self.backend.read(self.prefix + bundle["path"], parent)
                if data is None or digest(data) != bundle["sha256"]:
                    raise StorageError("previous checkpoint verification still failed")
                self.last_manifest = current["files"]
            hashes = {name: digest(data) for name, data in snapshot.items()}
            if hashes == self.last_manifest and current:
                self.verified_revision = parent
                return parent
            stream = io.BytesIO()
            with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
                for name, data in snapshot.items():
                    archive.writestr(safe_name(name), data)
            packed = stream.getvalue()
            bundle_path = "snapshots/" + digest(packed) + ".zip"
            manifest = {
                "schema_version": 1,
                "files": hashes,
                "attempt": self.attempt,
                "operator": self.operator,
                "bundle": {"path": bundle_path, "sha256": digest(packed)},
            }
            files = {
                self.prefix + bundle_path: packed,
                self.prefix + "checkpoint.json": encoded(manifest),
            }
            self.pending_manifest = hashes
            revision = self.backend.commit(files, parent)
            # Verify exactly this revision, including newly uploaded contents.
            if self.backend.read(self.prefix + "checkpoint.json", revision) != encoded(manifest):
                raise StorageError("remote checkpoint verification failed")
            for name, data in files.items():
                if self.backend.read(name, revision) != data:
                    raise StorageError("remote file verification failed")
            self.last_manifest, self.verified_revision = hashes, revision
            return revision

    def paused(self) -> bool:
        control = self._read_json("control.json", self.backend.head())
        if control is None:
            return False
        if set(control) != {"pause"} or type(control["pause"]) is not bool:
            raise ValueError("campaign control must contain only a boolean pause")
        return control["pause"]

    def release(self):
        with self.lock:
            parent = self.backend.head()
            self._assert_owner(parent)
            lease = {"attempt": self.attempt, "operator": self.operator, "released": True}
            self.backend.commit({self.prefix + "writer.json": encoded(lease)}, parent)
