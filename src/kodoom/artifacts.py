"""Versioned, checksummed workflow bundles; extraction never trusts archive paths."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact bundles cannot contain symlinks")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": checksum(path),
            }
    return result


def export_bundle(root: Path, destination: Path) -> Path:
    """Copy only this stage's artifacts, not checkout/cache or other project data."""
    if destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("bundle destination must be outside the artifact root")
    destination.mkdir(parents=True, exist_ok=True)
    files = inventory(root)
    required = sum(item["size"] for item in files.values())
    if shutil.disk_usage(destination).free < required + 1024**3:
        raise ValueError("not enough space to export with a 1 GB reserve")
    output = destination / f"{root.name}-{uuid.uuid4().hex[:12]}.zip"
    temporary = output.with_suffix(".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in files:
                archive.write(root / name, "artifacts/" + name)
            archive.writestr("manifest.json", json.dumps({"version": 1, "files": files}))
        verify_bundle(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _manifest(archive: zipfile.ZipFile) -> dict:
    names = archive.namelist()
    if len(names) != len(set(names)) or "manifest.json" not in names:
        raise ValueError("bundle has duplicate entries or no manifest")
    if archive.getinfo("manifest.json").file_size > 8 * 1024**2:
        raise ValueError("bundle manifest is too large")
    manifest = json.loads(archive.read("manifest.json"))
    if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("unsupported bundle manifest")
    for name, item in manifest["files"].items():
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or path.as_posix() != name
        ):
            raise ValueError("unsafe artifact path")
        if (
            not isinstance(item, dict)
            or type(item.get("size")) is not int
            or item["size"] < 0
            or not isinstance(item.get("sha256"), str)
        ):
            raise ValueError("invalid artifact metadata")
    if set(names) != {"manifest.json", *("artifacts/" + n for n in manifest["files"])}:
        raise ValueError("bundle contains unlisted or missing files")
    return manifest


def verify_bundle(bundle: Path) -> dict:
    with zipfile.ZipFile(bundle) as archive:
        manifest = _manifest(archive)
        for name, item in manifest["files"].items():
            info = archive.getinfo("artifacts/" + name)
            if info.file_size != item["size"]:
                raise ValueError(f"artifact size mismatch: {name}")
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"artifact checksum mismatch: {name}")
    return manifest


def restore_bundle(bundle: Path, destination: Path) -> None:
    """Verify first, extract to staging, then rename; never overwrite live artifacts."""
    if destination.exists():
        raise ValueError("restore destination exists; use an empty stage directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = verify_bundle(bundle)
    size = sum(item["size"] for item in manifest["files"].values())
    if shutil.disk_usage(destination.parent).free < size + 1024**3:
        raise ValueError("not enough space to restore with a 1 GB reserve")
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        staged = Path(scratch) / "stage"
        staged.mkdir()
        with zipfile.ZipFile(bundle) as archive:
            for name in manifest["files"]:
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open("artifacts/" + name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        if inventory(staged) != manifest["files"]:
            raise ValueError("restored files do not match the manifest")
        staged.rename(destination)
