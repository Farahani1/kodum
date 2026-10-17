"""Consume an exact external release without running dataset quality checks."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import replace
from pathlib import Path

from kodoom.package_import import CONFIG_SOURCES, checked_path, row_records, verify_inventory
from kodoom.schema import Record, RecordError
from kodoom.sources import check_record, get_source
from kodoom.typed_decisions import read_parquet_rows

TRANSPORT = "hf-data-bilingual-parquet-v1"
KEYS = {
    "dataset",
    "release_version",
    "schema_version",
    "transport",
    "manifest_sha256",
    "configurations",
    "languages",
    "splits",
}


def read_selection(path: Path) -> dict:
    with path.open("rb") as stream:
        selection = tomllib.load(stream)
    validate_selection(selection)
    return selection


def validate_selection(selection: dict) -> None:
    if set(selection) != KEYS or selection.get("dataset") != "hf-data":
        raise RecordError("dataset selection requires the exact hf-data contract keys")
    if selection["schema_version"] != "1.0.0" or selection["transport"] != TRANSPORT:
        raise RecordError("unsupported dataset schema/transport; migrate the consumer adapter")
    version = selection["release_version"]
    if (
        not isinstance(version, str)
        or len(version.split(".")) != 3
        or not all(part.isdigit() for part in version.split("."))
    ):
        raise RecordError("pin an exact dataset release version (MAJOR.MINOR.PATCH)")
    pin = selection["manifest_sha256"]
    if not isinstance(pin, str) or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
        raise RecordError("pin a lowercase SHA-256 release manifest hash")
    for key, allowed in (
        ("configurations", CONFIG_SOURCES),
        ("languages", ("en", "fa")),
        ("splits", ("train", "test")),
    ):
        values = selection[key]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(v, str) or v not in allowed for v in values)
            or len(set(values)) != len(values)
        ):
            raise RecordError(
                f"select unique supported {key} explicitly; available: {list(allowed)}"
            )


def consume_release(root: Path, selection: dict) -> tuple[list[Record], dict]:
    """Return raw Records and a portable lock; verify all inventoried bytes first.

    Only selected bilingual tables are read. Original aggregate/workflow views
    are deliberately unsupported because they overlap those same decisions.
    The local source root is recorded separately from the lock fingerprint.
    """
    validate_selection(selection)
    release = verify_inventory(root, selection["manifest_sha256"])
    for key in ("release_version", "schema_version"):
        if release.get(key) != selection[key]:
            raise RecordError(f"dataset {key} differs from exact selection")
    inventory = {entry["path"]: entry for entry in release["files"]}
    tables = {}
    for table in release["configurations"]:
        key = (table["config"], table["split"])
        if key in tables:
            raise RecordError(f"duplicate dataset table: {key}")
        tables[key] = table
    records, payloads = [], []
    seen, memberships = set(), {}
    for config in sorted(selection["configurations"]):
        for split in sorted(selection["splits"]):
            key = (config, split)
            if key not in tables:
                raise RecordError(f"selected resource unavailable in release: {config}/{split}")
            table = tables[key]
            if table["path"] not in inventory:
                raise RecordError("selected dataset table is not inventoried")
            count = 0
            for row in read_parquet_rows(checked_path(root, table["path"])):
                en, fa = row_records(row, preserve_text=True)
                if en.source != CONFIG_SOURCES[config] or en.split != split:
                    raise RecordError("dataset table source/split differs from row")
                decision = (en.source, en.source_revision, en.id)
                if decision in seen:
                    raise RecordError(f"duplicate decision: {decision}")
                seen.add(decision)
                case = (en.source, en.source_revision, en.source_id)
                if memberships.setdefault(case, split) != split:
                    raise RecordError(f"case crosses splits: {case}")
                for record in (en, fa):
                    check_record(record)
                    if record.question_lang in selection["languages"]:
                        # Retain supplied declarations, including unresolved historical
                        # eligibility, rather than turning acceptance into an audit.
                        extra = dict(record.extra)
                        extra.pop("training_eligibility", None)
                        extra["dataset_release"] = {
                            "manifest_sha256": selection["manifest_sha256"],
                            "release_version": selection["release_version"],
                            "config": config,
                            "source_role": get_source(record.source).role,
                        }
                        records.append(replace(record, extra=extra))
                count += 1
            if count != table["rows"]:
                raise RecordError(f"dataset row count differs from manifest: {key}")
            payloads.append({**inventory[table["path"]], "config": config, "split": split})
    identity = {
        **selection,
        "configurations": sorted(selection["configurations"]),
        "languages": sorted(selection["languages"]),
        "splits": sorted(selection["splits"]),
        "adapter_version": 1,
        "payloads": payloads,
        "records": len(records),
        "source_roles": {
            config: get_source(CONFIG_SOURCES[config]).role
            for config in sorted(selection["configurations"])
        },
        "translation_license_grant": release.get("translation_license_grant"),
        "consumption_acceptance": "project-plan-v31-owner-decision-2026-10-16",
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return records, {**identity, "fingerprint": fingerprint, "source_root": str(root.resolve())}
