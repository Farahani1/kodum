import json
from pathlib import Path

import pytest

from kodoom.dataset_release import build_resources, translation_grant
from kodoom.package_import import file_sha, import_package
from kodoom.schema import RecordError, read_jsonl
from kodoom.transport import read_parquet
from tests.test_package_import import make_package


def source_package(root):
    make_package(root)
    names = ("LICENSE", "NOTICE", "quality/finding.json", "source_snapshots/upstream/LICENSE")
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unresolved supplied evidence\n", encoding="utf-8")
    path = root / "provenance/release.json"
    manifest = json.loads(path.read_text("utf-8"))
    for name in names:
        p = root / name
        manifest["files"].append({"path": name, "bytes": p.stat().st_size, "sha256": file_sha(p)})
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return file_sha(path)


def test_export_preserves_source_evidence_and_prepares_gold_blind_check_requests(tmp_path):
    source, imported, out = (tmp_path / name for name in ("source", "import", "release"))
    pin = source_package(source)
    import_package(source, imported, pin)
    report = build_resources(source, imported, out)
    assert report["resources"]["typed_decisions"]["paired_decisions"] == 1
    assert not report["publication_authorized"]
    root = out / "typed_decisions"
    assert list(read_jsonl(root / "jsonl/fa/train.jsonl")) == list(
        read_parquet(root / "data/fa/train.parquet")
    )
    assert (root / "provenance/inputs/source.parquet").exists()
    assert (root / "source_snapshots/upstream/LICENSE").read_bytes() == (
        source / "source_snapshots/upstream/LICENSE"
    ).read_bytes()
    assert (root / "provenance/supplied/quality/finding.json").read_text("utf-8") == (
        source / "quality/finding.json"
    ).read_text("utf-8")
    request = json.loads((root / "review/meaning-check-requests.jsonl").read_text("utf-8"))
    assert request["questions"][0]["id"] == "cs-1:route"
    assert "gold" not in json.dumps(request)
    assert (
        json.loads((root / "review/status.json").read_text("utf-8"))["meaning_check"] == "not_run"
    )
    for line in (root / "SHA256SUMS").read_text("utf-8").splitlines():
        digest, name = line.split("  ", 1)
        assert file_sha(root / name) == digest


def test_export_refuses_changed_import(tmp_path):
    source, imported = tmp_path / "source", tmp_path / "import"
    pin = source_package(source)
    import_package(source, imported, pin)
    path = imported / "jsonl/typed_decisions/fa/train.jsonl"
    path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RecordError, match="checksum mismatch"):
        build_resources(source, imported, tmp_path / "release")


def add_translation_grant(source):
    repo = Path(__file__).resolve().parents[1]
    files = {
        "provenance/translation_license.json": repo / "docs/translation-license-grant.json",
        "licenses/Apache-2.0.txt": repo / "licenses/Apache-2.0.txt",
        "TRANSLATION-LICENSE.md": repo / "TRANSLATION-LICENSE.md",
    }
    path = source / "provenance/release.json"
    release = json.loads(path.read_text("utf-8"))
    for name, original in files.items():
        dest = source / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(original.read_bytes())
        release["files"].append(
            {"path": name, "bytes": dest.stat().st_size, "sha256": file_sha(dest)}
        )
    path.write_text(json.dumps(release), encoding="utf-8")
    return file_sha(path), release


def test_export_preserves_explicit_contribution_grant_and_training_gate(tmp_path):
    source, imported, out = (tmp_path / name for name in ("source", "import", "release"))
    source_package(source)
    pin, _ = add_translation_grant(source)
    import_package(source, imported, pin)
    result = build_resources(source, imported, out)
    root = out / "typed_decisions"
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    assert manifest["translation_license_grant"]["license"] == "Apache-2.0"
    assert result["training_eligibility"] == manifest["training_eligibility"] == "unresolved"
    assert (root / "licenses/Apache-2.0.txt").read_bytes() == (
        source / "licenses/Apache-2.0.txt"
    ).read_bytes()
    assert (root / "provenance/translation_license.json").read_bytes() == (
        source / "provenance/translation_license.json"
    ).read_bytes()
    card = (root / "README.md").read_text("utf-8")
    assert "owner granted Apache-2.0" in card
    assert "grant has not been supplied" not in card
    assert "license: other" in card  # combined resource still preserves component terms
    assert "licenses/*.txt text eol=lf" in (root / ".gitattributes").read_text("utf-8")


def test_grant_must_be_inventoried_and_match_license_text(tmp_path):
    root = tmp_path / "source"
    source_package(root)
    _, release = add_translation_grant(root)
    assert translation_grant(root, release)["status"] == "granted"
    missing = {
        **release,
        "files": [
            f for f in release["files"] if f["path"] != "provenance/translation_license.json"
        ],
    }
    with pytest.raises(RecordError, match="not inventoried"):
        translation_grant(root, missing)
    (root / "licenses/Apache-2.0.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(RecordError, match="differs"):
        translation_grant(root, release)
