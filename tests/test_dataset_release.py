import json

import pytest

from kodoom.dataset_release import build_resources
from kodoom.package_import import file_sha, import_package
from kodoom.schema import RecordError, read_jsonl
from kodoom.transport import read_parquet
from tests.test_package_import import make_package


def source_package(root):
    make_package(root)
    for name in ("LICENSE", "NOTICE", "quality/finding.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unresolved supplied evidence\n", encoding="utf-8")
    path = root / "provenance/release.json"
    manifest = json.loads(path.read_text("utf-8"))
    for name in ("LICENSE", "NOTICE", "quality/finding.json"):
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
