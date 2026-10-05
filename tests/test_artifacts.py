import hashlib
import json
import zipfile

import pytest

from kodoom.artifacts import export_bundle, inventory, restore_bundle, verify_bundle


def test_export_restore_preserves_files_and_hidden_bookkeeping(tmp_path):
    root = tmp_path / "stage"
    (root / ".kodoom").mkdir(parents=True)
    (root / ".kodoom/history.json").write_text("{}", encoding="utf-8")
    (root / "output.txt").write_text("a result", encoding="utf-8")
    bundle = export_bundle(root, tmp_path / "bundles")
    restored = tmp_path / "restored"
    restore_bundle(bundle, restored)
    assert inventory(restored) == inventory(root)
    assert export_bundle(root, tmp_path / "bundles") != bundle


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/outside", "a\\b"])
def test_archive_paths_cannot_escape_the_restore_root(tmp_path, name):
    bundle = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "version": 1,
                    "files": {name: {"size": 1, "sha256": hashlib.sha256(b"x").hexdigest()}},
                }
            ),
        )
        archive.writestr("artifacts/" + name, "x")
    with pytest.raises(ValueError, match="unsafe artifact"):
        restore_bundle(bundle, tmp_path / "restore")
    assert not (tmp_path / "restore").exists()


def test_bad_checksums_and_unlisted_members_are_rejected(tmp_path):
    bundle = tmp_path / "bad.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps({"version": 1, "files": {"data.txt": {"size": 1, "sha256": "wrong"}}}),
        )
        archive.writestr("artifacts/data.txt", "x")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_bundle(bundle)
    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr("unexpected", "x")
    with pytest.raises(ValueError, match="unlisted"):
        verify_bundle(bundle)
