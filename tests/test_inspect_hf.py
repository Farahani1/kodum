import json
from types import SimpleNamespace

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from kodoom.cli import main  # noqa: E402
from kodoom.inspect_hf import (  # noqa: E402
    InspectError,
    format_report,
    inspect_dataset,
    parse_json_cells,
    select_parquet,
)

FILES = [
    "README.md",
    "customer_service/test-00000-of-00001.parquet",
    "customer_service/train-00000-of-00001.parquet",
    "all/test-00000-of-00002.parquet",
    "all/test-00001-of-00002.parquet",
    "data/validation-00000-of-00001.parquet",
]


def test_select_parquet_by_split_and_config():
    assert select_parquet(FILES, "test", "customer_service") == [
        "customer_service/test-00000-of-00001.parquet"
    ]
    assert select_parquet(FILES, "test") == [  # no config asked: the first, alphabetically
        "all/test-00000-of-00002.parquet",
        "all/test-00001-of-00002.parquet",
    ]
    assert select_parquet(FILES, "validation") == ["data/validation-00000-of-00001.parquet"]


def test_select_parquet_errors_name_what_exists():
    with pytest.raises(InspectError, match=r"configs with it: .*customer_service"):
        select_parquet(FILES, "test", "nope")
    with pytest.raises(InspectError, match="no parquet file for split 'dev'"):
        select_parquet(FILES, "dev")


def test_json_strings_are_parsed_but_other_text_is_not():
    row = {
        "state": '{"body": "سلام", "n": 1}',
        "questions": '[{"a": 1}]',
        "text": "just words {not json}",
        "broken": "{oops",
        "n": 5,
    }
    parsed = parse_json_cells(row)
    assert parsed["state"] == {"body": "سلام", "n": 1} and parsed["questions"] == [{"a": 1}]
    assert parsed["text"] == "just words {not json}" and parsed["broken"] == "{oops"


def test_long_rows_are_cut_with_a_note():
    text = format_report(
        "r", "abc", None, [("f", 3)], None, "test", "x: string", 1, [{"x": "y" * 500}], 100
    )
    assert (
        "more characters cut" in text and "license : not stated" in text and "commit  : abc" in text
    )


@pytest.fixture
def fake_hub(tmp_path):
    rows = [
        {
            "id": i,
            "state": json.dumps({"body": f"متن {i}"}, ensure_ascii=False),
            "gold": '{"q": {"p": [1]}}',
        }
        for i in range(5)
    ]
    path = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    siblings = [SimpleNamespace(rfilename="cfg/test-00000-of-00001.parquet", size=123)]
    info = SimpleNamespace(sha="deadbeef", siblings=siblings, card_data={"license": "apache-2.0"})
    calls = []

    class Api:
        def dataset_info(self, repo, revision=None, files_metadata=False):
            calls.append(("info", repo, revision))
            return info

    def download(**kwargs):
        calls.append(("download", kwargs["revision"]))
        return str(path)

    return Api(), download, calls


def test_inspect_dataset_reports_commit_license_columns_and_rows(fake_hub):
    api, download, calls = fake_hub
    text = inspect_dataset("me/data", split="test", rows=2, api=api, download=download)
    assert "commit  : deadbeef" in text and "license : apache-2.0" in text
    assert "config  : cfg   split: test   rows: 5" in text
    assert "id: int64" in text and "state: string" in text
    assert '"body": "متن 0"' in text and '"body": "متن 1"' in text and "متن 2" not in text
    assert calls[1] == ("download", "deadbeef")  # files are read at the resolved commit, not main


def test_the_command_prints_and_saves_the_report(fake_hub, tmp_path, capsys, monkeypatch):
    api, download, _ = fake_hub
    import kodoom.inspect_hf as module

    original = module.inspect_dataset
    monkeypatch.setattr(
        "kodoom.cli.inspect_dataset", lambda *a, **k: original(*a, api=api, download=download, **k)
    )
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "report.txt"
    assert main(["inspect", "me/data", "--profile", "dev", "--out", str(out)]) == 0
    assert "commit  : deadbeef" in capsys.readouterr().out
    assert "rows: 5" in out.read_text(encoding="utf-8")


def test_a_missing_library_gives_the_colab_fix(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_hub(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("no hub", name="huggingface_hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_hub)
    with pytest.raises(InspectError, match="pip install huggingface_hub pyarrow"):
        inspect_dataset("me/data")


def test_a_dataset_without_parquet_lists_its_files(fake_hub):
    api, download, _ = fake_hub
    with pytest.raises(
        InspectError, match=r"no parquet file for split 'train'[\s\S]*cfg/test-00000"
    ):
        inspect_dataset("me/data", split="train", api=api, download=download)
