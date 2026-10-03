import os
from pathlib import Path

from kodoom import datadir
from kodoom.cli import main


def write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")  # byte counts must not depend on the OS


def touch_later(path, text):
    st = path.stat()
    path.write_text(text, encoding="utf-8", newline="\n")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


def test_snapshot_skips_hidden_and_temporary_files(tmp_path):
    write(tmp_path / "skills" / "a.jsonl")
    write(tmp_path / ".kodoom" / "history.jsonl")
    write(tmp_path / "skills" / ".hidden")
    write(tmp_path / "skills" / "b.jsonl.tmp")
    assert set(datadir.snapshot(tmp_path)) == {"skills/a.jsonl"}
    assert datadir.snapshot(tmp_path / "missing") == {}


def test_diff_names_new_updated_and_removed():
    before = {"a": (1, 1), "b": (1, 1), "c": (1, 1)}
    after = {"a": (1, 1), "b": (2, 5), "d": (1, 1)}
    changes = datadir.diff(before, after)
    assert (changes.new, changes.updated, changes.removed) == (["d"], ["b"], ["c"])
    assert changes.marks() == {"d": "new", "b": "updated"}
    assert not datadir.diff(before, before)


def test_readme_lists_files_and_history_and_only_changes_when_files_do(tmp_path):
    write(tmp_path / "skills" / "a.jsonl", '{"id": 1}\n{"id": 2}\n')
    changes = datadir.update_readme(tmp_path, "kodoom generate", now="T1")
    assert changes.new == ["skills/a.jsonl"]
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "a.jsonl  20 B, 2 records  [new]" in text
    assert "skills/  — code-labeled Persian skill data" in text
    assert "- T1 `kodoom generate`: 1 new" in text
    assert "README.md" not in text.split("```")[1]  # the README does not list itself

    assert datadir.update_readme(tmp_path, "kodoom generate", now="T2") is None
    assert "T2" not in (tmp_path / "README.md").read_text(encoding="utf-8")

    touch_later(tmp_path / "skills" / "a.jsonl", '{"id": 1}\n')
    write(tmp_path / "typed-decisions" / "fa" / "gemma" / "test.jsonl")
    changes = datadir.update_readme(tmp_path, "kodoom translate", now="T3")
    assert changes.updated == ["skills/a.jsonl"]
    assert changes.new == ["typed-decisions/fa/gemma/test.jsonl"]
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "gemma/  — one translator's output" in text
    assert text.index("- T3") < text.index("- T1")  # newest first


def test_no_readme_for_a_missing_or_empty_data_dir(tmp_path):
    assert datadir.update_readme(tmp_path / "missing", "kodoom generate") is None
    assert datadir.update_readme(tmp_path, "kodoom generate") is None
    assert not (tmp_path / "README.md").exists()


def test_changes_since_start(tmp_path):
    write(tmp_path / "old.jsonl")
    assert datadir.changes_since_start(tmp_path) is None
    assert datadir.save_start(tmp_path) == 1
    write(tmp_path / "new.jsonl")
    _, changes = datadir.changes_since_start(tmp_path)
    assert changes.new == ["new.jsonl"] and not changes.updated


def test_tree_lines_draws_folders_first():
    files = {"z.json": (5, 0), "skills/a.jsonl": (2048, 0), "skills/b.jsonl": (1, 0)}
    lines = datadir.tree_lines(Path("data"), files, {"skills/b.jsonl": "new"})
    assert lines == [
        "data/",
        "├── skills/  — code-labeled Persian skill data (`kodoom generate`)",
        "│   ├── a.jsonl  2.0 KB",
        "│   └── b.jsonl  1 B  [new]",
        "└── z.json  5 B",
    ]


def test_generate_updates_the_readme_and_tree_marks_this_runs_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # the dev profile's data_dir is the relative "data"
    assert main(["generate", "--profile", "dev", "jalali-dates"]) == 0
    assert main(["tree", "--profile", "dev", "--start"]) == 0
    capsys.readouterr()
    assert main(["generate", "--profile", "dev", "digit-forms"]) == 0
    out = capsys.readouterr().out
    assert "README.md: updated (2 new, 0 updated, 0 removed)" in out
    readme = (tmp_path / "data" / "README.md").read_text(encoding="utf-8")
    assert "`kodoom generate --profile dev digit-forms`: 2 new" in readme

    assert main(["tree", "--profile", "dev"]) == 0
    out = capsys.readouterr().out
    assert "digit-forms.jsonl" in out and "[new]" in out
    assert "jalali-dates.jsonl  " in out
    assert "  new      skills/digit-forms.jsonl" in out
    assert "  updated  README.md" in out
    assert "jalali-dates.jsonl" not in out.split("since")[1]


def test_tree_without_a_start_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["tree", "--profile", "dev"]) == 0
    assert "no start recorded" in capsys.readouterr().out
