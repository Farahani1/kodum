"""The Colab notebook must stay thin and valid (plan: Colab stays thin)."""

import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab.ipynb"


def cells(kind):
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == kind]


def test_notebook_is_valid_and_its_python_parses():
    for src in cells("code"):
        # "!" shell lines are IPython syntax; the rest must be valid Python.
        python = "\n".join(line for line in src.splitlines() if not line.startswith("!"))
        ast.parse(python)


def test_every_kodoom_command_names_a_profile():
    commands = [
        line for src in cells("code") for line in src.splitlines() if line.startswith("!kodoom")
    ]
    assert commands
    assert all("--profile {PROFILE}" in line for line in commands)


def test_no_training_logic_in_the_notebook():
    code = "\n".join(cells("code"))
    for forbidden in ("import torch", "transformers", "Trainer", "def train"):
        assert forbidden not in code


def test_outputs_are_not_committed():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")


def test_data_tree_brackets_every_data_command():
    commands = [
        line for src in cells("code") for line in src.splitlines() if line.startswith("!kodoom")
    ]
    start = commands.index("!kodoom tree --profile {PROFILE} --start")
    writers = ("generate", "fetch", "translate ", "pilot-sheet", "export-units", "import-units")
    first_writer = min(i for i, c in enumerate(commands) if any(w in c for w in writers))
    assert start < first_writer
    assert commands[-1] == "!kodoom tree --profile {PROFILE}"
