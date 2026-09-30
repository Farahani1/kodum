"""Print the exact structure of a Hugging Face dataset: ``kodoom inspect``.

Run once on Colab (Hugging Face is not reachable from every environment) to learn
what a source dataset really contains before writing a loader for it: the pinned
commit, the license from its card, the files, the columns and types of one split, and
the first rows in full. Cells that hold a JSON string (typed-decisions stores its
state, questions and gold that way) are parsed and printed as JSON. Nothing here
changes any data.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

PARQUET = re.compile(r"^(?:(?P<config>[^/]+)/)?(?P<split>[A-Za-z0-9_]+)-\d+-of-\d+\.parquet$")


class InspectError(RuntimeError):
    """The dataset cannot be inspected as asked."""


def select_parquet(files: Iterable[str], split: str, config: str | None = None) -> list[str]:
    """The parquet files of ``split``, in ``config`` (or the first config that has the split)."""
    matches: dict[str | None, list[str]] = {}
    for name in sorted(files):
        m = PARQUET.match(name)
        if m and m["split"] == split:
            matches.setdefault(m["config"], []).append(name)
    if config is not None:
        if config not in matches:
            have = sorted(str(c) for c in matches) or "none"
            raise InspectError(
                f"no {split!r} parquet in config {config!r}; configs with it: {have}"
            )
        return matches[config]
    if not matches:
        raise InspectError(f"no parquet file for split {split!r}")
    return matches[sorted(matches, key=str)[0]]


def parse_json_cells(row: dict[str, Any]) -> dict[str, Any]:
    """Replace cells holding a JSON object or array (as text) by the parsed value."""
    parsed = {}
    for key, value in row.items():
        if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(value)
        parsed[key] = value
    return parsed


def format_report(
    repo: str,
    sha: str | None,
    license_: str | None,
    files: Sequence[tuple[str, int | None]],
    config: str | None,
    split: str,
    schema: str,
    total_rows: int,
    rows: Sequence[dict[str, Any]],
    max_chars: int = 6000,
) -> str:
    lines = [
        f"dataset : {repo}",
        f"commit  : {sha or 'unknown'}",
        f"license : {license_ or 'not stated in the card metadata'}",
        f"config  : {config or '(none)'}   split: {split}   rows: {total_rows}",
        "",
        "files:",
        *(f"  {size if size is not None else '?':>10}  {name}" for name, size in files),
        "",
        "columns:",
        *(f"  {line}" for line in schema.splitlines()),
    ]
    for n, row in enumerate(rows, start=1):
        text = json.dumps(parse_json_cells(row), ensure_ascii=False, indent=2, default=str)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [{len(text) - max_chars} more characters cut]"
        lines += ["", f"row {n} (JSON strings parsed):", text]
    return "\n".join(lines)


def inspect_dataset(
    repo: str,
    *,
    revision: str | None = None,
    config: str | None = None,
    split: str = "test",
    rows: int = 2,
    max_chars: int = 6000,
    api: Any = None,
    download: Callable[..., str] | None = None,
) -> str:
    """Fetch and describe ``repo``. ``api`` and ``download`` are replaceable for tests."""
    try:
        import pyarrow.parquet as pq

        if api is None or download is None:
            from huggingface_hub import HfApi, hf_hub_download

            api = api or HfApi()
            download = download or hf_hub_download
    except ImportError as e:
        raise InspectError(
            f"{e.name} is not installed; on Colab run: pip install huggingface_hub pyarrow"
        ) from e

    info = api.dataset_info(repo, revision=revision, files_metadata=True)
    files = [(s.rfilename, getattr(s, "size", None)) for s in info.siblings]
    card = getattr(info, "card_data", None)
    license_ = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
    if isinstance(license_, list):
        license_ = ", ".join(map(str, license_))

    names = [name for name, _ in files]
    try:
        chosen = select_parquet(names, split, config)
    except InspectError as e:  # show what is there, so the reader can pick a file by hand
        listing = "\n".join(f"  {n}" for n in names[:40])
        raise InspectError(f"{e}\nfiles in {repo} (first 40):\n{listing}") from e
    match = PARQUET.match(chosen[0])
    tables = []
    for name in chosen:
        path = download(repo_id=repo, filename=name, repo_type="dataset", revision=info.sha)
        tables.append(pq.read_table(Path(path)))
    table = tables[0] if len(tables) == 1 else _concat(tables)
    return format_report(
        repo,
        info.sha,
        license_,
        files,
        match["config"],
        split,
        str(table.schema),
        table.num_rows,
        table.slice(0, rows).to_pylist(),
        max_chars,
    )


def _concat(tables):
    import pyarrow as pa

    return pa.concat_tables(tables)
