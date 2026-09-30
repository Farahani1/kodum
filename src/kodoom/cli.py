"""The ``kodoom`` command: the same entry point on the laptop and on Colab.

Commands that depend on the environment take ``--profile``; there is no
default, so a Colab run can never start with laptop settings or the reverse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from kodoom import __version__
from kodoom.check import FAIL, apply_environment, run_checks
from kodoom.config import BUILTIN_PROFILES, Profile, ProfileError, load_profile
from kodoom.generators import GENERATORS
from kodoom.generators.common import DEFAULT_PAIRS_PER_KIND, GeneratorError
from kodoom.runs import RunError, list_runs
from kodoom.schema import RecordError, read_jsonl, write_jsonl
from kodoom.sources import SourceError, check_record


def main(argv: list[str] | None = None) -> int:
    # Persian in output must not crash a Windows console using a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ProfileError, RecordError, SourceError, RunError, GeneratorError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kodoom", description=__doc__)
    parser.add_argument("--version", action="version", version=f"kodoom {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="show the resolved settings of a profile")
    _add_profile_args(info)
    info.set_defaults(func=_info)

    check = commands.add_parser(
        "check", help="check that this machine is ready for a profile (run first on Colab)"
    )
    _add_profile_args(check)
    check.set_defaults(func=_check)

    runs = commands.add_parser("runs", help="list runs under runs_dir and what can be resumed")
    _add_profile_args(runs)
    runs.set_defaults(func=_runs)

    generate = commands.add_parser(
        "generate",
        help="generate the code-labeled Persian skill data (plan 1.1); needs no GPU or download",
    )
    generate.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help=f"generators to run (default: all): {', '.join(GENERATORS)}",
    )
    _add_profile_args(generate)
    generate.add_argument("--out", type=Path, default=Path("data/skills"), help="output directory")
    generate.add_argument("--seed", type=int, help="random seed (default: the profile's)")
    generate.add_argument(
        "--pairs-per-kind",
        type=int,
        help=f"minimal pairs per question kind (default {DEFAULT_PAIRS_PER_KIND}, or the "
        "profile's max_cases_per_source if that is smaller)",
    )
    generate.set_defaults(func=_generate)

    validate = commands.add_parser(
        "validate", help="check record files against the schema and source rules"
    )
    validate.add_argument("files", nargs="+", type=Path, metavar="FILE.jsonl")
    validate.set_defaults(func=_validate)
    return parser


def _add_profile_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--profile",
        required=True,
        help=f"one of {', '.join(BUILTIN_PROFILES)}, or a path to a profile TOML file",
    )
    p.add_argument("--runs-dir", type=Path, help="override the profile's output directory")


def _load(args: argparse.Namespace) -> Profile:
    profile = load_profile(args.profile, runs_dir=args.runs_dir)
    apply_environment(profile)
    return profile


def _info(args: argparse.Namespace) -> int:
    profile = _load(args)
    print(f"kodoom {__version__}")
    for key, value in vars(profile).items():
        print(f"{key:>22}: {value}")
    return 0


def _generate(args: argparse.Namespace) -> int:
    profile = _load(args)
    unknown = [n for n in args.names if n not in GENERATORS]
    if unknown:
        raise GeneratorError(
            f"unknown generator {unknown[0]!r}; choose from {', '.join(GENERATORS)}"
        )
    seed = profile.seed if args.seed is None else args.seed
    pairs = args.pairs_per_kind or DEFAULT_PAIRS_PER_KIND
    if args.pairs_per_kind is None and profile.max_cases_per_source is not None:
        pairs = min(pairs, profile.max_cases_per_source)

    for name in args.names or list(GENERATORS):
        generator = GENERATORS[name]
        records = generator.generate(seed, pairs)
        for record in records:
            check_record(record)
        path = args.out / f"{name}.jsonl"
        write_jsonl(path, records)
        manifest = {
            "generator": name,
            "generator_version": generator.VERSION,
            "kodoom_version": __version__,
            "seed": seed,
            "pairs_per_kind": pairs,
            "records": len(records),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "by_split": dict(Counter(r.split for r in records)),
            "by_kind": dict(Counter(r.extra["kind"] for r in records)),
        }
        (args.out / f"{name}.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{path}: {len(records)} records (seed {seed}, {pairs} pairs per kind)")
        print(f"  splits: {manifest['by_split']}")
        print(f"  sha256: {manifest['sha256']}")
    return 0


def _check(args: argparse.Namespace) -> int:
    profile = _load(args)
    print(f"profile: {profile.name}")
    checks = run_checks(profile)
    for c in checks:
        print(f"  [{c.status:>4}] {c.name}: {c.detail}")
    failed = [c for c in checks if c.status == FAIL]
    print("not ready: fix the FAIL lines above" if failed else "ready")
    return 1 if failed else 0


def _runs(args: argparse.Namespace) -> int:
    profile = _load(args)
    rows = list_runs(profile.runs_dir)
    if not rows:
        print(f"no runs under {profile.runs_dir}")
        return 0
    print(f"{'run':<24} {'status':<9} {'latest':>7} {'best':>7} {'metric':>9} {'size':>9}  updated")
    for r in rows:
        metric = "" if r["best_metric"] is None else f"{r['best_metric']:.4f}"
        print(
            f"{r['run_id']:<24} {r['status']:<9} {r['latest_step'] or '':>7} "
            f"{r['best_step'] or '':>7} {metric:>9} {r['size_bytes'] / 1024**3:>7.2f}GB  "
            f"{r['updated']}"
        )
    return 0


def _validate(args: argparse.Namespace) -> int:
    for path in args.files:
        counts: Counter[tuple[str, str, str]] = Counter()
        for record in read_jsonl(path):
            check_record(record)
            counts[record.source, record.split, record.origin] += 1
        print(f"{path}: {sum(counts.values())} records OK")
        for (source, split, origin), n in sorted(counts.items()):
            print(f"  {n:>7}  {source}  {split}  {origin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
