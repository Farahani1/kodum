"""Explicit data preparation, smoke, training, prediction and restore commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kodoom.training.data import prepare
from kodoom.training.settings import read_settings


def parser():
    p = argparse.ArgumentParser(prog="python -m kodoom.training")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="freeze case-grouped model splits")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--selection", type=Path, default=Path("configs/dataset-baseline.toml"))
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--exclude-context", action="append", default=[])
    prep.add_argument("--seed", type=int, default=1234)
    for command in ("smoke", "preflight", "train"):
        train = sub.add_parser(command)
        train.add_argument("--prepared", type=Path, required=True)
        train.add_argument("--settings", type=Path, default=Path("configs/laya-full.toml"))
        train.add_argument("--profile", required=True)
        train.add_argument("--run-id", required=True)
        train.add_argument("--hub-repo")
        train.add_argument("--runs-dir", type=Path)
    restore = sub.add_parser("restore")
    restore.add_argument("--hub-repo", required=True)
    restore.add_argument("--revision", required=True)
    restore.add_argument("--run-id", required=True)
    restore.add_argument("--out", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "prepare":
        m = prepare(
            args.source,
            args.selection,
            args.out,
            seed=args.seed,
            exclude_contexts=tuple(args.exclude_context),
        )
        result = {k: m[k] for k in ("fingerprint", "counts", "exclude_contexts")}
    elif args.command == "restore":
        from kodoom.training.remote import restore

        result = {"restored": str(restore(args.hub_repo, args.revision, args.run_id, args.out))}
    else:
        from kodoom.training.worker import train

        result = train(
            args.prepared,
            read_settings(args.settings),
            profile_name=args.profile,
            run_id=args.run_id,
            hub_repo=args.hub_repo,
            smoke=args.command == "smoke",
            preflight=args.command == "preflight",
            runs_dir=args.runs_dir,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
