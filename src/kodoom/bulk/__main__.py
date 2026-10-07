"""Child-process entry point; importing the notebook launcher never owns TPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run a private resumable translation campaign")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--repo", default="")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--provider", default="kaggle")
    parser.add_argument("--profile", default="kaggle-tpu")
    parser.add_argument("--operator", default="owner")
    parser.add_argument("--takeover", action="store_true")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--local-remote", type=Path)
    parser.add_argument("--max-units", type=int)
    args = parser.parse_args()
    from kodoom.bulk.worker import run_campaign

    result = run_campaign(
        root=args.root,
        campaign_id=args.campaign,
        repo=args.repo,
        provider=args.provider,
        profile=args.profile,
        operator=args.operator,
        takeover=args.takeover,
        deadline=args.deadline,
        dev=args.dev,
        request_path=args.request,
        local_remote=args.local_remote,
        max_units=args.max_units,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 1 if result["error_type"] or not result["persisted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
