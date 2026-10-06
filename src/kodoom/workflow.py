"""Current-stage execution, shared by Kaggle, Colab and explicit local profiles."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
import uuid
import zipfile
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from kodoom import datadir, helmo, typed_decisions
from kodoom.artifacts import checksum, export_bundle, inventory, restore_bundle, verify_bundle
from kodoom.check import FAIL, OK, Check, run_checks
from kodoom.config import Profile, load_profile
from kodoom.runs import git_commit
from kodoom.schema import read_jsonl
from kodoom.sources import check_record
from kodoom.translate.pipeline import case_items, cases, helmo_items, pick_cases, pick_helmo_records

ROOT = Path(__file__).resolve().parents[2]
REQUEST = ROOT / "workflows" / "current.toml"
CATALOG = ROOT / "workflows" / "recipes.toml"
MODELS = {
    "gemma3-4b-bf16": ("torch", "google/gemma-3-4b-it", "cuda"),
    "gemma3-27b-tpu-bf16": ("jax", "google/gemma-3/flax/gemma3-27b-it/1", "tpu"),
}


def load_request(path: Path = REQUEST) -> tuple[dict, list[dict]]:
    request = tomllib.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "stage",
        "plan",
        "stop",
        "recipes",
        "translator",
        "model",
        "model_revision",
        "batch_size",
        "helmo_limit",
        "typed_limit",
        "review",
    }
    if request.get("schema_version") == 2:
        expected |= {
            "backend",
            "precision",
            "input_tokens",
            "output_tokens",
            "cache_tokens",
            "max_seconds",
        }
    if set(request) != expected or request["schema_version"] not in (1, 2):
        raise ValueError("invalid current request schema")
    if (
        request["stage"] != "free-text-gate"
        or request["stop"] != "awaiting-review"
        or request["translator"] not in MODELS
        or request["recipes"] != ["helmo-gate", "typed-gate"]
    ):
        raise ValueError("this runner supports only the approved free-text gate")
    backend, model, _device = MODELS[request["translator"]]
    if request["model"] != model or request.get("backend", "torch") != backend:
        raise ValueError("model, translator and backend must agree")
    if request["schema_version"] == 2:
        if request["precision"] != "bf16":
            raise ValueError("the translation gate requires BF16")
        if (
            any(
                type(request[key]) is not int or request[key] < 1
                for key in ("input_tokens", "output_tokens", "cache_tokens", "max_seconds")
            )
            or request["input_tokens"] + request["output_tokens"] > request["cache_tokens"]
        ):
            raise ValueError("positive input/output limits must fit the cache budget")
    if backend == "jax" and (request["batch_size"] != 1 or request["model_revision"] != model):
        raise ValueError("TPU gate requires batch size 1 and the versioned Kaggle checkpoint")
    if (
        request["helmo_limit"] != 40
        or request["typed_limit"] != 20
        or type(request["batch_size"]) is not int
        or request["batch_size"] < 1
    ):
        raise ValueError("gate must use 40 helmo and 20 typed cases and a positive batch size")
    catalog = tomllib.loads((path.parent / "recipes.toml").read_text(encoding="utf-8"))
    rows = catalog["recipe"]
    lookup = {row["id"]: row for row in rows}
    if catalog["schema_version"] != 1 or len(lookup) != len(rows):
        raise ValueError("invalid recipe catalog")
    active = [lookup[name] for name in request["recipes"]]
    for recipe in active:
        if recipe["status"] != "active" or not all(isinstance(x, str) for x in recipe["args"]):
            raise ValueError("active recipes must contain command argument lists")
    return request, active


def write_profile(profile: Profile, path: Path) -> None:
    sections = {
        "run": {"seed": profile.seed, "runs_dir": str(profile.runs_dir.resolve())},
        "compute": {
            "device": profile.device,
            "precision": profile.precision,
            "backend": profile.backend,
        },
        "storage": {
            name: str(getattr(profile, name).resolve())
            for name in ("scratch_dir", "cache_dir", "data_dir")
        },
    }
    sections["storage"]["reserve_gb"] = profile.reserve_gb
    if profile.threads is not None:
        sections["compute"]["threads"] = profile.threads
    if profile.max_cases_per_source is not None:
        sections["data"] = {"max_cases_per_source": profile.max_cases_per_source}
    path.write_text(
        "\n".join(
            f"[{section}]\n"
            + "\n".join(f"{key} = {json.dumps(value)}" for key, value in values.items())
            for section, values in sections.items()
        )
        + "\n",
        encoding="utf-8",
    )


def command(args: list[str], profile: Path, log: Path, env: dict) -> int:
    words = [sys.executable, "-u", "-m", "kodoom.cli", *args, "--profile", str(profile)]
    print("kodoom " + " ".join(args), flush=True)
    with (
        log.open("a", encoding="utf-8") as output,
        subprocess.Popen(
            words,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as child,
    ):
        for line in child.stdout:
            for key in ("HF_TOKEN", "GITHUB_TOKEN", "KAGGLE_API_TOKEN", "KAGGLE_KEY"):
                if env.get(key):
                    line = line.replace(env[key], "[redacted]")
            print(line, end="", flush=True)
            output.write(line)
        return child.wait()


def model_preflight(request: dict, revision: str | None = None) -> dict:
    if request.get("backend") == "jax":
        from kodoom.tpu import probe_in_child
        from kodoom.translate.checkpoint import prepare_checkpoint

        if revision and revision != request["model_revision"]:
            raise ValueError("TPU model revision must match the immutable Kaggle version")
        info = probe_in_child()
        _directory, checkpoint = prepare_checkpoint()
        return {**info, "checkpoint": checkpoint, "model_revision": checkpoint["handle"]}
    import torch
    from huggingface_hub import HfApi, hf_hub_download

    if not torch.cuda.is_available():
        raise ValueError("GPU unavailable; select a GPU accelerator")
    # A native-support flag alone excludes emulated bf16 on T4. Probe the actual operation.
    probe = torch.ones((8, 8), device="cuda", dtype=torch.bfloat16)
    if not torch.isfinite(probe @ probe).all().item():
        raise ValueError("bf16 GPU probe failed; do not substitute fp16 for Gemma")
    try:
        info = HfApi().model_info(request["model"], revision=revision or request["model_revision"])
        hf_hub_download(request["model"], "config.json", revision=info.sha)
    except Exception as exc:
        raise ValueError("Gemma access failed: accept the model terms and enable HF_TOKEN") from exc
    return {
        "model_revision": info.sha,
        "device": torch.cuda.get_device_name(0),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "model_dtype": "bfloat16",
        "visible_devices": torch.cuda.device_count(),
    }


def selected_records(recipe: dict, root: Path, limit: int) -> list:
    records = list(read_jsonl(root / recipe["input"]))
    revision = helmo.REVISION if recipe["dataset"] == "helmo" else typed_decisions.REVISION
    for record in records:
        check_record(record)
        source = helmo.SOURCE if recipe["dataset"] == "helmo" else typed_decisions.SOURCE
        if (
            record.source_revision != revision
            or record.state_lang != "en"
            or record.question_lang != "en"
            or record.split != "train"
            or record.source != source
        ):
            raise ValueError(
                "input source revision/language differs from the pinned English source"
            )
    if recipe["dataset"] == "helmo":
        picked = pick_helmo_records(records, limit, True)
        if len(picked) != limit or {r.question_type for r in picked} != {"choice", "score", "noul"}:
            raise ValueError("helmo input cannot supply the requested balanced sample")
    else:
        groups = pick_cases(cases(records), limit, True)
        picked = [r for group in groups for r in group]
        if len(groups) != limit or {r.extra["workflow"] for r in picked} != set(
            typed_decisions.WORKFLOWS
        ):
            raise ValueError("typed input cannot supply the requested balanced workflows")
    if len({r.id for r in picked}) != len(picked):
        raise ValueError("source has duplicate selected IDs")
    return picked


def validate_output(expected: list, path: Path, translator: str) -> dict:
    rows = list(read_jsonl(path)) if path.exists() else []
    actual = {r.id: r for r in rows}
    if len(actual) != len(rows) or set(actual) != {r.id + ":fa" for r in expected}:
        raise ValueError(f"missing, duplicate or unexpected translated IDs: {path}")
    for en in expected:
        fa = actual[en.id + ":fa"]
        check_record(fa)
        if (
            fa.source_id,
            fa.source_revision,
            fa.split,
            fa.gold,
            fa.license,
            [o.id for o in fa.options],
            fa.extra.get("translator"),
        ) != (
            en.source_id,
            en.source_revision,
            en.split,
            en.gold,
            en.license,
            [o.id for o in en.options],
            translator,
        ):
            raise ValueError("translation changed provenance, labels or option IDs")
    return {
        "records": len(rows),
        "cases": len({r.source_id for r in rows}),
        "with_findings": sum(r.checks_passed is False for r in rows),
        "sha256": checksum(path),
    }


def review_sheet(root: Path, recipe: dict, selected: list, translator: str) -> None:
    review_path = root / f"review-{recipe['dataset']}.csv"
    if review_path.exists():
        return  # Preserve the reader's annotations when resuming the same semantic run.
    translated = {
        r.id: r for r in read_jsonl(root / recipe["output"].format(translator=translator))
    }
    groups = cases(selected)
    with review_path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "source_id",
                "english",
                "persian",
                "automatic_findings",
                "meaning_changed",
                "needs_edit",
                "notes",
            ]
        )
        for group in groups:
            fa = [translated[r.id + ":fa"] for r in group]

            def texts(rows):
                return {
                    "state": rows[0].state,
                    "questions": [
                        {
                            "text": r.question_text,
                            "options": [asdict(o) for o in r.options],
                            "gold": r.gold,
                        }
                        for r in rows
                    ],
                }

            writer.writerow(
                [
                    group[0].source_id,
                    json.dumps(texts(group), ensure_ascii=False),
                    json.dumps(texts(fa), ensure_ascii=False),
                    json.dumps([r.extra.get("check_findings", []) for r in fa], ensure_ascii=False),
                    "",
                    "",
                    "",
                ]
            )


def artifact_root(base: Profile, request: dict, mode: str, smoke: bool = False) -> Path:
    root = base.data_dir / "workflows" / request["stage"]
    if not smoke and request.get("backend") == "jax":
        root /= request["translator"]
    return root / ("smoke-" + mode if smoke else mode)


def check_provider_paths(provider: str, base: Profile) -> None:
    if provider == "kaggle" and not base.data_dir.as_posix().startswith("/kaggle/working/"):
        raise ValueError("Kaggle artifacts must be under /kaggle/working")


def restore_compatible(bundle: Path, root: Path, request: dict, mode: str, smoke: bool = False):
    manifest = verify_bundle(bundle)
    if "execution.json" not in manifest["files"]:
        raise ValueError("stage bundle has no execution identity")
    with zipfile.ZipFile(bundle) as archive:
        saved = json.loads(archive.read("artifacts/execution.json"))
    if (
        saved.get("request") != request
        or saved.get("mode") != mode
        or saved.get("semantic", {}).get("smoke", False) != smoke
    ):
        raise ValueError("bundle belongs to a different model/stage/mode; cannot restore")
    restore_bundle(bundle, root)


def prompt_manifest(recipe: dict, picked: list, target: Path) -> None:
    from kodoom.translate.glossary import load
    from kodoom.translate.hf import chat_messages

    glossary = load()
    groups = [[r] for r in picked] if recipe["dataset"] == "helmo" else cases(picked)
    items = [
        item
        for group in groups
        for item in (helmo_items(group[0]) if recipe["dataset"] == "helmo" else case_items(group))
    ]
    target.write_text(
        json.dumps([chat_messages(item, glossary) for item in items], ensure_ascii=False),
        encoding="utf-8",
    )


def restore_stage(provider: str, profile: str, mode: str, bundle: str | Path) -> Path:
    """Restore a saved stage bundle into its provider-owned, empty stage directory."""
    if provider not in ("kaggle", "colab", "generic") or mode not in ("preflight", "gate"):
        raise ValueError("select provider and preflight/gate mode explicitly")
    request, _ = load_request()
    base = load_profile(profile)
    check_provider_paths(provider, base)
    root = artifact_root(base, request, mode)
    restore_compatible(Path(bundle), root, request, mode)
    return root


def run_current(
    *,
    provider: str,
    profile: str,
    mode: str = "preflight",
    restore: str = "",
    request_path: Path = REQUEST,
    model_revision: str | None = None,
    smoke: bool = False,
    dry_run: bool = False,
) -> dict:
    if provider not in ("kaggle", "colab", "generic") or mode not in ("preflight", "gate"):
        raise ValueError("select provider and preflight/gate mode explicitly")
    request, recipes = load_request(request_path)
    base = load_profile(profile)
    if smoke and base.device != "cpu":
        raise ValueError("smoke mode requires a CPU profile")
    backend, _model, device = MODELS[request["translator"]]
    if not smoke and (base.device != device or base.backend != backend):
        raise ValueError(f"the selected translator requires a {backend}/{device} profile")
    if not smoke and backend == "jax" and provider != "kaggle":
        raise ValueError("the TPU checkpoint path requires provider=kaggle")
    check_provider_paths(provider, base)
    root = artifact_root(base, request, mode, smoke)
    limits = [3, 4] if mode == "preflight" else [request["helmo_limit"], request["typed_limit"]]
    if dry_run:
        return {
            "stage": request["stage"],
            "mode": mode,
            "provider": provider,
            "limits": dict(zip(request["recipes"], limits, strict=True)),
            "artifact_root": str(root),
            "stop": request["stop"],
        }
    if restore:
        restore_compatible(Path(restore), root, request, mode, smoke)
    root.mkdir(parents=True, exist_ok=True)
    resolved = replace(
        base,
        data_dir=root,
        cache_dir=base.cache_dir.resolve(),
        scratch_dir=base.scratch_dir.resolve(),
        runs_dir=base.runs_dir.resolve(),
    )
    if mode == "preflight":
        resolved = replace(resolved, max_cases_per_source=200)
    profile_path = root / "profile.toml"
    state_path = root / "execution.json"
    prior = json.loads(state_path.read_text("utf-8")) if state_path.exists() else None
    attempt = uuid.uuid4().hex[:12]
    state = {
        "stage": request["stage"],
        "mode": mode,
        "provider": provider,
        "attempt": attempt,
        "started": datetime.now(UTC).isoformat(),
        "status": "running",
        "persistence": "local-only; save output bundle and verify a fresh-session restore",
        "git_commit": git_commit(),
        "request": request,
        "commands": [],
        "inputs": {},
        "artifact_root": str(root),
    }
    env = dict(os.environ, HF_HOME=str(resolved.cache_dir))
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    os.environ["HF_HOME"] = str(resolved.cache_dir)
    if backend == "torch":
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif not smoke:
        env["JAX_PLATFORMS"] = os.environ["JAX_PLATFORMS"] = "tpu"
        env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    error = None
    try:
        # JAX in the notebook kernel would retain TPU devices after the probe.
        checks = (
            run_checks(resolved, probe_devices=False)
            if backend == "jax" and not smoke
            else run_checks(resolved)
        )
        state["checks"] = [asdict(c) for c in checks]
        if any(c.status == FAIL or (c.name == "device" and c.status != "ok") for c in checks):
            raise ValueError(
                "readiness failed: " + "; ".join(c.detail for c in checks if c.status != "ok")
            )
        if not smoke and (not state["git_commit"] or state["git_commit"].endswith("-dirty")):
            raise ValueError("real executions require a clean committed checkout")
        if smoke:
            state.update(model_revision="stub", model_dtype="fp32", device="cpu")
        else:
            state.update(
                model_preflight(request, model_revision or (prior or {}).get("model_revision"))
            )
            if backend == "jax":
                state["checks"].append(
                    asdict(Check("device", OK, f"{state['device']}, 8 chips, BF16"))
                )
        versions = {}
        for package in (
            "kodoom",
            "torch",
            "transformers",
            "accelerate",
            "huggingface_hub",
            "pyarrow",
            "jax",
            "jaxlib",
            "libtpu",
            "gemma",
            "kauldron",
            "flax",
            "orbax-checkpoint",
            "optax",
            "sentencepiece",
            "kagglehub",
            "numpy",
            "tensorflow",
        ):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "not-installed"
        state["versions"] = versions
        semantic = {
            "request": request,
            "mode": mode,
            "smoke": smoke,
            "seed": base.seed,
            "code": state["git_commit"],
            "model": state["model_revision"],
            "dtype": state["model_dtype"],
            "batch": request["batch_size"],
            "prompt": checksum(ROOT / "src/kodoom/translate/hf.py"),
            "glossary": checksum(ROOT / "src/kodoom/translate/glossary.toml"),
        }
        if backend == "jax" and not smoke:
            semantic.update(
                backend=backend,
                checkpoint=state["checkpoint"]["fingerprint"],
                versions=versions,
                generator=checksum(ROOT / "src/kodoom/translate/jax.py"),
                sharding="FSDPSharding / replicated batch-1 cache",
                allocator=env["XLA_PYTHON_CLIENT_MEM_FRACTION"],
            )
        state["semantic"] = semantic
        if prior and prior.get("semantic") and prior["semantic"] != semantic:
            raise ValueError("saved run has incompatible settings/code; use a new artifact root")
        if prior and not prior.get("semantic") and any(root.glob("*/fa/*/*.jsonl")):
            raise ValueError(
                "translated artifacts have no resume identity; use a new artifact root"
            )
        if backend == "jax" and not smoke and mode == "gate":
            preflight_path = root.parent / "preflight/execution.json"
            if not preflight_path.is_file():
                raise ValueError("Run and save the TPU preflight before the gate")
            preflight = json.loads(preflight_path.read_text("utf-8"))
            comparable = {k: v for k, v in semantic.items() if k != "mode"}
            prior_semantics = {
                k: v for k, v in preflight.get("semantic", {}).items() if k != "mode"
            }
            if preflight.get("status") != "preflight-passed" or comparable != prior_semantics:
                raise ValueError("TPU preflight failed or belongs to different model/code settings")
        write_profile(resolved, profile_path)
        env.update(
            KODOOM_MODEL_REVISION=state["model_revision"],
            KODOOM_TRANSLATION_BATCH_SIZE=str(request["batch_size"]),
        )
        if backend == "jax" and not smoke:
            env.update(
                KODOOM_CHECKPOINT_FINGERPRINT=state["checkpoint"]["fingerprint"],
                KODOOM_INPUT_TOKENS=str(request["input_tokens"]),
                KODOOM_OUTPUT_TOKENS=str(request["output_tokens"]),
                KODOOM_CACHE_TOKENS=str(request["cache_tokens"]),
                KODOOM_DEADLINE=str(time.monotonic() + request["max_seconds"]),
            )
        translator = "stub" if smoke else request["translator"]
        datadir.save_start(root)

        def invoke(args):
            code = command(args, profile_path, root / f"commands-{attempt}.log", env)
            state["commands"].append({"args": args, "returncode": code})
            return code

        for recipe, limit in zip(recipes, limits, strict=True):
            if backend == "jax" and not smoke:
                env["KODOOM_TPU_METRICS"] = str(root / f"tpu-{recipe['dataset']}-{attempt}.json")
            input_path = root / recipe["input"]
            if not input_path.exists():
                other_mode = "gate" if mode == "preflight" else "preflight"
                other_root = root.parent / ("smoke-" + other_mode if smoke else other_mode)
                prior_input = other_root / recipe["input"]
                if prior_input.exists():
                    # Reuse pinned English inputs, never another mode's translations.
                    selected_records(recipe, other_root, limit)
                    input_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(prior_input, input_path)
            if not input_path.exists():
                if smoke:
                    raise ValueError(
                        "CPU smoke requires fixture English files; no network or weights"
                    )
                if invoke(["fetch", recipe["dataset"]]):
                    raise ValueError("source fetch failed")
            picked = selected_records(recipe, root, limit)
            state["inputs"][recipe["dataset"]] = {
                "sha256": checksum(input_path),
                "ids": [r.id for r in picked],
            }
            if (
                prior
                and recipe["dataset"] in prior["inputs"]
                and prior["inputs"][recipe["dataset"]] != state["inputs"][recipe["dataset"]]
            ):
                raise ValueError("input checksum/selection changed; cannot resume")
            args = [arg.format(limit=limit, translator=translator) for arg in recipe["args"]]
            if backend == "jax" and not smoke:
                prompts = root / f"prompts-{recipe['dataset']}.json"
                prompt_manifest(recipe, picked, prompts)
                env["KODOOM_PROMPTS"] = str(prompts)
            code = invoke(args)
            if backend == "jax" and not smoke:
                metrics = Path(env["KODOOM_TPU_METRICS"])
                if metrics.exists():
                    state.setdefault("tpu_metrics", {})[recipe["dataset"]] = json.loads(
                        metrics.read_text("utf-8")
                    )
            output = root / recipe["output"].format(translator=translator)
            result = validate_output(picked, output, translator)
            # Code 1 is accepted only with all expected records and actual review findings.
            if code not in (0, 1) or (code == 1 and not result["with_findings"]):
                raise ValueError("translation command failed without a complete reviewable output")
            state.setdefault("outputs", {})[recipe["dataset"]] = result
            review_sheet(root, recipe, picked, translator)
            from kodoom.translate.gate_review import label_sheet

            label_sheet(root, recipe, picked, translator)
            args = [
                "translations",
                recipe["dataset"],
                "--translator",
                translator,
                "--show",
                str(limit),
            ]
            if recipe["dataset"] == "typed-decisions":
                args += ["--split", "train"]
            if invoke(args):
                raise ValueError("translation report failed")
        state["status"] = "preflight-passed" if mode == "preflight" else "awaiting-review"
        (root / "REVIEW.md").write_text(
            f"# {request['stage']} ({mode})\n\n{request['review']}\n\n"
            "Fill meaning_changed and needs_edit with yes/no in review-*.csv.\n"
            "Automatic checks do not measure meaning preservation. Do not start the full run.\n"
            "Smoke output is test data, never evidence that the scientific gate passed.\n",
            encoding="utf-8",
        )
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: execution failed; inspect command log and checks"
        error = exc
    finally:
        state["ended"] = datetime.now(UTC).isoformat()
        serialized = json.dumps(state, ensure_ascii=False, indent=2, default=str) + "\n"
        (root / f"attempt-{attempt}.json").write_text(serialized, encoding="utf-8")
        # A rejected rerun must not replace the resume contract of existing artifacts.
        if prior is None or error is None:
            state_path.write_text(serialized, encoding="utf-8")
        try:
            datadir.update_readme(root, f"workflow {request['stage']} {mode} attempt {attempt}")
            state["files"] = inventory(root)
            bundle = export_bundle(root, base.data_dir.parent / "bundles")
            state["bundle"] = str(bundle)
            print(f"Status: {state['status']}\nBundle ready for private saving: {bundle}")
        except Exception as final_error:
            print("Artifact finalization failed; local files remain at " + str(root))
            if error is None:
                error = final_error
    if error is not None:
        raise error
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=("kaggle", "colab", "generic"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mode", choices=("preflight", "gate"), default="preflight")
    parser.add_argument("--request", type=Path, default=REQUEST)
    parser.add_argument("--restore", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_current(
            provider=args.provider,
            profile=args.profile,
            mode=args.mode,
            restore=args.restore,
            smoke=args.smoke,
            dry_run=args.dry_run,
            request_path=args.request,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        print(
            f"workflow failed: {type(exc).__name__}; see the stage execution.json and logs",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {key: result[key] for key in ("stage", "mode", "status", "bundle") if key in result},
            indent=2,
        )
        if not args.dry_run
        else json.dumps(result, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
