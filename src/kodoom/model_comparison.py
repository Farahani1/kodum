"""Reproducible three-model comparisons on native and translated Persian samples.

Selection happens before inference, by whole source case. Choice conversion is
shared by all models because X1 has no native noul/score heads. No test-set
calibration, checkpoint selection, dataset editing or publication happens here.
"""

from __future__ import annotations

import csv
import gc
import importlib.util
import itertools
import json
import math
import platform
import re
import time
from collections import defaultdict
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from kodoom.dataset_lock import consume_release, read_selection
from kodoom.evaluation import evaluate
from kodoom.external_eval import (
    ExternalEvalError,
    load_parsinlu_task,
    load_persianqa_repository,
    predict_diba_x1,
    predict_laya,
)
from kodoom.metrics import paired_difference_ci
from kodoom.package_import import checked_path, file_sha
from kodoom.predictions import Prediction, write_predictions
from kodoom.schema import Record
from kodoom.training.data import case_key, context, digest, model_record

MODEL_NAMES = ("laya-multilingual", "kodoom", "dibaone-x1")
PROTOCOL = "shared-choice-persian-v2"


def choice_record(record: Record) -> Record:
    """Keep IDs, gold, option order and source split; declare converted heads."""
    protocol = record.extra.get("protocol", "native-choice")
    if record.question_type != "choice":
        protocol = f"{record.question_type}-to-choice-for-all-models"
    return replace(
        record,
        question_type="choice",
        extra={
            **record.extra,
            "original_question_type": record.question_type,
            "protocol": protocol,
        },
    )


def select_cases(records, *, cases_per_group: int, seed: int, translated: bool = False):
    """Stable hash sampling independent of row order; retain every case question.

    Native groups retain sentiment domains. Translated groups retain workflow
    or topic and split; no source train split can be relabelled as held-out.
    """
    if type(cases_per_group) is not int or cases_per_group < 1:
        raise ExternalEvalError("cases_per_group must be a positive integer")
    cases, seen = defaultdict(list), set()
    for record in records:
        if record.id in seen:
            raise ExternalEvalError(f"duplicate evaluation ID: {record.id}")
        seen.add(record.id)
        group = (
            [record.source, record.split, context(record)]
            if translated
            else [record.task_family, record.extra.get("source_file", "")]
        )
        cases[(json.dumps(group, ensure_ascii=False), case_key(record))].append(record)
    groups = defaultdict(list)
    for (group, key), members in cases.items():
        groups[group].append((key, members))
    selected, summary = [], {}
    for group, entries in sorted(groups.items()):
        ranked = sorted(entries, key=lambda entry: (digest([seed, group, entry[0]]), entry[0]))
        chosen = ranked[:cases_per_group]
        selected.extend(
            record for _, members in chosen for record in sorted(members, key=lambda r: r.id)
        )
        summary[group] = {
            "available_cases": len(entries),
            "selected_cases": len(chosen),
            "selected_questions": sum(len(members) for _, members in chosen),
        }
    if not selected:
        raise ExternalEvalError("sample selection produced no records")
    selected = sorted(selected, key=lambda record: record.id)
    return selected, {
        "seed": seed,
        "cases_per_group": cases_per_group,
        "unit": "whole-source-case",
        "groups": summary,
        "ids": [record.id for record in selected],
        "records_sha256": digest([record.to_dict() for record in selected]),
    }


def download_release(selection_path: Path, destination: Path, *, repo: str, revision: str):
    """Materialize an exact accepted release so containment/hash checks work."""
    from huggingface_hub import snapshot_download

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ExternalEvalError("translated dataset revision must be an immutable commit SHA")
    selection = read_selection(selection_path)
    snapshot_download(
        repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["provenance/release.json"],
        local_dir=str(destination),
    )
    manifest = destination / "provenance/release.json"
    if file_sha(manifest) != selection["manifest_sha256"]:
        raise ExternalEvalError("translated dataset manifest differs from the accepted release")
    release = json.loads(manifest.read_text("utf-8"))
    paths = [entry["path"] for entry in release["files"]]
    for relative in paths:
        checked_path(destination, relative)
    snapshot_download(
        repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["provenance/release.json", *paths],
        local_dir=str(destination),
    )
    return selection


def translated_benchmarks(root: Path, selection: dict, *, include_helmo: bool):
    """Consume released Persian text without cleaning/relabeling the dataset."""
    records, lock = consume_release(
        root,
        {
            **selection,
            "configurations": ["typed_decisions"],
            "languages": ["fa"],
            "splits": ["test"],
        },
    )
    result = {"translated-typed-decisions-test": records}
    locks = {"translated-typed-decisions-test": lock}
    if include_helmo:
        records, lock = consume_release(
            root,
            {**selection, "configurations": ["helmo"], "languages": ["fa"], "splits": ["train"]},
        )
        # Helmo has no held-out split in this release. Keep it as a diagnostic.
        result["translated-helmo-train-diagnostic"] = records
        locks["translated-helmo-train-diagnostic"] = lock
    return result, locks


def prepare_benchmarks(
    *,
    work: Path,
    selection_path: Path,
    parsinlu_revision: str,
    dataset_repo: str,
    dataset_revision: str,
    native_cases: int = 100,
    translated_cases: int = 3,
    seed: int = 17,
    include_helmo: bool = True,
    persianqa_revision: str | None = None,
):
    """Resolve inputs and select once, before downloading any model weights."""
    raw, sources = {}, {}
    for task in ("entailment", "paraphrase", "sentiment", "multiple-choice-qa"):
        name = f"parsinlu-{task}"
        records, revision, split = load_parsinlu_task(
            "persiannlp/parsinlu", task, revision=parsinlu_revision
        )
        raw[name] = records
        sources[name] = {
            "revision": revision,
            "split": split,
            "files": sorted({r.extra["source_file"] for r in records}),
            "excluded_unlabelled_rows": records[0].extra["excluded_unlabelled_rows"],
        }
    records, revision, splits = load_persianqa_repository(
        "https://github.com/sajjjadayobi/PersianQA.git",
        work / "PersianQA",
        revision=persianqa_revision,
    )
    raw["persianqa"] = records
    sources["persianqa"] = {"revision": revision, "splits": splits}
    root = work / "translated-release"
    selection = download_release(selection_path, root, repo=dataset_repo, revision=dataset_revision)
    translated, locks = translated_benchmarks(root, selection, include_helmo=include_helmo)
    raw.update(translated)
    sources.update(
        {
            name: {"repo": dataset_repo, "revision": dataset_revision, "lock": lock}
            for name, lock in locks.items()
        }
    )
    selected, samples = {}, {}
    for name, records in raw.items():
        is_translated = name.startswith("translated-")
        records, sample = select_cases(
            records,
            cases_per_group=(translated_cases if is_translated else native_cases),
            seed=seed,
            translated=is_translated,
        )
        selected[name] = [choice_record(record) for record in records]
        samples[name] = {
            **sample,
            "source": sources[name],
            "interpretation": (
                "train-only-source-diagnostic; not-held-out" if "diagnostic" in name else "held-out"
            ),
            "choice_records_sha256": digest([r.to_dict() for r in selected[name]]),
        }
        print(
            name,
            len(records),
            "questions",
            sum(g["selected_cases"] for g in sample["groups"].values()),
            "cases",
            flush=True,
        )
    return selected, samples


def load_kodoom(work: Path, *, repo: str, revision: str, run_id: str):
    from huggingface_hub import snapshot_download

    from kodoom.training import laya

    root = work / "kodoom-snapshot"
    snapshot_download(
        repo,
        revision=revision,
        local_dir=str(root),
        allow_patterns=[f"{run_id}/durable-manifest.json", f"{run_id}/best/**"],
    )
    snapshot = checked_path(root, run_id)
    manifest = json.loads((snapshot / "durable-manifest.json").read_text("utf-8"))
    entries = {rel: entry for rel, entry in manifest["files"].items() if rel.startswith("best/")}
    required = {"best/model.safetensors", "best/encoder/config.json", "best/rl_agent_config.json"}
    if not required.issubset(entries) or not any(
        rel.startswith("best/tokenizer/") for rel in entries
    ):
        raise ExternalEvalError("best checkpoint manifest omits required inference files")
    for relative, entry in entries.items():
        path = checked_path(snapshot, relative)
        if file_sha(path) != entry["sha256"]:
            raise ExternalEvalError(f"best-checkpoint hash mismatch: {relative}")
    return laya.load(work / "cache", artifact=snapshot / "best")


def load_diba(work: Path, revision: str, device: str):
    from huggingface_hub import snapshot_download

    root = work / "diba-x1-snapshot" / revision
    snapshot_download(
        "Dibachain/DibaOne-X1",
        revision=revision,
        local_dir=str(root),
        allow_patterns=["dibaone_x1.py", "*.json", "*.safetensors", "stage1/**"],
    )
    spec = importlib.util.spec_from_file_location("kodoom_pinned_diba_x1", root / "dibaone_x1.py")
    module = importlib.util.module_from_spec(spec)
    # dataclass implementations inspect their defining module during import.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DibaOneX1.from_pretrained(str(root), backend="torch", device=device)


def _clean_json(value):
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(_clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def comparison_report(records, predictions, *, seed=17, n_boot=2000):
    """Report all coverage plus pairwise and three-way common-item comparisons."""
    models = {}
    for name, mapping in predictions.items():
        evaluation = evaluate(records, mapping)
        models[name] = {
            **evaluation.summary(n_boot=n_boot, seed=seed).to_dict(),
            "coverage": len(evaluation.items) / len(records),
            "failed": len(evaluation.failed),
            "missing": len(evaluation.missing),
        }
    common = [
        r
        for r in records
        if all(r.id in mapping and not mapping[r.id].error for mapping in predictions.values())
    ]
    common_models = {
        name: evaluate(common, mapping).summary(n_boot=n_boot, seed=seed).to_dict()
        for name, mapping in predictions.items()
    }
    pairs = {}
    for first, second in itertools.combinations(predictions, 2):
        shared = [
            r
            for r in records
            if all(
                r.id in predictions[name] and not predictions[name][r.id].error
                for name in (first, second)
            )
        ]
        scores = [
            {
                item.record.id: float(item.metrics.correct)
                for item in evaluate(shared, predictions[name]).items
            }
            for name in (first, second)
        ]
        pairs[f"{first}-minus-{second}"] = {
            "n": len(shared),
            "accuracy_difference_ci95": paired_difference_ci(
                *[[score[r.id] for r in shared] for score in scores],
                [case_key(r) for r in shared],
                n_boot=n_boot,
                seed=seed,
            ),
        }
    return _clean_json(
        {
            "n_records": len(records),
            "models": models,
            "common_items": {"n": len(common), "models": common_models},
            "pairwise": pairs,
            "protocols": sorted({r.extra["protocol"] for r in records}),
        }
    )


def write_comparison_csv(output: Path, benchmarks, predictions):
    """One row per request: gold, each model's decision/probabilities and error."""
    names = list(predictions)
    fields = [
        "benchmark",
        "id",
        "source_id",
        "split",
        "group",
        "original_question_type",
        "protocol",
        "gold",
        *[
            f"{name}_{suffix}"
            for name in names
            for suffix in ("prediction", "probabilities", "error")
        ],
    ]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for benchmark, records in benchmarks.items():
            for record in records:
                row = {
                    "benchmark": benchmark,
                    "id": record.id,
                    "source_id": record.source_id,
                    "split": record.split,
                    "group": context(record),
                    "original_question_type": record.extra["original_question_type"],
                    "protocol": record.extra["protocol"],
                    "gold": json.dumps(record.gold),
                }
                for name in names:
                    prediction = predictions[name].get(benchmark, {}).get(record.id)
                    probs = prediction.probs if prediction else {}
                    row[f"{name}_prediction"] = max(probs, key=probs.get) if probs else ""
                    row[f"{name}_probabilities"] = json.dumps(probs)
                    row[f"{name}_error"] = prediction.error if prediction else "not-run"
                writer.writerow(row)


def run_comparison(
    benchmarks,
    samples,
    *,
    work: Path,
    output: Path,
    model_repo: str,
    model_revision: str,
    model_run_id: str,
    code_revision: str,
    seed: int = 17,
    microbatch: int = 4,
    diba_revision: str | None = None,
):
    """Load one model at a time, save each benchmark immediately, report failures."""
    import torch
    from huggingface_hub import HfApi

    from kodoom.config import load_profile
    from kodoom.runs import Run
    from kodoom.training import laya

    if not torch.cuda.is_available():
        raise ExternalEvalError("select a Kaggle GPU before running real-model inference")
    device = "cuda:0"
    diba_revision = diba_revision or HfApi().model_info("Dibachain/DibaOne-X1").sha
    if not re.fullmatch(r"[0-9a-f]{40}", diba_revision):
        raise ExternalEvalError("Diba revision must resolve to an immutable 40-character Hub SHA")
    identity = {
        "code_revision": code_revision,
        "protocol": PROTOCOL,
        "samples": samples,
        "models": {
            "laya-multilingual": {
                "repo": laya.MODEL_ID,
                "revision": laya.MODEL_REVISION,
                "inference_code_revision": laya.CODE_REVISION,
                "inference_code_sha256": laya.COMMON_SHA,
            },
            "kodoom": {
                "repo": model_repo,
                "revision": model_revision,
                "path": f"{model_run_id}/best",
                "inference_code_revision": laya.CODE_REVISION,
                "inference_code_sha256": laya.COMMON_SHA,
            },
            "dibaone-x1": {
                "repo": "Dibachain/DibaOne-X1",
                "revision": diba_revision,
                "mode": "best",
                "backend": "torch",
            },
        },
        "seed": seed,
        "microbatch": microbatch,
        "normalization": "kodoom.normalize.normalize for all three models",
        "probabilities": "Laya/Kodoom temperature 1; X1 as shipped; no test calibration",
        "latency": "model API/inference only; batched Laya time amortized per question",
    }
    output.mkdir(parents=True, exist_ok=True)
    # Refuse stale predictions under another selection/configuration.
    if (output / "manifest.json").exists():
        raise ExternalEvalError("results directory already contains a run; select a new WORK path")
    profile = replace(
        load_profile("kaggle"), runs_dir=output.parent / "runs", scratch_dir=work / "scratch"
    )
    run = Run.open(profile, "comparison-" + digest(identity)[:16], identity)
    save_json(output / "samples.json", samples)
    predictions = {name: {} for name in MODEL_NAMES}
    report = {"protocol": PROTOCOL, "benchmarks": {}}
    manifest = {
        **identity,
        "status": "running",
        "model_errors": {},
        "device": device,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
    }
    manifest["dependencies"] = {}
    for dependency in ("torch", "transformers", "huggingface_hub", "safetensors", "pyarrow"):
        try:
            manifest["dependencies"][dependency] = version(dependency)
        except PackageNotFoundError:
            manifest["dependencies"][dependency] = None
    save_json(output / "manifest.json", manifest)
    try:
        for name in MODEL_NAMES:
            print("Loading", name, flush=True)
            loaded = None
            started = time.perf_counter()
            try:
                if name == "dibaone-x1":
                    loaded = load_diba(work, diba_revision, device)
                elif name == "kodoom":
                    loaded = load_kodoom(
                        work, repo=model_repo, revision=model_revision, run_id=model_run_id
                    )
                else:
                    loaded = laya.load(work / "cache")
                manifest.setdefault("model_load_seconds", {})[name] = time.perf_counter() - started
                for benchmark, records in benchmarks.items():
                    print(name, benchmark, len(records), flush=True)
                    if name == "dibaone-x1":
                        # Laya.encode normalizes internally; give X1 the same model input.
                        result = predict_diba_x1(
                            [model_record(r) for r in records], loaded, mode="best"
                        )
                    else:
                        model, tokenizer, cfg, common = loaded
                        manifest.setdefault("inference_settings", {})[name] = {
                            "max_len": cfg["max_len"],
                            "head_max_len": cfg["head_max_len"],
                            "temperature": cfg.get("temperature"),
                        }
                        result = predict_laya(
                            records,
                            model,
                            tokenizer,
                            common,
                            cfg,
                            device=device,
                            microbatch=microbatch,
                        )
                    predictions[name][benchmark] = {p.id: p for p in result}
                    write_predictions(output / benchmark / f"{name}.jsonl", result)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                manifest["model_errors"][name] = message
                print(name, "failed:", message, flush=True)
                for benchmark, records in benchmarks.items():
                    if benchmark not in predictions[name]:
                        result = [
                            Prediction(r.id, {}, gold=dict(r.gold), error=message) for r in records
                        ]
                        predictions[name][benchmark] = {p.id: p for p in result}
                        write_predictions(output / benchmark / f"{name}.jsonl", result)
            finally:
                loaded = None
                # Tuple unpacking above would otherwise retain a Laya model on GPU.
                model = tokenizer = cfg = common = None
                gc.collect()
                torch.cuda.empty_cache()
        manifest["prediction_errors"] = {
            name: sum(bool(p.error) for mapping in by_benchmark.values() for p in mapping.values())
            for name, by_benchmark in predictions.items()
        }
        manifest["status"] = (
            "failed"
            if manifest["model_errors"]
            else "completed-with-errors"
            if any(manifest["prediction_errors"].values())
            else "completed"
        )
    finally:
        for benchmark, records in benchmarks.items():
            maps = {name: predictions[name].get(benchmark, {}) for name in MODEL_NAMES}
            report["benchmarks"][benchmark] = {
                **comparison_report(records, maps, seed=seed),
                "interpretation": samples[benchmark]["interpretation"],
                "by_original_question_type": {
                    kind: comparison_report(
                        [r for r in records if r.extra["original_question_type"] == kind],
                        maps,
                        seed=seed,
                        n_boot=200,
                    )
                    for kind in sorted({r.extra["original_question_type"] for r in records})
                },
                "by_group": {
                    group: comparison_report(
                        [r for r in records if context(r) == group], maps, seed=seed, n_boot=200
                    )
                    for group in sorted({context(r) for r in records})
                },
            }
        save_json(output / "report.json", report)
        write_comparison_csv(output, benchmarks, predictions)
        with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "benchmark",
                    "interpretation",
                    "model",
                    "n",
                    "accuracy",
                    "nll",
                    "brier",
                    "ece_15",
                    "coverage",
                    "failed",
                    "missing",
                    "common_n",
                    "common_accuracy",
                ],
            )
            writer.writeheader()
            for benchmark, result in report["benchmarks"].items():
                for name, summary in result["models"].items():
                    writer.writerow(
                        {
                            "benchmark": benchmark,
                            "interpretation": result["interpretation"],
                            "model": name,
                            **{
                                key: summary[key]
                                for key in (
                                    "n",
                                    "accuracy",
                                    "nll",
                                    "brier",
                                    "ece_15",
                                    "coverage",
                                    "failed",
                                    "missing",
                                )
                            },
                            "common_n": result["common_items"]["n"],
                            "common_accuracy": result["common_items"]["models"][name]["accuracy"],
                        }
                    )
        manifest["output_sha256"] = {
            str(path.relative_to(output)): file_sha(path)
            for path in output.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        if manifest["status"] == "running":
            manifest["status"] = "interrupted"
        save_json(output / "manifest.json", manifest)
        run.finish(evaluation_status=manifest["status"])
    if manifest["model_errors"]:
        raise ExternalEvalError(f"model failure; inspect {output / 'manifest.json'}")
    return report
