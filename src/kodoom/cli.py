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

from kodoom import __version__, baselines, helmo
from kodoom.calibration import (
    CalibrationError,
    fit_calibration,
    read_calibration,
    write_calibration,
)
from kodoom.check import FAIL, apply_environment, run_checks
from kodoom.config import BUILTIN_PROFILES, Profile, ProfileError, load_profile
from kodoom.evaluation import evaluate, format_table
from kodoom.generators import GENERATORS
from kodoom.generators.common import DEFAULT_PAIRS_PER_KIND, GeneratorError
from kodoom.inspect_hf import InspectError, inspect_dataset
from kodoom.metrics import MetricError
from kodoom.predictions import PredictionError, read_predictions, write_predictions
from kodoom.runs import RunError, list_runs
from kodoom.schema import RecordError, read_jsonl, write_jsonl
from kodoom.sources import SourceError, check_record, get_source
from kodoom.translate.exchange import (
    ExchangeError,
    PrecomputedTranslator,
    check_filled,
    read_units,
    units_for,
    write_units,
)
from kodoom.translate.exchange import instructions as exchange_instructions
from kodoom.translate.glossary import load as load_glossary
from kodoom.translate.pilot import (
    PilotError,
    build_sheet,
    read_sheet,
    score_sheet,
    write_key,
    write_sheet,
)
from kodoom.translate.pipeline import (
    MODEL_TRANSLATORS,
    TRANSLATORS,
    cases,
    pick_cases,
    translate_file,
    translator_factory,
)
from kodoom.translate.report import side_by_side, summarize
from kodoom.translate.rules import RuleError
from kodoom.typed_decisions import REVISION, SPLITS, TypedDecisionsError, field_stats, load_records


def main(argv: list[str] | None = None) -> int:
    # Persian in output must not crash a Windows console using a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        ProfileError,
        RecordError,
        SourceError,
        RunError,
        GeneratorError,
        InspectError,
        PredictionError,
        TypedDecisionsError,
        RuleError,
        ExchangeError,
        PilotError,
        CalibrationError,
        MetricError,
    ) as e:
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
    generate.add_argument(
        "--out", type=Path, help="output directory (default: <the profile's data_dir>/skills)"
    )
    generate.add_argument("--seed", type=int, help="random seed (default: the profile's)")
    generate.add_argument(
        "--pairs-per-kind",
        type=int,
        help=f"minimal pairs per question kind (default {DEFAULT_PAIRS_PER_KIND}, or the "
        "profile's max_cases_per_source if that is smaller)",
    )
    generate.set_defaults(func=_generate)

    inspect = commands.add_parser(
        "inspect", help="print the structure of a Hugging Face dataset (run once on Colab)"
    )
    inspect.add_argument("repo", help="dataset id, e.g. LocalLLaMA/typed-decisions")
    _add_profile_args(inspect)
    inspect.add_argument("--revision", help="branch, tag or commit (default: main)")
    inspect.add_argument(
        "--config", help="config (folder) to read; default: the first with the split"
    )
    inspect.add_argument("--split", default="test")
    inspect.add_argument("--rows", type=int, default=2, help="rows to print in full")
    inspect.add_argument(
        "--card-lines", type=int, default=0, help="also print the card's first N lines"
    )
    inspect.add_argument("--max-chars", type=int, default=6000, help="cut a printed row after this")
    inspect.add_argument("--out", type=Path, help="also write the report to this file")
    inspect.set_defaults(func=_inspect)

    fetch = commands.add_parser(
        "fetch", help="download a source dataset and convert it to records (run on Colab)"
    )
    fetch.add_argument("dataset", choices=["typed-decisions", "helmo"])
    _add_profile_args(fetch)
    fetch.add_argument(
        "--revision", help="commit to read (default: the one pinned for the dataset)"
    )
    fetch.add_argument(
        "--limit",
        type=int,
        help="typed-decisions: cases per workflow and split; helmo: records "
        "(default: the profile's cap)",
    )
    fetch.add_argument(
        "--out", type=Path, help="output directory (default: <data_dir>/<dataset>/en)"
    )
    fetch.set_defaults(func=_fetch)

    translate = commands.add_parser(
        "translate", help="translate fetched typed-decisions cases into Persian (resumable)"
    )
    translate.add_argument("dataset", choices=["typed-decisions"])
    _add_profile_args(translate)
    translate.add_argument(
        "--translator",
        choices=[*TRANSLATORS, *MODEL_TRANSLATORS],
        required=True,
        help="stub only pretends (dev and tests); the others load a model (Colab, GPU)",
    )
    translate.add_argument("--split", choices=list(SPLITS), help="default: both")
    translate.add_argument("--limit", type=int, help="cases per split (default: all fetched)")
    translate.add_argument(
        "--balanced", action="store_true", help="with --limit, take cases from every workflow"
    )
    translate.set_defaults(func=_translate)

    show = commands.add_parser(
        "translations", help="check counts and English next to Persian for a translator's output"
    )
    show.add_argument("dataset", choices=["typed-decisions"])
    _add_profile_args(show)
    show.add_argument("--translator", required=True, help="the folder under fa/ to read")
    show.add_argument("--split", choices=list(SPLITS), default="test")
    show.add_argument("--show", type=int, default=3, help="cases to print side by side")
    show.set_defaults(func=_translations)

    sheet = commands.add_parser(
        "pilot-sheet", help="write the blind review sheet for two translators' output"
    )
    sheet.add_argument("dataset", choices=["typed-decisions"])
    _add_profile_args(sheet)
    sheet.add_argument("--a", required=True, metavar="TRANSLATOR", help="a folder under fa/")
    sheet.add_argument("--b", required=True, metavar="TRANSLATOR", help="the other translator")
    sheet.add_argument("--split", choices=list(SPLITS), default="train")
    sheet.add_argument("--seed", type=int, default=1234, help="draws which one is A or B")
    sheet.set_defaults(func=_pilot_sheet)

    score = commands.add_parser("pilot-score", help="tally a filled review sheet against its key")
    score.add_argument("dataset", choices=["typed-decisions"])
    _add_profile_args(score)
    score.add_argument(
        "--sheet", default="sheet-filled.csv", help="file name in the pilot folder on Drive"
    )
    score.set_defaults(func=_pilot_score)

    export = commands.add_parser(
        "export-units",
        help="write the distinct texts to translate, for a translator outside kodoom",
    )
    export.add_argument("dataset", choices=["typed-decisions"])
    _add_profile_args(export)
    export.add_argument("--split", choices=list(SPLITS), default="train")
    export.add_argument("--limit", type=int, help="cases (default: all fetched)")
    export.add_argument("--balanced", action="store_true", help="with --limit, all workflows")
    export.set_defaults(func=_export_units)

    imp = commands.add_parser(
        "import-units", help="turn a filled units file into checked Persian records"
    )
    imp.add_argument("dataset", choices=["typed-decisions"])
    _add_profile_args(imp)
    imp.add_argument("--split", choices=list(SPLITS), default="train")
    imp.add_argument("--name", required=True, help="the translator's name (folder under fa/)")
    imp.add_argument("--units", default="units-filled.jsonl", help="file in the exchange folder")
    imp.add_argument("--limit", type=int, help="cases (default: all fetched)")
    imp.add_argument("--balanced", action="store_true", help="with --limit, all workflows")
    imp.set_defaults(func=_import_units)

    fields = commands.add_parser(
        "fields", help="statistics of the text fields of fetched typed-decisions records"
    )
    _add_profile_args(fields)
    fields.add_argument(
        "files",
        nargs="*",
        type=Path,
        metavar="FILE.jsonl",
        help="default: the train and test files `kodoom fetch typed-decisions` wrote",
    )
    fields.add_argument("--samples", type=int, default=3, help="example values per field")
    fields.set_defaults(func=_fields)

    baseline = commands.add_parser(
        "baseline", help="write trivial-baseline predictions (uniform, prior, oracle)"
    )
    baseline.add_argument("name", choices=["uniform", "prior", "oracle"])
    baseline.add_argument("--data", type=Path, required=True, help="gold records (JSONL)")
    baseline.add_argument("--split", help="only records of this split")
    baseline.add_argument("--train", type=Path, help="training records for the prior baseline")
    baseline.add_argument("--out", type=Path, required=True, help="prediction file to write")
    baseline.set_defaults(func=_baseline)

    score = commands.add_parser("score", help="score a prediction file against gold records")
    score.add_argument("--gold", type=Path, required=True, help="gold records (JSONL)")
    score.add_argument("--pred", type=Path, required=True, help="prediction file (JSONL)")
    score.add_argument("--split", help="only records of this split (default: all)")
    score.add_argument(
        "--by", action="append", default=[], help="break down by a field or extra.KEY"
    )
    score.add_argument("--calibration", type=Path, help="apply a calibration.json before scoring")
    score.add_argument("--seen-from", type=Path, help="training records: seen vs unseen families")
    score.add_argument("--json", type=Path, help="also write the full report as JSON")
    score.set_defaults(func=_score)

    calibrate = commands.add_parser(
        "calibrate", help="fit calibration.json (a temperature per question type) from predictions"
    )
    calibrate.add_argument("--gold", type=Path, required=True)
    calibrate.add_argument("--pred", type=Path, required=True)
    calibrate.add_argument("--split", default="calibration", help="split to fit on (never test)")
    calibrate.add_argument("--out", type=Path, required=True)
    calibrate.set_defaults(func=_calibrate)

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
    out_dir = args.out if args.out is not None else profile.data_dir / "skills"
    seed = profile.seed if args.seed is None else args.seed
    pairs = args.pairs_per_kind or DEFAULT_PAIRS_PER_KIND
    if args.pairs_per_kind is None and profile.max_cases_per_source is not None:
        pairs = min(pairs, profile.max_cases_per_source)

    for name in args.names or list(GENERATORS):
        generator = GENERATORS[name]
        records = generator.generate(seed, pairs)
        for record in records:
            check_record(record)
        path = out_dir / f"{name}.jsonl"
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
        (out_dir / f"{name}.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{path}: {len(records)} records (seed {seed}, {pairs} pairs per kind)")
        print(f"  splits: {manifest['by_split']}")
        print(f"  sha256: {manifest['sha256']}")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    if args.dataset == "helmo":
        return _fetch_helmo(args)
    profile = _load(args)
    args.revision = args.revision or REVISION
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise InspectError(
            "huggingface_hub is not installed: pip install huggingface_hub pyarrow"
        ) from e
    limit = args.limit if args.limit is not None else profile.max_cases_per_source
    out_dir = args.out if args.out is not None else profile.data_dir / "typed-decisions" / "en"
    by_split = load_records(hf_hub_download, revision=args.revision, limit=limit)
    manifest = {
        "source": "LocalLLaMA/typed-decisions",
        "revision": args.revision,
        "license": get_source("LocalLLaMA/typed-decisions").license,
        "kodoom_version": __version__,
        "cases_per_workflow_and_split": limit,
        "splits": {},
    }
    for split, records in by_split.items():
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, records)
        cases = {r.source_id for r in records}
        manifest["splits"][split] = {
            "cases": len(cases),
            "decisions": len(records),
            "by_workflow": dict(Counter(r.extra["workflow"] for r in records)),
            "by_type": dict(Counter(r.question_type for r in records)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"{path}: {len(cases)} cases, {len(records)} decisions")
        print(f"  by workflow: {manifest['splits'][split]['by_workflow']}")
        print(f"  by type:     {manifest['splits'][split]['by_type']}")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"commit {args.revision}")
    return 0


def _fetch_helmo(args: argparse.Namespace) -> int:
    profile = _load(args)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise InspectError(
            "huggingface_hub is not installed: pip install huggingface_hub pyarrow"
        ) from e
    revision = args.revision or helmo.REVISION
    limit = args.limit if args.limit is not None else profile.max_cases_per_source
    out_dir = args.out if args.out is not None else profile.data_dir / "helmo" / "en"
    records = helmo.load_records(hf_hub_download, revision=revision, limit=limit)
    path = out_dir / "train.jsonl"
    write_jsonl(path, records)
    manifest = {
        "source": helmo.SOURCE,
        "revision": revision,
        "license": get_source(helmo.SOURCE).license,
        "kodoom_version": __version__,
        "records_limit": limit,
        "records": len(records),
        "by_type": dict(Counter(r.question_type for r in records)),
        "topics": len({r.extra["topic"] for r in records}),
        "gold_rescaled": sum("gold_sum_in_source" in r.extra for r in records),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{path}: {len(records)} records, {manifest['topics']} topics")
    print(f"  by type: {manifest['by_type']}")
    print(f"  gold not summing to 1 in the source (rescaled, flagged): {manifest['gold_rescaled']}")
    print(f"commit {revision}")
    return 0


def _translate(args: argparse.Namespace) -> int:
    profile = _load(args)
    translator = translator_factory(args.translator)()
    base = profile.data_dir / "typed-decisions"
    failed = 0
    for split in [args.split] if args.split else list(SPLITS):
        source = base / "en" / f"{split}.jsonl"
        if not source.exists():
            raise InspectError(f"{source} does not exist; run `kodoom fetch typed-decisions` first")
        out = base / "fa" / translator.name / f"{split}.jsonl"
        stats = translate_file(source, out, translator, limit=args.limit, balanced=args.balanced)
        print(f"{out}: {stats['translated']} cases translated, {stats['skipped']} already done")
        failed += stats["with_findings"] + stats["failed"]
        if stats["with_findings"]:
            print(f"  {stats['with_findings']} cases have check findings (checks_passed=false)")
        if stats["failed"]:
            log = out.with_name(out.stem + ".failures.jsonl")
            print(f"  {stats['failed']} cases could not be translated; reasons in {log}")
    return 0 if not failed else 1


def _translations(args: argparse.Namespace) -> int:
    profile = _load(args)
    base = profile.data_dir / "typed-decisions"
    fa_path = base / "fa" / args.translator / f"{args.split}.jsonl"
    if not fa_path.exists():
        raise InspectError(f"{fa_path} does not exist; run `kodoom translate` first")
    persian = list(read_jsonl(fa_path))
    english_by_id = {r.id: r for r in read_jsonl(base / "en" / f"{args.split}.jsonl")}
    summary = summarize(persian)
    print(f"{fa_path}: {summary['cases']} cases, {summary['decisions']} decisions")
    print(f"  cases with check findings: {summary['cases_with_findings']}")
    print(f"  findings by check: {summary['findings_by_check']}")
    for fa_case in cases(persian)[: args.show]:
        english = [english_by_id[r.id.removesuffix(":fa")] for r in fa_case]
        print()
        print(side_by_side(english, fa_case))
    return 0


def _pilot_dir(profile: Profile) -> Path:
    return profile.data_dir / "typed-decisions" / "pilot"


def _pilot_sheet(args: argparse.Namespace) -> int:
    profile = _load(args)
    base = profile.data_dir / "typed-decisions"
    english = cases(read_jsonl(base / "en" / f"{args.split}.jsonl"))
    candidates = {}
    for name in (args.a, args.b):
        path = base / "fa" / name / f"{args.split}.jsonl"
        if not path.exists():
            raise InspectError(f"{path} does not exist; run `kodoom translate` first")
        candidates[name] = {c[0].source_id: c for c in cases(read_jsonl(path))}
    rows, key = build_sheet(english, candidates, seed=args.seed)
    out = _pilot_dir(profile)
    write_sheet(out / "sheet.csv", rows)
    write_key(out / "key-do-not-open.json", key)
    print(f"{out / 'sheet.csv'}: {len(rows)} cases, two translations each, in random order")
    print("Fill the columns better (A, B or tie) and meaning_errors_A / _B, then save the")
    print(f"file as {out / 'sheet-filled.csv'} and run `kodoom pilot-score`.")
    print(f"Do not open {out / 'key-do-not-open.json'}: it tells which translator was A or B.")
    return 0


def _pilot_score(args: argparse.Namespace) -> int:
    profile = _load(args)
    folder = _pilot_dir(profile)
    sheet_path, key_path = folder / args.sheet, folder / "key-do-not-open.json"
    for path in (sheet_path, key_path):
        if not path.exists():
            raise InspectError(f"{path} does not exist")
    result = score_sheet(read_sheet(sheet_path), json.loads(key_path.read_text(encoding="utf-8")))
    print(f"{result['cases']} cases: {result['ties']} ties, {result['unrated']} not rated")
    for name, wins in result["wins"].items():
        errors = result["mean_meaning_errors"][name]
        shown = "n/a" if errors is None else f"{errors:.2f}"
        print(f"  {name}: {wins} wins, mean meaning errors per case {shown}")
    for workflow, counts in result["wins_by_workflow"].items():
        print(f"  {workflow}: {counts}")
    return 0


def _exchange_dir(profile: Profile) -> Path:
    return profile.data_dir / "typed-decisions" / "exchange"


def _export_units(args: argparse.Namespace) -> int:
    profile = _load(args)
    source = profile.data_dir / "typed-decisions" / "en" / f"{args.split}.jsonl"
    if not source.exists():
        raise InspectError(f"{source} does not exist; run `kodoom fetch typed-decisions` first")
    chosen = pick_cases(cases(read_jsonl(source)), args.limit, args.balanced)
    units = units_for(chosen, load_glossary())
    folder = _exchange_dir(profile)
    write_units(folder / "units.jsonl", units)
    (folder / "INSTRUCTIONS.md").write_text(exchange_instructions(len(units)), encoding="utf-8")
    print(f"{folder / 'units.jsonl'}: {len(units)} distinct texts from {len(chosen)} cases")
    print(f"Give the translator units.jsonl and {folder / 'INSTRUCTIONS.md'}; save the filled file")
    print(f"as {folder / 'units-filled.jsonl'} and run `kodoom import-units --name NAME`.")
    return 0


def _import_units(args: argparse.Namespace) -> int:
    profile = _load(args)
    base = profile.data_dir / "typed-decisions"
    folder = _exchange_dir(profile)
    filled, original = folder / args.units, folder / "units.jsonl"
    for path in (filled, original):
        if not path.exists():
            raise InspectError(f"{path} does not exist")
    units = read_units(filled)
    problems = check_filled(read_units(original), units)
    for kind, ids in problems.items():
        if ids:
            print(f"{len(ids)} units {kind}: {', '.join(ids[:8])}{' ...' if len(ids) > 8 else ''}")
    source = base / "en" / f"{args.split}.jsonl"
    translator = PrecomputedTranslator(units, args.name)
    out = base / "fa" / args.name / f"{args.split}.jsonl"
    stats = translate_file(source, out, translator, limit=args.limit, balanced=args.balanced)
    print(f"{out}: {stats['translated']} cases, {stats['skipped']} already there")
    if stats["failed"]:
        print(
            f"  {stats['failed']} cases could not be built (reasons in {out.with_suffix('')}"
            ".failures.jsonl)"
        )
    if stats["with_findings"]:
        print(f"  {stats['with_findings']} cases have check findings (checks_passed=false)")
    return 0 if not (stats["failed"] or stats["with_findings"] or any(problems.values())) else 1


def _fields(args: argparse.Namespace) -> int:
    profile = _load(args)
    files = args.files or [
        profile.data_dir / "typed-decisions" / "en" / f"{split}.jsonl" for split in SPLITS
    ]
    records = [r for path in files for r in read_jsonl(path)]
    print(f"{len({r.source_id for r in records})} cases, {len(records)} decisions")
    for workflow, rows in field_stats(records, samples=args.samples).items():
        print(f"\n== {workflow}")
        for r in rows:
            print(
                f"{r['path']}: coverage {r['coverage']:.2f}, {r['distinct']} distinct "
                f"in {r['occurrences']}, mean {r['mean_chars']:.0f} chars"
            )
            for sample in r["samples"]:
                print(f"    {sample!r}")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    _load(args)  # points HF_HOME at the profile's cache_dir before anything downloads
    report = inspect_dataset(
        args.repo,
        revision=args.revision,
        config=args.config,
        split=args.split,
        rows=args.rows,
        max_chars=args.max_chars,
        card_lines=args.card_lines,
    )
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
    return 0


def _records(path: Path, split: str | None):
    return [r for r in read_jsonl(path) if split is None or r.split == split]


def _baseline(args: argparse.Namespace) -> int:
    records = _records(args.data, args.split)
    if args.name == "prior":
        if args.train is None:
            raise MetricError("the prior baseline needs --train")
        predictions = baselines.prior(records, _records(args.train, "train"))
    else:
        predictions = getattr(baselines, args.name)(records)
    count = write_predictions(args.out, predictions)
    print(f"{args.out}: {count} predictions ({args.name})")
    return 0


def _score(args: argparse.Namespace) -> int:
    records = _records(args.gold, args.split)
    seen = None
    if args.seen_from:
        seen = {r.task_family for r in read_jsonl(args.seen_from) if r.split == "train"}
    calibration = read_calibration(args.calibration) if args.calibration else None
    result = evaluate(
        records, read_predictions(args.pred), calibration=calibration, seen_families=seen
    )
    summary = result.summary()
    by = list(args.by) or ["question_type"]
    if seen is not None:
        by.append("family_status")
    print(format_table({"all": summary}, "scored"))
    report: dict = {"summary": summary.to_dict(), "by": {}}
    for name in by:
        rows = result.by(name)
        print("\n" + format_table(rows, name))
        report["by"][name] = {k: v.to_dict() for k, v in rows.items()}
    pairs = result.pairs()
    if pairs:
        print(
            f"\nminimal pairs: {pairs.pairs} pairs, both right {pairs.pair_accuracy:.3f} "
            f"(item accuracy {pairs.item_accuracy:.3f}, exactly one right in {pairs.one_right})"
        )
        report["pairs"] = vars(pairs)
    print(
        f"\nmissing {len(result.missing)}, failed {len(result.failed)}, "
        f"unexpected {len(result.unexpected)}"
    )
    report["missing"], report["failed"] = len(result.missing), len(result.failed)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        args.json.write_text(text, encoding="utf-8")
    return 1 if result.missing or result.failed else 0


def _calibrate(args: argparse.Namespace) -> int:
    if args.split == "test":
        raise MetricError("never fit a calibration on the test split")
    predictions = read_predictions(args.pred)
    examples = [
        (r.question_type, predictions[r.id].probs, r.gold)
        for r in _records(args.gold, args.split)
        if r.id in predictions and not predictions[r.id].error
    ]
    calibration = fit_calibration(examples)
    write_calibration(args.out, calibration)
    print(
        f"{args.out}: global T {calibration.temperature:.3f}, per type {calibration.temperatures}"
    )
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
