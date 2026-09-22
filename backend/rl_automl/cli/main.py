"""Production command line interface for RL AutoML.

Planning and execution intentionally remain separate around an explicit approval gate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import pandas as pd

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.errors import AutoMLError
from rl_automl.core.types import RunObject, RunStatus, TaskType
from rl_automl.dataset import remote
from rl_automl.dataset.loaders import list_sources, load_source, save_frame
from rl_automl.dataset.validation import sha256_file
from rl_automl.orchestrator import AutoMLOrchestrator, PlanRequest

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_NOT_APPROVED = 3
EXIT_UNAVAILABLE = 4


def _configure_console() -> None:
    """Avoid UnicodeEncodeError on legacy Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with suppress(OSError, ValueError):
                reconfigure(errors="backslashreplace")


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str, ensure_ascii=True))


def _config(path: str | None) -> AutoMLConfig:
    return AutoMLConfig.load(path)


def _read_path(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    separator = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def _load_dataset(
    value: str, *, source: bool, allow_network: bool, config: AutoMLConfig
) -> tuple[pd.DataFrame, str, str, str]:
    if source:
        frame, descriptor = load_source(
            value,
            allow_network=allow_network,
            cache_dir=config.remote_dataset_dir(),
        )
        destination = config.artifacts_dir() / "datasets" / f"{descriptor.name}.csv"
        save_frame(frame, destination)
        resolved = destination.resolve()
        return frame, descriptor.name, str(resolved), sha256_file(resolved)

    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    return _read_path(path), path.name, str(path), sha256_file(path)


def _plan(args: argparse.Namespace) -> int:
    config = _config(args.config)
    frame, name, path, digest = _load_dataset(
        args.dataset, source=args.source, allow_network=args.allow_network, config=config
    )
    orchestrator = AutoMLOrchestrator(config)
    run = orchestrator.plan(
        PlanRequest(
            problem_statement=args.problem,
            frame=frame,
            dataset_name=name,
            dataset_path=path,
            dataset_sha256=digest,
            target=args.target,
            task_type=args.task_type,
            metric=args.metric,
            use_policy=not args.no_policy,
        )
    )
    _render_plan(run, args.json)
    return EXIT_OK


def _render_plan(run: RunObject, as_json: bool) -> None:
    if as_json:
        _print_json(run.model_dump(mode="json"))
        return
    task = run.task
    print(f"Run: {run.run_id}")
    print(f"Status: {run.status.value}")
    if task:
        print(f"Task: {task.task_type.value}; target={task.target or 'none'}; metric={task.metric}")
    print("Recommended pipelines:")
    for item in run.rl_recommendations:
        expected = (
            "n/a" if item.expected_performance is None else f"{item.expected_performance:.4f}"
        )
        print(f"  {item.rank}. {item.model} (cost={item.expected_cost.value}, expected={expected})")
        if item.reason:
            print(f"     {item.reason}")
    print(f"No models were trained. To execute: automl run {run.run_id}")


def _parse_selection(raw: str) -> str | list[int] | list[str]:
    text = raw.strip()
    if text.lower() in {"all", "*"}:
        return "all"
    tokens = [part.strip() for part in text.split(",") if part.strip()]
    if not tokens:
        raise ValueError("approval selection is empty")
    if all(token.isdigit() for token in tokens):
        return [int(token) for token in tokens]
    return tokens


def _interactive_selection(run: RunObject) -> str | list[int] | list[str] | None:
    if not sys.stdin.isatty():
        print(
            "Approval required. Re-run with --approve all (or ranks/models) to train.",
            file=sys.stderr,
        )
        return None
    _render_plan(run, False)
    print("Training may consume substantial CPU, memory, and time.")
    try:
        answer = input("Approve pipelines (all, comma-separated ranks/models, or no)? ").strip()
    except EOFError:
        return None
    if answer.lower() in {"", "n", "no", "cancel"}:
        return None
    return _parse_selection(answer)


def _run(args: argparse.Namespace) -> int:
    config = _config(args.config)
    orchestrator = AutoMLOrchestrator(config)
    run = orchestrator.load_run(args.run_id)
    if run.status.is_terminal:
        raise ValueError(f"run '{run.run_id}' is already {run.status.value}")

    selection = (
        _parse_selection(args.approve) if args.approve is not None else _interactive_selection(run)
    )
    if selection is None:
        print("Run not approved; no training started.", file=sys.stderr)
        return EXIT_NOT_APPROVED

    orchestrator.approve(run, selection=selection, approved_by="cli")
    print(f"Approved {len(run.approved_experiments)} pipeline(s); starting training.")
    completed = orchestrator.execute(run)
    if args.json:
        _print_json(completed.model_dump(mode="json"))
    else:
        print(f"Run {completed.run_id}: {completed.status.value}")
        if completed.best_model:
            best = completed.best_model
            print(f"Best model: {best.model}; validation {best.primary_metric}")
            print(f"Validation score: {best.validation_score}")
        if completed.artifacts.model_zip:
            print(f"Model package: {completed.artifacts.model_zip}")
        if completed.error:
            print(f"Error: {completed.error}", file=sys.stderr)
    return EXIT_OK if completed.status is RunStatus.COMPLETE else EXIT_FAILED


def _status(args: argparse.Namespace) -> int:
    orchestrator = AutoMLOrchestrator(_config(args.config))
    if args.run_id:
        payload: Any = orchestrator.load_run(args.run_id).model_dump(mode="json")
    else:
        payload = orchestrator.list_runs()
    if args.json:
        _print_json(payload)
        return EXIT_OK
    if args.run_id:
        print(f"Run: {payload['run_id']}")
        print(f"Status: {payload['status']}")
        print(f"Dataset: {payload.get('dataset_name') or '-'}")
        best = payload.get("best_model") or {}
        if best:
            print(f"Best model: {best.get('model')} ({best.get('validation_score')})")
        if payload.get("error"):
            print(f"Error: {payload['error']}")
    elif not payload:
        print("No runs found.")
    else:
        print("RUN ID                 STATUS              DATASET              BEST MODEL")
        for item in payload:
            print(
                f"{item.get('run_id') or '-'!s:<22} "
                f"{item.get('status') or '-'!s:<19} "
                f"{item.get('dataset') or '-'!s:<20} "
                f"{item.get('best_model') or '-'!s}"
            )
    return EXIT_OK


def _download(args: argparse.Namespace) -> int:
    run = AutoMLOrchestrator(_config(args.config)).load_run(args.run_id)
    source_text = run.artifacts.model_zip if args.kind == "model" else run.artifacts.results_zip
    expected = (
        run.artifacts.model_zip_sha256 if args.kind == "model" else run.artifacts.results_zip_sha256
    )
    if not source_text:
        raise FileNotFoundError(f"run '{run.run_id}' has no {args.kind} archive")
    source = Path(source_text).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"recorded archive is missing: {source}")
    actual = sha256_file(source)
    if expected and actual != expected:
        raise ValueError("archive checksum does not match the run record")
    destination = Path(args.output).expanduser() if args.output else Path.cwd() / source.name
    destination = destination.resolve()
    if destination != source:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    result = {"run_id": run.run_id, "kind": args.kind, "path": str(destination), "sha256": actual}
    _print_json(result) if args.json else print(
        f"Downloaded {args.kind} archive to {destination}\nSHA256: {actual}"
    )
    return EXIT_OK


def _datasets(args: argparse.Namespace) -> int:
    config = _config(args.config)
    cache_dir = config.remote_dataset_dir()

    if args.download:
        targets = (
            remote.remote_names() if args.download == ["all"] else list(args.download)
        )
        unknown = [name for name in targets if not remote.is_remote(name)]
        if unknown:
            print(f"error: not curated datasets: {', '.join(unknown)}", file=sys.stderr)
            return EXIT_USAGE
        if not args.yes:
            print(
                "This downloads data from the public URLs listed below. "
                "Re-run with --yes to confirm.",
                file=sys.stderr,
            )
            for name in targets:
                spec = remote.REMOTE_DATASETS[name]
                print(f"  {name}: {spec.url}  ({spec.license})", file=sys.stderr)
            return EXIT_USAGE
        failures: list[str] = []
        for name in targets:
            spec = remote.REMOTE_DATASETS[name]
            if remote.is_cached(name, cache_dir):
                print(f"{name}: already cached at {remote.cached_frame_path(name, cache_dir)}")
                continue
            try:
                path = remote.download(name, cache_dir)
            except AutoMLError as exc:
                # One unreachable mirror should not abandon the rest of the batch.
                failures.append(name)
                print(f"{name}: FAILED {exc}", file=sys.stderr)
                continue
            print(f"{name}: {path}  sha256 verified  licence: {spec.license}")
        if failures:
            print(f"error: {len(failures)} download(s) failed: {', '.join(failures)}", file=sys.stderr)
            return EXIT_FAILED
        return EXIT_OK

    payload = list_sources(include_network=args.include_network, cache_dir=cache_dir)
    if args.json:
        _print_json(payload)
    else:
        print("NAME                              TASK                 TARGET               NETWORK")
        for item in payload:
            print(
                f"{item['name']:<33} {item['task_type']:<20} "
                f"{item.get('target') or '-'!s:<20} {'yes' if item['requires_network'] else 'no'}"
            )
        curated = remote.remote_descriptors(cache_dir)
        if curated:
            print("\nCURATED PUBLIC DATASETS")
            for item in curated:
                state = "cached" if item["cached"] else "not downloaded"
                print(f"  {item['name']:<28} {item['task_type']:<16} {state:<16} {item['license']}")
    return EXIT_OK


def _delegate(args: argparse.Namespace, module_name: str, expensive_label: str) -> int:
    if not args.approve:
        print(
            f"Explicit approval required. Re-run with --approve to start {expensive_label}.",
            file=sys.stderr,
        )
        return EXIT_NOT_APPROVED
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            print(f"Command unavailable: module '{module_name}' is not installed.", file=sys.stderr)
            return EXIT_UNAVAILABLE
        raise
    return int(module.main(args.arguments) or 0)


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("The API extra is required: pip install rl-automl[api]", file=sys.stderr)
        return EXIT_UNAVAILABLE
    config = _config(args.config)
    uvicorn.run(
        "rl_automl.api.app:app",
        host=args.host or config.api.host,
        port=args.port or config.api.port,
    )
    return EXIT_OK


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="path to configuration YAML")


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automl", description="RL-driven AutoML with explicit human approval"
    )
    parser.add_argument("--version", action="version", version="rl-automl 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="profile data and recommend pipelines; trains nothing")
    plan.add_argument("dataset", help="dataset path, or registered name with --source")
    plan.add_argument("--problem", required=True, help="plain-language ML objective")
    plan.add_argument(
        "--source", action="store_true", help="resolve dataset from the source registry"
    )
    plan.add_argument("--allow-network", action="store_true", help="allow a network-backed source")
    plan.add_argument("--target")
    plan.add_argument("--task-type", choices=[item.value for item in TaskType])
    plan.add_argument("--metric")
    plan.add_argument("--no-policy", action="store_true", help="use only the surrogate planner")
    _add_config(plan)
    _add_json(plan)
    plan.set_defaults(handler=_plan)

    run = sub.add_parser("run", help="approve and execute an existing plan")
    run.add_argument("run_id")
    run.add_argument(
        "--approve",
        nargs="?",
        const="all",
        metavar="SELECTION",
        help="explicitly approve all, comma-separated ranks, or model names",
    )
    _add_config(run)
    _add_json(run)
    run.set_defaults(handler=_run)

    status = sub.add_parser("status", help="show one run or list all runs")
    status.add_argument("run_id", nargs="?")
    _add_config(status)
    _add_json(status)
    status.set_defaults(handler=_status)

    download = sub.add_parser("download", help="copy and verify a completed run archive")
    download.add_argument("run_id")
    download.add_argument("--kind", choices=("model", "results"), default="model")
    download.add_argument("--output", help="destination file path")
    _add_config(download)
    _add_json(download)
    download.set_defaults(handler=_download)

    datasets = sub.add_parser("datasets", help="list registered datasets")
    datasets.add_argument("--include-network", action="store_true")
    datasets.add_argument(
        "--download",
        nargs="+",
        metavar="NAME",
        help="fetch curated public datasets by name, or 'all'; requires --yes",
    )
    datasets.add_argument(
        "--yes", action="store_true", help="confirm downloading from the listed public URLs"
    )
    _add_config(datasets)
    _add_json(datasets)
    datasets.set_defaults(handler=_datasets)

    for name, help_text, module_name, label in (
        ("train-rl", "train PPO planner policies", "rl_automl.training.train_rl", "RL training"),
        (
            "benchmark",
            "run the evaluation benchmark",
            "rl_automl.evaluation.benchmark",
            "benchmarking",
        ),
    ):
        delegated = sub.add_parser(name, help=help_text, add_help=False)
        delegated.add_argument("--help", action="store_true", dest="delegate_help")
        delegated.add_argument(
            "--approve", action="store_true", help="explicitly permit this expensive command"
        )
        delegated.add_argument("arguments", nargs=argparse.REMAINDER)
        delegated.set_defaults(
            handler=lambda ns, m=module_name, label=label: _delegate_command(ns, m, label)
        )

    serve = sub.add_parser("serve", help="start the optional HTTP API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    _add_config(serve)
    serve.set_defaults(handler=_serve)
    return parser


def _delegate_command(args: argparse.Namespace, module_name: str, label: str) -> int:
    if args.delegate_help:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                print(
                    f"Command unavailable: module '{module_name}' is not installed.",
                    file=sys.stderr,
                )
                return EXIT_UNAVAILABLE
            raise
        parser_builder = getattr(module, "build_parser", None)
        if parser_builder is not None:
            parser_builder().print_help()
            return EXIT_OK
        return int(module.main(["--help"]) or 0)
    return _delegate(args, module_name, label)


def main(argv: Sequence[str] | None = None) -> int:
    _configure_console()
    parser = build_parser()
    try:
        args, unknown = parser.parse_known_args(argv)
        if args.command in {"train-rl", "benchmark"}:
            args.arguments = unknown + args.arguments
        elif unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Cancelled; no further work started.", file=sys.stderr)
        return 130
    except AutoMLError as exc:
        _print_json(exc.to_dict()) if getattr(locals().get("args", None), "json", False) else print(
            f"error [{exc.code}]: {exc}", file=sys.stderr
        )
        return EXIT_USAGE
    except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
