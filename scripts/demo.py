#!/usr/bin/env python3
"""Guarded end-to-end RL-AutoML demonstration.

The script always plans and prints recommendations before it asks for approval. No call to
approve() or execute() is made unless the user confirms interactively or passes --yes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.types import RunStatus
from rl_automl.dataset.loaders import load_source
from rl_automl.orchestrator import AutoMLOrchestrator, PlanRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan a local AutoML run, print recommendations, require explicit approval, "
            "then execute approved pipelines and explain the generated artifacts."
        )
    )
    parser.add_argument("--config", help="path to a configuration YAML")
    parser.add_argument(
        "--source",
        default="churn_demo_small",
        help="bundled or synthetic dataset source (default: churn_demo_small)",
    )
    parser.add_argument(
        "--problem",
        default="Predict which customers will churn",
        help="natural-language problem statement",
    )
    parser.add_argument(
        "--approve",
        default="all",
        metavar="SELECTION",
        help="approval selection: all, comma-separated ranks, or comma-separated model names",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="explicitly approve training without an interactive prompt",
    )
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help="skip a saved PPO policy and use surrogate planning",
    )
    parser.add_argument("--time-budget-s", type=float, help="real execution time budget")
    parser.add_argument("--memory-budget-mb", type=float, help="real execution memory budget")
    parser.add_argument(
        "--max-experiments",
        type=int,
        help="maximum experiments allowed by the run configuration",
    )
    return parser


def parse_selection(raw: str) -> str | list[int] | list[str]:
    value = raw.strip()
    if value.lower() in {"all", "*"}:
        return "all"
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("approval selection cannot be empty")
    if all(token.isdigit() for token in tokens):
        return [int(token) for token in tokens]
    return tokens


def print_plan(run: object) -> None:
    task = run.task
    profile = run.dataset_profile
    recommendation = run.recommendation_set
    print("\n=== PLAN COMPLETE: NO MODEL HAS BEEN TRAINED ===")
    print(f"run id   : {run.run_id}")
    print(f"dataset  : {run.dataset_name}")
    if profile is not None:
        print(f"shape    : {profile.n_rows:,} rows x {profile.n_cols:,} columns")
    if task is not None:
        print(f"task     : {task.task_type.value}")
        print(f"target   : {task.target}")
        print(f"metric   : {task.metric} ({task.metric_direction.value})")
    if recommendation is not None:
        print(f"planner  : {recommendation.planner}")
        for note in recommendation.notes:
            print(f"note     : {note}")

    print("\nRecommended pipelines:")
    for item in run.rl_recommendations:
        estimate = (
            "n/a" if item.expected_performance is None else f"{item.expected_performance:.4f}"
        )
        pre = ", ".join(item.preprocessing) if item.preprocessing else "default"
        print(
            f"  {item.rank}. {item.model} | preset={item.preset_index} | "
            f"estimated={estimate} | cost={item.expected_cost.value}"
        )
        print(f"     preprocessing={pre}; feature_selection={item.feature_selection}")
        if item.reason:
            print(f"     reason={item.reason}")
    print("\nEstimated scores above are surrogate estimates, not measured performance.")


def confirm_training(selection: str, assume_yes: bool) -> bool:
    if assume_yes:
        print("\n--yes supplied: explicit non-interactive approval recorded.")
        return True
    if not sys.stdin.isatty():
        print(
            "\nRefusing to train: input is non-interactive and --yes was not supplied.",
            file=sys.stderr,
        )
        return False
    print(f"\nRequested selection: {selection}")
    answer = input("Type APPROVE to authorize real model training, or press Enter to stop: ")
    return answer.strip() == "APPROVE"


def print_results(run: object) -> None:
    print("\n=== REAL EXECUTION RESULTS ===")
    print(f"status: {run.status.value}")
    for result in run.results:
        score = "n/a" if result.validation_score is None else f"{result.validation_score:.4f}"
        test = "n/a" if result.test_score is None else f"{result.test_score:.4f}"
        print(
            f"  {result.experiment.model}: {result.status.value}; "
            f"validation={score}; test={test}; train_s={result.training_time_s:.3f}"
        )
        if result.error:
            print(f"    error: {result.error}")

    print("\n=== ARTIFACT WALKTHROUGH ===")
    artifacts_dir = run.config_snapshot["runtime"]["artifacts_dir"]
    run_json = Path(artifacts_dir) / "runs" / run.run_id / "run.json"
    print(f"run JSON   : {run_json}")
    print(f"model dir  : {run.artifacts.model_dir or 'not produced'}")
    print(f"model ZIP  : {run.artifacts.model_zip or 'not produced'}")
    print(f"model SHA  : {run.artifacts.model_zip_sha256 or 'n/a'}")
    print(f"results ZIP: {run.artifacts.results_zip or 'not produced'}")
    print(f"results SHA: {run.artifacts.results_zip_sha256 or 'n/a'}")
    print(f"verified   : {run.artifacts.verified}")
    print("The model ZIP contains the serialized pipeline, metadata, loader, prediction CLI,")
    print("requirements, and package README. The results ZIP contains the comparison and history.")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AutoMLConfig.load(args.config)
    if args.time_budget_s is not None:
        config.search.time_budget_s = args.time_budget_s
    if args.memory_budget_mb is not None:
        config.search.memory_budget_mb = args.memory_budget_mb
    if args.max_experiments is not None:
        config.search.max_experiments = args.max_experiments
        config.search.min_experiments = min(config.search.min_experiments, args.max_experiments)
    config.validate_all()

    frame, source = load_source(args.source)
    orchestrator = AutoMLOrchestrator(config)
    run = orchestrator.plan(
        PlanRequest(
            problem_statement=args.problem,
            frame=frame,
            dataset_name=source.name,
            target=source.target,
            task_type=source.task_type,
            metric=source.effective_metric(),
            use_policy=not args.no_policy,
        )
    )
    print_plan(run)

    if not confirm_training(args.approve, args.yes):
        print("No approval recorded. Stopping before training.")
        return 0

    try:
        selection = parse_selection(args.approve)
        run = orchestrator.approve(run, selection=selection, approved_by="scripts/demo.py")
    except ValueError as exc:
        print(f"invalid approval selection: {exc}", file=sys.stderr)
        return 2

    print(f"Approved {len(run.approved_experiments)} pipeline(s). Real training starts now.")
    run = orchestrator.execute(run, frame=frame)
    print_results(run)
    return 0 if run.status is RunStatus.COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
