#!/usr/bin/env python3
"""Collect budgeted real experiment outcomes for SurrogateModel.fit.

This utility is intentionally non-interactive and requires --yes. Every output row comes
from the real execution layer; it is not a surrogate prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.types import ExperimentSource, ExperimentSpec
from rl_automl.dataset.fingerprint import build_fingerprint
from rl_automl.dataset.loaders import get_source
from rl_automl.dataset.profiler import profile_dataset
from rl_automl.execution.executor import ExecutionLimits, PipelineExecutor
from rl_automl.search.model_registry import REGISTRY

DEFAULT_DATASETS = ("iris", "breast_cancer", "diabetes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled set of real model fits and write JSONL rows consumable by "
            "rl_automl.environment.surrogate.SurrogateModel.fit."
        )
    )
    parser.add_argument("--config", help="path to a configuration YAML")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="local bundled/synthetic source names",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="optional model-key allowlist; defaults to configured available models",
    )
    parser.add_argument(
        "--experiments-per-dataset",
        type=int,
        default=3,
        help="maximum real model fits per dataset (default: 3)",
    )
    parser.add_argument(
        "--time-budget-s",
        type=float,
        default=180.0,
        help="real execution time budget per dataset (default: 180)",
    )
    parser.add_argument(
        "--memory-budget-mb",
        type=float,
        default=2048.0,
        help="real execution memory budget per dataset (default: 2048)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="deterministically cap each source before profiling/training (default: 5000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/meta/surrogate_meta.jsonl"),
        help="destination JSONL path",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append rows instead of replacing the output file",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="required explicit confirmation that real model training is authorized",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.yes:
        raise ValueError("real experiments require explicit --yes confirmation")
    if args.experiments_per_dataset < 1:
        raise ValueError("--experiments-per-dataset must be at least 1")
    if args.time_budget_s <= 0:
        raise ValueError("--time-budget-s must be positive")
    if args.memory_budget_mb <= 0:
        raise ValueError("--memory-budget-mb must be positive")
    if args.max_rows < 50:
        raise ValueError("--max-rows must be at least 50")


def choose_experiments(
    task_type: Any,
    config: AutoMLConfig,
    count: int,
    model_allowlist: Sequence[str] | None,
) -> list[ExperimentSpec]:
    allowed = list(model_allowlist) if model_allowlist else list(config.models.enabled)
    candidates = []
    for spec in REGISTRY.for_task(task_type, enabled=allowed):
        available, reason = spec.is_available()
        if not available:
            print(f"  skip {spec.key}: {reason}")
            continue
        preset = spec.preset(0, task_type)
        preprocessing = list(spec.required_preprocessing)
        candidates.append(
            ExperimentSpec(
                model=spec.key,
                preprocessing=preprocessing,
                feature_selection="none",
                preset_index=0,
                hyperparameters=dict(preset.params),
                expected_cost=preset.cost,
                source=ExperimentSource.USER,
                rationale="controlled real meta-dataset collection",
            )
        )
    if not candidates:
        raise ValueError(f"no available configured models support {task_type.value}")
    return candidates[:count]


def finite_score(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return float(value)


def result_row(
    *,
    source_name: str,
    task_type: Any,
    fingerprint: Any,
    profile: Any,
    result: Any,
) -> dict[str, Any]:
    experiment = result.experiment
    spec = REGISTRY.get(experiment.model)
    return {
        "dataset": source_name,
        "model_key": experiment.model,
        "family": spec.family,
        "task_type": task_type.value,
        "preset_index": experiment.preset_index,
        "n_presets": spec.n_presets(task_type),
        "preprocessing": list(experiment.preprocessing),
        "feature_selection": experiment.feature_selection,
        "fingerprint": [float(value) for value in fingerprint.values],
        "score": finite_score(result.validation_score),
        "cost_s": max(float(result.training_time_s), 0.0),
        "peak_memory_mb": max(float(result.peak_memory_mb), 0.0),
        "failed": not result.succeeded,
        "error_type": result.error_type,
        "error": result.error,
        "metric": result.primary_metric,
        "dataset_profile": {"n_rows": profile.n_rows, "n_cols": profile.n_cols},
        "measured": True,
    }


def collect_rows(args: argparse.Namespace, config: AutoMLConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_index, source_name in enumerate(args.datasets):
        source = get_source(source_name)
        frame = source.load()
        if len(frame) > args.max_rows:
            frame = frame.sample(n=args.max_rows, random_state=config.runtime.seed).reset_index(
                drop=True
            )
        profile = profile_dataset(frame, source.target, config.dataset)
        fingerprint = build_fingerprint(profile, source.task_type)
        experiments = choose_experiments(
            source.task_type,
            config,
            args.experiments_per_dataset,
            args.models,
        )
        print(
            f"{source_name}: {len(frame):,} rows, {len(experiments)} real experiment(s), "
            f"budget={args.time_budget_s:g}s/{args.memory_budget_mb:g}MB"
        )
        executor = PipelineExecutor(
            task=source_task(source),
            config=config,
            run_id=f"meta_{dataset_index}_{source_name}",
            seed=config.runtime.seed + dataset_index,
            limits=ExecutionLimits(
                max_experiments=len(experiments),
                time_budget_s=args.time_budget_s,
                memory_budget_mb=args.memory_budget_mb,
            ),
        )
        executor.prepare(frame)
        summary = executor.run(experiments)
        for result in summary.results:
            row = result_row(
                source_name=source_name,
                task_type=source.task_type,
                fingerprint=fingerprint,
                profile=profile,
                result=result,
            )
            rows.append(row)
            print(
                f"  {row['model_key']}: failed={row['failed']} "
                f"score={row['score']:.5f} cost_s={row['cost_s']:.3f}"
            )
    return rows


def source_task(source: Any) -> Any:
    from rl_automl.core.metrics import direction_for
    from rl_automl.core.types import TaskSpec

    return TaskSpec(
        task_type=source.task_type,
        target=source.target,
        metric=source.effective_metric(),
        metric_direction=direction_for(source.effective_metric()),
        confidence=1.0,
        source="declared-dataset-source",
        notes=["task contract taken from dataset source metadata"],
    )


def write_rows(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="ascii", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        config = AutoMLConfig.load(args.config)
        rows = collect_rows(args, config)
        write_rows(args.output, rows, args.append)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    failures = sum(1 for row in rows if row["failed"])
    print(f"\nWrote {len(rows)} measured row(s) to {args.output} ({failures} failed experiments).")
    if len(rows) < 50:
        print("SurrogateModel.fit needs at least 50 rows; combine additional controlled batches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
