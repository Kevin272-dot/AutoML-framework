"""Honest, budget-matched evaluation of PPO against search baselines.

Simulated results use the analytic surrogate and are labelled ``simulated``. Real results
train models through :class:`AutoMLEnv`; they are never started unless both ``--real`` and
``--yes`` are supplied. The report keeps those two evidence sources separate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from rl_automl.agent.agent import PPOAgent
from rl_automl.baselines.searchers import (
    GreedySearcher,
    GridSearcher,
    PolicySearcher,
    RandomSearcher,
    SearchResult,
)
from rl_automl.core.config import AutoMLConfig
from rl_automl.core.metrics import direction_for
from rl_automl.core.types import TaskSpec, TaskType, utc_now_iso
from rl_automl.dataset.loaders import get_source
from rl_automl.environment.action_space import ActionSpace
from rl_automl.environment.automl_env import AutoMLEnv, RealEnvConfig
from rl_automl.environment.base import SearchTracker
from rl_automl.environment.reward import RewardCalculator
from rl_automl.environment.state_encoder import StateEncoder
from rl_automl.environment.surrogate import AnalyticPrior, SurrogateModel
from rl_automl.search.model_registry import REGISTRY
from rl_automl.training.train_rl import (
    CorpusEntry,
    build_action_space_for,
    build_corpus,
    make_environment,
    split_corpus,
)

DEFAULT_SEEDS = (101, 202, 303, 404, 505)
BOOTSTRAP_SAMPLES = 2000
PERMUTATION_SAMPLES = 10_000


def _bootstrap_mean_ci(
    values: list[float], *, seed: int = 0, samples: int = BOOTSTRAP_SAMPLES
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    means = np.mean(rng.choice(array, size=(samples, len(array)), replace=True), axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _paired_permutation_pvalue(
    left: list[float], right: list[float], *, seed: int = 0
) -> float | None:
    if len(left) != len(right) or not left:
        return None
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    observed = abs(float(np.mean(differences)))
    if np.allclose(differences, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(PERMUTATION_SAMPLES, len(differences)))
    permuted = np.abs(np.mean(signs * differences, axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (len(permuted) + 1))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_searcher: dict[str, list[float]] = defaultdict(list)
    paired: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        score = row.get("best_score")
        if score is None:
            continue
        by_searcher[row["searcher"]].append(float(score))
        paired[(row["dataset"], row["searcher"])].append(float(score))

    searchers: dict[str, Any] = {}
    for index, (name, values) in enumerate(sorted(by_searcher.items())):
        low, high = _bootstrap_mean_ci(values, seed=1000 + index)
        searchers[name] = {
            "n": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci95": [low, high],
        }

    comparisons: dict[str, Any] = {}
    ppo_datasets = sorted({dataset for dataset, searcher in paired if searcher == "ppo"})
    for baseline in sorted(name for name in by_searcher if name != "ppo"):
        ppo_values: list[float] = []
        baseline_values: list[float] = []
        for dataset in ppo_datasets:
            left = paired.get((dataset, "ppo"), [])
            right = paired.get((dataset, baseline), [])
            for ppo_score, baseline_score in zip(left, right, strict=False):
                ppo_values.append(ppo_score)
                baseline_values.append(baseline_score)
        differences = [
            ppo_score - baseline_score
            for ppo_score, baseline_score in zip(ppo_values, baseline_values, strict=True)
        ]
        comparisons[f"ppo_vs_{baseline}"] = {
            "n_pairs": len(differences),
            "mean_difference": float(np.mean(differences)) if differences else None,
            "p_value_paired_permutation": _paired_permutation_pvalue(
                ppo_values, baseline_values, seed=42
            ),
            "ppo_wins": int(sum(value > 0 for value in differences)),
            "ties": int(sum(np.isclose(value, 0.0) for value in differences)),
            "ppo_losses": int(sum(value < 0 for value in differences)),
        }
    return {"searchers": searchers, "comparisons": comparisons}


def _searcher_factories(
    agent: PPOAgent,
) -> dict[str, Callable[[int], Any]]:
    return {
        "ppo": lambda seed: PolicySearcher(agent),
        "random": lambda seed: RandomSearcher(seed),
        "greedy": lambda seed: GreedySearcher(),
        "grid": lambda seed: GridSearcher(),
    }


def _load_agent(
    task_type: TaskType,
    config: AutoMLConfig,
    action_space: ActionSpace,
    encoder: StateEncoder,
    device: str | None,
) -> PPOAgent:
    path = config.policy_path(task_type)
    if not path.is_file():
        raise FileNotFoundError(
            f"no checkpoint for {task_type.value}: expected {config.policy_destination(task_type)}"
        )
    agent, _extra = PPOAgent.load(
        path,
        action_space=action_space,
        state_encoder=encoder,
        rl_config=config.rl,
        device=device,
        strict=True,
    )
    return agent


def benchmark_simulated(
    entries: list[CorpusEntry],
    *,
    config: AutoMLConfig,
    budget: int,
    seeds: Sequence[int],
    device: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    families = {spec.key: spec.family for spec in REGISTRY.all_specs()}
    encoder = StateEncoder(families)
    surrogate = SurrogateModel(AnalyticPrior(), cost_reference_s=config.models.cost_reference_s)

    for task_type in sorted({entry.task_type for entry in entries}, key=lambda task: task.value):
        action_space = build_action_space_for(task_type, config)
        agent = _load_agent(task_type, config, action_space, encoder, device)
        factories = _searcher_factories(agent)
        for entry in [item for item in entries if item.task_type is task_type]:
            context = {
                "config": config,
                "action_space": action_space,
                "encoder": encoder,
                "surrogate": surrogate,
                "max_experiments": budget,
            }
            for seed in seeds:
                for factory in factories.values():
                    env = make_environment(entry, seed=seed, **context)
                    result: SearchResult = factory(seed).search(env, action_space, budget=budget)
                    rows.append(
                        {
                            "environment": "simulated",
                            "dataset": entry.name,
                            "task_type": task_type.value,
                            "seed": seed,
                            **result.to_dict(),
                        }
                    )
    return {
        "environment": "simulated",
        "warning": "surrogate estimates; not measured model-training performance",
        "budget": budget,
        "rows": rows,
        "summary": _summary(rows),
    }


def _real_environment(
    entry: CorpusEntry,
    *,
    config: AutoMLConfig,
    action_space: ActionSpace,
    encoder: StateEncoder,
    budget: int,
    seed: int,
    searcher: str,
) -> AutoMLEnv:
    source = get_source(entry.name, cache_dir=config.remote_dataset_dir())
    frame = source.load()
    task = TaskSpec(
        learning_type=entry.task_type.learning_type,
        task_type=entry.task_type,
        objective=f"benchmark {entry.name}",
        target=entry.target,
        metric=entry.metric,
        metric_direction=direction_for(entry.metric),
        confidence=1.0,
        source="benchmark",
    )
    tracker = SearchTracker(
        reward_calculator=RewardCalculator(
            config.rl.reward,
            entry.metric_direction,
            max_experiments=budget,
            patience=config.search.patience,
        ),
        max_experiments=budget,
        n_possible_models=len(action_space.layout.model_keys),
    )
    return AutoMLEnv(
        task=task,
        frame=frame,
        config=config,
        action_space=action_space,
        encoder=encoder,
        tracker=tracker,
        run_id=f"benchmark_{entry.name}_{searcher}_{seed}",
        env_config=RealEnvConfig(
            max_experiments=budget,
            min_experiments_before_stop=config.search.min_experiments,
            time_budget_s=config.search.time_budget_s,
            memory_budget_mb=config.search.memory_budget_mb,
        ),
        fingerprint=entry.fingerprint,
    )


def benchmark_real(
    entries: list[CorpusEntry],
    *,
    config: AutoMLConfig,
    budget: int,
    seeds: Sequence[int],
    device: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    families = {spec.key: spec.family for spec in REGISTRY.all_specs()}
    encoder = StateEncoder(families)
    for task_type in sorted({entry.task_type for entry in entries}, key=lambda task: task.value):
        action_space = build_action_space_for(task_type, config)
        agent = _load_agent(task_type, config, action_space, encoder, device)
        factories = _searcher_factories(agent)
        for entry in [item for item in entries if item.task_type is task_type]:
            for seed in seeds:
                for name, factory in factories.items():
                    env = _real_environment(
                        entry,
                        config=config,
                        action_space=action_space,
                        encoder=encoder,
                        budget=budget,
                        seed=seed,
                        searcher=name,
                    )
                    result = factory(seed).search(env, action_space, budget=budget)
                    rows.append(
                        {
                            "environment": "real",
                            "dataset": entry.name,
                            "task_type": task_type.value,
                            "seed": seed,
                            **result.to_dict(),
                        }
                    )
    return {
        "environment": "real",
        "budget": budget,
        "rows": rows,
        "summary": _summary(rows),
    }


def run_benchmark(
    config: AutoMLConfig,
    *,
    datasets: Sequence[str] | None = None,
    task_types: Sequence[TaskType] | None = None,
    budget: int = 6,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    real: bool = False,
    approved: bool = False,
    device: str | None = None,
    include_remote: bool = False,
) -> dict[str, Any]:
    if real and not approved:
        raise PermissionError("real benchmark training requires explicit approval")
    corpus = build_corpus(
        config, task_types=task_types, only=datasets, include_remote=include_remote
    )
    _train, holdout = split_corpus(corpus)
    evaluation_entries = holdout or corpus
    simulated = benchmark_simulated(
        evaluation_entries,
        config=config,
        budget=budget,
        seeds=seeds,
        device=device,
    )
    report: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "datasets": [entry.to_dict() for entry in evaluation_entries],
        "simulated": simulated,
        "real": None,
    }
    if real:
        report["real"] = benchmark_real(
            evaluation_entries,
            config=config,
            budget=budget,
            seeds=seeds,
            device=device,
        )
    destination = config.artifacts_dir() / "benchmarks" / "benchmark.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(destination)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PPO against matched search baselines.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--task-type", action="append", choices=[task.value for task in TaskType])
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--remote",
        action="store_true",
        help=(
            "include curated public datasets that are already cached "
            "(fetch them first with: automl datasets --download all)"
        ),
    )
    parser.add_argument("--real", action="store_true", help="also train models for real")
    parser.add_argument("--yes", action="store_true", help="explicitly approve real model training")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.real and not arguments.yes:
        print("error: --real requires --yes because it trains models", file=sys.stderr)
        return 2
    config = AutoMLConfig.load(arguments.config)
    try:
        report = run_benchmark(
            config,
            datasets=arguments.datasets,
            task_types=(
                [TaskType(value) for value in arguments.task_type] if arguments.task_type else None
            ),
            budget=arguments.budget,
            seeds=arguments.seeds,
            real=arguments.real,
            approved=arguments.yes,
            device=arguments.device,
            include_remote=arguments.remote,
        )
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for environment in ("simulated", "real"):
        result = report.get(environment)
        if not result:
            continue
        print(f"\n== {environment} ==")
        if environment == "simulated":
            print("SURROGATE ESTIMATES - NOT MEASURED PERFORMANCE")
        for name, summary in result["summary"]["searchers"].items():
            low, high = summary["ci95"]
            print(
                f"  {name:<8} mean={summary['mean']:.4f} "
                f"95% CI=[{low:.4f}, {high:.4f}] n={summary['n']}"
            )
        for name, comparison in result["summary"]["comparisons"].items():
            print(
                f"  {name}: delta={comparison['mean_difference']} "
                f"p={comparison['p_value_paired_permutation']}"
            )
    print(f"\nreport: {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "benchmark_real",
    "benchmark_simulated",
    "run_benchmark",
]
