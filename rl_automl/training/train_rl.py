"""Pretrain the PPO planner across the bundled and synthetic corpus (spec §24, §31C).

What this trains against, stated plainly: the **surrogate**, not real training runs. That is
what makes thousands of episodes affordable, and it is also the reason a trained policy is
not, by itself, evidence of anything. The honest comparison against random search and an
untrained policy is printed and written to disk alongside the checkpoint, and it is labelled
``simulated`` everywhere it appears.

One checkpoint per task type. A policy is tied to one action space, and the action space
depends on the task type, so a single file cannot serve classification and regression at
once. Each checkpoint is written to ``ppo_policy_<task>.pt`` next to the configured policy
path, and the orchestrator resolves the task-specific file first.

Corpus datasets are used with the task type and metric they *declare*, not with the task
inferred from a problem statement. Deployment infers the task, so a misclassified run
simply fails the checkpoint's metadata guard and falls back to the surrogate planner --
which is the designed behaviour, and better than training against a guess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_automl.agent.agent import PPOAgent
from rl_automl.core.config import AutoMLConfig
from rl_automl.core.logging import get_logger
from rl_automl.core.metrics import direction_for, primary_metric_for
from rl_automl.core.seeding import seed_everything
from rl_automl.core.types import (
    DatasetFingerprint,
    MetricDirection,
    TaskType,
    utc_now_iso,
)
from rl_automl.dataset import remote
from rl_automl.dataset.fingerprint import build_fingerprint
from rl_automl.dataset.loaders import get_source, synthetic_recipes, synthetic_source
from rl_automl.dataset.profiler import profile_dataset
from rl_automl.environment.action_space import ActionSpace, build_action_space
from rl_automl.environment.base import SearchTracker
from rl_automl.environment.reward import RewardCalculator
from rl_automl.environment.simulator import SimulatorConfig, SimulatorEnv
from rl_automl.environment.state_encoder import StateEncoder
from rl_automl.environment.surrogate import AnalyticPrior, SurrogateModel
from rl_automl.search.model_registry import REGISTRY

logger = get_logger("training.train_rl")

#: Bundled scikit-learn datasets used for pretraining, grouped by their declared task.
BUNDLED_CORPUS: dict[TaskType, tuple[str, ...]] = {
    TaskType.CLASSIFICATION: (
        "iris",
        "breast_cancer",
        "wine",
        "digits",
        "churn_demo_small",
        "churn_demo",
    ),
    TaskType.REGRESSION: ("diabetes", "california_housing"),
}

#: Fraction of each task type's corpus reserved for the honest comparison.
HOLDOUT_FRACTION = 0.34
QUICK_STEPS = 4_000
EVAL_SEEDS = (101, 202, 303)

#: How often a long run reports progress and writes a durable checkpoint. Progress is
#: time-based on purpose: printing every N steps means a slow run shows nothing for hours.
PROGRESS_INTERVAL_S = 30.0
CHECKPOINT_INTERVAL_S = 900.0


def _format_duration(seconds: float) -> str:
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


@dataclass
class CorpusEntry:
    """One dataset, reduced to what the surrogate environment needs."""

    name: str
    task_type: TaskType
    target: str | None
    metric: str
    metric_direction: MetricDirection
    n_rows: int
    n_cols: int
    fingerprint: DatasetFingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_type": self.task_type.value,
            "target": self.target,
            "metric": self.metric,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
        }


# --------------------------------------------------------------------------------------
# Corpus construction
# --------------------------------------------------------------------------------------


def build_corpus(
    config: AutoMLConfig,
    *,
    task_types: Iterable[TaskType] | None = None,
    only: Sequence[str] | None = None,
    include_unsupervised: bool = False,
    include_remote: bool = False,
) -> list[CorpusEntry]:
    """Load and profile every corpus dataset that matches the request.

    ``include_remote`` adds the curated public datasets that are **already cached**. It
    never triggers a download: fetching is an explicit act (``automl datasets --download``),
    so building a corpus stays offline and reproducible.
    """
    wanted = set(task_types) if task_types else None
    wanted_names = set(only) if only else None
    entries: list[CorpusEntry] = []
    cache_dir = config.remote_dataset_dir()

    for task_type, names in BUNDLED_CORPUS.items():
        if wanted is not None and task_type not in wanted:
            continue
        for name in names:
            if wanted_names is not None and name not in wanted_names:
                continue
            entries.extend(_entry_from_source(name, config, expected=task_type))

    if include_remote:
        curated = [
            name
            for name in remote.remote_names()
            if remote.is_cached(name, cache_dir)
            and (wanted_names is None or name in wanted_names)
        ]
        for name in curated:
            spec = remote.REMOTE_DATASETS[name]
            if wanted is not None and spec.task_type not in wanted:
                continue
            entries.extend(_entry_from_source(name, config, expected=spec.task_type))

    for recipe in synthetic_recipes():
        if not include_unsupervised and not recipe.task_type.is_supervised:
            continue
        if wanted is not None and recipe.task_type not in wanted:
            continue
        if wanted_names is not None and recipe.name not in wanted_names:
            continue
        entries.extend(
            _entry_from_source(
                recipe.name,
                config,
                expected=recipe.task_type,
                source=synthetic_source(recipe),
            )
        )

    if not entries:
        raise ValueError("no corpus datasets matched the request")

    logger.info(
        "corpus built",
        extra={
            "context": {
                "datasets": len(entries),
                "tasks": sorted({entry.task_type.value for entry in entries}),
            }
        },
    )
    return entries


def _entry_from_source(
    name: str,
    config: AutoMLConfig,
    *,
    expected: TaskType,
    source: Any | None = None,
) -> list[CorpusEntry]:
    try:
        source = source or get_source(name, cache_dir=config.remote_dataset_dir())
        frame = source.load()
    except Exception as exc:  # pragma: no cover - a missing optional dependency
        logger.warning(
            "skipping corpus dataset",
            extra={"context": {"dataset": name, "error": f"{type(exc).__name__}: {exc}"}},
        )
        return []

    if source.task_type is not expected:  # pragma: no cover - guards a registry edit
        logger.warning(
            "corpus dataset declares an unexpected task type",
            extra={"context": {"dataset": name, "declared": source.task_type.value}},
        )
        return []

    # primary_metric_for is the authoritative registry; core.types also carries a default
    # table, but it disagrees for anomaly detection and this is the only place either is
    # consulted at runtime.
    metric = source.metric or primary_metric_for(source.task_type)
    profile = profile_dataset(frame, source.target, config.dataset)
    return [
        CorpusEntry(
            name=name,
            task_type=source.task_type,
            target=source.target,
            metric=metric,
            metric_direction=direction_for(metric),
            n_rows=int(frame.shape[0]),
            n_cols=int(frame.shape[1]),
            fingerprint=build_fingerprint(profile, source.task_type),
        )
    ]


def split_corpus(entries: list[CorpusEntry]) -> tuple[list[CorpusEntry], list[CorpusEntry]]:
    """Split each task type's datasets so the evaluation never trains on its own test set.

    Datasets are ordered deterministically, and the smallest slice that still leaves
    something to train on is held out, because a corpus of two datasets must still be able
    to demonstrate the comparison honestly rather than reporting a training score.
    """
    train: list[CorpusEntry] = []
    holdout: list[CorpusEntry] = []
    by_task: dict[TaskType, list[CorpusEntry]] = {}
    for entry in entries:
        by_task.setdefault(entry.task_type, []).append(entry)

    for task_type in sorted(by_task, key=lambda item: item.value):
        group = sorted(by_task[task_type], key=lambda item: item.name)
        n_holdout = max(1, round(len(group) * HOLDOUT_FRACTION))
        n_holdout = min(n_holdout, len(group) - 1) if len(group) > 1 else 1
        holdout.extend(group[-n_holdout:])
        train.extend(group[: len(group) - n_holdout] or group[:1])
    return train, holdout


# --------------------------------------------------------------------------------------
# Environments
# --------------------------------------------------------------------------------------


def build_action_space_for(task_type: TaskType, config: AutoMLConfig) -> ActionSpace:
    return build_action_space(
        task_type,
        config.models.enabled,
        min_experiments_before_stop=config.search.min_experiments,
    )


def make_environment(
    entry: CorpusEntry,
    *,
    config: AutoMLConfig,
    action_space: ActionSpace,
    encoder: StateEncoder,
    surrogate: SurrogateModel,
    seed: int,
    max_experiments: int,
) -> SimulatorEnv:
    tracker = SearchTracker(
        reward_calculator=RewardCalculator(
            config.rl.reward,
            entry.metric_direction,
            max_experiments=max_experiments,
            patience=config.search.patience,
        ),
        max_experiments=max_experiments,
        n_possible_models=len(action_space.layout.model_keys),
    )
    return SimulatorEnv(
        task_type=entry.task_type,
        fingerprint=entry.fingerprint,
        action_space=action_space,
        encoder=encoder,
        tracker=tracker,
        surrogate=surrogate,
        config=SimulatorConfig(
            max_experiments=max_experiments,
            min_experiments_before_stop=config.search.min_experiments,
            seed=seed,
        ),
        n_rows=entry.n_rows,
        n_cols=entry.n_cols,
    )


# --------------------------------------------------------------------------------------
# Evaluation (surrogate only -- this is not the spec's benchmark)
# --------------------------------------------------------------------------------------


def _rollout(
    entry: CorpusEntry,
    context: dict[str, Any],
    *,
    agent: PPOAgent | None,
    seed: int,
) -> float | None:
    env = make_environment(entry, seed=seed, **context)
    state = env.reset()
    rng = np.random.default_rng(seed)
    for _ in range(context["max_experiments"]):
        if agent is None:
            action = context["action_space"].sample_random_action(env.mask_table, rng)
        else:
            action, _log_prob, _value = agent.act(state, env.mask_table, deterministic=True)
        result = env.step(action)
        state = result.state
        if result.done:
            break
    return env.best_score


def compare_planners(
    holdout: list[CorpusEntry],
    *,
    trained: PPOAgent,
    untrained: PPOAgent,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Trained vs untrained vs random on held-out datasets. Surrogate scores, not measurements."""
    per_dataset: list[dict[str, Any]] = []
    for entry in holdout:
        trained_scores: list[float] = []
        untrained_scores: list[float] = []
        random_scores: list[float] = []
        for seed in EVAL_SEEDS:
            dataset_seed = (
                int.from_bytes(hashlib.sha256(entry.name.encode("utf-8")).digest()[:4], "big")
                % 1000
            )
            for agent, sink in ((trained, trained_scores), (untrained, untrained_scores)):
                score = _rollout(entry, context, agent=agent, seed=seed + dataset_seed)
                if score is not None:
                    sink.append(score)
            score = _rollout(entry, context, agent=None, seed=seed + dataset_seed)
            if score is not None:
                random_scores.append(score)

        def _mean(values: list[float]) -> float | None:
            return float(np.mean(values)) if values else None

        per_dataset.append(
            {
                **entry.to_dict(),
                "trained": _mean(trained_scores),
                "untrained": _mean(untrained_scores),
                "random": _mean(random_scores),
            }
        )

    def _overall(key: str) -> float | None:
        values = [row[key] for row in per_dataset if row[key] is not None]
        return float(np.mean(values)) if values else None

    trained_mean = _overall("trained")
    random_mean = _overall("random")
    return {
        "environment": "simulated",
        "note": (
            "surrogate estimates, not measurements; the RL-vs-baselines study that mixes in "
            "real training runs belongs to evaluation.benchmark"
        ),
        "seeds_per_dataset": list(EVAL_SEEDS),
        "n_holdout_datasets": len(per_dataset),
        "trained_mean": trained_mean,
        "untrained_mean": _overall("untrained"),
        "random_mean": random_mean,
        "beats_random": (
            None
            if trained_mean is None or random_mean is None
            else bool(trained_mean > random_mean)
        ),
        "per_dataset": per_dataset,
    }


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def train_task_type(
    task_type: TaskType,
    train_entries: list[CorpusEntry],
    holdout_entries: list[CorpusEntry],
    *,
    config: AutoMLConfig,
    steps: int,
    device: str | None,
    seed: int,
    verbose: bool,
    resume: bool = True,
) -> dict[str, Any]:
    families = {spec.key: spec.family for spec in REGISTRY.all_specs()}
    encoder = StateEncoder(families)
    action_space = build_action_space_for(task_type, config)
    surrogate = SurrogateModel(AnalyticPrior(), cost_reference_s=config.models.cost_reference_s)
    max_experiments = min(config.search.max_experiments, config.rl.max_episode_steps)

    context = {
        "config": config,
        "action_space": action_space,
        "encoder": encoder,
        "surrogate": surrogate,
        "max_experiments": max_experiments,
    }

    destination = config.policy_destination(task_type)
    resumed_from_steps = 0
    if resume and destination.is_file():
        agent, _extra = PPOAgent.load(
            destination,
            action_space=action_space,
            state_encoder=encoder,
            rl_config=config.rl,
            device=device,
            strict=True,
        )
        resumed_from_steps = agent.metadata.trained_steps
    else:
        agent = PPOAgent.from_action_space(
            action_space, encoder.dim, config.rl, device=device, seed=seed
        )
    untrained = PPOAgent.from_action_space(
        action_space, encoder.dim, config.rl, device=device, seed=seed + 1
    )

    if verbose:
        print(
            f"  action space: {action_space.layout.n_model_slots} model slot(s), "
            f"state dim {encoder.dim}, params {agent.describe()['n_parameters']:,}",
            flush=True,
        )
        if resumed_from_steps:
            print(f"  resuming checkpoint after {resumed_from_steps:,} prior step(s)", flush=True)
        print(
            f"  training on {len(train_entries)} dataset(s), "
            f"checkpoint comparison on {len(holdout_entries)}",
            flush=True,
        )

    # Progress is reported on a *time* basis, not every N steps. A multi-hour run that
    # prints nothing until it finishes is indistinguishable from a hung one, and the first
    # version of this file made exactly that mistake.
    started_at = time.perf_counter()
    last_report_at = started_at
    last_checkpoint_at = started_at

    progress_path = destination.with_name(f"{destination.stem}.progress.json")

    def report_update(step: int, stats: Any, _report: Any) -> None:
        nonlocal last_report_at, last_checkpoint_at
        now = time.perf_counter()
        elapsed = now - started_at
        if step >= steps:
            return
        if now - last_report_at >= PROGRESS_INTERVAL_S:
            rate = step / elapsed if elapsed > 0 else 0.0
            remaining = (steps - step) / rate if rate > 0 else 0.0
            if verbose:
                print(
                    f"    step {step:>8,}/{steps:,}  loss {stats.total_loss:8.4f}  "
                    f"entropy {stats.entropy:6.4f}  {rate:7.1f} steps/s  "
                    f"elapsed {_format_duration(elapsed)}  eta {_format_duration(remaining)}",
                    flush=True,
                )
            # Also written to disk: stdout is block-buffered when redirected, and a long run
            # must be observable no matter how it was launched.
            progress_path.write_text(
                json.dumps(
                    {
                        "task_type": task_type.value,
                        "step": step,
                        "steps": steps,
                        "steps_per_second": round(rate, 3),
                        "elapsed_s": round(elapsed, 1),
                        "eta_s": round(remaining, 1),
                        "loss": round(float(stats.total_loss), 6),
                        "entropy": round(float(stats.entropy), 6),
                        "updated_at": utc_now_iso(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            last_report_at = now
        # Durable progress: a long run should not lose everything if it is interrupted.
        if now - last_checkpoint_at >= CHECKPOINT_INTERVAL_S:
            agent.save(destination, extra={"partial": True, "saved_at": utc_now_iso()})
            last_checkpoint_at = now
            if verbose:
                print(f"    (checkpoint saved after {step:,} steps)", flush=True)

    envs = [
        make_environment(entry, seed=1000 + index, **context)
        for index, entry in enumerate(train_entries)
    ]
    try:
        training = agent.train(envs, total_steps=steps, on_update=report_update)
    except KeyboardInterrupt:
        agent.save(destination, extra={"partial": True, "interrupted_at": utc_now_iso()})
        if verbose:
            print(
                f"  interrupted after {agent.metadata.trained_steps:,} step(s); "
                f"partial checkpoint saved to {destination}",
                flush=True,
            )
        raise

    evaluation = compare_planners(
        holdout_entries, trained=agent, untrained=untrained, context=context
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    agent.save(
        destination,
        extra={
            "reward": config.rl.reward.model_dump(mode="json"),
            "trained_at": utc_now_iso(),
            "corpus": [entry.to_dict() for entry in train_entries],
            "holdout": [entry.to_dict() for entry in holdout_entries],
            "evaluation": evaluation,
            "simulated": True,
        },
    )

    return {
        "task_type": task_type.value,
        "checkpoint": str(destination),
        "resumed_from_steps": resumed_from_steps,
        "trained_steps_total": agent.metadata.trained_steps,
        "training": training.to_dict(),
        "corpus": [entry.to_dict() for entry in train_entries],
        "holdout": [entry.to_dict() for entry in holdout_entries],
        "evaluation": evaluation,
    }


def run_training(
    config: AutoMLConfig,
    *,
    steps: int | None = None,
    task_types: Sequence[TaskType] | None = None,
    only: Sequence[str] | None = None,
    device: str | None = None,
    seed: int | None = None,
    verbose: bool = True,
    resume: bool = True,
    include_remote: bool = False,
) -> dict[str, Any]:
    """Train one policy per task type. Returns the full report."""
    resolved_seed = config.runtime.seed if seed is None else seed
    resolved_steps = int(steps or config.rl.total_steps)
    seed_everything(resolved_seed)

    entries = build_corpus(
        config, task_types=task_types, only=only, include_remote=include_remote
    )
    train_entries, holdout_entries = split_corpus(entries)

    results: list[dict[str, Any]] = []
    for task_type in sorted({entry.task_type for entry in train_entries}, key=lambda t: t.value):
        train_group = [entry for entry in train_entries if entry.task_type is task_type]
        holdout_group = [entry for entry in holdout_entries if entry.task_type is task_type]
        if verbose:
            print(f"\n== {task_type.value} ==")
        results.append(
            train_task_type(
                task_type,
                train_group,
                holdout_group,
                config=config,
                steps=resolved_steps,
                device=device,
                seed=resolved_seed,
                verbose=verbose,
                resume=resume,
            )
        )

    report = {
        "trained_at": utc_now_iso(),
        "environment": "simulated (surrogate)",
        "steps_requested": resolved_steps,
        "seed": resolved_seed,
        "device": device or config.rl.device,
        "results": results,
    }

    report_path = config.policy_path().parent / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if verbose:
        print(f"\nreport written to {report_path}")
    return report


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train-rl",
        description=(
            "Pretrain the PPO planner against the surrogate. The checkpoint is only useful "
            "alongside a comparison, so one is printed and saved with it."
        ),
    )
    parser.add_argument("--config", default=None, help="path to a config YAML")
    parser.add_argument("--steps", type=int, default=None, help="environment steps per task type")
    parser.add_argument(
        "--task-type",
        action="append",
        dest="task_types",
        choices=[task.value for task in TaskType],
        help="restrict training to this task type (repeatable)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="restrict the corpus to these dataset names",
    )
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"short run for a smoke check ({QUICK_STEPS} steps, small corpus)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore compatible task-specific checkpoints and train from new weights",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help=(
            "include curated public datasets that are already cached "
            "(fetch them first with: automl datasets --download all)"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = AutoMLConfig.load(arguments.config)

    steps = arguments.steps
    only = arguments.datasets
    if arguments.quick:
        steps = steps or QUICK_STEPS
        only = only or [
            "iris",
            "wine",
            "churn_demo_small",
            "synth_clf_balanced_small",
            "diabetes",
            "synth_reg_small",
        ]

    task_types = (
        [TaskType(value) for value in arguments.task_types] if arguments.task_types else None
    )

    verbose = not arguments.quiet
    if verbose:
        print("RL pretraining against the SURROGATE environment (simulated outcomes).")
        print(
            "A trained policy is not evidence of superiority on its own; the comparison below is."
        )
        print(f"steps per task type: {steps or config.rl.total_steps:,}")

    try:
        report = run_training(
            config,
            steps=steps,
            task_types=task_types,
            only=only,
            device=arguments.device,
            seed=arguments.seed,
            verbose=verbose,
            resume=not arguments.fresh,
            include_remote=arguments.remote,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("\n=== summary (simulated) ===")
    for result in report["results"]:
        evaluation = result["evaluation"]
        training = result["training"]

        def _fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        print(f"\n{result['task_type']}:")
        print(f"  checkpoint      : {result['checkpoint']}")
        print(f"  steps / updates : {training['total_steps']:,} / {training['n_updates']}")
        print(f"  episodes        : {training['n_episodes']}")
        print(f"  elapsed         : {training['elapsed_s']}s")
        print(f"  holdout datasets: {evaluation['n_holdout_datasets']}")
        print(f"  trained  (sim)  : {_fmt(evaluation['trained_mean'])}")
        print(f"  untrained(sim)  : {_fmt(evaluation['untrained_mean'])}")
        print(f"  random   (sim)  : {_fmt(evaluation['random_mean'])}")
        print(f"  beats random    : {evaluation['beats_random']}")

    print(
        "\nThese are surrogate estimates. Reporting them as measured performance would be "
        "the single easiest way to make this project dishonest."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
