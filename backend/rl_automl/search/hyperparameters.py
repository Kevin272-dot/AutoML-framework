"""Hyperparameter presets and post-selection refinement (spec §7).

Two responsibilities:

* **Preset validation** - the RL action space selects a preset *index*, so the registry's
  preset tables are the finite set of hyperparameter configurations the agent can choose
  between. This module provides the guardrails those tables must satisfy.
* **Refinement** - once a winning configuration is known, a small budgeted randomized
  search around it can beat the nearest preset without re-opening the action space. This
  is the bridge between a finite, learnable action space and continuous hyperparameters.

Refinement runs *after* selection and only when explicitly enabled
(``search.post_selection_refinement``), so it never inflates the cost of a run the user did
not ask for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_automl.core.config import SearchConfig
from rl_automl.core.logging import get_logger
from rl_automl.core.types import ExperimentSource, ExperimentSpec
from rl_automl.search.model_registry import ModelSpec, get_spec

logger = get_logger("search.hyperparameters")

#: name -> (kind, low, high, spacing). Sampling is done in log space where the spacing is
#: "log", because that is how these hyperparameters actually behave.
NUMERIC_RANGES: dict[str, tuple[str, float, float, str]] = {
    "n_estimators": ("int", 100, 2000, "log"),
    "learning_rate": ("float", 0.005, 0.3, "log"),
    "max_depth": ("int", 3, 32, "linear"),
    "min_samples_leaf": ("int", 1, 20, "log"),
    "min_samples_split": ("int", 2, 40, "log"),
    "min_child_samples": ("int", 5, 100, "log"),
    "min_child_weight": ("float", 0.5, 20.0, "log"),
    "num_leaves": ("int", 8, 256, "log"),
    "subsample": ("float", 0.5, 1.0, "linear"),
    "colsample_bytree": ("float", 0.4, 1.0, "linear"),
    "reg_lambda": ("float", 0.0, 20.0, "linear"),
    "reg_alpha": ("float", 0.0, 10.0, "linear"),
    "C": ("float", 0.01, 50.0, "log"),
    "max_iter": ("int", 500, 5000, "log"),
    "max_epochs": ("int", 20, 200, "linear"),
    "dropout": ("float", 0.0, 0.5, "linear"),
    "batch_size": ("int", 64, 1024, "log"),
    "n_clusters": ("int", 2, 24, "linear"),
    "eps": ("float", 0.1, 1.5, "linear"),
    "min_samples": ("int", 2, 30, "log"),
    "n_neighbors": ("int", 5, 60, "linear"),
    "nu": ("float", 0.01, 0.5, "linear"),
    "epsilon": ("float", 0.01, 1.0, "log"),
}

#: Parameters that are structural rather than numeric and are never perturbed.
NON_NUMERIC_PARAMS = frozenset(
    {
        "kernel",
        "penalty",
        "solver",
        "linkage",
        "covariance_type",
        "contamination",
        "hidden_sizes",
        "n_components",
        "n_init",
        "random_state",
        "n_jobs",
        "verbose",
        "novelty",
        "gamma",
    }
)


@dataclass
class ValidationReport:
    model: str
    ok: bool
    problems: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "ok": self.ok, "problems": list(self.problems)}


def validate_presets(spec: ModelSpec) -> ValidationReport:
    """Check a registry entry's preset table is usable by the RL action space."""
    problems: list[str] = []
    if not spec.presets:
        problems.append("no presets declared")

    seen: set[str] = set()
    for preset in spec.presets:
        if preset.name in seen:
            problems.append(f"duplicate preset name '{preset.name}'")
        seen.add(preset.name)
        if not isinstance(preset.params, dict):
            problems.append(f"preset '{preset.name}' params must be a dict")
        for key in preset.params:
            if key in NON_NUMERIC_PARAMS:
                continue
            if key not in NUMERIC_RANGES:
                # Not fatal: refinement simply will not touch this parameter.
                continue

    return ValidationReport(model=spec.key, ok=not problems, problems=problems)


class HyperparameterRefiner:
    """Randomized local search around a winning preset."""

    def __init__(self, config: SearchConfig | None = None, seed: int = 0) -> None:
        self.config = config or SearchConfig()
        self.seed = seed

    def refine(
        self,
        base_experiment: ExperimentSpec,
        *,
        trials: int | None = None,
        seed: int | None = None,
    ) -> list[ExperimentSpec]:
        """Propose perturbations of ``base_experiment``.

        Deterministic for a given seed, so a refined run is reproducible.
        """
        spec = get_spec(base_experiment.model)
        trials = int(trials or self.config.refinement_trials)
        if trials <= 0:
            return []

        rng = np.random.default_rng(self.seed if seed is None else seed)
        candidates: list[ExperimentSpec] = []
        seen: set[str] = set()

        for _ in range(trials):
            params = dict(base_experiment.hyperparameters)
            perturbation = self._perturb(params, rng)
            if not perturbation:
                continue

            candidate = ExperimentSpec(
                model=base_experiment.model,
                preprocessing=list(base_experiment.preprocessing),
                feature_selection=base_experiment.feature_selection,
                preset_index=base_experiment.preset_index,
                hyperparameters=params,
                expected_cost=base_experiment.expected_cost,
                source=ExperimentSource.RL,
                rationale="refinement: " + ", ".join(perturbation),
            )
            signature = candidate.signature()
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(candidate)

        logger.info(
            "refinement candidates generated",
            extra={
                "context": {
                    "model": spec.key,
                    "trials_requested": trials,
                    "generated": len(candidates),
                }
            },
        )
        return candidates

    def _perturb(self, params: dict[str, Any], rng: np.random.Generator) -> list[str]:
        """Perturb 1-3 numeric parameters in place and describe what changed."""
        adjustable = [
            key
            for key, value in params.items()
            if key not in NON_NUMERIC_PARAMS
            and (key in NUMERIC_RANGES or isinstance(value, int | float))
        ]
        if not adjustable:
            return []

        n_changes = int(rng.integers(1, min(3, len(adjustable)) + 1))
        chosen = rng.choice(adjustable, size=n_changes, replace=False)
        changes: list[str] = []

        for key in chosen:
            key = str(key)
            current = params.get(key)
            range_spec = NUMERIC_RANGES.get(key)
            new_value = _perturb_value(key, current, range_spec, rng)
            if new_value is None or new_value == current:
                continue
            params[key] = new_value
            changes.append(f"{key} {current}->{new_value}")

        return changes


def _perturb_value(
    key: str,
    current: Any,
    range_spec: tuple[str, float, float, str] | None,
    rng: np.random.Generator,
) -> Any:
    """Nudge one hyperparameter, staying near ``current``."""
    if range_spec is None:
        if not isinstance(current, int | float):
            return None
        if isinstance(current, bool):
            return None
        # Unknown numeric parameter: multiplicative jitter with a sane floor.
        factor = float(np.clip(rng.normal(1.0, 0.25), 0.4, 2.5))
        value = current * factor
        return int(max(1, round(value))) if isinstance(current, int) else round(float(value), 6)

    kind, low, high, spacing = range_spec

    if current is None:
        # e.g. max_depth=None means unlimited; sampling a finite depth is the useful move.
        base = high if kind == "int" else (low + high) / 2.0
    else:
        base = float(current)

    if spacing == "log":
        base = max(base, low)
        factor = float(np.exp(rng.normal(0.0, 0.45)))
        value = base * factor
    else:
        span = high - low
        value = base + float(rng.normal(0.0, 0.18 * span))

    value = float(np.clip(value, low, high))
    if kind == "int":
        return round(value)
    return round(value, 6)


def describe_preset(model_key: str, preset_index: int) -> dict[str, Any]:
    """Human-readable preset description, used in the recommendation reasons."""
    preset = get_spec(model_key).preset(preset_index)
    return {
        "name": preset.name,
        "params": dict(preset.params),
        "cost": preset.cost.value,
        "description": preset.description,
    }


__all__ = [
    "NON_NUMERIC_PARAMS",
    "NUMERIC_RANGES",
    "HyperparameterRefiner",
    "ValidationReport",
    "describe_preset",
    "validate_presets",
]
