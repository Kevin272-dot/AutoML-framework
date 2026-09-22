"""Intelligent stopping (spec §25).

The agent can choose ``STOP`` itself; this module is the independent, auditable second
opinion. It exists because a learned policy can be over-optimistic about the value of one
more experiment, and because "we ran out of budget" and "we decided further search was
pointless" are different outcomes that the run report should distinguish.

Important asymmetry, straight from the spec: stopping here governs *search*. It never
cancels the explicitly approved pipelines -- the execution phase always runs what the user
authorised.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MIN_NOVELTY = 0.05


@dataclass
class StoppingDecision:
    should_stop: bool
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"should_stop": self.should_stop, "reason": self.reason, "detail": self.detail}


class StoppingPolicy:
    """Decides when another experiment is no longer worth its cost."""

    def __init__(
        self,
        *,
        patience: int = 4,
        min_improvement: float = 0.001,
        min_experiments: int = 3,
        max_experiments: int = 12,
        target_score: float | None = None,
        value_threshold: float = 0.0,
    ) -> None:
        self.patience = max(1, patience)
        self.min_improvement = float(min_improvement)
        self.min_experiments = max(1, min_experiments)
        self.max_experiments = max(1, max_experiments)
        self.target_score = target_score
        self.value_threshold = float(value_threshold)

    def evaluate(
        self,
        *,
        n_experiments: int,
        recent_scores: list[float] | None = None,
        budget_fraction: float = 0.0,
        elapsed_fraction: float = 0.0,
        mean_novelty: float = 1.0,
        expected_value_fn: Callable[[], float] | None = None,
        oriented: bool = True,
    ) -> StoppingDecision:
        """Evaluate the stopping criteria in order of how conclusive they are."""
        detail: dict[str, Any] = {
            "n_experiments": n_experiments,
            "budget_fraction": round(budget_fraction, 4),
            "elapsed_fraction": round(elapsed_fraction, 4),
            "mean_novelty": round(mean_novelty, 4),
        }

        if n_experiments >= self.max_experiments:
            return StoppingDecision(True, "experiment budget exhausted", detail)

        if budget_fraction >= 1.0:
            return StoppingDecision(True, "search budget exhausted", detail)
        if elapsed_fraction >= 1.0:
            return StoppingDecision(True, "time budget exhausted", detail)

        scores = [score for score in (recent_scores or []) if score is not None]
        if scores:
            best = max(scores) if oriented else min(scores)
            detail["best_recent_score"] = round(best, 5)
            if self.target_score is not None:
                reached = best >= self.target_score if oriented else best <= self.target_score
                if reached:
                    return StoppingDecision(True, "target score reached", detail)

        # Never stop before the minimum: the planner must have something to recommend.
        if n_experiments < self.min_experiments:
            detail["reason_not_stopping"] = "below the minimum experiment count"
            return StoppingDecision(False, "below minimum experiments", detail)

        if self._stagnated(scores):
            detail["stagnation_window"] = self.patience
            detail["min_improvement"] = self.min_improvement
            return StoppingDecision(True, "search has stagnated", detail)

        if mean_novelty < MIN_NOVELTY:
            return StoppingDecision(True, "experiment space is exhausted", detail)

        if expected_value_fn is not None:
            value = float(expected_value_fn())
            detail["expected_value"] = round(value, 5)
            detail["value_threshold"] = self.value_threshold
            if value < self.value_threshold:
                return StoppingDecision(
                    True, "expected value of another experiment is below the threshold", detail
                )

        return StoppingDecision(False, "continuing", detail)

    def _stagnated(self, scores: list[float]) -> bool:
        """True when the best score has not moved by ``min_improvement`` for ``patience`` steps."""
        if len(scores) <= self.patience:
            return False
        recent = scores[-self.patience :]
        reference = max(scores[: -self.patience])
        return (max(recent) - reference) < self.min_improvement


def improvement_slope(scores: list[float]) -> float:
    """Least-squares slope of the score trace. Negative means the search is getting worse."""
    values = [score for score in scores if score is not None and np.isfinite(score)]
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def expected_value_of_search(
    *,
    best_score: float | None,
    expected_gain: float,
    expected_cost_s: float,
    cost_weight: float = 0.05,
) -> float:
    """A crude expected-value estimate used as the last stopping criterion.

    ``expected_gain`` is supplied by the caller (the surrogate's ceiling minus the current
    best); the cost term subtracts the predicted training time scaled by ``cost_weight``.
    """
    if best_score is None:
        return float("inf")
    return float(expected_gain - cost_weight * max(expected_cost_s, 0.0))


__all__ = [
    "StoppingDecision",
    "StoppingPolicy",
    "expected_value_of_search",
    "improvement_slope",
]
