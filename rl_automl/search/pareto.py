"""Pareto analysis (spec §14, §22).

The comparison table answers "which model won". The Pareto frontier answers the more
useful question: "which models are not beaten on *every* axis at once". A model that is
2% worse but trains in a fifth of the time is not dominated, and hiding that trade-off
would defeat the purpose of the report (spec §14: "do not hide trade-offs").
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from rl_automl.core.metrics import orient
from rl_automl.core.types import (
    ExperimentResult,
    MetricDirection,
    ParetoPoint,
)


@dataclass
class ParetoAnalysis:
    frontier: list[ParetoPoint] = field(default_factory=list)
    dominated: list[ParetoPoint] = field(default_factory=list)

    @property
    def n_frontier(self) -> int:
        return len(self.frontier)

    def cheapest(self) -> ParetoPoint | None:
        return min(self.frontier, key=lambda point: point.cost_s) if self.frontier else None

    def best_score_point(self) -> ParetoPoint | None:
        return max(self.frontier, key=lambda point: point.score) if self.frontier else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_frontier": self.n_frontier,
            "n_dominated": len(self.dominated),
            "frontier": [point.model_dump(mode="json") for point in self.frontier],
        }


def compute_pareto(
    results: Iterable[ExperimentResult],
    direction: MetricDirection,
    *,
    include_failures: bool = False,
) -> ParetoAnalysis:
    """Non-dominated set over (score, training time, peak memory).

    A point is dominated when another point is at least as good on every objective and
    strictly better on at least one.
    """
    points: list[ParetoPoint] = []

    for result in results:
        if not result.succeeded and not include_failures:
            continue
        if result.validation_score is None:
            continue
        points.append(
            ParetoPoint(
                model=result.experiment.model,
                experiment_id=result.experiment.experiment_id,
                # Stored oriented so that "larger is better" holds for every objective.
                score=float(orient(result.validation_score, direction)),
                cost_s=float(max(result.training_time_s, 1e-9)),
                peak_memory_mb=float(max(result.peak_memory_mb, 0.0)),
                on_frontier=True,
            )
        )

    if not points:
        return ParetoAnalysis()

    frontier: list[ParetoPoint] = []
    dominated: list[ParetoPoint] = []
    for candidate in points:
        is_dominated = any(
            _dominates(other, candidate) for other in points if other is not candidate
        )
        candidate.on_frontier = not is_dominated
        (dominated if is_dominated else frontier).append(candidate)

    # Stable, readable ordering: strongest first.
    frontier.sort(key=lambda point: point.score, reverse=True)
    dominated.sort(key=lambda point: point.score, reverse=True)
    return ParetoAnalysis(frontier=frontier, dominated=dominated)


def _dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
    """True when ``left`` is no worse everywhere and better somewhere."""
    better_or_equal = (
        left.score >= right.score
        and left.cost_s <= right.cost_s
        and left.peak_memory_mb <= right.peak_memory_mb
    )
    strictly_better = (
        left.score > right.score
        or left.cost_s < right.cost_s
        or left.peak_memory_mb < right.peak_memory_mb
    )
    return better_or_equal and strictly_better


def efficiency_alternative(
    results: Iterable[ExperimentResult],
    direction: MetricDirection,
    *,
    reference_model: str,
    max_score_loss_fraction: float = 0.03,
) -> dict[str, Any] | None:
    """Find a materially cheaper model that is only slightly worse than the winner.

    This is what powers the "efficiency alternative" line in the final report (spec §15):
    the user should be told when 97% of the performance is available for a fraction of the
    compute.
    """
    candidates = [
        result for result in results if result.succeeded and result.validation_score is not None
    ]
    if len(candidates) < 2:
        return None

    reference = next(
        (result for result in candidates if result.experiment.model == reference_model), None
    )
    if reference is None or reference.validation_score is None:
        return None

    reference_oriented = orient(reference.validation_score, direction)
    best: tuple[float, ExperimentResult] | None = None

    for candidate in candidates:
        if candidate.experiment.experiment_id == reference.experiment.experiment_id:
            continue
        if candidate.validation_score is None or reference.training_time_s <= 0:
            continue
        if candidate.training_time_s >= reference.training_time_s:
            continue

        candidate_oriented = orient(candidate.validation_score, direction)
        # Only consider options that are genuinely close in quality.
        if abs(reference_oriented) > 1e-9:
            relative_loss = abs(reference_oriented - candidate_oriented) / abs(reference_oriented)
        else:
            relative_loss = 0.0
        if relative_loss > max_score_loss_fraction:
            continue

        speedup = reference.training_time_s / max(candidate.training_time_s, 1e-9)
        if best is None or speedup > best[0]:
            best = (speedup, candidate)

    if best is None:
        return None
    speedup, candidate = best
    return {
        "model": candidate.experiment.model,
        "experiment_id": candidate.experiment.experiment_id,
        "validation_score": candidate.validation_score,
        "test_score": candidate.test_score,
        "speedup": round(speedup, 2),
        "score_delta": round(
            float(candidate.validation_score) - float(reference.validation_score), 6
        ),
        "reason": (
            f"trains {speedup:.1f}x faster than {reference_model} with a validation "
            f"{reference.primary_metric} difference of "
            f"{abs(float(candidate.validation_score) - float(reference.validation_score)):.4f}"
        ),
    }


__all__ = ["ParetoAnalysis", "compute_pareto", "efficiency_alternative"]
