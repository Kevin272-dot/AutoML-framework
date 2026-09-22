"""Fair comparison and final selection (spec §13, §14, §15).

Two rules that the code enforces rather than merely documents:

* **Selection is made on validation only.** The reason text attached to the chosen model
  never cites the test score, because the test set was reserved for final reporting
  (spec §15). Test scores appear in the table for transparency and are labelled as held-out.
* **Trade-offs are surfaced, not hidden.** The summary always names the fastest, smallest
  and most compute-efficient model even when they are not the winner, and the chosen model
  is accompanied by a cheaper alternative when one exists within a small quality margin.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from rl_automl.core.logging import get_logger
from rl_automl.core.metrics import orient
from rl_automl.core.types import (
    BestModel,
    ComparisonRow,
    ComparisonSummary,
    EfficiencyAlternative,
    ExperimentResult,
    MetricDirection,
    TaskSpec,
)
from rl_automl.search.pareto import compute_pareto, efficiency_alternative

logger = get_logger("execution.comparison")


@dataclass
class ComparisonOutcome:
    summary: ComparisonSummary
    best_model: BestModel | None
    pareto: list[Any]


class ModelComparator:
    def __init__(self, task: TaskSpec) -> None:
        self.task = task
        self.direction: MetricDirection = task.metric_direction or MetricDirection.MAXIMIZE
        self.metric = task.metric

    # -- comparison --------------------------------------------------------------

    def compare(self, results: Iterable[ExperimentResult]) -> ComparisonSummary:
        results = list(results)
        rows = [self._row(result) for result in results]

        successful = [
            result for result in results if result.succeeded and result.validation_score is not None
        ]
        summary = ComparisonSummary(
            rows=rows,
            objective=self.task.objective or self.task.task_type.value,
            primary_metric=self.metric,
            metric_direction=self.direction,
        )

        if successful:
            summary.best_validation = max(successful, key=lambda r: r.rank_key()).experiment.model
            summary.fastest = min(successful, key=lambda r: r.training_time_s).experiment.model
            summary.most_compute_efficient = min(
                successful,
                key=lambda r: _compute_cost(r),
            ).experiment.model
            sized = [result for result in successful if result.model_size_bytes > 0]
            if sized:
                summary.smallest = min(sized, key=lambda r: r.model_size_bytes).experiment.model

            scored_on_test = [r for r in successful if r.test_score is not None]
            if scored_on_test:
                summary.best_test = max(
                    scored_on_test,
                    key=lambda r: orient(r.test_score or 0.0, self.direction),
                ).experiment.model

            summary.trade_offs = self._trade_offs(successful, summary)

        summary.table = self.render_table(results)
        return summary

    def _row(self, result: ExperimentResult) -> ComparisonRow:
        return ComparisonRow(
            model=result.experiment.model,
            experiment_id=result.experiment.experiment_id,
            status=result.status,
            primary_metric=result.primary_metric or self.metric,
            validation_score=result.validation_score,
            test_score=result.test_score,
            training_time_s=round(result.training_time_s, 4),
            inference_time_per_1k_ms=round(result.inference_time_per_1k_ms, 4),
            peak_memory_mb=round(result.peak_memory_mb, 2),
            model_size_bytes=int(result.model_size_bytes),
            error=result.error,
        )

    def render_table(self, results: Iterable[ExperimentResult]) -> str:
        """Plain-text table, safe to print in a terminal or paste into a report."""
        results = list(results)
        if not results:
            return "No experiments were executed."

        has_test = any(result.test_score is not None for result in results)
        metric_header = self.metric
        header = (
            f"{'model':<22} {metric_header:>10} {'train(s)':>10} {'mem(MB)':>10} {'size(KB)':>10}"
        )
        if has_test:
            header += f" {'test ' + metric_header:>14}"
        header += "  status"

        lines = ["MODEL COMPARISON", "", header, "-" * len(header)]
        for result in sorted(
            results,
            key=lambda item: (
                item.validation_score is None,
                -orient(item.validation_score or 0.0, self.direction),
            ),
        ):
            score = _format(result.validation_score)
            line = (
                f"{result.experiment.model:<22} {score:>10} "
                f"{result.training_time_s:>10.2f} {result.peak_memory_mb:>10.0f} "
                f"{result.model_size_bytes / 1024:>10.1f}"
            )
            if has_test:
                line += f" {_format(result.test_score):>14}"
            line += f"  {result.status.value}"
            if result.error:
                line += f" ({result.error_type})"
            lines.append(line)

        if has_test:
            lines.append("")
            lines.append(
                "test column is the held-out split, reported for transparency only; "
                "selection used validation"
            )
        return "\n".join(lines)

    def _trade_offs(
        self, successful: list[ExperimentResult], summary: ComparisonSummary
    ) -> list[str]:
        """Explicit statements about what was given up by the winning choice."""
        notes: list[str] = []
        best = max(successful, key=lambda r: r.rank_key())

        slowest = max(successful, key=lambda r: r.training_time_s)
        fastest = min(successful, key=lambda r: r.training_time_s)
        if slowest.experiment.model == best.experiment.model and fastest is not best:
            ratio = best.training_time_s / max(fastest.training_time_s, 1e-9)
            notes.append(
                f"{best.experiment.model} has the best validation {self.metric} but also the "
                f"longest training time ({best.training_time_s:.2f}s, {ratio:.1f}x "
                f"{fastest.experiment.model})"
            )

        hungriest = max(successful, key=lambda r: r.peak_memory_mb)
        if hungriest.experiment.model == best.experiment.model and len(successful) > 1:
            lightest = min(successful, key=lambda r: r.peak_memory_mb)
            notes.append(
                f"{best.experiment.model} used the most memory "
                f"({best.peak_memory_mb:.0f} MB vs {lightest.peak_memory_mb:.0f} MB for "
                f"{lightest.experiment.model})"
            )

        if summary.fastest and summary.fastest != best.experiment.model:
            fastest_result = next(
                result for result in successful if result.experiment.model == summary.fastest
            )
            delta = _score_delta(
                best.validation_score, fastest_result.validation_score, self.direction
            )
            notes.append(
                f"{summary.fastest} trains fastest ({fastest_result.training_time_s:.2f}s) with a "
                f"validation {self.metric} difference of {delta:.4f}"
            )

        failed = [result for result in successful if result.warnings]
        if failed:
            notes.append(
                f"{len(failed)} model(s) carried warnings; see the run report before trusting "
                "their scores as directly comparable"
            )
        return notes

    # -- selection ---------------------------------------------------------------

    def select(
        self, results: Iterable[ExperimentResult]
    ) -> tuple[BestModel | None, list[Any], ComparisonSummary]:
        """Choose the final model using the objective, on validation evidence."""
        results = list(results)
        summary = self.compare(results)
        pareto = compute_pareto(results, self.direction).frontier

        successful = [
            result for result in results if result.succeeded and result.validation_score is not None
        ]
        if not successful:
            logger.warning("no experiment succeeded; no model can be selected")
            return None, pareto, summary

        best = max(successful, key=lambda result: result.rank_key())
        alternative = efficiency_alternative(
            successful, self.direction, reference_model=best.experiment.model
        )

        reason = (
            f"highest validation {self.metric} "
            f"({_format(best.validation_score)}) among the "
            f"{len(successful)} approved pipeline(s) that trained successfully"
        )
        if self.task.metric:
            reason = (
                f"best validation {self.task.metric} per the problem statement's objective; "
                + reason
            )
        if len(successful) == 1:
            reason = (
                f"the only approved pipeline that trained successfully; validation "
                f"{self.metric} {_format(best.validation_score)}"
            )

        best_model = BestModel(
            model=best.experiment.model,
            experiment_id=best.experiment.experiment_id,
            primary_metric=best.primary_metric or self.metric,
            metric_direction=self.direction,
            validation_score=best.validation_score,
            test_score=best.test_score,
            reason=reason,
            # Explicit, so the report can state that held-out data did not drive the choice.
            selection_basis="validation only (test split reserved for final evaluation)",
            efficiency_alternative=(EfficiencyAlternative(**alternative) if alternative else None),
            hyperparameters=dict(best.experiment.hyperparameters),
            preprocessing=list(best.experiment.preprocessing),
            feature_selection=best.experiment.feature_selection,
        )
        return best_model, pareto, summary


def _compute_cost(result: ExperimentResult) -> float:
    """A single scalar for "how expensive was this", used for the efficiency award.

    Training time dominates; peak memory is folded in at a lower weight so a model that is
    marginally faster but far hungrier does not win the efficiency label.
    """
    return result.training_time_s + 0.001 * result.peak_memory_mb


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _score_delta(left: float | None, right: float | None, direction: MetricDirection) -> float:
    if left is None or right is None:
        return 0.0
    return abs(orient(left, direction) - orient(right, direction))


__all__ = ["ComparisonOutcome", "ModelComparator"]
