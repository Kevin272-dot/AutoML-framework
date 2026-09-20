"""Pipeline execution engine (spec §11, §12, §26, §29).

Responsibilities, and nothing more:

* load the dataset once and build the split once, so every pipeline is compared on
  identical rows
* train, evaluate on validation, measure time/memory/size, save artifacts
* enforce the compute budget (max experiments, max runtime, max memory)
* isolate failures: one pipeline blowing up must not take the run with it

It deliberately knows nothing about the RL agent. It receives an
:class:`ExperimentSpec` and returns an :class:`ExperimentResult`.

The test split is reachable only through :meth:`begin_finalize` /
:meth:`evaluate_test`. Calling :meth:`evaluate_test` before finalisation raises, which
turns "the test set never influenced the search" from a convention into an enforced
invariant that the test suite checks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.errors import (
    BudgetExceededError,
    DatasetValidationError,
    NotApprovedError,
    RunStateError,
)
from rl_automl.core.logging import get_logger
from rl_automl.core.metrics import direction_for, normalise_metric_name, primary_metric_for
from rl_automl.core.types import (
    ColumnKind,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    TaskSpec,
    TaskType,
    utc_now_iso,
)
from rl_automl.dataset.profiler import classify_column
from rl_automl.execution.evaluator import EvaluationResult, Evaluator
from rl_automl.execution.splits import DataSplits, make_splits
from rl_automl.execution.trainer import ModelTrainer, TrainedModel
from rl_automl.search.model_registry import get_spec

logger = get_logger("execution.executor")


@dataclass
class ExecutionLimits:
    """Hard caps enforced by the executor (spec §26)."""

    max_experiments: int = 12
    time_budget_s: float = 3600.0
    memory_budget_mb: float = 8192.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_experiments": self.max_experiments,
            "time_budget_s": self.time_budget_s,
            "memory_budget_mb": self.memory_budget_mb,
        }


@dataclass
class ExecutionSummary:
    results: list[ExperimentResult] = field(default_factory=list)
    n_success: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    total_training_time_s: float = 0.0
    elapsed_s: float = 0.0
    stopping_reason: str = "completed"
    warnings: list[str] = field(default_factory=list)

    @property
    def successful(self) -> list[ExperimentResult]:
        return [result for result in self.results if result.succeeded]

    def best_by_validation(self) -> ExperimentResult | None:
        candidates = [r for r in self.successful if r.validation_score is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.rank_key())

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "n_skipped": self.n_skipped,
            "total_training_time_s": round(self.total_training_time_s, 3),
            "elapsed_s": round(self.elapsed_s, 3),
            "stopping_reason": self.stopping_reason,
            "warnings": list(self.warnings),
        }


class PipelineExecutor:
    def __init__(
        self,
        task: TaskSpec,
        config: AutoMLConfig | None = None,
        run_id: str = "run",
        seed: int | None = None,
        limits: ExecutionLimits | None = None,
    ) -> None:
        self.task = task
        self.config = config or AutoMLConfig()
        self.run_id = run_id
        self.seed = self.config.runtime.seed if seed is None else seed

        self.limits = limits or ExecutionLimits(
            max_experiments=self.config.search.max_experiments,
            time_budget_s=self.config.search.time_budget_s,
            memory_budget_mb=self.config.search.memory_budget_mb,
        )

        self.splits: DataSplits | None = None
        self.column_kinds: dict[str, ColumnKind] = {}
        self.feature_names_raw: list[str] = []

        self._trainer = ModelTrainer(self.config, seed=self.seed)
        self._evaluator = Evaluator(
            task,
            class_labels=None,
        )
        self._fitted: dict[str, TrainedModel] = {}
        self._results: dict[str, ExperimentResult] = {}

        self._n_executed = 0
        self._started_at = 0.0
        self._finalizing = False
        self._test_evaluations = 0
        self._search_touched_test = False

        self._classes: np.ndarray | None = None
        self._needs_label_encoding = False
        self._run_dir = self.config.artifacts_dir() / "runs" / run_id

    # -- setup -------------------------------------------------------------------

    def prepare(self, frame: pd.DataFrame, holdout: pd.DataFrame | None = None) -> DataSplits:
        """Build the split once. Must be called before :meth:`execute`.

        ``holdout`` is an optional supplied test table. It is appended to the training
        frame and marked as the reserved test split, so a train/test pair that arrived as
        two files is honoured exactly rather than re-split.
        """
        combined, holdout_rows, note = self._combine_holdout(frame, holdout)
        self.splits = make_splits(
            combined,
            self.task,
            self.config.dataset,
            seed=self.seed,
            holdout_rows=holdout_rows,
        )
        if note:
            self.splits.warnings.append(note)

        feature_columns = [
            str(column) for column in frame.columns if str(column) != self.task.target
        ]
        self.feature_names_raw = feature_columns
        self.column_kinds = {
            name: classify_column(frame[name])
            for name in feature_columns
            if column_is_usable(frame[name])
        }

        if not self.column_kinds:
            raise RunStateError(
                "no usable feature columns were found",
                n_columns=len(feature_columns),
            )

        self._prepare_target_encoding()
        self._evaluator = Evaluator(self.task, class_labels=self._classes)
        self._started_at = time.perf_counter()
        self._run_dir.mkdir(parents=True, exist_ok=True)
        return self.splits

    def _combine_holdout(
        self, frame: pd.DataFrame, holdout: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, int | None, str | None]:
        """Append a supplied test table to the training frame, aligned to its columns.

        Alignment is by training-table columns: a test file routinely carries an id column
        the training file has not got, and just as routinely omits one. Extra columns are
        dropped and missing ones become NaN for the imputers, rather than refusing the run.
        """
        if holdout is None or len(holdout) == 0:
            return frame, None, None

        if self.task.task_type.is_supervised and (
            not self.task.target or self.task.target not in holdout.columns
        ):
            raise DatasetValidationError(
                f"the supplied test table has no '{self.task.target}' column, so its rows "
                "cannot be scored. Predictions-only files (the usual Kaggle test.csv) need "
                "labels to be a test set; leave the test set empty to split the training "
                "table internally instead.",
                target=self.task.target,
                columns=[str(column) for column in holdout.columns][:50],
            )

        extra = [str(column) for column in holdout.columns if column not in frame.columns]
        absent = [
            str(column)
            for column in frame.columns
            if column not in holdout.columns and str(column) != self.task.target
        ]
        aligned = holdout.reindex(columns=frame.columns)
        combined = pd.concat([frame, aligned], ignore_index=True)

        notes = [f"using the supplied test table: {len(holdout)} rows held out verbatim"]
        if extra:
            notes.append(f"ignored {len(extra)} column(s) not present in the training table")
        if absent:
            notes.append(
                f"{len(absent)} training column(s) were absent from the test table and are "
                "imputed"
            )
        logger.info(
            "supplied holdout attached",
            extra={
                "context": {
                    "holdout_rows": len(holdout),
                    "extra_columns": extra[:10],
                    "absent_columns": absent[:10],
                }
            },
        )
        return combined, len(holdout), "; ".join(notes)

    def _prepare_target_encoding(self) -> None:
        """Encode non-contiguous / non-numeric class labels to ``0..k-1``.

        XGBoost and LightGBM require integer class labels, so a dataset whose target is
        ``["yes", "no"]`` or ``[1, 2, 3]`` must be encoded. Predictions are decoded back to
        the original labels before they are scored, and the mapping ships in the model
        package.
        """
        if self.task.task_type is not TaskType.CLASSIFICATION or self.splits is None:
            return
        y_train = self.splits.target_for("train")
        if y_train is None:
            return

        values = np.asarray(y_train)
        classes = np.unique(values)

        already_contiguous = (
            values.dtype.kind in "iu"
            and int(values.min()) == 0
            and len(classes) == int(values.max()) + 1
        )
        self._classes = classes
        self._needs_label_encoding = not already_contiguous

    def _encode_labels(self, values: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return encoded labels plus a mask of which rows had a known label."""
        array = np.asarray(values)
        if not self._needs_label_encoding or self._classes is None:
            return array, np.ones(len(array), dtype=bool)
        codes = pd.Index(self._classes).get_indexer(array)
        return codes, codes >= 0

    # -- execution ---------------------------------------------------------------

    def execute(self, experiment: ExperimentSpec) -> ExperimentResult:
        """Train and validate one experiment. Never raises for pipeline failures."""
        if self.splits is None:
            raise RunStateError("prepare() must be called before execute()")

        started_at = utc_now_iso()
        result = ExperimentResult(
            experiment=experiment,
            primary_metric=normalise_metric_name(
                self.task.metric or primary_metric_for(self.task.task_type)
            ),
            metric_direction=self.task.metric_direction or direction_for(self.task.metric),
            started_at=started_at,
        )

        try:
            spec = get_spec(experiment.model)
        except Exception as exc:
            return self._fail(result, exc, "unknown model")

        if not spec.supports(self.task.task_type):
            return self._fail(
                result,
                ValueError(f"{spec.display_name} does not support {self.task.task_type.value}"),
                "unsupported task",
            )

        if not spec.fit_requires_target and experiment.feature_selection in (
            "kbest",
            "mutual_info",
        ):
            result.warnings.append(
                f"feature selection '{experiment.feature_selection}' needs a target and was "
                "ignored for this unsupervised task"
            )

        try:
            self.enforce_budget()
            y_train = self.splits.target_for("train")
            if y_train is not None and self._needs_label_encoding:
                y_train, _known = self._encode_labels(y_train)
            trained = self._trainer.train(
                spec,
                experiment,
                self.task.task_type,
                self.splits.features("train"),
                y_train,
                self.column_kinds,
                memory_limit_mb=self.limits.memory_budget_mb,
            )
        except BudgetExceededError:
            raise
        except Exception as exc:
            return self._fail(result, exc, "training")

        self._fitted[experiment.experiment_id] = trained
        self._n_executed += 1
        result.warnings.extend(trained.warnings)
        result.training_time_s = trained.training_time_s
        result.peak_memory_mb = trained.peak_memory_mb
        result.n_parameters = trained.n_parameters
        result.n_features_used = trained.n_features_out

        try:
            evaluation = self._evaluate_split(trained, "val")
        except Exception as exc:
            self._fitted.pop(experiment.experiment_id, None)
            return self._fail(result, exc, "validation")

        result.validation_metrics = evaluation.metrics
        result.validation_score = self._evaluator.primary_score(evaluation.metrics)
        # Evaluation warnings (in-sample clustering, missing probabilities, metric
        # fallbacks) belong on the result, otherwise the comparison table silently
        # presents incomparable numbers as comparable.
        result.warnings.extend(evaluation.warnings)
        if result.validation_score is None:
            result.warnings.append("no primary metric could be computed on the validation split")

        try:
            inference_total, inference_per_1k = ModelTrainer.measure_inference(
                trained, self.splits.features("val"), self.config.execution.inference_benchmark_rows
            )
            result.inference_time_ms = inference_total * 1000
            result.inference_time_per_1k_ms = inference_per_1k * 1000
        except Exception as exc:  # pragma: no cover - timing is best effort
            result.warnings.append(f"inference timing failed: {type(exc).__name__}: {exc}")

        try:
            artifact_dir = self._save_experiment_artifacts(trained, result)
            result.artifact_path = str(artifact_dir)
            result.model_size_bytes = (artifact_dir / "model.joblib").stat().st_size
        except Exception as exc:  # pragma: no cover - disk issues
            result.warnings.append(f"could not save experiment artifacts: {exc}")

        result.status = ExperimentStatus.SUCCESS
        result.finished_at = utc_now_iso()
        self._results[experiment.experiment_id] = result

        logger.info(
            "experiment finished",
            extra={
                "context": {
                    "model": experiment.model,
                    "validation_score": result.validation_score,
                    "training_time_s": round(result.training_time_s, 3),
                }
            },
        )
        return result

    def run(self, experiments: list[ExperimentSpec]) -> ExecutionSummary:
        """Execute a list of approved experiments under the configured budget."""
        if self.splits is None:
            raise RunStateError("prepare() must be called before run()")

        summary = ExecutionSummary()
        for experiment in experiments:
            if self._budget_exhausted():
                reason = self._budget_reason()
                skipped = ExperimentResult(
                    experiment=experiment,
                    primary_metric=result_metric(self.task),
                    metric_direction=self.task.metric_direction,
                    status=ExperimentStatus.SKIPPED,
                    error=f"not executed: {reason}",
                    error_type="budget_exhausted",
                )
                summary.results.append(skipped)
                summary.n_skipped += 1
                summary.warnings.append(reason)
                continue

            try:
                result = self.execute(experiment)
            except BudgetExceededError as exc:
                skipped = ExperimentResult(
                    experiment=experiment,
                    primary_metric=result_metric(self.task),
                    metric_direction=self.task.metric_direction,
                    status=ExperimentStatus.SKIPPED,
                    error=str(exc),
                    error_type="budget_exceeded",
                )
                summary.results.append(skipped)
                summary.n_skipped += 1
                continue

            summary.results.append(result)
            if result.succeeded:
                summary.n_success += 1
            else:
                summary.n_failed += 1
                logger.warning(
                    "experiment failed and was isolated",
                    extra={
                        "context": {
                            "model": experiment.model,
                            "error_type": result.error_type,
                            "error": (result.error or "")[:200],
                        }
                    },
                )

        summary.total_training_time_s = sum(r.training_time_s for r in summary.results)
        summary.elapsed_s = self.elapsed_s
        summary.stopping_reason = self._budget_reason() if self._budget_exhausted() else "completed"
        return summary

    @property
    def elapsed_s(self) -> float:
        if not self._started_at:
            return 0.0
        return time.perf_counter() - self._started_at

    # -- budget ------------------------------------------------------------------

    def _budget_exhausted(self) -> bool:
        return self._budget_reason() is not None

    def _budget_reason(self) -> str | None:
        if self._n_executed >= self.limits.max_experiments:
            return f"experiment limit reached ({self.limits.max_experiments})"
        if self.elapsed_s > self.limits.time_budget_s:
            return f"runtime budget exceeded ({self.limits.time_budget_s:.0f}s)"
        return None

    def enforce_budget(self) -> None:
        reason = self._budget_reason()
        if reason is not None:
            raise BudgetExceededError(reason, limits=self.limits.to_dict())

    @property
    def n_executed(self) -> int:
        return self._n_executed

    # -- finalisation ------------------------------------------------------------

    def begin_finalize(self) -> None:
        """Open the test-set gate. Called only after the final model is selected."""
        self._finalizing = True

    def evaluate_test(self, result: ExperimentResult) -> ExperimentResult:
        """Score one already-trained model on the held-out test set (spec §12, §15)."""
        if not self._finalizing:
            self._search_touched_test = True
            raise RunStateError(
                "the test set can only be evaluated during finalisation, after model "
                "selection, so that search decisions are never informed by it"
            )
        if self.splits is None or not self.splits.test_reserved:
            result.warnings.append("this task type has no reserved test split")
            return result

        trained = self._fitted.get(result.experiment.experiment_id)
        if trained is None:
            result.warnings.append("model is no longer in memory; test evaluation skipped")
            return result

        try:
            evaluation = self._evaluate_split(trained, "test")
        except Exception as exc:
            result.warnings.append(f"test evaluation failed: {type(exc).__name__}: {exc}")
            return result

        self._test_evaluations += 1
        result.test_metrics = evaluation.metrics
        result.test_score = self._evaluator.primary_score(evaluation.metrics)
        result.warnings.extend(evaluation.warnings)
        return result

    def _evaluate_split(self, trained: TrainedModel, which: str) -> EvaluationResult:
        """Evaluate a trained model on one split, per the task's protocol."""
        self._guard_split_access(which)
        assert self.splits is not None

        if self.task.task_type.is_supervised:
            X = self.splits.features(which)
            bundle = self._trainer.predict(trained, X)
            y_true = np.asarray(self.splits.target_for(which))
            y_pred = self.decode_labels(bundle.predictions)

            _codes, known = self._encode_labels(y_true)
            if not known.all():
                n_unknown = int((~known).sum())
                logger.warning(
                    "evaluation rows carry labels unseen during training and were excluded",
                    extra={"context": {"split": which, "n_excluded": n_unknown}},
                )
                y_true = y_true[known]
                y_pred = y_pred[known]
                if bundle.probabilities is not None:
                    bundle.probabilities = bundle.probabilities[known]
                if bundle.scores is not None:
                    bundle.scores = np.asarray(bundle.scores)[known]

            return self._evaluator.evaluate(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=bundle.probabilities,
                y_score=bundle.scores,
            )

        if self.task.task_type is TaskType.CLUSTERING:
            return self._evaluate_clustering(trained, which)
        if self.task.task_type is TaskType.ANOMALY_DETECTION:
            return self._evaluate_anomaly(trained, which)
        if self.task.task_type is TaskType.DIMENSIONALITY_REDUCTION:
            return self._evaluate_dimred(trained, which)
        raise RunStateError(f"no evaluation protocol for {self.task.task_type.value}")

    def _evaluate_clustering(self, trained: TrainedModel, which: str) -> EvaluationResult:
        """Clustering metrics need labels; not every algorithm can label new points."""
        assert self.splits is not None
        if hasattr(trained.model, "predict"):
            X = self.splits.features(which)
            transformed = trained.preprocessor.transform(X)
            labels = trained.model.predict(transformed)
            return self._evaluator.evaluate(X=transformed, labels=labels, model=trained.model)

        # DBSCAN and Agglomerative Clustering have no predict(); their labels only exist
        # for the data they were fitted on, so the score is explicitly in-sample.
        transformed = trained.preprocessor.transform(self.splits.features("train"))
        labels = getattr(trained.model, "labels_", None)
        result = self._evaluator.evaluate(X=transformed, labels=labels, model=trained.model)
        result.warnings.append(
            f"{trained.model_key} cannot label unseen points; clustering metrics are "
            "in-sample and are not directly comparable with models that can"
        )
        return result

    def _evaluate_anomaly(self, trained: TrainedModel, which: str) -> EvaluationResult:
        assert self.splits is not None
        X = self.splits.features(which)
        truth = self.splits.label_hint(which)
        if truth is None and self.task.target and self.splits.target_for(which) is not None:
            truth = self.splits.target_for(which)
        return self._evaluator.evaluate(
            X=trained.preprocessor.transform(X), labels=truth, model=trained.model
        )

    def _evaluate_dimred(self, trained: TrainedModel, which: str) -> EvaluationResult:
        assert self.splits is not None
        X = self.splits.features(which)
        return self._evaluator.evaluate(X=trained.preprocessor.transform(X), model=trained.model)

    def _guard_split_access(self, which: str) -> None:
        if which == "test" and not self._finalizing:
            self._search_touched_test = True
            raise RunStateError(
                "the test split was accessed before finalisation; this would invalidate "
                "the held-out evaluation"
            )

    # -- label decoding ----------------------------------------------------------

    def decode_labels(self, values: Any) -> np.ndarray:
        """Map encoded class indices back to the original labels.

        Public because the packaging layer must express the probe's expected predictions in
        the same label space the shipped loader will produce.
        """
        array = np.asarray(values)
        if not self._needs_label_encoding or self._classes is None:
            return array
        if self.task.task_type is not TaskType.CLASSIFICATION:
            return array
        try:
            indices = array.astype(int)
        except (TypeError, ValueError):
            return array
        if indices.min() < 0 or indices.max() >= len(self._classes):
            return array
        return self._classes[indices]

    # -- artifacts ---------------------------------------------------------------

    def _save_experiment_artifacts(self, trained: TrainedModel, result: ExperimentResult) -> Path:
        directory = self._run_dir / "models" / trained.experiment_id
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained.preprocessor, directory / "preprocessor.joblib")
        joblib.dump(trained.model, directory / "model.joblib")
        metadata = {
            **trained.describe(),
            "validation_score": result.validation_score,
            "validation_metrics": result.validation_metrics,
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
        return directory

    # -- accessors ---------------------------------------------------------------

    def get_trained(self, experiment_id: str) -> TrainedModel | None:
        return self._fitted.get(experiment_id)

    def get_result(self, experiment_id: str) -> ExperimentResult | None:
        return self._results.get(experiment_id)

    @property
    def evaluator(self) -> Evaluator:
        return self._evaluator

    @property
    def class_labels(self) -> np.ndarray | None:
        return self._classes

    @property
    def label_encoding_required(self) -> bool:
        return self._needs_label_encoding

    @property
    def test_evaluations(self) -> int:
        return self._test_evaluations

    @property
    def test_was_touched_during_search(self) -> bool:
        return self._search_touched_test

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    # -- internals ---------------------------------------------------------------

    @staticmethod
    def _fail(result: ExperimentResult, exc: Exception, stage: str) -> ExperimentResult:
        result.status = ExperimentStatus.FAILED
        result.error_type = type(exc).__name__
        result.error = f"{stage}: {exc}"
        result.finished_at = utc_now_iso()
        return result


def result_metric(task: TaskSpec) -> str:
    return normalise_metric_name(task.metric or primary_metric_for(task.task_type))


def column_is_usable(series: pd.Series) -> bool:
    """A column with no usable values cannot be a feature."""
    return not (series.isna().all() or series.nunique(dropna=True) == 0)


def require_approval(approved: list[ExperimentSpec] | None) -> list[ExperimentSpec]:
    """The approval gate in one function, so both the API and the CLI enforce it."""
    if not approved:
        raise NotApprovedError(
            "execution requires an explicit list of approved experiments; "
            "call /approve with the selected pipelines first"
        )
    return approved


__all__ = [
    "ExecutionLimits",
    "ExecutionSummary",
    "PipelineExecutor",
    "require_approval",
    "result_metric",
]
