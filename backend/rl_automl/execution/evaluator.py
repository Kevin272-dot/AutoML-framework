"""Evaluation (spec §13).

Computes the metrics for each task family and, critically, keeps the two evaluation
contexts apart:

* ``evaluate_validation`` -- used by the RL search and by model selection.
* ``evaluate_test`` -- final reporting only, never a selection input.

Every metric is computed defensively: a metric that cannot be computed for a given
prediction (a single-class fold, no positive predictions, negative values under a log
transform) is omitted with a recorded reason rather than raising. That is what lets a
run survive one model behaving badly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn import metrics as skm

from rl_automl.core.logging import get_logger
from rl_automl.core.metrics import (
    direction_for,
    normalise_metric_name,
    primary_metric_for,
    secondary_metrics_for,
)
from rl_automl.core.types import MetricDirection, TaskSpec, TaskType

logger = get_logger("execution.evaluator")

MAX_CLUSTER_METRIC_ROWS = 20_000


@dataclass
class EvaluationResult:
    metrics: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": dict(self.metrics), "skipped": dict(self.skipped)}


class Evaluator:
    """Task-aware metric computation."""

    def __init__(self, task: TaskSpec, class_labels: np.ndarray | None = None) -> None:
        self.task = task
        self.task_type = task.task_type
        self.primary_metric = normalise_metric_name(
            task.metric or primary_metric_for(self.task_type)
        )
        self.direction = task.metric_direction or direction_for(self.primary_metric)
        self.class_labels = class_labels
        self._is_binary = bool(class_labels is not None and len(np.unique(class_labels)) <= 2)

    def _positive_label(self, classes: np.ndarray) -> Any:
        """The label treated as the positive class for binary metrics.

        Prefers the training classes recorded by the executor so that validation, test
        and packaged inference all agree on which class "positive" means.
        """
        if self.class_labels is not None and len(self.class_labels) == 2:
            return self.class_labels[-1]
        return classes[-1]

    # -- public API --------------------------------------------------------------

    def evaluate(
        self,
        y_true: Any = None,
        y_pred: Any = None,
        y_proba: Any = None,
        y_score: Any = None,
        X: Any = None,
        labels: Any = None,
        model: Any = None,
    ) -> EvaluationResult:
        """Dispatch to the right protocol for this task type.

        ``y_proba`` holds calibrated probabilities where the model provides them;
        ``y_score`` holds unbounded decision-function values as a fallback so that models
        like ``SVC`` still get ranking metrics.
        """
        if self.task_type.is_supervised:
            if y_true is None or y_pred is None:
                raise ValueError("supervised evaluation requires y_true and y_pred")
            return self._supervised(np.asarray(y_true), np.asarray(y_pred), y_proba, y_score)
        if self.task_type is TaskType.CLUSTERING:
            return self._clustering(X, labels)
        if self.task_type is TaskType.ANOMALY_DETECTION:
            return self._anomaly(X, labels, model)
        if self.task_type is TaskType.DIMENSIONALITY_REDUCTION:
            return self._dimensionality_reduction(X, model)
        raise ValueError(f"no evaluation protocol for {self.task_type.value}")

    def primary_score(self, metrics: dict[str, float]) -> float | None:
        """The metric the objective is expressed in, oriented so larger is better."""
        value = metrics.get(self.primary_metric)
        if value is None:
            return None
        return float(value)

    def oriented_primary(self, metrics: dict[str, float]) -> float | None:
        value = self.primary_score(metrics)
        if value is None:
            return None
        return value if self.direction is MetricDirection.MAXIMIZE else -value

    # -- supervised --------------------------------------------------------------

    def _supervised(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Any, y_score: Any = None
    ) -> EvaluationResult:
        if self.task_type is TaskType.CLASSIFICATION:
            return self._classification(y_true, y_pred, y_proba, y_score)
        return self._regression(y_true, y_pred)

    def _classification(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Any, y_score: Any = None
    ) -> EvaluationResult:
        result = EvaluationResult()
        unique_true = np.unique(y_true)
        binary = len(unique_true) <= 2
        average = "binary" if binary else "macro"

        # Explicit positive label matters: with a target like ["churned", "stayed"] the
        # default pos_label of 1 is invalid, and scikit-learn's ranking metrics refuse to
        # guess. The indicator array gives roc_auc/average_precision an unambiguous {0,1}
        # truth to score against.
        pos_label = self._positive_label(unique_true) if binary else None
        label_kwargs = {"pos_label": pos_label} if binary else {}
        y_indicator = (y_true == pos_label).astype(int) if binary else None

        def record(name: str, fn) -> None:
            try:
                value = fn()
                if value is None:
                    result.skipped[name] = "not applicable"
                    return
                value = float(value)
                if not np.isfinite(value):
                    result.skipped[name] = "non-finite result"
                    return
                result.metrics[name] = value
            except Exception as exc:
                result.skipped[name] = f"{type(exc).__name__}: {exc}"

        record("accuracy", lambda: skm.accuracy_score(y_true, y_pred))
        record("balanced_accuracy", lambda: skm.balanced_accuracy_score(y_true, y_pred))
        record(
            "precision",
            lambda: skm.precision_score(
                y_true, y_pred, average=average, zero_division=0, **label_kwargs
            ),
        )
        record(
            "recall",
            lambda: skm.recall_score(
                y_true, y_pred, average=average, zero_division=0, **label_kwargs
            ),
        )
        record(
            "f1",
            lambda: skm.f1_score(y_true, y_pred, average=average, zero_division=0, **label_kwargs),
        )
        record("matthews", lambda: skm.matthews_corrcoef(y_true, y_pred))

        probabilities = _as_2d(y_proba)
        if probabilities is not None:
            if binary:
                positive = _positive_class_column(probabilities, y_true)
                record("roc_auc", lambda: skm.roc_auc_score(y_indicator, positive))
                record(
                    "average_precision",
                    lambda: skm.average_precision_score(y_indicator, positive),
                )
                record("pr_auc", lambda: skm.average_precision_score(y_indicator, positive))
                record(
                    "log_loss",
                    lambda: skm.log_loss(y_indicator, np.column_stack([1 - positive, positive])),
                )
            else:
                record(
                    "roc_auc",
                    lambda: skm.roc_auc_score(y_true, probabilities, average="macro"),
                )
                record(
                    "average_precision",
                    lambda: skm.average_precision_score(y_true, probabilities, average="macro"),
                )
                record("log_loss", lambda: skm.log_loss(y_true, probabilities))
        elif y_score is not None:
            # SVC and similar estimators expose decision_function instead of probabilities.
            # Ranking metrics are still valid; log_loss is not, and is reported as skipped.
            scores = np.asarray(y_score, dtype=np.float64)
            if binary:
                ranking = scores[:, -1] if scores.ndim == 2 else scores.reshape(-1)
                record("roc_auc", lambda: skm.roc_auc_score(y_indicator, ranking))
                record(
                    "average_precision",
                    lambda: skm.average_precision_score(y_indicator, ranking),
                )
                record("pr_auc", lambda: skm.average_precision_score(y_indicator, ranking))
            else:
                record(
                    "roc_auc",
                    lambda: skm.roc_auc_score(y_true, scores, average="macro"),
                )
                record(
                    "average_precision",
                    lambda: skm.average_precision_score(y_true, scores, average="macro"),
                )
            result.skipped["log_loss"] = "model exposes no probabilities"
            result.warnings.append(
                "ranking metrics were computed from decision_function scores rather than "
                "calibrated probabilities"
            )
        else:
            result.warnings.append(
                "model exposes no probabilities; probability-based metrics "
                "(roc_auc, pr_auc, log_loss) were not computed"
            )

        self._ensure_primary(result)
        return result

    def _regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationResult:
        result = EvaluationResult()

        def record(name: str, fn) -> None:
            try:
                value = float(fn())
                if not np.isfinite(value):
                    result.skipped[name] = "non-finite result"
                    return
                result.metrics[name] = value
            except Exception as exc:
                result.skipped[name] = f"{type(exc).__name__}: {exc}"

        record("mae", lambda: skm.mean_absolute_error(y_true, y_pred))
        mse = float(skm.mean_squared_error(y_true, y_pred))
        result.metrics["rmse"] = float(np.sqrt(mse))
        record("r2", lambda: skm.r2_score(y_true, y_pred))

        if np.all(y_true >= 0) and np.all(y_pred >= 0):
            record("rmsle", lambda: skm.mean_squared_log_error(y_true, y_pred) ** 0.5)
        else:
            result.skipped["rmsle"] = "requires non-negative targets and predictions"

        non_zero = np.abs(y_true) > 1e-12
        if non_zero.any():
            record(
                "mape",
                lambda: float(
                    np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]))
                ),
            )
        else:
            result.skipped["mape"] = "targets are all zero"

        self._ensure_primary(result)
        return result

    def _ensure_primary(self, result: EvaluationResult) -> None:
        if self.primary_metric in result.metrics:
            return
        for fallback in secondary_metrics_for(self.task_type):
            if fallback in result.metrics:
                result.warnings.append(
                    f"primary metric '{self.primary_metric}' could not be computed; "
                    f"falling back to '{fallback}'"
                )
                return
        if result.metrics:
            result.warnings.append(
                f"primary metric '{self.primary_metric}' could not be computed for this model"
            )

    # -- clustering --------------------------------------------------------------

    def _clustering(self, X: Any, labels: Any) -> EvaluationResult:
        result = EvaluationResult()
        if X is None or labels is None:
            raise ValueError("clustering evaluation requires X and the fitted labels")

        matrix = _to_dense_array(X)
        assignments = np.asarray(labels)
        n_labels = len(np.unique(assignments))

        if n_labels < 2:
            result.skipped["silhouette"] = "fewer than two clusters were produced"
            result.skipped["davies_bouldin"] = "fewer than two clusters were produced"
            result.skipped["calinski_harabasz"] = "fewer than two clusters were produced"
            result.warnings.append(
                "the model produced a single cluster; clustering metrics are undefined"
            )
            return result

        if len(matrix) > MAX_CLUSTER_METRIC_ROWS:
            rng = np.random.default_rng(1234)
            subset = rng.choice(len(matrix), MAX_CLUSTER_METRIC_ROWS, replace=False)
            matrix = matrix[subset]
            assignments = assignments[subset]
            result.warnings.append(
                f"clustering metrics computed on a {MAX_CLUSTER_METRIC_ROWS:,}-row sample"
            )

        def record(name: str, fn) -> None:
            try:
                result.metrics[name] = float(fn())
            except Exception as exc:
                result.skipped[name] = f"{type(exc).__name__}: {exc}"

        record("silhouette", lambda: skm.silhouette_score(matrix, assignments))
        record("davies_bouldin", lambda: skm.davies_bouldin_score(matrix, assignments))
        record("calinski_harabasz", lambda: skm.calinski_harabasz_score(matrix, assignments))
        result.metrics["n_clusters"] = float(n_labels)
        self._ensure_primary(result)
        return result

    # -- anomaly detection -------------------------------------------------------

    def _anomaly(self, X: Any, labels: Any, model: Any) -> EvaluationResult:
        result = EvaluationResult()
        if model is None:
            raise ValueError("anomaly evaluation requires the fitted model")

        predictions = np.asarray(model.predict(X)) if hasattr(model, "predict") else None
        if predictions is not None:
            # scikit-learn conventions: -1 is an anomaly, 1 is normal.
            flagged = (predictions == -1).astype(int)
            result.metrics["anomaly_rate"] = float(flagged.mean())

        scores = _anomaly_scores(model, X)

        if labels is not None:
            truth = _as_anomaly_truth(np.asarray(labels))
            if truth is not None and scores is not None:
                try:
                    result.metrics["anomaly_roc_auc"] = float(skm.roc_auc_score(truth, scores))
                except Exception as exc:
                    result.skipped["anomaly_roc_auc"] = f"{type(exc).__name__}: {exc}"
                try:
                    result.metrics["anomaly_average_precision"] = float(
                        skm.average_precision_score(truth, scores)
                    )
                except Exception as exc:
                    result.skipped["anomaly_average_precision"] = f"{type(exc).__name__}: {exc}"
            elif truth is None:
                result.skipped["anomaly_roc_auc"] = "no ground-truth anomaly labels available"
                result.skipped["anomaly_average_precision"] = (
                    "no ground-truth anomaly labels available"
                )
            else:
                result.warnings.append(
                    "labels were provided but the model exposes no anomaly score; "
                    "ranking metrics were skipped"
                )

        self._ensure_primary(result)
        return result

    # -- dimensionality reduction ------------------------------------------------

    def _dimensionality_reduction(self, X: Any, model: Any) -> EvaluationResult:
        result = EvaluationResult()
        if model is None:
            raise ValueError("dimensionality reduction evaluation requires the fitted model")

        if hasattr(model, "explained_variance_ratio_"):
            result.metrics["explained_variance"] = float(
                np.sum(np.asarray(model.explained_variance_ratio_))
            )

        if hasattr(model, "inverse_transform"):
            try:
                matrix = _to_dense_array(X)
                reduced = model.transform(matrix)
                reconstructed = model.inverse_transform(reduced)
                result.metrics["reconstruction_error"] = float(
                    np.mean((matrix - reconstructed) ** 2)
                )
                result.metrics["n_components"] = float(np.asarray(reduced).shape[1])
            except Exception as exc:
                result.skipped["reconstruction_error"] = f"{type(exc).__name__}: {exc}"

        self._ensure_primary(result)
        return result


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _as_2d(probabilities: Any) -> np.ndarray | None:
    if probabilities is None:
        return None
    array = np.asarray(probabilities)
    if array.ndim == 1:
        array = np.column_stack([1 - array, array])
    if array.ndim != 2 or array.shape[1] < 2:
        return None
    return array


def _positive_class_column(probabilities: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Pick the column corresponding to the positive class, handling arbitrary labels."""
    classes = np.unique(y_true)
    if len(classes) < 2:
        return probabilities[:, -1]
    # scikit-learn orders classes ascending; the "positive" class is the larger label,
    # except for the common 0/1 case where it is 1.
    if probabilities.shape[1] == 2:
        return probabilities[:, -1]
    return probabilities[:, int(np.argmax(classes))]


def _to_dense_array(X: Any) -> np.ndarray:
    array = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return np.asarray(array, dtype=np.float64)


def _anomaly_scores(model: Any, X: Any) -> np.ndarray | None:
    """Higher score means more anomalous, normalised across sklearn's API variants."""
    try:
        if hasattr(model, "decision_function"):
            return -np.asarray(model.decision_function(X), dtype=np.float64)
        if hasattr(model, "score_samples"):
            return -np.asarray(model.score_samples(X), dtype=np.float64)
    except Exception:  # pragma: no cover - model specific
        return None
    return None


def _as_anomaly_truth(labels: np.ndarray) -> np.ndarray | None:
    """Map ground-truth labels onto 1 = anomaly, 0 = normal. Returns None if unusable."""
    unique = np.unique(labels[~_is_nan(labels)]) if labels.dtype.kind == "f" else np.unique(labels)
    if len(unique) < 2:
        return None
    if set(unique.tolist()) <= {-1, 1}:
        return (labels == -1).astype(int)
    return (labels != 0).astype(int)


def _is_nan(array: np.ndarray) -> np.ndarray:
    return np.isnan(array) if array.dtype.kind == "f" else np.zeros(len(array), dtype=bool)


def metric_direction_for(task: TaskSpec) -> MetricDirection:
    return task.metric_direction or direction_for(task.metric)


__all__ = ["EvaluationResult", "Evaluator", "metric_direction_for"]
