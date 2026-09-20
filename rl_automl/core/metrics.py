"""Metric definitions shared by the evaluator and the task-understanding layer.

Metrics live in ``core`` because both the evaluator (``execution``) and the task
classifier (``task``) need to agree on names, directions and applicability, and neither
should depend on the other.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_automl.core.types import MetricDirection, TaskType


@dataclass(frozen=True)
class MetricSpec:
    name: str
    display_name: str
    direction: MetricDirection
    tasks: tuple[TaskType, ...]
    requires_probabilities: bool = False
    requires_labels: bool = False
    description: str = ""

    def applicable_to(self, task: TaskType) -> bool:
        return task in self.tasks


_CLS = (TaskType.CLASSIFICATION,)
_REG = (TaskType.REGRESSION,)
_CLUSTER = (TaskType.CLUSTERING,)
_ANOMALY = (TaskType.ANOMALY_DETECTION,)
_DIMRED = (TaskType.DIMENSIONALITY_REDUCTION,)


METRICS: dict[str, MetricSpec] = {
    # -- classification ---------------------------------------------------------
    "accuracy": MetricSpec("accuracy", "Accuracy", MetricDirection.MAXIMIZE, _CLS),
    "balanced_accuracy": MetricSpec(
        "balanced_accuracy", "Balanced Accuracy", MetricDirection.MAXIMIZE, _CLS
    ),
    "precision": MetricSpec("precision", "Precision", MetricDirection.MAXIMIZE, _CLS),
    "recall": MetricSpec("recall", "Recall", MetricDirection.MAXIMIZE, _CLS),
    "f1": MetricSpec("f1", "F1", MetricDirection.MAXIMIZE, _CLS),
    "roc_auc": MetricSpec(
        "roc_auc", "ROC-AUC", MetricDirection.MAXIMIZE, _CLS, requires_probabilities=True
    ),
    "pr_auc": MetricSpec(
        "pr_auc", "PR-AUC", MetricDirection.MAXIMIZE, _CLS, requires_probabilities=True
    ),
    "average_precision": MetricSpec(
        "average_precision",
        "Average Precision",
        MetricDirection.MAXIMIZE,
        _CLS,
        requires_probabilities=True,
    ),
    "matthews": MetricSpec("matthews", "Matthews Correlation", MetricDirection.MAXIMIZE, _CLS),
    "log_loss": MetricSpec(
        "log_loss", "Log Loss", MetricDirection.MINIMIZE, _CLS, requires_probabilities=True
    ),
    # -- regression -------------------------------------------------------------
    "mae": MetricSpec("mae", "MAE", MetricDirection.MINIMIZE, _REG),
    "rmse": MetricSpec("rmse", "RMSE", MetricDirection.MINIMIZE, _REG),
    "r2": MetricSpec("r2", "R2", MetricDirection.MAXIMIZE, _REG),
    "mape": MetricSpec("mape", "MAPE", MetricDirection.MINIMIZE, _REG),
    "rmsle": MetricSpec("rmsle", "RMSLE", MetricDirection.MINIMIZE, _REG),
    # -- clustering -------------------------------------------------------------
    "silhouette": MetricSpec("silhouette", "Silhouette", MetricDirection.MAXIMIZE, _CLUSTER),
    "davies_bouldin": MetricSpec(
        "davies_bouldin", "Davies-Bouldin", MetricDirection.MINIMIZE, _CLUSTER
    ),
    "calinski_harabasz": MetricSpec(
        "calinski_harabasz", "Calinski-Harabasz", MetricDirection.MAXIMIZE, _CLUSTER
    ),
    # -- anomaly detection ------------------------------------------------------
    "anomaly_roc_auc": MetricSpec(
        "anomaly_roc_auc",
        "ROC-AUC (anomaly)",
        MetricDirection.MAXIMIZE,
        _ANOMALY,
        requires_labels=True,
    ),
    "anomaly_average_precision": MetricSpec(
        "anomaly_average_precision",
        "PR-AUC (anomaly)",
        MetricDirection.MAXIMIZE,
        _ANOMALY,
        requires_labels=True,
    ),
    # -- dimensionality reduction ----------------------------------------------
    "explained_variance": MetricSpec(
        "explained_variance", "Explained Variance", MetricDirection.MAXIMIZE, _DIMRED
    ),
    "reconstruction_error": MetricSpec(
        "reconstruction_error", "Reconstruction Error", MetricDirection.MINIMIZE, _DIMRED
    ),
}

#: Alias spellings that may appear in a problem statement.
METRIC_ALIASES: dict[str, str] = {
    "auc": "roc_auc",
    "roc-auc": "roc_auc",
    "rocauc": "roc_auc",
    "auroc": "roc_auc",
    "pr-auc": "pr_auc",
    "prauc": "pr_auc",
    "average-precision": "average_precision",
    "avg_precision": "average_precision",
    "mcc": "matthews",
    "matthews_correlation": "matthews",
    "cross_entropy": "log_loss",
    "logloss": "log_loss",
    "mean_absolute_error": "mae",
    "mean_squared_error": "rmse",
    "root_mean_squared_error": "rmse",
    "mse": "rmse",
    "r_squared": "r2",
    "r-squared": "r2",
    "r2": "r2",
    "db_index": "davies_bouldin",
    "calinski": "calinski_harabasz",
    "explained_variance_ratio": "explained_variance",
}

#: Primary metric chosen when the problem statement names none.
PRIMARY_METRIC_BY_TASK: dict[TaskType, str] = {
    TaskType.CLASSIFICATION: "f1",
    TaskType.REGRESSION: "rmse",
    TaskType.CLUSTERING: "silhouette",
    TaskType.ANOMALY_DETECTION: "anomaly_average_precision",
    TaskType.DIMENSIONALITY_REDUCTION: "explained_variance",
}

#: Secondary metrics reported alongside the primary one.
SECONDARY_METRICS_BY_TASK: dict[TaskType, tuple[str, ...]] = {
    TaskType.CLASSIFICATION: ("accuracy", "precision", "recall", "roc_auc"),
    TaskType.REGRESSION: ("mae", "r2"),
    TaskType.CLUSTERING: ("davies_bouldin", "calinski_harabasz"),
    TaskType.ANOMALY_DETECTION: ("anomaly_roc_auc",),
    TaskType.DIMENSIONALITY_REDUCTION: ("reconstruction_error",),
}


def normalise_metric_name(raw: str) -> str:
    """Map an alias or loose spelling onto a canonical metric name."""
    key = raw.strip().lower().replace(" ", "_")
    if key in METRICS:
        return key
    return METRIC_ALIASES.get(key, key)


def get_metric(name: str) -> MetricSpec | None:
    return METRICS.get(normalise_metric_name(name))


def direction_for(name: str) -> MetricDirection:
    spec = get_metric(name)
    if spec is not None:
        return spec.direction
    return MetricDirection.MAXIMIZE


def primary_metric_for(task: TaskType) -> str:
    return PRIMARY_METRIC_BY_TASK[task]


def secondary_metrics_for(task: TaskType) -> tuple[str, ...]:
    return SECONDARY_METRICS_BY_TASK.get(task, ())


def is_better(candidate: float, reference: float, direction: MetricDirection) -> bool:
    return candidate > reference if direction is MetricDirection.MAXIMIZE else candidate < reference


def orient(score: float, direction: MetricDirection) -> float:
    """Flip a score so that larger is always better. Used for rewards and ranking."""
    return score if direction is MetricDirection.MAXIMIZE else -score


def metrics_for_task(task: TaskType) -> list[str]:
    return [name for name, spec in METRICS.items() if spec.applicable_to(task)]


__all__ = [
    "METRICS",
    "METRIC_ALIASES",
    "PRIMARY_METRIC_BY_TASK",
    "SECONDARY_METRICS_BY_TASK",
    "MetricSpec",
    "direction_for",
    "get_metric",
    "is_better",
    "metrics_for_task",
    "normalise_metric_name",
    "orient",
    "primary_metric_for",
    "secondary_metrics_for",
]
