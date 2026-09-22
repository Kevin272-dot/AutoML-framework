from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_automl.core.types import ColumnKind, MetricDirection, TaskType
from rl_automl.dataset.fingerprint import (
    FINGERPRINT_FEATURES,
    FINGERPRINT_VERSION,
    build_fingerprint,
)
from rl_automl.dataset.profiler import classify_column, profile_dataset
from rl_automl.task.task_classifier import TaskClassifier


@pytest.mark.parametrize(
    ("statement", "columns", "expected", "metric", "target"),
    [
        (
            "Predict whether a customer will churn.",
            ["age", "churn"],
            TaskType.CLASSIFICATION,
            "f1",
            "churn",
        ),
        (
            "Minimize RMSE when predicting house_price.",
            ["area", "house_price"],
            TaskType.REGRESSION,
            "rmse",
            "house_price",
        ),
        (
            "Segment customers into groups.",
            ["spend", "visits"],
            TaskType.CLUSTERING,
            "silhouette",
            None,
        ),
        (
            "Detect anomalous transactions.",
            ["amount", "merchant"],
            TaskType.ANOMALY_DETECTION,
            # The canonical primary metric from core.metrics, not the `anomaly_rate`
            # secondary the evaluator also reports.
            "anomaly_average_precision",
            None,
        ),
        (
            "Use PCA for dimensionality reduction.",
            ["a", "b"],
            TaskType.DIMENSIONALITY_REDUCTION,
            "explained_variance",
            None,
        ),
    ],
)
def test_task_classifier_covers_task_families(statement, columns, expected, metric, target):
    task = TaskClassifier().classify(statement, columns=columns)
    assert task.task_type is expected
    assert task.metric == metric
    assert task.target == target


def test_task_classifier_hints_override_heuristics():
    task = TaskClassifier().classify(
        "cluster these records",
        columns=["x", "label"],
        task_hint="classification",
        target_hint="label",
        metric_hint="accuracy",
    )
    assert task.task_type is TaskType.CLASSIFICATION
    assert task.target == "label"
    assert task.metric == "accuracy"
    assert task.metric_direction is MetricDirection.MAXIMIZE
    assert task.source == "user"


def test_profiler_detects_types_missing_duplicates_imbalance(config):
    frame = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, np.nan, 4.0] * 30,
            "category": ["a", "a", "b", "a"] * 30,
            "event_date": pd.date_range("2023-01-01", periods=120).astype(str),
            "constant": [1] * 120,
            "outcome": [0] * 108 + [1] * 12,
        }
    )
    profile = profile_dataset(frame, "outcome", config.dataset)
    assert profile.n_rows == 120
    assert "numeric" in profile.numeric_features
    assert "category" in profile.categorical_features
    assert "event_date" in profile.datetime_features
    assert "constant" in profile.constant_features
    assert profile.missing_ratio > 0
    assert profile.class_distribution.is_imbalanced
    assert any("imbalanced" in warning for warning in profile.warnings)
    assert classify_column(frame["event_date"]) is ColumnKind.DATETIME


def test_fingerprint_is_versioned_bounded_finite_and_task_aware(binary_frame, config):
    profile = profile_dataset(binary_frame, "outcome", config.dataset)
    fp = build_fingerprint(profile, TaskType.CLASSIFICATION)
    assert fp.schema_version == FINGERPRINT_VERSION
    assert fp.feature_names == list(FINGERPRINT_FEATURES)
    assert len(fp.values) == 40
    assert np.isfinite(fp.values).all()
    assert np.asarray(fp.values).min() >= 0
    assert np.asarray(fp.values).max() <= 1
    values = fp.as_dict()
    assert values["has_target"] == 1
    assert values["task_is_classification"] == 1
    assert values["task_is_regression"] == 0
