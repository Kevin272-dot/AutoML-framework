from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.types import LearningType, MetricDirection, TaskSpec, TaskType


@pytest.fixture
def config(tmp_path):
    cfg = AutoMLConfig.load()
    cfg.runtime.artifacts_dir = str(tmp_path / "artifacts")
    cfg.dataset.min_rows = 20
    cfg.models.enabled = ["logistic_regression", "linear_regression", "random_forest"]
    cfg.search.min_experiments = 1
    cfg.search.max_experiments = 2
    cfg.search.n_recommendations = 1
    cfg.search.time_budget_s = 30
    cfg.packaging.include_onnx = False
    cfg.ensure_directories()
    return cfg


@pytest.fixture
def classification_task():
    return TaskSpec(
        learning_type=LearningType.SUPERVISED,
        task_type=TaskType.CLASSIFICATION,
        objective="predict outcome",
        target="outcome",
        metric="f1",
        metric_direction=MetricDirection.MAXIMIZE,
    )


@pytest.fixture
def regression_task():
    return TaskSpec(
        learning_type=LearningType.SUPERVISED,
        task_type=TaskType.REGRESSION,
        objective="estimate target",
        target="target",
        metric="rmse",
        metric_direction=MetricDirection.MINIMIZE,
    )


@pytest.fixture
def binary_frame():
    rng = np.random.default_rng(7)
    n = 120
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "number": x,
            "number_with_missing": np.where(np.arange(n) % 11 == 0, np.nan, rng.normal(size=n)),
            "category": np.where(x > 0, "north", "south"),
            "when": pd.date_range("2024-01-01", periods=n, freq="h"),
            "outcome": np.where(x + rng.normal(scale=0.25, size=n) > 0, "yes", "no"),
        }
    )
