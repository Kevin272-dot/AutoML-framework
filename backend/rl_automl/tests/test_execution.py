from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics

from rl_automl.core.errors import RunStateError
from rl_automl.core.types import ColumnKind, ExperimentSpec, ExperimentStatus, TaskType
from rl_automl.execution.evaluator import Evaluator
from rl_automl.execution.executor import ExecutionLimits, PipelineExecutor
from rl_automl.execution.preprocessing import PreprocessorBuilder
from rl_automl.execution.splits import make_splits
from rl_automl.search.model_registry import get_spec


def test_splits_are_deterministic_disjoint_and_stratified(
    binary_frame, classification_task, config
):
    one = make_splits(binary_frame, classification_task, config.dataset, seed=13)
    two = make_splits(binary_frame, classification_task, config.dataset, seed=13)
    assert np.array_equal(one.train_idx, two.train_idx)
    assert not (
        set(one.train_idx) & set(one.val_idx)
        | set(one.train_idx) & set(one.test_idx)
        | set(one.val_idx) & set(one.test_idx)
    )
    assert len(set(one.train_idx) | set(one.val_idx) | set(one.test_idx)) == len(binary_frame)
    assert one.stratified and one.test_reserved
    assert "outcome" not in one.features("train")
    assert set(one.target_for("train").unique()) == {"yes", "no"}


def test_preprocessor_handles_missing_unknown_categories_and_datetimes(binary_frame, config):
    kinds = {
        "number": ColumnKind.NUMERIC,
        "number_with_missing": ColumnKind.NUMERIC,
        "category": ColumnKind.CATEGORICAL,
        "when": ColumnKind.DATETIME,
    }
    pipeline, metadata = PreprocessorBuilder(config.dataset).build(
        get_spec("logistic_regression"),
        [
            "impute_numeric",
            "encode_categorical",
            "scale_standard",
            "datetime_features",
            "missing_indicator",
        ],
        task_type=TaskType.CLASSIFICATION,
        column_kinds=kinds,
    )
    X = binary_frame.drop(columns="outcome")
    transformed = pipeline.fit_transform(X.iloc[:80], binary_frame.outcome.iloc[:80])
    inference = X.iloc[[90]].copy()
    inference["category"] = "previously-unseen"
    predicted_input = pipeline.transform(inference)
    assert transformed.shape[0] == 80
    assert predicted_input.shape[0] == 1
    assert np.isfinite(np.asarray(predicted_input, dtype=float)).all()
    assert metadata["datetime_features"] == ["when"]


def test_evaluator_matches_sklearn_and_handles_string_positive_label(classification_task):
    truth = np.array(["no", "yes", "yes", "no", "yes"])
    pred = np.array(["no", "yes", "no", "no", "yes"])
    proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.8, 0.2], [0.2, 0.8]])
    result = Evaluator(classification_task, class_labels=np.array(["no", "yes"])).evaluate(
        y_true=truth, y_pred=pred, y_proba=proba
    )
    assert result.metrics["f1"] == pytest.approx(metrics.f1_score(truth, pred, pos_label="yes"))
    assert result.metrics["roc_auc"] == pytest.approx(
        metrics.roc_auc_score(truth == "yes", proba[:, 1])
    )


def test_executor_isolates_failures_enforces_budget_and_guards_test(
    binary_frame, classification_task, config
):
    executor = PipelineExecutor(
        classification_task,
        config,
        run_id="executor",
        limits=ExecutionLimits(max_experiments=1, time_budget_s=30),
    )
    executor.prepare(binary_frame.drop(columns="when"))
    valid = ExperimentSpec(
        model="logistic_regression",
        preset_index=0,
        preprocessing=["impute_numeric", "scale_standard", "encode_categorical"],
    )
    result = executor.execute(valid)
    assert result.succeeded
    with pytest.raises(RunStateError):
        executor.evaluate_test(result)
    assert executor.test_was_touched_during_search
    executor.begin_finalize()
    assert executor.evaluate_test(result).test_score is not None

    budgeted = PipelineExecutor(
        classification_task, config, run_id="budget", limits=ExecutionLimits(max_experiments=1)
    )
    budgeted.prepare(binary_frame.drop(columns="when"))
    summary = budgeted.run([valid, ExperimentSpec(model="not-a-model")])
    assert summary.n_success == 1
    assert summary.n_skipped == 1
    assert summary.results[-1].status is ExperimentStatus.SKIPPED


def test_executor_unknown_model_failure_is_returned_not_raised(
    binary_frame, classification_task, config
):
    executor = PipelineExecutor(classification_task, config, run_id="failure")
    executor.prepare(binary_frame.drop(columns="when"))
    result = executor.execute(ExperimentSpec(model="not-a-model"))
    assert result.status is ExperimentStatus.FAILED
    assert result.error_type
    assert "unknown model" in result.error
