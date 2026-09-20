"""Smoke check for the execution layer (temporary scratch script)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.errors import RunStateError
from rl_automl.core.types import ExperimentSpec, ExperimentStatus, TaskType
from rl_automl.dataset.loaders import load_source
from rl_automl.dataset.profiler import profile_dataset
from rl_automl.execution.executor import ExecutionLimits, PipelineExecutor
from rl_automl.search.model_registry import get_spec
from rl_automl.task.task_classifier import TaskClassifier

cfg = AutoMLConfig.load()
cfg.runtime.artifacts_dir = "artifacts"
cfg.dataset.min_rows = 30

frame, source = load_source("breast_cancer")
statement = f"Predict whether a patient's diagnosis is malignant based on cell measurements."
task = TaskClassifier().classify(statement, columns=list(frame.columns),
                                class_distribution=profile_dataset(frame, source.target, cfg.dataset).class_distribution)
print("task:", task.task_type.value, "| metric:", task.metric, "| target:", task.target)

print("\n== experiments ==")
experiments = [
    ExperimentSpec(model="logistic_regression", preset_index=0,
                   preprocessing=["impute_numeric", "scale_standard", "encode_categorical"]),
    ExperimentSpec(model="random_forest", preset_index=1, preprocessing=["impute_numeric"]),
    ExperimentSpec(model="lightgbm", preset_index=1, preprocessing=["impute_numeric", "encode_categorical"]),
    ExperimentSpec(model="mlp", preset_index=1,
                   preprocessing=["impute_numeric", "scale_standard", "encode_categorical"]),
    ExperimentSpec(model="totally_not_a_model", preset_index=0),
    ExperimentSpec(model="svm", preset_index=0,
                   preprocessing=["impute_numeric", "scale_standard", "encode_categorical"]),
]

ex = PipelineExecutor(task, cfg, run_id="smoke_run", seed=cfg.runtime.seed,
                      limits=ExecutionLimits(max_experiments=10, time_budget_s=600, memory_budget_mb=6000))
splits = ex.prepare(frame)
print("splits:", splits.summary())

print("\n== test-set guard (before finalize) ==")
try:
    ex.evaluate_test(ex.execute(experiments[0]))
    print("  FAIL: test access was allowed before finalisation")
except RunStateError as exc:
    print("  OK: blocked ->", str(exc)[:80])

print("\n== execute ==")
summary = ex.run(experiments)
for r in summary.results:
    if r.succeeded:
        print(f"  {r.experiment.model:<20} {r.primary_metric}={r.validation_score:.4f} "
              f"train={r.training_time_s:.2f}s mem={r.peak_memory_mb:.0f}MB "
              f"size={r.model_size_bytes/1024:.1f}KB params={r.n_parameters}")
    else:
        print(f"  {r.experiment.model:<20} {r.status.value.upper():<8} {r.error_type}: {(r.error or '')[:90]}")
print("summary:", summary.to_dict())

print("\n== finalize + test ==")
ex.begin_finalize()
best = summary.best_by_validation()
print("best by validation:", best.experiment.model, best.validation_score)
for r in summary.successful:
    ex.evaluate_test(r)
    print(f"  {r.experiment.model:<20} val={r.validation_score:.4f} test={r.test_score:.4f} "
          f"| acc={r.test_metrics.get('accuracy', float('nan')):.4f} roc_auc={r.test_metrics.get('roc_auc', float('nan')):.4f}")
print("test evaluations:", ex.test_evaluations)
print("test touched during search:", ex.test_was_touched_during_search)

print("\n== regression path ==")
rframe, rsource = load_source("diabetes")
rprofile = profile_dataset(rframe, "disease_progression", cfg.dataset)
rtask = TaskClassifier().classify(
    "Estimate disease progression from clinical measurements.",
    columns=list(rframe.columns),
    class_distribution=rprofile.class_distribution,
)
print("  task:", rtask.task_type.value, "| metric:", rtask.metric, "| target:", rtask.target)
rex = PipelineExecutor(rtask, cfg, run_id="smoke_reg", limits=ExecutionLimits(max_experiments=5))
rex.prepare(rframe)
rsum = rex.run([
    ExperimentSpec(model="linear_regression", preset_index=0, preprocessing=["impute_numeric"]),
    ExperimentSpec(model="random_forest", preset_index=0, preprocessing=["impute_numeric"]),
    ExperimentSpec(model="xgboost", preset_index=0, preprocessing=["impute_numeric"]),
])
rex.begin_finalize()
for r in rsum.results:
    if r.succeeded:
        rex.evaluate_test(r)
        r2 = r.validation_metrics.get("r2")
        print(f"  {r.experiment.model:<20} rmse={r.validation_score:.3f} test_rmse={r.test_score:.3f} "
              f"r2={r2 if r2 is None else round(r2, 3)} size={r.model_size_bytes/1024:.1f}KB")
    else:
        print(f"  {r.experiment.model:<20} FAILED {r.error}")

print("\n== clustering path ==")
cdf = pd.DataFrame(np.random.default_rng(0).normal(size=(300, 6)),
                   columns=[f"f{i}" for i in range(6)])
ctask = TaskClassifier().classify("Segment customers into groups based on behaviour.", columns=list(cdf.columns))
cex = PipelineExecutor(ctask, cfg, run_id="smoke_clu", limits=ExecutionLimits(max_experiments=5))
cex.prepare(cdf)
csum = cex.run([
    ExperimentSpec(model="kmeans", preset_index=1, preprocessing=["impute_numeric", "scale_standard"]),
    ExperimentSpec(model="agglomerative", preset_index=0, preprocessing=["impute_numeric", "scale_standard"]),
])
for r in csum.results:
    if r.succeeded:
        print(f"  {r.experiment.model:<20} silhouette={r.validation_score:.4f} "
              f"n_clusters={r.validation_metrics.get('n_clusters')} | {'; '.join(r.warnings)[:70]}")
    else:
        print(f"  {r.experiment.model:<20} FAILED {r.error}")

print("\n== label encoding path (string target) ==")
sdf = pd.DataFrame({
    "tenure": np.random.default_rng(1).integers(1, 60, 400),
    "charges": np.random.default_rng(2).normal(50, 20, 400).round(2),
    "plan": np.random.default_rng(3).choice(["basic", "premium", "gold"], 400),
    "outcome": np.random.default_rng(4).choice(["stayed", "churned"], 400),
})
stask = TaskClassifier().classify("Predict whether a customer will churn.", columns=list(sdf.columns))
sex = PipelineExecutor(stask, cfg, run_id="smoke_labels", limits=ExecutionLimits(max_experiments=5))
sex.prepare(sdf)
print("  needs label encoding:", sex.label_encoding_required, "| classes:", sex.class_labels)
ssum = sex.run([
    ExperimentSpec(model="lightgbm", preset_index=0, preprocessing=["impute_numeric", "encode_categorical"]),
    ExperimentSpec(model="logistic_regression", preset_index=0,
                   preprocessing=["impute_numeric", "scale_standard", "encode_categorical"]),
])
sex.begin_finalize()
for r in ssum.successful:
    sex.evaluate_test(r)
    v = r.validation_score
    t = r.test_score
    print(f"  {r.experiment.model:<20} f1={None if v is None else round(v, 4)} "
          f"test_f1={None if t is None else round(t, 4)} skipped={list(r.validation_metrics.get('__x', '') or [])}")
    print(f"      val_metrics={ {k: round(x, 4) for k, x in r.validation_metrics.items()} }")

assert sex.decode_labels(np.array([0, 1])).tolist() == ["churned", "stayed"], sex.decode_labels(np.array([0, 1]))

print("\n== budget enforcement ==")
bex = PipelineExecutor(task, cfg, run_id="smoke_budget", limits=ExecutionLimits(max_experiments=1))
bex.prepare(frame)
bsum = bex.run([
    ExperimentSpec(model="logistic_regression", preset_index=0,
                   preprocessing=["impute_numeric", "scale_standard", "encode_categorical"]),
    ExperimentSpec(model="random_forest", preset_index=0, preprocessing=["impute_numeric"]),
])
print("  executed:", bsum.n_success, "skipped:", bsum.n_skipped, "reason:", bsum.stopping_reason)
assert bsum.n_skipped == 1 and bsum.results[1].status is ExperimentStatus.SKIPPED

print("\nEXEC SMOKE OK")
