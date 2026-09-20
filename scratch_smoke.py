"""Smoke check for the layers written so far (temporary scratch script)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.metrics import direction_for, primary_metric_for
from rl_automl.core.types import TaskType
from rl_automl.dataset.fingerprint import FINGERPRINT_FEATURES, build_fingerprint
from rl_automl.dataset.loaders import load_source, synthetic_recipes
from rl_automl.dataset.profiler import profile_dataset
from rl_automl.search.model_registry import REGISTRY
from rl_automl.task.task_classifier import TaskClassifier

print("== config ==")
cfg = AutoMLConfig.load()
print("source:", cfg.source_path)
print("artifact dir:", cfg.artifacts_dir())
print("enabled models:", cfg.models.enabled)
print("fingerprint features:", len(FINGERPRINT_FEATURES))

print("\n== registry ==")
print("registered:", REGISTRY.keys())
for spec in REGISTRY.all_specs():
    ok, reason = spec.is_available()
    print(f"  {spec.key:<22} tasks={[t.value for t in spec.tasks]} presets={spec.n_presets()} avail={ok}")

print("\n== bundled sources ==")
for name in ("iris", "breast_cancer", "diabetes"):
    frame, source = load_source(name)
    print(f"  {name:<15} shape={frame.shape} target={source.target} metric={source.effective_metric()}")

print("\n== synthetic ==")
print(f"  {len(synthetic_recipes())} recipes")

print("\n== task classifier ==")
clf = TaskClassifier()
prompts = [
    ("Predict whether a customer will churn based on transaction history.", ["customer_id", "tenure", "monthly_charges", "churn"]),
    ("Estimate the median house value from demographic block features.", ["median_income", "house_age", "median_house_value"]),
    ("Segment our customers into groups based on purchasing behaviour.", ["spend", "frequency", "recency"]),
    ("Detect fraudulent transactions in the payment stream.", ["amount", "merchant", "is_fraud"]),
    ("Minimize RMSE when predicting daily energy demand.", ["temperature", "day_of_week", "energy_demand"]),
    ("Classify emails into spam or not spam.", ["subject_length", "sender_score", "label"]),
]
for text, cols in prompts:
    spec = clf.classify(text, columns=cols)
    print(f"  {spec.task_type.value:<22} metric={spec.metric:<10} dir={spec.metric_direction.value:<9} target={str(spec.target):<22} conf={spec.confidence:.2f}")

print("\n== profiling + fingerprint ==")
frame = pd.DataFrame(
    {
        "age": pd.Series([25, 40, 33, 52, 61, 29, 48, 37, 55, 44] * 20, dtype="float64"),
        "income": pd.Series(np.linspace(1000, 90000, 200)),
        "city": pd.Series(["london", "paris", "berlin", "rome", "madrid"] * 40),
        "signup_date": pd.Series(pd.date_range("2020-01-01", periods=200, freq="D")),
        "churn": pd.Series([0] * 160 + [1] * 40),
    }
)
frame.loc[0:12, "income"] = np.nan
profile = profile_dataset(frame, target="churn", config=cfg.dataset)
print("  ", profile.summary_line())
print("   numeric:", profile.numeric_features)
print("   categorical:", profile.categorical_features)
print("   datetime:", profile.datetime_features)
print("   missing_ratio:", round(profile.missing_ratio, 4), "dups:", profile.n_duplicate_rows)
print("   class dist:", profile.class_distribution)
print("   warnings:", profile.warnings)

fp = build_fingerprint(profile, TaskType.CLASSIFICATION)
print("   fingerprint len:", len(fp.values), "all finite:", all(np.isfinite(fp.values)))
print("   sample:", {k: round(v, 3) for k, v in list(fp.as_dict().items())[:6]})

print("\n== metric helpers ==")
print("  f1 dir:", direction_for("f1").value, "| rmse dir:", direction_for("rmse").value)
print("  primary cls:", primary_metric_for(TaskType.CLASSIFICATION))

print("\nSMOKE OK")
