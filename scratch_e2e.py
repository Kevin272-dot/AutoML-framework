"""End-to-end vertical-slice smoke test (temporary scratch script).

Exercises: churn data -> task understanding -> profiling -> RL/surrogate planning ->
approval gate -> execution -> comparison -> selection -> packaging -> ZIP verification,
plus the two invariants that matter most (no training before approval; no test leakage).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.errors import NotApprovedError, RunStateError
from rl_automl.core.types import RunStatus
from rl_automl.dataset.loaders import load_source
from rl_automl.orchestrator import AutoMLOrchestrator, PlanRequest

STATEMENT = (
    "Predict whether a customer will churn based on transaction history and account usage."
)

cfg = AutoMLConfig.load()
cfg.runtime.artifacts_dir = "artifacts"
cfg.search.max_experiments = 8
cfg.search.min_experiments = 3
cfg.search.n_recommendations = 3
cfg.rl.enabled = True

frame, source = load_source("churn_demo")
print("== dataset ==")
print("  ", source.name, frame.shape, "| churn rate:", round(frame["churn"].mean(), 4))
print("   dtypes:", dict(frame.dtypes.astype(str).value_counts()))

orch = AutoMLOrchestrator(cfg)

print("\n== PHASE A: plan ==")
run = orch.plan(
    PlanRequest(
        problem_statement=STATEMENT,
        frame=frame,
        dataset_name="customers.csv",
        dataset_sha256="deadbeef",
    )
)
print("  status:", run.status.value)
print("  task:", run.task.task_type.value, "| metric:", run.task.metric,
      f"({run.task.metric_direction.value})", "| target:", run.task.target,
      "| confidence:", run.task.confidence)
print("  dataset:", run.dataset_profile.summary_line())
print("   numeric:", len(run.dataset_profile.numeric_features),
      "categorical:", len(run.dataset_profile.categorical_features),
      "datetime:", len(run.dataset_profile.datetime_features))
print("   class dist:", run.dataset_profile.class_distribution.n_classes,
      "| minority:", round(run.dataset_profile.class_distribution.minority_ratio, 3),
      "| imbalanced:", run.dataset_profile.class_distribution.is_imbalanced)
print("   profile warnings:", run.dataset_profile.warnings)
print("  planner:", run.recommendation_set.planner)
print("  requires_user_approval:", run.recommendation_set.requires_user_approval)
print("  estimated:", run.recommendation_set.estimated_compute)
for item in run.rl_recommendations:
    print(f"   #{item.rank} {item.model:<20} cost={item.expected_cost.value:<7} "
          f"expected={item.expected_performance:.4f} fs={item.feature_selection}")
    print(f"       pre={item.preprocessing}")
    print(f"       reason: {item.reason}")
print("  baseline:", run.recommendation_set.baseline_expectation)
print("  notes:", run.recommendation_set.notes)
print("  progress stages:", [event["stage"] for event in run.progress])

print("\n== invariant: plan phase produced no model artifacts ==")
run_dir = orch.run_dir(run.run_id)
model_files = list((run_dir / "package").rglob("*.joblib")) if (run_dir / "package").exists() else []
print("   joblib files after planning:", len(model_files))
print("   zip archives after planning:", list(run_dir.glob("*.zip")))
assert not model_files, "planning must not train or save models"
assert not list(run_dir.glob("*.zip")), "planning must not produce archives"

print("\n== invariant: execution without approval is refused ==")
unapproved = orch.load_run(run.run_id)
unapproved.approved_experiments = []
try:
    orch.execute(unapproved, frame=frame)
    print("   FAIL: execution proceeded without approval")
except NotApprovedError as exc:
    print("   OK: blocked ->", str(exc)[:80])

print("\n== approval gate ==")
try:
    orch.approve(run, selection=[1, 99])
    print("   FAIL: invalid rank accepted")
except RunStateError as exc:
    print("   OK: bad rank rejected ->", str(exc)[:70])

run = orch.approve(run, selection=[1, 2, 3])
print("  approved:", [e.model for e in run.approved_experiments])
print("  status:", run.status.value, "| approved_at:", run.approved_at)

# A deliberately failing pipeline, to prove failure isolation (spec §29).
print("\n== failure isolation ==")
from rl_automl.core.types import ExperimentSpec

run = orch.approve(
    run,
    selection="all",
    extra_experiments=[
        ExperimentSpec(model="svm", preset_index=0,
                       preprocessing=["impute_numeric", "scale_standard", "encode_categorical"],
                       hyperparameters={"C": 1.0, "kernel": "rbf", "gamma": "scale"}),
    ],
)
print("  approved with an extra pipeline:", [e.model for e in run.approved_experiments])

print("\n== PHASE B: execute ==")
run = orch.execute(run, frame=frame)
print("  status:", run.status.value)
print("  error:", run.error)

print("\n" + (run.comparison.table if run.comparison else "no comparison"))

print("\n  trade-offs:")
for note in (run.comparison.trade_offs if run.comparison else []):
    print("   -", note)

print("\n  best model:", run.best_model.model,
      "| validation:", round(run.best_model.validation_score, 4),
      "| test:", None if run.best_model.test_score is None else round(run.best_model.test_score, 4))
print("  reason:", run.best_model.reason)
print("  selection basis:", run.best_model.selection_basis)
assert "test" not in (run.best_model.reason or "").lower(), "reason must not cite test performance"

print("\n  pareto frontier:")
for point in run.pareto_frontier:
    print(f"   {point.model:<20} score={point.score:.4f} cost={point.cost_s:.2f}s "
          f"mem={point.peak_memory_mb:.0f}MB on_frontier={point.on_frontier}")

print("\n  search statistics:", run.search_statistics.model_dump(
    include={"n_experiments", "n_failed", "best_model", "stopping_reason", "total_training_time_s"}))

print("\n  failure isolation check:",
      [(r.experiment.model, r.status.value) for r in run.results])

print("\n== artifacts ==")
print("  model zip:", run.artifacts.model_zip)
print("  verified:", run.artifacts.verified, "| sha256:", run.artifacts.model_zip_sha256)
print("  results zip:", run.artifacts.results_zip)
print("  messages:", run.messages)

assert run.artifacts.verified, "the model archive must be verified before download"

print("\n== packaged contents ==")
import zipfile

with zipfile.ZipFile(run.artifacts.model_zip) as archive:
    for name in sorted(archive.namelist()):
        print("   ", name)

with zipfile.ZipFile(run.artifacts.results_zip) as archive:
    print("   results bundle:", sorted(archive.namelist()))

print("\n== results.json keys (§22) ==")
with zipfile.ZipFile(run.artifacts.results_zip) as archive:
    payload = json.loads(archive.read("results/results.json"))
required_keys = {"run_id", "task", "dataset_profile", "rl_recommendations",
                 "approved_experiments", "results", "best_model", "pareto_frontier",
                 "search_statistics", "artifacts"}
print("   present:", sorted(required_keys & set(payload)))
print("   missing:", sorted(required_keys - set(payload)))
assert not (required_keys - set(payload)), "results.json is missing required keys"

print("\n== clean-environment inference check ==")
package_dir = Path(run.artifacts.model_dir).resolve()
head = frame.drop(columns=["churn"]).head(5)
head.to_csv(package_dir / "inference" / "to_predict.csv", index=False)
script = textwrap.dedent(
    f"""
    import sys
    sys.path.insert(0, r"{package_dir / 'inference'}")
    from model_loader import load_model
    model = load_model(r"{package_dir}")
    print("  describe:", model.describe())
    rows = model.predict(r"{package_dir / 'inference' / 'to_predict.csv'}")
    print("  predictions:", list(rows))
    proba = model.predict_proba(r"{package_dir / 'inference' / 'to_predict.csv'}")
    print("  probabilities:", None if proba is None else proba.round(3).tolist())
    """
)
result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                        cwd=str(package_dir), timeout=300)
print(result.stdout.rstrip() or result.stderr[-2000:])

print("\n== generated README (first 45 lines) ==")
readme = (package_dir / "README.md").read_text(encoding="utf-8").splitlines()
print("\n".join("   " + line for line in readme[:45]))

print("\n== generated requirements.txt ==")
print("   ", (package_dir / "requirements.txt").read_text(encoding="utf-8").replace("\n", " | "))

print("\n== run persistence round trip ==")
reloaded = orch.load_run(run.run_id)
print("   status:", reloaded.status.value, "| best:", reloaded.best_model.model,
      "| results:", len(reloaded.results), "| progress events:", len(reloaded.progress))
print("   list_runs:", orch.list_runs()[:2])

assert reloaded.status is RunStatus.COMPLETE
print("\nEND-TO-END OK")
