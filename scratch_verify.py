"""Verify per-task presets and compile the whole package (temporary scratch script)."""

from __future__ import annotations

import compileall
import sys

from rl_automl.core.types import TaskType
from rl_automl.search.hyperparameters import validate_presets
from rl_automl.search.model_registry import REGISTRY

for key in ("logistic_regression", "random_forest", "extra_trees", "xgboost", "lightgbm", "mlp"):
    spec = REGISTRY.get(key)
    print(
        f"{key:<20} clf={spec.n_presets(TaskType.CLASSIFICATION)} "
        f"reg={spec.n_presets(TaskType.REGRESSION)}"
    )

print("xgboost clf presets:", [p.name for p in REGISTRY.get("xgboost").presets_for(TaskType.CLASSIFICATION)])
print("xgboost reg presets:", [p.name for p in REGISTRY.get("xgboost").presets_for(TaskType.REGRESSION)])
print("random_forest clf presets:", [p.name for p in REGISTRY.get("random_forest").presets_for(TaskType.CLASSIFICATION)])

problems = {
    spec.key: report.problems
    for spec in REGISTRY.all_specs()
    if not (report := validate_presets(spec)).ok
}
print("preset validation problems:", problems or "none")

ok = compileall.compile_dir("rl_automl", quiet=1, force=True)
print("compileall:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
