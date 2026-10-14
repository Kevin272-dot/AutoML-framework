"""Serving trained models: load the packaged model (run artifacts) and predict.

Uses the generated `model_loader.py` from the run's model package — the exact
runtime the platform ships for external integration — so API predictions match
package predictions exactly.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import get_settings


class ModelNotAvailable(Exception):
    def __init__(self, run_id: str, reason: str):
        self.run_id = run_id
        self.reason = reason
        super().__init__(reason)


def _package_dir(run_id: str) -> Path:
    settings = get_settings()
    return (
        Path(settings.local_storage_dir).parent
        / "automl-artifacts"
        / "runs"
        / run_id
        / "package"
        / "model_package"
    )


_loaded: dict[str, Any] = {}


def _load_model(run_id: str) -> Any:
    """Import the generated model_loader.py and return a LoadedModel (cached)."""
    if run_id in _loaded:
        return _loaded[run_id]

    package = _package_dir(run_id)
    loader_file = package / "inference" / "model_loader.py"
    if not loader_file.is_file():
        raise ModelNotAvailable(run_id, "This run has no packaged model yet.")

    try:
        spec = importlib.util.spec_from_file_location(f"model_loader_{run_id}", loader_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"model_loader_{run_id}"] = module
        spec.loader.exec_module(module)
        loaded = module.load_model(package)
    except Exception as exc:
        raise ModelNotAvailable(run_id, f"The packaged model failed to load: {exc}") from exc

    _loaded[run_id] = loaded
    return loaded


def get_features(run_id: str) -> dict[str, Any]:
    """Feature schema for building dynamic prediction forms."""
    package = _package_dir(run_id)
    schema_file = package / "metadata" / "feature_schema.json"
    if not schema_file.is_file():
        raise ModelNotAvailable(run_id, "This run has no packaged model yet.")
    schema = json.loads(schema_file.read_text())
    features = [
        {
            "name": f.get("name"),
            "kind": f.get("kind"),
            "required": True,
        }
        for f in schema.get("features", [])
    ]
    return {
        "task_type": schema.get("task_type"),
        "target": schema.get("target"),
        "features": features,
        "classes": schema.get("output", {}).get("classes"),
    }


def predict(run_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    loaded = _load_model(run_id)

    frame = _to_frame(records, loaded.feature_names)
    predictions = loaded.predict(frame)
    probabilities = loaded.predict_proba(frame)

    result: dict[str, Any] = {
        "predictions": [_safe(v) for v in predictions.tolist()],
        "count": len(records),
    }
    if probabilities is not None:
        result["probabilities"] = [[round(float(p), 6) for p in row] for row in probabilities.tolist()]
    classes = loaded.schema.get("output", {}).get("classes")
    if classes:
        result["classes"] = classes
    return result


def _to_frame(records: list[dict[str, Any]], feature_names: list[str]) -> pd.DataFrame:
    import pandas as pd

    if not records:
        raise ValueError("records must contain at least one row")
    return pd.DataFrame(records, columns=feature_names)


def _safe(value: Any) -> Any:
    import math

    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) else f
    return value
