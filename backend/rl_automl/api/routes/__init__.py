"""HTTP routers, grouped by resource."""

from __future__ import annotations

from rl_automl.api.routes import artifacts, datasets, models, runs

ROUTERS = (datasets.router, models.router, runs.router, artifacts.router)

__all__ = ["ROUTERS", "artifacts", "datasets", "models", "runs"]
