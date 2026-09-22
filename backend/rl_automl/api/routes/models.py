"""Model catalog: what this installation can actually train.

The registry knows every algorithm; the config decides which are enabled; the environment
decides which optional dependencies are importable. A client that has to *attempt* a run to
discover that clustering has no enabled model is a client that wastes a plan. This endpoint
answers the question directly.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rl_automl.api.dependencies import get_manager
from rl_automl.api.run_manager import RunManager
from rl_automl.api.schemas import ModelCatalog
from rl_automl.core.types import TaskType
from rl_automl.search.model_registry import get_registry

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelCatalog)
def list_models(manager: RunManager = Depends(get_manager)) -> ModelCatalog:
    """Enabled model keys, grouped by the task types they can serve."""
    registry = get_registry()
    enabled = list(manager.config.models.enabled)
    by_task = {
        task.value: [spec.key for spec in registry.for_task(task, enabled=enabled)]
        for task in TaskType
    }
    return ModelCatalog(
        enabled=enabled,
        runnable_tasks=[task for task, keys in by_task.items() if keys],
        by_task=by_task,
    )


__all__ = ["router"]
