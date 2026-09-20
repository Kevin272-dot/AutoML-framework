"""Shared FastAPI dependencies.

The manager is created once by :func:`rl_automl.api.app.create_app` and hung off
``app.state``. Routers reach it through this dependency instead of importing a module-level
singleton, so tests can build an app against a temporary artifact directory without any
global state leaking between them.
"""

from __future__ import annotations

from fastapi import Request

from rl_automl.api.run_manager import RunManager


def get_manager(request: Request) -> RunManager:
    manager = getattr(request.app.state, "manager", None)
    if manager is None:  # pragma: no cover - only if the app was built incorrectly
        raise RuntimeError("the application was created without a RunManager")
    return manager


__all__ = ["get_manager"]
