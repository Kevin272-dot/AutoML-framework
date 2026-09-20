"""FastAPI application factory.

``create_app`` is a factory rather than a module-level singleton so a test can point an app
at a temporary artifact directory and get a fully isolated RunManager. The module-level
``app`` below exists for ``uvicorn rl_automl.api.app:app``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from rl_automl.api.routes import ROUTERS
from rl_automl.api.run_manager import RunManager
from rl_automl.api.schemas import HealthResponse
from rl_automl.core.config import AutoMLConfig
from rl_automl.core.errors import (
    AutoMLError,
    DatasetValidationError,
    NotApprovedError,
    RunStateError,
    SecurityError,
)
from rl_automl.core.logging import get_logger

logger = get_logger("api.app")

#: The bundled dashboard. Plain HTML next to this module, so it ships with the package and
#: needs no build step and no network access.
DASHBOARD_HTML = Path(__file__).parent / "static" / "index.html"

#: AutoML error type -> HTTP status. The taxonomy exists so this mapping can be exact.
_STATUS_BY_ERROR: tuple[tuple[type[AutoMLError], int], ...] = (
    (NotApprovedError, 409),
    (RunStateError, 409),
    (SecurityError, 400),
    (DatasetValidationError, 400),
)


def _status_for(exc: AutoMLError) -> int:
    for error_type, code in _STATUS_BY_ERROR:
        if isinstance(exc, error_type):
            return code
    return 500


def create_app(
    config: AutoMLConfig | None = None,
    *,
    manager_factory: Callable[[AutoMLConfig], RunManager] = RunManager,
) -> FastAPI:
    resolved = config or AutoMLConfig.load()
    resolved.ensure_directories()
    manager = manager_factory(resolved)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "api started",
            extra={"context": {"artifacts_dir": str(resolved.artifacts_dir())}},
        )
        try:
            yield
        finally:
            manager.shutdown()
            logger.info("api stopped")

    app = FastAPI(
        title="RL-AutoML",
        version="0.1.0",
        description=(
            "Problem-aware, dataset-aware experiment planning with a human approval gate. "
            "Nothing is trained until a run's plan is explicitly approved."
        ),
        lifespan=lifespan,
    )
    app.state.manager = manager
    app.state.config = resolved

    if resolved.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved.api.cors_origins),
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    for router in ROUTERS:
        app.include_router(router)

    @app.exception_handler(AutoMLError)
    async def _handle_automl_error(_request: Request, exc: AutoMLError) -> JSONResponse:
        return JSONResponse(status_code=_status_for(exc), content={"detail": exc.to_dict()})

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        """Serve the bundled single-page dashboard, which drives this same API."""
        if not DASHBOARD_HTML.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="the dashboard is not bundled with this installation",
            )
        return FileResponse(DASHBOARD_HTML, media_type="text/html")

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            runs=manager.run_count,
            datasets=manager.dataset_count,
        )

    return app


app = create_app()

__all__ = ["app", "create_app"]
