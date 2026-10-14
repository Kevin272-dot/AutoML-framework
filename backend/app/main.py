import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import automl as automl_router
from app.api import connections as connections_router
from app.api import datasets as datasets_router
from app.api import discovery as discovery_router
from app.api import jobs as jobs_router
from app.config import get_settings
from app.schemas import ErrorEnvelope

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("automl")

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Dataset Discovery + Automated Machine Learning",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never expose raw stack traces to the client."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorEnvelope(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred.",
            what_happened="The server hit an unexpected error while processing the request.",
            why="This is usually a bug in the application or an unavailable dependency.",
            what_to_do="Retry the operation. If it persists, check the backend logs.",
        ).model_dump(),
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.include_router(discovery_router.router)
app.include_router(datasets_router.router)
app.include_router(jobs_router.router)
app.include_router(connections_router.router)
app.include_router(automl_router.router)


@app.on_event("startup")
def create_tables():
    """Dev convenience: create tables when they don't exist. Alembic handles migrations."""
    from app.db import Base, engine

    import app.models  # noqa: F401  (registers ORM models on Base metadata)

    Base.metadata.create_all(bind=engine)
    _ensure_schema_upgrades(engine)
    _recover_stale_jobs()


def _job_pid(job) -> int | None:
    import json

    try:
        detail = json.loads(job.detail) if job.detail else {}
        pid = detail.get("pid")
        return int(pid) if pid else None
    except (ValueError, TypeError):
        return None


def _pid_alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, ValueError):
        return False


def _ensure_schema_upgrades(engine) -> None:
    """Lightweight column additions for existing dev databases (create_all does not
    ALTER existing tables). Production should use Alembic migrations instead."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "automl_runs" not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns("automl_runs")}
    with engine.begin() as conn:
        if "approved" not in existing:
            conn.execute(text("ALTER TABLE automl_runs ADD COLUMN approved JSON"))


def _recover_stale_jobs():
    """Eager in-process jobs die with the backend process; mark them failed so
    runs are never permanently blocked by a job that no longer exists.
    AutoML run mirrors are reset from the orchestrator's authoritative run file,
    so a run interrupted mid-execution returns to its approval gate.
    (With JOB_BACKEND=celery, jobs survive restarts and are left alone.)"""
    import json
    from pathlib import Path

    from app.config import get_settings
    from app.models import AutoMLRun, Job, utcnow

    if get_settings().job_backend != "eager":
        return
    from app.db import SessionLocal

    IN_FLIGHT_RUN_STATUSES = {"queued", "executing", "evaluating", "comparing", "selecting", "packaging"}

    db = SessionLocal()
    try:
        stale = db.query(Job).filter(Job.status.in_(["QUEUED", "RUNNING"])).all()
        for job in stale:
            # AutoML runs execute in detached subprocesses; if the pid is still
            # alive the run legitimately outlived a backend restart — leave it.
            pid = _job_pid(job)
            if pid is not None and _pid_alive(pid):
                continue
            job.status = "FAILED"
            job.error_code = "STALE_JOB"
            job.error_message = "The process running this job is gone. Re-run it."
            job.finished_at = utcnow()

        settings = get_settings()
        runs_root = Path(settings.local_storage_dir).parent / "automl-artifacts" / "runs"
        stuck_runs = (
            db.query(AutoMLRun)
            .filter(AutoMLRun.status.in_(IN_FLIGHT_RUN_STATUSES))
            .all()
        )
        for run_row in stuck_runs:
            # A live detached subprocess owns this run — leave it alone.
            if run_row.job_id:
                job = db.get(Job, run_row.job_id)
                pid = _job_pid(job) if job else None
                if pid is not None and _pid_alive(pid):
                    continue
            run_file = runs_root / run_row.id / "run.json"
            recovered = None
            if run_file.is_file():
                try:
                    recovered = json.loads(run_file.read_text()).get("status")
                except (OSError, ValueError):
                    recovered = None
            if recovered == "awaiting_approval" or (
                recovered in (None, "created") and run_row.recommendations
            ):
                # The orchestrator never started (or never finished) execution;
                # its saved state is still at the approval gate.
                run_row.status = "awaiting_approval"
                run_row.error = None
                run_row.job_id = None
            else:
                run_row.status = "failed"
                run_row.error = "The backend restarted while this run was executing."
        db.commit()
    finally:
        db.close()
