"""Background task execution.

Uses Celery+Redis when settings.job_backend == "celery" (requires Docker stack).
Falls back to "eager" in-process execution (separate thread) when Redis/Celery is
unavailable — same task functions, different dispatch. Job state is always
persisted in the database either way.
"""

import json
import subprocess
import sys
import threading
from pathlib import Path

from app.config import get_settings
from app.models import Job


def get_celery_app():
    from app.workers.celery_app import celery_app

    return celery_app


def _run_in_thread(target, *args) -> None:
    threading.Thread(target=target, args=args, daemon=True).start()


def dispatch_download(dataset_id: str, job_id: str, db_factory) -> str:
    settings = get_settings()
    if settings.job_backend == "celery":
        from app.workers.tasks import download_dataset_task

        download_dataset_task.delay(dataset_id, job_id)
        return "queued"
    from app.datasets.download import run_download

    _run_in_thread(run_download, dataset_id, job_id, db_factory)
    return "eager"


def dispatch_eda(dataset_id: str, job_id: str, db_factory) -> str:
    settings = get_settings()
    if settings.job_backend == "celery":
        from app.workers.tasks import eda_task

        eda_task.delay(dataset_id, job_id)
        return "queued"
    from app.datasets.eda import run_eda

    _run_in_thread(run_eda, dataset_id, job_id, db_factory)
    return "eager"


def dispatch_preprocess(dataset_id: str, job_id: str, db_factory) -> str:
    settings = get_settings()
    if settings.job_backend == "celery":
        from app.workers.tasks import preprocess_task

        preprocess_task.delay(dataset_id, job_id)
        return "queued"
    from app.datasets.preprocess import run_preprocessing

    _run_in_thread(run_preprocessing, dataset_id, job_id, db_factory)
    return "eager"


def dispatch_automl_run(run_id: str, job_id: str, db_factory) -> str:
    """Eager profile: execute in a DETACHED subprocess so backend restarts
    (--reload, deploys) never kill an in-flight training run. The pid is recorded
    on the job so recovery can tell a live run from a dead one."""
    settings = get_settings()
    if settings.job_backend == "celery":
        from app.workers.tasks import automl_run_task

        automl_run_task.delay(run_id, job_id)
        return "queued"

    import json
    import subprocess
    import sys

    proc = subprocess.Popen(
        [sys.executable, "-m", "app.automl.exec_one", run_id, job_id],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Record the pid on the job so recovery can check liveness.
    session = db_factory()
    try:
        job = session.get(Job, job_id)
        if job is not None:
            detail: dict = json.loads(job.detail) if job.detail and job.detail.startswith("{") else {}
            detail["pid"] = proc.pid
            job.detail = json.dumps(detail)
            session.commit()
    finally:
        session.close()
    return "subprocess"


def dispatch_discovery_search(request_id: str, job_id: str, db_factory) -> str:
    settings = get_settings()
    if settings.job_backend == "celery":
        from app.workers.tasks import discovery_search_task

        discovery_search_task.delay(request_id, job_id)
        return "queued"
    from app.discovery.service import run_discovery_search

    _run_in_thread(run_discovery_search, request_id, job_id, db_factory)
    return "eager"
