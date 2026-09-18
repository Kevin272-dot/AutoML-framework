"""Background task execution.

Uses Celery+Redis when settings.job_backend == "celery" (requires Docker stack).
Falls back to "eager" in-process execution (separate thread) when Redis/Celery is
unavailable — same task functions, different dispatch. Job state is always
persisted in the database either way.
"""

import threading

from app.config import get_settings


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


def dispatch_discovery_search(request_id: str, job_id: str, db_factory) -> str:
    settings = get_settings()
    if settings.job_backend == "celery":
        from app.workers.tasks import discovery_search_task

        discovery_search_task.delay(request_id, job_id)
        return "queued"
    from app.discovery.service import run_discovery_search

    _run_in_thread(run_discovery_search, request_id, job_id, db_factory)
    return "eager"
