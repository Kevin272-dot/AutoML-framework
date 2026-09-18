"""Celery task wrappers. Each task opens its own DB session (workers are separate
processes; sessions must never be shared)."""

from app.db import SessionLocal
from app.datasets.download import run_download
from app.datasets.eda import run_eda
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.tasks.download_dataset_task", bind=True, max_retries=0)
def download_dataset_task(self, dataset_id: str, job_id: str):
    run_download(dataset_id, job_id, SessionLocal)


@celery_app.task(name="app.workers.tasks.eda_task", bind=True, max_retries=0)
def eda_task(self, dataset_id: str, job_id: str):
    run_eda(dataset_id, job_id, SessionLocal)


@celery_app.task(name="app.workers.tasks.preprocess_task", bind=True, max_retries=0)
def preprocess_task(self, dataset_id: str, job_id: str):
    from app.datasets.preprocess import run_preprocessing

    run_preprocessing(dataset_id, job_id, SessionLocal)


@celery_app.task(name="app.workers.tasks.discovery_search_task", bind=True, max_retries=0)
def discovery_search_task(self, request_id: str, job_id: str):
    from app.discovery.service import run_discovery_search

    run_discovery_search(request_id, job_id, SessionLocal)
