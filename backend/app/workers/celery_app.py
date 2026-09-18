from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "automl",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    worker_max_tasks_per_child=20,
    task_time_limit=3600,
    task_soft_time_limit=3300,
)
