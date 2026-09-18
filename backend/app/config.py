from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "INTELLIGENT AUTOML"
    # Postgres via docker-compose; SQLite fallback works for dev without Docker.
    database_url: str = "sqlite:///./automl.db"
    redis_url: str = "redis://localhost:6379/0"

    # S3 / MinIO
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "automl"
    s3_secret_key: str = "automl-dev-secret"
    s3_bucket: str = "automl-datasets"
    # When False, artifacts go to local storage (dev profile without Docker).
    use_s3_storage: bool = False
    local_storage_dir: str = "./data/artifacts"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # External service tuning
    audit_cache_ttl_seconds: int = 3600
    http_timeout_seconds: float = 20.0
    max_dataset_download_bytes: int = 512 * 1024 * 1024  # 512 MB
    max_preview_rows: int = 100

    # Hugging Face endpoints (documented public APIs only)
    hf_api_base: str = "https://huggingface.co/api"
    hf_datasets_server_base: str = "https://datasets-server.huggingface.co"

    # Background jobs: "celery" requires Redis; "eager" runs tasks in-process (dev fallback).
    job_backend: str = "eager"

    # Cap on concurrently stored (downloaded) datasets per workspace.
    max_stored_datasets: int = 25


@lru_cache
def get_settings() -> Settings:
    return Settings()
