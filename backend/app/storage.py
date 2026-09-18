"""Artifact storage: S3-compatible (MinIO) or local filesystem fallback.

Backend-only credentials; the frontend never receives storage config.
"""

import hashlib
import os
import shutil
from pathlib import Path

import boto3

from app.config import get_settings


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 512), b""):
            h.update(chunk)
    return h.hexdigest()


def store_artifact(local_path: str, key: str) -> dict:
    """Move a downloaded file into object storage (or local storage in the dev profile).

    Returns {storage_key, storage_backend, size, hash}.
    """
    settings = get_settings()
    src = Path(local_path)
    size = src.stat().st_size
    digest = _sha256(src)

    if settings.use_s3_storage:
        client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
        )
        bucket = settings.s3_bucket
        existing = client.list_buckets().get("Buckets", [])
        if not any(b["Name"] == bucket for b in existing):
            client.create_bucket(Bucket=bucket)
        client.upload_file(str(src), bucket, key)
        src.unlink()
        return {"storage_key": f"s3://{bucket}/{key}", "storage_backend": "s3", "size": size, "hash": digest}

    dest_dir = Path(settings.local_storage_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return {"storage_key": str(dest), "storage_backend": "local", "size": size, "hash": digest}


def delete_artifact(storage_key: str, storage_backend: str) -> None:
    """Remove a stored artifact (best effort — missing files are ignored)."""
    settings = get_settings()
    if storage_backend == "s3":
        bucket, key = storage_key.removeprefix("s3://").split("/", 1)
        client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
        )
        client.delete_object(Bucket=bucket, Key=key)
        return
    path = Path(storage_key)
    if path.exists():
        path.unlink()
        # Clean up the dataset's now-empty artifact directory if possible.
        try:
            path.parent.rmdir()
        except OSError:
            pass


def open_artifact(storage_key: str, storage_backend: str):
    """Return a local file path for reading. Downloads from S3 to a temp file if needed."""
    settings = get_settings()
    if storage_backend == "local":
        return storage_key
    # s3://bucket/key -> download to temp
    bucket, key = storage_key.removeprefix("s3://").split("/", 1)
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )
    local = Path(settings.local_storage_dir) / "cache" / key
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        client.download_file(bucket, key, str(local))
    return str(local)
