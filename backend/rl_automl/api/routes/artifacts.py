"""Artifact endpoints.

Downloads are served from the run's recorded path, but never *trusting* that path: the
resolved file must live inside the configured artifact tree. A run record is data, and data
that has been on disk is not the same as data this process just produced.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from rl_automl.api.dependencies import get_manager
from rl_automl.api.run_manager import RunManager
from rl_automl.api.schemas import ArtifactInfo, ArtifactList
from rl_automl.core.errors import RunStateError
from rl_automl.core.logging import get_logger

logger = get_logger("api.routes.artifacts")

router = APIRouter(prefix="/runs", tags=["artifacts"])

MEDIA_TYPES = {
    "model": "application/zip",
    "results": "application/zip",
}
DOWNLOAD_NAMES = {
    "model": "model_package.zip",
    "results": "results.zip",
}


def _describe(run_id: str, kind: str, manager: RunManager) -> ArtifactInfo:
    path = manager.artifact_path(run_id, kind)
    return ArtifactInfo(
        kind=kind,
        url=f"/runs/{run_id}/artifacts/{kind}.zip",
        sha256=_recorded_checksum(run_id, kind, manager),
        size_bytes=path.stat().st_size,
    )


def _recorded_checksum(run_id: str, kind: str, manager: RunManager) -> str | None:
    run = manager.get_run(run_id)
    if kind == "model":
        return run.artifacts.model_zip_sha256
    return run.artifacts.results_zip_sha256


@router.get("/{run_id}/artifacts", response_model=ArtifactList)
def list_artifacts(run_id: str, manager: RunManager = Depends(get_manager)) -> ArtifactList:
    try:
        run = manager.get_run(run_id)
    except RunStateError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.to_dict()) from exc

    available: list[ArtifactInfo] = []
    for kind in ("model", "results"):
        try:
            available.append(_describe(run_id, kind, manager))
        except RunStateError:
            continue
    return ArtifactList(run_id=run_id, verified=run.artifacts.verified, artifacts=available)


def _serve(run_id: str, kind: str, manager: RunManager) -> FileResponse:
    try:
        path: Path = manager.artifact_path(run_id, kind)
    except RunStateError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.to_dict()) from exc

    return FileResponse(
        path,
        media_type=MEDIA_TYPES.get(kind, "application/octet-stream"),
        filename=DOWNLOAD_NAMES.get(kind, path.name),
    )


@router.get("/{run_id}/artifacts/model.zip")
def download_model(run_id: str, manager: RunManager = Depends(get_manager)) -> FileResponse:
    return _serve(run_id, "model", manager)


@router.get("/{run_id}/artifacts/results.zip")
def download_results(run_id: str, manager: RunManager = Depends(get_manager)) -> FileResponse:
    return _serve(run_id, "results", manager)


__all__ = ["router"]
