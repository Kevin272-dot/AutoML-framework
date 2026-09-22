"""Dataset endpoints.

Nothing reaches the rest of the system without passing the upload gate in
:mod:`rl_automl.dataset.validation`; these handlers only decide *where* the bytes land and
what the client is told afterwards.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from rl_automl.api.dependencies import get_manager
from rl_automl.api.run_manager import RunManager, StoredDataset
from rl_automl.api.schemas import (
    ArchiveMemberRejection,
    DatasetArchiveResponse,
    DatasetFromSourceRequest,
    DatasetSourceInfo,
    DatasetSummary,
)
from rl_automl.core.errors import ConfigError, DatasetValidationError, SecurityError
from rl_automl.core.logging import get_logger
from rl_automl.dataset.archive import extract_tabular_members, looks_like_archive
from rl_automl.dataset.loaders import list_sources

logger = get_logger("api.routes.datasets")

router = APIRouter(prefix="/datasets", tags=["datasets"])

UPLOAD_CHUNK_BYTES = 1 << 20


def _summarise(manager: RunManager, record: StoredDataset) -> DatasetSummary:
    return DatasetSummary(
        dataset_id=record.dataset_id,
        name=record.name,
        path=record.path,
        n_rows=record.n_rows,
        n_cols=record.n_cols,
        sha256=record.sha256,
        columns=list(record.columns),
        warnings=list(record.warnings),
        preview=manager.dataset_preview(record),
        created_at=record.created_at,
    )


def _copy_within_limit(upload: UploadFile, limit_bytes: int) -> Path:
    """Stream the upload to a temporary file, aborting as soon as it exceeds the cap.

    The limit is enforced while writing rather than after, so an oversized body never
    occupies disk space in the first place.
    """
    suffix = Path(upload.filename or "upload.csv").suffix
    written = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        while chunk := upload.file.read(UPLOAD_CHUNK_BYTES):
            written += len(chunk)
            if written > limit_bytes:
                raise SecurityError(
                    f"upload exceeds the {limit_bytes / 1024 / 1024:.0f} MB limit",
                    size_bytes=written,
                    limit_bytes=limit_bytes,
                )
            handle.write(chunk)
        return Path(handle.name)


@router.post("", response_model=DatasetSummary, status_code=status.HTTP_201_CREATED)
def upload_dataset(
    file: UploadFile = File(...),
    manager: RunManager = Depends(get_manager),
) -> DatasetSummary:
    """Upload a dataset file. It is validated, sanitised, then stored cleaned."""
    limit_bytes = int(manager.config.security.max_upload_mb * 1024 * 1024)
    temporary = _copy_within_limit(file, limit_bytes)
    try:
        record = manager.register_upload(temporary, file.filename or temporary.name)
    except (SecurityError, DatasetValidationError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.to_dict()) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _summarise(manager, record)


@router.post("/from-source", response_model=DatasetSummary, status_code=status.HTTP_201_CREATED)
def dataset_from_source(
    payload: DatasetFromSourceRequest,
    manager: RunManager = Depends(get_manager),
) -> DatasetSummary:
    """Register one of the bundled or synthetic datasets by name."""
    try:
        record = manager.register_source(payload.source, name=payload.name)
    except ConfigError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.to_dict()) from exc
    return _summarise(manager, record)


@router.post("/archive", response_model=DatasetArchiveResponse, status_code=status.HTTP_201_CREATED)
def upload_archive(
    file: UploadFile = File(...),
    manager: RunManager = Depends(get_manager),
) -> DatasetArchiveResponse:
    """Unpack an archive and register every usable table inside it.

    The archive is guarded on the way in (see :mod:`rl_automl.dataset.archive`), and each
    member it yields is then put through the ordinary single-file gate before it is stored,
    so an unpacked file has no more privilege than an uploaded one.
    """
    limit_bytes = int(manager.config.security.max_upload_mb * 1024 * 1024)
    temporary = _copy_within_limit(file, limit_bytes)
    try:
        if not looks_like_archive(temporary):
            raise DatasetValidationError(
                "that upload is not a readable ZIP archive; use POST /datasets for a single table",
                filename=file.filename,
            )

        with tempfile.TemporaryDirectory(prefix="automl-archive-") as workspace:
            extraction = extract_tabular_members(
                temporary, workspace, manager.config.security
            )
            registered: list[DatasetSummary] = []
            rejected = [
                ArchiveMemberRejection(name=item.name, reason=item.reason)
                for item in extraction.rejected
            ]
            for member in extraction.members:
                try:
                    record = manager.register_upload(member.path, member.name)
                except (SecurityError, DatasetValidationError) as exc:
                    rejected.append(
                        ArchiveMemberRejection(name=member.name, reason=exc.message)
                    )
                    continue
                registered.append(_summarise(manager, record))

            if not registered:
                raise DatasetValidationError(
                    "no table inside the archive passed validation",
                    rejected=[item.model_dump() for item in rejected],
                )
    except (SecurityError, DatasetValidationError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.to_dict()) from exc
    finally:
        temporary.unlink(missing_ok=True)

    return DatasetArchiveResponse(
        archive_name=file.filename or temporary.name,
        datasets=registered,
        rejected=rejected,
    )


@router.get("", response_model=list[DatasetSummary])
def list_datasets(manager: RunManager = Depends(get_manager)) -> list[DatasetSummary]:
    return [_summarise(manager, record) for record in manager.list_datasets()]


@router.get("/sources", response_model=list[DatasetSourceInfo])
def list_dataset_sources(manager: RunManager = Depends(get_manager)) -> list[DatasetSourceInfo]:
    """Names accepted by ``POST /datasets/from-source`` and ``POST /runs``.

    Declared before ``/{dataset_id}`` so the literal path is matched first; otherwise the
    dynamic route would swallow "sources" as a dataset id.
    """
    return [
        DatasetSourceInfo(**source)
        for source in list_sources(cache_dir=manager.config.remote_dataset_dir())
    ]


@router.get("/{dataset_id}", response_model=DatasetSummary)
def get_dataset(dataset_id: str, manager: RunManager = Depends(get_manager)) -> DatasetSummary:
    return _summarise(manager, manager.get_dataset(dataset_id))


__all__ = ["router"]
