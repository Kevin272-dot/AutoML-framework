"""Dataset endpoints: candidate detail, preview, selection, download, EDA."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal, get_db
from app.discovery.adapters.base import AdapterError
from app.discovery.adapters.huggingface import HuggingFaceAdapter
from app.discovery.service import select_dataset
from app.models import (
    Dataset,
    DatasetCandidate,
    DatasetFile,
    DataSource,
    DiscoveryRequest,
    EDAReport,
    Job,
)
from app.schemas import (
    ConfirmTargetIn,
    DatasetOut,
    EDAOut,
    JobOut,
    PreviewOut,
    SelectDatasetIn,
)
from app.models import utcnow
from app.workers.dispatch import dispatch_download

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _dataset_out(ds: Dataset, source: DataSource | None) -> DatasetOut:
    return DatasetOut(
        id=ds.id,
        name=ds.name,
        description=ds.description,
        source=source.name if source else None,
        source_dataset_id=ds.source_dataset_id,
        source_url=ds.source_url,
        license=ds.license,
        row_count=ds.row_count,
        column_count=ds.column_count,
        file_format=ds.file_format,
        selected_target=ds.selected_target,
        selected_task=ds.selected_task,
        created_at=ds.created_at,
    )


def _job_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        kind=job.kind,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        detail=job.detail,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str, db: Session = Depends(get_db)):
    cand = db.get(DatasetCandidate, candidate_id)
    if cand is None:
        raise HTTPException(status_code=404, detail="Dataset candidate not found.")
    source = db.get(DataSource, cand.source_id)
    return {
        "id": cand.id,
        "source": source.name if source else cand.source_id,
        "source_slug": source.slug if source else None,
        "source_dataset_id": cand.source_dataset_id,
        "name": cand.name,
        "description": cand.description,
        "url": cand.canonical_url,
        "license": cand.license,
        "owner": cand.owner,
        "tags": cand.tags or [],
        "date_start": cand.date_start,
        "date_end": cand.date_end,
        "row_count": cand.row_count,
        "column_count": cand.column_count,
        "file_format": cand.file_format,
        "file_size_bytes": cand.file_size_bytes,
        "columns": cand.columns_meta or [],
        "download_available": cand.download_available,
        "preview_available": cand.preview_available,
        "overall_score": cand.score_total,
        "score_components": cand.score_components,
        "raw_metadata_keys": list((cand.raw_metadata or {}).keys()),
        "retrieved_at": cand.retrieved_at,
    }


@router.get("/candidates/{candidate_id}/preview", response_model=PreviewOut)
async def preview_candidate(candidate_id: str, db: Session = Depends(get_db)):
    cand = db.get(DatasetCandidate, candidate_id)
    if cand is None:
        raise HTTPException(status_code=404, detail="Dataset candidate not found.")
    source = db.get(DataSource, cand.source_id)
    if source is None or source.adapter != "huggingface":
        raise HTTPException(status_code=400, detail="Preview is not available for this source.")
    adapter = HuggingFaceAdapter()
    try:
        columns, rows = await adapter.get_preview(cand.source_dataset_id, limit=get_settings().max_preview_rows)
    except AdapterError as exc:
        raise HTTPException(status_code=502, detail={"code": exc.code, "message": exc.message}) from exc
    return PreviewOut(
        candidate_id=cand.id,
        columns=columns,
        dtypes={},
        rows=rows,
        row_count_total=cand.row_count,
    )


@router.post("/select")
def select(payload: SelectDatasetIn, db: Session = Depends(get_db)):
    cand = db.get(DatasetCandidate, payload.candidate_id)
    if cand is None:
        raise HTTPException(status_code=404, detail="Dataset candidate not found.")
    if not cand.download_available:
        raise HTTPException(status_code=400, detail="This dataset does not expose downloadable files.")
    request = db.get(DiscoveryRequest, cand.request_id)
    dataset, download_job, eda_job = select_dataset(db, cand, request.project_id if request else None)
    if request:
        request.status = "DATASET_SELECTED"
    db.commit()
    dispatch_download(dataset.id, download_job.id, SessionLocal)
    return {
        "dataset_id": dataset.id,
        "download_job_id": download_job.id,
        "eda_job_id": eda_job.id,
        "status": "DATASET_SELECTED",
    }


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    source = db.get(DataSource, ds.source_id) if ds.source_id else None
    return _dataset_out(ds, source)


@router.get("/{dataset_id}/files")
def get_dataset_files(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    files = db.query(DatasetFile).filter(DatasetFile.dataset_id == ds.id).all()
    return [
        {
            "id": f.id,
            "file_name": f.file_name,
            "file_format": f.file_format,
            "file_size_bytes": f.file_size_bytes,
            "file_hash_sha256": f.file_hash_sha256,
            "source_file_url": f.source_file_url,
            "validated": f.validated,
            "validation_report": f.validation_report,
        }
        for f in files
    ]


@router.get("/{dataset_id}/jobs", response_model=list[JobOut])
def get_dataset_jobs(dataset_id: str, db: Session = Depends(get_db)):
    jobs = db.query(Job).filter(Job.dataset_id == dataset_id).order_by(Job.created_at).all()
    return [_job_out(j) for j in jobs]


@router.get("/{dataset_id}/eda", response_model=EDAOut)
def get_eda(dataset_id: str, db: Session = Depends(get_db)):
    report = (
        db.query(EDAReport)
        .filter(EDAReport.dataset_id == dataset_id)
        .order_by(EDAReport.created_at.desc())
        .first()
    )
    if report is None:
        raise HTTPException(status_code=404, detail="EDA report not available yet.")
    r = report.report
    return EDAOut(
        dataset_id=dataset_id,
        shape=tuple(r.get("shape", [0, 0])),
        column_stats=r.get("column_stats", []),
        duplicate_rows=r.get("duplicate_rows", 0),
        correlation_matrix=r.get("correlation_matrix"),
        class_balance=r.get("class_balance"),
        quality_warnings=r.get("quality_warnings", []),
        target_candidates=r.get("target_candidates", []),
        suggested_task=r.get("suggested_task"),
        suggested_target=r.get("suggested_target"),
    )


@router.post("/{dataset_id}/confirm-target")
def confirm_target(dataset_id: str, payload: ConfirmTargetIn, db: Session = Depends(get_db)):
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    report = (
        db.query(EDAReport).filter(EDAReport.dataset_id == dataset_id).order_by(EDAReport.created_at.desc()).first()
    )
    valid_columns = {c["name"] for c in (report.report.get("column_stats", []) if report else [])}
    if valid_columns and payload.target not in valid_columns:
        raise HTTPException(status_code=400, detail=f"Column '{payload.target}' does not exist in this dataset.")
    ds.selected_target = payload.target
    ds.selected_task = payload.task
    ds.selected_at_target = utcnow()
    db.commit()
    return {"dataset_id": ds.id, "selected_target": ds.selected_target, "selected_task": ds.selected_task}
