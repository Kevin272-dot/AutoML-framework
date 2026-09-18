"""Jobs + dashboard endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import DataSource, Dataset, DiscoveryRequest, Job, SourceApproval
from app.schemas import DashboardStatsOut, DatasetOut, DiscoveryRequestOut, JobOut, ParsedRequirements

router = APIRouter(prefix="/api", tags=["jobs"])


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


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_out(job)


@router.get("/dashboard/stats", response_model=DashboardStatsOut)
def dashboard_stats(db: Session = Depends(get_db)):
    total_datasets = db.query(Dataset).count()
    total_requests = db.query(DiscoveryRequest).count()
    total_sources = db.query(DataSource).count()
    active_jobs = db.query(Job).filter(Job.status.in_(["QUEUED", "RUNNING"])).all()

    recent_datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).limit(5).all()
    recent_requests = db.query(DiscoveryRequest).order_by(DiscoveryRequest.created_at.desc()).limit(5).all()

    def _ds_out(ds: Dataset) -> DatasetOut:
        source = db.get(DataSource, ds.source_id) if ds.source_id else None
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

    def _req_out(req: DiscoveryRequest) -> DiscoveryRequestOut:
        return DiscoveryRequestOut(
            id=req.id,
            query=req.query,
            parsed_requirements=ParsedRequirements(**req.parsed_requirements),
            status=req.status,
            error_code=req.error_code,
            error_message=req.error_message,
            created_at=req.created_at,
        )

    return DashboardStatsOut(
        total_datasets=total_datasets,
        total_discovery_requests=total_requests,
        total_sources=total_sources,
        active_jobs=len(active_jobs),
        recent_datasets=[_ds_out(d) for d in recent_datasets],
        recent_requests=[_req_out(r) for r in recent_requests],
        active_job_list=[_job_out(j) for j in active_jobs],
    )


@router.get("/sources", response_model=list[dict])
def list_sources(db: Session = Depends(get_db)):
    sources = db.query(DataSource).order_by(DataSource.name).all()
    return [
        {
            "id": s.id,
            "slug": s.slug,
            "name": s.name,
            "source_type": s.source_type,
            "adapter": s.adapter,
            "status": s.status,
            "last_audited_at": s.last_audited_at,
        }
        for s in sources
    ]


@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    out = []
    for ds in datasets:
        source = db.get(DataSource, ds.source_id) if ds.source_id else None
        out.append(
            DatasetOut(
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
        )
    return out
