"""Discovery + source approval endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db, new_uuid
from app.discovery import service
from app.models import DataSource, DiscoveryRequest, Job, SourceApproval, SourceAudit, utcnow
from app.schemas import (
    ApproveSourcesIn,
    ApproveSourcesOut,
    DiscoveryRequestCreate,
    DiscoveryRequestOut,
    ParsedRequirements,
    RankedCandidateOut,
    ScoreComponents,
    SourceWithAuditOut,
)
from app.workers.dispatch import dispatch_discovery_search

router = APIRouter(prefix="/api/discovery", tags=["discovery"])


def _audit_to_out(audit: SourceAudit):
    from app.schemas import AuditOut

    return AuditOut(
        id=audit.id,
        source_id=audit.source_id,
        reachable=audit.reachable,
        api_available=audit.api_available,
        search_available=audit.search_available,
        metadata_available=audit.metadata_available,
        preview_available=audit.preview_available,
        download_available=audit.download_available,
        robots_status=audit.robots_status,
        robots_allowed=audit.robots_allowed,
        terms_status=audit.terms_status,
        license_status=audit.license_status,
        authentication_required=audit.authentication_required,
        recommended_access_method=audit.recommended_access_method,
        technical_status=audit.technical_status,
        policy_status=audit.policy_status,
        notes=audit.notes or [],
        audited_at=audit.audited_at,
    )


def _source_with_audit(source: DataSource, audit: SourceAudit | None) -> SourceWithAuditOut:
    return SourceWithAuditOut(
        id=source.id,
        slug=source.slug,
        name=source.name,
        base_url=source.base_url,
        source_type=source.source_type,
        access_method=source.access_method,
        supports_search=source.supports_search,
        supports_metadata=source.supports_metadata,
        supports_preview=source.supports_preview,
        supports_download=source.supports_download,
        requires_auth=source.requires_auth,
        status=source.status,
        audit=_audit_to_out(audit) if audit else None,
    )


def _request_out(req: DiscoveryRequest) -> DiscoveryRequestOut:
    return DiscoveryRequestOut(
        id=req.id,
        query=req.query,
        parsed_requirements=ParsedRequirements(**req.parsed_requirements),
        status=req.status,
        error_code=req.error_code,
        error_message=req.error_message,
        created_at=req.created_at,
    )


def _get_request(db: Session, request_id: str) -> DiscoveryRequest:
    req = db.get(DiscoveryRequest, request_id)
    if req is None:
        raise HTTPException(status_code=404, detail="Discovery request not found.")
    return req


@router.post("/requests", response_model=DiscoveryRequestOut)
def create_discovery_request(payload: DiscoveryRequestCreate, db: Session = Depends(get_db)):
    req = service.create_request(db, payload.query.strip())
    return _request_out(req)


@router.get("/requests/{request_id}", response_model=DiscoveryRequestOut)
def get_discovery_request(request_id: str, db: Session = Depends(get_db)):
    return _request_out(_get_request(db, request_id))


@router.get("/requests/{request_id}/sources", response_model=list[SourceWithAuditOut])
def get_request_sources(request_id: str, db: Session = Depends(get_db)):
    req = _get_request(db, request_id)
    approvals = db.query(SourceApproval).filter(SourceApproval.request_id == req.id).all()
    out = []
    for approval in approvals:
        source = db.get(DataSource, approval.source_id)
        latest_audit = (
            db.query(SourceAudit)
            .filter(SourceAudit.source_id == source.id)
            .order_by(SourceAudit.audited_at.desc())
            .first()
        )
        item = _source_with_audit(source, latest_audit)
        out.append(item)
    return out


@router.post("/requests/{request_id}/audit", response_model=DiscoveryRequestOut)
def audit_request_sources(request_id: str, db: Session = Depends(get_db)):
    """Run (or re-run) discovery + audit for the request; stops at source approval."""
    req = _get_request(db, request_id)
    service.discover_and_audit_sources(db, req)
    return _request_out(req)


@router.post("/requests/{request_id}/approve-sources", response_model=ApproveSourcesOut)
def approve_sources(request_id: str, payload: ApproveSourcesIn, db: Session = Depends(get_db)):
    req = _get_request(db, request_id)
    if req.status not in ("WAITING_FOR_SOURCE_APPROVAL", "RESOURCES_AUDITED"):
        raise HTTPException(
            status_code=409,
            detail=f"Request is in status {req.status}; source approval is not allowed now.",
        )
    result = service.approve_sources(db, req, payload.source_ids)
    return ApproveSourcesOut(
        request_id=req.id,
        status=req.status,
        approved_source_ids=result["approved"],
        rejected_source_ids=list(result["rejected"].keys()),
        rejected_reasons=result["rejected"],
    )


@router.post("/requests/{request_id}/search")
def search_datasets(request_id: str, db: Session = Depends(get_db)):
    from app.db import SessionLocal

    req = _get_request(db, request_id)
    if req.status != "SOURCES_APPROVED":
        raise HTTPException(
            status_code=409,
            detail="Source approval is required before dataset search.",
        )
    job = Job(id=new_uuid(), request_id=req.id, kind="DISCOVERY_SEARCH", status="QUEUED")
    db.add(job)
    service.start_search(db, req, job)
    db.commit()
    dispatch_discovery_search(req.id, job.id, SessionLocal)
    return {"request_id": req.id, "job_id": job.id, "status": job.status}


@router.get("/requests/{request_id}/results", response_model=list[RankedCandidateOut])
def get_results(request_id: str, db: Session = Depends(get_db)):
    req = _get_request(db, request_id)
    if req.status not in ("DATASETS_READY", "WAITING_FOR_DATASET_SELECTION", "DATASET_SELECTED", "DATASET_DOWNLOADING", "DATASET_READY"):
        return []
    from app.models import DatasetCandidate

    candidates = (
        db.query(DatasetCandidate)
        .filter(DatasetCandidate.request_id == req.id)
        .order_by(DatasetCandidate.score_total.desc())
        .all()
    )
    return [
        RankedCandidateOut(
            id=c.id,
            source=c.source.name if c.source else c.source_id,
            source_dataset_id=c.source_dataset_id,
            name=c.name,
            description=c.description,
            url=c.canonical_url,
            license=c.license,
            owner=c.owner,
            tags=c.tags or [],
            date_start=c.date_start,
            date_end=c.date_end,
            row_count=c.row_count,
            column_count=c.column_count,
            file_format=c.file_format,
            file_size_bytes=c.file_size_bytes,
            download_available=c.download_available,
            preview_available=c.preview_available,
            overall_score=c.score_total or 0.0,
            score_components=ScoreComponents(**(c.score_components or {})),
        )
        for c in candidates
    ]
