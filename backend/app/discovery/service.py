"""Discovery orchestration: the approval-gated state machine.

REQUEST_CREATED → REQUIREMENTS_PARSED → RESOURCES_DISCOVERED → RESOURCES_AUDITED
→ WAITING_FOR_SOURCE_APPROVAL → SOURCES_APPROVED → DATASET_SEARCHING → DATASETS_READY
→ WAITING_FOR_DATASET_SELECTION → DATASET_SELECTED → DATASET_DOWNLOADING → DATASET_READY

The service NEVER searches a source that the user has not approved, and never
touches a source whose audit failed.
"""

import asyncio

from sqlalchemy.orm import Session

from app.db import new_uuid
from app.datasets.dedupe import dedupe_candidates
from app.datasets.normalize import normalize_search_result
from app.datasets.rank import score_candidates
from app.discovery.adapters.huggingface import HuggingFaceAdapter
from app.discovery.audit import audit_source, persist_audit
from app.discovery.registry import seed_sources
from app.models import (
    DataSource,
    Dataset,
    DatasetCandidate,
    DiscoveryRequest,
    Job,
    SourceApproval,
    SourceAudit,
    utcnow,
)
from app.parsing.requirement_parser import parse_requirements
from app.schemas import ParsedRequirements

ADAPTERS = {"huggingface": HuggingFaceAdapter}


def create_request(db: Session, query: str, project_id: str | None = None) -> DiscoveryRequest:
    seed_sources(db)
    req = DiscoveryRequest(
        id=new_uuid(),
        project_id=project_id,
        query=query,
        parsed_requirements=parse_requirements(query).model_dump(),
        status="REQUEST_CREATED",
    )
    db.add(req)
    db.commit()
    db.refresh(req)
    return req


def discover_and_audit_sources(db: Session, request: DiscoveryRequest) -> list[dict]:
    """Match requirements to registered sources, audit each, persist approval rows."""
    parsed = ParsedRequirements(**request.parsed_requirements)
    sources = db.query(DataSource).all()
    matched = _match_sources(sources, parsed)

    audit_results = asyncio.run(
        _audit_all([s for s in matched])
    )

    approvals = []
    for source in matched:
        result = audit_results[source.id]
        approval = SourceApproval(id=new_uuid(), request_id=request.id, source_id=source.id, approved=False)
        db.add(approval)
        persist_audit(db, source, result)
        approvals.append({"source": source, "approval": approval, "audit": result})

    request.status = "WAITING_FOR_SOURCE_APPROVAL"
    db.commit()
    return approvals


async def _audit_all(sources):
    results = {}
    for source in sources:
        results[source.id] = await audit_source(source)
    return results


def _match_sources(sources, parsed: ParsedRequirements):
    """Score sources by domain overlap; sources with adapters always stay in the pool."""
    matched = []
    for source in sources:
        overlap = len(set(source.domains or []) & set(parsed.domain))
        if source.adapter or overlap > 0:
            matched.append(source)
    if not matched:
        matched = [s for s in sources if s.adapter]
    return matched


def approve_sources(db: Session, request: DiscoveryRequest, source_ids: list[str]) -> dict:
    """Approve only sources that were part of this request AND passed the audit."""
    eligible = {
        a.source_id: a
        for a in db.query(SourceApproval).filter(SourceApproval.request_id == request.id).all()
    }
    approved, rejected = [], {}
    for sid in source_ids:
        approval = eligible.get(sid)
        if approval is None:
            rejected[sid] = "Source was not part of this discovery request."
            continue
        source = db.get(DataSource, sid)
        latest_audit = (
            db.query(SourceAudit)
            .filter(SourceAudit.source_id == sid)
            .order_by(SourceAudit.audited_at.desc())
            .first()
        )
        usable = (
            source is not None
            and source.adapter is not None
            and latest_audit is not None
            and latest_audit.technical_status in ("API_AVAILABLE", "DOWNLOAD_AVAILABLE", "ELIGIBLE")
        )
        if not usable:
            rejected[sid] = "Source failed the audit or has no implemented adapter; it cannot be approved."
            continue
        approval.approved = True
        approval.approved_at = utcnow()
        approved.append(sid)

    request.status = "SOURCES_APPROVED" if approved else "RESOURCES_AUDITED"
    db.commit()
    return {"approved": approved, "rejected": rejected}


def start_search(db: Session, request: DiscoveryRequest, job: Job) -> None:
    request.status = "SOURCES_APPROVED"
    db.commit()


def run_discovery_search(request_id: str, job_id: str, db_factory) -> None:
    """Search ONLY approved sources; normalize; dedupe; rank; persist candidates."""
    db: Session = db_factory()
    try:
        request = db.get(DiscoveryRequest, request_id)
        job = db.get(Job, job_id)
        if request is None or job is None:
            return
        try:
            approved_ids = [
                a.source_id
                for a in db.query(SourceApproval)
                .filter(SourceApproval.request_id == request.id, SourceApproval.approved == True)  # noqa: E712
                .all()
            ]
            if not approved_ids:
                raise RuntimeError("No approved sources; search is blocked.")

            request.status = "DATASET_SEARCHING"
            job.status = "RUNNING"
            job.stage = "SEARCHING"
            job.progress = 0.2
            job.detail = f"Searching {len(approved_ids)} approved source(s)."
            db.commit()

            parsed = ParsedRequirements(**request.parsed_requirements)
            sources = db.query(DataSource).filter(DataSource.id.in_(approved_ids)).all()

            raw_results = []
            for source in sources:
                adapter_cls = ADAPTERS.get(source.adapter)
                if adapter_cls is None:
                    continue
                adapter = adapter_cls()
                found = asyncio.run(adapter.search(parsed))
                for r in found:
                    rec = normalize_search_result(source.id, r)
                    rec["source_slug"] = source.slug
                    raw_results.append(rec)

            job.progress = 0.6
            job.stage = "NORMALIZING"
            job.detail = f"Normalizing {len(raw_results)} raw results."
            db.commit()

            unique, dropped = dedupe_candidates(raw_results)
            ranked = score_candidates(unique, parsed)

            for rec in ranked:
                rec.pop("name_normalized", None)
                rec.pop("source_slug", None)
                db.add(DatasetCandidate(id=new_uuid(), request_id=request.id, **rec))

            request.status = "DATASETS_READY"
            job.status = "COMPLETED"
            job.stage = "DONE"
            job.progress = 1.0
            job.detail = f"{len(ranked)} unique datasets found ({len(dropped)} duplicates dropped)."
            job.finished_at = utcnow()
            db.commit()
        except Exception as exc:
            db.rollback()
            request.status = "FAILED"
            request.error_code = "SEARCH_FAILURE"
            request.error_message = str(exc)[:500]
            job.status = "FAILED"
            job.error_code = "SEARCH_FAILURE"
            job.error_message = str(exc)[:500]
            job.finished_at = utcnow()
            db.commit()
    finally:
        db.close()


def select_dataset(db: Session, candidate: DatasetCandidate, project_id: str | None) -> tuple[Dataset, Job, Job, Job]:
    """Create the Dataset plus a chained job set: download → EDA → preprocessing."""
    source = db.get(DataSource, candidate.source_id)
    dataset = Dataset(
        id=new_uuid(),
        project_id=project_id,
        candidate_id=candidate.id,
        name=candidate.name,
        description=candidate.description,
        source_id=candidate.source_id,
        source_dataset_id=candidate.source_dataset_id,
        source_url=candidate.canonical_url,
        access_method=source.access_method if source else None,
        license=candidate.license,
        row_count=candidate.row_count,
        column_count=candidate.column_count,
        file_format=candidate.file_format,
    )
    db.add(dataset)

    download_job = Job(id=new_uuid(), dataset_id=dataset.id, kind="DATASET_DOWNLOAD", status="QUEUED")
    eda_job = Job(id=new_uuid(), dataset_id=dataset.id, kind="EDA", status="QUEUED")
    preprocess_job = Job(id=new_uuid(), dataset_id=dataset.id, kind="PREPROCESS", status="QUEUED")
    db.add(download_job)
    db.add(eda_job)
    db.add(preprocess_job)
    db.commit()
    db.refresh(dataset)
    return dataset, download_job, eda_job, preprocess_job
