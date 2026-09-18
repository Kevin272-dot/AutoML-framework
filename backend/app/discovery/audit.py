"""Resource audit engine.

For each candidate source, probe (with plain HTTP only — no scraping):
  reachability, documented API liveness, robots.txt posture, terms/license availability.

Produces a SourceAudit row with explicit technical_status and policy_status.
robots.txt results are reported as a technical signal only.
"""

from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy.orm import Session

from app.config import get_settings
from app.discovery.robots import check_url_reachable, evaluate_robots, fetch_robots, probe_api
from app.models import DataSource, SourceAudit, utcnow


def _technical_status(source: DataSource, audit: dict) -> str:
    if not audit["reachable"]:
        return "SOURCE_UNREACHABLE"
    if audit["robots_allowed"] is False:
        return "ROBOTS_DISALLOWED"
    if source.requires_auth:
        return "AUTH_REQUIRED"
    if audit["api_available"] and audit["search_available"]:
        return "API_AVAILABLE"
    if audit["download_available"]:
        return "DOWNLOAD_AVAILABLE"
    return "UNSUPPORTED"


def _policy_status(robots_allowed: bool | None, terms_found: bool) -> str:
    if robots_allowed is False:
        return "RESTRICTED"
    if terms_found and robots_allowed is not None:
        return "KNOWN"
    return "UNKNOWN"


def _api_probe_path(source: DataSource) -> str | None:
    """A cheap documented endpoint that proves API liveness + search capability."""
    probes = {
        "huggingface": "https://huggingface.co/api/datasets?limit=1",
        "data-gov": "https://catalog.data.gov/api/3/action/package_search?q=agriculture&rows=1",
        "eu-open-data": "https://data.europa.eu/api/hub/search/search?q=agriculture&limit=1",
    }
    return probes.get(source.slug)


async def audit_source(source: DataSource) -> dict:
    settings = get_settings()
    notes: list[str] = []
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
        reachable, status_code, _ = await check_url_reachable(client, source.base_url)
        if not reachable:
            notes.append(f"Base URL {source.base_url} did not respond (HTTP {status_code}).")

        api_available = False
        search_available = False
        if source.api_url:
            probe = _api_probe_path(source) or source.api_url
            api_available, err = await probe_api(client, probe)
            if api_available:
                search_available = source.supports_search
            elif err:
                notes.append(f"API probe failed: {err}.")
        elif source.supports_api:
            notes.append("No documented API endpoint registered for this source.")

        if source.robots_url:
            robots_txt, robots_code = await fetch_robots(client, source.robots_url)
            target_path = source.api_url or source.base_url
            robots_result = evaluate_robots(robots_txt, robots_code, target_path)
        else:
            robots_result = evaluate_robots(None, None, source.base_url)

        terms_found = False
        if source.terms_url:
            ok, _, _ = await check_url_reachable(client, source.terms_url)
            terms_found = ok
            if not ok:
                notes.append("Terms URL unreachable.")

        license_found = False
        if source.license_url:
            license_found, _, _ = (await check_url_reachable(client, source.license_url))[:3]

    download_available = source.supports_download and reachable and robots_result.allowed is not False

    audit_dict = {
        "reachable": reachable,
        "api_available": api_available,
        "search_available": search_available,
        "metadata_available": source.supports_metadata and api_available,
        "preview_available": source.supports_preview and api_available,
        "download_available": download_available,
        "robots_status": robots_result.status,
        "robots_allowed": robots_result.allowed,
        "terms_status": "FOUND" if terms_found else ("NOT_FOUND" if source.terms_url else "UNKNOWN"),
        "license_status": "FOUND" if license_found else ("NOT_FOUND" if source.license_url else "UNKNOWN"),
        "authentication_required": source.requires_auth,
        "recommended_access_method": source.access_method,
    }
    if robots_result.note:
        notes.append(robots_result.note)

    audit_dict["technical_status"] = _technical_status(source, audit_dict)
    audit_dict["policy_status"] = _policy_status(robots_result.allowed, terms_found)
    audit_dict["notes"] = notes
    return audit_dict


def persist_audit(db: Session, source: DataSource, result: dict) -> SourceAudit:
    settings = get_settings()
    notes = result.get("notes", [])
    audit = SourceAudit(
        id=_new_id(),
        source_id=source.id,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=settings.audit_cache_ttl_seconds),
        notes=notes,
        **{k: v for k, v in result.items() if k not in ("notes",)},
    )
    db.add(audit)
    source.last_audited_at = utcnow()
    source.status = "AUDITED"
    db.commit()
    db.refresh(audit)
    return audit


def _new_id() -> str:
    import uuid

    return str(uuid.uuid4())
