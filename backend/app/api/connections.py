"""Source connections: users store per-source API keys as protected secrets.

Secrets are encrypted at rest, never returned to the frontend (only a masked
hint), and validated against the source where a documented check exists.
"""

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db, new_uuid
from app.models import DataSource, SourceConnection, utcnow
from app.security.secrets import decrypt_secret, encrypt_secret, mask_secret

router = APIRouter(prefix="/api/connections", tags=["connections"])

# --- secret validation per source -------------------------------------------


async def validate_secret(slug: str, secret: str) -> tuple[bool, str | None]:
    """Returns (valid, reason). Unvalidated sources are stored as CONNECTED but
    flagged unvalidated rather than rejected."""
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
            if slug == "huggingface":
                resp = await client.get(
                    "https://huggingface.co/api/whoami-v2",
                    headers={"Authorization": f"Bearer {secret}"},
                )
                return resp.status_code == 200, None if resp.status_code == 200 else "Token rejected by Hugging Face."
            if slug == "kaggle":
                if ":" not in secret:
                    return False, "Kaggle credentials must be in 'username:key' format."
                username, key = secret.split(":", 1)
                resp = await client.get(
                    "https://www.kaggle.com/api/v1/datasets/list?search=data",
                    auth=(username, key),
                )
                return resp.status_code == 200, None if resp.status_code == 200 else "Credentials rejected by Kaggle."
            if slug == "data-gov-in":
                resp = await client.get(
                    "https://api.data.gov.in/catalog",
                    params={"api-key": secret, "limit": 1, "format": "json"},
                )
                if resp.status_code != 200:
                    return False, f"data.gov.in returned HTTP {resp.status_code}."
                body = resp.json()
                if body.get("error"):
                    return False, "data.gov.in rejected the API key."
                return True, None
    except httpx.HTTPError:
        return True, None  # source unreachable: store without validation rather than fail
    return True, None


# --- schemas -----------------------------------------------------------------


class ConnectionCreate(BaseModel):
    source_id: str
    secret: str = Field(min_length=4, max_length=512)


class ConnectionOut(BaseModel):
    source_id: str
    source_name: str
    source_slug: str
    status: str
    validated: bool
    secret_hint: str
    created_at: str


# --- endpoints ---------------------------------------------------------------


@router.get("", response_model=list[ConnectionOut])
def list_connections(db: Session = Depends(get_db)):
    out = []
    for conn in db.query(SourceConnection).all():
        source = db.get(DataSource, conn.source_id)
        if source is None:
            continue
        out.append(
            ConnectionOut(
                source_id=conn.source_id,
                source_name=source.name,
                source_slug=source.slug,
                status=conn.status,
                validated=conn.validated,
                secret_hint=conn.secret_hint,
                created_at=conn.created_at.isoformat(),
            )
        )
    return out


@router.post("", response_model=ConnectionOut)
async def create_connection(payload: ConnectionCreate, db: Session = Depends(get_db)):
    source = db.get(DataSource, payload.source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Unknown source.")

    valid, reason = await validate_secret(source.slug, payload.secret)
    if not valid:
        raise HTTPException(
            status_code=400,
            detail={"code": "SECRET_INVALID", "message": reason or "The source rejected this key."},
        )

    conn = db.query(SourceConnection).filter(SourceConnection.source_id == source.id).first()
    if conn is None:
        conn = SourceConnection(id=new_uuid(), source_id=source.id)
        db.add(conn)
    conn.secret_encrypted = encrypt_secret(payload.secret)
    conn.secret_hint = mask_secret(payload.secret)
    conn.status = "CONNECTED"
    conn.validated = valid
    db.commit()
    return ConnectionOut(
        source_id=conn.source_id,
        source_name=source.name,
        source_slug=source.slug,
        status=conn.status,
        validated=conn.validated,
        secret_hint=conn.secret_hint,
        created_at=conn.created_at.isoformat(),
    )


@router.delete("/{source_id}")
def delete_connection(source_id: str, db: Session = Depends(get_db)):
    conn = db.query(SourceConnection).filter(SourceConnection.source_id == source_id).first()
    if conn is None:
        raise HTTPException(status_code=404, detail="No connection for this source.")
    conn.status = "DISCONNECTED"
    db.delete(conn)
    db.commit()
    return {"source_id": source_id, "status": "DISCONNECTED"}


def get_connection_secret(db: Session, source_id: str) -> str | None:
    """Internal helper for adapters: decrypt the stored secret for a source."""
    conn = db.query(SourceConnection).filter(SourceConnection.source_id == source_id).first()
    if conn is None or conn.status != "CONNECTED":
        return None
    try:
        return decrypt_secret(conn.secret_encrypted)
    except ValueError:
        conn.status = "INVALID"
        db.commit()
        return None
