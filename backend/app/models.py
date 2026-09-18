"""SQLAlchemy ORM models.

All primary/foreign keys are CHAR(36) UUID strings for cross-database compatibility
(PostgreSQL in production, SQLite for the dev profile without Docker).
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Users / projects (no auth this round; single seeded default project)
# ---------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(255), default="Local User")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(255), default="Default Project")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# Data sources (a source is NOT a dataset)
# ---------------------------------------------------------------------------


class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # e.g. "huggingface"
    name: Mapped[str] = mapped_column(String(255))
    base_url: Mapped[str] = mapped_column(String(512))
    source_type: Mapped[str] = mapped_column(String(32))  # API|DATASET_CATALOG|OPEN_DATA_PORTAL|REPOSITORY|DIRECT_FILE|WEB_PAGE
    api_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    robots_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    terms_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    license_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    access_method: Mapped[str] = mapped_column(String(32), default="API")  # preferred access
    domains: Mapped[list] = mapped_column(JSON, default=list)  # topic areas this source covers
    supports_search: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_metadata: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_preview: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_download: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_api: Mapped[bool] = mapped_column(Boolean, default=False)
    requires_auth: Mapped[bool] = mapped_column(Boolean, default=False)
    adapter: Mapped[str | None] = mapped_column(String(64), nullable=True)  # adapter registry key
    status: Mapped[str] = mapped_column(String(32), default="REGISTERED")  # REGISTERED|AUDITED|DISABLED
    last_audited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class SourceAudit(Base):
    __tablename__ = "source_audits"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(36), ForeignKey("data_sources.id"), index=True)
    reachable: Mapped[bool] = mapped_column(Boolean, default=False)
    api_available: Mapped[bool] = mapped_column(Boolean, default=False)
    search_available: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_available: Mapped[bool] = mapped_column(Boolean, default=False)
    preview_available: Mapped[bool] = mapped_column(Boolean, default=False)
    download_available: Mapped[bool] = mapped_column(Boolean, default=False)
    robots_status: Mapped[str] = mapped_column(String(32), default="UNKNOWN")  # FOUND|NOT_FOUND|UNKNOWN|ERROR
    robots_allowed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    terms_status: Mapped[str] = mapped_column(String(32), default="UNKNOWN")  # FOUND|NOT_FOUND|UNKNOWN
    license_status: Mapped[str] = mapped_column(String(32), default="UNKNOWN")
    authentication_required: Mapped[bool] = mapped_column(Boolean, default=False)
    recommended_access_method: Mapped[str] = mapped_column(String(32), default="UNKNOWN")
    technical_status: Mapped[str] = mapped_column(String(32), default="AUDIT_PENDING")
    policy_status: Mapped[str] = mapped_column(String(32), default="UNKNOWN")
    notes: Mapped[list] = mapped_column(JSON, default=list)
    audited_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    source = relationship("DataSource", backref="audits")


# ---------------------------------------------------------------------------
# Discovery requests and source approvals
# ---------------------------------------------------------------------------


class DiscoveryRequest(Base):
    __tablename__ = "discovery_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("projects.id"), nullable=True, index=True)
    query: Mapped[str] = mapped_column(Text)
    parsed_requirements: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(48), default="REQUEST_CREATED", index=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class SourceApproval(Base):
    __tablename__ = "source_approvals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), ForeignKey("discovery_requests.id"), index=True)
    source_id: Mapped[str] = mapped_column(String(36), ForeignKey("data_sources.id"), index=True)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    source = relationship("DataSource")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("discovery_requests.id"), nullable=True, index=True)
    dataset_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    kind: Mapped[str] = mapped_column(String(48))  # DISCOVERY_SEARCH|DATASET_DOWNLOAD|EDA
    status: Mapped[str] = mapped_column(String(32), default="QUEUED", index=True)
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Dataset candidates (normalized search results) and registered datasets
# ---------------------------------------------------------------------------


class DatasetCandidate(Base):
    __tablename__ = "dataset_candidates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("discovery_requests.id"), nullable=True, index=True)
    source_id: Mapped[str] = mapped_column(String(36), ForeignKey("data_sources.id"), index=True)
    source_dataset_id: Mapped[str] = mapped_column(String(512))
    canonical_url: Mapped[str] = mapped_column(String(1024))
    dedupe_key: Mapped[str] = mapped_column(String(1024), index=True)
    name: Mapped[str] = mapped_column(String(512))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    license: Mapped[str | None] = mapped_column(String(255), nullable=True)
    owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    domains: Mapped[list] = mapped_column(JSON, default=list)
    date_start: Mapped[str | None] = mapped_column(String(32), nullable=True)
    date_end: Mapped[str | None] = mapped_column(String(32), nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_format: Mapped[str | None] = mapped_column(String(32), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    columns_meta: Mapped[list] = mapped_column(JSON, default=list)
    download_available: Mapped[bool] = mapped_column(Boolean, default=False)
    preview_available: Mapped[bool] = mapped_column(Boolean, default=False)
    raw_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    score_total: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_components: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    source = relationship("DataSource")
    __table_args__ = (UniqueConstraint("request_id", "dedupe_key", name="uq_request_dedupe"),)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("projects.id"), nullable=True, index=True)
    candidate_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("dataset_candidates.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(512))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("data_sources.id"), nullable=True)
    source_dataset_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    access_method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    license: Mapped[str | None] = mapped_column(String(255), nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_format: Mapped[str | None] = mapped_column(String(32), nullable=True)
    selected_target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    selected_task: Mapped[str | None] = mapped_column(String(32), nullable=True)
    selected_at_target: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DatasetFile(Base):
    __tablename__ = "dataset_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"), index=True)
    file_name: Mapped[str] = mapped_column(String(512))
    storage_key: Mapped[str] = mapped_column(String(1024))
    storage_backend: Mapped[str] = mapped_column(String(32))  # s3|local
    file_format: Mapped[str | None] = mapped_column(String(32), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_hash_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_file_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DatasetColumn(Base):
    __tablename__ = "dataset_columns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    dtype: Mapped[str | None] = mapped_column(String(64), nullable=True)
    semantic_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    position: Mapped[int] = mapped_column(Integer, default=0)


class SourceConnection(Base):
    """Per-source credentials (API keys / tokens), stored as protected secrets.
    The plaintext secret never leaves the backend; only a masked hint is returned."""

    __tablename__ = "source_connections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(36), ForeignKey("data_sources.id"), unique=True, index=True)
    secret_encrypted: Mapped[str] = mapped_column(Text)  # Fernet-encrypted
    secret_hint: Mapped[str] = mapped_column(String(64))  # masked hint, e.g. "abcd…wxyz"
    status: Mapped[str] = mapped_column(String(32), default="CONNECTED")  # CONNECTED | INVALID | DISCONNECTED
    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    source = relationship("DataSource")


class EDAReport(Base):
    __tablename__ = "eda_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"), index=True)
    report: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PreprocessReport(Base):
    __tablename__ = "preprocess_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"), index=True)
    report: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
