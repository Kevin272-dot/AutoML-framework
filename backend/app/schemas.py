"""Pydantic request/response schemas. These define the API contract that
src/lib/api-types.ts mirrors on the frontend."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Requirement parsing
# ---------------------------------------------------------------------------

TaskType = Literal["classification", "regression", None]


class ParsedRequirements(BaseModel):
    keywords: list[str] = []
    domain: list[str] = []
    location: list[str] = []
    date_start: str | None = None
    date_end: str | None = None
    task: TaskType = None
    target_hints: list[str] = []
    required_features: list[str] = []
    preferred_format: str | None = None
    min_rows: int | None = None
    max_rows: int | None = None
    parser: str = "deterministic"


# ---------------------------------------------------------------------------
# Sources + audit
# ---------------------------------------------------------------------------


class SourceOut(BaseModel):
    id: str
    slug: str
    name: str
    base_url: str
    source_type: str
    access_method: str
    supports_search: bool
    supports_metadata: bool
    supports_preview: bool
    supports_download: bool
    requires_auth: bool
    status: str


class AuditOut(BaseModel):
    id: str
    source_id: str
    reachable: bool
    api_available: bool
    search_available: bool
    metadata_available: bool
    preview_available: bool
    download_available: bool
    robots_status: str
    robots_allowed: bool | None
    terms_status: str
    license_status: str
    authentication_required: bool
    recommended_access_method: str
    technical_status: str
    policy_status: str
    notes: list[str] = []
    audited_at: datetime | None = None


class SourceWithAuditOut(SourceOut):
    audit: AuditOut | None = None


# ---------------------------------------------------------------------------
# Discovery flow
# ---------------------------------------------------------------------------


class DiscoveryRequestCreate(BaseModel):
    query: str = Field(min_length=3, max_length=2000)


class DiscoveryRequestOut(BaseModel):
    id: str
    query: str
    parsed_requirements: ParsedRequirements
    status: str
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime


class ApproveSourcesIn(BaseModel):
    source_ids: list[str] = Field(min_length=1)


class ApproveSourcesOut(BaseModel):
    request_id: str
    status: str
    approved_source_ids: list[str]
    rejected_source_ids: list[str] = []
    rejected_reasons: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Dataset candidates + ranking
# ---------------------------------------------------------------------------


class ScoreComponents(BaseModel):
    keyword_match: float
    date_match: float
    domain_match: float
    completeness: float
    size_score: float
    feature_quality: float


class RankedCandidateOut(BaseModel):
    id: str
    source: str
    source_dataset_id: str
    name: str
    description: str | None
    url: str
    license: str | None
    owner: str | None
    tags: list[str] = []
    date_start: str | None
    date_end: str | None
    row_count: int | None
    column_count: int | None
    file_format: str | None
    file_size_bytes: int | None
    download_available: bool
    preview_available: bool
    overall_score: float
    score_components: ScoreComponents


class PreviewOut(BaseModel):
    candidate_id: str
    columns: list[str]
    dtypes: dict[str, str]
    rows: list[dict[str, Any]]
    row_count_total: int | None


# ---------------------------------------------------------------------------
# Selection / download / jobs
# ---------------------------------------------------------------------------


class SelectDatasetIn(BaseModel):
    candidate_id: str


class DatasetOut(BaseModel):
    id: str
    name: str
    description: str | None
    source: str | None
    source_dataset_id: str | None
    source_url: str | None
    license: str | None
    row_count: int | None
    column_count: int | None
    file_format: str | None
    selected_target: str | None
    selected_task: str | None
    created_at: datetime


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    stage: str | None
    progress: float
    detail: str | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------


class ColumnStats(BaseModel):
    name: str
    dtype: str
    semantic_type: str
    missing_count: int
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    histogram: list[tuple[float, float]] | None = None  # (bin_start, count)
    top_values: list[tuple[str, int]] | None = None
    is_constant: bool = False
    is_potential_id: bool = False
    outlier_count: int | None = None


class TargetCandidate(BaseModel):
    column: str
    task: str
    confidence: float
    rationale: str


class EDAOut(BaseModel):
    dataset_id: str
    shape: tuple[int, int]
    column_stats: list[ColumnStats]
    duplicate_rows: int
    correlation_matrix: dict[str, dict[str, float]] | None = None
    class_balance: dict[str, Any] | None = None
    quality_warnings: list[str] = []
    target_candidates: list[TargetCandidate] = []
    suggested_task: str | None = None
    suggested_target: str | None = None


class ConfirmTargetIn(BaseModel):
    target: str
    task: Literal["classification", "regression"]


# ---------------------------------------------------------------------------
# Dashboard / errors
# ---------------------------------------------------------------------------


class DashboardStatsOut(BaseModel):
    total_datasets: int
    total_discovery_requests: int
    total_sources: int
    active_jobs: int
    recent_datasets: list[DatasetOut] = []
    recent_requests: list[DiscoveryRequestOut] = []
    active_job_list: list[JobOut] = []


class ErrorEnvelope(BaseModel):
    code: str
    message: str
    what_happened: str
    why: str
    what_to_do: str
    details: dict[str, Any] = {}
