"""API request and response models.

These are the wire contract, deliberately separate from the domain models in
:mod:`rl_automl.core.types`. The domain objects are internal and change freely; a request
body is a promise to clients, so it gets its own, smaller, validated shape here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rl_automl.core.types import utc_now_iso


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------


class DatasetFromSourceRequest(_Request):
    """Register a bundled or synthetic dataset by name, without uploading a file."""

    source: str = Field(description="Name of a bundled dataset, e.g. 'churn_demo'")
    name: str | None = Field(default=None, description="Optional display name override")


class DatasetSourceInfo(_Response):
    """A dataset that can be registered by name, whether or not it has been used yet.

    Distinct from :class:`DatasetSummary` on purpose: this describes what is *available*,
    while a summary describes bytes that have already passed the upload gate.
    """

    name: str
    task_type: str
    target: str | None = None
    metric: str = ""
    description: str = ""
    requires_network: bool = False
    tags: list[str] = Field(default_factory=list)


class ModelCatalog(_Response):
    """Which algorithms can actually run here.

    Exists because the enabled model list and the optional dependencies together decide
    which task types are runnable, and a client should not have to attempt a run to learn
    that, say, clustering has no enabled model.
    """

    enabled: list[str] = Field(default_factory=list)
    runnable_tasks: list[str] = Field(default_factory=list)
    by_task: dict[str, list[str]] = Field(default_factory=dict)


class DatasetSummary(_Response):
    dataset_id: str
    name: str
    path: str
    n_rows: int
    n_cols: int
    sha256: str
    columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    preview: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class ArchiveMemberRejection(_Response):
    """A file inside an archive that was not registered, and why."""

    name: str
    reason: str


class DatasetArchiveResponse(_Response):
    """The result of unpacking an archive upload.

    A download commonly holds several tables (``train.csv``, ``test.csv``, a sample
    submission). Each readable table is registered as its own dataset so the caller can
    choose which one to train on.
    """

    archive_name: str
    datasets: list[DatasetSummary] = Field(default_factory=list)
    rejected: list[ArchiveMemberRejection] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------


class RunCreateRequest(_Request):
    problem_statement: str = Field(min_length=3)
    dataset_id: str | None = None
    source: str | None = Field(
        default=None, description="Bundled dataset name; alternative to dataset_id"
    )
    target: str | None = None
    task_type: (
        Literal[
            "classification",
            "regression",
            "clustering",
            "anomaly_detection",
            "dimensionality_reduction",
        ]
        | None
    ) = None
    metric: str | None = None
    use_policy: bool = True


class ApproveRequest(_Request):
    """Which recommended pipelines the user authorises.

    ``selection`` is ``"all"``, a list of 1-based ranks, or a list of model names.
    """

    selection: str | list[int] | list[str] = "all"
    approved_by: str = "user"


class CancelRequest(_Request):
    reason: str = "cancelled by user"


class RunSummary(_Response):
    run_id: str
    status: str
    created_at: str
    updated_at: str
    problem_statement: str = ""
    dataset_name: str = ""
    task_type: str | None = None
    metric: str | None = None
    best_model: str | None = None
    validation_score: float | None = None
    test_score: float | None = None
    n_recommended: int = 0
    error: str | None = None


class RecommendationResponse(_Response):
    run_id: str
    status: str
    planner: str = ""
    task_type: str = ""
    metric: str = ""
    metric_direction: str = ""
    requires_user_approval: bool = True
    estimated_runtime_s: float | None = None
    estimated_compute: str = ""
    baseline_expectation: str = ""
    notes: list[str] = Field(default_factory=list)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)


class EventMessage(_Response):
    run_id: str
    stage: str
    message: str
    sequence: int = 0
    progress: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utc_now_iso)


class ArtifactInfo(_Response):
    kind: str
    url: str
    sha256: str | None = None
    size_bytes: int | None = None


class ArtifactList(_Response):
    run_id: str
    verified: bool = False
    artifacts: list[ArtifactInfo] = Field(default_factory=list)


class HealthResponse(_Response):
    status: str = "ok"
    runs: int = 0
    datasets: int = 0


__all__ = [
    "ApproveRequest",
    "ArchiveMemberRejection",
    "ArtifactInfo",
    "ArtifactList",
    "CancelRequest",
    "DatasetArchiveResponse",
    "DatasetFromSourceRequest",
    "DatasetSourceInfo",
    "DatasetSummary",
    "EventMessage",
    "HealthResponse",
    "ModelCatalog",
    "RecommendationResponse",
    "RunCreateRequest",
    "RunSummary",
]
