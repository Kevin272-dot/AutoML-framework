"""Domain contract.

``ExperimentSpec`` is deliberately the *only* representation of an experiment in the
system. The RL agent emits it, the executor consumes it, memory stores it and the
packaging engine describes it. Nothing downstream re-invents its own version, which is
what keeps the RL agent swappable (spec §35).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


def utc_now_iso() -> str:
    """Timezone-aware ISO-8601 timestamp, second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class _Base(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)


# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------


class LearningType(str, Enum):
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"

    @property
    def learning_type(self) -> LearningType:
        if self in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
            return LearningType.SUPERVISED
        return LearningType.UNSUPERVISED

    @property
    def is_supervised(self) -> bool:
        return self.learning_type is LearningType.SUPERVISED


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ColumnKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    OTHER = "other"


class CostTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExperimentSource(str, Enum):
    RL = "rl"
    BASELINE = "baseline"
    USER = "user"


class ExperimentStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    """Run state machine. Ordering is meaningful; see ``orchestrator``."""

    CREATED = "created"
    PROFILING = "profiling"
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPARING = "comparing"
    SELECTING = "selecting"
    PACKAGING = "packaging"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.CANCELLED)


# --------------------------------------------------------------------------------------
# Task understanding
# --------------------------------------------------------------------------------------


class TaskSpec(_Base):
    """What the problem statement is actually asking for (spec §3)."""

    learning_type: LearningType = LearningType.SUPERVISED
    task_type: TaskType = TaskType.CLASSIFICATION
    objective: str = ""
    target: str | None = None
    metric: str = "f1"
    metric_direction: MetricDirection = MetricDirection.MAXIMIZE
    confidence: float = 0.0
    source: str = "heuristic"
    notes: list[str] = Field(default_factory=list)
    problem_statement: str = ""

    def to_example(self) -> dict[str, Any]:
        return {
            "learning_type": self.learning_type.value,
            "task_type": self.task_type.value,
            "objective": self.objective,
            "target": self.target,
            "metric": self.metric,
        }


# --------------------------------------------------------------------------------------
# Dataset profiling
# --------------------------------------------------------------------------------------


class ColumnProfile(_Base):
    name: str
    kind: ColumnKind
    dtype: str
    n_missing: int = 0
    missing_ratio: float = 0.0
    n_unique: int = 0
    cardinality_ratio: float = 0.0
    is_constant: bool = False
    # numeric summaries
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    median: float | None = None
    skew: float | None = None
    n_outliers: int = 0
    # categorical / datetime summaries
    top_values: list[dict[str, Any]] = Field(default_factory=list)
    is_high_cardinality: bool = False


class ClassDistribution(_Base):
    n_classes: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    minority_ratio: float | None = None
    imbalance_ratio: float | None = None
    entropy: float | None = None
    is_imbalanced: bool = False


class FeatureCorrelation(_Base):
    feature_a: str
    feature_b: str
    correlation: float


class DatasetProfile(_Base):
    """Full profiling output (spec §4). Rendered in the UI and stored in the run."""

    n_rows: int = 0
    n_cols: int = 0
    numeric_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    datetime_features: list[str] = Field(default_factory=list)
    other_features: list[str] = Field(default_factory=list)
    n_missing_total: int = 0
    missing_ratio: float = 0.0
    columns_with_missing: list[str] = Field(default_factory=list)
    n_duplicate_rows: int = 0
    duplicate_ratio: float = 0.0
    cardinality: dict[str, int] = Field(default_factory=dict)
    high_cardinality_features: list[str] = Field(default_factory=list)
    constant_features: list[str] = Field(default_factory=list)
    class_distribution: ClassDistribution | None = None
    top_correlations: list[FeatureCorrelation] = Field(default_factory=list)
    max_abs_correlation: float = 0.0
    mean_abs_skew: float = 0.0
    dimensionality_ratio: float = 0.0
    size_bytes: int = 0
    memory_mb: float = 0.0
    target: str | None = None
    columns: list[ColumnProfile] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    profiled_at: str = Field(default_factory=utc_now_iso)

    def summary_line(self) -> str:
        return f"{self.n_rows:,} rows x {self.n_cols:,} features"


class DatasetFingerprint(_Base):
    """Compact, fixed-shape numeric description of a dataset.

    Feeds the RL state and the cross-dataset similarity index. ``schema_version``
    guards against a saved policy meeting a differently-shaped fingerprint.
    """

    schema_version: str = SCHEMA_VERSION
    feature_names: list[str] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0

    def vector(self) -> list[float]:
        return list(self.values)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.values, strict=False))


# --------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------


class ExperimentSpec(_Base):
    """A single, fully-specified experiment. The shared contract of the system."""

    experiment_id: str = Field(default_factory=lambda: new_id("exp"))
    model: str
    preprocessing: list[str] = Field(default_factory=list)
    feature_selection: str = "none"
    preset_index: int = 0
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    expected_cost: CostTier = CostTier.MEDIUM
    source: ExperimentSource = ExperimentSource.RL
    rationale: str = ""

    def signature(self) -> str:
        """Stable identity used for de-duplication and novelty scoring.

        Hyperparameters are part of the key so that two different presets of the same
        family count as genuinely different experiments.
        """
        pre = ",".join(sorted(self.preprocessing))
        hp = ",".join(f"{k}={self.hyperparameters[k]}" for k in sorted(self.hyperparameters))
        return f"{self.model}|{pre}|{self.feature_selection}|{hp}"

    def short_label(self) -> str:
        return self.model

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ExperimentResult(_Base):
    """Outcome of executing one :class:`ExperimentSpec` (spec §13)."""

    experiment: ExperimentSpec
    status: ExperimentStatus = ExperimentStatus.SUCCESS
    error: str | None = None
    error_type: str | None = None

    primary_metric: str = ""
    metric_direction: MetricDirection = MetricDirection.MAXIMIZE
    validation_score: float | None = None
    test_score: float | None = None
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    test_metrics: dict[str, float] = Field(default_factory=dict)

    training_time_s: float = 0.0
    inference_time_ms: float = 0.0
    inference_time_per_1k_ms: float = 0.0
    peak_memory_mb: float = 0.0
    model_size_bytes: int = 0
    n_parameters: int | None = None
    n_features_used: int | None = None

    artifact_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is ExperimentStatus.SUCCESS

    def rank_key(self) -> float:
        """Score oriented so that larger is always better."""
        score = self.validation_score
        if score is None:
            return float("-inf")
        return score if self.metric_direction is MetricDirection.MAXIMIZE else -score


# --------------------------------------------------------------------------------------
# Recommendation (spec §9)
# --------------------------------------------------------------------------------------


class RecommendedExperiment(_Base):
    rank: int
    model: str
    preprocessing: list[str] = Field(default_factory=list)
    feature_selection: str = "none"
    preset_index: int = 0
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    expected_cost: CostTier = CostTier.MEDIUM
    expected_performance: float | None = None
    experiment: ExperimentSpec | None = None


class SearchStatistics(_Base):
    searcher: str = "rl"
    n_experiments: int = 0
    n_failed: int = 0
    n_distinct_models: int = 0
    best_validation_score: float | None = None
    best_model: str | None = None
    total_training_time_s: float = 0.0
    total_compute_s: float = 0.0
    time_to_best_s: float | None = None
    stopping_reason: str = "budget"
    history: list[dict[str, Any]] = Field(default_factory=list)


class RecommendationSet(_Base):
    """The plan-phase output. Contains no trained models by construction."""

    task: TaskType
    metric: str = "f1"
    metric_direction: MetricDirection = MetricDirection.MAXIMIZE
    recommended_experiments: list[RecommendedExperiment] = Field(default_factory=list)
    estimated_experiments: int = 0
    estimated_compute: str = ""
    estimated_runtime_s: float | None = None
    estimated_cost_tier: CostTier = CostTier.MEDIUM
    requires_user_approval: bool = True
    planner: str = "rl-ppo"
    search_statistics: SearchStatistics | None = None
    baseline_expectation: str = ""
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


# --------------------------------------------------------------------------------------
# Comparison and selection (spec §14, §15)
# --------------------------------------------------------------------------------------


class ComparisonRow(_Base):
    model: str
    experiment_id: str
    status: ExperimentStatus
    primary_metric: str
    validation_score: float | None = None
    test_score: float | None = None
    training_time_s: float = 0.0
    inference_time_per_1k_ms: float = 0.0
    peak_memory_mb: float = 0.0
    model_size_bytes: int = 0
    error: str | None = None


class ParetoPoint(_Base):
    model: str
    experiment_id: str
    score: float
    cost_s: float
    peak_memory_mb: float = 0.0
    on_frontier: bool = True


class EfficiencyAlternative(_Base):
    model: str
    validation_score: float | None = None
    speedup: float | None = None
    score_delta: float | None = None
    reason: str = ""


class BestModel(_Base):
    model: str
    experiment_id: str
    primary_metric: str
    metric_direction: MetricDirection = MetricDirection.MAXIMIZE
    validation_score: float | None = None
    test_score: float | None = None
    reason: str = ""
    selection_basis: str = "validation"
    efficiency_alternative: EfficiencyAlternative | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    preprocessing: list[str] = Field(default_factory=list)
    feature_selection: str = "none"


class ComparisonSummary(_Base):
    rows: list[ComparisonRow] = Field(default_factory=list)
    objective: str = ""
    primary_metric: str = ""
    metric_direction: MetricDirection = MetricDirection.MAXIMIZE
    best_validation: str | None = None
    best_test: str | None = None
    fastest: str | None = None
    smallest: str | None = None
    most_compute_efficient: str | None = None
    trade_offs: list[str] = Field(default_factory=list)
    table: str = ""


# --------------------------------------------------------------------------------------
# Artifacts and the run object (spec §22, §23)
# --------------------------------------------------------------------------------------


class ArtifactBundle(_Base):
    model_zip: str | None = None
    model_zip_sha256: str | None = None
    model_dir: str | None = None
    results_zip: str | None = None
    results_zip_sha256: str | None = None
    verified: bool = False
    extras: dict[str, str] = Field(default_factory=dict)


class RunObject(_Base):
    """Everything an AutoML execution produced (spec §22)."""

    run_id: str
    status: RunStatus = RunStatus.CREATED
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    problem_statement: str = ""
    dataset_path: str = ""
    dataset_name: str = ""
    dataset_sha256: str = ""

    #: An optional second table used verbatim as the held-out test set, instead of carving
    #: one out of ``dataset_path``. This is how a supplied train/test pair is honoured: the
    #: user's own test rows are the ones that must never be seen during selection.
    holdout_path: str | None = None
    holdout_name: str | None = None

    task: TaskSpec | None = None
    dataset_profile: DatasetProfile | None = None
    dataset_fingerprint: DatasetFingerprint | None = None

    rl_recommendations: list[RecommendedExperiment] = Field(default_factory=list)
    recommendation_set: RecommendationSet | None = None
    approved_experiments: list[ExperimentSpec] = Field(default_factory=list)
    approved_at: str | None = None

    results: list[ExperimentResult] = Field(default_factory=list)
    comparison: ComparisonSummary | None = None
    best_model: BestModel | None = None
    pareto_frontier: list[ParetoPoint] = Field(default_factory=list)
    search_statistics: SearchStatistics | None = None

    artifacts: ArtifactBundle = Field(default_factory=ArtifactBundle)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    progress: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    messages: list[str] = Field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = utc_now_iso()


class ExecutionEvent(_Base):
    """One progress event, streamed to a frontend (spec §28)."""

    run_id: str
    stage: str
    message: str
    sequence: int = 0
    progress: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utc_now_iso)


DEFAULT_METRIC_BY_TASK: dict[TaskType, str] = {
    TaskType.CLASSIFICATION: "f1",
    TaskType.REGRESSION: "rmse",
    TaskType.CLUSTERING: "silhouette",
    TaskType.ANOMALY_DETECTION: "average_precision",
    TaskType.DIMENSIONALITY_REDUCTION: "explained_variance",
}

FALLBACK_METRIC_BY_TASK: dict[TaskType, str] = {
    TaskType.CLASSIFICATION: "accuracy",
    TaskType.REGRESSION: "r2",
    TaskType.CLUSTERING: "silhouette",
    TaskType.ANOMALY_DETECTION: "average_precision",
    TaskType.DIMENSIONALITY_REDUCTION: "explained_variance",
}


__all__ = [
    "DEFAULT_METRIC_BY_TASK",
    "FALLBACK_METRIC_BY_TASK",
    "SCHEMA_VERSION",
    "ArtifactBundle",
    "BestModel",
    "ClassDistribution",
    "ColumnKind",
    "ColumnProfile",
    "ComparisonRow",
    "ComparisonSummary",
    "CostTier",
    "DatasetFingerprint",
    "DatasetProfile",
    "EfficiencyAlternative",
    "ExecutionEvent",
    "ExperimentResult",
    "ExperimentSource",
    "ExperimentSpec",
    "ExperimentStatus",
    "FeatureCorrelation",
    "LearningType",
    "MetricDirection",
    "ParetoPoint",
    "RecommendationSet",
    "RecommendedExperiment",
    "RunObject",
    "RunStatus",
    "SearchStatistics",
    "TaskSpec",
    "TaskType",
    "new_id",
    "utc_now_iso",
]
