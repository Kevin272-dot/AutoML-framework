"""Configuration loading.

Layering, lowest precedence first:

1. Field defaults on the pydantic models below
2. ``--config`` path, ``$AUTOML_CONFIG``, or ``./configs/default.yaml``
3. Environment overrides ``AUTOML_<SECTION>__<FIELD>`` (double underscore nests)

``pydantic-settings`` is deliberately not a dependency; the layering is small enough
to implement directly and keeps the runtime dependency surface minimal.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rl_automl.core.errors import ConfigError
from rl_automl.core.types import TaskType

ENV_PREFIX = "AUTOML_"
ENV_CONFIG_VAR = "AUTOML_CONFIG"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=False)


class RuntimeConfig(_Section):
    artifacts_dir: str = "artifacts"
    seed: int = 1234
    n_jobs: int = -1
    log_level: str = "INFO"
    log_json: bool = False
    num_threads: int = 0  # 0 = leave the environment alone


class SecurityConfig(_Section):
    max_upload_mb: float = 512.0
    max_rows: int = 5_000_000
    max_cols: int = 5_000
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".csv", ".tsv", ".txt", ".parquet", ".jsonl", ".json"]
    )
    max_column_name_length: int = 128
    reject_dunder_columns: bool = True
    max_zip_mb: float = 2048.0
    #: Uploads are often an archive of several tables (a Kaggle download). These bound the
    #: expansion effort: how many members are considered, and how far a member may inflate.
    max_archive_members: int = 25
    max_compression_ratio: float = 200.0


class TaskConfig(_Section):
    default_metric_by_task: dict[str, str] = Field(default_factory=dict)
    llm_extractor: bool = False
    llm_model: str = "gpt-4o-mini"
    min_confidence: float = 0.35


class DatasetConfig(_Section):
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    min_rows: int = 50
    stratify: bool = True
    temporal_split: bool = False
    max_cardinality_for_ohe: int = 50
    high_cardinality_threshold: float = 0.5
    near_duplicate_correlation: float = 0.98

    @field_validator("test_ratio")
    @classmethod
    def _check_ratios(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("test_ratio must be in (0, 1)")
        return v

    def validate_ratios(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                "dataset split ratios must sum to 1.0",
                train=self.train_ratio,
                val=self.val_ratio,
                test=self.test_ratio,
            )


class SearchConfig(_Section):
    max_experiments: int = 12
    min_experiments: int = 3
    n_recommendations: int = 3
    time_budget_s: float = 3600.0
    memory_budget_mb: float = 8192.0
    target_score: float | None = None
    patience: int = 4
    stop_min_improvement: float = 0.001
    recommendation_horizon: int = 6
    recommendation_rollouts: int = 64
    recommendation_temperature: float = 1.0
    post_selection_refinement: bool = False
    refinement_trials: int = 8

    @field_validator("max_experiments")
    @classmethod
    def _check_max(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_experiments must be >= 1")
        return v


class RewardConfig(_Section):
    w_perf: float = 1.0
    w_cost: float = 0.2
    w_novelty: float = 0.1
    stagnation_penalty: float = 0.05
    invalid_penalty: float = 0.5
    failure_penalty: float = 0.2
    terminal_bonus: float = 1.0
    perf_scale: float = 0.05


class RLConfig(_Section):
    enabled: bool = True
    #: Relative to ``runtime.artifacts_dir``, so the whole artifact tree stays relocatable.
    policy_path: str = "policies/ppo_policy.pt"
    device: str = "auto"
    seed: int | None = None  # None means "inherit runtime.seed"
    hidden_sizes: list[int] = Field(default_factory=lambda: [256, 256])
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.02
    entropy_coef_final: float = 0.002
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 256
    rollout_steps: int = 2048
    #: Environment steps per task type for a default run. Measured throughput on this
    #: implementation is roughly 24 steps/s on CPU, so this is ~17 minutes per task type;
    #: raise it for an overnight run (200_000 is ~2h20m per task type).
    total_steps: int = 25_000
    max_episode_steps: int = 20
    normalize_reward: bool = True
    reward: RewardConfig = Field(default_factory=RewardConfig)
    warm_start_recommendation: bool = True


class ModelsConfig(_Section):
    enabled: list[str] = Field(
        default_factory=lambda: [
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "xgboost",
            "lightgbm",
            "mlp",
        ]
    )
    cost_reference_s: float = 60.0


class ExecutionConfig(_Section):
    cv_folds: int = 0
    fail_fast: bool = False
    max_parallel: int = 1
    inference_benchmark_rows: int = 1000
    memory_sample_interval_s: float = 0.1
    allow_dimensionality_reduction: bool = True


class PackagingConfig(_Section):
    include_onnx: bool = False
    include_readme: bool = True
    include_results: bool = True
    inference_requirements: list[str] = Field(
        default_factory=lambda: [
            "numpy>=1.26",
            "pandas>=2.2",
            "scipy>=1.11",
            "scikit-learn>=1.4",
            "joblib>=1.3",
        ]
    )


class MemoryConfig(_Section):
    enabled: bool = True
    #: Relative to ``runtime.artifacts_dir``, like every other artifact path.
    db_path: str = "memory.db"
    similarity_k: int = 5
    min_similarity: float = 0.0


class ApiConfig(_Section):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_concurrent_runs: int = 2


class AutoMLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    packaging: PackagingConfig = Field(default_factory=PackagingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

    # populated by :meth:`load`
    source_path: str | None = None

    # -- loading -----------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None, apply_env: bool = True) -> AutoMLConfig:
        """Build a config from the default file, an optional override, and the env."""
        data: dict[str, Any] = {}
        resolved = _resolve_config_path(path)
        if resolved is not None:
            data = _read_yaml(resolved)
        elif path is not None:
            raise ConfigError(f"config file not found: {path}")

        if apply_env:
            data = _deep_merge(data, _env_overrides())

        config = cls.model_validate(data)
        config.source_path = str(resolved) if resolved is not None else None
        config.validate_all()
        return config

    def validate_all(self) -> None:
        self.dataset.validate_ratios()
        if self.search.min_experiments > self.search.max_experiments:
            raise ConfigError(
                "search.min_experiments cannot exceed search.max_experiments",
                min_experiments=self.search.min_experiments,
                max_experiments=self.search.max_experiments,
            )
        unknown = sorted(set(self.models.enabled) - set(_known_model_keys()))
        if unknown:
            raise ConfigError("unknown model keys in models.enabled", unknown=unknown)

    # -- helpers -----------------------------------------------------------------

    def artifacts_dir(self) -> Path:
        return Path(self.runtime.artifacts_dir).expanduser()

    def artifact_path(self, raw: str | Path) -> Path:
        """Anchor a configured artifact path on the artifact tree.

        An absolute path is honoured as given; a relative one is resolved under
        :meth:`artifacts_dir`. That is what makes the artifact tree relocatable, and it is
        why pointing ``artifacts_dir`` at a temporary directory during a test also moves
        the policy checkpoints and the memory database there instead of leaving them in
        the real tree.
        """
        candidate = Path(raw).expanduser()
        return candidate if candidate.is_absolute() else self.artifacts_dir() / candidate

    def policy_destination(self, task_type: TaskType | str | None = None) -> Path:
        """Return the canonical write path for a generic or task-specific checkpoint."""
        base = self.artifact_path(self.rl.policy_path)
        if task_type is None:
            return base
        name = task_type.value if isinstance(task_type, TaskType) else str(task_type)
        return base.with_name(f"{base.stem}_{name}{base.suffix}")

    def policy_path(self, task_type: TaskType | str | None = None) -> Path:
        """Resolve a checkpoint for loading, with legacy generic-file fallback."""
        candidate = self.policy_destination(task_type)
        if task_type is not None and not candidate.is_file():
            return self.policy_destination()
        return candidate

    def available_policy_paths(self) -> list[Path]:
        """List task-specific checkpoints plus the legacy generic checkpoint if present."""
        candidates = [self.policy_destination(task) for task in TaskType]
        candidates.append(self.policy_destination())
        return [path for path in candidates if path.is_file()]

    def memory_db_path(self) -> Path:
        return self.artifact_path(self.memory.db_path)

    def remote_dataset_dir(self) -> Path:
        """Where curated public downloads are cached inside the artifact tree."""
        return self.artifact_path("datasets/remote")

    def ensure_directories(self) -> Path:
        """Create the artifact tree and return its root."""
        root = self.artifacts_dir()
        for child in ("datasets", "runs", "meta", "policies"):
            (root / child).mkdir(parents=True, exist_ok=True)
        if self.memory.enabled:
            self.memory_db_path().parent.mkdir(parents=True, exist_ok=True)
        return root

    def to_snapshot(self) -> dict[str, Any]:
        """Serialisable config, stored in every run object for reproducibility."""
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------


def _known_model_keys() -> set[str]:
    from rl_automl.search.model_registry import registered_keys

    return set(registered_keys())


def _resolve_config_path(path: str | Path | None) -> Path | None:
    if path is not None:
        candidate = Path(path).expanduser()
        return candidate if candidate.is_file() else None

    env_path = os.environ.get(ENV_CONFIG_VAR)
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return candidate

    for candidate in (
        Path.cwd() / "configs" / "default.yaml",
        Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
    ):
        if candidate.is_file():
            return candidate
    return None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - malformed user file
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"config root must be a mapping, got {type(loaded).__name__}")
    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce(raw: str) -> Any:
    """Interpret an env string as YAML so ints, floats, bools and lists all work."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Translate ``AUTOML_RL__REWARD__W_COST=0.3`` into ``{"rl": {"reward": {"w_cost": 0.3}}}``."""
    environ = os.environ if environ is None else environ
    out: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX) or key in (ENV_CONFIG_VAR,):
            continue
        remainder = key[len(ENV_PREFIX) :]
        if "__" not in remainder:
            continue
        parts = [segment.lower() for segment in remainder.split("__")]
        cursor = out
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce(raw)
    return out


__all__ = [
    "ENV_PREFIX",
    "ApiConfig",
    "AutoMLConfig",
    "DatasetConfig",
    "ExecutionConfig",
    "MemoryConfig",
    "ModelsConfig",
    "PackagingConfig",
    "RLConfig",
    "RewardConfig",
    "RuntimeConfig",
    "SearchConfig",
    "SecurityConfig",
    "TaskConfig",
]
