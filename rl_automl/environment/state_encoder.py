"""State encoding (spec §5).

The observation is a fixed-length, bounded vector assembled from four sources:

1. **Task** - one-hot over the five supported task types.
2. **Dataset fingerprint** - the 40-dimension condensation from ``dataset.fingerprint``.
3. **Search progress** - how far through the budget we are, how good the best result is,
   how much it has improved recently, whether the search has stalled, how diverse it has
   been, and which *model families* are empirically doing well on this dataset. That last
   block is what makes the policy adapt within an episode rather than just at step zero.
4. **Memory** - best score and best family from historically similar datasets
   (spec §24). Zeros until the memory database has entries, which keeps the state shape
   stable whether or not memory is enabled.

Every component is in roughly ``[-1, 1]``. Score-like quantities are expressed *relative
to the first score observed in the episode*, which makes the encoding scale-free: an F1 of
0.9 and an RMSE of 53 both land in the same range, so one policy can serve every task
type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rl_automl.core.types import DatasetFingerprint, TaskType
from rl_automl.dataset.fingerprint import FINGERPRINT_FEATURES

STATE_SCHEMA_VERSION = "1.1.0"

#: Fixed task ordering. Appending to ``TaskType`` requires a schema version bump.
TASK_ORDER: tuple[TaskType, ...] = (
    TaskType.CLASSIFICATION,
    TaskType.REGRESSION,
    TaskType.CLUSTERING,
    TaskType.ANOMALY_DETECTION,
    TaskType.DIMENSIONALITY_REDUCTION,
)

#: Fixed model-family ordering. Families come from the registry, so an unknown family
#: (a newly registered algorithm) maps to the trailing "other" slot rather than changing
#: the state dimension.
FAMILY_ORDER: tuple[str, ...] = (
    "linear",
    "bagging",
    "boosting",
    "kernel",
    "neural",
    "centroid",
    "density",
    "hierarchical",
    "probabilistic",
    "ensemble",
    "neighborhood",
    "other",
)

PROGRESS_SCALARS: tuple[str, ...] = (
    "experiments_fraction",
    "budget_fraction",
    "elapsed_fraction",
    "best_relative",
    "best_improvement",
    "last_gain",
    "mean_gain",
    "stagnation",
    "model_diversity",
    "experiment_diversity",
    "mean_novelty",
    "last_failed",
)

MEMORY_SCALARS: tuple[str, ...] = (
    "memory_best_relative",
    "memory_similarity",
    "memory_attempts_to_best",
    "memory_n_similar",
)


@dataclass
class SearchProgress:
    """Everything the encoder needs to know about the episode so far."""

    step: int = 0
    max_steps: int = 1
    n_experiments: int = 0

    #: Oriented scores (larger is always better), so the encoder is task-agnostic.
    best_score: float | None = None
    first_score: float | None = None
    last_gain: float = 0.0
    mean_gain: float = 0.0
    stagnation: int = 0

    n_distinct_models: int = 0
    n_distinct_signatures: int = 0
    n_possible_models: int = 1
    mean_novelty: float = 1.0
    last_failed: bool = False

    elapsed_fraction: float = 0.0
    budget_fraction: float = 0.0

    #: family -> mean normalised performance observed in this episode.
    family_performance: dict[str, float] = field(default_factory=dict)

    #: Cross-dataset memory (spec §24). Left at defaults when memory is empty.
    memory_best_score: float | None = None
    memory_similarity: float = 0.0
    memory_attempts_to_best: float | None = None
    memory_n_similar: int = 0
    memory_family_affinity: dict[str, float] = field(default_factory=dict)


class StateEncoder:
    """Deterministic, versioned observation builder."""

    def __init__(self, model_families: dict[str, str] | None = None) -> None:
        self.model_families = dict(model_families or {})
        self._family_index = {family: index for index, family in enumerate(FAMILY_ORDER)}
        self.feature_names = self._build_feature_names()

    # -- shape -------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    @property
    def schema_version(self) -> str:
        return STATE_SCHEMA_VERSION

    def _build_feature_names(self) -> list[str]:
        names = [f"task_{task.value}" for task in TASK_ORDER]
        names.extend(f"fp_{name}" for name in FINGERPRINT_FEATURES)
        names.extend(f"progress_{name}" for name in PROGRESS_SCALARS)
        names.extend(f"memory_{name}" for name in MEMORY_SCALARS)
        names.extend(f"family_{family}" for family in FAMILY_ORDER)
        names.append("family_coverage")
        return names

    def describe(self) -> dict[str, object]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "dim": self.dim,
            "n_task_features": len(TASK_ORDER),
            "n_fingerprint_features": len(FINGERPRINT_FEATURES),
            "n_progress_features": len(PROGRESS_SCALARS),
            "n_memory_features": len(MEMORY_SCALARS),
            "n_family_features": len(FAMILY_ORDER),
        }

    # -- encoding ----------------------------------------------------------------

    def encode(
        self,
        fingerprint: DatasetFingerprint | None,
        task_type: TaskType,
        progress: SearchProgress | None = None,
    ) -> np.ndarray:
        progress = progress or SearchProgress()
        vector = np.zeros(self.dim, dtype=np.float64)
        cursor = 0

        # 1. task one-hot
        if task_type in TASK_ORDER:
            vector[cursor + TASK_ORDER.index(task_type)] = 1.0
        cursor += len(TASK_ORDER)

        # 2. dataset fingerprint
        n_fingerprint = len(FINGERPRINT_FEATURES)
        if fingerprint is not None and len(fingerprint.values) == n_fingerprint:
            vector[cursor : cursor + n_fingerprint] = np.asarray(
                fingerprint.values, dtype=np.float64
            )
        elif fingerprint is not None:
            # A fingerprint built by a different schema is unusable; leaving zeros is
            # better than raising inside a training loop.
            shared = min(n_fingerprint, len(fingerprint.values))
            vector[cursor : cursor + shared] = np.asarray(
                fingerprint.values[:shared], dtype=np.float64
            )
        cursor += n_fingerprint

        # 3. search progress
        vector[cursor : cursor + len(PROGRESS_SCALARS)] = self._progress_block(progress)
        cursor += len(PROGRESS_SCALARS)

        # 4. cross-dataset memory
        vector[cursor : cursor + len(MEMORY_SCALARS)] = self._memory_block(progress)
        cursor += len(MEMORY_SCALARS)

        # 5. family performance + coverage
        vector[cursor : cursor + len(FAMILY_ORDER)] = self._family_block(progress)
        cursor += len(FAMILY_ORDER)
        vector[cursor] = _clip01(len(progress.family_performance) / max(len(FAMILY_ORDER) - 1, 1))

        return np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)

    # -- blocks ------------------------------------------------------------------

    def _progress_block(self, progress: SearchProgress) -> np.ndarray:
        max_steps = max(progress.max_steps, 1)
        first = progress.first_score
        reference = max(abs(first), 1e-6) if first is not None else 1.0

        best_relative = (
            _tanh((progress.best_score or 0.0) / reference)
            if progress.best_score is not None
            else 0.0
        )
        best_improvement = (
            _tanh(((progress.best_score or 0.0) - first) / reference)
            if progress.best_score is not None and first is not None
            else 0.0
        )

        return np.asarray(
            [
                _clip01(progress.n_experiments / max_steps),
                _clip01(progress.budget_fraction),
                _clip01(progress.elapsed_fraction),
                best_relative,
                best_improvement,
                _tanh(progress.last_gain / reference),
                _tanh(progress.mean_gain / reference),
                _clip01(progress.stagnation / max_steps),
                _clip01(progress.n_distinct_models / max(progress.n_possible_models, 1)),
                _clip01(progress.n_distinct_signatures / max(progress.n_experiments, 1)),
                _clip01(progress.mean_novelty),
                1.0 if progress.last_failed else 0.0,
            ],
            dtype=np.float64,
        )

    def _memory_block(self, progress: SearchProgress) -> np.ndarray:
        reference = max(abs(progress.first_score or 0.0), 1e-6)
        memory_best = (
            _tanh(progress.memory_best_score / reference)
            if progress.memory_best_score is not None
            else 0.0
        )
        attempts = (
            _clip01((progress.memory_attempts_to_best or 0.0) / max(progress.max_steps, 1))
            if progress.memory_attempts_to_best is not None
            else 0.0
        )
        return np.asarray(
            [
                memory_best,
                _clip01(progress.memory_similarity),
                attempts,
                _clip01(progress.memory_n_similar / 10.0),
            ],
            dtype=np.float64,
        )

    def _family_block(self, progress: SearchProgress) -> np.ndarray:
        block = np.zeros(len(FAMILY_ORDER), dtype=np.float64)
        for family, value in progress.family_performance.items():
            index = self._family_index.get(family)
            if index is None:
                index = self._family_index["other"]
            block[index] = max(block[index], float(value))
        return block


def family_index(family: str) -> int:
    return FAMILY_ORDER.index(family) if family in FAMILY_ORDER else FAMILY_ORDER.index("other")


def _tanh(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.tanh(value))


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


__all__ = [
    "FAMILY_ORDER",
    "MEMORY_SCALARS",
    "PROGRESS_SCALARS",
    "STATE_SCHEMA_VERSION",
    "TASK_ORDER",
    "SearchProgress",
    "StateEncoder",
    "family_index",
]
