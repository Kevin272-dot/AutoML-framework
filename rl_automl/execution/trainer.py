"""Model training (spec §11).

One job: turn an :class:`ExperimentSpec` into a fitted model plus the measurements the
comparison table needs. It does not decide *what* to train, and it does not decide what
to do with the results -- those belong to the RL agent and the evaluator respectively.

The preprocessor is fitted here, on the training split only, and is then part of the
saved artifact, so inference reproduces training-time transformations exactly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.errors import ModelFitError
from rl_automl.core.logging import get_logger
from rl_automl.core.types import ColumnKind, ExperimentSpec, TaskType
from rl_automl.execution.preprocessing import PreprocessorBuilder, finalise_feature_names
from rl_automl.execution.resource_monitor import ResourceMonitor
from rl_automl.search.model_registry import ModelSpec

logger = get_logger("execution.trainer")


@dataclass
class TrainedModel:
    """A fitted pipeline plus its provenance."""

    experiment_id: str
    model_key: str
    model: Any
    preprocessor: Pipeline
    feature_names: list[str]
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    preprocessing: list[str] = field(default_factory=list)
    feature_selection: str = "none"
    preset_index: int = 0
    training_time_s: float = 0.0
    peak_memory_mb: float = 0.0
    mean_memory_mb: float = 0.0
    n_parameters: int | None = None
    n_features_in: int = 0
    n_features_out: int = 0
    native_categorical_columns: list[str] = field(default_factory=list)
    categorical_feature_indices: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preprocessing_metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "model": self.model_key,
            "hyperparameters": dict(self.hyperparameters),
            "preprocessing": list(self.preprocessing),
            "feature_selection": self.feature_selection,
            "preset_index": self.preset_index,
            "n_features_in": self.n_features_in,
            "n_features_out": self.n_features_out,
            "n_parameters": self.n_parameters,
            "training_time_s": round(self.training_time_s, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }


@dataclass
class PredictionBundle:
    predictions: np.ndarray
    probabilities: np.ndarray | None = None
    scores: np.ndarray | None = None
    inference_time_s: float = 0.0


class ModelTrainer:
    def __init__(self, config: AutoMLConfig | None = None, seed: int = 0) -> None:
        self.config = config or AutoMLConfig()
        self.seed = seed
        self.preprocessor_builder = PreprocessorBuilder(self.config.dataset)

    # -- training ----------------------------------------------------------------

    def train(
        self,
        spec: ModelSpec,
        experiment: ExperimentSpec,
        task_type: TaskType,
        X_train: Any,
        y_train: Any = None,
        column_kinds: dict[str, ColumnKind] | None = None,
        memory_limit_mb: float | None = None,
    ) -> TrainedModel:
        available, reason = spec.is_available()
        if not available:
            raise ModelFitError(f"{spec.display_name} is unavailable: {reason}", model=spec.key)

        preset = spec.preset(experiment.preset_index, task_type)
        hyperparameters = {**preset.params, **experiment.hyperparameters}

        pipeline, metadata = self.preprocessor_builder.build(
            spec,
            experiment.preprocessing,
            experiment.feature_selection,
            task_type,
            column_kinds,
        )
        warnings: list[str] = list(metadata.get("warnings", []))

        n_rows = len(X_train)
        if n_rows == 0:
            raise ModelFitError("training split is empty")

        artifact_seed = _derive_model_seed(self.seed, experiment.experiment_id, spec.key)
        monitor = ResourceMonitor(memory_limit_mb=memory_limit_mb, interval_s=0.05)

        started = time.perf_counter()
        try:
            with monitor:
                # Forced through np.asarray: scikit-learn 1.9 may hand back a DataFrame
                # from a ColumnTransformer, and an estimator that sees named columns at
                # fit time but a bare array at predict time warns and can behave
                # inconsistently. One representation in both directions removes the
                # whole class of problem.
                transformed = np.asarray(pipeline.fit_transform(X_train, y_train), dtype=np.float64)
                feature_names = finalise_feature_names(pipeline, metadata)
                categorical_indices = _categorical_indices(metadata)

                estimator = spec.builder(task_type, hyperparameters, artifact_seed)
                fit_kwargs = _native_categorical_kwargs(
                    spec, metadata, categorical_indices, transformed
                )
                if spec.fit_requires_target:
                    if y_train is None:
                        raise ModelFitError(
                            f"{spec.display_name} requires a target but none was provided"
                        )
                    estimator.fit(transformed, y_train, **fit_kwargs)
                else:
                    estimator.fit(transformed, **fit_kwargs)
        except (ModelFitError, MemoryError):
            raise
        except Exception as exc:
            raise ModelFitError(
                f"{spec.display_name} failed to fit: {type(exc).__name__}: {exc}",
                model=spec.key,
                hyperparameters=hyperparameters,
            ) from exc

        training_time = time.perf_counter() - started
        snapshot = monitor.snapshot()

        if snapshot.memory_limit_exceeded:
            raise ModelFitError(
                f"training exceeded the {memory_limit_mb:.0f} MB memory budget",
                peak_memory_mb=snapshot.peak_memory_mb,
            )

        n_out = int(np.asarray(transformed).shape[1]) if len(transformed) else len(feature_names)
        if n_out != len(feature_names):
            warnings.append(
                f"feature-name count ({len(feature_names)}) differs from the transformed "
                f"width ({n_out}); regenerating generic names"
            )
            feature_names = [f"feature_{i}" for i in range(n_out)]

        return TrainedModel(
            experiment_id=experiment.experiment_id,
            model_key=spec.key,
            model=estimator,
            preprocessor=pipeline,
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            preprocessing=sorted(set(metadata.get("toggles", []))),
            feature_selection=metadata.get("feature_selection", "none"),
            preset_index=experiment.preset_index,
            training_time_s=training_time,
            peak_memory_mb=snapshot.peak_memory_mb,
            mean_memory_mb=snapshot.mean_memory_mb,
            n_parameters=_count_parameters(estimator),
            n_features_in=int(metadata.get("n_features_in", len(column_kinds or {})))
            or len(column_kinds or {}),
            n_features_out=n_out,
            native_categorical_columns=list(metadata.get("native_categorical_columns", [])),
            categorical_feature_indices=categorical_indices,
            warnings=warnings,
            preprocessing_metadata=metadata,
        )

    # -- inference ---------------------------------------------------------------

    def predict(self, trained: TrainedModel, X: Any) -> PredictionBundle:
        """Transform (never re-fit) and predict. Used for validation, test and timing."""
        if len(X) == 0:
            return PredictionBundle(predictions=np.empty(0))

        started = time.perf_counter()
        try:
            transformed = np.asarray(trained.preprocessor.transform(X), dtype=np.float64)
        except Exception as exc:
            raise ModelFitError(
                f"preprocessing failed at inference time: {type(exc).__name__}: {exc}",
                model=trained.model_key,
            ) from exc

        model = trained.model
        predictions = np.asarray(model.predict(transformed))

        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = np.asarray(model.predict_proba(transformed))
            except Exception:  # pragma: no cover - model specific
                probabilities = None

        scores = None
        if probabilities is None and hasattr(model, "decision_function"):
            try:
                scores = np.asarray(model.decision_function(transformed))
            except Exception:  # pragma: no cover - model specific
                scores = None

        return PredictionBundle(
            predictions=predictions,
            probabilities=probabilities,
            scores=scores,
            inference_time_s=time.perf_counter() - started,
        )

    # -- sizing ------------------------------------------------------------------

    @staticmethod
    def measure_inference(trained: TrainedModel, X: Any, n_rows: int = 1000) -> tuple[float, float]:
        """Return (total_seconds, seconds_per_1000_rows) for a timed prediction pass."""
        if len(X) == 0 or not hasattr(trained.model, "predict"):
            return 0.0, 0.0
        sample = X.iloc[:n_rows] if hasattr(X, "iloc") else X[:n_rows]
        started = time.perf_counter()
        trained.model.predict(np.asarray(trained.preprocessor.transform(sample), dtype=np.float64))
        elapsed = time.perf_counter() - started
        return elapsed, elapsed / max(len(sample), 1) * 1000


def _categorical_indices(metadata: dict[str, Any]) -> list[int]:
    """Positions of the categorical block inside the transformed feature matrix.

    Used to tell LightGBM which columns are categorical. The column transformer emits
    blocks in declaration order: numeric, categorical, datetime, missing indicators.
    """
    if not metadata.get("native_categorical_columns"):
        return []
    start = len(metadata.get("numeric_features", []))
    n_categorical = len(metadata.get("categorical_features", []))
    return list(range(start, start + n_categorical))


def _native_categorical_kwargs(
    spec: ModelSpec,
    metadata: dict[str, Any],
    indices: list[int],
    transformed: Any,
) -> dict[str, Any]:
    """LightGBM can consume integer-encoded categorical columns natively."""
    if not indices or spec.framework != "lightgbm":
        return {}
    width = int(np.asarray(transformed).shape[1]) if len(transformed) else 0
    if width and max(indices) >= width:
        return {}
    return {"categorical_feature": indices}


def _count_parameters(model: Any) -> int | None:
    """Parameter count where it is meaningful, otherwise ``None``."""
    n_parameters = getattr(model, "n_parameters", None)
    if isinstance(n_parameters, int) and n_parameters > 0:
        return n_parameters

    total = 0
    for attribute in ("coef_", "intercept_"):
        if hasattr(model, attribute):
            try:
                total += int(np.asarray(getattr(model, attribute)).size)
            except Exception:  # pragma: no cover
                continue
    return total or None


def _derive_model_seed(base_seed: int, experiment_id: str, model_key: str) -> int:
    from rl_automl.core.seeding import derive_seed

    return derive_seed(base_seed, experiment_id, model_key)


__all__ = ["ModelTrainer", "PredictionBundle", "TrainedModel"]
