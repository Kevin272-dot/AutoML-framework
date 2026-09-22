"""Model package export (spec §16, §17, §21).

Builds the ``model_package/`` directory: the fitted model, the fitted preprocessor, the
metadata that documents both, and the schema that says what the inputs must look like.

Design rules:

* **Metadata is not the training environment.** No raw rows, no sample values beyond a
  handful of illustrative category labels, no file paths from the user's machine. Column
  *names* are kept because inference cannot work without them (spec §17).
* **Optional exports are verified or dropped.** ONNX is only included when the converted
  model reproduces the native predictions within tolerance on a probe sample; otherwise it
  is skipped with a recorded reason rather than shipping an artifact that silently
  disagrees with the model (spec §21).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.errors import PackagingError
from rl_automl.core.logging import get_logger
from rl_automl.core.types import (
    ColumnKind,
    DatasetProfile,
    ExperimentResult,
    TaskSpec,
    TaskType,
    utc_now_iso,
)
from rl_automl.execution.trainer import TrainedModel
from rl_automl.packaging.runtime_bundle import write_runtime_bundle
from rl_automl.search.model_registry import get_spec

logger = get_logger("packaging.model_exporter")

PACKAGE_DIRNAME = "model_package"
ONNX_ATOL = 1e-4
SCHEMA_MAX_EXAMPLES = 8


@dataclass
class FeatureField:
    """One input column the caller must supply."""

    name: str
    kind: str
    dtype: str
    required: bool = True
    nullable: bool = True
    n_unique: int = 0
    example_values: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
            "required": self.required,
            "nullable": self.nullable,
            "n_unique": self.n_unique,
            "example_values": list(self.example_values),
            "notes": self.notes,
        }


@dataclass
class FeatureSchema:
    """The inference contract: exactly which columns go in, and what comes out."""

    features: list[FeatureField] = field(default_factory=list)
    target: str | None = None
    task_type: str = ""
    metric: str = ""
    excluded_columns: list[str] = field(default_factory=list)
    n_features_expected: int = 0
    output: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "metric": self.metric,
            "target": self.target,
            "n_features_expected": self.n_features_expected,
            "excluded_columns": list(self.excluded_columns),
            "output": dict(self.output),
            "features": [feature.to_dict() for feature in self.features],
        }


@dataclass
class ModelPackage:
    directory: Path
    metadata: dict[str, Any]
    schema: FeatureSchema
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def model_path(self) -> Path:
        return self.directory / "model" / "model.joblib"

    @property
    def preprocessor_path(self) -> Path:
        return self.directory / "preprocessing" / "preprocessor.joblib"

    @property
    def inference_path(self) -> Path:
        return self.directory / "inference"

    def files(self) -> list[Path]:
        return sorted(path for path in self.directory.rglob("*") if path.is_file())

    def has_onnx(self) -> bool:
        return any(self.directory.glob("model/*.onnx"))


class ModelExporter:
    def __init__(self, config: AutoMLConfig | None = None) -> None:
        self.config = config or AutoMLConfig()

    # -- public ------------------------------------------------------------------

    def export(
        self,
        *,
        trained: TrainedModel,
        result: ExperimentResult,
        task: TaskSpec,
        profile: DatasetProfile,
        destination: str | Path,
        run_id: str = "run",
        dataset_name: str = "",
        dataset_sha256: str = "",
        column_kinds: dict[str, ColumnKind] | None = None,
        label_classes: np.ndarray | None = None,
        label_encoding_required: bool = False,
        split_summary: dict[str, Any] | None = None,
        include_onnx: bool | None = None,
        probe_frame: pd.DataFrame | None = None,
    ) -> ModelPackage:
        """Write the package directory. Does not create the archive."""
        directory = Path(destination)
        if directory.name != PACKAGE_DIRNAME:
            directory = directory / PACKAGE_DIRNAME
        if directory.exists():
            # Rebuilt from scratch so a stale file can never survive into the archive.
            import shutil

            shutil.rmtree(directory)
        for sub in ("model", "preprocessing", "metadata", "inference"):
            (directory / sub).mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []

        joblib.dump(trained.model, directory / "model" / "model.joblib")
        joblib.dump(trained.preprocessor, directory / "preprocessing" / "preprocessor.joblib")

        schema = self.build_schema(
            trained=trained,
            task=task,
            profile=profile,
            column_kinds=column_kinds or {},
            label_classes=label_classes,
        )
        model_metadata = self.build_model_metadata(
            trained=trained,
            result=result,
            task=task,
            run_id=run_id,
            dataset_name=dataset_name,
            dataset_sha256=dataset_sha256,
            label_classes=label_classes,
            label_encoding_required=label_encoding_required,
        )
        dataset_metadata = self.build_dataset_metadata(
            profile=profile,
            dataset_name=dataset_name,
            dataset_sha256=dataset_sha256,
            split_summary=split_summary or {},
        )

        # The artifact carries its own copies of the classes it pickles, so it loads on a
        # machine that has never installed AutoML.
        write_runtime_bundle(directory)

        self._write_json(directory / "metadata" / "feature_schema.json", schema.to_dict())
        self._write_json(directory / "metadata" / "model_metadata.json", model_metadata)
        self._write_json(directory / "metadata" / "dataset_metadata.json", dataset_metadata)

        extras: dict[str, str] = {}
        should_try_onnx = (
            self.config.packaging.include_onnx if include_onnx is None else include_onnx
        )
        if should_try_onnx:
            onnx_note = self._try_export_onnx(
                trained=trained, directory=directory, probe_frame=probe_frame
            )
            if onnx_note:
                warnings.append(onnx_note)
            elif (directory / "model" / "model.onnx").is_file():
                extras["onnx"] = "model/model.onnx"
                model_metadata["onnx"] = {
                    "available": True,
                    "verified": True,
                    "tolerance": ONNX_ATOL,
                }

        return ModelPackage(
            directory=directory,
            metadata=model_metadata,
            schema=schema,
            warnings=warnings,
            extras=extras,
        )

    # -- metadata ----------------------------------------------------------------

    def build_schema(
        self,
        *,
        trained: TrainedModel,
        task: TaskSpec,
        profile: DatasetProfile,
        column_kinds: dict[str, ColumnKind],
        label_classes: np.ndarray | None,
    ) -> FeatureSchema:
        by_name = {column.name: column for column in profile.columns}
        features: list[FeatureField] = []

        for name, kind in column_kinds.items():
            column = by_name.get(name)
            examples = self._example_values(column)
            features.append(
                FeatureField(
                    name=name,
                    kind=kind.value if isinstance(kind, ColumnKind) else str(kind),
                    dtype=str(column.dtype) if column else "unknown",
                    required=True,
                    nullable=bool(column is None or column.n_missing > 0),
                    n_unique=int(column.n_unique) if column else 0,
                    example_values=examples,
                    notes=(
                        "any value not seen during training is routed to the encoder's "
                        "infrequent category"
                        if column is not None and column.kind is ColumnKind.CATEGORICAL
                        else ""
                    ),
                )
            )

        excluded = sorted(
            str(column) for column in profile.columns if str(column.name) not in set(column_kinds)
        )
        output: dict[str, Any] = {
            "type": "label" if task.task_type.is_supervised else "labels",
            "column": "prediction",
        }
        if label_classes is not None and len(label_classes):
            output["classes"] = [str(label) for label in label_classes]
        if task.task_type is TaskType.CLASSIFICATION:
            output["probabilities"] = "predict_proba() returns one column per class"
        if task.task_type is TaskType.REGRESSION:
            output["description"] = "continuous value"

        return FeatureSchema(
            features=features,
            target=task.target,
            task_type=task.task_type.value,
            metric=task.metric,
            excluded_columns=excluded,
            n_features_expected=len(features),
            output=output,
        )

    def build_model_metadata(
        self,
        *,
        trained: TrainedModel,
        result: ExperimentResult,
        task: TaskSpec,
        run_id: str,
        dataset_name: str,
        dataset_sha256: str,
        label_classes: np.ndarray | None,
        label_encoding_required: bool = False,
    ) -> dict[str, Any]:
        spec = get_spec(trained.model_key)
        preset = spec.preset(trained.preset_index, task.task_type)
        metadata: dict[str, Any] = {
            "model": spec.display_name,
            "model_key": spec.key,
            "task": task.task_type.value,
            "learning_type": task.learning_type.value,
            "framework": spec.framework,
            "objective": task.objective,
            "primary_metric": result.primary_metric,
            "metric_direction": result.metric_direction.value,
            "validation_score": result.validation_score,
            "test_score": result.test_score,
            "validation_metrics": result.validation_metrics,
            "test_metrics": result.test_metrics,
            "features": trained.n_features_out,
            "raw_features": len(trained.preprocessing_metadata.get("numeric_features", []))
            + len(trained.preprocessing_metadata.get("categorical_features", []))
            + len(trained.preprocessing_metadata.get("datetime_features", [])),
            "preprocessing": list(trained.preprocessing),
            "feature_selection": trained.feature_selection,
            "hyperparameters": dict(trained.hyperparameters),
            "preset": {"index": trained.preset_index, "name": preset.name},
            "training_timestamp": utc_now_iso(),
            "run_id": run_id,
            "dataset": {"name": dataset_name, "sha256": dataset_sha256},
            "resource_usage": {
                "training_time_s": round(trained.training_time_s, 4),
                "peak_memory_mb": round(trained.peak_memory_mb, 2),
            },
            "n_parameters": trained.n_parameters,
            "artifacts": {
                "model": "model/model.joblib",
                "preprocessor": "preprocessing/preprocessor.joblib",
            },
            "library_versions": _library_versions(spec.framework),
            "excluded_raw_columns": trained.preprocessing_metadata.get("dropped_features", []),
        }
        # Only recorded when the target actually had to be encoded. A target that was already
        # 0..k-1 needs no decoding, and emitting a mapping anyway would make the shipped
        # loader return strings where the in-process model returns integers.
        if label_encoding_required and label_classes is not None and len(label_classes):
            metadata["label_mapping"] = {
                str(index): str(label) for index, label in enumerate(label_classes)
            }
        if trained.warnings:
            metadata["warnings"] = list(dict.fromkeys(trained.warnings))
        return metadata

    def build_dataset_metadata(
        self,
        *,
        profile: DatasetProfile,
        dataset_name: str,
        dataset_sha256: str,
        split_summary: dict[str, Any],
    ) -> dict[str, Any]:
        distribution = profile.class_distribution
        payload: dict[str, Any] = {
            "name": dataset_name,
            "sha256": dataset_sha256,
            "n_rows": profile.n_rows,
            "n_cols": profile.n_cols,
            "target": profile.target,
            "missing_ratio": round(profile.missing_ratio, 6),
            "duplicate_ratio": round(profile.duplicate_ratio, 6),
            "numeric_features": len(profile.numeric_features),
            "categorical_features": len(profile.categorical_features),
            "datetime_features": len(profile.datetime_features),
            "high_cardinality_features": len(profile.high_cardinality_features),
            "split": split_summary,
            "warnings": list(profile.warnings),
        }
        if distribution is not None:
            payload["class_distribution"] = {
                "n_classes": distribution.n_classes,
                "minority_ratio": distribution.minority_ratio,
                "is_imbalanced": distribution.is_imbalanced,
            }
        return payload

    # -- ONNX --------------------------------------------------------------------

    def _try_export_onnx(
        self,
        *,
        trained: TrainedModel,
        directory: Path,
        probe_frame: pd.DataFrame | None,
    ) -> str | None:
        """Export ONNX only if conversion works and predictions still agree.

        Returns a note describing why it was skipped, or ``None`` on success.
        """
        spec = get_spec(trained.model_key)
        if not spec.supports_onnx:
            return f"ONNX export skipped: {spec.display_name} has no reliable converter"

        if probe_frame is None or probe_frame.empty:
            return "ONNX export skipped: no probe data available to verify the conversion"

        try:
            from skl2onnx import to_onnx
        except ImportError:
            return (
                "ONNX export skipped: the 'onnx' extra is not installed "
                "(pip install rl-automl[onnx])"
            )

        try:
            sample = trained.preprocessor.transform(probe_frame.head(64))
            sample = np.asarray(sample, dtype=np.float32)
            native = np.asarray(trained.model.predict(sample))

            onnx_model = to_onnx(trained.model, sample[:1], target_opset=None)
            destination = directory / "model" / "model.onnx"
            with destination.open("wb") as handle:
                handle.write(onnx_model.SerializeToString())

            if not self._verify_onnx(destination, sample, native):
                destination.unlink(missing_ok=True)
                return (
                    "ONNX export skipped: the converted model disagreed with the native "
                    f"model beyond {ONNX_ATOL} on the probe sample"
                )
        except Exception as exc:
            return f"ONNX export skipped: conversion failed ({type(exc).__name__}: {exc})"

        return None

    @staticmethod
    def _verify_onnx(path: Path, sample: np.ndarray, native: np.ndarray) -> bool:
        """Compare ONNX Runtime output against the native predictions."""
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: sample})
        except Exception:  # pragma: no cover - runtime/driver dependent
            return False

        if not outputs:
            return False
        converted = np.asarray(outputs[0])

        try:
            if native.dtype.kind in "iu" or native.dtype.kind == "O":
                # Class predictions: compare the argmax of whatever the graph returns.
                if converted.ndim > 1:
                    converted = np.argmax(converted, axis=1)
                return bool(np.array_equal(np.asarray(converted).reshape(-1), native.reshape(-1)))
            return bool(np.allclose(converted.reshape(native.shape), native, atol=ONNX_ATOL))
        except Exception:  # pragma: no cover - shape mismatch means "not equivalent"
            return False

    # -- helpers -----------------------------------------------------------------

    @staticmethod
    def _example_values(column: Any) -> list[str]:
        if column is None:
            return []
        values = [
            str(entry.get("value"))
            for entry in getattr(column, "top_values", [])[:SCHEMA_MAX_EXAMPLES]
        ]
        return values

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        try:
            path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk issues
            raise PackagingError(f"could not write {path}: {exc}") from exc


def _json_default(value: Any) -> Any:
    """Make numpy scalars and arrays JSON-serialisable."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    return str(value)


def _library_versions(framework: str) -> dict[str, str]:
    import importlib.metadata as metadata

    packages = ["numpy", "pandas", "scikit-learn", "joblib"]
    if framework in ("xgboost", "lightgbm"):
        packages.append(framework)
    if framework == "torch":
        packages.append("torch")

    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover
            continue
    return versions


__all__ = [
    "PACKAGE_DIRNAME",
    "FeatureField",
    "FeatureSchema",
    "ModelExporter",
    "ModelPackage",
]
