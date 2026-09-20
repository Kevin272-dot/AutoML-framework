"""Preprocessing construction (spec §7, §11).

The RL agent chooses preprocessing *toggles*; this module turns those toggles into a
fitted scikit-learn pipeline. Design rules:

* **Fit on train only.** The preprocessor is always fitted inside the pipeline that
  includes the model, using the training split alone. It is transformed (never
  re-fitted) for validation and test data, which is what keeps the test set untouched.
* **Missing values are always handled.** Every model in the registry either requires
  imputation or is safer with it, so imputation always runs; the toggles select the
  *strategy* (median vs mean, most-frequent vs explicit marker) rather than toggling it
  on and off. This guarantees no NaN ever reaches an estimator.
* **Unknown categories never crash inference.** One-hot encoding uses
  ``infrequent_if_exist`` bucketing and ``handle_unknown`` so unseen categories at
  predict time degrade gracefully instead of raising.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from rl_automl.core.config import DatasetConfig
from rl_automl.core.errors import ModelFitError
from rl_automl.core.logging import get_logger
from rl_automl.core.types import ColumnKind, TaskType
from rl_automl.core.vocabulary import (
    SUPERVISED_ONLY_FEATURE_SELECTION,
    FeatureSelectionOption,
    PreprocessingOption,
)
from rl_automl.execution.runtime_transformers import (
    DATETIME_FEATURE_SUFFIXES,
    DatetimeExpander,
    OutlierClipper,
    SignedLog1p,
)
from rl_automl.search.model_registry import ModelSpec

logger = get_logger("execution.preprocessing")

MAX_SELECTED_FEATURES = 60


class PreprocessorBundle:
    """A fitted preprocessing + optional feature-selection pipeline."""

    def __init__(
        self,
        pipeline: Pipeline,
        feature_names: list[str],
        numeric_features: list[str],
        categorical_features: list[str],
        datetime_features: list[str],
        dropped_features: list[str],
        toggles: list[str],
        feature_selection: str,
        native_categorical_columns: list[str],
        n_features_in: int,
        warnings: list[str],
    ) -> None:
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.datetime_features = datetime_features
        self.dropped_features = dropped_features
        self.toggles = toggles
        self.feature_selection = feature_selection
        self.native_categorical_columns = native_categorical_columns
        self.n_features_in = n_features_in
        self.warnings = warnings

    # -- sklearn-like surface ----------------------------------------------------

    def fit(self, X: Any, y: Any = None) -> PreprocessorBundle:
        self.pipeline.fit(X, y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(self.pipeline.transform(X), dtype=np.float64)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return np.asarray(self.pipeline.fit_transform(X, y), dtype=np.float64)

    def get_feature_names_out(self) -> list[str]:
        return list(self.feature_names)

    @property
    def n_features_out(self) -> int:
        return len(self.feature_names)

    def describe(self) -> dict[str, Any]:
        return {
            "toggles": list(self.toggles),
            "feature_selection": self.feature_selection,
            "n_features_in": self.n_features_in,
            "n_features_out": self.n_features_out,
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "datetime_features": list(self.datetime_features),
            "dropped_features": list(self.dropped_features),
            "native_categorical_columns": list(self.native_categorical_columns),
            "warnings": list(self.warnings),
        }


class PreprocessorBuilder:
    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()

    def build(
        self,
        spec: ModelSpec,
        toggles: list[str] | tuple[str, ...],
        feature_selection: str = "none",
        task_type: TaskType = TaskType.CLASSIFICATION,
        column_kinds: dict[str, ColumnKind] | None = None,
    ) -> tuple[Pipeline, dict[str, Any]]:
        """Construct an unfitted pipeline plus build metadata."""
        warnings: list[str] = []
        active = set(toggles)

        forbidden = set(spec.forbidden_preprocessing)
        if active & forbidden:
            dropped = sorted(active & forbidden)
            warnings.append(f"{spec.display_name} does not support {', '.join(dropped)}; ignored")
            active -= forbidden

        kinds = column_kinds or {}
        numeric = [name for name, kind in kinds.items() if kind is ColumnKind.NUMERIC]
        categorical = [name for name, kind in kinds.items() if kind is ColumnKind.CATEGORICAL]
        datetimes = [name for name, kind in kinds.items() if kind is ColumnKind.DATETIME]
        other = [name for name, kind in kinds.items() if kind is ColumnKind.OTHER]

        dropped: list[str] = list(other)
        if other:
            warnings.append(f"dropped {len(other)} column(s) of unsupported type")
        if datetimes and PreprocessingOption.DATETIME_FEATURES.value not in active:
            dropped.extend(datetimes)
            warnings.append(
                f"{len(datetimes)} datetime column(s) dropped because datetime feature "
                "expansion was not selected"
            )

        native_categorical: list[str] = []
        expand_datetime = bool(datetimes) and PreprocessingOption.DATETIME_FEATURES.value in active

        transformers: list[tuple[str, Any, Any]] = []
        output_names: list[str] = []

        if numeric:
            numeric_pipe = self._numeric_pipeline(active)
            transformers.append(("numeric", numeric_pipe, numeric))
            output_names.extend(numeric)

        if categorical:
            encode = PreprocessingOption.ENCODE_CATEGORICAL.value in active
            use_native = spec.supports_native_categorical and not encode
            if use_native:
                native_categorical = list(categorical)
                cat_pipe = self._ordinal_pipeline()
                output_names.extend(categorical)
            else:
                cat_pipe = self._categorical_pipeline(active)
                output_names.extend(self._onehot_output_names(categorical))
            transformers.append(("categorical", cat_pipe, categorical))

        if expand_datetime:
            transformers.append(("datetime", self._datetime_pipeline(active), datetimes))
            output_names.extend(
                f"{column}_{suffix}" for column in datetimes for suffix in DATETIME_FEATURE_SUFFIXES
            )

        # Indicators are computed on numeric columns only: MissingIndicator casts its
        # input to float64 internally, so it cannot be pointed at categorical text
        # columns. Categorical missingness is still visible to the model through the
        # explicit "__missing__" fill value.
        if PreprocessingOption.MISSING_INDICATOR.value in active and numeric:
            transformers.append(("missing_indicator", MissingIndicator(features="all"), numeric))
            output_names.extend(f"{column}__isna" for column in numeric)

        if not transformers:
            raise ModelFitError(
                "no usable features remain after preprocessing",
                n_numeric=len(numeric),
                n_categorical=len(categorical),
                n_datetime=len(datetimes),
            )

        column_transformer = ColumnTransformer(
            transformers=transformers, remainder="drop", sparse_threshold=0.0
        )

        selection = self._resolve_feature_selection(
            feature_selection, task_type, output_names, warnings
        )
        steps: list[tuple[str, Any]] = [("columns", column_transformer)]
        if selection is not None:
            steps.append(("selection", selection))

        pipeline = Pipeline(steps)

        metadata = {
            "numeric_features": numeric,
            "categorical_features": categorical,
            "datetime_features": datetimes if expand_datetime else [],
            "dropped_features": dropped,
            "native_categorical_columns": native_categorical,
            "toggles": sorted(active),
            "feature_selection": feature_selection,
            "output_names": output_names,
            "warnings": warnings,
        }
        return pipeline, metadata

    # -- sub-pipelines -----------------------------------------------------------

    def _numeric_pipeline(self, active: set[str]) -> Pipeline:
        steps: list[tuple[str, Any]] = [
            (
                "impute",
                SimpleImputer(
                    strategy="median"
                    if PreprocessingOption.IMPUTE_NUMERIC.value in active
                    else "mean"
                ),
            )
        ]
        if PreprocessingOption.CLIP_OUTLIERS.value in active:
            steps.append(("clip", OutlierClipper()))
        if PreprocessingOption.LOG_TRANSFORM.value in active:
            steps.append(("log", SignedLog1p()))
        scaler = self._scaler(active)
        if scaler is not None:
            steps.append(("scale", scaler))
        return Pipeline(steps)

    def _categorical_pipeline(self, active: set[str]) -> Pipeline:
        return Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="most_frequent"
                        if PreprocessingOption.IMPUTE_CATEGORICAL.value in active
                        else "constant",
                        fill_value="__missing__",
                    ),
                ),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=5,
                        max_categories=self.config.max_cardinality_for_ohe,
                        sparse_output=False,
                    ),
                ),
            ]
        )

    @staticmethod
    def _ordinal_pipeline() -> Pipeline:
        return Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encode",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )

    def _datetime_pipeline(self, active: set[str]) -> Pipeline:
        steps: list[tuple[str, Any]] = [
            ("expand", DatetimeExpander()),
            ("impute", SimpleImputer(strategy="constant", fill_value=-1.0)),
        ]
        scaler = self._scaler(active)
        if scaler is not None:
            steps.append(("scale", scaler))
        return Pipeline(steps)

    @staticmethod
    def _scaler(active: set[str]) -> Any | None:
        if PreprocessingOption.SCALE_ROBUST.value in active:
            return RobustScaler()
        if PreprocessingOption.SCALE_STANDARD.value in active:
            return StandardScaler()
        return None

    def _onehot_output_names(self, categorical: list[str]) -> list[str]:
        """Approximate output names; exact names require a fitted encoder.

        ``min_frequency`` bucketing means the true count is only known after fitting, so
        this is an upper bound used for reporting and for ``SelectKBest`` sizing. The
        authoritative names are refreshed after fitting in :func:`finalise_feature_names`.
        """
        return [f"{column}__onehot" for column in categorical]

    def _resolve_feature_selection(
        self,
        feature_selection: str,
        task_type: TaskType,
        output_names: list[str],
        warnings: list[str],
    ) -> Any | None:
        strategy = (feature_selection or FeatureSelectionOption.NONE.value).lower()
        if strategy == FeatureSelectionOption.NONE.value:
            return None

        if strategy in SUPERVISED_ONLY_FEATURE_SELECTION and not task_type.is_supervised:
            warnings.append(
                f"feature selection '{strategy}' needs a target; "
                "falling back to variance thresholding"
            )
            strategy = FeatureSelectionOption.VARIANCE.value

        if strategy == FeatureSelectionOption.VARIANCE.value:
            return VarianceThreshold(threshold=0.0)

        if strategy == FeatureSelectionOption.PCA.value:
            return PCA(n_components=0.95, random_state=0)

        k = max(1, min(MAX_SELECTED_FEATURES, max(len(output_names), 1)))
        if strategy == FeatureSelectionOption.MUTUAL_INFO.value:
            # Classification and regression need different estimators: `mutual_info_classif`
            # treats the target as discrete, which is meaningless for a continuous target.
            score_func = (
                mutual_info_classif
                if task_type is TaskType.CLASSIFICATION
                else mutual_info_regression
            )
            return SelectKBest(score_func=score_func, k=k)
        if strategy == FeatureSelectionOption.KBEST.value:
            score_func = f_classif if task_type is TaskType.CLASSIFICATION else f_regression
            return SelectKBest(score_func=score_func, k=k)

        warnings.append(f"unknown feature selection '{feature_selection}'; using none")
        return None


def finalise_feature_names(pipeline: Pipeline, metadata: dict[str, Any]) -> list[str]:
    """Read the true output names off a fitted pipeline.

    One-hot encoding with frequency bucketing collapses rare categories, so the real
    width is only known after fitting.
    """
    columns: ColumnTransformer = pipeline.named_steps["columns"]
    names: list[str] = []
    for name, transformer, _columns in columns.transformers_:
        if name == "remainder":
            continue
        if name == "datetime":
            names.extend(transformer.get_feature_names_out())
            continue
        if name == "missing_indicator":
            names.extend(f"{column}__isna" for column in metadata.get("output_names", []))
            continue
        try:
            produced = transformer.get_feature_names_out()
            names.extend(str(item) for item in produced)
        except Exception:
            n = getattr(transformer, "n_features_in_", 0)
            names.extend(f"{name}_{i}" for i in range(int(n)))

    if "selection" in pipeline.named_steps:
        selection = pipeline.named_steps["selection"]
        if hasattr(selection, "get_support"):
            support = np.asarray(selection.get_support())
            if len(support) == len(names):
                names = [name for name, keep in zip(names, support, strict=False) if keep]
            else:
                names = [f"selected_{i}" for i in range(int(support.sum()))]
        else:
            names = [f"component_{i}" for i in range(int(getattr(selection, "n_components_", 0)))]

    return names or [f"feature_{i}" for i in range(_output_width(pipeline))]


def _output_width(pipeline: Pipeline) -> int:
    columns = pipeline.named_steps.get("columns")
    if columns is None:
        return 0
    widths: list[int] = []
    for _name, transformer, _cols in columns.transformers_:
        widths.append(int(getattr(transformer, "n_features_in_", 0)))
    return sum(widths)


__all__ = [
    "MAX_SELECTED_FEATURES",
    "DatetimeExpander",
    "OutlierClipper",
    "PreprocessorBuilder",
    "PreprocessorBundle",
    "SignedLog1p",
    "finalise_feature_names",
]
