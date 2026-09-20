"""Shared vocabulary for preprocessing, feature selection and export formats.

Lives in ``core`` because both the registry (``search``) and the transformer
implementations (``execution``) need the same names, and neither should depend on the
other. Adding an option here makes it visible to the RL action space automatically.
"""

from __future__ import annotations

from enum import Enum


class PreprocessingOption(str, Enum):
    """Toggles the RL agent may switch on for a pipeline."""

    IMPUTE_NUMERIC = "impute_numeric"
    IMPUTE_CATEGORICAL = "impute_categorical"
    MISSING_INDICATOR = "missing_indicator"
    SCALE_STANDARD = "scale_standard"
    SCALE_ROBUST = "scale_robust"
    ENCODE_CATEGORICAL = "encode_categorical"
    CLIP_OUTLIERS = "clip_outliers"
    LOG_TRANSFORM = "log_transform"
    DATETIME_FEATURES = "datetime_features"


class FeatureSelectionOption(str, Enum):
    NONE = "none"
    VARIANCE = "variance"
    MUTUAL_INFO = "mutual_info"
    KBEST = "kbest"
    PCA = "pca"


class ExportFormat(str, Enum):
    JOBLIB = "joblib"
    ONNX = "onnx"


PREPROCESSING_OPTIONS: tuple[str, ...] = tuple(option.value for option in PreprocessingOption)
FEATURE_SELECTION_OPTIONS: tuple[str, ...] = tuple(
    option.value for option in FeatureSelectionOption
)

#: Options that address missing values. A model that cannot tolerate NaN should be
#: paired with at least one of these; the registry declares the requirement and the
#: action space masks illegal combinations.
MISSING_VALUE_OPTIONS: frozenset[str] = frozenset(
    {PreprocessingOption.IMPUTE_NUMERIC.value, PreprocessingOption.IMPUTE_CATEGORICAL.value}
)

#: Feature selection strategies that require the target and therefore cannot be used
#: for unsupervised tasks.
SUPERVISED_ONLY_FEATURE_SELECTION: frozenset[str] = frozenset(
    {FeatureSelectionOption.MUTUAL_INFO.value, FeatureSelectionOption.KBEST.value}
)


def is_valid_preprocessing_option(name: str) -> bool:
    return name in PREPROCESSING_OPTIONS


def is_valid_feature_selection(name: str) -> bool:
    return name in FEATURE_SELECTION_OPTIONS


__all__ = [
    "FEATURE_SELECTION_OPTIONS",
    "MISSING_VALUE_OPTIONS",
    "PREPROCESSING_OPTIONS",
    "SUPERVISED_ONLY_FEATURE_SELECTION",
    "ExportFormat",
    "FeatureSelectionOption",
    "PreprocessingOption",
    "is_valid_feature_selection",
    "is_valid_preprocessing_option",
]
