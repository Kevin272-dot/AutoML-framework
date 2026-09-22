"""Dataset fingerprint (spec §4).

A fixed-shape, bounded numeric summary of a dataset. Two properties matter:

1. **Stable shape.** ``FINGERPRINT_FEATURES`` fixes the order; the RL state encoder and
   the similarity index both depend on that order, so a change bumps
   ``FINGERPRINT_VERSION`` and invalidates saved policies rather than silently
   corrupting them.
2. **Bounded.** Every component lands in roughly ``[0, 1]`` so the policy sees a
   well-conditioned input regardless of whether the dataset has 500 rows or 5 million.
"""

from __future__ import annotations

import math

from rl_automl.core.types import ColumnKind, DatasetFingerprint, DatasetProfile, TaskType

FINGERPRINT_VERSION = "1.0.0"

#: Ordered feature names. Append only; reordering requires a version bump.
FINGERPRINT_FEATURES: tuple[str, ...] = (
    "log_rows",
    "log_cols",
    "log_rows_per_col",
    "missing_ratio",
    "frac_cols_with_missing",
    "duplicate_ratio",
    "frac_numeric",
    "frac_categorical",
    "frac_datetime",
    "frac_other",
    "frac_constant",
    "frac_high_cardinality",
    "mean_cardinality_ratio",
    "max_cardinality_ratio",
    "log_mean_cardinality",
    "frac_binary_features",
    "mean_abs_skew",
    "max_abs_skew",
    "mean_abs_correlation",
    "max_abs_correlation",
    "frac_near_duplicate_features",
    "log_size_mb",
    "log_memory_per_row",
    "class_log_n_classes",
    "class_minority_ratio",
    "class_imbalance_ratio",
    "class_entropy",
    "class_is_imbalanced",
    "target_missing_ratio",
    "target_cardinality_ratio",
    "log_outlier_ratio",
    "has_target",
    "task_is_classification",
    "task_is_regression",
    "task_is_clustering",
    "task_is_anomaly",
    "task_is_dimred",
    "samples_per_feature",
    "frac_low_cardinality_categorical",
    "numeric_fraction_of_signal",
)

assert len(FINGERPRINT_FEATURES) == len(set(FINGERPRINT_FEATURES)), "duplicate fingerprint feature"


def _scale_log(value: float, saturation: float = 1e7) -> float:
    """Map a non-negative count onto [0, 1] with a log curve."""
    if value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(saturation))


def _ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not denominator:
        return default
    return float(numerator) / float(denominator)


def _clamp(value: float | None, low: float = 0.0, high: float = 1.0) -> float:
    if value is None:
        return low
    if not math.isfinite(value):
        return low
    return max(low, min(high, float(value)))


def build_fingerprint(
    profile: DatasetProfile, task_type: TaskType | None = None
) -> DatasetFingerprint:
    """Condense a :class:`DatasetProfile` into the RL-visible fingerprint."""
    values = fingerprint_values(profile, task_type)
    return DatasetFingerprint(
        schema_version=FINGERPRINT_VERSION,
        feature_names=list(FINGERPRINT_FEATURES),
        values=values,
        n_rows=profile.n_rows,
        n_cols=profile.n_cols,
    )


def fingerprint_values(profile: DatasetProfile, task_type: TaskType | None = None) -> list[float]:
    n_rows = max(profile.n_rows, 1)
    n_cols = max(profile.n_cols, 1)
    feature_cols = max(n_cols - (1 if profile.target else 0), 1)

    numeric = set(profile.numeric_features)
    categorical = set(profile.categorical_features)
    constant = set(profile.constant_features)

    by_name = {column.name: column for column in profile.columns}
    feature_columns = [column for column in profile.columns if column.name != profile.target]
    feature_names = [column.name for column in feature_columns]

    binary_features = sum(1 for column in feature_columns if column.n_unique == 2)
    low_cardinality_categorical = sum(
        1
        for column in feature_columns
        if column.kind is ColumnKind.CATEGORICAL and 2 <= column.n_unique <= 10
    )

    cardinalities = [column.n_unique for column in feature_columns] or [0]
    outlier_ratios = [
        _ratio(by_name[name].n_outliers, n_rows)
        for name in numeric
        if name in by_name and by_name[name].n_outliers
    ]

    target_column = by_name.get(profile.target) if profile.target else None
    distribution = profile.class_distribution

    raw: dict[str, float] = {
        "log_rows": _scale_log(profile.n_rows),
        "log_cols": _scale_log(profile.n_cols, saturation=5000),
        "log_rows_per_col": _scale_log(_ratio(n_rows, feature_cols), saturation=1e5),
        "missing_ratio": _clamp(profile.missing_ratio),
        "frac_cols_with_missing": _clamp(_ratio(len(profile.columns_with_missing), n_cols)),
        "duplicate_ratio": _clamp(profile.duplicate_ratio),
        "frac_numeric": _clamp(_ratio(len(numeric), n_cols)),
        "frac_categorical": _clamp(_ratio(len(categorical), n_cols)),
        "frac_datetime": _clamp(_ratio(len(profile.datetime_features), n_cols)),
        "frac_other": _clamp(_ratio(len(profile.other_features), n_cols)),
        "frac_constant": _clamp(_ratio(len(constant), n_cols)),
        "frac_high_cardinality": _clamp(_ratio(len(profile.high_cardinality_features), n_cols)),
        "mean_cardinality_ratio": _clamp(_ratio(sum(cardinalities), len(cardinalities) * n_rows)),
        "max_cardinality_ratio": _clamp(_ratio(max(cardinalities), n_rows)),
        "log_mean_cardinality": _scale_log(
            float(sum(cardinalities)) / len(cardinalities), saturation=1e5
        ),
        "frac_binary_features": _clamp(_ratio(binary_features, len(feature_names))),
        "mean_abs_skew": _clamp(profile.mean_abs_skew, 0.0, 10.0) / 10.0,
        "max_abs_skew": _clamp(
            max((abs(c.skew) for c in profile.columns if c.skew is not None), default=0.0),
            0.0,
            20.0,
        )
        / 20.0,
        "mean_abs_correlation": _clamp(
            sum(abs(pair.correlation) for pair in profile.top_correlations)
            / max(len(profile.top_correlations), 1)
        ),
        "max_abs_correlation": _clamp(profile.max_abs_correlation),
        "frac_near_duplicate_features": _clamp(
            _ratio(
                sum(1 for pair in profile.top_correlations if abs(pair.correlation) >= 0.98),
                max(len(profile.top_correlations), 1),
            )
        ),
        "log_size_mb": _scale_log(profile.memory_mb, saturation=4096),
        "log_memory_per_row": _scale_log(
            _ratio(profile.size_bytes, n_rows) / 1024, saturation=4096
        ),
        "class_log_n_classes": (
            _scale_log(distribution.n_classes, saturation=1000) if distribution else 0.0
        ),
        "class_minority_ratio": _clamp(distribution.minority_ratio if distribution else None),
        "class_imbalance_ratio": (
            _scale_log(distribution.imbalance_ratio or 0.0, saturation=1000)
            if distribution
            else 0.0
        ),
        "class_entropy": _clamp(distribution.entropy if distribution else None),
        "class_is_imbalanced": 1.0 if distribution and distribution.is_imbalanced else 0.0,
        "target_missing_ratio": _clamp(target_column.missing_ratio if target_column else None),
        "target_cardinality_ratio": _clamp(
            target_column.cardinality_ratio if target_column else None
        ),
        "log_outlier_ratio": _scale_log(
            (sum(outlier_ratios) / len(outlier_ratios) * 100) if outlier_ratios else 0.0,
            saturation=100,
        ),
        "has_target": 1.0 if profile.target else 0.0,
        "task_is_classification": 1.0 if task_type is TaskType.CLASSIFICATION else 0.0,
        "task_is_regression": 1.0 if task_type is TaskType.REGRESSION else 0.0,
        "task_is_clustering": 1.0 if task_type is TaskType.CLUSTERING else 0.0,
        "task_is_anomaly": 1.0 if task_type is TaskType.ANOMALY_DETECTION else 0.0,
        "task_is_dimred": 1.0 if task_type is TaskType.DIMENSIONALITY_REDUCTION else 0.0,
        "samples_per_feature": _scale_log(_ratio(n_rows, feature_cols), saturation=1e4),
        "frac_low_cardinality_categorical": _clamp(
            _ratio(low_cardinality_categorical, max(len(feature_names), 1))
        ),
        "numeric_fraction_of_signal": _clamp(_ratio(len(numeric), len(feature_names))),
    }

    missing = [name for name in FINGERPRINT_FEATURES if name not in raw]
    if missing:  # pragma: no cover - guards against a half-edited feature list
        raise KeyError(f"fingerprint features not computed: {missing}")
    return [raw[name] for name in FINGERPRINT_FEATURES]


def fingerprint_array(profile: DatasetProfile, task_type: TaskType | None = None):
    """Same values as a 1-D NumPy array, for the similarity index."""
    import numpy as np

    return np.asarray(fingerprint_values(profile, task_type), dtype=np.float64)


def describe_fingerprint(fingerprint: DatasetFingerprint) -> dict[str, float]:
    """Human-readable view, handy in logs and the results export."""
    return {
        name: round(value, 4)
        for name, value in zip(fingerprint.feature_names, fingerprint.values, strict=False)
    }


__all__ = [
    "FINGERPRINT_FEATURES",
    "FINGERPRINT_VERSION",
    "build_fingerprint",
    "describe_fingerprint",
    "fingerprint_array",
    "fingerprint_values",
]
