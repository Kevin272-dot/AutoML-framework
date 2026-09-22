"""Dataset profiling (spec §4).

Produces the human-readable profile shown in the UI *and* the numeric inputs the
fingerprint condenses. Everything here is computed once per run and reused.

Cost control matters: correlations and outlier counts are computed on a bounded sample
for large frames, which keeps profiling O(1)-ish in practice without meaningfully
changing the fingerprint.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from rl_automl.core.config import DatasetConfig
from rl_automl.core.logging import get_logger
from rl_automl.core.types import (
    ClassDistribution,
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    FeatureCorrelation,
)

logger = get_logger("dataset.profiler")

#: Frames larger than this are sampled for correlation and outlier analysis.
SAMPLE_THRESHOLD = 200_000
CORRELATION_SAMPLE = 50_000
TOP_CORRELATIONS = 10
TOP_VALUES = 5
OUTLIER_SAMPLE = 50_000

_DATE_NAME_HINT = re.compile(
    r"(date|time|_at$|_dt$|timestamp|created|updated|day|month|year|quarter|week)",
    re.IGNORECASE,
)
_DATE_SHAPE = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}:\d{2}")


class DatasetProfiler:
    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()

    # -- public ------------------------------------------------------------------

    def profile(self, frame: pd.DataFrame, target: str | None = None) -> DatasetProfile:
        if frame is None or frame.empty:
            raise ValueError("cannot profile an empty dataframe")

        frame = frame.reset_index(drop=True)
        n_rows, n_cols = int(frame.shape[0]), int(frame.shape[1])
        warnings_found: list[str] = []

        feature_frame = (
            frame.drop(columns=[target]) if target and target in frame.columns else frame
        )

        kinds: dict[str, ColumnKind] = {}
        columns: list[ColumnProfile] = []

        numeric_names: list[str] = []
        categorical_names: list[str] = []
        datetime_names: list[str] = []
        other_names: list[str] = []

        cardinality: dict[str, int] = {}
        high_cardinality: list[str] = []
        constant_features: list[str] = []
        columns_with_missing: list[str] = []
        n_missing_total = 0

        for column in frame.columns:
            series = frame[column]
            kind = classify_column(series)
            column_name = str(column)
            kinds[column_name] = kind

            # The target is described by the class distribution, not by the feature
            # lists, so it is deliberately excluded from them.
            is_target = target is not None and column_name == str(target)
            if not is_target:
                (
                    numeric_names,
                    categorical_names,
                    datetime_names,
                    other_names,
                )[_KIND_INDEX[kind]].append(column_name)

            profile = self._profile_column(column_name, series, kind)
            columns.append(profile)

            cardinality[column_name] = profile.n_unique
            n_missing_total += profile.n_missing
            if profile.n_missing:
                columns_with_missing.append(column_name)
            if profile.is_constant:
                constant_features.append(column_name)
            if profile.is_high_cardinality:
                high_cardinality.append(column_name)

        n_duplicates = int(frame.duplicated().sum())
        total_cells = max(n_rows * n_cols, 1)

        class_distribution = None
        if target and target in frame.columns:
            class_distribution = self._class_distribution(frame[target], target)

        correlations, max_abs_corr = self._correlations(feature_frame, numeric_names)

        numeric_profiles = [c for c in columns if c.kind is ColumnKind.NUMERIC]
        skews = [abs(c.skew) for c in numeric_profiles if c.skew is not None]
        mean_abs_skew = float(np.mean(skews)) if skews else 0.0

        if n_missing_total == 0 and n_duplicates:
            warnings_found.append(
                f"{n_duplicates:,} duplicate rows detected; consider whether they are legitimate"
            )
        if class_distribution is not None and class_distribution.is_imbalanced:
            warnings_found.append(
                "class distribution is imbalanced "
                f"(minority class {class_distribution.minority_ratio:.1%}); "
                "prefer precision/recall/F1 or PR-AUC over accuracy"
            )
        if high_cardinality:
            warnings_found.append(
                f"{len(high_cardinality)} high-cardinality feature(s) detected; "
                "one-hot encoding may explode dimensionality"
            )
        if n_rows < self.config.min_rows:
            warnings_found.append(f"only {n_rows:,} rows available; results will be noisy")

        profile = DatasetProfile(
            n_rows=n_rows,
            n_cols=n_cols,
            numeric_features=numeric_names,
            categorical_features=categorical_names,
            datetime_features=datetime_names,
            other_features=other_names,
            n_missing_total=int(n_missing_total),
            missing_ratio=float(n_missing_total / total_cells),
            columns_with_missing=columns_with_missing,
            n_duplicate_rows=n_duplicates,
            duplicate_ratio=float(n_duplicates / max(n_rows, 1)),
            cardinality=cardinality,
            high_cardinality_features=high_cardinality,
            constant_features=constant_features,
            class_distribution=class_distribution,
            top_correlations=correlations,
            max_abs_correlation=max_abs_corr,
            mean_abs_skew=mean_abs_skew,
            dimensionality_ratio=float(n_cols / max(n_rows, 1)),
            size_bytes=int(frame.memory_usage(deep=True).sum()),
            memory_mb=float(frame.memory_usage(deep=True).sum() / 1024 / 1024),
            target=target,
            columns=columns,
            warnings=warnings_found,
        )
        logger.info(
            "dataset profiled",
            extra={
                "context": {
                    "rows": n_rows,
                    "cols": n_cols,
                    "numeric": len(numeric_names),
                    "categorical": len(categorical_names),
                    "missing_ratio": round(profile.missing_ratio, 4),
                }
            },
        )
        return profile

    # -- per column --------------------------------------------------------------

    def _profile_column(self, name: str, series: pd.Series, kind: ColumnKind) -> ColumnProfile:
        n_rows = len(series)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        cardinality_ratio = float(n_unique / max(n_rows, 1))

        profile = ColumnProfile(
            name=name,
            kind=kind,
            dtype=str(series.dtype),
            n_missing=n_missing,
            missing_ratio=float(n_missing / max(n_rows, 1)),
            n_unique=n_unique,
            cardinality_ratio=cardinality_ratio,
            is_constant=n_unique <= 1,
            is_high_cardinality=(
                kind is ColumnKind.CATEGORICAL
                and (
                    n_unique > self.config.max_cardinality_for_ohe
                    or cardinality_ratio >= self.config.high_cardinality_threshold
                )
            ),
        )

        if kind is ColumnKind.NUMERIC:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                profile.mean = _safe_float(numeric.mean())
                profile.std = _safe_float(numeric.std())
                profile.minimum = _safe_float(numeric.min())
                profile.maximum = _safe_float(numeric.max())
                profile.median = _safe_float(numeric.median())
                profile.skew = _safe_float(numeric.skew())
                profile.n_outliers = self._count_outliers(numeric)
        elif kind is ColumnKind.CATEGORICAL:
            counts = series.astype("string").value_counts().head(TOP_VALUES)
            profile.top_values = [
                {"value": str(value), "count": int(count), "ratio": float(count / max(n_rows, 1))}
                for value, count in counts.items()
            ]
        elif kind is ColumnKind.DATETIME:
            parsed = _to_datetime(series)
            valid = parsed.dropna()
            if not valid.empty:
                # Stored as epoch seconds so the range survives serialisation.
                profile.minimum = float(valid.min().timestamp())
                profile.maximum = float(valid.max().timestamp())

        return profile

    @staticmethod
    def _count_outliers(numeric: pd.Series) -> int:
        values = numeric.dropna()
        if values.empty:
            return 0
        if len(values) > OUTLIER_SAMPLE:
            values = values.sample(OUTLIER_SAMPLE, random_state=1234)
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            return 0
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return int(((values < lower) | (values > upper)).sum())

    # -- class distribution ------------------------------------------------------

    @staticmethod
    def _class_distribution(series: pd.Series, target: str) -> ClassDistribution:
        counts = series.value_counts(dropna=False)
        n_classes = int(counts.shape[0])
        total = int(counts.sum()) or 1
        proportions = (counts / total).astype(float)

        minority_ratio = float(proportions.min()) if n_classes else None
        imbalance_ratio = (
            float(proportions.max() / proportions.min())
            if n_classes > 1 and proportions.min() > 0
            else None
        )
        entropy = float(-(proportions * np.log(proportions + 1e-12)).sum())
        max_entropy = float(np.log(n_classes)) if n_classes > 1 else 1.0
        normalised_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        is_imbalanced = bool(
            n_classes > 1
            and minority_ratio is not None
            and (minority_ratio < 0.25 or (imbalance_ratio or 0.0) > 4.0)
        )

        return ClassDistribution(
            n_classes=n_classes,
            counts={str(k): int(v) for k, v in counts.items()},
            minority_ratio=minority_ratio,
            imbalance_ratio=imbalance_ratio,
            entropy=normalised_entropy,
            is_imbalanced=is_imbalanced,
        )

    # -- correlations ------------------------------------------------------------

    def _correlations(
        self, frame: pd.DataFrame, numeric_names: list[str]
    ) -> tuple[list[FeatureCorrelation], float]:
        usable = [c for c in numeric_names if c in frame.columns]
        if len(usable) < 2:
            return [], 0.0

        working = frame[usable]
        if len(working) > CORRELATION_SAMPLE:
            working = working.sample(CORRELATION_SAMPLE, random_state=1234)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = working.corr(numeric_only=True)
        if matrix.empty:
            return [], 0.0

        pairs: list[FeatureCorrelation] = []
        names = list(matrix.columns)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                value = matrix.at[left, right]
                if pd.isna(value):
                    continue
                pairs.append(
                    FeatureCorrelation(
                        feature_a=str(left), feature_b=str(right), correlation=float(value)
                    )
                )

        if not pairs:
            return [], 0.0
        pairs.sort(key=lambda pair: abs(pair.correlation), reverse=True)
        return pairs[:TOP_CORRELATIONS], float(abs(pairs[0].correlation))


# --------------------------------------------------------------------------------------
# Column typing helpers
# --------------------------------------------------------------------------------------

_KIND_INDEX = {
    ColumnKind.NUMERIC: 0,
    ColumnKind.CATEGORICAL: 1,
    ColumnKind.DATETIME: 2,
    ColumnKind.OTHER: 3,
}


def classify_column(series: pd.Series) -> ColumnKind:
    """Decide how a column should be treated downstream."""
    if pd.api.types.is_bool_dtype(series):
        return ColumnKind.CATEGORICAL
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnKind.DATETIME
    if pd.api.types.is_numeric_dtype(series):
        return ColumnKind.NUMERIC
    if isinstance(series.dtype, pd.CategoricalDtype):
        return ColumnKind.CATEGORICAL
    if _is_datetime_like(series):
        return ColumnKind.DATETIME
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return ColumnKind.CATEGORICAL
    return ColumnKind.OTHER


def _is_datetime_like(series: pd.Series) -> bool:
    """Heuristically recognise date columns stored as text.

    Requires both a high parse rate and date-shaped strings (or a suggestive column
    name), which prevents a column of integer-like strings being read as dates.
    """
    sample = series.dropna()
    if sample.empty:
        return False
    if len(sample) > 200:
        sample = sample.sample(200, random_state=1234)

    as_text = sample.astype("string")
    shaped = as_text.str.match(_DATE_SHAPE).fillna(False).mean()
    name_hint = bool(_DATE_NAME_HINT.search(str(series.name or "")))
    if shaped < 0.8 and not name_hint:
        return False

    parsed = _to_datetime(sample)
    return bool(parsed.notna().mean() >= 0.9)


def _to_datetime(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            return pd.to_datetime(series, errors="coerce")


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def profile_dataset(
    frame: pd.DataFrame, target: str | None = None, config: DatasetConfig | None = None
) -> DatasetProfile:
    return DatasetProfiler(config).profile(frame, target=target)


__all__ = ["SAMPLE_THRESHOLD", "DatasetProfiler", "classify_column", "profile_dataset"]
