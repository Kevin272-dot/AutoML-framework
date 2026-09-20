"""Custom transformers that must survive outside this project.

These classes end up inside a fitted pipeline, and a fitted pipeline is pickled into the
model package. Pickle records a class by its dotted path, so unpickling needs the module
that defined it. Shipping this file inside the artifact is what makes that possible.

The rule this module exists to enforce: **it imports nothing from ``rl_automl``**, and it
depends only on numpy, pandas and scikit-learn -- the same libraries the package's
``requirements.txt`` already pins. Anything added here must keep that property, or the
artifact stops being loadable on a machine that has never installed AutoML.

The pickle stores the dotted name ``rl_automl.execution.runtime_transformers``; the
generated ``model_loader`` registers the shipped copy under that name at load time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DATETIME_FEATURE_SUFFIXES = ("year", "month", "day", "dayofweek", "is_weekend")


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Quantile clipping fitted on the training split."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: Any, y: Any = None) -> OutlierClipper:
        array = np.asarray(X, dtype=np.float64)
        with np.errstate(all="ignore"):
            self.lower_ = np.nanquantile(array, self.lower_quantile, axis=0)
            self.upper_ = np.nanquantile(array, self.upper_quantile, axis=0)
        self.n_features_in_ = array.shape[1] if array.ndim == 2 else 1
        return self

    def transform(self, X: Any) -> np.ndarray:
        array = np.asarray(X, dtype=np.float64)
        return np.clip(array, self.lower_, self.upper_)


class SignedLog1p(BaseEstimator, TransformerMixin):
    """Signed ``log1p``: compresses heavy tails while keeping sign and NaN."""

    def fit(self, X: Any, y: Any = None) -> SignedLog1p:
        array = np.asarray(X)
        self.n_features_in_ = array.shape[1] if array.ndim == 2 else 1
        return self

    def transform(self, X: Any) -> np.ndarray:
        array = np.asarray(X, dtype=np.float64)
        with np.errstate(all="ignore"):
            return np.sign(array) * np.log1p(np.abs(array))


class DatetimeExpander(BaseEstimator, TransformerMixin):
    """Expands datetime columns into calendar features.

    Emits NaN for unparseable values so a downstream imputer decides their fate; this
    transformer never silently invents a date.
    """

    def __init__(self, columns: tuple[str, ...] | list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, X: Any, y: Any = None) -> DatetimeExpander:
        frame = _as_frame(X)
        self.columns_ = list(self.columns) if self.columns else [str(c) for c in frame.columns]
        self.feature_names_out_ = [
            f"{column}_{suffix}" for column in self.columns_ for suffix in DATETIME_FEATURE_SUFFIXES
        ]
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = _as_frame(X)
        blocks: list[np.ndarray] = []
        for column in self.columns_:
            if column not in frame.columns:
                blocks.append(np.full((len(frame), len(DATETIME_FEATURE_SUFFIXES)), np.nan))
                continue
            with np.errstate(all="ignore"):
                parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed")
            blocks.append(
                np.column_stack(
                    [
                        parsed.dt.year.to_numpy(dtype=np.float64),
                        parsed.dt.month.to_numpy(dtype=np.float64),
                        parsed.dt.day.to_numpy(dtype=np.float64),
                        parsed.dt.dayofweek.to_numpy(dtype=np.float64),
                        parsed.dt.dayofweek.isin([5, 6]).to_numpy(dtype=np.float64),
                    ]
                )
            )
        if not blocks:
            return np.empty((len(frame), 0), dtype=np.float64)
        return np.hstack(blocks)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


__all__ = [
    "DATETIME_FEATURE_SUFFIXES",
    "DatetimeExpander",
    "OutlierClipper",
    "SignedLog1p",
]
