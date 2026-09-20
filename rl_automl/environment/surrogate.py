"""Surrogate experiment outcomes (spec §30).

PPO needs thousands of episodes; real training gives you tens per hour. The surrogate is
what makes RL training tractable, and it is deliberately **two-tier**:

* :class:`AnalyticPrior` - a documented, hand-specified model of "which algorithm
  families do well on which kind of tabular problem", with explicit adjustments for size,
  width, cardinality, missingness, imbalance and preset strength. It is the floor: cheap,
  always available, and honest about being a prior rather than a measurement.
* :class:`SurrogateModel` - optional gradient-boosted regressors fitted on real
  experiment outcomes collected into a meta-dataset. When present it is blended with the
  prior.

**This is a simulator, not ground truth.** Results obtained against it are labelled as
simulated everywhere they surface, and the benchmark in ``evaluation.benchmark`` reports
real-environment numbers separately (spec §30).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_automl.core.logging import get_logger
from rl_automl.core.types import CostTier, TaskType
from rl_automl.core.vocabulary import PREPROCESSING_OPTIONS, PreprocessingOption

logger = get_logger("environment.surrogate")


@dataclass
class SurrogatePrediction:
    score: float
    cost_s: float
    failure_probability: float
    uncertainty: float = 0.0
    source: str = "prior"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 5),
            "cost_s": round(self.cost_s, 4),
            "failure_probability": round(self.failure_probability, 4),
            "uncertainty": round(self.uncertainty, 4),
            "source": self.source,
        }


#: Base expected score per (task type, model family), before dataset adjustments.
#: Grounded in the usual tabular results: boosted trees lead on medium/large heterogeneous
#: tabular data, linear models are a strong small-data baseline, kernel methods shine on
#: small dense problems and degrade as n grows.
FAMILY_BASE: dict[TaskType, dict[str, float]] = {
    TaskType.CLASSIFICATION: {
        "boosting": 0.900,
        "bagging": 0.872,
        "kernel": 0.860,
        "neural": 0.852,
        "linear": 0.815,
        "ensemble": 0.850,
        "probabilistic": 0.830,
        "centroid": 0.780,
        "other": 0.820,
    },
    TaskType.REGRESSION: {
        "boosting": 0.835,
        "bagging": 0.815,
        "neural": 0.790,
        "kernel": 0.775,
        "linear": 0.740,
        "other": 0.760,
    },
    TaskType.CLUSTERING: {
        "centroid": 0.520,
        "probabilistic": 0.510,
        "density": 0.480,
        "hierarchical": 0.495,
        "other": 0.470,
    },
    TaskType.ANOMALY_DETECTION: {
        "ensemble": 0.640,
        "kernel": 0.615,
        "neighborhood": 0.600,
        "other": 0.560,
    },
    TaskType.DIMENSIONALITY_REDUCTION: {
        "linear": 0.880,
        "other": 0.800,
    },
}

#: Relative training cost per family, in units of the configured cost reference.
FAMILY_COST: dict[str, float] = {
    "linear": 0.15,
    "centroid": 0.35,
    "bagging": 0.85,
    "boosting": 0.70,
    "ensemble": 0.60,
    "probabilistic": 0.70,
    "neural": 1.10,
    "kernel": 1.60,
    "density": 0.95,
    "hierarchical": 1.90,
    "neighborhood": 1.05,
    "other": 0.80,
}


class AnalyticPrior:
    """Hand-specified outcome model. Documented, deterministic, cheap."""

    def predict(
        self,
        *,
        task_type: TaskType,
        family: str,
        fingerprint: np.ndarray,
        preset_index: int,
        n_presets: int,
        preprocessing: tuple[str, ...] | list[str],
        feature_selection: str,
        preset_params: dict[str, Any] | None = None,
        preset_cost: CostTier | None = None,
        cost_reference_s: float = 60.0,
        n_rows: int = 1000,
        n_cols: int = 10,
    ) -> SurrogatePrediction:
        features = fingerprint_features(fingerprint)
        strength = _preset_strength(preset_index, n_presets, preset_cost)
        score = FAMILY_BASE.get(task_type, {}).get(
            family, FAMILY_BASE.get(task_type, {}).get("other", 0.6)
        )

        scale = _scale_factor(features)
        score += scale * _size_adjustment(task_type, family, features)
        score += _width_adjustment(task_type, family, features)
        score += _categorical_adjustment(task_type, family, features)
        score += _missing_adjustment(task_type, family, features)
        score += _imbalance_adjustment(task_type, family, features)
        score += _preset_adjustment(strength)
        score += _preprocessing_adjustment(family, preprocessing, feature_selection)
        score += _class_weight_adjustment(family, preset_params, features)
        score += _noise(task_type, family, fingerprint, preset_index, n_presets)

        # Clipped below 1.0 on purpose: a prior that can reach the ceiling stops
        # discriminating between candidates, which turns every ranking into a coin toss.
        score = float(np.clip(score, 0.01, 0.985))

        cost = (
            FAMILY_COST.get(family, 0.8)
            * float(np.clip(_rows_factor(features), 0.25, 12.0))
            * cost_reference_s
        )
        cost *= 1.0 + 0.35 * strength
        cost = float(max(cost, 0.01))

        failure = _failure_probability(task_type, family, features, strength)
        return SurrogatePrediction(
            score=score,
            cost_s=cost,
            failure_probability=failure,
            uncertainty=0.05 + 0.05 * (1.0 - scale),
            source="prior",
        )


class SurrogateModel:
    """Prior, optionally blended with regressors fitted on real outcomes."""

    def __init__(
        self,
        prior: AnalyticPrior | None = None,
        *,
        cost_reference_s: float = 60.0,
        blend: float = 0.7,
    ) -> None:
        self.prior = prior or AnalyticPrior()
        self.cost_reference_s = cost_reference_s
        self.blend = blend
        self._score_model: Any = None
        self._cost_model: Any = None
        self._failure_model: Any = None
        self._model_keys: tuple[str, ...] = ()
        self.n_samples = 0

    # -- fitting -----------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._score_model is not None

    def fit(self, rows: list[dict[str, Any]]) -> SurrogateModel:
        """Fit on meta-dataset rows collected from real runs.

        Each row needs: ``fingerprint``, ``task_type``, ``model_key``, ``preset_index``,
        ``preprocessing``, ``feature_selection``, ``score``, ``cost_s``, ``failed``.
        """
        if len(rows) < 50:
            logger.info(
                "not enough meta-dataset rows to fit a surrogate; keeping the analytic prior",
                extra={"context": {"n_rows": len(rows)}},
            )
            return self

        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError:  # pragma: no cover
            return self

        self._model_keys = tuple(sorted({str(row["model_key"]) for row in rows}))
        X = np.vstack([self._row_features(row) for row in rows])
        y_score = np.asarray([float(row["score"]) for row in rows])
        y_cost = np.log1p(np.asarray([float(row.get("cost_s", 1.0)) for row in rows]))
        y_failure = np.asarray([1.0 if row.get("failed") else 0.0 for row in rows])

        self._score_model = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(
            X, y_score
        )
        self._cost_model = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(
            X, y_cost
        )
        if 0 < y_failure.sum() < len(y_failure):
            from sklearn.ensemble import HistGradientBoostingClassifier

            self._failure_model = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(
                X, y_failure
            )

        self.n_samples = len(rows)
        logger.info(
            "surrogate fitted on meta-dataset",
            extra={"context": {"n_rows": len(rows), "n_models": len(self._model_keys)}},
        )
        return self

    # -- prediction --------------------------------------------------------------

    def predict(
        self,
        *,
        task_type: TaskType,
        model_key: str,
        family: str,
        fingerprint: np.ndarray,
        preset_index: int,
        n_presets: int,
        preprocessing: tuple[str, ...] | list[str],
        feature_selection: str,
        preset_params: dict[str, Any] | None = None,
        preset_cost: CostTier | None = None,
        n_rows: int = 1000,
        n_cols: int = 10,
    ) -> SurrogatePrediction:
        prior = self.prior.predict(
            task_type=task_type,
            family=family,
            fingerprint=fingerprint,
            preset_index=preset_index,
            n_presets=n_presets,
            preprocessing=preprocessing,
            feature_selection=feature_selection,
            preset_params=preset_params,
            preset_cost=preset_cost,
            cost_reference_s=self.cost_reference_s,
            n_rows=n_rows,
            n_cols=n_cols,
        )

        if not self.is_fitted or model_key not in self._model_keys:
            return prior

        row = {
            "task_type": task_type,
            "model_key": model_key,
            "preset_index": preset_index,
            "n_presets": n_presets,
            "preprocessing": preprocessing,
            "feature_selection": feature_selection,
            "fingerprint": fingerprint,
        }
        features = self._row_features(row).reshape(1, -1)
        fitted_score = float(self._score_model.predict(features)[0])
        fitted_cost = float(np.expm1(self._cost_model.predict(features)[0]))
        fitted_failure = (
            float(self._failure_model.predict_proba(features)[0][1])
            if self._failure_model is not None
            else prior.failure_probability
        )

        blended_score = (1 - self.blend) * prior.score + self.blend * np.clip(
            fitted_score, 0.0, 0.999
        )
        blended_cost = (1 - self.blend) * prior.cost_s + self.blend * max(fitted_cost, 0.01)

        return SurrogatePrediction(
            score=float(blended_score),
            cost_s=float(blended_cost),
            failure_probability=float(
                (1 - self.blend) * prior.failure_probability + self.blend * fitted_failure
            ),
            uncertainty=0.02,
            source="blended",
        )

    def _row_features(self, row: dict[str, Any]) -> np.ndarray:
        fingerprint = np.asarray(row["fingerprint"], dtype=np.float64).reshape(-1)
        task_type = row["task_type"]
        task_one_hot = np.zeros(len(TaskType), dtype=np.float64)
        task_one_hot[list(TaskType).index(task_type)] = 1.0

        model_one_hot = np.zeros(max(len(self._model_keys), 1), dtype=np.float64)
        if self._model_keys and str(row["model_key"]) in self._model_keys:
            model_one_hot[self._model_keys.index(str(row["model_key"]))] = 1.0

        preprocessing = set(row.get("preprocessing") or [])
        bits = np.asarray(
            [1.0 if option in preprocessing else 0.0 for option in PREPROCESSING_OPTIONS]
        )
        n_presets = max(int(row.get("n_presets", 1)), 1)
        preset = np.asarray([float(row.get("preset_index", 0)) / n_presets])
        feature_selection = str(row.get("feature_selection", "none"))
        return np.concatenate(
            [
                fingerprint,
                task_one_hot,
                model_one_hot,
                preset,
                bits,
                np.asarray([1.0 if feature_selection != "none" else 0.0]),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_fitted": self.is_fitted,
            "n_samples": self.n_samples,
            "model_keys": list(self._model_keys),
            "blend": self.blend,
        }


# --------------------------------------------------------------------------------------
# Prior internals
# --------------------------------------------------------------------------------------

#: Fingerprint indices, mirroring ``dataset.fingerprint.FINGERPRINT_FEATURES``.
_FP = {
    "log_rows": 0,
    "log_cols": 1,
    "log_rows_per_col": 2,
    "missing_ratio": 3,
    "frac_cols_with_missing": 4,
    "frac_numeric": 6,
    "frac_categorical": 7,
    "frac_datetime": 8,
    "frac_constant": 10,
    "frac_high_cardinality": 11,
    "mean_cardinality_ratio": 12,
    "max_cardinality_ratio": 13,
    "mean_abs_skew": 16,
    "max_abs_correlation": 19,
    "class_minority_ratio": 24,
    "class_imbalance_ratio": 25,
    "class_is_imbalanced": 27,
    "log_outlier_ratio": 30,
}


def fingerprint_features(fingerprint: np.ndarray | list[float]) -> dict[str, float]:
    """Pull the few fingerprint components the prior cares about."""
    values = np.asarray(fingerprint, dtype=np.float64).reshape(-1)
    out: dict[str, float] = {}
    for name, index in _FP.items():
        out[name] = float(values[index]) if index < len(values) else 0.0
    return out


def _scale_factor(features: dict[str, float]) -> float:
    """How far the dataset is from the "typical" tabular problem the base scores assume."""
    return 0.5 + 0.5 * (1.0 - abs(features.get("log_rows", 0.3) - 0.35) * 2.0)


def _rows_factor(features: dict[str, float]) -> float:
    return 0.35 + 3.5 * features.get("log_rows", 0.3)


def _size_adjustment(task_type: TaskType, family: str, features: dict[str, float]) -> float:
    """Small data favours low-variance learners; large data favours boosting."""
    log_rows = features.get("log_rows", 0.3)
    tiny = max(0.0, 0.45 - log_rows) / 0.45
    large = max(0.0, log_rows - 0.55) / 0.45

    if family in ("linear", "kernel"):
        return 0.05 * tiny - 0.06 * large
    if family in ("boosting",):
        return -0.05 * tiny + 0.04 * large
    if family == "neural":
        return -0.06 * tiny + 0.02 * large
    if family in ("bagging", "ensemble"):
        return 0.02 * tiny + 0.01 * large
    return 0.0


def _width_adjustment(task_type: TaskType, family: str, features: dict[str, float]) -> float:
    """Kernel and neural methods degrade when dimensions grow relative to samples."""
    rows_per_col = features.get("log_rows_per_col", 0.3)
    crowded = max(0.0, 0.55 - rows_per_col) / 0.55
    if family == "kernel":
        return -0.12 * crowded
    if family == "neural":
        return -0.07 * crowded
    if family == "linear":
        return -0.04 * crowded
    if family == "boosting":
        return 0.03 * crowded
    return 0.0


def _categorical_adjustment(task_type: TaskType, family: str, features: dict[str, float]) -> float:
    """High-cardinality categoricals punish one-hot-hungry learners."""
    pressure = features.get("frac_high_cardinality", 0.0) + features.get(
        "mean_cardinality_ratio", 0.0
    )
    if family in ("linear", "kernel"):
        return -0.10 * min(pressure, 1.0)
    if family == "neural":
        return -0.05 * min(pressure, 1.0)
    if family == "boosting":
        return 0.04 * min(pressure, 1.0)
    if family in ("bagging", "ensemble"):
        return 0.02 * min(pressure, 1.0)
    return 0.0


def _missing_adjustment(task_type: TaskType, family: str, features: dict[str, float]) -> float:
    missing = features.get("missing_ratio", 0.0)
    if missing <= 0.0:
        return 0.0
    if family == "boosting":
        return 0.02 * min(missing * 5, 1.0)  # handles NaN natively
    if family in ("linear", "kernel", "neural"):
        return -0.04 * min(missing * 5, 1.0)
    return 0.0


def _imbalance_adjustment(task_type: TaskType, family: str, features: dict[str, float]) -> float:
    if task_type is not TaskType.CLASSIFICATION:
        return 0.0
    imbalanced = features.get("class_is_imbalanced", 0.0)
    if not imbalanced:
        return 0.0
    if family == "linear":
        return -0.06
    if family == "kernel":
        return -0.03
    if family == "boosting":
        return 0.02
    return 0.0


#: Declared cost tiers, in the same 0..1 units the surrogate used to use for "strength".
_PRESET_STRENGTH: dict[CostTier, float] = {
    CostTier.LOW: 0.0,
    CostTier.MEDIUM: 0.5,
    CostTier.HIGH: 1.0,
}


def _preset_strength(preset_index: int, n_presets: int, preset_cost: CostTier | None) -> float:
    """How strong a preset is, taken from what it declares about itself.

    Presets are appended to the registry tables in the order they were written, not in
    order of capability -- ``class_balanced`` sits last for every classifier. Scoring by
    position therefore rewarded whichever preset happened to be appended last. The declared
    :class:`CostTier` is the honest signal, and position is only a fallback for callers that
    cannot resolve a spec.
    """
    if preset_cost is not None:
        return _PRESET_STRENGTH.get(preset_cost, 0.5)
    return _preset_rank(preset_index, n_presets)


def _preset_rank(preset_index: int, n_presets: int) -> float:
    """Positional fallback, used only when a preset's cost tier is unknown."""
    if n_presets <= 1:
        return 0.0
    return float(preset_index) / float(n_presets - 1)


def _preset_adjustment(strength: float) -> float:
    """Stronger presets help, with clearly diminishing returns.

    Kept small relative to the family base scores: preset strength is a second-order effect
    next to algorithm choice, and inflating it would make the agent chase presets instead of
    exploring families.
    """
    return 0.030 * np.sqrt(max(0.0, float(strength)))


def _preprocessing_adjustment(
    family: str, preprocessing: tuple[str, ...] | list[str], feature_selection: str
) -> float:
    """Reward preprocessing that genuinely helps this family, not toggle-stacking.

    The total is clipped, because eight simultaneously-enabled toggles should not beat a
    sensible two-toggle configuration.
    """
    selected = set(preprocessing)
    adjustment = 0.0

    if family in ("linear", "kernel", "neural"):
        if PreprocessingOption.SCALE_STANDARD.value in selected or (
            PreprocessingOption.SCALE_ROBUST.value in selected
        ):
            adjustment += 0.020
        else:
            adjustment -= 0.025
    elif (
        family in ("boosting", "bagging", "ensemble")
        and PreprocessingOption.SCALE_STANDARD.value in selected
    ):
        adjustment -= 0.008  # harmless but wasted work for trees

    if PreprocessingOption.CLIP_OUTLIERS.value in selected:
        adjustment += 0.007 if family in ("linear", "kernel", "neural") else 0.003
    if PreprocessingOption.LOG_TRANSFORM.value in selected:
        adjustment += 0.008
    if PreprocessingOption.MISSING_INDICATOR.value in selected:
        adjustment += 0.004
    if PreprocessingOption.DATETIME_FEATURES.value in selected:
        adjustment += 0.006

    if feature_selection != "none":
        # Helpful when there is plenty of data and many features, mildly risky otherwise.
        adjustment += 0.012

    return float(np.clip(adjustment, -0.05, 0.05))


def _class_weight_adjustment(
    family: str, preset_params: dict[str, Any] | None, features: dict[str, float]
) -> float:
    """Class weighting helps exactly when the classes are unbalanced.

    On balanced data it costs a little, because it distorts the decision threshold for no
    reason; on imbalanced data it is one of the cheapest real wins available.
    """
    if not preset_params:
        return 0.0

    weighted = any(key in preset_params for key in ("class_weight", "scale_pos_weight"))
    if not weighted:
        return 0.0

    if features.get("class_is_imbalanced", 0.0):
        return 0.025 if family in ("boosting", "bagging", "linear", "ensemble") else 0.015
    return -0.010


def _failure_probability(
    task_type: TaskType,
    family: str,
    features: dict[str, float],
    preset_strength: float,
) -> float:
    """Rough chance the pipeline blows up: memory blowouts and unworkable settings."""
    probability = 0.01

    log_rows = features.get("log_rows", 0.3)
    if family in ("kernel", "hierarchical", "neighborhood"):
        # Quadratic-ish memory in the number of samples.
        probability += 0.22 * max(0.0, log_rows - 0.60) / 0.40
    if family == "neural" and log_rows > 0.75:
        probability += 0.05

    dimensionality = features.get("log_cols", 0.2)
    if dimensionality > 0.5 and family in ("kernel", "neural"):
        probability += 0.08

    probability += 0.04 * float(preset_strength)
    return float(np.clip(probability, 0.0, 0.85))


def _noise(
    task_type: TaskType,
    family: str,
    fingerprint: np.ndarray | list[float],
    preset_index: int,
    n_presets: int,
) -> float:
    """Small deterministic jitter so the surrogate is not perfectly smooth.

    Deterministic on (dataset, task, family, preset) so repeated evaluations of the same
    experiment agree -- which matters for reproducible training and for the baselines
    being compared fairly.
    """
    digest = hashlib.sha256(
        f"{task_type.value}|{family}|{preset_index}|{n_presets}|"
        f"{np.round(fingerprint_features(fingerprint).get('log_rows', 0.0), 4)}".encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return float(rng.normal(0.0, 0.012))


__all__ = ["AnalyticPrior", "SurrogateModel", "SurrogatePrediction", "fingerprint_features"]
