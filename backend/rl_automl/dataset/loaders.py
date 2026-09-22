"""Dataset sources: bundled offline data, synthetic generators, curated downloads, OpenML.

Four tiers, deliberately:

* **Bundled** -- small real datasets shipped with scikit-learn. No network, always
  available, used by the test suite and by smoke runs.
* **Synthetic** -- parameterised generators spanning sizes, widths, informative
  fractions, separation and imbalance. This is where most pretraining diversity comes
  from, since there are only a handful of bundled datasets.
* **Curated** -- a short, hard-coded list of openly-licensed public datasets, each pinned
  by SHA-256 and normalised to CSV on first download (see
  :mod:`rl_automl.dataset.remote`). This is the tier that gives the *measured* half of the
  benchmark something real to run on: mixed dtypes, genuine missing values and
  high-cardinality text, rather than the clean numeric matrices scikit-learn ships.
* **OpenML** -- opt-in, requires the ``openml`` extra and network access, cached under
  ``artifacts/meta/openml`` so a corpus is fetched once and reused offline.

The bundled and synthetic tiers never touch the network; the curated tier only does so on
first use, and never unless network access is explicitly allowed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl_automl.core.errors import ConfigError, DatasetValidationError
from rl_automl.core.logging import get_logger
from rl_automl.core.metrics import primary_metric_for
from rl_automl.core.types import TaskType
from rl_automl.dataset import remote

logger = get_logger("dataset.loaders")

TargetName = str


@dataclass(frozen=True)
class DatasetSource:
    """A named, loadable dataset plus the task metadata needed to use it."""

    name: str
    task_type: TaskType
    loader: Callable[[], pd.DataFrame]
    target: TargetName | None = None
    metric: str | None = None
    description: str = ""
    requires_network: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def effective_metric(self) -> str:
        return self.metric or primary_metric_for(self.task_type)

    def load(self) -> pd.DataFrame:
        frame = self.loader()
        if self.target and self.target not in frame.columns:
            raise DatasetValidationError(
                f"source '{self.name}' did not produce expected target column '{self.target}'",
                columns=list(frame.columns),
            )
        return frame

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_type": self.task_type.value,
            "target": self.target,
            "metric": self.effective_metric(),
            "description": self.description,
            "requires_network": self.requires_network,
            "tags": list(self.tags),
        }


# --------------------------------------------------------------------------------------
# Bundled scikit-learn datasets
# --------------------------------------------------------------------------------------


def _sklearn_frame(
    loader_name: str,
    *,
    target: str = "target",
    positive: str | None = None,
) -> pd.DataFrame:
    """Materialise a scikit-learn ``Bunch`` as a tidy DataFrame.

    Feature names are used verbatim so that problem statements mentioning domain terms
    ("mean radius", "worst concave points") can match against real columns.
    """
    import sklearn.datasets as sk_datasets

    bunch = getattr(sk_datasets, loader_name)()
    feature_names = [str(name) for name in bunch.feature_names]
    frame = pd.DataFrame(np.asarray(bunch.data, dtype=np.float64), columns=feature_names)

    target_values = np.asarray(bunch.target)
    frame[target] = target_values

    if positive is not None and hasattr(bunch, "target_names"):
        names = [str(n) for n in bunch.target_names]
        if positive in names:
            index = names.index(positive)
            frame[target] = (target_values == index).astype(int)
    return frame


def _load_iris() -> pd.DataFrame:
    return _sklearn_frame("load_iris", target="species")


def _load_breast_cancer() -> pd.DataFrame:
    return _sklearn_frame("load_breast_cancer", target="diagnosis", positive="malignant")


def _load_wine() -> pd.DataFrame:
    return _sklearn_frame("load_wine", target="cultivar")


def _load_digits() -> pd.DataFrame:
    return _sklearn_frame("load_digits", target="digit")


def _load_diabetes() -> pd.DataFrame:
    return _sklearn_frame("load_diabetes", target="disease_progression")


def _load_california_housing() -> pd.DataFrame:
    """Requires a one-time download; only used when network access is allowed."""
    from sklearn.datasets import fetch_california_housing

    bunch = fetch_california_housing()
    frame = pd.DataFrame(np.asarray(bunch.data, dtype=np.float64), columns=bunch.feature_names)
    frame["median_house_value"] = np.asarray(bunch.target, dtype=np.float64)
    return frame


BUNDLED_SOURCES: dict[str, DatasetSource] = {
    "iris": DatasetSource(
        name="iris",
        task_type=TaskType.CLASSIFICATION,
        loader=_load_iris,
        target="species",
        description="3-class flower measurements, 150 rows",
        tags=("tiny", "multiclass", "numeric"),
    ),
    "breast_cancer": DatasetSource(
        name="breast_cancer",
        task_type=TaskType.CLASSIFICATION,
        loader=_load_breast_cancer,
        target="diagnosis",
        description="Binary tumour classification, 569 rows x 30 features",
        tags=("binary", "numeric", "wide"),
    ),
    "wine": DatasetSource(
        name="wine",
        task_type=TaskType.CLASSIFICATION,
        loader=_load_wine,
        target="cultivar",
        description="3-class wine chemistry, 178 rows",
        tags=("tiny", "multiclass", "numeric"),
    ),
    "digits": DatasetSource(
        name="digits",
        task_type=TaskType.CLASSIFICATION,
        loader=_load_digits,
        target="digit",
        description="10-class handwritten digits, 1797 rows x 64 features",
        tags=("multiclass", "wide", "image-derived"),
    ),
    "diabetes": DatasetSource(
        name="diabetes",
        task_type=TaskType.REGRESSION,
        loader=_load_diabetes,
        target="disease_progression",
        description="Regression on diabetes progression, 442 rows",
        tags=("small", "numeric"),
    ),
    "california_housing": DatasetSource(
        name="california_housing",
        task_type=TaskType.REGRESSION,
        loader=_load_california_housing,
        target="median_house_value",
        description="Regression on housing prices, 20k rows (downloads once)",
        requires_network=True,
        tags=("medium", "numeric", "skewed-target"),
    ),
}


# --------------------------------------------------------------------------------------
# Synthetic generators
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticRecipe:
    """A parameterised call to a scikit-learn generator."""

    name: str
    task_type: TaskType
    generator: str
    params: dict[str, Any]
    target: str = "target"
    metric: str | None = None
    description: str = ""

    def build(self, seed: int = 0) -> pd.DataFrame:
        frame, _target = self.build_with_metadata(seed)
        return frame

    def build_with_metadata(self, seed: int = 0) -> tuple[pd.DataFrame, str | None]:
        import sklearn.datasets as sk_datasets

        rng = np.random.default_rng(seed)
        params = dict(self.params)
        params.setdefault("random_state", int(rng.integers(0, 2**31 - 1)))

        if self.generator == "make_classification":
            data, target = sk_datasets.make_classification(**params)
        elif self.generator == "make_regression":
            data, target = sk_datasets.make_regression(**params)
        elif self.generator == "make_blobs":
            data, target = sk_datasets.make_blobs(**params)
        elif self.generator == "make_moons":
            data, target = sk_datasets.make_moons(**params)
        else:  # pragma: no cover - guarded by construction
            raise ConfigError(f"unknown synthetic generator '{self.generator}'")

        n_features = data.shape[1]
        columns = [f"feature_{i:03d}" for i in range(n_features)]
        frame = pd.DataFrame(np.asarray(data, dtype=np.float64), columns=columns)

        if self.task_type.is_supervised:
            frame[self.target] = target
            return frame, self.target

        # Unsupervised: synthetic generators still produce a label, but it is not a
        # training target. Keep it as a clearly-marked evaluation hint only.
        frame["_group_hint"] = target
        return frame, None


def make_churn_frame(n_rows: int = 20_000, seed: int = 7) -> pd.DataFrame:
    """A churn-shaped table: mixed dtypes, missing values, imbalance and correlated features.

    The bundled scikit-learn datasets are all numeric matrices, so they exercise none of the
    interesting paths -- categorical encoding, datetime expansion, high-cardinality
    handling, class weighting. This generator exists to give the end-to-end demo and the
    integration tests a realistic mixed-type problem (spec §34).

    The target is genuinely predictable from ``contract_type``, ``tenure_months`` and
    ``support_calls``, with noise, so models should land clearly above chance without any
    of them being perfect. That makes the comparison table meaningful rather than a
    coin-flip.
    """
    rng = np.random.default_rng(seed)

    contract = rng.choice(
        ["month-to-month", "one_year", "two_year"], size=n_rows, p=[0.55, 0.25, 0.20]
    )
    payment = rng.choice(["card", "bank_transfer", "electronic_check", "mailed_check"], size=n_rows)
    internet = rng.choice(["fiber", "dsl", "none"], size=n_rows, p=[0.45, 0.4, 0.15])

    tenure = np.clip(rng.gamma(shape=2.0, scale=14.0, size=n_rows), 1, 72).round()
    monthly = np.clip(rng.normal(70, 28, n_rows), 18, 130).round(2)
    support_calls = rng.poisson(1.3, n_rows)
    age = np.clip(rng.normal(44, 15, n_rows), 18, 90).round()
    total = (tenure * monthly * rng.normal(1.0, 0.05, n_rows)).round(2)

    # Tuned so the demo carries a real minority class (~21% churn, comfortable margin below
    # the 25% imbalance threshold) with enough signal for the comparison table to separate
    # models. A demo at ~27% would look better on F1 but would leave the profiler's
    # imbalance path and every class-weighted preset unexercised.
    logit = (
        1.15 * (contract == "month-to-month")
        - 0.85 * (contract == "two_year")
        - 0.070 * tenure
        + 0.46 * support_calls
        + 0.018 * (monthly - 70)
        + 0.30 * (internet == "fiber")
        - 1.20
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(n_rows) < probability).astype(int)

    frame = pd.DataFrame(
        {
            "contract_type": pd.Categorical(contract),
            "payment_method": pd.Categorical(payment),
            "internet_service": pd.Categorical(internet),
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_calls": support_calls,
            "age": age,
            "signup_date": pd.to_datetime("2019-01-01")
            + pd.to_timedelta(rng.integers(0, 6 * 365, n_rows), unit="D"),
            "churn": churn,
        }
    )

    # ~6% missing at random, concentrated where it could plausibly happen.
    for column, fraction in (("total_charges", 0.11), ("monthly_charges", 0.04), ("age", 0.02)):
        mask = rng.random(n_rows) < fraction
        frame.loc[mask, column] = np.nan

    return frame


def _classification_recipes() -> list[SyntheticRecipe]:
    recipes = [
        SyntheticRecipe(
            "synth_clf_balanced_small",
            TaskType.CLASSIFICATION,
            "make_classification",
            {
                "n_samples": 800,
                "n_features": 12,
                "n_informative": 6,
                "n_redundant": 3,
                "n_classes": 2,
                "class_sep": 1.0,
                "flip_y": 0.03,
            },
            description="Small, balanced, well-separated binary problem",
        ),
        SyntheticRecipe(
            "synth_clf_imbalanced",
            TaskType.CLASSIFICATION,
            "make_classification",
            {
                "n_samples": 4000,
                "n_features": 20,
                "n_informative": 8,
                "n_redundant": 5,
                "n_classes": 2,
                "weights": [0.9, 0.1],
                "class_sep": 0.8,
                "flip_y": 0.05,
            },
            description="Imbalanced binary problem with overlapping classes",
        ),
        SyntheticRecipe(
            "synth_clf_multiclass_wide",
            TaskType.CLASSIFICATION,
            "make_classification",
            {
                "n_samples": 6000,
                "n_features": 60,
                "n_informative": 20,
                "n_redundant": 15,
                "n_classes": 4,
                "class_sep": 1.2,
            },
            description="Wide multiclass problem",
        ),
        SyntheticRecipe(
            "synth_clf_hard_boundary",
            TaskType.CLASSIFICATION,
            "make_classification",
            {
                "n_samples": 2500,
                "n_features": 25,
                "n_informative": 10,
                "n_redundant": 5,
                "n_classes": 2,
                "class_sep": 0.5,
                "flip_y": 0.15,
            },
            description="Weakly separable binary problem, noisy labels",
        ),
        SyntheticRecipe(
            "synth_clf_high_dimensional_sparse",
            TaskType.CLASSIFICATION,
            "make_classification",
            {
                "n_samples": 1500,
                "n_features": 120,
                "n_informative": 6,
                "n_redundant": 0,
                "n_classes": 3,
                "n_clusters_per_class": 1,
            },
            description="Many uninformative features, good for feature-selection arms",
            metric="f1",
        ),
        SyntheticRecipe(
            "synth_clf_moons",
            TaskType.CLASSIFICATION,
            "make_moons",
            {"n_samples": 1500, "noise": 0.25},
            description="Non-linear boundary where linear baselines struggle",
        ),
        SyntheticRecipe(
            "synth_clf_blobs",
            TaskType.CLASSIFICATION,
            "make_blobs",
            {"n_samples": 3000, "n_features": 10, "centers": 4, "cluster_std": 1.8},
            description="Well-separated clusters usable as classification",
        ),
    ]
    return recipes


def _regression_recipes() -> list[SyntheticRecipe]:
    return [
        SyntheticRecipe(
            "synth_reg_small",
            TaskType.REGRESSION,
            "make_regression",
            {
                "n_samples": 600,
                "n_features": 10,
                "n_informative": 5,
                "noise": 10.0,
            },
            description="Small linear problem",
        ),
        SyntheticRecipe(
            "synth_reg_noisy",
            TaskType.REGRESSION,
            "make_regression",
            {
                "n_samples": 5000,
                "n_features": 30,
                "n_informative": 12,
                "noise": 45.0,
            },
            description="Noisy mid-size regression",
        ),
        SyntheticRecipe(
            "synth_reg_high_dimensional",
            TaskType.REGRESSION,
            "make_regression",
            {
                "n_samples": 1200,
                "n_features": 150,
                "n_informative": 8,
                "noise": 20.0,
            },
            description="Wide regression, tests feature-selection arms",
        ),
        SyntheticRecipe(
            "synth_reg_nonlinear",
            TaskType.REGRESSION,
            "make_regression",
            {
                "n_samples": 3000,
                "n_features": 20,
                "n_informative": 10,
                "noise": 30.0,
                "effective_rank": 6,
            },
            description="Rank-deficient design matrix, tree-friendly",
        ),
    ]


def _unsupervised_recipes() -> list[SyntheticRecipe]:
    return [
        SyntheticRecipe(
            "synth_cluster_blobs",
            TaskType.CLUSTERING,
            "make_blobs",
            {"n_samples": 2000, "n_features": 8, "centers": 5, "cluster_std": 1.0},
            description="Five isotropic clusters",
        ),
        SyntheticRecipe(
            "synth_cluster_dense",
            TaskType.CLUSTERING,
            "make_blobs",
            {"n_samples": 4000, "n_features": 15, "centers": 8, "cluster_std": 2.5},
            description="Overlapping clusters, higher intrinsic dimension",
        ),
    ]


def _demo_sources() -> list[DatasetSource]:
    """Mixed-dtype generators, used by the demo and the integration tests."""

    def churn_small() -> pd.DataFrame:
        return make_churn_frame(n_rows=3_000, seed=7)

    def churn_full() -> pd.DataFrame:
        return make_churn_frame(n_rows=20_000, seed=7)

    return [
        DatasetSource(
            name="churn_demo",
            task_type=TaskType.CLASSIFICATION,
            loader=churn_full,
            target="churn",
            metric="f1",
            description=(
                "20k-row churn table with categorical, numeric and datetime columns, "
                "~6% missing values and a 22% minority class"
            ),
            tags=("synthetic", "mixed-dtype", "imbalanced", "demo"),
        ),
        DatasetSource(
            name="churn_demo_small",
            task_type=TaskType.CLASSIFICATION,
            loader=churn_small,
            target="churn",
            metric="f1",
            description="3k-row version of churn_demo, for fast tests",
            tags=("synthetic", "mixed-dtype", "demo", "small"),
        ),
    ]


# Registered alongside the scikit-learn datasets: these are local, offline generators with
# the same interface, so every code path that resolves a dataset by name works unchanged.
for _demo_source in _demo_sources():
    BUNDLED_SOURCES[_demo_source.name] = _demo_source


def synthetic_recipes(task_type: TaskType | None = None) -> list[SyntheticRecipe]:
    recipes = _classification_recipes() + _regression_recipes() + _unsupervised_recipes()
    if task_type is None:
        return recipes
    return [recipe for recipe in recipes if recipe.task_type is task_type]


def synthetic_source(recipe: SyntheticRecipe) -> DatasetSource:
    def loader() -> pd.DataFrame:
        return recipe.build(seed=0)

    return DatasetSource(
        name=recipe.name,
        task_type=recipe.task_type,
        loader=loader,
        target=recipe.target if recipe.task_type.is_supervised else None,
        metric=recipe.metric,
        description=recipe.description,
        tags=("synthetic",),
    )


# --------------------------------------------------------------------------------------
# OpenML (opt-in)
# --------------------------------------------------------------------------------------


OPENML_IDS: dict[str, dict[str, Any]] = {
    "openml_titanic": {
        "data_id": 40945,
        "task_type": TaskType.CLASSIFICATION,
        "target": "survived",
    },
    "openml_credit_g": {"data_id": 31, "task_type": TaskType.CLASSIFICATION, "target": "class"},
    "openml_diabetes_130": {"data_id": 37, "task_type": TaskType.CLASSIFICATION, "target": "class"},
    "openml_blood_transfusion": {
        "data_id": 1464,
        "task_type": TaskType.CLASSIFICATION,
        "target": "Class",
    },
    "openml_phoneme": {"data_id": 1489, "task_type": TaskType.CLASSIFICATION, "target": "Class"},
    "openml_pima": {"data_id": 37, "task_type": TaskType.CLASSIFICATION, "target": "class"},
    "openml_kin8nm": {"data_id": 189, "task_type": TaskType.REGRESSION, "target": "y"},
    "openml_cpu": {"data_id": 561, "task_type": TaskType.REGRESSION, "target": "class"},
}


def openml_source(name: str, cache_dir: str | Path | None = None) -> DatasetSource:
    """Build a source that fetches one OpenML dataset, caching it locally."""
    if name not in OPENML_IDS:
        raise ConfigError(f"unknown OpenML source '{name}'", known=sorted(OPENML_IDS))
    meta = OPENML_IDS[name]
    cache = Path(cache_dir) if cache_dir else Path("artifacts/meta/openml")
    cache_file = cache / f"{name}.parquet"

    def loader() -> pd.DataFrame:
        if cache_file.is_file():
            return pd.read_parquet(cache_file)
        try:
            from sklearn.datasets import fetch_openml
        except ImportError as exc:  # pragma: no cover - guarded by extra install
            raise ConfigError(
                "the 'openml' extra is required for OpenML sources: pip install rl-automl[openml]"
            ) from exc

        bunch = fetch_openml(data_id=meta["data_id"], as_frame=True, parser="auto")
        frame = bunch.frame.copy()
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            frame.to_parquet(cache_file, index=False)
        except Exception:  # pragma: no cover - optional pyarrow
            logger.warning("could not cache OpenML dataset", extra={"context": {"name": name}})
        return frame

    target = str(meta["target"])
    return DatasetSource(
        name=name,
        task_type=meta["task_type"],
        loader=loader,
        target=target,
        description=f"OpenML data_id={meta['data_id']} (cached after first fetch)",
        requires_network=True,
        tags=("openml", "real"),
    )


# --------------------------------------------------------------------------------------
# Public helpers
# --------------------------------------------------------------------------------------


def get_source(
    name: str, *, allow_network: bool = False, cache_dir: str | Path | None = None
) -> DatasetSource:
    """Resolve a source by name across all tiers."""
    if name in BUNDLED_SOURCES:
        return BUNDLED_SOURCES[name]
    if remote.is_remote(name):
        if not remote.is_cached(name, cache_dir) and not allow_network:
            raise ConfigError(
                f"'{name}' is a curated remote dataset and is not cached yet; "
                "pass allow_network=True to download it once",
                source=name,
            )
        return remote.remote_source(name, cache_dir)
    if name in OPENML_IDS:
        if not allow_network:
            raise ConfigError(
                f"'{name}' requires network access; pass allow_network=True or download it first",
                source=name,
            )
        return openml_source(name, cache_dir=cache_dir)
    for recipe in synthetic_recipes():
        if recipe.name == name:
            return synthetic_source(recipe)
    raise ConfigError(f"unknown dataset source '{name}'", known=sorted(list_sources_names()))


def list_sources_names(
    include_network: bool = False, cache_dir: str | Path | None = None
) -> list[str]:
    """Names usable right now; network-only sources appear when ``include_network``.

    Curated remote datasets are listed as soon as they are cached, because from that point
    on they are ordinary offline sources.
    """
    names = list(BUNDLED_SOURCES)
    names.extend(recipe.name for recipe in synthetic_recipes())
    names.extend(
        name for name in remote.remote_names() if include_network or remote.is_cached(name, cache_dir)
    )
    if include_network:
        names.extend(OPENML_IDS)
    return names


def list_sources(
    include_network: bool = False, cache_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    sources = [source.to_dict() for source in BUNDLED_SOURCES.values()]
    sources.extend(synthetic_source(recipe).to_dict() for recipe in synthetic_recipes())
    sources.extend(
        remote.remote_source(name, cache_dir).to_dict()
        for name in remote.remote_names()
        if include_network or remote.is_cached(name, cache_dir)
    )
    if include_network:
        sources.extend(openml_source(name).to_dict() for name in OPENML_IDS)
    return sources


def load_source(
    name: str, *, allow_network: bool = False, cache_dir: str | Path | None = None
) -> tuple[pd.DataFrame, DatasetSource]:
    source = get_source(name, allow_network=allow_network, cache_dir=cache_dir)
    if source.requires_network and not allow_network:
        raise ConfigError(f"source '{name}' requires network access")
    frame = source.load()
    logger.info(
        "dataset source loaded",
        extra={"context": {"source": name, "rows": int(frame.shape[0])}},
    )
    return frame, source


def save_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a frame to CSV or Parquet based on the suffix."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".parquet":
        frame.to_parquet(destination, index=False)
    else:
        frame.to_csv(destination, index=False)
    return destination


__all__ = [
    "BUNDLED_SOURCES",
    "OPENML_IDS",
    "DatasetSource",
    "SyntheticRecipe",
    "get_source",
    "list_sources",
    "list_sources_names",
    "load_source",
    "openml_source",
    "remote",
    "save_frame",
    "synthetic_recipes",
    "synthetic_source",
]
