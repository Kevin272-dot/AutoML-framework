"""Declarative model registry.

The registry is the single extension point of the system: adding an algorithm means
appending a :class:`ModelSpec` here. The RL action space, the executor, the evaluator
and the packaging layer all read from it, so no algorithm-specific branch exists in the
RL core (spec §8, §35).

Optional heavy dependencies are imported *inside* the builders, so importing the
registry never requires ``xgboost``, ``lightgbm`` or ``torch`` to be installed. The
executor surfaces a missing dependency as an isolated pipeline failure, not a crash.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rl_automl.core.errors import RegistryError
from rl_automl.core.types import CostTier, TaskType
from rl_automl.core.vocabulary import PreprocessingOption

Builder = Callable[[TaskType, dict[str, Any], int], Any]


@dataclass(frozen=True)
class Preset:
    """A named hyperparameter configuration. Presets keep the RL action space finite."""

    name: str
    params: dict[str, Any]
    cost: CostTier = CostTier.MEDIUM
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "cost": self.cost.value,
            "description": self.description,
        }


@functools.cache
def _dependency_installed(module_name: str) -> bool:
    """Cached availability probe.

    ``importlib.util.find_spec`` walks the import system and touches the filesystem, and the
    action-space masks call it once per model slot *per environment step*. During PPO
    pretraining that made it one of the hottest paths in the loop, for an answer that cannot
    change while the process runs -- a package does not appear half way through a training
    run. Caching it is therefore free of correctness risk.
    """
    return importlib.util.find_spec(module_name) is not None


@dataclass(frozen=True)
class ModelSpec:
    """Everything the rest of the system needs to know about an algorithm."""

    key: str
    display_name: str
    tasks: tuple[TaskType, ...]
    builder: Builder
    presets: tuple[Preset, ...]
    family: str
    cost: CostTier = CostTier.MEDIUM
    framework: str = "scikit-learn"
    #: Preprocessing the algorithm cannot work correctly without.
    required_preprocessing: tuple[str, ...] = ()
    #: Preprocessing that is meaningless for this algorithm and must be masked out.
    forbidden_preprocessing: tuple[str, ...] = ()
    supports_native_categorical: bool = False
    requires_dense_input: bool = False
    #: Import name that must be importable for this model to run, if not sklearn.
    optional_dependency: str | None = None
    supports_onnx: bool = False
    fit_requires_target: bool = True
    supports_predict_proba: bool = True
    notes: str = ""
    presets_by_task: dict[TaskType, tuple[Preset, ...]] = field(default_factory=dict)

    # -- queries -----------------------------------------------------------------

    def supports(self, task: TaskType) -> bool:
        return task in self.tasks

    def presets_for(self, task: TaskType) -> tuple[Preset, ...]:
        if task in self.presets_by_task:
            return self.presets_by_task[task]
        return self.presets

    def preset(self, index: int, task: TaskType | None = None) -> Preset:
        presets = self.presets if task is None else self.presets_for(task)
        if not presets:
            raise RegistryError(f"model '{self.key}' has no presets", model=self.key)
        if not 0 <= index < len(presets):
            raise RegistryError(
                f"preset index {index} out of range for '{self.key}'",
                model=self.key,
                n_presets=len(presets),
            )
        return presets[index]

    def n_presets(self, task: TaskType | None = None) -> int:
        presets = self.presets if task is None else self.presets_for(task)
        return len(presets)

    def is_available(self) -> tuple[bool, str | None]:
        """Whether the algorithm can actually be instantiated in this environment."""
        if self.optional_dependency is None:
            return True, None
        if not _dependency_installed(self.optional_dependency):
            return False, (
                f"optional dependency '{self.optional_dependency}' is not installed; "
                f"install with `pip install rl-automl[models]`"
            )
        return True, None

    def required_missing_handling(self) -> bool:
        """True when the algorithm cannot see NaN and so needs imputation."""
        return PreprocessingOption.IMPUTE_NUMERIC.value in self.required_preprocessing

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "family": self.family,
            "tasks": [task.value for task in self.tasks],
            "cost": self.cost.value,
            "framework": self.framework,
            "required_preprocessing": list(self.required_preprocessing),
            "forbidden_preprocessing": list(self.forbidden_preprocessing),
            "supports_native_categorical": self.supports_native_categorical,
            "requires_dense_input": self.requires_dense_input,
            "supports_onnx": self.supports_onnx,
            "fit_requires_target": self.fit_requires_target,
            "presets": [preset.to_dict() for preset in self.presets],
        }


class ModelRegistry:
    """Ordered registry. Iteration order defines the RL action-space layout."""

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec, replace: bool = False) -> ModelSpec:
        if spec.key in self._specs and not replace:
            raise RegistryError(f"model '{spec.key}' is already registered", model=spec.key)
        if not spec.presets and not spec.presets_by_task:
            raise RegistryError(f"model '{spec.key}' must declare presets", model=spec.key)
        self._specs[spec.key] = spec
        return spec

    def get(self, key: str) -> ModelSpec:
        try:
            return self._specs[key]
        except KeyError as exc:
            raise RegistryError(
                f"unknown model '{key}'", model=key, known=sorted(self._specs)
            ) from exc

    def has(self, key: str) -> bool:
        return key in self._specs

    def keys(self) -> list[str]:
        return list(self._specs)

    def all_specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def for_task(self, task: TaskType, enabled: list[str] | None = None) -> list[ModelSpec]:
        """Specs supporting ``task``, in registration order, optionally filtered."""
        allowed = set(enabled) if enabled else None
        return [
            spec
            for spec in self._specs.values()
            if spec.supports(task) and (allowed is None or spec.key in allowed)
        ]

    def build(
        self,
        key: str,
        task: TaskType,
        params: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> Any:
        spec = self.get(key)
        if not spec.supports(task):
            raise RegistryError(
                f"model '{key}' does not support task '{task.value}'",
                model=key,
                task=task.value,
            )
        return spec.builder(task, dict(params or {}), seed)

    def describe(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self._specs.values()]


# --------------------------------------------------------------------------------------
# Builders. Optional imports live inside so this module stays dependency-light.
# --------------------------------------------------------------------------------------


def _accepts_random_state(cls: Any) -> bool:
    """Whether an estimator's constructor takes ``random_state``.

    Not every sklearn estimator does -- ``AgglomerativeClustering``, ``DBSCAN``,
    ``OneClassSVM`` and ``LocalOutlierFactor`` all reject it -- so injection has to be
    signature-aware rather than unconditional.
    """
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover - C extension constructors
        return False
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return True
    return "random_state" in parameters


def _instantiate(cls: Any, params: dict[str, Any], seed: int) -> Any:
    kwargs = dict(params)
    if "random_state" not in kwargs and _accepts_random_state(cls):
        kwargs["random_state"] = seed
    return cls(**kwargs)


def _mlp_builder(task: TaskType, params: dict[str, Any], seed: int) -> Any:
    from rl_automl.execution.torch_models import TorchMLPClassifier, TorchMLPRegressor

    cls = TorchMLPClassifier if task is TaskType.CLASSIFICATION else TorchMLPRegressor
    return cls(random_state=seed, **params)


def _xgboost_builder(task: TaskType, params: dict[str, Any], seed: int) -> Any:
    if task is TaskType.CLASSIFICATION:
        from xgboost import XGBClassifier

        return XGBClassifier(random_state=seed, **params)
    from xgboost import XGBRegressor

    return XGBRegressor(random_state=seed, **params)


def _lightgbm_builder(task: TaskType, params: dict[str, Any], seed: int) -> Any:
    if task is TaskType.CLASSIFICATION:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=seed, verbose=-1, **params)
    from lightgbm import LGBMRegressor

    return LGBMRegressor(random_state=seed, verbose=-1, **params)


def _sklearn_builder(class_name: str, module: str = "sklearn.ensemble") -> Builder:
    def build(task: TaskType, params: dict[str, Any], seed: int) -> Any:
        import importlib

        cls = getattr(importlib.import_module(module), class_name)
        return _instantiate(cls, params, seed)

    return build


def _sklearn_task_builder(
    classification_class: str,
    regression_class: str,
    module: str = "sklearn.ensemble",
) -> Builder:
    def build(task: TaskType, params: dict[str, Any], seed: int) -> Any:
        import importlib

        name = classification_class if task is TaskType.CLASSIFICATION else regression_class
        cls = getattr(importlib.import_module(module), name)
        return _instantiate(cls, params, seed)

    return build


def _unsupervised_builder(class_name: str, module: str) -> Builder:
    def build(task: TaskType, params: dict[str, Any], seed: int) -> Any:
        import importlib

        cls = getattr(importlib.import_module(module), class_name)
        return _instantiate(cls, params, seed)

    return build


# --------------------------------------------------------------------------------------
# Preset tables
# --------------------------------------------------------------------------------------

_DENSE_IMPUTE = (PreprocessingOption.IMPUTE_NUMERIC.value,)
_TREE_REQUIRED = (PreprocessingOption.IMPUTE_NUMERIC.value,)

_RF_PRESETS = (
    Preset(
        "fast",
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
        CostTier.LOW,
        "Shallow ensemble, quick baseline",
    ),
    Preset(
        "balanced",
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
        CostTier.MEDIUM,
        "Default workhorse configuration",
    ),
    Preset(
        "regularized",
        {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 4},
        CostTier.MEDIUM,
        "Depth-limited to reduce variance",
    ),
    Preset(
        "deep",
        {"n_estimators": 900, "max_depth": 24, "min_samples_leaf": 1},
        CostTier.HIGH,
        "Large ensemble, highest ceiling",
    ),
)

_LINEAR_PRESETS = (
    Preset("default", {"max_iter": 2000}, CostTier.LOW, "L2 penalty, unit regularization"),
    Preset("strong_reg", {"C": 0.1, "max_iter": 3000}, CostTier.LOW, "Heavy L2 shrinkage"),
    Preset("weak_reg", {"C": 10.0, "max_iter": 3000}, CostTier.LOW, "Light L2 shrinkage"),
    Preset(
        "l1",
        {"C": 1.0, "penalty": "l1", "solver": "liblinear", "max_iter": 5000},
        CostTier.LOW,
        "Sparse solution, implicit feature selection",
    ),
    Preset(
        "balanced",
        {"C": 1.0, "class_weight": "balanced", "max_iter": 3000},
        CostTier.LOW,
        "Reweights the minority class; useful when the target is imbalanced",
    ),
)

_MLP_PRESETS = (
    Preset(
        "small",
        {
            "hidden_sizes": (64,),
            "learning_rate": 1e-3,
            "max_epochs": 60,
            "batch_size": 256,
            "dropout": 0.0,
        },
        CostTier.LOW,
        "Single small hidden layer",
    ),
    Preset(
        "medium",
        {
            "hidden_sizes": (128, 64),
            "learning_rate": 1e-3,
            "max_epochs": 80,
            "batch_size": 256,
            "dropout": 0.1,
        },
        CostTier.MEDIUM,
        "Two-layer MLP",
    ),
    Preset(
        "deep",
        {
            "hidden_sizes": (256, 128, 64),
            "learning_rate": 5e-4,
            "max_epochs": 100,
            "batch_size": 256,
            "dropout": 0.2,
        },
        CostTier.MEDIUM,
        "Deeper MLP with dropout",
    ),
    Preset(
        "wide",
        {
            "hidden_sizes": (512, 256),
            "learning_rate": 5e-4,
            "max_epochs": 120,
            "batch_size": 512,
            "dropout": 0.3,
        },
        CostTier.HIGH,
        "Wide and regularized",
    ),
)

_SVM_PRESETS = (
    Preset(
        "rbf_small",
        {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        CostTier.MEDIUM,
        "RBF kernel, default regularization",
    ),
    Preset(
        "rbf_wide",
        {"C": 10.0, "kernel": "rbf", "gamma": "auto"},
        CostTier.MEDIUM,
        "Wider RBF, less regularization",
    ),
    Preset("linear", {"C": 1.0, "kernel": "linear"}, CostTier.LOW, "Linear kernel"),
)

_SVR_PRESETS = (
    Preset("rbf", {"C": 1.0, "kernel": "rbf", "epsilon": 0.1}, CostTier.MEDIUM, "RBF kernel"),
    Preset("linear", {"C": 1.0, "kernel": "linear", "epsilon": 0.1}, CostTier.LOW, "Linear kernel"),
)

_KM_PRESETS = (
    Preset("k3", {"n_clusters": 3, "n_init": 10}, CostTier.LOW, "Three clusters"),
    Preset("k5", {"n_clusters": 5, "n_init": 10}, CostTier.LOW, "Five clusters"),
    Preset("k8", {"n_clusters": 8, "n_init": 10}, CostTier.LOW, "Eight clusters"),
    Preset("k12", {"n_clusters": 12, "n_init": 10}, CostTier.MEDIUM, "Twelve clusters"),
)

_DBSCAN_PRESETS = (
    Preset("tight", {"eps": 0.3, "min_samples": 5}, CostTier.MEDIUM, "Tight neighbourhood"),
    Preset("medium", {"eps": 0.5, "min_samples": 5}, CostTier.MEDIUM, "Medium neighbourhood"),
    Preset("loose", {"eps": 0.8, "min_samples": 10}, CostTier.MEDIUM, "Loose neighbourhood"),
)

_AGGLO_PRESETS = (
    Preset("ward5", {"n_clusters": 5, "linkage": "ward"}, CostTier.MEDIUM, "Ward linkage"),
    Preset("avg5", {"n_clusters": 5, "linkage": "average"}, CostTier.HIGH, "Average linkage"),
    Preset(
        "complete8", {"n_clusters": 8, "linkage": "complete"}, CostTier.HIGH, "Complete linkage"
    ),
)

_GMM_PRESETS = (
    Preset(
        "g3", {"n_components": 3, "covariance_type": "full"}, CostTier.MEDIUM, "Three components"
    ),
    Preset(
        "g5", {"n_components": 5, "covariance_type": "full"}, CostTier.MEDIUM, "Five components"
    ),
    Preset(
        "g8diag",
        {"n_components": 8, "covariance_type": "diag"},
        CostTier.MEDIUM,
        "Eight diagonal components",
    ),
)

_IFOREST_PRESETS = (
    Preset(
        "default", {"n_estimators": 200, "contamination": "auto"}, CostTier.LOW, "Default forest"
    ),
    Preset(
        "deep",
        {"n_estimators": 500, "max_samples": 512, "contamination": "auto"},
        CostTier.MEDIUM,
        "Larger forest",
    ),
    Preset(
        "sensitive",
        {"n_estimators": 300, "contamination": 0.1},
        CostTier.MEDIUM,
        "Assume 10% anomalies",
    ),
)

_OCSVM_PRESETS = (
    Preset("rbf", {"kernel": "rbf", "nu": 0.1, "gamma": "scale"}, CostTier.MEDIUM, "RBF kernel"),
    Preset("linear", {"kernel": "linear", "nu": 0.1}, CostTier.MEDIUM, "Linear kernel"),
)

_LOF_PRESETS = (
    Preset(
        "k20",
        {"n_neighbors": 20, "contamination": "auto", "novelty": True},
        CostTier.MEDIUM,
        "20 neighbours",
    ),
    Preset(
        "k35",
        {"n_neighbors": 35, "contamination": "auto", "novelty": True},
        CostTier.HIGH,
        "35 neighbours",
    ),
)

_PCA_PRESETS = (
    Preset("var95", {"n_components": 0.95}, CostTier.LOW, "Retain 95% variance"),
    Preset("var99", {"n_components": 0.99}, CostTier.LOW, "Retain 99% variance"),
    Preset("k10", {"n_components": 10}, CostTier.LOW, "Ten components"),
)


_XGB_PRESETS = (
    Preset(
        "fast",
        {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 5},
        CostTier.LOW,
        "Fewer, shallower trees",
    ),
    Preset(
        "balanced",
        {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        CostTier.MEDIUM,
        "Standard boosted configuration",
    ),
    Preset(
        "regularized",
        {
            "n_estimators": 700,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_lambda": 2.0,
        },
        CostTier.MEDIUM,
        "Heavier regularization for small or noisy data",
    ),
    Preset(
        "deep",
        {
            "n_estimators": 1200,
            "learning_rate": 0.02,
            "max_depth": 9,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        CostTier.HIGH,
        "Slow learning rate, many trees",
    ),
)

_LGBM_PRESETS = (
    Preset(
        "fast",
        {"n_estimators": 250, "learning_rate": 0.08, "num_leaves": 31},
        CostTier.LOW,
        "Quick boosted baseline",
    ),
    Preset(
        "balanced",
        {
            "n_estimators": 600,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        CostTier.MEDIUM,
        "Standard LightGBM configuration",
    ),
    Preset(
        "wide",
        {
            "n_estimators": 700,
            "learning_rate": 0.03,
            "num_leaves": 96,
            "min_child_samples": 20,
            "subsample": 0.85,
        },
        CostTier.MEDIUM,
        "Wide trees for high-cardinality interactions",
    ),
    Preset(
        "regularized",
        {
            "n_estimators": 900,
            "learning_rate": 0.02,
            "num_leaves": 48,
            "min_child_samples": 40,
            "reg_lambda": 3.0,
            "subsample": 0.8,
        },
        CostTier.HIGH,
        "Strong regularization, robust on noisy data",
    ),
)


_LINEAR_PRESETS = (
    Preset("default", {"max_iter": 2000}, CostTier.LOW, "L2 penalty, unit regularization"),
    Preset("strong_reg", {"C": 0.1, "max_iter": 3000}, CostTier.LOW, "Heavy L2 shrinkage"),
    Preset("weak_reg", {"C": 10.0, "max_iter": 3000}, CostTier.LOW, "Light L2 shrinkage"),
    Preset(
        "l1",
        {"C": 1.0, "penalty": "l1", "solver": "liblinear", "max_iter": 5000},
        CostTier.LOW,
        "Sparse solution, implicit feature selection",
    ),
    Preset(
        "balanced",
        {"C": 1.0, "class_weight": "balanced", "max_iter": 3000},
        CostTier.LOW,
        "Reweights the minority class; useful when the target is imbalanced",
    ),
)

_MLP_PRESETS = (
    Preset(
        "small",
        {
            "hidden_sizes": (64,),
            "learning_rate": 1e-3,
            "max_epochs": 60,
            "batch_size": 256,
            "dropout": 0.0,
        },
        CostTier.LOW,
        "Single small hidden layer",
    ),
    Preset(
        "medium",
        {
            "hidden_sizes": (128, 64),
            "learning_rate": 1e-3,
            "max_epochs": 80,
            "batch_size": 256,
            "dropout": 0.1,
        },
        CostTier.MEDIUM,
        "Two-layer MLP",
    ),
    Preset(
        "deep",
        {
            "hidden_sizes": (256, 128, 64),
            "learning_rate": 5e-4,
            "max_epochs": 100,
            "batch_size": 256,
            "dropout": 0.2,
        },
        CostTier.MEDIUM,
        "Deeper MLP with dropout",
    ),
    Preset(
        "wide",
        {
            "hidden_sizes": (512, 256),
            "learning_rate": 5e-4,
            "max_epochs": 120,
            "batch_size": 512,
            "dropout": 0.3,
        },
        CostTier.HIGH,
        "Wide and regularized",
    ),
)

_SVM_PRESETS = (
    Preset(
        "rbf_small",
        {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        CostTier.MEDIUM,
        "RBF kernel, default regularization",
    ),
    Preset(
        "rbf_wide",
        {"C": 10.0, "kernel": "rbf", "gamma": "auto"},
        CostTier.MEDIUM,
        "Wider RBF, less regularization",
    ),
    Preset("linear", {"C": 1.0, "kernel": "linear"}, CostTier.LOW, "Linear kernel"),
)

_SVR_PRESETS = (
    Preset("rbf", {"C": 1.0, "kernel": "rbf", "epsilon": 0.1}, CostTier.MEDIUM, "RBF kernel"),
    Preset("linear", {"C": 1.0, "kernel": "linear", "epsilon": 0.1}, CostTier.LOW, "Linear kernel"),
)

_KM_PRESETS = (
    Preset("k3", {"n_clusters": 3, "n_init": 10}, CostTier.LOW, "Three clusters"),
    Preset("k5", {"n_clusters": 5, "n_init": 10}, CostTier.LOW, "Five clusters"),
    Preset("k8", {"n_clusters": 8, "n_init": 10}, CostTier.LOW, "Eight clusters"),
    Preset("k12", {"n_clusters": 12, "n_init": 10}, CostTier.MEDIUM, "Twelve clusters"),
)

_DBSCAN_PRESETS = (
    Preset("tight", {"eps": 0.3, "min_samples": 5}, CostTier.MEDIUM, "Tight neighbourhood"),
    Preset("medium", {"eps": 0.5, "min_samples": 5}, CostTier.MEDIUM, "Medium neighbourhood"),
    Preset("loose", {"eps": 0.8, "min_samples": 10}, CostTier.MEDIUM, "Loose neighbourhood"),
)

_AGGLO_PRESETS = (
    Preset("ward5", {"n_clusters": 5, "linkage": "ward"}, CostTier.MEDIUM, "Ward linkage"),
    Preset("avg5", {"n_clusters": 5, "linkage": "average"}, CostTier.HIGH, "Average linkage"),
    Preset(
        "complete8", {"n_clusters": 8, "linkage": "complete"}, CostTier.HIGH, "Complete linkage"
    ),
)

_GMM_PRESETS = (
    Preset(
        "g3", {"n_components": 3, "covariance_type": "full"}, CostTier.MEDIUM, "Three components"
    ),
    Preset(
        "g5", {"n_components": 5, "covariance_type": "full"}, CostTier.MEDIUM, "Five components"
    ),
    Preset(
        "g8diag",
        {"n_components": 8, "covariance_type": "diag"},
        CostTier.MEDIUM,
        "Eight diagonal components",
    ),
)

_IFOREST_PRESETS = (
    Preset(
        "default", {"n_estimators": 200, "contamination": "auto"}, CostTier.LOW, "Default forest"
    ),
    Preset(
        "deep",
        {"n_estimators": 500, "max_samples": 512, "contamination": "auto"},
        CostTier.MEDIUM,
        "Larger forest",
    ),
    Preset(
        "sensitive",
        {"n_estimators": 300, "contamination": 0.1},
        CostTier.MEDIUM,
        "Assume 10% anomalies",
    ),
)

_OCSVM_PRESETS = (
    Preset("rbf", {"kernel": "rbf", "nu": 0.1, "gamma": "scale"}, CostTier.MEDIUM, "RBF kernel"),
    Preset("linear", {"kernel": "linear", "nu": 0.1}, CostTier.MEDIUM, "Linear kernel"),
)

_LOF_PRESETS = (
    Preset(
        "k20",
        {"n_neighbors": 20, "contamination": "auto", "novelty": True},
        CostTier.MEDIUM,
        "20 neighbours",
    ),
    Preset(
        "k35",
        {"n_neighbors": 35, "contamination": "auto", "novelty": True},
        CostTier.HIGH,
        "35 neighbours",
    ),
)

_PCA_PRESETS = (
    Preset("var95", {"n_components": 0.95}, CostTier.LOW, "Retain 95% variance"),
    Preset("var99", {"n_components": 0.99}, CostTier.LOW, "Retain 99% variance"),
    Preset("k10", {"n_components": 10}, CostTier.LOW, "Ten components"),
)


_XGB_PRESETS = (
    Preset(
        "fast",
        {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 5},
        CostTier.LOW,
        "Fewer, shallower trees",
    ),
    Preset(
        "balanced",
        {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        CostTier.MEDIUM,
        "Standard boosted configuration",
    ),
    Preset(
        "regularized",
        {
            "n_estimators": 700,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_lambda": 2.0,
        },
        CostTier.MEDIUM,
        "Heavier regularization for small or noisy data",
    ),
    Preset(
        "deep",
        {
            "n_estimators": 1200,
            "learning_rate": 0.02,
            "max_depth": 9,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        CostTier.HIGH,
        "Slow learning rate, many trees",
    ),
)

_LGBM_PRESETS = (
    Preset(
        "fast",
        {"n_estimators": 250, "learning_rate": 0.08, "num_leaves": 31},
        CostTier.LOW,
        "Quick boosted baseline",
    ),
    Preset(
        "balanced",
        {
            "n_estimators": 600,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        CostTier.MEDIUM,
        "Standard LightGBM configuration",
    ),
    Preset(
        "wide",
        {
            "n_estimators": 700,
            "learning_rate": 0.03,
            "num_leaves": 96,
            "min_child_samples": 20,
            "subsample": 0.85,
        },
        CostTier.MEDIUM,
        "Wide trees for high-cardinality interactions",
    ),
    Preset(
        "regularized",
        {
            "n_estimators": 900,
            "learning_rate": 0.02,
            "num_leaves": 48,
            "min_child_samples": 40,
            "reg_lambda": 3.0,
            "subsample": 0.8,
        },
        CostTier.HIGH,
        "Strong regularization, robust on noisy data",
    ),
)


#: Class-weighted variants. Kept separate from the regression tables because regressors
#: reject ``class_weight`` outright -- offering it there would produce a pipeline that
#: fails at fit time for no reason.
_CLASS_BALANCED_RF = Preset(
    "class_balanced",
    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 1,
        "class_weight": "balanced_subsample",
    },
    CostTier.MEDIUM,
    "Reweights the minority class; useful when the target is imbalanced",
)

_RF_PRESETS_CLF = (*_RF_PRESETS, _CLASS_BALANCED_RF)
_XGB_PRESETS_CLF = (
    *_XGB_PRESETS,
    Preset(
        "class_balanced",
        {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 4.0,
        },
        CostTier.MEDIUM,
        "Upweights the positive class for imbalanced binary problems",
    ),
)
_LGBM_PRESETS_CLF = (
    *_LGBM_PRESETS,
    Preset(
        "class_balanced",
        {
            "n_estimators": 600,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
        },
        CostTier.MEDIUM,
        "Reweights the minority class; useful when the target is imbalanced",
    ),
)


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------

_CLS = (TaskType.CLASSIFICATION,)
_REG = (TaskType.REGRESSION,)
_CLS_REG = (TaskType.CLASSIFICATION, TaskType.REGRESSION)


def _build_default_registry() -> ModelRegistry:
    registry = ModelRegistry()

    registry.register(
        ModelSpec(
            key="logistic_regression",
            display_name="Logistic Regression",
            tasks=_CLS,
            builder=_sklearn_builder("LogisticRegression", "sklearn.linear_model"),
            presets=_LINEAR_PRESETS,
            family="linear",
            cost=CostTier.LOW,
            required_preprocessing=(
                PreprocessingOption.IMPUTE_NUMERIC.value,
                PreprocessingOption.SCALE_STANDARD.value,
                PreprocessingOption.ENCODE_CATEGORICAL.value,
            ),
            supports_onnx=True,
            notes="Strong, fast baseline; needs scaled inputs and encoded categoricals.",
        )
    )

    registry.register(
        ModelSpec(
            key="linear_regression",
            display_name="Linear Regression",
            tasks=_REG,
            builder=_sklearn_builder("LinearRegression", "sklearn.linear_model"),
            presets=(Preset("default", {}, CostTier.LOW, "Ordinary least squares"),),
            family="linear",
            cost=CostTier.LOW,
            required_preprocessing=(
                PreprocessingOption.IMPUTE_NUMERIC.value,
                PreprocessingOption.ENCODE_CATEGORICAL.value,
            ),
            supports_predict_proba=False,
            supports_onnx=True,
            notes="Reference baseline for regression.",
        )
    )

    registry.register(
        ModelSpec(
            key="random_forest",
            display_name="Random Forest",
            tasks=_CLS_REG,
            builder=_sklearn_task_builder(
                "RandomForestClassifier", "RandomForestRegressor", "sklearn.ensemble"
            ),
            presets=_RF_PRESETS,
            presets_by_task={TaskType.CLASSIFICATION: _RF_PRESETS_CLF},
            family="bagging",
            cost=CostTier.MEDIUM,
            required_preprocessing=_TREE_REQUIRED,
            notes="Robust default; handles mixed scales without standardization.",
        )
    )

    registry.register(
        ModelSpec(
            key="extra_trees",
            display_name="Extra Trees",
            tasks=_CLS_REG,
            builder=_sklearn_task_builder(
                "ExtraTreesClassifier", "ExtraTreesRegressor", "sklearn.ensemble"
            ),
            presets=_RF_PRESETS,
            presets_by_task={TaskType.CLASSIFICATION: _RF_PRESETS_CLF},
            family="bagging",
            cost=CostTier.MEDIUM,
            required_preprocessing=_TREE_REQUIRED,
            notes="Cheaper than Random Forest with a similar ceiling.",
        )
    )

    registry.register(
        ModelSpec(
            key="xgboost",
            display_name="XGBoost",
            tasks=_CLS_REG,
            builder=_xgboost_builder,
            presets=_XGB_PRESETS,
            presets_by_task={TaskType.CLASSIFICATION: _XGB_PRESETS_CLF},
            family="boosting",
            cost=CostTier.MEDIUM,
            framework="xgboost",
            required_preprocessing=_TREE_REQUIRED,
            supports_onnx=True,
            optional_dependency="xgboost",
            notes="Strong on tabular data; needs absent categories handled by encoding.",
        )
    )

    registry.register(
        ModelSpec(
            key="lightgbm",
            display_name="LightGBM",
            tasks=_CLS_REG,
            builder=_lightgbm_builder,
            presets=_LGBM_PRESETS,
            presets_by_task={TaskType.CLASSIFICATION: _LGBM_PRESETS_CLF},
            family="boosting",
            cost=CostTier.MEDIUM,
            framework="lightgbm",
            required_preprocessing=_TREE_REQUIRED,
            supports_native_categorical=True,
            supports_onnx=True,
            optional_dependency="lightgbm",
            notes="Fast on wide data; can consume categorical features natively.",
        )
    )

    registry.register(
        ModelSpec(
            key="mlp",
            display_name="Neural Network (MLP)",
            tasks=_CLS_REG,
            builder=_mlp_builder,
            presets=_MLP_PRESETS,
            family="neural",
            cost=CostTier.MEDIUM,
            framework="torch",
            required_preprocessing=(
                PreprocessingOption.IMPUTE_NUMERIC.value,
                PreprocessingOption.SCALE_STANDARD.value,
                PreprocessingOption.ENCODE_CATEGORICAL.value,
            ),
            optional_dependency="torch",
            supports_onnx=True,
            notes="Learns interactions linear models miss; needs scaled inputs.",
        )
    )

    registry.register(
        ModelSpec(
            key="svm",
            display_name="Support Vector Machine",
            tasks=_CLS,
            builder=_sklearn_builder("SVC", "sklearn.svm"),
            presets=_SVM_PRESETS,
            family="kernel",
            cost=CostTier.HIGH,
            required_preprocessing=(
                PreprocessingOption.IMPUTE_NUMERIC.value,
                PreprocessingOption.SCALE_STANDARD.value,
                PreprocessingOption.ENCODE_CATEGORICAL.value,
            ),
            requires_dense_input=True,
            notes="Quadratic-ish training cost; strong on small, well-scaled data.",
        )
    )

    registry.register(
        ModelSpec(
            key="svr",
            display_name="Support Vector Regression",
            tasks=_REG,
            builder=_sklearn_builder("SVR", "sklearn.svm"),
            presets=_SVR_PRESETS,
            family="kernel",
            cost=CostTier.HIGH,
            required_preprocessing=(
                PreprocessingOption.IMPUTE_NUMERIC.value,
                PreprocessingOption.SCALE_STANDARD.value,
                PreprocessingOption.ENCODE_CATEGORICAL.value,
            ),
            requires_dense_input=True,
            supports_predict_proba=False,
            notes="Does not scale past a few tens of thousands of rows.",
        )
    )

    registry.register(
        ModelSpec(
            key="kmeans",
            display_name="KMeans",
            tasks=(TaskType.CLUSTERING,),
            builder=_unsupervised_builder("KMeans", "sklearn.cluster"),
            presets=_KM_PRESETS,
            family="centroid",
            cost=CostTier.LOW,
            required_preprocessing=(
                PreprocessingOption.IMPUTE_NUMERIC.value,
                PreprocessingOption.SCALE_STANDARD.value,
                PreprocessingOption.ENCODE_CATEGORICAL.value,
            ),
            fit_requires_target=False,
            supports_predict_proba=False,
            notes="Centroid clustering; k is part of the action space.",
        )
    )

    registry.register(
        ModelSpec(
            key="dbscan",
            display_name="DBSCAN",
            tasks=(TaskType.CLUSTERING,),
            builder=_unsupervised_builder("DBSCAN", "sklearn.cluster"),
            presets=_DBSCAN_PRESETS,
            family="density",
            cost=CostTier.MEDIUM,
            required_preprocessing=(
                PreprocessingOption.IMPUTE_NUMERIC.value,
                PreprocessingOption.SCALE_STANDARD.value,
                PreprocessingOption.ENCODE_CATEGORICAL.value,
            ),
            fit_requires_target=False,
            supports_predict_proba=False,
            notes="Density-based; no predict() for new points, evaluation uses fit labels.",
        )
    )

    registry.register(
        ModelSpec(
            key="agglomerative",
            display_name="Agglomerative Clustering",
            tasks=(TaskType.CLUSTERING,),
            builder=_unsupervised_builder("AgglomerativeClustering", "sklearn.cluster"),
            presets=_AGGLO_PRESETS,
            family="hierarchical",
            cost=CostTier.HIGH,
            required_preprocessing=(
                PreprocessingOption.IMPUTE_NUMERIC.value,
                PreprocessingOption.SCALE_STANDARD.value,
                PreprocessingOption.ENCODE_CATEGORICAL.value,
            ),
            fit_requires_target=False,
            supports_predict_proba=False,
            notes="Hierarchical; quadratic memory in the number of samples.",
        )
    )

    registry.register(
        ModelSpec(
            key="gaussian_mixture",
            display_name="Gaussian Mixture",
            tasks=(TaskType.CLUSTERING,),
            builder=_unsupervised_builder("GaussianMixture", "sklearn.mixture"),
            presets=_GMM_PRESETS,
            family="probabilistic",
            cost=CostTier.MEDIUM,
            required_preprocessing=(
                PreprocessingOption.IMPUTE_NUMERIC.value,
                PreprocessingOption.SCALE_STANDARD.value,
                PreprocessingOption.ENCODE_CATEGORICAL.value,
            ),
            fit_requires_target=False,
            notes="Soft assignment; exposes predict_proba style responsibilities.",
        )
    )

    registry.register(
        ModelSpec(
            key="isolation_forest",
            display_name="Isolation Forest",
            tasks=(TaskType.ANOMALY_DETECTION,),
            builder=_unsupervised_builder("IsolationForest", "sklearn.ensemble"),
            presets=_IFOREST_PRESETS,
            family="ensemble",
            cost=CostTier.LOW,
            required_preprocessing=(
                PreprocessingOption.IMPUTE_NUMERIC.value,
                PreprocessingOption.ENCODE_CATEGORICAL.value,
            ),
            fit_requires_target=False,
            supports_predict_proba=False,
            notes="Tree-based anomaly detector; robust default for tabular outliers.",
        )
    )

    registry.register(
        ModelSpec(
            key="one_class_svm",
            display_name="One-Class SVM",
            tasks=(TaskType.ANOMALY_DETECTION,),
            builder=_unsupervised_builder("OneClassSVM", "sklearn.svm"),
            presets=_OCSVM_PRESETS,
            family="kernel",
            cost=CostTier.HIGH,
            required_preprocessing=(
                PreprocessingOption.IMPUTE_NUMERIC.value,
                PreprocessingOption.SCALE_STANDARD.value,
                PreprocessingOption.ENCODE_CATEGORICAL.value,
            ),
            requires_dense_input=True,
            fit_requires_target=False,
            supports_predict_proba=False,
            notes="Boundary method; needs scaling and does not scale to large n.",
        )
    )

    registry.register(
        ModelSpec(
            key="local_outlier_factor",
            display_name="Local Outlier Factor",
            tasks=(TaskType.ANOMALY_DETECTION,),
            builder=_unsupervised_builder("LocalOutlierFactor", "sklearn.neighbors"),
            presets=_LOF_PRESETS,
            family="neighborhood",
            cost=CostTier.MEDIUM,
            required_preprocessing=(
                PreprocessingOption.IMPUTE_NUMERIC.value,
                PreprocessingOption.SCALE_STANDARD.value,
                PreprocessingOption.ENCODE_CATEGORICAL.value,
            ),
            fit_requires_target=False,
            supports_predict_proba=False,
            notes="Local density; novelty=True enables scoring new points.",
        )
    )

    registry.register(
        ModelSpec(
            key="pca_transform",
            display_name="PCA",
            tasks=(TaskType.DIMENSIONALITY_REDUCTION,),
            builder=_unsupervised_builder("PCA", "sklearn.decomposition"),
            presets=_PCA_PRESETS,
            family="linear",
            cost=CostTier.LOW,
            required_preprocessing=(
                PreprocessingOption.IMPUTE_NUMERIC.value,
                PreprocessingOption.SCALE_STANDARD.value,
            ),
            fit_requires_target=False,
            supports_predict_proba=False,
            notes="Linear projection used both as a standalone reducer and a pipeline stage.",
        )
    )

    return registry


REGISTRY: ModelRegistry = _build_default_registry()


def get_registry() -> ModelRegistry:
    return REGISTRY


def registered_keys() -> list[str]:
    return REGISTRY.keys()


def get_spec(key: str) -> ModelSpec:
    return REGISTRY.get(key)


def specs_for_task(task: TaskType, enabled: list[str] | None = None) -> list[ModelSpec]:
    return REGISTRY.for_task(task, enabled)


__all__ = [
    "REGISTRY",
    "ModelRegistry",
    "ModelSpec",
    "Preset",
    "get_registry",
    "get_spec",
    "registered_keys",
    "specs_for_task",
]
