"""Dataset splitting (spec §12).

Two rules drive this module:

* **Splits are built once per run and reused by every pipeline.** Model comparison is
  only fair if every model sees identical rows.
* **The test set is reserved for final evaluation.** The RL search and model selection
  operate on train/validation only. ``DataSplits`` carries the test indices so the
  executor can evaluate them after selection, but nothing in the search path reads them.

Unsupervised tasks do not get supervised train/val/test semantics bolted on. They get a
fit set and a held-out evaluation set, and their metrics are computed on the held-out
set using the appropriate protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from rl_automl.core.config import DatasetConfig
from rl_automl.core.errors import DatasetValidationError
from rl_automl.core.logging import get_logger
from rl_automl.core.types import TaskSpec, TaskType

logger = get_logger("execution.splits")


@dataclass
class DataSplits:
    """Index-based splits over one immutable frame.

    Frames are materialised lazily per split by :meth:`features` / :meth:`target`.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    frame: pd.DataFrame
    target: str | None = None
    strategy: str = "random"
    stratified: bool = False
    seed: int = 0
    test_reserved: bool = True
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- sizes -------------------------------------------------------------------

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        return len(self.val_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)

    # -- materialisation ---------------------------------------------------------

    def features(self, which: str) -> pd.DataFrame:
        indices = self._indices(which)
        frame = self.frame.iloc[indices]
        if self.target and self.target in frame.columns:
            return frame.drop(columns=[self.target])
        return frame

    def target_for(self, which: str) -> pd.Series | None:
        """The target column for a split. Named ``target_for`` because ``target`` is
        already the name of the target column itself."""
        if not self.target:
            return None
        indices = self._indices(which)
        return self.frame.iloc[indices][self.target]

    def label_hint(self, which: str) -> pd.Series | None:
        """Unsupervised evaluation labels, when the source supplied them."""
        if "_group_hint" not in self.frame.columns:
            return None
        return self.frame.iloc[self._indices(which)]["_group_hint"]

    def frame_for(self, which: str) -> pd.DataFrame:
        return self.frame.iloc[self._indices(which)]

    def _indices(self, which: str) -> np.ndarray:
        mapping = {
            "train": self.train_idx,
            "val": self.val_idx,
            "validation": self.val_idx,
            "test": self.test_idx,
            "fit": self.train_idx,
            "eval": self.val_idx,
        }
        if which not in mapping:
            raise KeyError(f"unknown split '{which}'; expected one of {sorted(mapping)}")
        index = mapping[which]
        if which == "test" and not self.test_reserved:
            raise DatasetValidationError("test split is not available for this task type")
        return index

    def summary(self) -> dict[str, Any]:
        total = self.n_train + self.n_val + self.n_test
        return {
            "strategy": self.strategy,
            "stratified": self.stratified,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "total": total,
            "seed": self.seed,
            "test_reserved": self.test_reserved,
            "warnings": list(self.warnings),
        }


def make_splits(
    frame: pd.DataFrame,
    task: TaskSpec,
    config: DatasetConfig | None = None,
    seed: int = 0,
) -> DataSplits:
    """Build the split for a task. Deterministic for a given ``seed``."""
    config = config or DatasetConfig()
    config.validate_ratios()

    n_rows = len(frame)
    if n_rows < config.min_rows:
        raise DatasetValidationError(
            f"dataset has {n_rows} rows but the configured minimum is {config.min_rows}",
            n_rows=n_rows,
            min_rows=config.min_rows,
        )

    warnings: list[str] = []
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_rows)

    if not task.task_type.is_supervised:
        return _unsupervised_splits(frame, task, config, seed, rng, all_idx)

    target_series = None
    if task.target and task.target in frame.columns:
        target_series = frame[task.target]
    else:
        raise DatasetValidationError(
            "supervised task requires a target column that exists in the dataset",
            target=task.target,
            columns=[str(c) for c in frame.columns][:50],
        )

    if target_series.isna().any():
        n_missing = int(target_series.isna().sum())
        warnings.append(f"dropped {n_missing} rows with a missing target value")
        all_idx = all_idx[~target_series.isna().to_numpy()]
        target_series = target_series.dropna()

    if len(all_idx) < config.min_rows:
        raise DatasetValidationError(
            f"only {len(all_idx)} rows remain after dropping missing targets; "
            f"minimum is {config.min_rows}"
        )

    # Chronological split when the data is a time series and the caller asked for it.
    if config.temporal_split:
        datetime_column = _first_datetime_column(frame, task)
        if datetime_column is not None:
            return _temporal_splits(frame, task, config, seed, all_idx, datetime_column, warnings)
        warnings.append(
            "temporal_split requested but no datetime column was found; "
            "falling back to a random split"
        )

    n_test = round(len(all_idx) * config.test_ratio)
    n_val = round(len(all_idx) * config.val_ratio)
    n_train = len(all_idx) - n_val - n_test

    if min(n_train, n_val, n_test) < 1:
        raise DatasetValidationError(
            f"dataset of {len(all_idx)} rows is too small for a "
            f"{config.train_ratio:.0%}/{config.val_ratio:.0%}/{config.test_ratio:.0%} split"
        )

    stratify = config.stratify and task.task_type is TaskType.CLASSIFICATION
    labels = target_series.to_numpy()

    if stratify:
        counts = pd.Series(labels).value_counts()
        # Each class needs at least one member in each of the three splits.
        if counts.min() < 3 or counts.shape[0] > 100:
            warnings.append(
                "stratification skipped: a class has fewer than 3 members or "
                "there are too many classes"
            )
            stratify = False

    if stratify:
        train_idx, val_idx, test_idx = _stratified_three_way(
            all_idx, labels, n_train, n_val, n_test, seed
        )
        strategy = "stratified"
    else:
        shuffled = rng.permutation(all_idx)
        train_idx = np.sort(shuffled[:n_train])
        val_idx = np.sort(shuffled[n_train : n_train + n_val])
        test_idx = np.sort(shuffled[n_train + n_val :])
        strategy = "random"

    splits = DataSplits(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        frame=frame,
        target=task.target,
        strategy=strategy,
        stratified=stratify,
        seed=seed,
        test_reserved=True,
        warnings=warnings,
        meta={
            "test_ratio": config.test_ratio,
            "val_ratio": config.val_ratio,
            "held_out_rows": len(test_idx),
        },
    )
    _log_splits(splits, task)
    return splits


def _stratified_three_way(
    all_idx: np.ndarray,
    labels: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratify with plain NumPy so the split does not depend on sklearn internals."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    total = len(all_idx)
    for label in np.unique(labels):
        class_idx = all_idx[labels == label]
        shuffled = rng.permutation(class_idx)
        share = len(shuffled) / total

        n_test_class = max(1, round(n_test * share))
        n_val_class = max(1, round(n_val * share))
        if n_test_class + n_val_class >= len(shuffled):
            n_test_class = max(1, min(n_test_class, len(shuffled) - 2))
            n_val_class = max(1, min(n_val_class, len(shuffled) - n_test_class - 1))

        test_parts.append(shuffled[:n_test_class])
        val_parts.append(shuffled[n_test_class : n_test_class + n_val_class])
        train_parts.append(shuffled[n_test_class + n_val_class :])

    return (
        np.sort(np.concatenate(train_parts)),
        np.sort(np.concatenate(val_parts)),
        np.sort(np.concatenate(test_parts)),
    )


def _temporal_splits(
    frame: pd.DataFrame,
    task: TaskSpec,
    config: DatasetConfig,
    seed: int,
    all_idx: np.ndarray,
    datetime_column: str,
    warnings: list[str],
) -> DataSplits:
    """Chronological train/val/test so the model is never trained on the future."""
    with np.errstate(all="ignore"):
        parsed = pd.to_datetime(frame[datetime_column], errors="coerce", format="mixed")
    order = np.argsort(parsed.to_numpy(), kind="stable")
    ordered = all_idx[order]
    ordered = ordered[~pd.isna(parsed.to_numpy()[order])]

    n_test = round(len(ordered) * config.test_ratio)
    n_val = round(len(ordered) * config.val_ratio)
    n_train = len(ordered) - n_val - n_test
    if min(n_train, n_val, n_test) < 1:
        raise DatasetValidationError(
            f"dataset of {len(ordered)} timed rows is too small to split chronologically"
        )

    warnings.append(
        f"chronological split on '{datetime_column}'; the test set is the most recent "
        f"{config.test_ratio:.0%} of rows, so it must not be used for tuning"
    )
    splits = DataSplits(
        train_idx=np.sort(ordered[:n_train]),
        val_idx=np.sort(ordered[n_train : n_train + n_val]),
        test_idx=np.sort(ordered[n_train + n_val :]),
        frame=frame,
        target=task.target,
        strategy=f"temporal({datetime_column})",
        stratified=False,
        seed=seed,
        test_reserved=True,
        warnings=warnings,
        meta={"datetime_column": datetime_column, "held_out_rows": n_test},
    )
    _log_splits(splits, task)
    return splits


def _unsupervised_splits(
    frame: pd.DataFrame,
    task: TaskSpec,
    config: DatasetConfig,
    seed: int,
    rng: np.random.Generator,
    all_idx: np.ndarray,
) -> DataSplits:
    """Fit/evaluate split for clustering, anomaly detection and dimensionality reduction.

    Metrics like silhouette are computed on data the model did not fit on, which is the
    closest unsupservised analogue of a held-out set. There is no "test" split because a
    second reserved set has no meaning without labels to score against.
    """
    n_eval = max(1, round(len(all_idx) * (config.val_ratio + config.test_ratio)))
    n_fit = len(all_idx) - n_eval
    if n_fit < 1:
        raise DatasetValidationError(
            f"dataset of {len(all_idx)} rows is too small for an unsupervised fit/eval split"
        )

    shuffled = rng.permutation(all_idx)
    splits = DataSplits(
        train_idx=np.sort(shuffled[:n_fit]),
        val_idx=np.sort(shuffled[n_fit:]),
        test_idx=np.array([], dtype=np.int64),
        frame=frame,
        target=None,
        strategy="fit_eval",
        stratified=False,
        seed=seed,
        test_reserved=False,
        warnings=[
            "unsupervised task: evaluation uses the held-out fit/eval protocol, "
            "not a supervised test set"
        ],
        meta={"topic": task.task_type.value, "eval_rows": n_eval},
    )
    _log_splits(splits, task)
    return splits


def _first_datetime_column(frame: pd.DataFrame, task: TaskSpec) -> str | None:
    from rl_automl.core.types import ColumnKind
    from rl_automl.dataset.profiler import classify_column

    for column in frame.columns:
        name = str(column)
        if name == task.target:
            continue
        if classify_column(frame[column]) is ColumnKind.DATETIME:
            return name
    return None


def _log_splits(splits: DataSplits, task: TaskSpec) -> None:
    logger.info(
        "splits created",
        extra={
            "context": {
                "task": task.task_type.value,
                "strategy": splits.strategy,
                **splits.summary(),
            }
        },
    )


__all__ = ["DataSplits", "make_splits"]
