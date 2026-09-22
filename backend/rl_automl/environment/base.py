"""The environment interface shared by the real and simulated AutoML environments.

Kept in its own module so that ``simulator`` and the PPO agent can depend on the
interface without dragging in ``execution`` (and therefore torch, xgboost, lightgbm).
That is what lets PPO training run on a machine that only has the core install.

The real environment (:class:`~rl_automl.environment.automl_env.AutoMLEnv`) and the
surrogate (:class:`~rl_automl.environment.simulator.SimulatorEnv`) both satisfy this
contract, which is why the baselines can be compared against the RL planner under
identical budgeting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rl_automl.core.metrics import orient
from rl_automl.core.types import ExperimentResult, ExperimentSpec, TaskType
from rl_automl.environment.action_space import ActionMaskTable, ActionSpaceLayout
from rl_automl.environment.reward import RewardBreakdown, novelty_score
from rl_automl.environment.state_encoder import SearchProgress


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False


@dataclass
class SearchHistoryEntry:
    """One attempted experiment, as recorded by the environment."""

    step: int
    model: str
    preset_index: int
    preprocessing: tuple[str, ...]
    feature_selection: str
    score: float | None
    oriented_score: float | None
    cost_s: float
    reward: float
    failed: bool
    duplicate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model": self.model,
            "preset_index": self.preset_index,
            "preprocessing": list(self.preprocessing),
            "feature_selection": self.feature_selection,
            "score": self.score,
            "oriented_score": self.oriented_score,
            "cost_s": round(self.cost_s, 4),
            "reward": round(self.reward, 4),
            "failed": self.failed,
            "duplicate": self.duplicate,
        }


class AutoMLEnvironment(ABC):
    """Minimal contract: reset, step, and enough introspection for masking and logging."""

    def __init__(self, task_type: TaskType, max_experiments: int) -> None:
        self.task_type = task_type
        self.max_experiments = max_experiments

    # -- required ----------------------------------------------------------------

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Length of the observation vector."""

    @property
    @abstractmethod
    def layout(self) -> ActionSpaceLayout:
        """The action-space layout, including which model each index refers to."""

    @property
    @abstractmethod
    def mask_table(self) -> ActionMaskTable:
        """Per-slot action masks, indexed by the sampled model slot."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Start a new search episode and return the initial observation."""

    @abstractmethod
    def step(self, action: np.ndarray | tuple[int, ...]) -> StepResult:
        """Apply one action: run (or simulate) one experiment and return the transition."""

    # -- optional ----------------------------------------------------------------

    @property
    def history(self) -> list[SearchHistoryEntry]:
        return getattr(self, "_history", [])

    def observation(self) -> np.ndarray:
        """Current observation without advancing the episode."""
        if not hasattr(self, "_state"):
            raise RuntimeError("reset() must be called before observation()")
        return self._state

    def action_masks(self) -> ActionMaskTable:
        return self.mask_table

    # -- shared bookkeeping ------------------------------------------------------

    @property
    def n_experiments(self) -> int:
        return len(self.history)

    @property
    def best_score(self) -> float | None:
        scores = [
            entry.oriented_score for entry in self.history if entry.oriented_score is not None
        ]
        return max(scores) if scores else None

    @property
    def best_entry(self) -> SearchHistoryEntry | None:
        candidates = [entry for entry in self.history if entry.oriented_score is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda entry: entry.oriented_score or float("-inf"))

    def record(self, entry: SearchHistoryEntry) -> None:
        self._history.append(entry)

    def executed_results(self) -> list[ExperimentResult]:
        """Full results for experiments that were actually trained, if any."""
        return list(getattr(self, "_results", []))


class SearchTracker:
    """Shared episode bookkeeping for both environments.

    Owns the history, the novelty computation, the running best score and the reward
    accounting. Keeping it here -- rather than duplicating it in the real and simulated
    environments -- is what guarantees the RL agent sees identical dynamics in both, and
    therefore that the simulator is a legitimate stand-in during training.
    """

    def __init__(
        self,
        *,
        reward_calculator: Any,
        max_experiments: int,
        n_possible_models: int = 1,
        direction: Any = None,
    ) -> None:
        self.reward_calculator = reward_calculator
        self.max_experiments = max(1, max_experiments)
        self.n_possible_models = max(1, n_possible_models)
        self.direction = direction if direction is not None else reward_calculator.direction
        self.reset()

    def reset(self) -> None:
        self._history: list[SearchHistoryEntry] = []
        self._results: list[ExperimentResult] = []
        self._signatures: list[str] = []
        self._signature_set: set[str] = set()
        self._family_scores: dict[str, float] = {}
        self._gains: list[float] = []
        self._first_score: float | None = None
        self._best_score: float | None = None
        self._last_gain = 0.0
        self._stagnation = 0
        self._mean_novelty = 1.0
        self._last_failed = False
        self._elapsed_fraction = 0.0
        self._memory: dict[str, Any] = {}

    # -- state -------------------------------------------------------------------

    @property
    def history(self) -> list[SearchHistoryEntry]:
        return self._history

    @property
    def results(self) -> list[ExperimentResult]:
        return self._results

    @property
    def n_experiments(self) -> int:
        return len(self._history)

    @property
    def best_oriented_score(self) -> float | None:
        return self._best_score

    @property
    def signatures(self) -> list[str]:
        return self._signatures

    def novelty(self, signature: str) -> float:
        return novelty_score(signature, self._signature_set)

    def is_duplicate(self, signature: str) -> bool:
        return signature in self._signature_set

    def set_elapsed_fraction(self, fraction: float) -> None:
        self._elapsed_fraction = float(fraction)

    def set_memory_features(
        self,
        *,
        best_score: float | None = None,
        similarity: float = 0.0,
        attempts_to_best: float | None = None,
        n_similar: int = 0,
        family_affinity: dict[str, float] | None = None,
    ) -> None:
        """Attach cross-dataset memory (spec §24). Safe to call with defaults."""
        self._memory = {
            "best_score": best_score,
            "similarity": similarity,
            "attempts_to_best": attempts_to_best,
            "n_similar": n_similar,
            "family_affinity": dict(family_affinity or {}),
        }

    # -- recording ---------------------------------------------------------------

    def register(
        self,
        experiment: ExperimentSpec,
        *,
        score: float | None,
        cost_s: float,
        family: str,
        failed: bool = False,
        invalid: bool = False,
        result: ExperimentResult | None = None,
    ) -> tuple[RewardBreakdown, SearchHistoryEntry]:
        """Record one attempted experiment and return its shaped reward."""
        signature = experiment.signature()
        duplicate = self.is_duplicate(signature)
        novelty = self.novelty(signature)

        oriented = None if score is None else float(orient(score, self.direction))
        best_before = self._best_score

        breakdown = self.reward_calculator.compute(
            score=oriented,
            best_before=best_before,
            cost_s=cost_s,
            novelty=novelty,
            stagnation=self._stagnation,
            failed=failed,
            invalid=invalid,
        )
        self.reward_calculator.observe(breakdown.raw_total)

        # Progress bookkeeping.
        if oriented is not None and not failed:
            if self._first_score is None:
                self._first_score = oriented
            gain = oriented - best_before if best_before is not None else 0.0
            self._last_gain = gain
            self._gains.append(gain)
            if best_before is None or oriented > best_before:
                self._best_score = oriented
                self._stagnation = 0
            else:
                self._stagnation += 1
            self._update_family(family, oriented)
        else:
            self._last_gain = 0.0
            self._stagnation += 1

        self._last_failed = bool(failed or invalid)
        self._signatures.append(signature)
        self._signature_set.add(signature)
        n = len(self._history) + 1
        self._mean_novelty += (novelty - self._mean_novelty) / n

        entry = SearchHistoryEntry(
            step=n,
            model=experiment.model,
            preset_index=experiment.preset_index,
            preprocessing=tuple(experiment.preprocessing),
            feature_selection=experiment.feature_selection,
            score=score,
            oriented_score=oriented,
            cost_s=cost_s,
            reward=breakdown.total,
            failed=bool(failed or invalid),
            duplicate=duplicate,
        )
        self._history.append(entry)
        if result is not None:
            self._results.append(result)
        return breakdown, entry

    def _update_family(self, family: str, oriented: float) -> None:
        """Track which model families are actually working on this dataset.

        Stored as a bounded [0, 1] score where 0.5 is neutral, so the encoder can consume
        it directly.
        """
        reference = max(abs(self._first_score or 0.0), 1e-6)
        value = 0.5 + 0.5 * float(np.tanh((oriented - (self._first_score or 0.0)) / reference))
        previous = self._family_scores.get(family)
        self._family_scores[family] = value if previous is None else (previous + value) / 2.0

    # -- observation -------------------------------------------------------------

    def progress(self) -> SearchProgress:
        mean_gain = float(np.mean(self._gains)) if self._gains else 0.0
        memory = self._memory
        return SearchProgress(
            step=self.n_experiments,
            max_steps=self.max_experiments,
            n_experiments=self.n_experiments,
            best_score=self._best_score,
            first_score=self._first_score,
            last_gain=self._last_gain,
            mean_gain=mean_gain,
            stagnation=self._stagnation,
            n_distinct_models=len({entry.model for entry in self._history}),
            n_distinct_signatures=len(self._signature_set),
            n_possible_models=self.n_possible_models,
            mean_novelty=self._mean_novelty,
            last_failed=self._last_failed,
            elapsed_fraction=self._elapsed_fraction,
            budget_fraction=self.n_experiments / self.max_experiments,
            family_performance=dict(self._family_scores),
            memory_best_score=memory.get("best_score"),
            memory_similarity=float(memory.get("similarity", 0.0)),
            memory_attempts_to_best=memory.get("attempts_to_best"),
            memory_n_similar=int(memory.get("n_similar", 0)),
            memory_family_affinity=dict(memory.get("family_affinity", {})),
        )

    def terminal_breakdown(self, stopping_reason: str) -> RewardBreakdown:
        return self.reward_calculator.compute_terminal(
            best_score=self._best_score,
            n_experiments=self.n_experiments,
            stopping_reason=stopping_reason,
        )


__all__ = [
    "AutoMLEnvironment",
    "SearchHistoryEntry",
    "SearchTracker",
    "StepResult",
]
