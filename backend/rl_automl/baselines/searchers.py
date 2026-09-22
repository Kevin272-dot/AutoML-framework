"""Budget-matched baseline searchers over the shared AutoML environment contract.

Every searcher consumes exactly one environment step per attempted experiment and receives
no information that is unavailable to the PPO policy. This matters more than algorithmic
sophistication: a baseline with privileged access to surrogate predictions would make the
comparison invalid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rl_automl.agent.agent import PPOAgent
from rl_automl.core.types import CostTier
from rl_automl.environment.action_space import ActionSpace
from rl_automl.environment.base import AutoMLEnvironment

_COST_ORDER = {CostTier.LOW: 0, CostTier.MEDIUM: 1, CostTier.HIGH: 2}


@dataclass
class SearchResult:
    searcher: str
    budget: int
    n_experiments: int
    best_score: float | None
    best_model: str | None
    total_reward: float
    stopped_early: bool
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "searcher": self.searcher,
            "budget": self.budget,
            "n_experiments": self.n_experiments,
            "best_score": self.best_score,
            "best_model": self.best_model,
            "total_reward": self.total_reward,
            "stopped_early": self.stopped_early,
            "history": list(self.history),
        }


class Searcher(ABC):
    name = "searcher"

    @abstractmethod
    def choose_action(
        self,
        state: np.ndarray,
        env: AutoMLEnvironment,
        action_space: ActionSpace,
        step: int,
    ) -> np.ndarray:
        """Choose one legal experiment action."""

    def search(
        self,
        env: AutoMLEnvironment,
        action_space: ActionSpace,
        *,
        budget: int,
    ) -> SearchResult:
        state = env.reset()
        total_reward = 0.0
        stopped_early = False
        for step in range(max(1, int(budget))):
            action = self.choose_action(state, env, action_space, step)
            transition = env.step(action)
            state = transition.state
            total_reward += transition.reward
            if transition.done:
                stopped_early = step + 1 < budget
                break

        best = env.best_entry
        return SearchResult(
            searcher=self.name,
            budget=budget,
            n_experiments=env.n_experiments,
            best_score=env.best_score,
            best_model=best.model if best else None,
            total_reward=float(total_reward),
            stopped_early=stopped_early,
            history=[entry.to_dict() for entry in env.history],
        )


class RandomSearcher(Searcher):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def choose_action(
        self,
        state: np.ndarray,
        env: AutoMLEnvironment,
        action_space: ActionSpace,
        step: int,
    ) -> np.ndarray:
        del state, step
        return action_space.sample_random_action(env.mask_table, self.rng)


class GridSearcher(Searcher):
    name = "grid"

    def __init__(self) -> None:
        self._actions: list[np.ndarray] = []

    def choose_action(
        self,
        state: np.ndarray,
        env: AutoMLEnvironment,
        action_space: ActionSpace,
        step: int,
    ) -> np.ndarray:
        del state
        if not self._actions:
            specs = action_space.all_legal_specs(env.mask_table)
            specs.sort(
                key=lambda spec: (
                    _COST_ORDER.get(spec.expected_cost, 1),
                    spec.model,
                    spec.preset_index,
                    spec.feature_selection,
                )
            )
            self._actions = [np.asarray(action_space.encode(spec), dtype=int) for spec in specs]
        return self._actions[step % len(self._actions)].copy()


class GreedySearcher(Searcher):
    """Cheap model coverage first, then variants of the best observed model."""

    name = "greedy"

    def choose_action(
        self,
        state: np.ndarray,
        env: AutoMLEnvironment,
        action_space: ActionSpace,
        step: int,
    ) -> np.ndarray:
        del state, step
        specs = action_space.all_legal_specs(env.mask_table)
        attempted = {entry.model for entry in env.history}
        untried = [spec for spec in specs if spec.model not in attempted]
        if untried:
            chosen = min(
                untried,
                key=lambda spec: (
                    _COST_ORDER.get(spec.expected_cost, 1),
                    spec.preset_index,
                    spec.model,
                ),
            )
            return np.asarray(action_space.encode(chosen), dtype=int)

        best_model = env.best_entry.model if env.best_entry else None
        attempted_signatures = {
            (
                entry.model,
                entry.preset_index,
                entry.preprocessing,
                entry.feature_selection,
            )
            for entry in env.history
        }
        candidates = [
            spec
            for spec in specs
            if spec.model == best_model
            and (
                spec.model,
                spec.preset_index,
                tuple(spec.preprocessing),
                spec.feature_selection,
            )
            not in attempted_signatures
        ]
        if not candidates:
            candidates = specs
        chosen = min(
            candidates,
            key=lambda spec: (
                _COST_ORDER.get(spec.expected_cost, 1),
                spec.preset_index,
                spec.model,
            ),
        )
        return np.asarray(action_space.encode(chosen), dtype=int)


class PolicySearcher(Searcher):
    name = "ppo"

    def __init__(self, agent: PPOAgent, deterministic: bool = True) -> None:
        self.agent = agent
        self.deterministic = deterministic

    def choose_action(
        self,
        state: np.ndarray,
        env: AutoMLEnvironment,
        action_space: ActionSpace,
        step: int,
    ) -> np.ndarray:
        del action_space, step
        action, _log_prob, _value = self.agent.act(
            state, env.mask_table, deterministic=self.deterministic
        )
        return action


__all__ = [
    "GreedySearcher",
    "GridSearcher",
    "PolicySearcher",
    "RandomSearcher",
    "SearchResult",
    "Searcher",
]
