"""Simulated AutoML environment (spec §30).

Same contract as the real environment, but outcomes come from the surrogate instead of
from training. This is what PPO trains against: thousands of episodes across many dataset
fingerprints, at a cost of microseconds per step.

Being explicit about what this is: **episodes here are simulated**. Anything derived from
them is labelled ``simulated``, and the benchmark that compares the RL planner against
random / greedy / grid search runs in *both* environments, reporting them separately so
that a claim of superiority can never rest on surrogate numbers alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_automl.core.logging import get_logger
from rl_automl.core.types import DatasetFingerprint, TaskType
from rl_automl.environment.action_space import ActionMaskTable, ActionSpace, ActionSpaceLayout
from rl_automl.environment.base import (
    AutoMLEnvironment,
    SearchTracker,
    StepResult,
)
from rl_automl.environment.state_encoder import StateEncoder
from rl_automl.environment.surrogate import SurrogateModel
from rl_automl.search.model_registry import get_spec

logger = get_logger("environment.simulator")


@dataclass
class SimulatorConfig:
    max_experiments: int = 12
    min_experiments_before_stop: int = 1
    seed: int = 0
    add_cost_jitter: bool = True
    simulated: bool = True


class SimulatorEnv(AutoMLEnvironment):
    """Surrogate environment. Cheap enough to run thousands of episodes."""

    def __init__(
        self,
        *,
        task_type: TaskType,
        fingerprint: DatasetFingerprint | np.ndarray,
        action_space: ActionSpace,
        encoder: StateEncoder,
        tracker: SearchTracker,
        surrogate: SurrogateModel,
        config: SimulatorConfig | None = None,
        n_rows: int = 1000,
        n_cols: int = 10,
    ) -> None:
        self.config = config or SimulatorConfig()
        super().__init__(task_type, self.config.max_experiments)

        self.action_space = action_space
        self.encoder = encoder
        self.tracker = tracker
        self.surrogate = surrogate

        self.fingerprint_vector = _as_vector(fingerprint)
        self.fingerprint = fingerprint if isinstance(fingerprint, DatasetFingerprint) else None
        self.n_rows = n_rows
        self.n_cols = n_cols

        self._episode = 0
        self._rng = np.random.default_rng(self.config.seed)
        self._state = np.zeros(self.encoder.dim, dtype=np.float64)

    # -- interface ---------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return self.encoder.dim

    @property
    def layout(self) -> ActionSpaceLayout:
        return self.action_space.layout

    @property
    def mask_table(self) -> ActionMaskTable:
        return self.action_space.mask_table(
            n_experiments=self.tracker.n_experiments,
            elapsed_fraction=self.tracker._elapsed_fraction,
        )

    @property
    def history(self):
        return self.tracker.history

    @property
    def simulated(self) -> bool:
        return True

    def reset(self) -> np.ndarray:
        self._episode += 1
        # Reseeding per episode keeps training reproducible while still giving each
        # episode different failure and cost draws.
        self._rng = np.random.default_rng(self.config.seed + self._episode)
        self.tracker.reset()
        self._state = self._encode()
        return self._state

    def step(self, action: np.ndarray | tuple[int, ...]) -> StepResult:
        masks = self.mask_table
        values = [int(value) for value in np.asarray(action).reshape(-1)]

        # Head 0 is the model slot; reject anything the mask forbids rather than letting
        # the trajectory silently drift off-manifold.
        if values[0] >= len(masks.model) or not masks.is_legal_model(values[0]):
            breakdown = self.tracker.reward_calculator.compute(
                score=None, best_before=None, cost_s=0.0, novelty=0.0, invalid=True
            )
            return StepResult(
                state=self._state,
                reward=breakdown.total,
                done=False,
                info={"invalid_action": True, "action": values},
            )

        experiment = self.action_space.decode(action)
        if experiment is None:  # STOP
            breakdown = self.tracker.terminal_breakdown("agent_stop")
            return StepResult(
                state=self._state,
                reward=breakdown.total,
                done=True,
                terminated=True,
                info={"stopped": True, "reward": breakdown.to_dict()},
            )

        spec = get_spec(experiment.model)
        prediction = self.surrogate.predict(
            task_type=self.task_type,
            model_key=experiment.model,
            family=spec.family,
            fingerprint=self.fingerprint_vector,
            preset_index=experiment.preset_index,
            n_presets=spec.n_presets(self.task_type),
            preprocessing=experiment.preprocessing,
            feature_selection=experiment.feature_selection,
            preset_params=experiment.hyperparameters,
            preset_cost=spec.preset(experiment.preset_index, self.task_type).cost,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
        )

        failed = bool(self._rng.random() < prediction.failure_probability)
        cost_s = prediction.cost_s
        if self.config.add_cost_jitter and cost_s > 0:
            cost_s *= float(np.clip(self._rng.normal(1.0, 0.12), 0.3, 3.0))

        score = None if failed else float(prediction.score)

        breakdown, entry = self.tracker.register(
            experiment,
            score=score,
            cost_s=cost_s,
            family=spec.family,
            failed=failed,
        )

        done = self.tracker.n_experiments >= self.max_experiments
        terminal_reward = 0.0
        if done:
            terminal = self.tracker.terminal_breakdown("max_experiments")
            terminal_reward = terminal.total

        self._state = self._encode()
        return StepResult(
            state=self._state,
            reward=breakdown.total + terminal_reward,
            done=done,
            truncated=done,
            info={
                "experiment": experiment.to_dict(),
                "prediction": prediction.to_dict(),
                "failure_probability": prediction.failure_probability,
                "reward": breakdown.to_dict(),
                "duplicate": entry.duplicate,
                "simulated": True,
                "best_score": self.tracker.best_oriented_score,
            },
        )

    # -- internals ---------------------------------------------------------------

    def _encode(self) -> np.ndarray:
        return self.encoder.encode(self.fingerprint, self.task_type, self.tracker.progress())


def _as_vector(fingerprint: DatasetFingerprint | np.ndarray | list[float]) -> np.ndarray:
    if isinstance(fingerprint, DatasetFingerprint):
        return np.asarray(fingerprint.values, dtype=np.float64)
    return np.asarray(fingerprint, dtype=np.float64).reshape(-1)


def build_simulator(
    *,
    task_type: TaskType,
    fingerprint: DatasetFingerprint | np.ndarray,
    enabled_models: list[str] | None,
    encoder: StateEncoder,
    tracker: SearchTracker,
    surrogate: SurrogateModel,
    max_experiments: int = 12,
    min_experiments_before_stop: int = 1,
    seed: int = 0,
    n_rows: int = 1000,
    n_cols: int = 10,
) -> SimulatorEnv:
    """Convenience constructor that builds the matching action space."""
    from rl_automl.environment.action_space import build_action_space

    action_space = build_action_space(
        task_type,
        enabled_models,
        min_experiments_before_stop=min_experiments_before_stop,
    )
    return SimulatorEnv(
        task_type=task_type,
        fingerprint=fingerprint,
        action_space=action_space,
        encoder=encoder,
        tracker=tracker,
        surrogate=surrogate,
        config=SimulatorConfig(
            max_experiments=max_experiments,
            min_experiments_before_stop=min_experiments_before_stop,
            seed=seed,
        ),
        n_rows=n_rows,
        n_cols=n_cols,
    )


__all__ = ["SimulatorConfig", "SimulatorEnv", "build_simulator"]
