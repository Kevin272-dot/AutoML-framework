"""Real AutoML environment (spec §5, §35).

Same contract as the simulator, but :meth:`step` performs actual training through the
:class:`~rl_automl.execution.executor.PipelineExecutor`. This is the bridge between the RL
agent and the execution engine -- and the *only* place the two meet. The agent never
imports the executor: it sees an :class:`AutoMLEnvironment`, and whether that is real or
simulated is a wiring decision made by the caller.

Used for three things:

* collecting the meta-dataset that fits the surrogate (``scripts/build_meta_dataset.py``)
* running the honest, real-training half of the benchmark in ``evaluation.benchmark``
* optionally letting the agent search with real training when a user asks for it
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.logging import get_logger
from rl_automl.core.types import (
    DatasetFingerprint,
    ExperimentResult,
    TaskSpec,
)
from rl_automl.dataset.fingerprint import build_fingerprint
from rl_automl.dataset.profiler import DatasetProfiler
from rl_automl.environment.action_space import ActionMaskTable, ActionSpace, ActionSpaceLayout
from rl_automl.environment.base import AutoMLEnvironment, SearchTracker, StepResult
from rl_automl.environment.state_encoder import StateEncoder
from rl_automl.execution.executor import ExecutionLimits, PipelineExecutor
from rl_automl.search.model_registry import get_spec

logger = get_logger("environment.automl_env")


@dataclass
class RealEnvConfig:
    max_experiments: int = 12
    min_experiments_before_stop: int = 1
    time_budget_s: float = 3600.0
    memory_budget_mb: float = 8192.0


class AutoMLEnv(AutoMLEnvironment):
    """Environment that trains for real."""

    def __init__(
        self,
        *,
        task: TaskSpec,
        frame: pd.DataFrame,
        config: AutoMLConfig | None = None,
        action_space: ActionSpace,
        encoder: StateEncoder,
        tracker: SearchTracker,
        run_id: str = "env",
        env_config: RealEnvConfig | None = None,
        fingerprint: DatasetFingerprint | None = None,
    ) -> None:
        self.config = config or AutoMLConfig()
        self.env_config = env_config or RealEnvConfig(
            max_experiments=self.config.search.max_experiments,
            time_budget_s=self.config.search.time_budget_s,
            memory_budget_mb=self.config.search.memory_budget_mb,
        )
        super().__init__(task.task_type, self.env_config.max_experiments)

        self.task = task
        self.frame = frame
        self.action_space = action_space
        self.encoder = encoder
        self.tracker = tracker
        self.run_id = run_id

        self.profile = DatasetProfiler(self.config.dataset).profile(frame, target=task.target)
        self.fingerprint = fingerprint or build_fingerprint(self.profile, task.task_type)

        self.executor = PipelineExecutor(
            task,
            self.config,
            run_id=run_id,
            seed=self.config.runtime.seed,
            limits=ExecutionLimits(
                max_experiments=self.env_config.max_experiments,
                time_budget_s=self.env_config.time_budget_s,
                memory_budget_mb=self.env_config.memory_budget_mb,
            ),
        )
        self._prepared = False
        self._meta_rows: list[dict[str, Any]] = []
        self._state = np.zeros(encoder.dim, dtype=np.float64)

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
            elapsed_fraction=self._elapsed_fraction(),
        )

    @property
    def history(self):
        return self.tracker.history

    @property
    def simulated(self) -> bool:
        return False

    def reset(self) -> np.ndarray:
        if not self._prepared:
            self.executor.prepare(self.frame)
            self._prepared = True
        self.tracker.reset()
        self._state = self._encode()
        return self._state

    def step(self, action: np.ndarray | tuple[int, ...]) -> StepResult:
        masks = self.mask_table
        values = [int(value) for value in np.asarray(action).reshape(-1)]

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
        if experiment is None:
            breakdown = self.tracker.terminal_breakdown("agent_stop")
            return StepResult(
                state=self._state,
                reward=breakdown.total,
                done=True,
                terminated=True,
                info={"stopped": True, "reward": breakdown.to_dict()},
            )

        spec = get_spec(experiment.model)

        try:
            result = self.executor.execute(experiment)
        except Exception as exc:  # budget exhaustion and friends
            logger.warning(
                "real environment halting on executor exception",
                extra={"context": {"error": f"{type(exc).__name__}: {exc}"}},
            )
            breakdown = self.tracker.terminal_breakdown("executor_error")
            return StepResult(
                state=self._state,
                reward=breakdown.total,
                done=True,
                truncated=True,
                info={"error": f"{type(exc).__name__}: {exc}"},
            )

        failed = not result.succeeded
        score = result.validation_score if not failed else None
        cost_s = result.training_time_s

        breakdown, entry = self.tracker.register(
            experiment,
            score=score,
            cost_s=cost_s,
            family=spec.family,
            failed=failed,
            result=result,
        )
        self._record_meta_row(experiment, result, spec.family)

        done = self.tracker.n_experiments >= self.max_experiments
        terminal_reward = 0.0
        if done:
            terminal_reward = self.tracker.terminal_breakdown("max_experiments").total

        self._state = self._encode()
        return StepResult(
            state=self._state,
            reward=breakdown.total + terminal_reward,
            done=done,
            truncated=done,
            info={
                "experiment": experiment.to_dict(),
                "result": result.model_dump(mode="json"),
                "reward": breakdown.to_dict(),
                "duplicate": entry.duplicate,
                "simulated": False,
                "best_score": self.tracker.best_oriented_score,
            },
        )

    # -- metadata collection -----------------------------------------------------

    def _record_meta_row(self, experiment: Any, result: ExperimentResult, family: str) -> None:
        """Rows that ``surrogate.SurrogateModel.fit`` can consume (spec §24, §30)."""
        self._meta_rows.append(
            {
                "model_key": experiment.model,
                "family": family,
                "task_type": self.task.task_type,
                "preset_index": experiment.preset_index,
                "n_presets": get_spec(experiment.model).n_presets(self.task.task_type),
                "preprocessing": list(experiment.preprocessing),
                "feature_selection": experiment.feature_selection,
                "fingerprint": list(self.fingerprint.values),
                "score": result.validation_score,
                "cost_s": result.training_time_s,
                "peak_memory_mb": result.peak_memory_mb,
                "failed": not result.succeeded,
                "error_type": result.error_type,
                "dataset_profile": {
                    "n_rows": self.profile.n_rows,
                    "n_cols": self.profile.n_cols,
                },
            }
        )

    @property
    def meta_rows(self) -> list[dict[str, Any]]:
        return self._meta_rows

    def execution_results(self) -> list[ExperimentResult]:
        return self.executor.results() if hasattr(self.executor, "results") else []

    # -- internals ---------------------------------------------------------------

    def _encode(self) -> np.ndarray:
        self.tracker.set_elapsed_fraction(self._elapsed_fraction())
        return self.encoder.encode(self.fingerprint, self.task.task_type, self.tracker.progress())

    def _elapsed_fraction(self) -> float:
        if not self._prepared or self.executor.limits.time_budget_s <= 0:
            return 0.0
        return float(min(1.0, self.executor.elapsed_s / self.executor.limits.time_budget_s))


__all__ = ["AutoMLEnv", "RealEnvConfig"]
