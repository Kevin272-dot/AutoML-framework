"""PPO agent: composition, training loop, and checkpointing (spec §5, §8).

The agent is the composition root for the RL half of the system. Two properties matter
most:

* **Modularity (spec §35).** The agent depends on the environment *interface*, the action
  space and the registry-derived masks. It never imports ``execution``, so the policy can
  be trained, saved and inspected with no training stack installed.
* **Versioned checkpoints.** A saved policy is only meaningful alongside the state schema,
  the action space and the registry contents it was trained against. All four are recorded
  and checked on load, and a mismatch raises rather than silently producing nonsense
  recommendations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rl_automl.agent.actor import Actor, MlpTrunk
from rl_automl.agent.critic import Critic
from rl_automl.agent.ppo import PPO, PPOConfig, UpdateStats, explained_variance
from rl_automl.agent.rollout_buffer import MaskBatch, RolloutBuffer
from rl_automl.core.config import RLConfig
from rl_automl.core.errors import PolicyLoadError
from rl_automl.core.logging import get_logger
from rl_automl.core.seeding import torch_device
from rl_automl.core.types import utc_now_iso
from rl_automl.environment.action_space import ACTION_SPACE_VERSION, ActionMaskTable, ActionSpace
from rl_automl.environment.base import AutoMLEnvironment
from rl_automl.environment.state_encoder import STATE_SCHEMA_VERSION, StateEncoder

logger = get_logger("agent.agent")

POLICY_FORMAT_VERSION = "1.0.0"


@dataclass
class PolicyMetadata:
    """Everything needed to decide whether a checkpoint can still be used."""

    format_version: str = POLICY_FORMAT_VERSION
    state_schema_version: str = STATE_SCHEMA_VERSION
    action_space_version: str = ACTION_SPACE_VERSION
    obs_dim: int = 0
    head_sizes: list[int] = field(default_factory=list)
    hidden_sizes: list[int] = field(default_factory=list)
    model_keys: list[str] = field(default_factory=list)
    task_type: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    trained_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "state_schema_version": self.state_schema_version,
            "action_space_version": self.action_space_version,
            "obs_dim": self.obs_dim,
            "head_sizes": list(self.head_sizes),
            "hidden_sizes": list(self.hidden_sizes),
            "model_keys": list(self.model_keys),
            "task_type": self.task_type,
            "created_at": self.created_at,
            "trained_steps": self.trained_steps,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyMetadata:
        return cls(**payload)

    def incompatibility(self, expected: PolicyMetadata) -> str | None:
        """Describe the first incompatibility, or ``None`` if the checkpoint is usable."""
        if self.format_version != expected.format_version:
            return f"policy format {self.format_version} != {expected.format_version}"
        if self.state_schema_version != expected.state_schema_version:
            return (
                f"state schema {self.state_schema_version} != {expected.state_schema_version}; "
                "retrain or migrate the policy"
            )
        if self.action_space_version != expected.action_space_version:
            return f"action space {self.action_space_version} != {expected.action_space_version}"
        if self.obs_dim != expected.obs_dim:
            return f"observation dim {self.obs_dim} != {expected.obs_dim}"
        if list(self.head_sizes) != list(expected.head_sizes):
            return f"head sizes {self.head_sizes} != {expected.head_sizes}"
        if list(self.model_keys) != list(expected.model_keys):
            return (
                "the registry's model list changed since this policy was trained "
                f"({len(self.model_keys)} vs {len(expected.model_keys)} models)"
            )
        if self.task_type and expected.task_type and self.task_type != expected.task_type:
            return f"task type {self.task_type} != {expected.task_type}"
        return None


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic. One feature extractor, four policy heads, one value head."""

    def __init__(
        self,
        obs_dim: int,
        head_sizes: tuple[int, ...],
        hidden_sizes: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "tanh",
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.trunk = MlpTrunk(obs_dim, hidden_sizes, activation)
        self.actor = Actor(self.trunk, head_sizes, head_hidden)
        self.critic = Critic(self.trunk)

    def forward(self, state: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        logits = self.actor(state)
        values = self.critic(state)
        return logits, values


@dataclass
class TrainingReport:
    total_steps: int = 0
    n_updates: int = 0
    n_episodes: int = 0
    elapsed_s: float = 0.0
    episode_returns: list[float] = field(default_factory=list)
    episode_best_scores: list[float | None] = field(default_factory=list)
    update_history: list[dict[str, Any]] = field(default_factory=list)
    explained_variance: list[float] = field(default_factory=list)
    final_entropy_coef: float = 0.0

    @property
    def mean_episode_return(self) -> float:
        return float(np.mean(self.episode_returns)) if self.episode_returns else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "n_updates": self.n_updates,
            "n_episodes": self.n_episodes,
            "elapsed_s": round(self.elapsed_s, 3),
            "mean_episode_return": round(self.mean_episode_return, 4),
            "n_episode_returns": len(self.episode_returns),
            "final_entropy_coef": round(self.final_entropy_coef, 6),
            "first_updates": self.update_history[:3],
            "last_updates": self.update_history[-3:],
        }


class PPOAgent:
    """Wraps :class:`ActorCritic` and :class:`PPO` with acting, training and persistence."""

    def __init__(
        self,
        *,
        obs_dim: int,
        head_sizes: tuple[int, ...],
        model_keys: list[str],
        task_type: str,
        rl_config: RLConfig | None = None,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.rl_config = rl_config or RLConfig()
        self.seed = self.rl_config.seed if seed is None else seed
        self.device = self._resolve_device(device or self.rl_config.device)

        self.metadata = PolicyMetadata(
            obs_dim=int(obs_dim),
            head_sizes=[int(size) for size in head_sizes],
            hidden_sizes=[int(size) for size in self.rl_config.hidden_sizes],
            model_keys=list(model_keys),
            task_type=task_type,
        )

        torch.manual_seed(int(self.seed) if self.seed is not None else 0)
        self.policy = ActorCritic(
            obs_dim=self.metadata.obs_dim,
            head_sizes=head_sizes,
            hidden_sizes=self.rl_config.hidden_sizes,
        ).to(self.device)

        ppo_config = PPOConfig.from_rl_config(self.rl_config)
        ppo_config.seed = int(self.seed or 0)
        self.ppo = PPO(self.policy, ppo_config, device=self.device)

    # -- construction ------------------------------------------------------------

    @classmethod
    def from_action_space(
        cls,
        action_space: ActionSpace,
        state_dim: int,
        rl_config: RLConfig | None = None,
        device: str | None = None,
        seed: int | None = None,
    ) -> PPOAgent:
        return cls(
            obs_dim=state_dim,
            head_sizes=action_space.layout.head_sizes,
            model_keys=list(action_space.layout.model_keys),
            task_type=action_space.layout.task_type,
            rl_config=rl_config,
            device=device,
            seed=seed,
        )

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """``auto`` picks CUDA when present; anything explicit is honoured verbatim."""
        if requested in ("auto", ""):
            return torch_device(prefer_cuda=True)
        return requested

    @property
    def head_sizes(self) -> tuple[int, ...]:
        return tuple(self.metadata.head_sizes)

    @property
    def n_slots(self) -> int:
        return self.head_sizes[0]

    # -- acting ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        mask_table: ActionMaskTable,
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """Single-observation action. Returns (action, log_prob, value)."""
        batch, log_prob, value = self.act_batch(
            np.asarray(state, dtype=np.float32).reshape(1, -1),
            mask_table,
            deterministic=deterministic,
        )
        return batch.row(0), float(log_prob[0]), float(value[0])

    @torch.no_grad()
    def act_batch(
        self,
        states: np.ndarray,
        mask_table: ActionMaskTable,
        *,
        deterministic: bool = False,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        """Batched action. Used by the recommendation phase to roll out many plans at once."""
        tensor = torch.as_tensor(np.atleast_2d(states), dtype=torch.float32, device=self.device)
        masks = MaskBatch.from_table(mask_table, self.device)
        sample = (
            self.policy.actor.greedy(tensor, masks)
            if deterministic
            else self.policy.actor.act(tensor, masks)
        )
        values = self.policy.critic(tensor)
        return (
            sample.actions,
            sample.log_prob.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
        )

    @torch.no_grad()
    def value(self, state: np.ndarray) -> float:
        tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32).reshape(1, -1), device=self.device
        )
        return float(self.policy.critic(tensor)[0])

    # -- training ----------------------------------------------------------------

    def train(
        self,
        environments: list[AutoMLEnvironment],
        *,
        total_steps: int | None = None,
        on_update: Callable[[int, UpdateStats, TrainingReport], None] | None = None,
        on_episode: Callable[[int, AutoMLEnvironment], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> TrainingReport:
        """Train across a set of environments.

        Cycling through *many* environments is the point: swapping datasets on every
        episode boundary is what makes the policy learn to condition on the dataset
        fingerprint instead of memorising one problem (spec §24, §31C).
        """
        import time

        if not environments:
            raise ValueError("training requires at least one environment")

        # A rollout buffer stores one static mask table, so every environment in a training
        # mix must share the same action space. Mixing task types here would silently
        # misalign masks, so it is rejected up front.
        reference = environments[0]
        for candidate in environments[1:]:
            if candidate.layout.head_sizes != reference.layout.head_sizes:
                raise PolicyLoadError(
                    "all training environments must share one action space; got head sizes "
                    f"{candidate.layout.head_sizes} and {reference.layout.head_sizes}"
                )
            if candidate.layout.model_keys != reference.layout.model_keys:
                raise PolicyLoadError("all training environments must expose the same model list")

        total_steps = int(total_steps or self.rl_config.total_steps)
        rollout_steps = max(1, int(self.rl_config.rollout_steps))
        report = TrainingReport()
        started = time.perf_counter()

        env_index = 0
        env = environments[env_index]
        state = env.reset()
        episode_return = 0.0
        global_step = 0

        while global_step < total_steps:
            buffer = RolloutBuffer(
                capacity=rollout_steps,
                obs_dim=self.metadata.obs_dim,
                head_sizes=self.head_sizes,
                mask_tables=MaskBatch.from_table(env.mask_table, "cpu"),
                device=self.device,
            )

            last_value = 0.0
            for _ in range(rollout_steps):
                mask_table = env.mask_table
                action, log_prob, value = self.act(state, mask_table)
                result = env.step(action)

                buffer.add(
                    state=np.asarray(state, dtype=np.float32),
                    action=np.asarray(action, dtype=np.float64),
                    log_prob=log_prob,
                    reward=result.reward,
                    value=value,
                    done=result.done,
                    model_mask=np.asarray(mask_table.model, dtype=bool),
                )

                state = result.state
                episode_return += result.reward
                global_step += 1

                if result.done:
                    report.episode_returns.append(episode_return)
                    report.episode_best_scores.append(env.best_score)
                    report.n_episodes += 1
                    if on_episode is not None:
                        on_episode(report.n_episodes, env)

                    env_index = (env_index + 1) % len(environments)
                    env = environments[env_index]
                    state = env.reset()
                    episode_return = 0.0
                    last_value = 0.0

                if global_step >= total_steps:
                    break

            if buffer.is_empty():
                continue

            # Bootstrap from the current state. When the rollout ended exactly on an
            # episode boundary this is the *next* episode's first state, which is correct:
            # compute_gae multiplies the bootstrap by (1 - done) and that done flag is 1.
            last_value = self.value(state)

            # Entropy decays over the whole run so the policy explores early and commits late.
            self.ppo.schedule_entropy(global_step / max(total_steps, 1))
            stats = self.ppo.update(buffer, last_value=last_value, seed=int(self.seed or 0))

            with torch.no_grad():
                values = torch.as_tensor(buffer.values[: buffer.size], device=self.device)
                returns = torch.as_tensor(buffer.returns[: buffer.size], device=self.device)
                report.explained_variance.append(
                    explained_variance(values.cpu().numpy(), returns.cpu().numpy())
                )

            report.n_updates += 1
            report.update_history.append({**stats.to_dict(), "step": global_step})
            report.final_entropy_coef = stats.entropy_coef

            if on_update is not None:
                on_update(global_step, stats, report)
            if progress_callback is not None:
                progress_callback(global_step, total_steps)

        report.total_steps = global_step
        report.elapsed_s = time.perf_counter() - started
        self.metadata.trained_steps += global_step
        logger.info(
            "policy trained",
            extra={
                "context": {
                    "steps": global_step,
                    "updates": report.n_updates,
                    "episodes": report.n_episodes,
                    "elapsed_s": round(report.elapsed_s, 2),
                }
            },
        )
        return report

    # -- persistence -------------------------------------------------------------

    def save(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> Path:
        """Write a checkpoint. ``reward`` state is stored by the caller via ``extra``."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata.to_dict(),
            "model_state": self.policy.state_dict(),
            "optimizer_state": self.ppo.optimizer_state_dict(),
            "n_ppo_updates": self.ppo.n_updates,
            "extra": extra or {},
        }
        torch.save(payload, destination)
        logger.info("policy saved", extra={"context": {"path": str(destination)}})
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        action_space: ActionSpace,
        state_encoder: StateEncoder,
        rl_config: RLConfig | None = None,
        device: str | None = None,
        strict: bool = True,
    ) -> tuple[PPOAgent, dict[str, Any]]:
        """Load a checkpoint after checking it still matches this registry and schema.

        ``weights_only=False`` is deliberate and safe here: the only files this ever loads
        are checkpoints written by :meth:`save` inside this process's own artifact tree,
        never a user-supplied file (spec §32).
        """
        source = Path(path)
        if not source.is_file():
            raise PolicyLoadError(f"policy checkpoint not found: {source}")

        try:
            payload = torch.load(source, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise PolicyLoadError(f"could not read policy checkpoint: {exc}") from exc

        if not isinstance(payload, dict) or "metadata" not in payload:
            raise PolicyLoadError("checkpoint is missing its metadata block")

        stored = PolicyMetadata.from_dict(payload["metadata"])
        expected = PolicyMetadata(
            obs_dim=state_encoder.dim,
            head_sizes=list(action_space.layout.head_sizes),
            hidden_sizes=list((rl_config or RLConfig()).hidden_sizes),
            model_keys=list(action_space.layout.model_keys),
            task_type=action_space.layout.task_type,
        )

        problem = stored.incompatibility(expected)
        if problem is not None:
            message = f"policy checkpoint is incompatible with the current configuration: {problem}"
            if strict:
                raise PolicyLoadError(message, path=str(source))
            logger.warning(message, extra={"context": {"path": str(source)}})

        agent = cls(
            obs_dim=stored.obs_dim,
            head_sizes=tuple(stored.head_sizes),
            model_keys=list(stored.model_keys),
            task_type=stored.task_type,
            rl_config=rl_config,
            device=device,
        )
        try:
            agent.policy.load_state_dict(payload["model_state"])
        except Exception as exc:
            raise PolicyLoadError(f"policy weights could not be loaded: {exc}") from exc

        # Restore the stored metadata verbatim so provenance (creation time, number of
        # training steps) survives a save/load round trip rather than resetting to defaults.
        agent.metadata = stored

        if "optimizer_state" in payload:
            try:
                agent.ppo.load_optimizer_state_dict(payload["optimizer_state"])
            except Exception:  # pragma: no cover - optimiser state is not essential
                logger.warning("optimizer state could not be restored; continuing without it")
        agent.ppo.n_updates = int(payload.get("n_ppo_updates", 0))
        agent.policy.eval()

        return agent, payload.get("extra", {})

    def describe(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "metadata": self.metadata.to_dict(),
            "hyperparameters": {
                "learning_rate": self.ppo.config.learning_rate,
                "gamma": self.ppo.config.gamma,
                "clip_ratio": self.ppo.config.clip_ratio,
                "rollout_steps": self.ppo.config.rollout_steps,
                "update_epochs": self.ppo.config.update_epochs,
            },
            "n_parameters": int(sum(p.numel() for p in self.policy.parameters())),
        }


__all__ = [
    "POLICY_FORMAT_VERSION",
    "ActorCritic",
    "PPOAgent",
    "PolicyMetadata",
    "TrainingReport",
]
