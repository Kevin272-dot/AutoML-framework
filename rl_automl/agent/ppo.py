"""PPO (spec §5).

Clipped-surrogate PPO with generalised advantage estimation, advantage normalisation,
gradient clipping, an entropy bonus that decays over training, and optional early stopping
on approximate KL divergence.

The implementation takes the policy as a module exposing ``actor`` and ``critic``, so it
does not depend on the concrete Agent class and can be unit-tested against a trivial
stand-in policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rl_automl.agent.rollout_buffer import RolloutBuffer
from rl_automl.core.logging import get_logger

logger = get_logger("agent.ppo")


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.02
    entropy_coef_final: float = 0.002
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 256
    rollout_steps: int = 2048
    normalize_advantages: bool = True
    target_kl: float | None = 0.03
    seed: int = 0

    @classmethod
    def from_rl_config(cls, rl_config: Any) -> PPOConfig:
        return cls(
            learning_rate=rl_config.lr,
            gamma=rl_config.gamma,
            gae_lambda=rl_config.gae_lambda,
            clip_ratio=rl_config.clip_ratio,
            entropy_coef=rl_config.entropy_coef,
            entropy_coef_final=rl_config.entropy_coef_final,
            value_coef=rl_config.value_coef,
            max_grad_norm=rl_config.max_grad_norm,
            update_epochs=rl_config.update_epochs,
            minibatch_size=rl_config.minibatch_size,
            rollout_steps=rl_config.rollout_steps,
            seed=rl_config.seed if hasattr(rl_config, "seed") else 0,
        )


@dataclass
class UpdateStats:
    """Averaged diagnostics from one PPO update."""

    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    entropy_coef: float = 0.0
    n_minibatches: int = 0
    early_stopped: bool = False
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_loss": round(self.policy_loss, 6),
            "value_loss": round(self.value_loss, 6),
            "entropy": round(self.entropy, 6),
            "total_loss": round(self.total_loss, 6),
            "approx_kl": round(self.approx_kl, 6),
            "clip_fraction": round(self.clip_fraction, 5),
            "entropy_coef": round(self.entropy_coef, 6),
            "n_minibatches": self.n_minibatches,
            "early_stopped": self.early_stopped,
        }


class PPO:
    def __init__(
        self,
        policy: nn.Module,
        config: PPOConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.policy = policy.to(device)
        self.config = config or PPOConfig()
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.entropy_coef = self.config.entropy_coef
        self.n_updates = 0

    # -- entropy schedule --------------------------------------------------------

    def schedule_entropy(self, progress: float) -> float:
        """Linearly decay the entropy bonus from its initial to its final value."""
        progress = float(min(max(progress, 0.0), 1.0))
        self.entropy_coef = self.config.entropy_coef + progress * (
            self.config.entropy_coef_final - self.config.entropy_coef
        )
        return self.entropy_coef

    # -- update ------------------------------------------------------------------

    def update(
        self,
        buffer: RolloutBuffer,
        *,
        last_value: float = 0.0,
        seed: int | None = None,
    ) -> UpdateStats:
        """One PPO update over a filled rollout buffer."""
        if buffer.is_empty():
            raise ValueError("cannot update PPO from an empty rollout buffer")

        buffer.compute_gae(last_value, gamma=self.config.gamma, lam=self.config.gae_lambda)
        if self.config.normalize_advantages:
            buffer.normalize_advantages()

        seed = self.config.seed if seed is None else seed
        batches = buffer.batches(
            self.config.minibatch_size, self.config.update_epochs, seed=seed + self.n_updates
        )

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        counted = 0
        early_stopped = False

        for batch in batches:
            log_probs, entropy = self.policy.actor.evaluate_actions(
                batch.states, batch.actions, batch.masks
            )
            values = self.policy.critic(batch.states)

            log_ratio = log_probs - batch.log_probs
            ratio = torch.exp(log_ratio)
            advantages = batch.advantages

            unclipped = ratio * advantages
            clipped = (
                torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                * advantages
            )
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = torch.nn.functional.mse_loss(values, batch.returns)
            entropy_mean = entropy.mean()

            loss = (
                policy_loss + self.config.value_coef * value_loss - self.entropy_coef * entropy_mean
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                approx_kl = float(((ratio - 1.0) - log_ratio).mean())
                clip_fraction = float(((ratio - 1.0).abs() > self.config.clip_ratio).float().mean())

            totals["policy_loss"] += float(policy_loss.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["entropy"] += float(entropy_mean.detach())
            totals["total_loss"] += float(loss.detach())
            totals["approx_kl"] += approx_kl
            totals["clip_fraction"] += clip_fraction
            counted += 1

            # Standard PPO safeguard: if the policy has already moved too far, further
            # epochs would be optimising against stale advantages.
            if self.config.target_kl is not None and approx_kl > 1.5 * self.config.target_kl:
                early_stopped = True
                break

        self.n_updates += 1
        denominator = max(counted, 1)
        stats = UpdateStats(
            policy_loss=totals["policy_loss"] / denominator,
            value_loss=totals["value_loss"] / denominator,
            entropy=totals["entropy"] / denominator,
            total_loss=totals["total_loss"] / denominator,
            approx_kl=totals["approx_kl"] / denominator,
            clip_fraction=totals["clip_fraction"] / denominator,
            entropy_coef=self.entropy_coef,
            n_minibatches=counted,
            early_stopped=early_stopped,
        )
        return stats

    # -- persistence -------------------------------------------------------------

    def optimizer_state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_optimizer_state_dict(self, payload: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(payload)


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """1.0 means the critic predicts returns perfectly; 0.0 means it is no better than the mean."""
    if len(values) < 2:
        return 0.0
    variance = float(np.var(returns))
    if variance <= 1e-12:
        return 0.0
    return float(1.0 - np.var(returns - values) / variance)


__all__ = ["PPO", "PPOConfig", "UpdateStats", "explained_variance"]
