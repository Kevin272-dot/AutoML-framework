"""Critic: state-value head over the shared trunk.

Kept deliberately small. With a handful of heads and a few hundred input dimensions, a
two-layer MLP is already more than enough capacity, and a smaller critic trains faster and
is less prone to chasing the non-stationary targets that AutoML search produces.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rl_automl.agent.actor import MlpTrunk


class Critic(nn.Module):
    def __init__(self, trunk: MlpTrunk, hidden_size: int = 128) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = nn.Sequential(
            nn.Linear(trunk.output_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return state values with shape ``(batch,)``."""
        features = self.trunk(state)
        return self.head(features).squeeze(-1)


__all__ = ["Critic"]
