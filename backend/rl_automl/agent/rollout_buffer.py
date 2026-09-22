"""Rollout storage and mask batching.

Two non-obvious requirements shape this module:

1. **Masks must be replayable.** PPO recomputes log-probabilities during the update, and
   with a masked, autoregressive action space those recomputed probabilities are only
   correct if the *same* masks are used. The static per-slot mask tables are therefore
   stored once on the buffer, and only the per-step model mask (whose STOP entry changes
   as the search progresses) is stored per timestep.
2. **Actions are factored.** The action is 4 head values, not one integer, so the buffer
   stores them separately and reassembles them into an :class:`ActionBatch`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rl_automl.core.logging import get_logger

logger = get_logger("agent.rollout_buffer")


@dataclass
class MaskBatch:
    """Action masks in tensor form, indexed by model slot.

    ``model`` is one-dimensional for the common case where the whole batch shares one mask
    (it is identical for every row drawn from the same environment state). When replaying
    stored steps, ``batched_model`` carries the *per-step* model mask instead, so a step
    where STOP was illegal is not re-scored with STOP probability mass. The three tables
    are indexed by the sampled slot to produce per-row masks.
    """

    model: torch.Tensor  # (n_slots,)
    preprocessing_table: torch.Tensor  # (n_slots, n_preprocessing)
    preset_table: torch.Tensor  # (n_slots, max_presets)
    feature_selection_table: torch.Tensor  # (n_slots, n_feature_selection)
    batched_model: torch.Tensor | None = None  # (B, n_slots) when replaying stored steps

    @classmethod
    def from_table(cls, table: Any, device: torch.device | str = "cpu") -> MaskBatch:
        return cls(
            model=torch.as_tensor(np.asarray(table.model), dtype=torch.bool, device=device),
            preprocessing_table=torch.as_tensor(
                np.asarray(table.preprocessing), dtype=torch.bool, device=device
            ),
            preset_table=torch.as_tensor(np.asarray(table.preset), dtype=torch.bool, device=device),
            feature_selection_table=torch.as_tensor(
                np.asarray(table.feature_selection), dtype=torch.bool, device=device
            ),
        )

    @property
    def n_slots(self) -> int:
        return int(self.model.shape[0])

    def model_mask(self, batch_size: int) -> torch.Tensor:
        if self.batched_model is not None and self.batched_model.shape[0] == batch_size:
            return self.batched_model
        return self.model.unsqueeze(0).expand(batch_size, -1)

    def preprocessing_for(self, slots: torch.Tensor) -> torch.Tensor:
        return self.preprocessing_table[slots]

    def preset_for(self, slots: torch.Tensor) -> torch.Tensor:
        return self.preset_table[slots]

    def feature_selection_for(self, slots: torch.Tensor) -> torch.Tensor:
        return self.feature_selection_table[slots]

    def to(self, device: torch.device | str) -> MaskBatch:
        return MaskBatch(
            model=self.model.to(device),
            preprocessing_table=self.preprocessing_table.to(device),
            preset_table=self.preset_table.to(device),
            feature_selection_table=self.feature_selection_table.to(device),
            batched_model=None if self.batched_model is None else self.batched_model.to(device),
        )


@dataclass
class ActionBatch:
    """One or more factored actions."""

    model: torch.Tensor  # (B,) long
    preprocessing: torch.Tensor  # (B, n_preprocessing) float bits
    preset: torch.Tensor  # (B,) long
    feature_selection: torch.Tensor  # (B,) long

    @classmethod
    def from_matrix(cls, actions: np.ndarray, device: torch.device | str = "cpu") -> ActionBatch:
        """Build from the flat ``(model, *preprocessing_bits, preset, feature)`` layout."""
        matrix = np.atleast_2d(np.asarray(actions))
        n_preprocessing = matrix.shape[1] - 3
        return cls(
            model=torch.as_tensor(matrix[:, 0].astype(np.int64), device=device),
            preprocessing=torch.as_tensor(
                matrix[:, 1 : 1 + n_preprocessing].astype(np.float32), device=device
            ),
            preset=torch.as_tensor(matrix[:, 1 + n_preprocessing].astype(np.int64), device=device),
            feature_selection=torch.as_tensor(
                matrix[:, 2 + n_preprocessing].astype(np.int64), device=device
            ),
        )

    def to_matrix(self) -> np.ndarray:
        model = self.model.detach().cpu().numpy().reshape(-1, 1)
        preprocessing = self.preprocessing.detach().cpu().numpy()
        if preprocessing.ndim == 1:
            preprocessing = preprocessing.reshape(1, -1)
        tail = np.stack(
            [
                self.preset.detach().cpu().numpy().reshape(-1),
                self.feature_selection.detach().cpu().numpy().reshape(-1),
            ],
            axis=1,
        )
        return np.concatenate([model, preprocessing, tail], axis=1)

    def row(self, index: int) -> np.ndarray:
        return self.to_matrix()[index]


@dataclass
class RolloutBatch:
    """A minibatch ready for a PPO update."""

    states: torch.Tensor
    actions: ActionBatch
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    masks: MaskBatch


class RolloutBuffer:
    """Fixed-capacity on-policy buffer with GAE."""

    def __init__(
        self,
        *,
        capacity: int,
        obs_dim: int,
        head_sizes: tuple[int, ...],
        mask_tables: MaskBatch,
        device: torch.device | str = "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.head_sizes = tuple(head_sizes)
        self.device = device
        self.mask_tables = mask_tables.to(device)

        self.n_preprocessing = self.head_sizes[1]
        self._reset_arrays()

    def _reset_arrays(self) -> None:
        cap = self.capacity
        self.states = np.zeros((cap, self.obs_dim), dtype=np.float32)
        self.model = np.zeros(cap, dtype=np.int64)
        self.preprocessing = np.zeros((cap, self.n_preprocessing), dtype=np.float32)
        self.preset = np.zeros(cap, dtype=np.int64)
        self.feature_selection = np.zeros(cap, dtype=np.int64)
        self.log_probs = np.zeros(cap, dtype=np.float32)
        self.rewards = np.zeros(cap, dtype=np.float32)
        self.values = np.zeros(cap, dtype=np.float32)
        self.dones = np.zeros(cap, dtype=np.float32)
        self.model_masks = np.zeros((cap, self.head_sizes[0]), dtype=bool)
        self.advantages = np.zeros(cap, dtype=np.float32)
        self.returns = np.zeros(cap, dtype=np.float32)
        self._cursor = 0
        self._computed = False

    # -- writing -----------------------------------------------------------------

    def add(
        self,
        *,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        model_mask: np.ndarray,
    ) -> None:
        if self._cursor >= self.capacity:
            raise RuntimeError(
                f"rollout buffer is full ({self.capacity} steps); call reset() or a larger capacity"
            )
        index = self._cursor
        self.states[index] = state
        self.model[index] = int(action[0])
        self.preprocessing[index] = np.asarray(
            action[1 : 1 + self.n_preprocessing], dtype=np.float32
        )
        self.preset[index] = int(action[1 + self.n_preprocessing])
        self.feature_selection[index] = int(action[2 + self.n_preprocessing])
        self.log_probs[index] = float(log_prob)
        self.rewards[index] = float(reward)
        self.values[index] = float(value)
        self.dones[index] = 1.0 if done else 0.0
        self.model_masks[index] = np.asarray(model_mask, dtype=bool)
        self._cursor += 1

    # -- generalised advantage estimation ----------------------------------------

    def compute_gae(
        self,
        last_value: float,
        *,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        """Standard GAE-lambda, treating both terminals and truncations as episode ends."""
        advantage = 0.0
        for step in reversed(range(self._cursor)):
            if step == self._cursor - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[step]
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]

            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            advantage = delta + gamma * lam * next_non_terminal * advantage
            self.advantages[step] = advantage

        self.returns[: self._cursor] = self.advantages[: self._cursor] + self.values[: self._cursor]
        self._computed = True

    def normalize_advantages(self, epsilon: float = 1e-8) -> None:
        if self._cursor < 2:
            return
        advantages = self.advantages[: self._cursor]
        std = float(advantages.std()) or 1.0
        self.advantages[: self._cursor] = (advantages - advantages.mean()) / (std + epsilon)

    # -- reading -----------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._cursor

    def is_empty(self) -> bool:
        return self._cursor == 0

    def batches(self, minibatch_size: int, epochs: int, seed: int = 0) -> list[RolloutBatch]:
        """Materialise shuffled minibatches for one PPO update."""
        if not self._computed:
            raise RuntimeError("compute_gae() must be called before batches()")

        n = self._cursor
        if n == 0:
            return []

        slices = self._slice_indices(n, minibatch_size, epochs, seed)
        batches: list[RolloutBatch] = []
        for index in slices:
            batches.append(self._materialise(index))
        return batches

    def _slice_indices(
        self, n: int, minibatch_size: int, epochs: int, seed: int
    ) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        minibatch_size = max(1, min(minibatch_size, n))
        out: list[np.ndarray] = []
        for _ in range(max(1, epochs)):
            permutation = rng.permutation(n)
            for start in range(0, n, minibatch_size):
                chunk = permutation[start : start + minibatch_size]
                if len(chunk) > 0:
                    out.append(np.sort(chunk))
        return out

    def _materialise(self, index: np.ndarray) -> RolloutBatch:
        states = torch.as_tensor(self.states[index], dtype=torch.float32, device=self.device)
        actions = ActionBatch(
            model=torch.as_tensor(self.model[index], dtype=torch.int64, device=self.device),
            preprocessing=torch.as_tensor(
                self.preprocessing[index], dtype=torch.float32, device=self.device
            ),
            preset=torch.as_tensor(self.preset[index], dtype=torch.int64, device=self.device),
            feature_selection=torch.as_tensor(
                self.feature_selection[index], dtype=torch.int64, device=self.device
            ),
        )
        # The per-step model mask is what makes replay exact: with a different STOP mask
        # the recomputed log-probability would not match the one that generated the action.
        batched_model = torch.as_tensor(
            self.model_masks[index], dtype=torch.bool, device=self.device
        )
        masks = MaskBatch(
            model=batched_model.any(dim=0),
            preprocessing_table=self.mask_tables.preprocessing_table,
            preset_table=self.mask_tables.preset_table,
            feature_selection_table=self.mask_tables.feature_selection_table,
            batched_model=batched_model,
        )
        return RolloutBatch(
            states=states,
            actions=actions,
            log_probs=torch.as_tensor(
                self.log_probs[index], dtype=torch.float32, device=self.device
            ),
            values=torch.as_tensor(self.values[index], dtype=torch.float32, device=self.device),
            advantages=torch.as_tensor(
                self.advantages[index], dtype=torch.float32, device=self.device
            ),
            returns=torch.as_tensor(self.returns[index], dtype=torch.float32, device=self.device),
            masks=masks,
        )

    def reset(self) -> None:
        self._reset_arrays()

    def summary(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "head_sizes": list(self.head_sizes),
            "mean_reward": float(self.rewards[: self._cursor].mean()) if self._cursor else 0.0,
        }


__all__ = ["ActionBatch", "MaskBatch", "RolloutBatch", "RolloutBuffer"]
