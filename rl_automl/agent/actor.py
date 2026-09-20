"""Actor: the factored, masked policy (spec §7).

Four heads share one trunk. Head 0 (model, including STOP) is sampled first; heads 1-3 are
then sampled with the masks belonging to that model slot. That makes the policy genuinely
autoregressive over the hierarchy -- the preprocessing options and preset offered for
LightGBM are not the ones offered for SVM.

Action masking is applied by setting disallowed logits to a large negative value rather
than by renormalising probabilities. This is the standard masked-softmax formulation: it
preserves the gradient for legal actions and gives ~0 probability to illegal ones. The one
danger case -- a row with every action masked -- is handled explicitly, because a silent
``NaN`` there would poison the whole update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl_automl.agent.rollout_buffer import ActionBatch, MaskBatch

NEGATIVE_INFINITY = -1e9


class MlpTrunk(nn.Module):
    """Shared feature extractor for the actor and the critic."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for size in hidden_sizes:
            layers.append(nn.Linear(previous, int(size)))
            layers.append(_activation(activation))
            previous = int(size)
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = previous

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(state)


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    return nn.Tanh()


def masked_logits(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Apply a boolean mask. Rows with no legal action fall back to unmasked."""
    if mask is None:
        return logits
    mask = mask.bool()
    if mask.shape != logits.shape:
        mask = mask.expand_as(logits)

    fully_masked = ~mask.any(dim=-1, keepdim=True)
    if bool(fully_masked.any()):
        # Never produce NaN: fall back to unmasked logits for those rows.
        mask = torch.where(fully_masked, torch.ones_like(mask), mask)

    return logits.masked_fill(~mask, NEGATIVE_INFINITY)


@dataclass
class ActionSample:
    actions: ActionBatch
    log_prob: torch.Tensor  # (B,)
    entropy: torch.Tensor  # (B,)


class Actor(nn.Module):
    def __init__(
        self,
        trunk: MlpTrunk,
        head_sizes: tuple[int, ...],
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.head_sizes = tuple(int(size) for size in head_sizes)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(trunk.output_dim, head_hidden),
                    nn.Tanh(),
                    nn.Linear(head_hidden, size),
                )
                for size in self.head_sizes
            ]
        )

    def forward(self, state: torch.Tensor) -> list[torch.Tensor]:
        features = self.trunk(state)
        return [head(features) for head in self.heads]

    # -- sampling ----------------------------------------------------------------

    def act(self, state: torch.Tensor, masks: MaskBatch) -> ActionSample:
        """Sample a factored action, masking each head with its own legal set."""
        logits = self.forward(state)
        batch_size = state.shape[0]

        model_dist = Categorical(logits=masked_logits(logits[0], masks.model_mask(batch_size)))
        model = model_dist.sample()

        preprocessing_mask = masks.preprocessing_for(model)
        preprocessing_logits = masked_logits(logits[1], preprocessing_mask)
        # The multi-binary head samples each toggle independently; a Bernoulli per toggle
        # is the right factorisation and keeps the masks per-toggle.
        preprocessing = torch.bernoulli(torch.sigmoid(preprocessing_logits))
        preprocessing_log_prob = _bernoulli_log_prob(preprocessing_logits, preprocessing)

        preset_dist = Categorical(logits=masked_logits(logits[2], masks.preset_for(model)))
        preset = preset_dist.sample()

        feature_dist = Categorical(
            logits=masked_logits(logits[3], masks.feature_selection_for(model))
        )
        feature_selection = feature_dist.sample()

        log_prob = (
            model_dist.log_prob(model)
            + preprocessing_log_prob
            + preset_dist.log_prob(preset)
            + feature_dist.log_prob(feature_selection)
        )
        entropy = (
            model_dist.entropy()
            + _bernoulli_entropy(preprocessing_logits)
            + preset_dist.entropy()
            + feature_dist.entropy()
        )

        return ActionSample(
            actions=ActionBatch(
                model=model,
                preprocessing=preprocessing,
                preset=preset,
                feature_selection=feature_selection,
            ),
            log_prob=log_prob,
            entropy=entropy,
        )

    def greedy(self, state: torch.Tensor, masks: MaskBatch) -> ActionSample:
        """Deterministic argmax policy. Used for the recommendation phase."""
        logits = self.forward(state)
        batch_size = state.shape[0]

        model = torch.argmax(masked_logits(logits[0], masks.model_mask(batch_size)), dim=-1)
        preprocessing_mask = masks.preprocessing_for(model)
        preprocessing_logits = masked_logits(logits[1], preprocessing_mask)
        preprocessing = (torch.sigmoid(preprocessing_logits) >= 0.5).float()
        preset = torch.argmax(masked_logits(logits[2], masks.preset_for(model)), dim=-1)
        feature_selection = torch.argmax(
            masked_logits(logits[3], masks.feature_selection_for(model)), dim=-1
        )

        return ActionSample(
            actions=ActionBatch(
                model=model,
                preprocessing=preprocessing,
                preset=preset,
                feature_selection=feature_selection,
            ),
            log_prob=torch.zeros(batch_size, device=state.device),
            entropy=torch.zeros(batch_size, device=state.device),
        )

    # -- evaluation --------------------------------------------------------------

    def evaluate_actions(
        self, state: torch.Tensor, actions: ActionBatch, masks: MaskBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute log-probabilities and entropy for a stored action.

        The masks are rebuilt from the stored model indices, so the recomputed
        log-probability is over exactly the same conditional distribution that produced
        the action.
        """
        logits = self.forward(state)
        batch_size = state.shape[0]

        model_dist = Categorical(logits=masked_logits(logits[0], masks.model_mask(batch_size)))
        preprocessing_logits = masked_logits(logits[1], masks.preprocessing_for(actions.model))
        preset_dist = Categorical(logits=masked_logits(logits[2], masks.preset_for(actions.model)))
        feature_dist = Categorical(
            logits=masked_logits(logits[3], masks.feature_selection_for(actions.model))
        )

        log_prob = (
            model_dist.log_prob(actions.model)
            + _bernoulli_log_prob(preprocessing_logits, actions.preprocessing)
            + preset_dist.log_prob(actions.preset)
            + feature_dist.log_prob(actions.feature_selection)
        )
        entropy = (
            model_dist.entropy()
            + _bernoulli_entropy(preprocessing_logits)
            + preset_dist.entropy()
            + feature_dist.entropy()
        )
        return log_prob, entropy


def _bernoulli_log_prob(logits: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Sum of per-toggle Bernoulli log-probabilities for the multi-binary head."""
    log_prob = -torch.nn.functional.binary_cross_entropy_with_logits(
        logits, values, reduction="none"
    )
    return log_prob.sum(dim=-1)


def _bernoulli_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    entropy = -(
        probabilities * torch.log(probabilities + 1e-8)
        + (1 - probabilities) * torch.log(1 - probabilities + 1e-8)
    )
    return entropy.sum(dim=-1)


__all__ = ["NEGATIVE_INFINITY", "ActionSample", "Actor", "MlpTrunk", "masked_logits"]
