"""Reward shaping (spec §6).

The agent is rewarded for *making the search better*, not for picking a universally good
model. The reward decomposes into interpretable terms:

======================  ====================================================
performance             improvement in the primary metric over the best so far
cost                    penalty proportional to training time (spec §6, §26)
novelty                 small bonus for trying something genuinely different
stagnation              penalty once recent experiments stop improving
failure                 penalty for a pipeline that failed or was invalid
terminal                bonus on finishing, and a larger one for hitting the target
======================  ====================================================

Scores are oriented (larger is always better) before they enter the reward, so the same
weights work for F1 and RMSE. The combined reward is optionally normalised with a running
mean/std that is persisted alongside the policy, which keeps PPO's value targets stable
across datasets with very different metric scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_automl.core.config import RewardConfig
from rl_automl.core.metrics import orient
from rl_automl.core.types import MetricDirection


@dataclass
class RewardBreakdown:
    total: float
    performance: float = 0.0
    cost: float = 0.0
    novelty: float = 0.0
    stagnation: float = 0.0
    penalty: float = 0.0
    terminal: float = 0.0
    raw_total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "total": round(self.total, 5),
            "raw_total": round(self.raw_total, 5),
            "performance": round(self.performance, 5),
            "cost": round(self.cost, 5),
            "novelty": round(self.novelty, 5),
            "stagnation": round(self.stagnation, 5),
            "penalty": round(self.penalty, 5),
            "terminal": round(self.terminal, 5),
        }


class RewardNormalizer:
    """Welford running normaliser, serialisable with the policy.

    PPO is sensitive to reward scale; a dataset scored by RMSE and one scored by F1 differ
    by orders of magnitude. Normalising with running statistics keeps the advantage
    estimates comparable across both.
    """

    def __init__(self, clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.clip = clip
        self.epsilon = epsilon

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return float(np.sqrt(self.m2 / (self.count - 1))) or 1.0

    def normalize(self, value: float) -> float:
        if self.count < 2:
            return value
        scaled = (value - self.mean) / (self.std + self.epsilon)
        return float(np.clip(scaled, -self.clip, self.clip))

    def state_dict(self) -> dict[str, float]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.count = int(payload.get("count", 0))
        self.mean = float(payload.get("mean", 0.0))
        self.m2 = float(payload.get("m2", 0.0))


class RewardCalculator:
    def __init__(
        self,
        config: RewardConfig | None = None,
        direction: MetricDirection = MetricDirection.MAXIMIZE,
        *,
        max_experiments: int = 12,
        patience: int = 4,
        cost_reference_s: float = 60.0,
        target_score: float | None = None,
        normalize: bool = True,
    ) -> None:
        self.config = config or RewardConfig()
        self.direction = direction
        self.max_experiments = max(1, max_experiments)
        self.patience = max(1, patience)
        self.cost_reference_s = max(cost_reference_s, 1e-6)
        self.target_score = target_score
        self.normalizer = RewardNormalizer()
        self.normalize = normalize

    # -- scoring -----------------------------------------------------------------

    def compute(
        self,
        *,
        score: float | None,
        best_before: float | None,
        cost_s: float,
        novelty: float,
        stagnation: int = 0,
        failed: bool = False,
        invalid: bool = False,
        terminal: bool = False,
    ) -> RewardBreakdown:
        """Compute the shaped reward for one experiment.

        ``score`` and ``best_before`` are *oriented* (larger is better). ``None`` means the
        pipeline produced no usable metric.
        """
        cfg = self.config

        if invalid:
            raw = -cfg.invalid_penalty
            breakdown = RewardBreakdown(total=raw, raw_total=raw, penalty=-cfg.invalid_penalty)
            return self._finalise(breakdown)

        if failed or score is None:
            raw = -cfg.failure_penalty
            breakdown = RewardBreakdown(total=raw, raw_total=raw, penalty=-cfg.failure_penalty)
            if terminal:
                breakdown.terminal = self._terminal_term(best_before)
                breakdown.total += breakdown.terminal
                breakdown.raw_total += breakdown.terminal
            return self._finalise(breakdown)

        # Performance is measured as improvement over the best so far. On the very first
        # experiment there is no baseline, so the absolute quality is used as shaping --
        # otherwise a strong first model and a weak one would be indistinguishable.
        if best_before is None:
            gain = score
        else:
            gain = score - best_before

        performance = cfg.w_perf * _clip(gain / max(cfg.perf_scale, 1e-6), -1.0, 1.0)

        normalized_cost = _clip(cost_s / self.cost_reference_s, 0.0, 1.0)
        cost = -cfg.w_cost * normalized_cost

        novelty_term = cfg.w_novelty * _clip(novelty, 0.0, 1.0)

        stagnation_term = 0.0
        if stagnation >= self.patience:
            # Grows slowly with the length of the stall so the agent is nudged, then
            # pushed, rather than hitting a cliff.
            overshoot = min(stagnation - self.patience + 1, self.patience)
            stagnation_term = -cfg.stagnation_penalty * overshoot

        raw = performance + cost + novelty_term + stagnation_term

        breakdown = RewardBreakdown(
            total=raw,
            raw_total=raw,
            performance=performance,
            cost=cost,
            novelty=novelty_term,
            stagnation=stagnation_term,
        )

        if terminal:
            breakdown.terminal = self._terminal_term(best_before if best_before else score)
            breakdown.total += breakdown.terminal
            breakdown.raw_total += breakdown.terminal

        return self._finalise(breakdown)

    def compute_terminal(
        self,
        *,
        best_score: float | None,
        n_experiments: int,
        stopping_reason: str = "stop",
    ) -> RewardBreakdown:
        """Terminal reward for ending an episode (spec §25)."""
        breakdown = RewardBreakdown(total=0.0, raw_total=0.0)
        if best_score is not None:
            breakdown.terminal = self._terminal_term(best_score)
        # A small bonus for producing a usable recommendation at all.
        breakdown.terminal += (
            self.config.terminal_bonus * 0.1 * _clip(n_experiments / self.max_experiments, 0.0, 1.0)
        )
        breakdown.total = breakdown.terminal
        breakdown.raw_total = breakdown.terminal
        return self._finalise(breakdown)

    def _terminal_term(self, best_score: float | None) -> float:
        bonus = self.config.terminal_bonus
        if best_score is None:
            return -bonus
        if self.target_score is not None and best_score >= self.target_score:
            return bonus * 2.0
        return bonus * 0.5

    def _finalise(self, breakdown: RewardBreakdown) -> RewardBreakdown:
        if self.normalize:
            breakdown.raw_total = breakdown.total
            breakdown.total = self.normalizer.normalize(breakdown.total)
        return breakdown

    def observe(self, raw_total: float) -> None:
        """Feed a raw reward into the running statistics."""
        if self.normalize:
            self.normalizer.update(raw_total)

    def orient_score(self, score: float | None) -> float | None:
        if score is None:
            return None
        return orient(score, self.direction)

    def state_dict(self) -> dict[str, Any]:
        return {
            "normalizer": self.normalizer.state_dict(),
            "direction": self.direction.value,
            "normalize": self.normalize,
            "cost_reference_s": self.cost_reference_s,
            "max_experiments": self.max_experiments,
            "target_score": self.target_score,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.normalizer.load_state_dict(payload.get("normalizer", {}))
        if "direction" in payload:
            self.direction = MetricDirection(payload["direction"])
        self.normalize = bool(payload.get("normalize", self.normalize))
        self.cost_reference_s = float(payload.get("cost_reference_s", self.cost_reference_s))
        self.max_experiments = int(payload.get("max_experiments", self.max_experiments))
        self.target_score = payload.get("target_score", self.target_score)


def novelty_score(signature: str, previous_signatures: list[str] | set[str]) -> float:
    """1.0 for something brand new, decaying toward 0.0 for a near-repeat.

    Uses Jaccard overlap over the signature's components, so "same model, different
    preset" is treated as partially novel rather than either identical or entirely new.
    """
    if not previous_signatures:
        return 1.0
    parts = set(signature.split("|"))
    best_overlap = 0.0
    for previous in previous_signatures:
        other = set(previous.split("|"))
        if not parts or not other:
            continue
        union = len(parts | other)
        if not union:
            continue
        best_overlap = max(best_overlap, len(parts & other) / union)
    return float(max(0.0, 1.0 - best_overlap))


def _clip(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(high, max(low, value)))


__all__ = ["RewardBreakdown", "RewardCalculator", "RewardNormalizer", "novelty_score"]
