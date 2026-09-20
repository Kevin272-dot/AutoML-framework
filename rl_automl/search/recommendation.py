"""Recommendation engine - Phase A (spec §9).

This is the module that makes the two-phase design real. It answers *"which experiments are
worth running?"* and it does so **without training a single model**. Everything it reports
comes from the surrogate's expected-value estimates, which cost microseconds.

Candidate plans come from one of two planners:

* ``rl-ppo`` - a trained policy is rolled out over a simulated search, and the experiments
  it proposes are collected. Because planning runs through the simulator, the policy sees
  realistic search states without any real training happening.
* ``surrogate-greedy`` - no policy available yet (or RL disabled), so every legal
  experiment is scored by the surrogate and the best are taken. This is what makes the
  system usable before RL training has been run, and it is labelled as such rather than
  dressed up as a learned recommendation.

Either way the candidate pool is ranked by expected primary metric with cost as the
tie-break, and the output always carries ``requires_user_approval: true``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_automl.core.config import SearchConfig
from rl_automl.core.logging import get_logger
from rl_automl.core.metrics import orient
from rl_automl.core.types import (
    CostTier,
    DatasetFingerprint,
    DatasetProfile,
    ExperimentSpec,
    RecommendationSet,
    RecommendedExperiment,
    SearchStatistics,
    TaskSpec,
)
from rl_automl.environment.action_space import ActionSpace
from rl_automl.environment.simulator import SimulatorConfig, SimulatorEnv
from rl_automl.environment.state_encoder import StateEncoder
from rl_automl.environment.surrogate import SurrogateModel
from rl_automl.search.hyperparameters import HyperparameterRefiner
from rl_automl.search.model_registry import get_spec

logger = get_logger("search.recommendation")


@dataclass
class RecommendationConfig:
    n_recommendations: int = 3
    min_recommendations: int = 2
    horizon: int = 6
    rollouts: int = 64
    include_refinement: bool = False
    refinement_trials: int = 8
    cost_reference_s: float = 60.0
    max_experiments: int = 12


@dataclass
class PlannedExperiment:
    """A proposed experiment plus its surrogate-evaluated expectation."""

    experiment: ExperimentSpec
    expected_performance: float
    expected_cost_s: float
    failure_probability: float
    origin: str = "surrogate"
    uncertainty: float = 0.0

    @property
    def oriented_performance(self) -> float:
        return self.expected_performance


class RecommendationEngine:
    def __init__(
        self,
        *,
        task: TaskSpec,
        action_space: ActionSpace,
        state_encoder: StateEncoder,
        surrogate: SurrogateModel,
        profile: DatasetProfile | None = None,
        search_config: SearchConfig | None = None,
        config: RecommendationConfig | None = None,
        target_score: float | None = None,
    ) -> None:
        self.task = task
        self.action_space = action_space
        self.encoder = state_encoder
        self.surrogate = surrogate
        self.profile = profile
        self.search_config = search_config or SearchConfig()

        self.config = config or RecommendationConfig(
            n_recommendations=self.search_config.n_recommendations,
            horizon=self.search_config.recommendation_horizon,
            rollouts=self.search_config.recommendation_rollouts,
            include_refinement=self.search_config.post_selection_refinement,
            refinement_trials=self.search_config.refinement_trials,
            cost_reference_s=60.0,
            max_experiments=self.search_config.max_experiments,
        )
        self.target_score = target_score

    # -- public ------------------------------------------------------------------

    def recommend(
        self,
        *,
        fingerprint: DatasetFingerprint,
        agent: Any | None = None,
        n_rows: int = 1000,
        n_cols: int = 10,
    ) -> RecommendationSet:
        """Produce the plan-phase recommendation. Trains nothing, ever."""
        candidates: dict[str, PlannedExperiment] = {}
        planner = "surrogate-greedy"
        notes: list[str] = []

        if agent is not None:
            policy_candidates = self._plan_with_policy(agent, fingerprint, n_rows, n_cols)
            if policy_candidates:
                planner = "rl-ppo"
                for candidate in policy_candidates:
                    _keep_best(candidates, candidate)
            else:
                notes.append(
                    "the policy proposed no usable experiments; fell back to the surrogate"
                )

        # Always enrich with the surrogate's own view of the space. When RL is driving this
        # acts as the safety net that guarantees a plan even if the policy stops immediately.
        for candidate in self._plan_with_surrogate(fingerprint, n_rows, n_cols):
            _keep_best(candidates, candidate)

        if self.config.include_refinement and candidates:
            for candidate in self._refine(list(candidates.values()), fingerprint, n_rows, n_cols):
                _keep_best(candidates, candidate)

        ranked = self._rank(list(candidates.values()))
        selected = self._select_diverse(ranked, self._recommended_count(ranked))
        if not selected:
            raise RuntimeError(
                "no experiment could be proposed; check that the registry has models "
                "available for this task"
            )
        if len(selected) < self.config.min_recommendations:
            notes.append(f"only {len(selected)} distinct experiment(s) were available to recommend")

        recommended = [
            RecommendedExperiment(
                rank=index + 1,
                model=planned.experiment.model,
                preprocessing=list(planned.experiment.preprocessing),
                feature_selection=planned.experiment.feature_selection,
                preset_index=planned.experiment.preset_index,
                hyperparameters=dict(planned.experiment.hyperparameters),
                reason=self._explain(planned, index + 1),
                expected_cost=_cost_tier(planned.expected_cost_s, self.config.cost_reference_s),
                expected_performance=round(planned.expected_performance, 6),
                experiment=planned.experiment.model_copy(
                    update={"rationale": self._explain(planned, index + 1)}
                ),
            )
            for index, planned in enumerate(selected)
        ]

        total_cost = sum(planned.expected_cost_s for planned in selected)
        return RecommendationSet(
            task=self.task.task_type,
            metric=self.task.metric,
            metric_direction=self.task.metric_direction,
            recommended_experiments=recommended,
            estimated_experiments=len(recommended),
            estimated_compute=self._describe_compute(selected, total_cost, n_rows),
            estimated_runtime_s=round(total_cost, 2),
            estimated_cost_tier=_cost_tier(
                total_cost / max(len(selected), 1), self.config.cost_reference_s
            ),
            requires_user_approval=True,
            planner=planner,
            search_statistics=self._search_statistics(selected, candidates),
            baseline_expectation=self._baseline_expectation(fingerprint, selected, n_rows, n_cols),
            notes=notes,
        )

    # -- planners ----------------------------------------------------------------

    def _plan_with_policy(
        self,
        agent: Any,
        fingerprint: DatasetFingerprint,
        n_rows: int,
        n_cols: int,
    ) -> list[PlannedExperiment]:
        """Roll the policy out through the simulator and collect what it proposes."""
        horizon = max(1, self.config.horizon)
        n_episodes = max(1, self.config.rollouts // horizon)
        collected: dict[str, PlannedExperiment] = {}

        for episode in range(n_episodes):
            env = self._planning_env(fingerprint, n_rows, n_cols, seed=1000 + episode)
            state = env.reset()
            for _ in range(horizon):
                action, _log_prob, _value = agent.act(state, env.mask_table, deterministic=False)
                spec = self.action_space.decode(action)
                if spec is not None:
                    planned = self._evaluate_candidate(spec, fingerprint, n_rows, n_cols, "policy")
                    _keep_best(collected, planned)

                result = env.step(action)
                state = result.state
                if result.done:
                    break

        return list(collected.values())

    def _plan_with_surrogate(
        self,
        fingerprint: DatasetFingerprint,
        n_rows: int,
        n_cols: int,
    ) -> list[PlannedExperiment]:
        """Score every legal experiment with the surrogate and keep the best."""
        masks = self.action_space.mask_table(n_experiments=self.search_config.min_experiments)
        specs = self.action_space.all_legal_specs(masks)
        planned = [
            self._evaluate_candidate(spec, fingerprint, n_rows, n_cols, "surrogate")
            for spec in specs
        ]
        planned.sort(key=lambda item: item.expected_performance, reverse=True)

        # The top of the ranking is dominated by whichever family scores highest, so a
        # purely size-based cut can hand the diversity selector a pool made of a single
        # model. Every model contributes its best configuration, which is what makes a
        # "pick k distinct algorithms" pass possible in the first place.
        pool_size = max(self.config.n_recommendations * 4, 12)
        pool = planned[:pool_size]
        pooled_models = {item.experiment.model for item in pool}
        for candidate in planned[pool_size:]:
            if candidate.experiment.model in pooled_models:
                continue
            pool.append(candidate)
            pooled_models.add(candidate.experiment.model)
        pool.sort(key=lambda item: item.expected_performance, reverse=True)
        return pool

    def _refine(
        self,
        base_candidates: list[PlannedExperiment],
        fingerprint: DatasetFingerprint,
        n_rows: int,
        n_cols: int,
    ) -> list[PlannedExperiment]:
        """Optional randomized refinement around the strongest candidates."""
        refiner = HyperparameterRefiner(
            self.search_config, seed=self.search_config.__dict__.get("seed", 0) or 0
        )
        best = max(base_candidates, key=lambda item: item.expected_performance)
        proposals = refiner.refine(best.experiment, trials=self.config.refinement_trials)
        return [
            self._evaluate_candidate(spec, fingerprint, n_rows, n_cols, "refinement")
            for spec in proposals
        ]

    # -- scoring -----------------------------------------------------------------

    def _evaluate_candidate(
        self,
        spec: ExperimentSpec,
        fingerprint: DatasetFingerprint,
        n_rows: int,
        n_cols: int,
        origin: str,
    ) -> PlannedExperiment:
        model_spec = get_spec(spec.model)
        prediction = self.surrogate.predict(
            task_type=self.task.task_type,
            model_key=spec.model,
            family=model_spec.family,
            fingerprint=np.asarray(fingerprint.values, dtype=np.float64),
            preset_index=spec.preset_index,
            n_presets=model_spec.n_presets(self.task.task_type),
            preprocessing=spec.preprocessing,
            feature_selection=spec.feature_selection,
            preset_params=spec.hyperparameters,
            preset_cost=model_spec.preset(spec.preset_index, self.task.task_type).cost,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        # Oriented so that "higher is better" holds for every metric, including RMSE.
        oriented = orient(prediction.score, self.task.metric_direction)
        return PlannedExperiment(
            experiment=spec,
            expected_performance=float(oriented),
            expected_cost_s=prediction.cost_s,
            failure_probability=prediction.failure_probability,
            origin=origin,
            uncertainty=prediction.uncertainty,
        )

    def _rank(self, candidates: list[PlannedExperiment]) -> list[PlannedExperiment]:
        """Expected performance first, then cheapness, then lower failure risk."""
        return sorted(
            candidates,
            key=lambda item: (
                -item.expected_performance,
                item.expected_cost_s,
                item.failure_probability,
            ),
        )

    def _select_diverse(self, ranked: list[PlannedExperiment], k: int) -> list[PlannedExperiment]:
        """Pick ``k`` candidates, preferring distinct algorithms.

        Recommending the same model three times with different preprocessing gives the user
        nothing to choose between. The first pass takes the best candidate from each
        distinct model, which is what makes a recommendation list look like the spec's
        example (LightGBM, XGBoost, Extra Trees); a second pass fills any remaining slots
        with the next best candidates regardless of model, so a registry with fewer than
        ``k`` usable entries still produces a full plan.
        """
        if k <= 0:
            return []

        selected: list[PlannedExperiment] = []
        chosen_signatures: set[str] = set()
        seen_models: set[str] = set()

        for candidate in ranked:
            if len(selected) >= k:
                break
            if candidate.experiment.model in seen_models:
                continue
            selected.append(candidate)
            chosen_signatures.add(candidate.experiment.signature())
            seen_models.add(candidate.experiment.model)

        for candidate in ranked:
            if len(selected) >= k:
                break
            signature = candidate.experiment.signature()
            if signature in chosen_signatures:
                continue
            selected.append(candidate)
            chosen_signatures.add(signature)

        return selected

    def _recommended_count(self, ranked: list[PlannedExperiment]) -> int:
        """Top-k, or fewer if the target score is already expected to be reached."""
        limit = max(self.config.n_recommendations, self.config.min_recommendations)
        if self.target_score is not None:
            for index, planned in enumerate(ranked[:limit]):
                if planned.expected_performance >= self.target_score:
                    return max(self.config.min_recommendations, index + 1)
        return limit

    # -- explanation -------------------------------------------------------------

    def _explain(self, planned: PlannedExperiment, rank: int) -> str:
        spec = get_spec(planned.experiment.model)
        preset = spec.preset(planned.experiment.preset_index, self.task.task_type)
        metric = self.task.metric
        parts = [
            f"expected {metric} ~ {abs(planned.expected_performance):.3f} "
            f"using the '{preset.name}' preset"
        ]

        tier = _cost_tier(planned.expected_cost_s, self.config.cost_reference_s)
        parts.append(f"{tier.value} cost (~{planned.expected_cost_s:.0f}s of training)")

        if self.profile is not None:
            parts.extend(self._dataset_reasons(spec, planned))

        if planned.failure_probability > 0.15:
            parts.append(
                f"note: {planned.failure_probability:.0%} chance of a resource failure at "
                "this dataset size"
            )
        if planned.origin == "refinement":
            parts.append("hyperparameters refined around the best preset")
        return "; ".join(parts)

    def _dataset_reasons(self, spec: Any, planned: PlannedExperiment) -> list[str]:
        """Dataset-aware justification, which is the point of a fingerprint-conditioned planner."""
        reasons: list[str] = []
        profile = self.profile
        if profile is None:
            return reasons

        distribution = profile.class_distribution
        if (
            distribution is not None
            and distribution.is_imbalanced
            and spec.family in ("boosting", "bagging", "ensemble")
        ):
            minority = distribution.minority_ratio or 0.0
            reasons.append(
                f"tree ensembles usually hold up well on the {minority:.0%} minority class"
            )

        if profile.high_cardinality_features and spec.supports_native_categorical:
            reasons.append(
                f"handles the {len(profile.high_cardinality_features)} high-cardinality "
                "categorical feature(s) natively"
            )
        elif profile.high_cardinality_features and spec.family in ("linear", "kernel"):
            reasons.append("one-hot expansion of high-cardinality features may hurt this family")

        if profile.missing_ratio > 0.01 and spec.family in ("boosting", "bagging"):
            reasons.append(f"tolerates the {profile.missing_ratio:.1%} missing values")

        if profile.n_rows < 2000 and spec.family in ("linear", "kernel"):
            reasons.append("low-variance choice for a small dataset")
        if profile.n_rows > 100_000 and spec.family in ("kernel", "hierarchical"):
            reasons.append("may be slow at this row count")

        if profile.max_abs_correlation > 0.95 and planned.experiment.feature_selection != "none":
            reasons.append("feature selection should help with the near-duplicate features")

        return reasons

    # -- summary fields ----------------------------------------------------------

    def _describe_compute(
        self, selected: list[PlannedExperiment], total_cost_s: float, n_rows: int
    ) -> str:
        models = ", ".join(planned.experiment.model for planned in selected)
        return (
            f"{len(selected)} experiment(s) on {n_rows:,} rows - "
            f"estimated {total_cost_s:.0f}s of training total ({models})"
        )

    def _search_statistics(
        self,
        selected: list[PlannedExperiment],
        pool: dict[str, PlannedExperiment],
    ) -> SearchStatistics:
        """Note ``n_experiments = 0``: planning trains nothing, by construction."""
        best = selected[0] if selected else None
        return SearchStatistics(
            searcher="rl-ppo-planning",
            n_experiments=0,
            n_failed=0,
            n_distinct_models=len({planned.experiment.model for planned in pool.values()}),
            best_validation_score=best.expected_performance if best else None,
            best_model=best.experiment.model if best else None,
            total_training_time_s=0.0,
            total_compute_s=0.0,
            stopping_reason="planning_complete",
            history=[
                {
                    "step": index + 1,
                    "model": planned.experiment.model,
                    "origin": planned.origin,
                    "expected_score": round(planned.expected_performance, 5),
                    "expected_cost_s": round(planned.expected_cost_s, 3),
                }
                for index, planned in enumerate(selected)
            ],
        )

    def _baseline_expectation(
        self,
        fingerprint: DatasetFingerprint,
        selected: list[PlannedExperiment],
        n_rows: int,
        n_cols: int,
    ) -> str:
        """Say what a naive search of the same size would be expected to achieve.

        Deliberately phrased as an expectation, and computed from the same surrogate, so it
        can never imply that RL was better than random search on real data (spec §30).
        """
        if not selected:
            return ""
        masks = self.action_space.mask_table(n_experiments=self.search_config.min_experiments)
        specs = self.action_space.all_legal_specs(masks)
        if not specs:
            return ""

        rng = np.random.default_rng(0)
        n_draws = max(len(selected) * 8, 24)
        indices = rng.integers(0, len(specs), size=n_draws)
        sampled = [
            self._evaluate_candidate(specs[index], fingerprint, n_rows, n_cols, "baseline")
            for index in indices
        ]
        random_mean = float(np.mean([item.expected_performance for item in sampled]))
        random_best = max(item.expected_performance for item in sampled)
        recommended_mean = float(np.mean([item.expected_performance for item in selected]))

        return (
            f"for reference, {n_draws} uniformly sampled experiments in the same space would "
            f"be expected to average {abs(random_mean):.3f} {self.task.metric} "
            f"(best single draw {abs(random_best):.3f}); the recommended set averages "
            f"{abs(recommended_mean):.3f}. These are surrogate estimates, not measurements."
        )

    # -- helpers -----------------------------------------------------------------

    def _planning_env(
        self,
        fingerprint: DatasetFingerprint,
        n_rows: int,
        n_cols: int,
        seed: int,
    ) -> SimulatorEnv:
        """A simulator used purely for planning. It never trains anything."""
        from rl_automl.environment.base import SearchTracker
        from rl_automl.environment.reward import RewardCalculator

        tracker = SearchTracker(
            reward_calculator=RewardCalculator(
                direction=self.task.metric_direction,
                max_experiments=self.config.max_experiments,
                patience=self.search_config.patience,
                target_score=self.target_score,
            ),
            max_experiments=self.config.max_experiments,
            n_possible_models=len(self.action_space.layout.model_keys),
        )
        return SimulatorEnv(
            task_type=self.task.task_type,
            fingerprint=fingerprint,
            action_space=self.action_space,
            encoder=self.encoder,
            tracker=tracker,
            surrogate=self.surrogate,
            config=SimulatorConfig(
                max_experiments=self.config.max_experiments,
                min_experiments_before_stop=self.search_config.min_experiments,
                seed=seed,
            ),
            n_rows=n_rows,
            n_cols=n_cols,
        )


def _keep_best(pool: dict[str, PlannedExperiment], candidate: PlannedExperiment) -> None:
    """De-duplicate by experiment signature, keeping the most optimistic expectation."""
    signature = candidate.experiment.signature()
    existing = pool.get(signature)
    if existing is None or candidate.expected_performance > existing.expected_performance:
        pool[signature] = candidate


def _cost_tier(cost_s: float, reference_s: float) -> CostTier:
    """Turn an absolute expected training time into the low/medium/high tier the UI shows."""
    reference_s = max(reference_s, 1e-6)
    ratio = cost_s / reference_s
    if ratio <= 0.35:
        return CostTier.LOW
    if ratio <= 1.5:
        return CostTier.MEDIUM
    return CostTier.HIGH


def build_recommendation_engine(
    *,
    task: TaskSpec,
    action_space: ActionSpace,
    state_encoder: StateEncoder,
    surrogate: SurrogateModel,
    profile: DatasetProfile | None = None,
    search_config: SearchConfig | None = None,
    target_score: float | None = None,
) -> RecommendationEngine:
    return RecommendationEngine(
        task=task,
        action_space=action_space,
        state_encoder=state_encoder,
        surrogate=surrogate,
        profile=profile,
        search_config=search_config,
        target_score=target_score,
    )


__all__ = [
    "PlannedExperiment",
    "RecommendationConfig",
    "RecommendationEngine",
    "build_recommendation_engine",
]
