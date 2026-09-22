from __future__ import annotations

import numpy as np

from rl_automl.core.config import RewardConfig
from rl_automl.core.types import MetricDirection, TaskType
from rl_automl.dataset.fingerprint import build_fingerprint
from rl_automl.dataset.profiler import profile_dataset
from rl_automl.environment.action_space import STOP_INDEX, build_action_space
from rl_automl.environment.reward import RewardCalculator, novelty_score
from rl_automl.environment.state_encoder import STATE_SCHEMA_VERSION, SearchProgress, StateEncoder
from rl_automl.search.hyperparameters import validate_presets
from rl_automl.search.model_registry import REGISTRY


def test_registry_keys_presets_and_availability_are_consistent():
    assert len(REGISTRY.keys()) >= 15
    assert len(REGISTRY.keys()) == len(set(REGISTRY.keys()))
    for spec in REGISTRY.all_specs():
        assert spec.tasks
        assert spec.n_presets() > 0
        report = validate_presets(spec)
        assert report.ok, (spec.key, report.problems)
        available, reason = spec.is_available()
        assert isinstance(available, bool)
        assert reason is None or isinstance(reason, str)


def test_action_masks_derive_from_registry_and_round_trip(config):
    enabled = ["logistic_regression", "random_forest"]
    space = build_action_space(TaskType.CLASSIFICATION, enabled, min_experiments_before_stop=2)
    assert list(space.layout.model_keys) == enabled
    early = space.mask_table(n_experiments=0)
    later = space.mask_table(n_experiments=2)
    assert not early.model[STOP_INDEX]
    assert later.model[STOP_INDEX]
    for slot in later.experiment_slots():
        spec = space.spec_for_slot(slot)
        assert later.preset[slot].sum() == spec.n_presets(TaskType.CLASSIFICATION)
    action = space.sample_random_action(later, np.random.default_rng(4))
    decoded = space.decode(action)
    assert decoded is not None
    assert space.encode(decoded) == tuple(int(x) for x in action)


def test_state_encoder_dimension_version_and_clipping(binary_frame, config):
    profile = profile_dataset(binary_frame, "outcome", config.dataset)
    fp = build_fingerprint(profile, TaskType.CLASSIFICATION)
    encoder = StateEncoder({spec.key: spec.family for spec in REGISTRY.all_specs()})
    state = encoder.encode(
        fp,
        TaskType.CLASSIFICATION,
        SearchProgress(max_steps=5, n_experiments=99, elapsed_fraction=2, last_gain=np.inf),
    )
    assert encoder.schema_version == STATE_SCHEMA_VERSION
    assert encoder.dim == 74
    assert state.shape == (74,)
    assert np.isfinite(state).all()
    assert np.max(np.abs(state)) <= 1
    assert state[:5].tolist() == [1, 0, 0, 0, 0]


def test_reward_is_monotonic_cost_sensitive_and_novelty_sensitive():
    calc = RewardCalculator(RewardConfig(), MetricDirection.MAXIMIZE, normalize=False)
    weak = calc.compute(score=0.501, best_before=0.5, cost_s=1, novelty=0)
    strong = calc.compute(score=0.8, best_before=0.5, cost_s=1, novelty=0)
    expensive = calc.compute(score=0.8, best_before=0.5, cost_s=60, novelty=0)
    novel = calc.compute(score=0.8, best_before=0.5, cost_s=1, novelty=1)
    assert strong.total > weak.total
    assert expensive.total < strong.total
    assert novel.total > strong.total
    assert calc.compute(score=None, best_before=0.5, cost_s=0, novelty=0, failed=True).total < 0
    assert novelty_score("rf|0|none", []) == 1
    assert novelty_score("rf|0|none", ["rf|0|none"]) == 0
