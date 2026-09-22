from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from rl_automl.agent.agent import PPOAgent
from rl_automl.core.errors import PolicyLoadError
from rl_automl.core.types import TaskType
from rl_automl.environment.action_space import build_action_space
from rl_automl.environment.state_encoder import StateEncoder
from rl_automl.search.model_registry import REGISTRY
from rl_automl.training.train_rl import build_corpus, split_corpus


@pytest.mark.rl
def test_ppo_action_smoke_and_checkpoint_guards(tmp_path, config):
    config.rl.hidden_sizes = [16, 16]
    space = build_action_space(
        TaskType.CLASSIFICATION,
        ["logistic_regression", "random_forest"],
        min_experiments_before_stop=1,
    )
    encoder = StateEncoder({spec.key: spec.family for spec in REGISTRY.all_specs()})
    agent = PPOAgent.from_action_space(space, encoder.dim, config.rl, device="cpu", seed=3)
    state = np.zeros(encoder.dim)
    action, log_prob, value = agent.act(state, space.mask_table(n_experiments=0))
    assert action.shape == (space.layout.n_preprocessing + 3,)
    assert np.isfinite([log_prob, value]).all()

    checkpoint = agent.save(tmp_path / "policy.pt", extra={"sentinel": 7})
    loaded, extra = PPOAgent.load(
        checkpoint, action_space=space, state_encoder=encoder, rl_config=config.rl, device="cpu"
    )
    assert loaded.metadata.model_keys == agent.metadata.model_keys
    assert extra == {"sentinel": 7}

    narrow = build_action_space(TaskType.CLASSIFICATION, ["logistic_regression"])
    with pytest.raises(PolicyLoadError, match="incompatible"):
        PPOAgent.load(
            checkpoint,
            action_space=narrow,
            state_encoder=encoder,
            rl_config=config.rl,
            device="cpu",
        )


def test_training_corpus_is_deterministic_and_holdout_is_disjoint(config):
    entries = build_corpus(config, task_types=[TaskType.CLASSIFICATION], only=["iris", "wine"])
    train, holdout = split_corpus(entries)
    assert {item.name for item in train}.isdisjoint(item.name for item in holdout)
    assert sorted(item.name for item in train + holdout) == ["iris", "wine"]
    assert all(len(item.fingerprint.values) == 40 for item in entries)


@pytest.mark.rl
@pytest.mark.slow
def test_ppo_tiny_training_writes_checkpoint_metadata(config):
    """Opt-in training smoke; deliberately tiny but marked slow/RL for default filtering."""
    from rl_automl.training.train_rl import run_training

    config.rl.hidden_sizes = [16, 16]
    config.rl.rollout_steps = 8
    config.rl.minibatch_size = 4
    config.rl.update_epochs = 1
    report = run_training(
        config,
        steps=8,
        task_types=[TaskType.CLASSIFICATION],
        only=["iris", "wine"],
        device="cpu",
        seed=2,
        verbose=False,
    )
    result = report["results"][0]
    assert result["training"]["total_steps"] == 8
    assert result["evaluation"]["environment"] == "simulated"
    assert config.policy_path(TaskType.CLASSIFICATION).is_file()
