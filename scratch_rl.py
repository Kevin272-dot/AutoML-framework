"""Smoke check for the RL core (temporary scratch script)."""

from __future__ import annotations

import numpy as np

from rl_automl.agent.agent import PPOAgent
from rl_automl.core.config import AutoMLConfig
from rl_automl.dataset.fingerprint import build_fingerprint
from rl_automl.dataset.loaders import load_source
from rl_automl.dataset.profiler import profile_dataset
from rl_automl.environment.action_space import build_action_space
from rl_automl.environment.base import SearchTracker
from rl_automl.environment.reward import RewardCalculator
from rl_automl.environment.simulator import SimulatorEnv, SimulatorConfig
from rl_automl.environment.state_encoder import StateEncoder
from rl_automl.environment.surrogate import AnalyticPrior, SurrogateModel
from rl_automl.search.model_registry import REGISTRY
from rl_automl.task.task_classifier import TaskClassifier

cfg = AutoMLConfig.load()
cfg.rl.rollout_steps = 256
cfg.rl.total_steps = 2500
cfg.rl.update_epochs = 4
cfg.rl.minibatch_size = 64
cfg.rl.normalize_reward = True

print("== action space ==")
frame, source = load_source("breast_cancer")
profile = profile_dataset(frame, source.target, cfg.dataset)
task = TaskClassifier().classify(
    "Predict whether a patient's diagnosis is malignant.",
    columns=list(frame.columns),
    class_distribution=profile.class_distribution,
)
fingerprint = build_fingerprint(profile, task.task_type)
space = build_action_space(task.task_type, cfg.models.enabled, min_experiments_before_stop=2)
print("  layout head sizes:", space.layout.head_sizes)
print("  model slots:", space.layout.n_model_slots, "| models:", list(space.layout.model_keys))

masks = space.mask_table(n_experiments=0)
print("  mask at step 0:", masks.summary(), "(STOP correctly illegal before min_experiments)")
masks_later = space.mask_table(n_experiments=5)
print("  mask at step 5:", masks_later.summary())

print("\n== decode / encode round trip ==")
rng = np.random.default_rng(0)
for _ in range(3):
    action = space.sample_random_action(masks_later, rng)
    spec = space.decode(action)
    reencoded = space.encode(spec)
    assert list(reencoded) == [int(v) for v in action], (reencoded, action)
    print(f"  {spec.model:<20} preset={spec.preset_index} fs={spec.feature_selection:<12} "
          f"pre={spec.preprocessing} cost={spec.expected_cost.value}")
print("  round trip OK")

print("\n== preprocessing conflict resolution ==")
conflict = np.zeros(space.layout.head_sizes[1], dtype=int)
from rl_automl.core.vocabulary import PREPROCESSING_OPTIONS
conflict[PREPROCESSING_OPTIONS.index("scale_standard")] = 1
conflict[PREPROCESSING_OPTIONS.index("scale_robust")] = 1
spec_lr = REGISTRY.get("logistic_regression")
resolved = space.resolve_preprocessing(conflict, spec_lr)
print("  both scalers requested ->", resolved)
assert "scale_standard" not in resolved and "scale_robust" in resolved

print("\n== state encoder ==")
families = {spec.key: spec.family for spec in REGISTRY.all_specs()}
encoder = StateEncoder(families)
tracker = SearchTracker(
    reward_calculator=RewardCalculator(
        cfg.rl.reward, task.metric_direction, max_experiments=8, patience=cfg.search.patience
    ),
    max_experiments=8,
    n_possible_models=len(space.layout.model_keys),
)
state = encoder.encode(fingerprint, task.task_type, tracker.progress())
print("  state dim:", encoder.dim, "| finite:", bool(np.all(np.isfinite(state))))

print("\n== surrogate prior sanity ==")
prior = AnalyticPrior()
surrogate = SurrogateModel(prior)
for model_key in ("lightgbm", "logistic_regression", "svm", "mlp"):
    spec = REGISTRY.get(model_key)
    p = surrogate.predict(
        task_type=task.task_type, model_key=model_key, family=spec.family,
        fingerprint=np.asarray(fingerprint.values), preset_index=0,
        n_presets=spec.n_presets(task.task_type), preprocessing=["impute_numeric"],
        feature_selection="none",
    )
    print(f"  {model_key:<20} score={p.score:.3f} cost={p.cost_s:6.1f}s fail={p.failure_probability:.3f}")

print("\n== simulator episode ==")
env = SimulatorEnv(
    task_type=task.task_type, fingerprint=fingerprint, action_space=space,
    encoder=encoder, tracker=tracker, surrogate=surrogate,
    config=SimulatorConfig(max_experiments=6, min_experiments_before_stop=2, seed=7),
)
s = env.reset()
rewards = []
for step in range(6):
    action = space.sample_random_action(env.mask_table, rng)
    result = env.step(action)
    rewards.append(round(result.reward, 3))
    if result.done:
        break
print("  rewards:", rewards, "| n_experiments:", env.n_experiments,
      "| best:", None if env.best_score is None else round(env.best_score, 4))

print("\n== PPO training (simulated) ==")
envs = []
for name in ("breast_cancer", "wine", "digits", "iris", "diabetes"):
    f, src = load_source(name)
    prof = profile_dataset(f, src.target, cfg.dataset)
    t = TaskClassifier().classify(
        f"Predict {src.target} from the provided features.", columns=list(f.columns),
        class_distribution=prof.class_distribution,
    )
    if t.task_type != task.task_type:
        continue
    fp = build_fingerprint(prof, t.task_type)
    sp = build_action_space(t.task_type, cfg.models.enabled, min_experiments_before_stop=2)
    enc = StateEncoder(families)
    trk = SearchTracker(
        reward_calculator=RewardCalculator(
            cfg.rl.reward, t.metric_direction, max_experiments=8, patience=cfg.search.patience
        ),
        max_experiments=8, n_possible_models=len(sp.layout.model_keys),
    )
    envs.append(SimulatorEnv(
        task_type=t.task_type, fingerprint=fp, action_space=sp, encoder=enc, tracker=trk,
        surrogate=surrogate,
        config=SimulatorConfig(max_experiments=8, min_experiments_before_stop=2, seed=11),
        n_rows=prof.n_rows, n_cols=prof.n_cols,
    ))
print(f"  {len(envs)} training environments, all same action space: "
      f"{len({e.layout.head_sizes for e in envs}) == 1}")

agent = PPOAgent.from_action_space(space, encoder.dim, cfg.rl, device="cpu", seed=1234)
print("  device:", agent.device, "| params:", agent.describe()["n_parameters"])

report = agent.train(envs, total_steps=cfg.rl.total_steps)
print("  steps:", report.total_steps, "updates:", report.n_updates, "episodes:", report.n_episodes)
print("  elapsed:", round(report.elapsed_s, 1), "s | mean episode return:", round(report.mean_episode_return, 3))
print("  explained variance (first/last):",
      round(report.explained_variance[0], 3), "->", round(report.explained_variance[-1], 3))
first_losses = [u["total_loss"] for u in report.update_history[:4]]
last_losses = [u["total_loss"] for u in report.update_history[-4:]]
print("  loss first 4:", [round(v, 3) for v in first_losses])
print("  loss last 4: ", [round(v, 3) for v in last_losses])
print("  all finite:", all(np.isfinite(v) for v in report.explained_variance))
print("  entropy coef:", round(report.update_history[0]["entropy_coef"], 4), "->",
      round(report.final_entropy_coef, 4))

print("\n== learning check: policy vs random on a held-out dataset ==")
f, src = load_source("wine")
prof = profile_dataset(f, src.target, cfg.dataset)
t = TaskClassifier().classify("Predict cultivar.", columns=list(f.columns),
                              class_distribution=prof.class_distribution)
fp = build_fingerprint(prof, t.task_type)
sp = build_action_space(t.task_type, cfg.models.enabled, min_experiments_before_stop=2)
enc = StateEncoder(families)


def run_policy(policy_agent, deterministic):
    trk = SearchTracker(
        reward_calculator=RewardCalculator(cfg.rl.reward, t.metric_direction, max_experiments=8),
        max_experiments=8, n_possible_models=len(sp.layout.model_keys),
    )
    e = SimulatorEnv(task_type=t.task_type, fingerprint=fp, action_space=sp, encoder=enc,
                     tracker=trk, surrogate=surrogate,
                     config=SimulatorConfig(max_experiments=8, min_experiments_before_stop=2, seed=99))
    st = e.reset()
    for _ in range(8):
        a, _, _ = policy_agent.act(st, e.mask_table, deterministic=deterministic)
        out = e.step(a)
        st = out.state
        if out.done:
            break
    return e.best_score


trained_scores = [run_policy(agent, True) for _ in range(5)]
untrained = PPOAgent.from_action_space(space, encoder.dim, cfg.rl, device="cpu", seed=5)
untrained_scores = [run_policy(untrained, True) for _ in range(5)]
random_scores = []
for seed in range(5):
    local_rng = np.random.default_rng(seed)
    trk = SearchTracker(
        reward_calculator=RewardCalculator(cfg.rl.reward, t.task_direction if hasattr(t, "task_direction") else t.metric_direction, max_experiments=8),
        max_experiments=8, n_possible_models=len(sp.layout.model_keys),
    )
    e = SimulatorEnv(task_type=t.task_type, fingerprint=fp, action_space=sp, encoder=enc,
                     tracker=trk, surrogate=surrogate,
                     config=SimulatorConfig(max_experiments=8, min_experiments_before_stop=2, seed=seed))
    st = e.reset()
    for _ in range(8):
        a = sp.sample_random_action(e.mask_table, local_rng)
        out = e.step(a)
        st = out.state
        if out.done:
            break
    random_scores.append(e.best_score)

print("  trained   :", [None if s is None else round(s, 4) for s in trained_scores],
      "mean:", round(float(np.mean([s for s in trained_scores if s is not None])), 4))
print("  untrained :", [None if s is None else round(s, 4) for s in untrained_scores],
      "mean:", round(float(np.mean([s for s in untrained_scores if s is not None])), 4))
print("  random    :", [None if s is None else round(s, 4) for s in random_scores],
      "mean:", round(float(np.mean([s for s in random_scores if s is not None])), 4))

print("\n== save / load + version guard ==")
path = cfg.artifacts_dir() / "policies" / "smoke_policy.pt"
agent.save(path, extra={"reward": tracker.reward_calculator.state_dict()})
loaded, extra = PPOAgent.load(path, action_space=space, state_encoder=encoder, rl_config=cfg.rl, device="cpu")
print("  loaded metadata:", loaded.metadata.to_dict())
print("  extra keys:", sorted(extra))

from rl_automl.core.errors import PolicyLoadError

try:
    narrow = build_action_space(task.task_type, ["logistic_regression", "random_forest"])
    PPOAgent.load(path, action_space=narrow, state_encoder=encoder, rl_config=cfg.rl, device="cpu")
    print("  FAIL: incompatible policy was accepted")
except PolicyLoadError as exc:
    print("  guard OK:", str(exc)[:110])

print("\nRL SMOKE OK")
