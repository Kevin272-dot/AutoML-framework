# CHECKPOINT - Adaptive RL-AutoML System

**Last session:** 2026-09-18
**Branch/state:** greenfield build in `RLAGENT/`. Phases 0 and most of Phase 1 are implemented
and smoke-verified. **One blocking bug** in artifact verification, plus the API/CLI/tests/docs
still to write.
**Nothing is committed to git yet.** No long training run has been started (per your request).

---

## 1. Where things stand

### Verified working today

| Area | Evidence |
|---|---|
| Config layering (YAML + `AUTOML_*` env) | `AutoMLConfig.load()` resolves `configs/default.yaml` |
| Task understanding | 6/6 problem statements resolve to the correct task, metric, direction, target column |
| Dataset profiling + 40-dim fingerprint | mixed-dtype frame profiled; all fingerprint values finite |
| Upload security gate | extension allowlist, magic-byte sniffing, row/col/size caps, column sanitisation |
| Model registry | 17 models, per-task presets, class-balanced presets validated |
| Execution engine | 6 models trained; label encoding for string targets; budget caps; failure isolation |
| Test-set isolation | `evaluate_test` before `begin_finalize` raises `RunStateError` |
| RL core | action masks, decode/encode round trip, 74-dim state, surrogate prior, PPO training |
| PPO learning signal | loss 2.19 -> 0.07, explained variance 0.001 -> 0.53, entropy 0.018 -> 0.002 |
| Policy checkpointing | save/load round trip; version guard rejects a narrowed action space |
| Approval gate | `execute()` without approval raises `NotApprovedError`; bad rank rejected |
| Plan phase writes nothing | 0 `.joblib` files and 0 archives after `plan()` |
| Packaging | `model_package/` tree + generated `model_loader.py`, `predict.py`, README, requirements |
| Archive verification | CRC, zip-slip, required members, size cap, and an end-to-end model reload |

### Directly measured numbers worth remembering

- PPO **does not beat random search** on the surrogate prior (trained 0.999 vs random 0.995
  vs untrained 0.921). This is an honest finding, not a defect - the spec forbids claiming
  superiority without evidence, and the benchmark harness is meant to report exactly this.
- End-to-end run was **fast**: 4 pipelines on 20k rows trained in 8.9s total.

---

## 2. BLOCKING BUG - artifact verification fails (diagnosed, not yet fixed)

`orchestrator.execute()` ends in:

```
status: failed
error: ArtifactIntegrityError: the generated archive failed verification and will not be
       offered for download
```

**Root cause (high confidence):** in `packaging/model_exporter.py`,
`build_model_metadata()` always writes a `label_mapping` whenever `label_classes` is
non-empty. The generated `model_loader._decode()` then converts predictions to **strings**
("0"/"1"), while the probe's expected predictions come from
`orchestrator._expected_predictions()` -> `trainer.predict()`, which returns **raw**
predictions (ints). `zip_builder._prediction_difference()` compares them with
`np.array_equal` -> mismatch -> difference `inf` -> verification fails.

For the churn demo the target is already contiguous `0/1`, so **no** label encoding was
applied and no decoding should happen at all.

**Fix (two parts, both small):**

1. Only emit `label_mapping` when encoding was actually applied. Thread a
   `label_encoding_required: bool` from `PipelineExecutor.label_encoding_required` through
   `ModelExporter.export()` into `build_model_metadata()`, and skip `label_mapping` when it
   is `False`.
2. Make `_prediction_difference()` tolerant: when dtypes differ, compare on
   `astype(str)` before declaring a mismatch.

**Also fix while in there (defect found during the same run):** `ArtifactIntegrityError`
carries `problems` in `.details`, but the orchestrator only records `str(exc)`, so the
actual reason is invisible. Surface `exc.details` into `run.error` / `run.messages` for all
`AutoMLError`s - losing the diagnostic made this bug much harder to read.

**How to reproduce quickly:** run `python scratch_e2e.py`, or call the packaging path
directly and print `archive.verification.problems`.

---

## 3. Known quality issues to address (not blocking)

1. **Preset bonus is index-based, so it privileges list order.**
   `surrogate._preset_adjustment()` uses `_preset_rank(preset_index, n_presets)`, which
   rewards whichever preset happens to be last. Because `class_balanced` was appended last,
   it now wins nearly every recommendation. Fix: derive strength from the preset's declared
   `CostTier` (low=0, medium=0.5, high=1) instead of its position, and/or order presets by
   cost within each table.
2. **Recommendation diversity is limited by the candidate pool.**
   `_select_diverse()` now prefers distinct models, but the top-3 were xgboost, lightgbm,
   xgboost because the pool handed to it was boosting-dominated. Widen the pool (e.g. take
   `max(3 * k, 12)` candidates) before diversity selection.
3. **Churn demo class balance is 27.5%**, so `is_imbalanced` is False (threshold is <25%).
   If the demo should showcase imbalance handling, push the intercept in
   `loaders.make_churn_frame()` from `-1.00` to about `-1.35`, or lower the threshold.
4. **Churn demo F1 ~0.55.** The generator's signal-to-noise is low. Acceptable, but if the
   demo should look stronger, increase the `tenure_months` / `support_calls` coefficients.
5. `scratch_*.py` at the repo root are throwaway harnesses. Convert them into real tests
   (see section 5) and delete them.

---

## 4. Architecture as built (file map)

```
rl_automl/
  core/         types.py (domain contract) config.py errors.py logging.py seeding.py
                metrics.py vocabulary.py
  task/         task_classifier.py           rule-based, LLM-optional hook
  dataset/      profiler.py fingerprint.py validation.py loaders.py
  environment/  base.py (interface + SearchTracker) action_space.py state_encoder.py
                reward.py surrogate.py simulator.py automl_env.py
  agent/        actor.py critic.py rollout_buffer.py ppo.py agent.py
  search/       model_registry.py hyperparameters.py stopping.py pareto.py
                recommendation.py
  execution/    splits.py preprocessing.py torch_models.py trainer.py evaluator.py
                resource_monitor.py executor.py comparison.py
  memory/       (empty - Phase 2)
  packaging/    model_exporter.py inference_generator.py zip_builder.py
  baselines/    (empty)
  training/     (empty)
  evaluation/   (empty)
  api/ cli/     (empty)
  orchestrator.py   the run state machine shared by API and CLI
  configs/default.yaml
```

**Non-negotiable layering (spec §35, test to be added):**
`agent` must never import `execution`; `environment.simulator` must never import `execution`.
`environment.automl_env` is the only bridge, and it is the real environment's job.

**Enforced invariants (keep them enforced):**
- no training before approval (`require_approval`)
- no test access before `begin_finalize()` (raises)
- action masks derived from the registry, never hardcoded

---

## 5. Remaining Phase 1 work

1. **Fix the blocking bug** (section 2).
2. **FastAPI service** (`api/`): `POST /datasets`, `POST /runs`, `GET /runs/{id}`,
   `GET /runs/{id}/recommendations`, `POST /runs/{id}/approve`, `POST /runs/{id}/cancel`,
   `GET /runs/{id}/events` (SSE), `GET /runs/{id}/artifacts/model.zip`,
   `GET /runs/{id}/artifacts/results.zip`. Runs are asynchronous with a background worker;
   the orchestrator stays synchronous and a `RunManager` owns threads + the SSE fan-out.
3. **CLI** (`cli/main.py`, argparse): `plan`, `run` (interactive approve), `status`,
   `download`, `train-rl`, `benchmark`, `datasets`.
4. **Tests** (`rl_automl/tests/`) - convert the scratch harnesses into pytest suites:
   `test_task_classifier`, `test_profiler`, `test_fingerprint`, `test_registry`,
   `test_action_space` (masking vs registry), `test_state_encoder` (dimension/version),
   `test_reward` (monotonicity, cost, novelty), `test_splits` (leakage, determinism),
   `test_preprocessing`, `test_evaluator` (agrees with sklearn, pos_label for string labels),
   `test_executor` (failure isolation, budget, test guard), `test_packaging` (ZIP round trip),
   `test_ppo_smoke`, **`test_import_boundaries`**, `test_api`, `test_end_to_end`.
5. **README.md** at the repo root: architecture diagram, quick start, config reference,
   honest benchmark results, known limitations.
6. **`scripts/`**: `demo.py` (the end-to-end walkthrough), `build_meta_dataset.py`.

## 6. Phase 2 (after Phase 1 is complete)

- `memory/`: SQLite persistence, fingerprint k-NN similarity, cross-dataset memory features
  fed into `StateTracker.set_memory_features()`
- `baselines/`: random / greedy / grid searchers over the shared `AutoMLEnvironment`
  interface, with identical budget accounting
- `evaluation/benchmark.py`: the RL-vs-baselines study in **both** simulator and real
  environments, with bootstrap CIs and a paired significance test
- `training/train_rl.py`: pretrain PPO across the bundled + synthetic corpus, save the
  checkpoint to `artifacts/policies/ppo_policy.pt`
- ONNX export verification pass (the code path exists; `skl2onnx` is **not installed** here,
  so it currently skips with a note)
- Unsupervised evaluation protocols for clustering / anomaly detection are implemented in the
  executor but were not exercised end-to-end beyond a small smoke run

---

## 7. Environment facts (do not re-litigate)

- Python **3.14.5**; numpy 2.4.6, pandas **3.0.3**, scikit-learn 1.9.0, scipy 1.17.1
- torch 2.11.0+cu128, xgboost 3.3.0, lightgbm 4.6.0
- fastapi 0.129, uvicorn 0.40, pydantic 2.12.5, typer 0.25, rich 14.3, ruff, mypy, pytest 9.1
- **`pydantic-settings` is NOT installed** - config layering is hand-rolled on pydantic + PyYAML
- **`skl2onnx` is NOT installed** - ONNX export is opt-in and currently skips
- Shell is **cmd.exe**, not PowerShell. Console is **cp1252**, so source and generated
  artifacts must stay ASCII (this already bit us once - see section 8)
- Package is not pip-installed; run everything from the repo root (`pythonpath = ["."]` is set
  for pytest)

## 8. Gotchas already hit (so they do not bite twice)

- `DataSplits.target` was both a field and a method - the field shadowed it. Method is now
  `target_for()`.
- `random_state` injection must be **signature-aware**; `AgglomerativeClustering`, `DBSCAN`,
  `OneClassSVM` and `LocalOutlierFactor` reject it.
- `MissingIndicator` casts to float64, so it can only be pointed at numeric columns.
- `ColumnTransformer` can return a DataFrame in scikit-learn 1.9; the trainer forces
  `np.asarray` so fit and predict see the same representation.
- Binary metrics need an explicit `pos_label` for string targets, and `roc_auc_score` has no
  `pos_label` argument - use a `(y_true == pos_label).astype(int)` indicator instead.
- The MLP must move its input tensors to the training device.
- Non-ASCII in source breaks the Windows console: **no `\uXXXX` escapes and no literal
  non-ASCII** in strings that get printed or written into artifacts. `§` is safe (cp1252).

## 9. How to resume

```cmd
cd "C:\Users\Ibhan Mukherjee\Desktop\RLAGENT"
set AUTOML_LOG_LEVEL=ERROR

python scratch_e2e.py      :: reproduces the blocking bug at the packaging step
python scratch_exec.py     :: execution layer, incl. test-isolation guard
python scratch_rl.py       :: RL core + PPO training + checkpoint guard
python scratch_verify.py   :: registry presets + full-package compile check
```

Then: fix section 2, re-run `scratch_e2e.py` until it prints `END-TO-END OK`, and move on to
the API and CLI. **Ask before starting a real PPO training run** - that was your instruction
and it still stands.
