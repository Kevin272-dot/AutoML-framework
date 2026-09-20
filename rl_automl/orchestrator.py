"""Run orchestration: the state machine behind both the API and the CLI.

There is exactly one implementation of the run lifecycle, and both surfaces call it. That
is deliberate: an approval gate that can be bypassed by a second code path is not a gate.

The two phases, and the gate between them (spec §2, §10):

    plan()    -> profile, understand, fingerprint, RL-plan, recommend      (trains nothing)
    approve() -> record which pipelines the user authorised                 (still nothing)
    execute() -> train approved pipelines, compare, select, package         (only after approval)

``execute`` calls ``require_approval`` and ``PipelineExecutor`` refuses to touch the test
split before ``begin_finalize``, so "no training before approval" and "no test leakage into
selection" are enforced by the code rather than by convention.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.errors import (
    AutoMLError,
    NotApprovedError,
    PolicyLoadError,
    RunStateError,
)
from rl_automl.core.logging import get_logger
from rl_automl.core.metrics import primary_metric_for
from rl_automl.core.seeding import seed_everything
from rl_automl.core.types import (
    ArtifactBundle,
    ExecutionEvent,
    ExperimentResult,
    ExperimentSpec,
    RunObject,
    RunStatus,
    TaskType,
    utc_now_iso,
)
from rl_automl.dataset.fingerprint import build_fingerprint
from rl_automl.dataset.profiler import DatasetProfiler, classify_column
from rl_automl.environment.action_space import build_action_space
from rl_automl.environment.state_encoder import StateEncoder
from rl_automl.environment.surrogate import AnalyticPrior, SurrogateModel
from rl_automl.execution.comparison import ModelComparator
from rl_automl.execution.executor import (
    ExecutionLimits,
    PipelineExecutor,
    require_approval,
)
from rl_automl.packaging.inference_generator import InferenceGenerator
from rl_automl.packaging.model_exporter import ModelExporter
from rl_automl.packaging.zip_builder import ZipBuilder
from rl_automl.search.model_registry import REGISTRY
from rl_automl.search.recommendation import RecommendationEngine
from rl_automl.task.task_classifier import TaskClassifier

logger = get_logger("orchestrator")

PROBE_ROWS = 32


@dataclass
class PlanRequest:
    problem_statement: str
    frame: pd.DataFrame
    dataset_name: str = ""
    dataset_sha256: str = ""
    dataset_path: str = ""
    target: str | None = None
    task_type: TaskType | str | None = None
    metric: str | None = None
    run_id: str | None = None
    use_policy: bool = True


class AutoMLOrchestrator:
    def __init__(
        self,
        config: AutoMLConfig | None = None,
        *,
        memory: Any | None = None,
        surrogate: SurrogateModel | None = None,
        listener: Callable[[ExecutionEvent], None] | None = None,
    ) -> None:
        self.config = config or AutoMLConfig()
        self.config.ensure_directories()
        self.memory = memory
        self.listener = listener

        self.surrogate = surrogate or self._build_surrogate()
        self.exporter = ModelExporter(self.config)
        self.inference_generator = InferenceGenerator(self.config)
        self.zip_builder = ZipBuilder(self.config)
        self.classifier = TaskClassifier(
            min_confidence=self.config.task.min_confidence,
            default_metric_by_task=self.config.task.default_metric_by_task,
        )
        self._executors: dict[str, PipelineExecutor] = {}

    # ==================================================================================
    # Phase A -- planning
    # ==================================================================================

    def plan(self, request: PlanRequest) -> RunObject:
        """Understand, profile, fingerprint and recommend. Trains nothing."""
        seed_everything(self.config.runtime.seed)
        frame = request.frame
        if frame is None or frame.empty:
            raise RunStateError("cannot plan without a dataset")

        run = RunObject(
            run_id=request.run_id or _new_run_id(),
            problem_statement=request.problem_statement,
            dataset_name=request.dataset_name,
            dataset_sha256=request.dataset_sha256,
            dataset_path=request.dataset_path,
            config_snapshot=self.config.to_snapshot(),
        )
        self._emit(run, RunStatus.PROFILING.value, "Understanding the problem statement")

        # Pass 1: statement + column names, before any profile exists.
        task = self.classifier.classify(
            request.problem_statement, columns=[str(c) for c in frame.columns]
        )

        self._emit(run, RunStatus.PROFILING.value, "Profiling the dataset")
        profiler = DatasetProfiler(self.config.dataset)
        profile = profiler.profile(frame, target=task.target)

        # Pass 2: refine now that the target's distribution is known. A continuous target
        # with hundreds of distinct values is strong evidence against classification, and
        # this is where that evidence gets used.
        column_kinds = {str(column): classify_column(frame[column]) for column in frame.columns}
        task = self.classifier.classify(
            request.problem_statement,
            columns=[str(c) for c in frame.columns],
            column_kinds={name: kind.value for name, kind in column_kinds.items()},
            class_distribution=profile.class_distribution,
            target_hint=task.target,
        )

        if request.target or request.task_type or request.metric:
            task = self.classifier.apply_overrides(
                task,
                target=request.target,
                task_type=request.task_type,
                metric=request.metric,
            )
            if request.target and request.target != profile.target:
                profile = profiler.profile(frame, target=request.target)

        self._validate_task(task, frame)

        run.task = task
        run.dataset_profile = profile
        run.dataset_fingerprint = build_fingerprint(profile, task.task_type)
        self._emit(
            run,
            RunStatus.UNDERSTANDING.value,
            f"{task.task_type.value} task, optimising {task.metric}",
            detail={"task": task.to_example(), "confidence": task.confidence},
        )

        # -- RL planning ------------------------------------------------------------
        self._emit(run, RunStatus.PLANNING.value, "Planning experiments")
        action_space = build_action_space(
            task.task_type,
            self.config.models.enabled,
            min_experiments_before_stop=self.config.search.min_experiments,
        )
        encoder = StateEncoder({spec.key: spec.family for spec in REGISTRY.all_specs()})

        agent, policy_note = self._load_policy(action_space, encoder, request.use_policy)

        engine = RecommendationEngine(
            task=task,
            action_space=action_space,
            state_encoder=encoder,
            surrogate=self.surrogate,
            profile=profile,
            search_config=self.config.search,
            target_score=self.config.search.target_score,
        )
        recommendation = engine.recommend(
            fingerprint=run.dataset_fingerprint,
            agent=agent,
            n_rows=profile.n_rows,
            n_cols=max(profile.n_cols - 1, 1),
        )
        if policy_note:
            recommendation.notes.append(policy_note)

        run.recommendation_set = recommendation
        run.rl_recommendations = list(recommendation.recommended_experiments)
        run.search_statistics = recommendation.search_statistics
        run.status = RunStatus.AWAITING_APPROVAL

        self._emit(
            run,
            RunStatus.AWAITING_APPROVAL.value,
            f"{len(run.rl_recommendations)} pipeline(s) recommended; awaiting approval",
            progress=0.25,
            detail={"planner": recommendation.planner},
        )
        self.save_run(run)

        logger.info(
            "plan complete",
            extra={
                "context": {
                    "run_id": run.run_id,
                    "task": task.task_type.value,
                    "planner": recommendation.planner,
                    "n_recommended": len(run.rl_recommendations),
                }
            },
        )
        return run

    # ==================================================================================
    # The approval gate
    # ==================================================================================

    def approve(
        self,
        run: RunObject,
        *,
        selection: str | Iterable[int] | Iterable[str] = "all",
        extra_experiments: Iterable[ExperimentSpec] | None = None,
        approved_by: str = "user",
    ) -> RunObject:
        """Record the pipelines the user authorised. Trains nothing.

        ``selection`` accepts ``"all"``, a list of 1-based ranks, or a list of model names.
        """
        if run.status not in (
            RunStatus.AWAITING_APPROVAL,
            RunStatus.CREATED,
            RunStatus.PLANNING,
        ):
            raise RunStateError(
                f"cannot approve pipelines for a run in state '{run.status.value}'",
                run_id=run.run_id,
                status=run.status.value,
            )
        if not run.rl_recommendations:
            raise RunStateError("there are no recommended pipelines to approve", run_id=run.run_id)

        approved = self._resolve_selection(run, selection)
        for experiment in extra_experiments or ():
            approved.append(experiment)

        if not approved:
            raise NotApprovedError(
                "the selection resolved to zero pipelines; pass 'all' or at least one rank",
                run_id=run.run_id,
            )
        if len(approved) > self.config.search.max_experiments:
            raise RunStateError(
                f"{len(approved)} pipelines were approved but the configured maximum is "
                f"{self.config.search.max_experiments}",
                run_id=run.run_id,
            )

        run.approved_experiments = approved
        run.approved_at = utc_now_iso()
        run.status = RunStatus.AWAITING_APPROVAL
        run.messages.append(
            f"{len(approved)} pipeline(s) approved by {approved_by} at {run.approved_at}"
        )
        self._emit(
            run,
            RunStatus.AWAITING_APPROVAL.value,
            f"{len(approved)} pipeline(s) approved; ready to execute",
            progress=0.3,
            detail={"approved": [experiment.model for experiment in approved]},
        )
        self.save_run(run)
        return run

    def _resolve_selection(
        self, run: RunObject, selection: str | Iterable[int] | Iterable[str]
    ) -> list[ExperimentSpec]:
        recommended = run.rl_recommendations
        if isinstance(selection, str):
            if selection.strip().lower() not in ("all", "*"):
                names = [part.strip() for part in selection.split(",") if part.strip()]
                return self._by_names(recommended, names)
            return [item.experiment for item in recommended if item.experiment is not None]

        tokens = list(selection)
        if all(isinstance(token, int) for token in tokens):
            chosen: list[ExperimentSpec] = []
            for rank in tokens:
                match = next((item for item in recommended if item.rank == rank), None)
                if match is None or match.experiment is None:
                    raise RunStateError(f"no recommended pipeline has rank {rank}")
                chosen.append(match.experiment)
            return chosen
        return self._by_names(recommended, [str(token) for token in tokens])

    @staticmethod
    def _by_names(recommended: list[Any], names: list[str]) -> list[ExperimentSpec]:
        chosen: list[ExperimentSpec] = []
        for name in names:
            matches = [
                item
                for item in recommended
                if item.experiment is not None
                and (item.model == name or item.model.lower() == name.lower())
            ]
            if not matches:
                raise RunStateError(
                    f"'{name}' was not among the recommended pipelines",
                    recommended=[item.model for item in recommended],
                )
            chosen.append(matches[0].experiment)
        return chosen

    # ==================================================================================
    # Phase B -- execution
    # ==================================================================================

    def execute(
        self,
        run: RunObject,
        *,
        frame: pd.DataFrame | None = None,
        holdout: pd.DataFrame | None = None,
        plan_only_no_artifacts: bool = False,
    ) -> RunObject:
        """Train the approved pipelines, compare, select and package."""
        approved = require_approval(run.approved_experiments)
        if run.task is None:
            raise RunStateError("the run has no task specification", run_id=run.run_id)

        dataset = frame if frame is not None else self._reload_dataset(run)

        try:
            executor = self._build_executor(run)
            executor.prepare(dataset, holdout)

            if holdout is not None and len(holdout):
                self._emit(
                    run,
                    RunStatus.EXECUTING.value,
                    f"Training {len(approved)} approved pipeline(s) and reserving "
                    f"{len(holdout)} supplied test row(s)",
                    progress=0.35,
                    detail={"holdout_rows": int(len(holdout))},
                )
            else:
                self._emit(
                    run,
                    RunStatus.EXECUTING.value,
                    f"Training {len(approved)} approved pipeline(s)",
                    progress=0.35,
                    detail={"approved": [experiment.model for experiment in approved]},
                )
            summary = executor.run(approved)
            run.results = summary.results
            run.search_statistics = _merge_statistics(run.search_statistics, summary)

            for result in summary.results:
                self._emit(
                    run,
                    RunStatus.EXECUTING.value,
                    _result_message(result),
                    detail={
                        "model": result.experiment.model,
                        "status": result.status.value,
                        "validation_score": result.validation_score,
                    },
                )

            # -- compare and select on validation evidence only ----------------------
            self._emit(run, RunStatus.COMPARING.value, "Comparing models")
            comparator = ModelComparator(run.task)
            summary_preview = comparator.compare(run.results)
            run.comparison = summary_preview

            self._emit(run, RunStatus.SELECTING.value, "Selecting the final model")
            best_model, pareto, comparison = comparator.select(run.results)
            run.best_model = best_model
            run.pareto_frontier = pareto
            run.comparison = comparison

            if best_model is None:
                run.status = RunStatus.FAILED
                run.error = (
                    "every approved pipeline failed; nothing could be packaged. See the "
                    "results for per-pipeline errors."
                )
                self._emit(run, RunStatus.FAILED.value, run.error, progress=1.0)
                self.save_run(run)
                return run

            # -- held-out evaluation, after selection (spec §12) ---------------------
            self._emit(
                run,
                RunStatus.EVALUATING.value,
                "Evaluating on the held-out test split",
            )
            executor.begin_finalize()
            for result in run.results:
                if result.succeeded:
                    executor.evaluate_test(result)
            # Re-render the table now that test scores exist; selection is untouched.
            run.comparison = comparator.compare(run.results)

            if plan_only_no_artifacts:
                run.status = RunStatus.COMPLETE
                self.save_run(run)
                return run

            # -- packaging -----------------------------------------------------------
            self._emit(run, RunStatus.PACKAGING.value, "Packaging the final model")
            run.artifacts = self._package(run, executor, best_model, comparison)
            run.status = RunStatus.COMPLETE

        except AutoMLError as exc:
            run.status = RunStatus.FAILED
            run.error = _error_message(exc)
            run.messages.append(run.error)
            self._emit(run, RunStatus.FAILED.value, run.error)
        except Exception as exc:  # pragma: no cover - unexpected, but a run must not vanish
            run.status = RunStatus.FAILED
            run.error = _error_message(exc)
            run.messages.append(run.error)
            logger.exception("run failed", extra={"context": {"run_id": run.run_id}})
            self._emit(run, RunStatus.FAILED.value, run.error)

        self._emit(
            run,
            run.status.value,
            "Run complete" if run.status is RunStatus.COMPLETE else "Run failed",
            progress=1.0,
        )
        run.touch()
        self.save_run(run)
        return run

    def cancel(self, run: RunObject, reason: str = "cancelled by user") -> RunObject:
        if run.status.is_terminal:
            return run
        run.status = RunStatus.CANCELLED
        run.messages.append(reason)
        self._emit(run, RunStatus.CANCELLED.value, reason)
        self.save_run(run)
        return run

    # ==================================================================================
    # Packaging
    # ==================================================================================

    def _package(
        self,
        run: RunObject,
        executor: PipelineExecutor,
        best_model: Any,
        comparison: Any,
    ) -> ArtifactBundle:
        result = next(
            (
                item
                for item in run.results
                if item.experiment.experiment_id == best_model.experiment_id
            ),
            None,
        )
        if result is None or run.task is None or run.dataset_profile is None:
            return ArtifactBundle()

        trained = executor.get_trained(best_model.experiment_id)
        if trained is None:
            run.messages.append(
                "the selected model was no longer in memory, so no package was produced"
            )
            return ArtifactBundle()

        run_dir = self.config.artifacts_dir() / "runs" / run.run_id
        probe = self._probe_frame(executor)
        expected = self._expected_predictions(executor, trained, probe, run.task.task_type)

        package = self.exporter.export(
            trained=trained,
            result=result,
            task=run.task,
            profile=run.dataset_profile,
            destination=run_dir / "package",
            run_id=run.run_id,
            dataset_name=run.dataset_name,
            dataset_sha256=run.dataset_sha256,
            column_kinds=executor.column_kinds,
            label_classes=executor.class_labels,
            label_encoding_required=executor.label_encoding_required,
            split_summary=executor.splits.summary() if executor.splits else {},
            probe_frame=probe,
        )
        self.inference_generator.generate(
            package, split_summary=executor.splits.summary() if executor.splits else {}
        )
        run.messages.extend(package.warnings)

        archive = self.zip_builder.build(
            package.directory,
            run_dir / "model_package.zip",
            probe_frame=probe,
            expected_predictions=expected,
            task_type=run.task.task_type.value,
        )

        self._emit(
            run,
            RunStatus.PACKAGING.value,
            "Model package verified and archived",
            detail={
                "zip": str(archive.path),
                "verified": archive.verified,
                "sha256": archive.sha256,
            },
        )

        results_archive = self._results_archive(run, run_dir)

        return ArtifactBundle(
            model_zip=str(archive.path),
            model_zip_sha256=archive.sha256,
            model_dir=str(package.directory),
            results_zip=str(results_archive.path) if results_archive else None,
            results_zip_sha256=results_archive.sha256 if results_archive else None,
            verified=archive.verified,
            extras=dict(package.extras),
        )

    def _results_archive(self, run: RunObject, run_dir: Path) -> Any | None:
        """The download-results bundle: results.json, comparison.csv, experiment_history.json."""
        import io

        payload: dict[str, Any] = {
            "results.json": json.dumps(run.model_dump(mode="json"), indent=2, default=str),
        }

        if run.comparison and run.comparison.rows:
            frame = pd.DataFrame([row.model_dump(mode="json") for row in run.comparison.rows])
            buffer = io.StringIO()
            frame.to_csv(buffer, index=False)
            payload["comparison.csv"] = buffer.getvalue()

        history = [
            {
                "experiment": result.experiment.model_dump(mode="json"),
                "status": result.status.value,
                "error": result.error,
                "error_type": result.error_type,
                "validation_score": result.validation_score,
                "test_score": result.test_score,
                "validation_metrics": result.validation_metrics,
                "test_metrics": result.test_metrics,
                "training_time_s": result.training_time_s,
                "peak_memory_mb": result.peak_memory_mb,
                "model_size_bytes": result.model_size_bytes,
                "started_at": result.started_at,
                "finished_at": result.finished_at,
                "warnings": result.warnings,
            }
            for result in run.results
        ]
        payload["experiment_history.json"] = json.dumps(history, indent=2, default=str)
        payload["search_statistics.json"] = json.dumps(
            run.search_statistics.model_dump(mode="json") if run.search_statistics else {},
            indent=2,
            default=str,
        )
        payload["comparison.txt"] = run.comparison.table if run.comparison else ""

        try:
            return self.zip_builder.build_file_archive(payload, run_dir / "results.zip")
        except AutoMLError as exc:
            run.messages.append(f"results archive could not be built: {exc}")
            return None

    def _probe_frame(self, executor: PipelineExecutor) -> pd.DataFrame | None:
        """Rows used to verify the packaged model end to end.

        Taken from the **validation** split, not the test split, so that verifying an
        artifact never re-reads held-out data.
        """
        if executor.splits is None:
            return None
        frame = executor.splits.features("val")
        return frame.head(PROBE_ROWS) if not frame.empty else None

    @staticmethod
    def _expected_predictions(
        executor: PipelineExecutor,
        trained: Any,
        probe: pd.DataFrame | None,
        task_type: TaskType,
    ) -> np.ndarray | None:
        if probe is None:
            return None
        try:
            bundle = executor._trainer.predict(trained, probe)
            predictions = np.asarray(bundle.predictions)
            if task_type is TaskType.CLASSIFICATION:
                # The shipped loader decodes class indices back to the original labels, so the
                # expectation it is compared against must live in that same space.
                predictions = executor.decode_labels(predictions)
            return predictions
        except Exception:  # pragma: no cover - verification degrades, packaging still runs
            return None

    # ==================================================================================
    # Persistence
    # ==================================================================================

    def run_dir(self, run_id: str) -> Path:
        return self.config.artifacts_dir() / "runs" / run_id

    def save_run(self, run: RunObject) -> Path:
        run.touch()
        directory = self.run_dir(run.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "run.json"
        path.write_text(
            json.dumps(run.model_dump(mode="json"), indent=2, default=str), encoding="utf-8"
        )
        return path

    def load_run(self, run_id: str) -> RunObject:
        path = self.run_dir(run_id) / "run.json"
        if not path.is_file():
            raise RunStateError(f"run '{run_id}' was not found", run_id=run_id)
        return RunObject.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def list_runs(self) -> list[dict[str, Any]]:
        root = self.config.artifacts_dir() / "runs"
        if not root.is_dir():
            return []
        summaries: list[dict[str, Any]] = []
        for directory in sorted(root.iterdir(), reverse=True):
            path = directory / "run.json"
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:  # pragma: no cover
                continue
            best = payload.get("best_model") or {}
            summaries.append(
                {
                    "run_id": payload.get("run_id"),
                    "status": payload.get("status"),
                    "created_at": payload.get("created_at"),
                    "dataset": payload.get("dataset_name"),
                    "task": (payload.get("task") or {}).get("task_type"),
                    "best_model": best.get("model"),
                    "validation_score": best.get("validation_score"),
                    "test_score": best.get("test_score"),
                    "model_zip": (payload.get("artifacts") or {}).get("model_zip"),
                }
            )
        return summaries

    # ==================================================================================
    # Internals
    # ==================================================================================

    def _build_surrogate(self) -> SurrogateModel:
        """Prior by default; upgraded to a fitted surrogate when memory has meta-data."""
        surrogate = SurrogateModel(
            AnalyticPrior(), cost_reference_s=self.config.models.cost_reference_s
        )
        if self.memory is None:
            return surrogate
        try:
            rows = self.memory.load_meta_rows()
        except Exception as exc:  # pragma: no cover - memory is optional
            logger.warning("could not load meta rows: %s", exc)
            return surrogate
        if rows:
            surrogate.fit(rows)
        return surrogate

    def _load_policy(
        self, action_space: Any, encoder: StateEncoder, use_policy: bool
    ) -> tuple[Any | None, str | None]:
        """Load a trained policy if one exists. Returns (agent, note)."""
        if not use_policy or not self.config.rl.enabled:
            return None, "RL planning disabled; using the surrogate planner"

        # A policy is task-specific, so prefer the checkpoint pretrained for this task type.
        path = self.config.policy_path(action_space.layout.task_type)
        if not path.is_file():
            return None, (
                "no trained policy checkpoint found; using the surrogate planner. "
                "Run `automl train-rl` to train one."
            )

        try:
            from rl_automl.agent.agent import PPOAgent

            agent, _extra = PPOAgent.load(
                path,
                action_space=action_space,
                state_encoder=encoder,
                rl_config=self.config.rl,
                device="cpu",
                strict=False,
            )
        except PolicyLoadError as exc:
            return None, f"the saved policy could not be used ({exc}); using the surrogate planner"

        trained_steps = agent.metadata.trained_steps
        if trained_steps <= 0 and not self.config.rl.warm_start_recommendation:
            return None, ("the saved policy has not been trained yet; using the surrogate planner")
        if trained_steps <= 0:
            return agent, (
                "the saved policy has not been trained, so its proposals are uninformative; "
                "the recommendation is ranked by the surrogate"
            )
        return agent, None

    def _build_executor(self, run: RunObject) -> PipelineExecutor:
        executor = PipelineExecutor(
            run.task,
            self.config,
            run_id=run.run_id,
            seed=self.config.runtime.seed,
            limits=ExecutionLimits(
                max_experiments=self.config.search.max_experiments,
                time_budget_s=self.config.search.time_budget_s,
                memory_budget_mb=self.config.search.memory_budget_mb,
            ),
        )
        self._executors[run.run_id] = executor
        return executor

    @staticmethod
    def _validate_task(task: Any, frame: pd.DataFrame) -> None:
        if task.task_type.is_supervised:
            if not task.target:
                raise RunStateError(
                    "no target column could be identified for this supervised task",
                    columns=[str(column) for column in frame.columns][:50],
                )
            if task.target not in frame.columns:
                raise RunStateError(
                    f"the target column '{task.target}' is not present in the dataset",
                    columns=[str(column) for column in frame.columns][:50],
                )
        if not task.metric:
            task.metric = primary_metric_for(task.task_type)

    @staticmethod
    def _reload_dataset(run: RunObject) -> pd.DataFrame:
        return _read_table(run.dataset_path, "dataset", run.run_id)

    @staticmethod
    def _reload_holdout(run: RunObject) -> pd.DataFrame | None:
        """The supplied test table, when the run was created with one."""
        if not run.holdout_path:
            return None
        return _read_table(run.holdout_path, "test table", run.run_id)

    def _emit(
        self,
        run: RunObject,
        stage: str,
        message: str,
        *,
        progress: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Record a progress event and notify any listener (the API's SSE stream)."""
        event = ExecutionEvent(
            run_id=run.run_id,
            stage=stage,
            message=message,
            sequence=len(run.progress) + 1,
            progress=progress if progress is not None else _stage_progress(stage),
            detail=detail or {},
        )
        run.progress.append(event.model_dump(mode="json"))
        run.touch()
        if self.listener is not None:
            try:
                self.listener(event)
            except Exception:  # pragma: no cover - a broken listener must not break a run
                logger.warning("progress listener raised; continuing")


def _error_message(exc: Exception) -> str:
    """Render a failure together with its structured ``details``.

    ``AutoMLError`` subclasses carry machine-readable detail (for example the list of
    problems behind an artifact verification failure). Recording only ``str(exc)`` throws
    that away and makes the failure almost impossible to diagnose from a run record.
    """
    message = f"{type(exc).__name__}: {exc}"
    details = getattr(exc, "details", None)
    if not details:
        return message
    try:
        rendered = json.dumps(details, default=str, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - details are plain data
        rendered = str(details)
    if len(rendered) > 2000:
        rendered = rendered[:2000] + "...(truncated)"
    return f"{message} | details: {rendered}"


def _new_run_id() -> str:
    from rl_automl.core.types import new_id

    return new_id("run")


_STAGE_PROGRESS = {
    RunStatus.CREATED.value: 0.0,
    RunStatus.PROFILING.value: 0.1,
    RunStatus.UNDERSTANDING.value: 0.18,
    RunStatus.PLANNING.value: 0.25,
    RunStatus.AWAITING_APPROVAL.value: 0.3,
    RunStatus.EXECUTING.value: 0.6,
    RunStatus.EVALUATING.value: 0.75,
    RunStatus.COMPARING.value: 0.82,
    RunStatus.SELECTING.value: 0.86,
    RunStatus.PACKAGING.value: 0.95,
    RunStatus.COMPLETE.value: 1.0,
    RunStatus.FAILED.value: 1.0,
    RunStatus.CANCELLED.value: 1.0,
}


def _stage_progress(stage: str) -> float:
    return _STAGE_PROGRESS.get(stage, 0.0)


def _result_message(result: ExperimentResult) -> str:
    if result.succeeded:
        score = "n/a" if result.validation_score is None else f"{result.validation_score:.4f}"
        return (
            f"{result.experiment.model} trained in {result.training_time_s:.1f}s "
            f"({result.primary_metric}={score})"
        )
    return f"{result.experiment.model} failed: {result.error}"


def _merge_statistics(existing: Any, summary: Any) -> Any:
    """Fold real execution numbers into the plan-phase statistics object."""
    from rl_automl.core.types import SearchStatistics

    if existing is None:
        existing = SearchStatistics(searcher="rl-ppo")
    existing.n_experiments = summary.n_success + summary.n_failed
    existing.n_failed = summary.n_failed
    existing.n_distinct_models = len({result.experiment.model for result in summary.results})
    existing.total_training_time_s = summary.total_training_time_s
    existing.total_compute_s = summary.total_training_time_s
    existing.stopping_reason = summary.stopping_reason
    best = summary.best_by_validation()
    if best is not None:
        existing.best_validation_score = best.validation_score
        existing.best_model = best.experiment.model
    return existing


__all__ = ["PROBE_ROWS", "AutoMLOrchestrator", "PlanRequest"]
