"""Adapter around the teammate's `rl_automl` orchestrator (RL-AutoML engine).

The orchestrator owns run state in its own artifact tree (authoritative);
this module bridges it into our product: job progress, DB mirror rows,
and artifact registration. Their approval gate (`awaiting_approval`) maps
directly onto our third user gate: pipeline approval.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.config import get_settings
from app.datasets.download import load_dataframe
from app.models import AutoMLRun, Dataset, DatasetFile, Job, utcnow
from app.storage import open_artifact

MAX_PLAN_ROWS = 200_000


def _build_config():
    from rl_automl.core.config import AutoMLConfig, RuntimeConfig, SearchConfig

    settings = get_settings()
    artifacts_root = Path(settings.local_storage_dir).parent / "automl-artifacts"
    return AutoMLConfig(
        runtime=RuntimeConfig(
            artifacts_dir=str(artifacts_root),
            # Cap thread pools: unbounded n_jobs/OpenMP inside the API process causes
            # pathological oversubscription (xgboost: 120s on 600 rows vs 1.2s capped).
            n_jobs=2,
            num_threads=4,
            log_level="WARNING",
        ),
        search=SearchConfig(max_experiments=8, n_recommendations=3),
    )

def _load_dataset_frame(dataset: Dataset, db: Session) -> tuple[pd.DataFrame, DatasetFile]:
    dfile = (
        db.query(DatasetFile)
        .filter(DatasetFile.dataset_id == dataset.id, DatasetFile.file_name != "processed.parquet")
        .order_by(DatasetFile.created_at.desc())
        .first()
    )
    if dfile is None:
        dfile = db.query(DatasetFile).filter(DatasetFile.dataset_id == dataset.id).first()
    if dfile is None:
        raise ValueError("dataset has no data file to train on")
    local = open_artifact(dfile.storage_key, dfile.storage_backend)
    frame, _skipped = load_dataframe(local, dfile.file_format or "csv")
    if frame.shape[0] > MAX_PLAN_ROWS:
        frame = frame.head(MAX_PLAN_ROWS)
    return frame, dfile


def _orchestrator(listener=None):
    from rl_automl.orchestrator import AutoMLOrchestrator

    return AutoMLOrchestrator(_build_config(), listener=listener)


def _mirror(run_obj, db: Session, row: AutoMLRun) -> AutoMLRun:
    """Copy the orchestrator run state into our DB mirror row."""
    row.status = run_obj.status.value
    row.task = run_obj.task.task_type.value if run_obj.task else row.task
    row.metric = run_obj.task.metric if run_obj.task else row.metric
    row.target = run_obj.task.target if run_obj.task and run_obj.task.target else row.target
    if run_obj.recommendation_set is not None:
        row.recommendations = [
            {
                "rank": rec.rank,
                "model": rec.model,
                "preprocessing": rec.preprocessing,
                "reason": rec.reason,
                "expected_performance": rec.expected_performance,
                "expected_cost": rec.expected_cost.value if hasattr(rec.expected_cost, "value") else str(rec.expected_cost),
            }
            for rec in run_obj.recommendation_set.recommended_experiments
        ]
    row.approved = [spec.model for spec in run_obj.approved_experiments]
    row.leaderboard = [
        {
            "model": r.experiment.model if r.experiment else "?",
            "status": r.status.value if hasattr(r.status, "value") else str(r.status),
            "validation_score": r.validation_score,
            "test_score": r.test_score,
            "primary_metric": r.primary_metric,
            "training_time_s": r.training_time_s,
            "inference_time_ms": r.inference_time_ms,
            "error": r.error,
        }
        for r in run_obj.results
    ]
    if run_obj.best_model is not None:
        row.best_model = run_obj.best_model.model
        row.best_test_score = run_obj.best_model.test_score
    row.artifacts = {
        "model_zip": run_obj.artifacts.model_zip,
        "results_zip": run_obj.artifacts.results_zip,
        "verified": run_obj.artifacts.verified,
    }
    row.error = run_obj.error
    db.commit()
    db.refresh(row)
    return row


def plan_run(
    db: Session,
    dataset_id: str,
    problem_statement: str = "",
    target: str | None = None,
    task: str | None = None,
) -> AutoMLRun:
    """Understand + profile + recommend. Trains nothing; stops at the approval gate."""
    from rl_automl.orchestrator import PlanRequest

    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise ValueError("dataset not found")
    frame, dfile = _load_dataset_frame(dataset, db)

    statement = problem_statement or (
        f"Train a model on {dataset.name}"
        + (f" to predict {target}" if target else "")
        + (f" ({task})" if task else "")
    )

    orchestrator = _orchestrator()
    run_obj = orchestrator.plan(
        PlanRequest(
            problem_statement=statement,
            frame=frame,
            dataset_name=dataset.name,
            dataset_sha256=dfile.file_hash_sha256 or "",
            dataset_path=dfile.storage_key,
            target=target,
            task_type=task,
        )
    )

    row = AutoMLRun(
        id=run_obj.run_id,
        dataset_id=dataset_id,
        problem_statement=statement,
        target=target,
        status=run_obj.status.value,
        job_id=None,
    )
    db.add(row)
    row = _mirror(run_obj, db, row)
    if run_obj.messages:
        row.planner_note = run_obj.messages[-1]
        db.commit()
    return row


def approve_run(db: Session, run_id: str, selection: str | list[int] | list[str]) -> AutoMLRun:
    row = db.get(AutoMLRun, run_id)
    if row is None:
        raise ValueError("run not found")
    orchestrator = _orchestrator()
    run_obj = orchestrator.load_run(run_id)
    run_obj = orchestrator.approve(run_obj, selection=selection, approved_by="user")
    return _mirror(run_obj, db, row)


def execute_run(run_id: str, job_id: str, db_factory) -> None:
    """Trains the approved pipelines (background task)."""
    db: Session = db_factory()
    try:
        row = db.get(AutoMLRun, run_id)
        job = db.get(Job, job_id)
        if row is None or job is None:
            return

        def listener(event) -> None:
            # Runs in the orchestrator's thread; refresh a short-lived session.
            session = db_factory()
            try:
                stage = str(event.stage)
                j = session.get(Job, job_id)
                if j is None:
                    return
                j.status = "RUNNING"
                j.stage = stage.upper()
                j.progress = min(1.0, max(0.0, float(event.progress)))
                j.detail = event.message
                # Keep the product-facing run mirror in sync so the UI shows the
                # live stage instead of a stale "queued" while training runs.
                r = session.get(AutoMLRun, run_id)
                if r is not None and stage in {
                    "profiling", "understanding", "planning", "awaiting_approval",
                    "executing", "evaluating", "comparing", "selecting", "packaging",
                }:
                    r.status = stage
                session.commit()
            finally:
                session.close()

        orchestrator = _orchestrator(listener=listener)
        run_obj = orchestrator.load_run(run_id)
        frame, _dfile = _load_dataset_frame(db.get(Dataset, row.dataset_id), db)

        job.status = "RUNNING"
        job.stage = "EXECUTING"
        job.detail = "Training approved pipelines."
        db.commit()

        run_obj = orchestrator.execute(run_obj, frame=frame)
        _mirror(run_obj, db, row)

        job.status = "COMPLETED" if run_obj.status.value == "complete" else "FAILED"
        job.stage = "DONE"
        job.progress = 1.0
        job.detail = (
            f"Best model: {row.best_model}" if row.best_model else (run_obj.error or "Run finished.")
        )
        job.finished_at = utcnow()
        db.commit()
    except Exception as exc:
        db.rollback()
        job = db.get(Job, job_id)
        if job is not None:
            job.status = "FAILED"
            job.error_code = "AUTOML_FAILURE"
            job.error_message = str(exc)[:500]
            job.finished_at = utcnow()
            db.commit()
        row = db.get(AutoMLRun, run_id)
        if row is not None:
            row.status = "failed"
            row.error = str(exc)[:500]
            db.commit()
    finally:
        db.close()
