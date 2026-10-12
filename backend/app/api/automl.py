"""AutoML run endpoints backed by the rl_automl orchestrator."""

import traceback

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.automl import runner
from app.db import SessionLocal, get_db, new_uuid
from app.models import AutoMLRun, Dataset, Job
from app.schemas import JobOut
from app.workers.dispatch import dispatch_automl_run

router = APIRouter(prefix="/api/automl", tags=["automl"])


class StartRunIn(BaseModel):
    dataset_id: str
    target: str | None = None
    task: str | None = None
    problem_statement: str | None = None


class ApproveIn(BaseModel):
    selection: str | list[int] | list[str] = Field(default="all")


def _run_out(row: AutoMLRun) -> dict:
    return {
        "run_id": row.id,
        "dataset_id": row.dataset_id,
        "problem_statement": row.problem_statement,
        "target": row.target,
        "task": row.task,
        "metric": row.metric,
        "status": row.status,
        "planner_note": row.planner_note,
        "recommendations": row.recommendations or [],
        "approved": row.approved or [],
        "leaderboard": row.leaderboard or [],
        "best_model": row.best_model,
        "best_test_score": row.best_test_score,
        "artifacts": row.artifacts or {},
        "error": row.error,
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


def _get_row(db: Session, run_id: str) -> AutoMLRun:
    row = db.get(AutoMLRun, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="AutoML run not found.")
    return row


@router.post("/runs")
def start_run(payload: StartRunIn, db: Session = Depends(get_db)):
    """Plan an AutoML run: profile the dataset and recommend pipelines.
    Stops at the approval gate — no training happens until /approve + /execute."""
    if db.get(Dataset, payload.dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    try:
        row = runner.plan_run(
            db,
            payload.dataset_id,
            problem_statement=payload.problem_statement or "",
            target=payload.target,
            task=payload.task,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"code": "AUTOML_PLAN_FAILURE", "message": f"Planning failed: {exc}"},
        ) from exc
    return _run_out(row)


@router.get("/runs")
def list_runs(db: Session = Depends(get_db)):
    rows = db.query(AutoMLRun).order_by(AutoMLRun.created_at.desc()).all()
    return [_run_out(r) for r in rows]


@router.get("/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)):
    return _run_out(_get_row(db, run_id))


@router.post("/runs/{run_id}/approve")
def approve_run(run_id: str, payload: ApproveIn, db: Session = Depends(get_db)):
    _get_row(db, run_id)
    try:
        row = runner.approve_run(db, run_id, payload.selection)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "AUTOML_APPROVE_FAILURE", "message": str(exc)},
        ) from exc

    # Approving the pipelines is the user's consent to train them — start execution
    # immediately instead of requiring another click.
    job = Job(id=new_uuid(), dataset_id=row.dataset_id, kind="AUTOML_RUN", status="QUEUED")
    db.add(job)
    row.job_id = job.id
    row.status = "queued"
    db.commit()
    dispatch_automl_run(run_id, job.id, SessionLocal)
    return _run_out(row)


@router.post("/runs/{run_id}/execute")
def execute_run(run_id: str, db: Session = Depends(get_db)):
    row = _get_row(db, run_id)
    if row.status != "awaiting_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Run is in status '{row.status}'; approve pipelines before executing.",
        )
    if not row.approved:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "NOTHING_APPROVED",
                "message": "No pipelines approved yet. Approve at least one recommended pipeline first.",
            },
        )
    existing = (
        db.query(Job)
        .filter(Job.dataset_id == row.dataset_id, Job.kind == "AUTOML_RUN", Job.status.in_(["QUEUED", "RUNNING"]))
        .first()
    )
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail={"code": "RUN_ALREADY_RUNNING", "message": "This run already has a job in progress."},
        )
    job = Job(id=new_uuid(), dataset_id=row.dataset_id, kind="AUTOML_RUN", status="QUEUED")
    db.add(job)
    row.job_id = job.id
    row.status = "queued"
    db.commit()
    dispatch_automl_run(run_id, job.id, SessionLocal)
    return {"run_id": run_id, "job_id": job.id, "status": "queued"}


@router.get("/runs/{run_id}/job", response_model=JobOut)
def run_job(run_id: str, db: Session = Depends(get_db)):
    row = _get_row(db, run_id)
    if row.job_id is None:
        raise HTTPException(status_code=404, detail="No execution job for this run yet.")
    job = db.get(Job, row.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobOut(
        id=job.id,
        kind=job.kind,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        detail=job.detail,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("/runs/{run_id}/artifacts/{kind}")
def download_artifact(run_id: str, kind: str, db: Session = Depends(get_db)):
    """Download the packaged model.zip or results.zip produced by the run."""
    if kind not in ("model", "results"):
        raise HTTPException(status_code=400, detail="kind must be 'model' or 'results'.")
    row = _get_row(db, run_id)
    path = (row.artifacts or {}).get(f"{kind}_zip")
    if not path:
        raise HTTPException(status_code=404, detail="This artifact was not produced by the run.")
    from pathlib import Path

    from fastapi.responses import FileResponse

    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Artifact file is missing from storage.")
    return FileResponse(path=p, media_type="application/zip", filename=p.name)
