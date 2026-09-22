"""Integration: our product flow around the teammate's rl_automl orchestrator.

plan -> user approves pipelines -> execute -> leaderboard + artifacts mirror.
Uses a tiny synthetic frame so the whole run is fast and fully offline.
"""

import numpy as np
import pandas as pd
import pytest

from app.automl import runner
from app.models import AutoMLRun, Dataset, DatasetFile, DataSource, Job


def _setup_dataset(db, tmp_path):
    import uuid

    uid = uuid.uuid4().hex[:8]
    df = pd.DataFrame({
        "tenure_months": np.random.default_rng(0).integers(1, 72, 120),
        "support_calls": np.random.default_rng(1).integers(0, 9, 120),
        "monthly_charges": np.random.default_rng(2).normal(70, 15, 120).round(2),
    })
    # churn increases with support calls, decreases with tenure
    logits = 0.09 * df["support_calls"] - 0.05 * df["tenure_months"] - 1.35
    df["churn"] = (np.random.default_rng(3).random(120) < 1 / (1 + np.exp(-logits))).astype(int)

    source = DataSource(
        id=f"src-{uid}", slug=f"hf-{uid}", name="HF", base_url="https://example.com",
        source_type="DATASET_CATALOG", access_method="API",
    )
    db.add(source)
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    ds = Dataset(id=f"ds-{uid}", candidate_id=None, name="Churn Demo", source_id=source.id)
    db.add(ds)
    import hashlib

    db.add(DatasetFile(
        id=f"f-{uid}", dataset_id=ds.id, file_name="churn.csv",
        storage_key=str(path), storage_backend="local", file_format="csv",
        file_hash_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        validated=True,
    ))
    db.commit()
    return ds.id


def test_full_automl_run_flow(db, db_factory, tmp_path):
    dataset_id = _setup_dataset(db, tmp_path)

    # 1. plan: profiles + recommends, stops at approval gate
    row = runner.plan_run(db, dataset_id, target="churn", task="classification")
    assert row.status == "awaiting_approval", row.error
    assert row.task == "classification"
    assert len(row.recommendations) >= 1
    assert row.recommendations[0]["model"]

    # 2. user approves all recommended pipelines
    row = runner.approve_run(db, row.id, "all")
    assert row.status == "awaiting_approval"

    # 3. execute: trains, compares, selects, packages
    job = Job(id="job-automl", dataset_id=dataset_id, kind="AUTOML_RUN", status="QUEUED")
    db.add(job)
    db.commit()
    runner.execute_run(row.id, job.id, db_factory)

    db.refresh(row)
    assert row.status == "complete", (row.status, row.error)
    assert row.best_model is not None
    assert len(row.leaderboard) >= 1
    succeeded = [entry for entry in row.leaderboard if entry["status"] == "success"]
    assert succeeded, row.leaderboard
    assert any(entry["validation_score"] is not None for entry in succeeded)
    assert row.artifacts.get("model_zip"), row.artifacts

    job = db.get(Job, "job-automl")
    db.expire_all()  # listener sessions updated the job; drop the cached instance
    job = db.get(Job, "job-automl")
    assert job.status == "COMPLETED", (job.error_code, job.error_message)

    # 4. DB mirror is queryable for the UI
    rows = db.query(AutoMLRun).filter(AutoMLRun.dataset_id == dataset_id).all()
    assert len(rows) == 1 and rows[0].best_model


def test_execute_blocked_without_approval(db, db_factory, tmp_path):
    dataset_id = _setup_dataset(db, tmp_path)
    row = runner.plan_run(db, dataset_id, target="churn", task="classification")
    job = Job(id="job-automl2", dataset_id=dataset_id, kind="AUTOML_RUN", status="QUEUED")
    db.add(job)
    db.commit()

    runner.execute_run(row.id, job.id, db_factory)

    db.refresh(row)
    assert row.status == "failed"
    assert "approve" in (row.error or "").lower()
