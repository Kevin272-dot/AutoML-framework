import numpy as np
import pandas as pd
import pytest

from app.datasets.eda import _classify_target_candidates, _correlations, run_eda
from app.models import Dataset, DatasetFile, Job


def _write_csv(tmp_path, df: pd.DataFrame) -> str:
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_target_candidates_binary_column(tmp_path):
    df = pd.DataFrame({
        "customer_id": range(100),
        "age": np.random.default_rng(0).integers(18, 80, 100),
        "churn": [0, 1] * 50,
        "notes": ["text"] * 100,  # constant text column
    })
    candidates, task, target = _classify_target_candidates(df)
    assert task == "classification"
    # Binary 'churn' should outrank the high-cardinality 'age'.
    assert candidates[0]["column"] == "churn"
    assert 0 < candidates[0]["confidence"] <= 1
    assert all(c["column"] != "customer_id" for c in candidates)  # ID excluded


def test_target_candidates_regression(tmp_path):
    df = pd.DataFrame({
        "price": np.random.default_rng(1).normal(100, 20, 200).round(2),
        "area": np.random.default_rng(2).normal(80, 15, 200).round(2),
    })
    candidates, task, target = _classify_target_candidates(df)
    assert task == "regression"


def test_correlations_matrix_shape():
    df = pd.DataFrame({
        "a": np.random.default_rng(3).normal(size=50),
        "b": np.random.default_rng(4).normal(size=50),
        "text": ["x"] * 50,
    })
    corr = _correlations(df)
    assert set(corr.keys()) == {"a", "b"}
    assert abs(corr["a"]["a"] - 1.0) < 1e-6


def _setup_dataset(db, tmp_path, df):
    import uuid

    ds_id = str(uuid.uuid4())
    ds = Dataset(id=ds_id, name="Test Dataset")
    db.add(ds)
    path = _write_csv(tmp_path, df)
    db.add(DatasetFile(
        id=str(uuid.uuid4()), dataset_id=ds_id, file_name="data.csv",
        storage_key=path, storage_backend="local", file_format="csv",
        validated=True,
    ))
    db.add(Job(id=str(uuid.uuid4()), dataset_id=ds_id, kind="EDA", status="QUEUED"))
    db.commit()
    job = db.query(Job).filter(Job.dataset_id == ds_id).first()
    return ds_id, job.id


def test_run_eda_full_report(db, tmp_path):
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "id": range(60),
        "yield": (rng.normal(4, 1, 60)).round(2),
        "rain": (rng.normal(800, 100, 60)).round(1),
        "region": rng.choice(["north", "south", "east"], 60),
        "constant": ["same"] * 60,
    })
    df.loc[3, "rain"] = np.nan
    ds_id, job_id = _setup_dataset(db, tmp_path, df)

    run_eda(ds_id, job_id, lambda: db)
    from app.models import EDAReport

    report_row = db.query(EDAReport).filter(EDAReport.dataset_id == ds_id).first()
    assert report_row is not None
    r = report_row.report
    assert r["shape"] == [60, 5]
    assert r["duplicate_rows"] == 0
    names = {c["name"] for c in r["column_stats"]}
    assert names == {"id", "yield", "rain", "region", "constant"}
    constant = next(c for c in r["column_stats"] if c["name"] == "constant")
    assert constant["is_constant"] is True
    assert any("constant" in w.lower() for w in r["quality_warnings"])
    assert any("identifier" in w.lower() for w in r["quality_warnings"])
    assert r["suggested_target"] is not None
    assert r["target_candidates"], "expected at least one target candidate"
    job = db.query(Job).filter(Job.dataset_id == ds_id, Job.kind == "EDA").first()
    assert job.status == "COMPLETED"


def test_run_eda_class_balance(db, tmp_path):
    df = pd.DataFrame({
        "label": ["yes"] * 90 + ["no"] * 10,
        "feature": np.random.default_rng(6).normal(size=100),
    })
    ds_id, job_id = _setup_dataset(db, tmp_path, df)
    run_eda(ds_id, job_id, lambda: db)
    from app.models import EDAReport

    r = db.query(EDAReport).filter(EDAReport.dataset_id == ds_id).first().report
    cb = r["class_balance"]
    assert cb["n_classes"] == 2
    assert cb["imbalanced"] is True
    job = db.query(Job).filter(Job.id == job_id).first()
    assert job.status == "COMPLETED"
