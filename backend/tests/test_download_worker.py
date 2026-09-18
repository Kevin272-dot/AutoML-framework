"""End-to-end download → validate → store → EDA chain with a fake adapter."""

import uuid

import pandas as pd
import pytest

from app.datasets.download import run_download
from app.models import Dataset, DatasetCandidate, DataSource, Job
import app.datasets.download as download_mod


class FakeAdapter:
    """Writes a small valid CSV, simulating the adapter contract."""

    async def download(self, dataset_id, dest_path, max_bytes):
        df = pd.DataFrame({
            "district": ["a", "b", "c"] * 10,
            "yield_tons": [1.5, 2.0, 3.5] * 10,
            "year": [2020] * 30,
        })
        df.to_csv(dest_path, index=False)
        import hashlib

        return {
            "file_path": dest_path,
            "size": 1234,
            "format": "csv",
            "hash": hashlib.sha256(b"x").hexdigest(),
            "source_file_url": "https://example.com/data.csv",
            "file_name": "data.csv",
        }


@pytest.fixture
def setup(db):
    uid = uuid.uuid4().hex[:8]
    source = DataSource(
        id=f"src-{uid}", slug=f"fake-hf-{uid}", name="HF", base_url="https://example.com",
        source_type="DATASET_CATALOG", access_method="API", adapter="huggingface",
    )
    db.add(source)
    cand = DatasetCandidate(
        id=f"cand-{uid}", source_id=source.id, source_dataset_id=f"u/ds-{uid}",
        canonical_url=f"https://example.com/ds/{uid}", dedupe_key=f"https://example.com/ds/{uid}",
        name="Test DS", download_available=True,
    )
    db.add(cand)
    ds = Dataset(id=f"ds-{uid}", candidate_id=cand.id, name="Test DS", source_id=source.id)
    db.add(ds)
    db.add(Job(id=f"job-dl-{uid}", dataset_id=ds.id, kind="DATASET_DOWNLOAD", status="QUEUED"))
    db.add(Job(id=f"job-eda-{uid}", dataset_id=ds.id, kind="EDA", status="QUEUED"))
    db.commit()
    return ds.id


def _run_chain(db, db_factory, ds_id):
    orig = download_mod.get_adapter
    download_mod.get_adapter = lambda source: FakeAdapter()
    try:
        job = db.query(Job).filter(Job.dataset_id == ds_id, Job.kind == "DATASET_DOWNLOAD").first()
        run_download(ds_id, job.id, db_factory)
        eda_job = db.query(Job).filter(Job.dataset_id == ds_id, Job.kind == "EDA").first()
        from app.datasets.eda import run_eda

        run_eda(ds_id, eda_job.id, db_factory)
    finally:
        download_mod.get_adapter = orig


def test_download_runs_and_registers(db, db_factory, setup):
    _run_chain(db, db_factory, setup)

    job = db.query(Job).filter(Job.dataset_id == setup, Job.kind == "DATASET_DOWNLOAD").first()
    assert job.status == "COMPLETED", (job.error_code, job.error_message)

    from app.models import DatasetFile, DatasetColumn

    f = db.query(DatasetFile).filter(DatasetFile.dataset_id == setup).first()
    assert f is not None and f.validated
    assert f.validation_report["rows"] == 30
    cols = db.query(DatasetColumn).filter(DatasetColumn.dataset_id == setup).all()
    assert [c.name for c in sorted(cols, key=lambda c: c.position)] == ["district", "yield_tons", "year"]


def test_eda_chains_after_download(db, db_factory, setup):
    from app.models import EDAReport

    _run_chain(db, db_factory, setup)

    report = db.query(EDAReport).filter(EDAReport.dataset_id == setup).first()
    assert report is not None
    assert report.report["shape"] == [30, 3]
    eda_job = db.query(Job).filter(Job.dataset_id == setup, Job.kind == "EDA").first()
    assert eda_job.status == "COMPLETED"
