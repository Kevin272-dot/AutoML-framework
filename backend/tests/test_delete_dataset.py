"""Dataset deletion: DB records + stored artifacts removed, candidate stays."""

from app.models import Dataset, DatasetCandidate, DatasetFile, DataSource
from app.api.datasets import delete_dataset  # route function used directly with a session


def _seed(db, tmp_path, unique: str):
    source = DataSource(
        id=f"src-{unique}", slug=f"hf-{unique}", name="HF", base_url="https://example.com",
        source_type="DATASET_CATALOG", access_method="API",
    )
    db.add(source)
    db.add(DatasetCandidate(
        id=f"cand-{unique}", source_id=source.id, source_dataset_id=f"u/ds-{unique}",
        canonical_url=f"https://example.com/{unique}", dedupe_key=f"https://example.com/{unique}",
        name="DS", download_available=True,
    ))
    artifact = tmp_path / f"artifact-{unique}.csv"
    artifact.write_text("a,b\n1,2\n")
    ds = Dataset(id=f"ds-{unique}", candidate_id=f"cand-{unique}", name="DS", source_id=source.id)
    db.add(ds)
    db.add(DatasetFile(
        id=f"f-{unique}", dataset_id=ds.id, file_name="data.csv",
        storage_key=str(artifact), storage_backend="local", file_format="csv", validated=True,
    ))
    db.commit()
    return ds.id, artifact


def test_delete_dataset_removes_records_and_artifact(db, tmp_path):
    ds_id, artifact = _seed(db, tmp_path, "del1")

    result = delete_dataset(ds_id, db)
    assert result["status"] == "DELETED"
    assert result["artifacts_removed"] == 1
    assert not artifact.exists()
    assert db.get(Dataset, ds_id) is None
    assert db.query(DatasetFile).filter(DatasetFile.dataset_id == ds_id).count() == 0


def test_delete_keeps_candidate_for_reselection(db, tmp_path):
    ds_id, _ = _seed(db, tmp_path, "del2")
    ds = db.get(Dataset, ds_id)
    candidate_id = ds.candidate_id

    delete_dataset(ds_id, db)
    assert db.get(DatasetCandidate, candidate_id) is not None


def test_delete_missing_dataset_404(db):
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        delete_dataset("does-not-exist", db)
    assert exc_info.value.status_code == 404
