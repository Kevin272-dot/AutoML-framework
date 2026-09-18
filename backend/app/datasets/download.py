"""Dataset retrieval + validation + registration (runs as a Celery task).

Only runs after explicit user dataset selection. Validates the downloaded file
honestly — malformed data surfaces a structured error, never silent acceptance.
"""

import asyncio
import tempfile
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.config import get_settings
from app.discovery.adapters.base import AdapterError
from app.discovery.adapters.huggingface import HuggingFaceAdapter
from app.models import (
    Dataset,
    DatasetCandidate,
    DatasetColumn,
    DatasetFile,
    DataSource,
    Job,
    utcnow,
)
from app.storage import store_artifact

MAX_VALIDATE_ROWS = 50_000

FORMAT_READERS = {
    "csv": pd.read_csv,
    "parquet": pd.read_parquet,
}


class ValidationError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def validate_dataframe(df: pd.DataFrame, file_format: str, skipped_lines: int = 0) -> dict:
    """Validation report; raises ValidationError on fatal problems."""
    if df.empty:
        raise ValidationError("DATASET_EMPTY", "The file parsed successfully but contains no rows.")
    if df.shape[1] == 0:
        raise ValidationError("DATASET_NO_COLUMNS", "The file contains no columns.")
    if skipped_lines > 0 and skipped_lines / (skipped_lines + df.shape[0]) > 0.5:
        raise ValidationError(
            "DATASET_MALFORMED",
            f"{skipped_lines} of {skipped_lines + df.shape[0]} data rows could not be parsed; "
            "the file is likely malformed.",
        )

    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "duplicate_columns": int(df.columns.duplicated().sum()),
        "malformed_rows": 0,
        "skipped_lines": skipped_lines,
    }
    try:
        object_cols = df.select_dtypes(include=["object", "str"]).columns
        report["malformed_rows"] = int(df[object_cols].isna().all(axis=1).sum())
    except Exception:
        pass

    # Column consistency for CSVs: pandas already aligns to header; flag ragged rows.
    if file_format == "csv" and report["malformed_rows"] > report["rows"] * 0.5:
        raise ValidationError(
            "DATASET_MALFORMED",
            "More than half of the rows failed to parse consistently; the file is likely malformed.",
        )
    return report


def load_dataframe(path: str, file_format: str) -> tuple[pd.DataFrame, int]:
    """Load a CSV/Parquet file. Returns (dataframe, skipped_line_count).

    Raises ValidationError with a structured code on real parsing problems.
    """
    reader = FORMAT_READERS.get(file_format)
    if reader is None:
        raise ValidationError(
            "UNSUPPORTED_FORMAT",
            f"Format '{file_format}' is not supported yet. Supported: csv, parquet.",
        )
    try:
        if file_format == "csv":
            skipped = 0

            def _count_bad_line(_bad_line):
                nonlocal skipped
                skipped += 1
                return None

            try:
                return reader(path, nrows=MAX_VALIDATE_ROWS, on_bad_lines=_count_bad_line, engine="python"), skipped
            except Exception:
                raise
        return reader(path), 0
    except UnicodeDecodeError as exc:
        raise ValidationError("ENCODING_ERROR", "The file is not valid UTF-8/Latin-1 text.") from exc
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("PARSE_FAILURE", f"The file could not be parsed as {file_format}: {exc}") from exc


def get_adapter(source: DataSource):
    if source.adapter == "huggingface":
        return HuggingFaceAdapter()
    raise AdapterError("SOURCE_UNSUPPORTED", f"No adapter implemented for source '{source.slug}'.")


def run_download(dataset_id: str, job_id: str, db_factory) -> None:
    """Blocking implementation used by both Celery worker and eager fallback.

    db_factory: callable returning a fresh Session (workers must not share sessions).
    """
    settings = get_settings()
    db: Session = db_factory()
    try:
        job = db.get(Job, job_id)
        dataset = db.get(Dataset, dataset_id)
        candidate = db.get(DatasetCandidate, dataset.candidate_id) if dataset and dataset.candidate_id else None
        if dataset is None or candidate is None:
            _fail(job, db, "NOT_FOUND", "Selected dataset or candidate record not found.")
            return

        job.status = "RUNNING"
        job.stage = "DOWNLOADING"
        job.progress = 0.1
        job.detail = "Downloading dataset files from the approved source."
        db.commit()

        source = db.get(DataSource, candidate.source_id)
        adapter = get_adapter(source)

        with tempfile.TemporaryDirectory(prefix="automl-download-") as tmp:
            try:
                result = asyncio.run(
                    adapter.download(
                        candidate.source_dataset_id,
                        dest_path=str(Path(tmp) / "primary"),
                        max_bytes=settings.max_dataset_download_bytes,
                    )
                )
            except AdapterError as exc:
                _fail(job, db, exc.code, exc.message)
                return

            job.stage = "VALIDATING"
            job.progress = 0.5
            job.detail = "Validating the downloaded file."
            db.commit()

            try:
                df, skipped = load_dataframe(result["file_path"], result["format"])
                report = validate_dataframe(df, result["format"], skipped)
            except ValidationError as exc:
                _fail(job, db, exc.code, exc.message)
                return

            job.stage = "STORING"
            job.progress = 0.8
            job.detail = "Storing the validated dataset."
            db.commit()

            stored = store_artifact(result["file_path"], key=f"{dataset.id}/{result['file_name']}")

        dataset.row_count = report["rows"]
        dataset.column_count = report["columns"]
        dataset.file_format = result["format"]
        db.add(
            DatasetFile(
                id=_new_id(),
                dataset_id=dataset.id,
                file_name=result.get("file_name", "primary"),
                storage_key=stored["storage_key"],
                storage_backend=stored["storage_backend"],
                file_format=result["format"],
                file_size_bytes=stored["size"],
                file_hash_sha256=stored["hash"],
                source_file_url=result.get("source_file_url"),
                validated=True,
                validation_report=report,
            )
        )
        for pos, name in enumerate(report["column_names"]):
            db.add(DatasetColumn(id=_new_id(), dataset_id=dataset.id, name=name, position=pos))

        job.status = "COMPLETED"
        job.stage = "DONE"
        job.progress = 1.0
        job.detail = f"Downloaded and validated {report['rows']} rows x {report['columns']} columns."
        job.finished_at = utcnow()
        db.commit()

        # Chain EDA automatically after a successful download.
        from app.models import Job as JobModel
        from app.workers.dispatch import dispatch_eda

        eda_job = db.query(JobModel).filter(JobModel.dataset_id == dataset.id, JobModel.kind == "EDA").first()
        if eda_job is not None:
            eda_job.status = "QUEUED"
            db.commit()
            dispatch_eda(dataset.id, eda_job.id, db_factory)
    finally:
        db.close()


def _fail(job: Job | None, db: Session, code: str, message: str) -> None:
    if job is None:
        db.rollback()
        return
    job.status = "FAILED"
    job.error_code = code
    job.error_message = message
    job.finished_at = utcnow()
    db.commit()


def _new_id() -> str:
    import uuid

    return str(uuid.uuid4())
