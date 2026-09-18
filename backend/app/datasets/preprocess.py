"""Dataset preprocessing: turns the raw downloaded file into an ML-ready one.

Applied steps (recorded in the report, never hidden):
  1. Drop exact duplicate rows
  2. Drop constant columns (no signal)
  3. Exclude identifier-like columns from the processed output
  4. Impute missing values (numeric -> median, categorical -> mode)
  5. Encode categoricals (one-hot <=12 uniques, ordinal codes otherwise)
  6. Report outliers (IQR) without silently altering values — scaling and
     model-aware transforms happen in the AutoML pipeline phase

The original file is preserved; the processed copy is a separate artifact.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.datasets.download import load_dataframe
from app.models import Dataset, DatasetFile, PreprocessReport, utcnow
from app.storage import open_artifact, store_artifact

MAX_PREPROCESS_ROWS = 200_000
ONE_HOT_MAX_UNIQUES = 12
_ID_LIKE = pd.Series(
    ["id", "uuid", "guid", "identifier", "index", "key", "code", "slug"]
)


def _is_id_like(name: str, unique_ratio: float) -> bool:
    lowered = name.lower()
    name_signal = any(lowered == t or lowered.endswith(f"_{t}") for t in _ID_LIKE)
    return name_signal and unique_ratio > 0.9


def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    input_shape = [int(df.shape[0]), int(df.shape[1])]
    missing_before = int(df.isna().sum().sum())
    report: dict = {
        "input_shape": input_shape,
        "missing_cells_before": missing_before,
        "rows_dropped_duplicates": 0,
        "constant_columns_removed": [],
        "identifier_columns_excluded": [],
        "imputed": {},  # column -> strategy
        "one_hot_encoded": {},  # column -> number of dummy columns
        "ordinal_encoded": {},  # column -> number of categories
        "outliers": {},  # column -> IQR outlier count
        "steps": [],
    }

    # 1. duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count:
        df = df.drop_duplicates().reset_index(drop=True)
    report["rows_dropped_duplicates"] = dup_count
    report["steps"].append(f"Dropped {dup_count} duplicate rows.")

    # 3. identifier-like columns
    id_cols = [
        str(c)
        for c in df.columns
        if _is_id_like(str(c), df[c].nunique(dropna=True) / max(1, df.shape[0]))
    ]
    if id_cols:
        df = df.drop(columns=id_cols)
        report["identifier_columns_excluded"] = id_cols
        report["steps"].append(f"Excluded identifier-like columns: {', '.join(id_cols)}.")

    # 2. constant columns
    constant_cols = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        report["constant_columns_removed"] = constant_cols
        report["steps"].append(f"Removed constant columns: {', '.join(constant_cols)}.")

    # 4/5. imputation + encoding per column
    encoded_frames: list[pd.DataFrame] = []
    for col in list(df.columns):
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            missing = int(series.isna().sum())
            if missing:
                df[col] = series.fillna(series.median())
                report["imputed"][str(col)] = "median"
            # outlier count (reported, not altered)
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
                if outliers:
                    report["outliers"][str(col)] = outliers
        else:
            missing = int(series.isna().sum())
            if missing:
                mode = series.mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else "missing"
                df[col] = series.fillna(fill)
                report["imputed"][str(col)] = "mode"
            n_unique = int(df[col].nunique(dropna=True))
            if 2 <= n_unique <= ONE_HOT_MAX_UNIQUES:
                dummies = pd.get_dummies(df[col], prefix=str(col), dtype="int8")
                encoded_frames.append(dummies)
                report["one_hot_encoded"][str(col)] = int(dummies.shape[1])
                df = df.drop(columns=[col])
            else:
                codes, _ = pd.factorize(df[col])
                df[col] = codes.astype("int32")
                report["ordinal_encoded"][str(col)] = n_unique

    if encoded_frames:
        df = pd.concat([df] + encoded_frames, axis=1)

    report["steps"].append("Imputed missing values (numeric: median, categorical: mode).")
    report["steps"].append(
        "Encoded categorical columns (one-hot for low cardinality, ordinal codes otherwise)."
    )
    report["steps"].append("Scaling and model-aware transforms are applied inside the AutoML pipeline (next phase).")

    report["output_shape"] = [int(df.shape[0]), int(df.shape[1])]
    report["missing_cells_after"] = int(df.isna().sum().sum())
    return df, report


def run_preprocessing(dataset_id: str, job_id: str, db_factory) -> None:
    from app.models import Job

    db: Session = db_factory()
    try:
        job = db.get(Job, job_id)
        dataset = db.get(Dataset, dataset_id)
        if dataset is None:
            if job:
                job.status = "FAILED"
                job.error_code = "NOT_FOUND"
                job.error_message = "Dataset not found."
                job.finished_at = utcnow()
                db.commit()
            return

        if job:
            job.status = "RUNNING"
            job.stage = "PREPROCESSING"
            job.progress = 0.2
            job.detail = "Building the ML-ready preprocessed copy."
            db.commit()

        raw_file = (
            db.query(DatasetFile)
            .filter(DatasetFile.dataset_id == dataset.id, DatasetFile.file_format.in_(["csv", "parquet"]))
            .order_by(DatasetFile.created_at.desc())
            .first()
        )
        if raw_file is None:
            if job:
                job.status = "FAILED"
                job.error_code = "FILE_NOT_FOUND"
                job.error_message = "No raw dataset file available to preprocess."
                job.finished_at = utcnow()
                db.commit()
            return

        try:
            local_path = open_artifact(raw_file.storage_key, raw_file.storage_backend)
            df, _skipped = load_dataframe(local_path, raw_file.file_format or "csv")
            if df.shape[0] > MAX_PREPROCESS_ROWS:
                df = df.head(MAX_PREPROCESS_ROWS)
            processed, report = preprocess_dataframe(df)
        except Exception as exc:
            if job:
                job.status = "FAILED"
                job.error_code = "PREPROCESS_FAILURE"
                job.error_message = f"Preprocessing failed: {exc}"[:500]
                job.finished_at = utcnow()
                db.commit()
            return

        with tempfile.TemporaryDirectory(prefix="automl-preprocess-") as tmp:
            tmp_path = Path(tmp) / "processed.parquet"
            processed.to_parquet(tmp_path, index=False)
            stored = store_artifact(str(tmp_path), key=f"{dataset.id}/processed.parquet")

        db.add(
            DatasetFile(
                id=_new_id(),
                dataset_id=dataset.id,
                file_name="processed.parquet",
                storage_key=stored["storage_key"],
                storage_backend=stored["storage_backend"],
                file_format="parquet",
                file_size_bytes=stored["size"],
                file_hash_sha256=stored["hash"],
                validated=True,
                validation_report={"kind": "processed", "report": report},
            )
        )
        db.add(PreprocessReport(id=_new_id(), dataset_id=dataset.id, report=report))

        if job:
            job.status = "COMPLETED"
            job.stage = "DONE"
            job.progress = 1.0
            job.detail = f"Preprocessed copy created: {report['output_shape'][0]} rows x {report['output_shape'][1]} columns."
            job.finished_at = utcnow()
        db.commit()
    finally:
        db.close()


def _new_id() -> str:
    import uuid

    return str(uuid.uuid4())
