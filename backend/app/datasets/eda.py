"""Automatic Exploratory Data Analysis (read-only over the stored dataset).

Produces a structured report: shape, per-column stats, missingness, duplicates,
distributions, correlations, outliers, class balance, potential IDs, and target
candidates with explicit confidence + rationale. Never modifies the source data.
"""

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.datasets.download import load_dataframe
from app.models import Dataset, DatasetFile, EDAReport, utcnow
from app.storage import open_artifact

MAX_EDA_ROWS = 100_000
MAX_CORRELATION_COLS = 25
HISTOGRAM_BINS = 20


def _potential_id(col: pd.Series, name: str) -> bool:
    if col.nunique(dropna=True) == col.dropna().shape[0] and col.dropna().shape[0] > 1:
        lowered = name.lower()
        return lowered.endswith("_id") or lowered in ("id", "uuid", "index", "key") or "identifier" in lowered
    return False


def _numeric_stats(col: pd.Series) -> dict:
    s = col.dropna()
    out: dict = {}
    if s.empty:
        return out
    try:
        out.update(
            min=float(s.min()),
            max=float(s.max()),
            mean=float(s.mean()),
            median=float(s.median()),
            std=float(s.std()) if s.count() > 1 else None,
        )
    except (TypeError, ValueError):
        return out
    # histogram
    try:
        counts, edges = np.histogram(s.astype(float), bins=HISTOGRAM_BINS)
        out["histogram"] = [
            (round(float(edges[i]), 4), int(counts[i])) for i in range(len(counts)) if counts[i] > 0
        ]
    except (ValueError, TypeError):
        pass
    # IQR outliers
    try:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
            out["outlier_count"] = int(outliers.shape[0])
    except (ValueError, TypeError):
        pass
    return out


def _categorical_stats(col: pd.Series) -> dict:
    vc = col.value_counts(dropna=False).head(10)
    return {"top_values": [(str(k), int(v)) for k, v in vc.items()]}


def _classify_target_candidates(df: pd.DataFrame) -> tuple[list[dict], str | None, str | None]:
    """Suggest target candidates with explicit rationale + confidence."""
    candidates: list[dict] = []
    for name in df.columns:
        col = df[name]
        if _potential_id(col, name):
            continue
        n_unique = int(col.nunique(dropna=True))
        missing_ratio = float(col.isna().mean())
        if missing_ratio > 0.5:
            continue
        dtype = col.dtype
        if pd.api.types.is_numeric_dtype(dtype):
            if n_unique == 2:
                candidates.append(
                    {
                        "column": str(name),
                        "task": "classification",
                        "confidence": 0.85 - 0.1 * missing_ratio,
                        "rationale": f"Binary numeric column ({n_unique} distinct values).",
                    }
                )
            elif n_unique <= 20 and pd.api.types.is_integer_dtype(dtype):
                candidates.append(
                    {
                        "column": str(name),
                        "task": "classification",
                        "confidence": 0.55 - 0.1 * missing_ratio,
                        "rationale": f"Low-cardinality integer column ({n_unique} distinct values).",
                    }
                )
            else:
                candidates.append(
                    {
                        "column": str(name),
                        "task": "regression",
                        "confidence": 0.45 - 0.1 * missing_ratio,
                        "rationale": "Continuous numeric column with many distinct values.",
                    }
                )
        elif pd.api.types.is_string_dtype(dtype) or str(dtype) == "category":
            if 2 <= n_unique <= 50:
                candidates.append(
                    {
                        "column": str(name),
                        "task": "classification",
                        "confidence": 0.6 - 0.1 * missing_ratio,
                        "rationale": f"Categorical column ({n_unique} distinct values).",
                    }
                )
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    top = candidates[:5]
    suggested_target = top[0]["column"] if top else None
    suggested_task = top[0]["task"] if top else None
    return top, suggested_task, suggested_target


def _class_balance(df: pd.DataFrame, target: str | None) -> dict | None:
    if not target or target not in df.columns:
        return None
    vc = df[target].value_counts(dropna=False)
    total = int(vc.sum())
    return {
        "counts": {str(k): int(v) for k, v in vc.head(20).items()},
        "ratios": {str(k): round(float(v) / total, 4) for k, v in vc.head(20).items()},
        "n_classes": int(vc.shape[0]),
        "imbalanced": bool(vc.iloc[0] / total >= 0.9) if total else False,
    }


def _correlations(df: pd.DataFrame) -> dict | None:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    if numeric.shape[1] > MAX_CORRELATION_COLS:
        # Keep columns with the most variance signal to keep the matrix readable.
        variances = numeric.var().sort_values(ascending=False)
        numeric = numeric[variances.index[:MAX_CORRELATION_COLS]]
    try:
        corr = numeric.corr(numeric_only=True).round(3)
    except Exception:
        return None
    return {c: {r: float(corr.loc[c, r]) for r in corr.columns} for c in corr.columns}


def run_eda(dataset_id: str, job_id: str, db_factory) -> None:
    db: Session = db_factory()
    try:
        from app.models import Job

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
            job.stage = "EDA"
            job.progress = 0.2
            job.detail = "Running exploratory data analysis."
            db.commit()

        dfile = db.query(DatasetFile).filter(DatasetFile.dataset_id == dataset.id).first()
        if dfile is None:
            if job:
                job.status = "FAILED"
                job.error_code = "FILE_NOT_FOUND"
                job.error_message = "No validated dataset file registered."
                job.finished_at = utcnow()
                db.commit()
            return

        local_path = open_artifact(dfile.storage_key, dfile.storage_backend)
        df, _skipped = load_dataframe(local_path, dfile.file_format or "csv")

        column_stats = []
        warnings: list[str] = []
        for name in df.columns:
            col = df[name]
            missing_count = int(col.isna().sum())
            unique_count = int(col.nunique(dropna=True))
            stats = {
                "name": str(name),
                "dtype": str(col.dtype),
                "semantic_type": "numeric" if pd.api.types.is_numeric_dtype(col) else (
                    "datetime" if pd.api.types.is_datetime64_any_dtype(col) else
                    ("categorical" if unique_count <= max(50, df.shape[0] * 0.05) else "text")
                ),
                "missing_count": missing_count,
                "missing_ratio": round(float(col.isna().mean()), 4),
                "unique_count": unique_count,
                "unique_ratio": round(float(unique_count) / df.shape[0], 4) if df.shape[0] else 0.0,
                "is_constant": bool(unique_count <= 1),
                "is_potential_id": _potential_id(col, str(name)),
            }
            if pd.api.types.is_numeric_dtype(col):
                stats.update(_numeric_stats(col))
            else:
                stats.update(_categorical_stats(col))
            if stats["missing_ratio"] > 0.4:
                warnings.append(f"Column '{name}' is missing {stats['missing_ratio'] * 100:.0f}% of values.")
            if stats["is_constant"]:
                warnings.append(f"Column '{name}' is constant and carries no signal.")
            if stats["is_potential_id"]:
                warnings.append(f"Column '{name}' looks like an identifier and should be excluded from features.")
            column_stats.append(stats)

        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            warnings.append(f"{dup_count} duplicate rows detected.")

        target_candidates, suggested_task, suggested_target = _classify_target_candidates(df)
        report = {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "column_stats": column_stats,
            "duplicate_rows": dup_count,
            "correlation_matrix": _correlations(df),
            "class_balance": _class_balance(df, suggested_target),
            "quality_warnings": warnings,
            "target_candidates": target_candidates,
            "suggested_task": suggested_task,
            "suggested_target": suggested_target,
            "row_limit_applied": df.shape[0] >= MAX_EDA_ROWS,
        }

        db.add(EDAReport(id=_new_id(), dataset_id=dataset.id, report=report))
        dataset.row_count = report["shape"][0]
        dataset.column_count = report["shape"][1]

        if job:
            job.status = "COMPLETED"
            job.stage = "DONE"
            job.progress = 1.0
            job.detail = "EDA complete."
            job.finished_at = utcnow()
        db.commit()
    finally:
        db.close()


def _new_id() -> str:
    import uuid

    return str(uuid.uuid4())
