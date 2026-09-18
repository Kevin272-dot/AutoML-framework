"""Normalize adapter results into the internal DatasetMetadata shape.

raw_metadata is always preserved — source-specific fields are never discarded.
"""

from typing import Any

from app.discovery.adapters.base import AdapterMetadata, AdapterSearchResult

SEMANTIC_NUMERIC = {"int", "float", "numerical"}
SEMANTIC_CATEGORICAL = {"categorical", "string", "bool"}


def _semantic_type(dtype: str | None) -> str:
    if not dtype:
        return "unknown"
    d = dtype.lower()
    if any(t in d for t in ("int", "float", "double", "numeric")):
        return "numeric"
    if any(t in d for t in ("string", "category", "bool", "enum")):
        return "categorical"
    if "date" in d or "time" in d:
        return "datetime"
    return "other"


def normalize_search_result(source_id: str, result: AdapterSearchResult) -> dict[str, Any]:
    """Normalized candidate record (pre-ranking)."""
    name_norm = result.name.strip().lower()
    return {
        "source_id": source_id,
        "source_dataset_id": result.source_dataset_id,
        "canonical_url": result.url,
        "dedupe_key": f"{result.url.rstrip('/').lower()}",
        "name": result.name,
        "name_normalized": name_norm,
        "description": result.description,
        "license": result.license,
        "owner": result.owner,
        "tags": result.tags or [],
        "domains": [],
        "row_count": result.row_count,
        "column_count": result.column_count,
        "file_format": result.file_format,
        "file_size_bytes": result.file_size_bytes,
        "download_available": result.download_available,
        "preview_available": result.preview_available,
        "raw_metadata": result.raw or {},
    }


def normalize_metadata(source_id: str, meta: AdapterMetadata) -> dict[str, Any]:
    """Full normalized metadata incl. column list (used at detail/enrichment time)."""
    columns = [
        {
            "name": c.name,
            "dtype": c.dtype,
            "semantic_type": _semantic_type(c.dtype),
            "nullable": c.nullable,
            "sample_values": c.sample_values,
        }
        for c in meta.columns
    ]
    return {
        "source_id": source_id,
        "source_dataset_id": meta.source_dataset_id,
        "canonical_url": meta.url,
        "dedupe_key": meta.url.rstrip("/").lower(),
        "name": meta.name,
        "name_normalized": meta.name.strip().lower(),
        "description": meta.description,
        "license": meta.license,
        "owner": meta.owner,
        "tags": meta.tags or [],
        "domains": [],
        "date_start": meta.date_start,
        "date_end": meta.date_end,
        "row_count": meta.row_count,
        "column_count": meta.column_count,
        "file_format": meta.file_format,
        "file_size_bytes": meta.file_size_bytes,
        "columns": columns,
        "download_available": True,
        "preview_available": True,
        "raw_metadata": meta.raw or {},
    }
