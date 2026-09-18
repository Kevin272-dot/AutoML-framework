"""Hugging Face Datasets adapter.

Uses exclusively documented public APIs:
  - Catalog search:   GET https://huggingface.co/api/datasets?search=...
  - Dataset metadata: GET https://huggingface.co/api/datasets/{id}
  - Preview/statistics/parquet: https://datasets-server.huggingface.co/...
No credentials, no HTML scraping.
"""

import hashlib
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings
from app.discovery.adapters.base import (
    AdapterColumn,
    AdapterError,
    AdapterMetadata,
    AdapterSearchResult,
    DatasetSourceAdapter,
)
from app.schemas import ParsedRequirements

_TAG_TO_FORMAT = {"parquet": "parquet", "csv": "csv", "json": "json"}


def _domain_tag_filters(req: ParsedRequirements) -> list[str]:
    """HF supports tag filters like 'domain:agriculture' on some datasets; we rely on
    keyword search primarily and keep domain terms as additional query terms."""
    return req.domain or []


def _build_query(req: ParsedRequirements) -> str:
    parts = list(req.keywords)
    for loc in req.location:
        parts.append(loc)
    if not parts:
        parts = req.domain
    return " ".join(parts).strip()


def _guess_format(raw: dict) -> str | None:
    tags = [t.lower() for t in (raw.get("tags") or [])]
    for tag in tags:
        for fmt, marker in _TAG_TO_FORMAT.items():
            if marker in tag:
                return fmt
    return None


def _extract_license(raw: dict) -> str | None:
    for tag in raw.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    if isinstance(raw.get("license"), str):
        return raw["license"]
    return None


def _extract_date_coverage(raw: dict) -> tuple[str | None, str | None]:
    """Many dataset cards expose no temporal metadata; return None rather than guessing."""
    return None, None


class HuggingFaceAdapter(DatasetSourceAdapter):
    slug = "huggingface"

    def __init__(self, client: httpx.AsyncClient | None = None):
        settings = get_settings()
        self._api_base = settings.hf_api_base
        self._ds_server = settings.hf_datasets_server_base
        self._timeout = settings.http_timeout_seconds
        self._client = client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def search(self, requirements: ParsedRequirements) -> list[AdapterSearchResult]:
        query = _build_query(requirements)
        client = await self._get_client()
        results: list[AdapterSearchResult] = []

        async def _run_search(params: dict) -> httpx.Response:
            resp = await client.get(f"{self._api_base}/datasets", params=params)
            resp.raise_for_status()
            return resp

        searches = [{"search": query, "limit": 25, "full": "false"}]
        if _domain_tag_filters(requirements):
            for domain in requirements.domain[:2]:
                searches.append({"search": domain, "limit": 10, "full": "false"})
        if requirements.task:
            searches.append({"filter": "task_categories", "search": query, "limit": 10})

        seen_ids: set[str] = set()
        for params in searches:
            try:
                resp = await _run_search(params)
            except httpx.HTTPError:
                continue
            for item in resp.json():
                ds_id = item.get("id")
                if not ds_id or ds_id in seen_ids:
                    continue
                seen_ids.add(ds_id)
                results.append(
                    AdapterSearchResult(
                        source_dataset_id=ds_id,
                        name=item.get("name") or ds_id.split("/")[-1],
                        url=f"https://huggingface.co/datasets/{ds_id}",
                        description=item.get("description") if isinstance(item.get("description"), str) else None,
                        license=_extract_license(item),
                        owner=ds_id.split("/")[0] if "/" in ds_id else None,
                        tags=item.get("tags") or [],
                        download_available=True,
                        preview_available=True,
                        raw=item,
                    )
                )
        return results

    async def get_metadata(self, dataset_id: str) -> AdapterMetadata:
        client = await self._get_client()
        try:
            resp = await client.get(f"{self._api_base}/datasets/{dataset_id}")
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            code = "DATASET_REMOVED" if exc.response.status_code == 404 else "API_FAILURE"
            raise AdapterError(code, f"Hugging Face metadata request failed for '{dataset_id}'.") from exc
        except httpx.HTTPError as exc:
            raise AdapterError("SOURCE_UNREACHABLE", "Hugging Face is unreachable.") from exc

        raw = resp.json()

        row_count = column_count = None
        file_format = _guess_format(raw)
        columns: list[AdapterColumn] = []
        try:
            info = await client.get(f"{self._ds_server}/parquet", params={"dataset": dataset_id})
            if info.status_code == 200:
                file_format = "parquet"
        except httpx.HTTPError:
            pass

        try:
            stats_resp = await client.get(f"{self._ds_server}/statistics", params={"dataset": dataset_id})
            if stats_resp.status_code == 200:
                stats = stats_resp.json().get("statistics") or []
                if stats:
                    first = stats[0]
                    col_stats = first.get("column_statistics") or []
                    column_count = len(col_stats)
                    row_count = first.get("num_rows")
                    for cs in col_stats:
                        col = cs.get("column_statistics") or {}
                        dt = cs.get("column_type", "unknown")
                        sample = (col.get("frequencies") or {})
                        top_keys = list(sample.keys())[:3] if isinstance(sample, dict) else []
                        columns.append(
                            AdapterColumn(
                                name=cs.get("column_name", ""),
                                dtype=dt,
                                sample_values=top_keys,
                            )
                        )
        except httpx.HTTPError:
            pass

        date_start, date_end = _extract_date_coverage(raw)
        return AdapterMetadata(
            source_dataset_id=dataset_id,
            name=raw.get("name") or dataset_id.split("/")[-1],
            url=f"https://huggingface.co/datasets/{dataset_id}",
            description=raw.get("description") if isinstance(raw.get("description"), str) else None,
            license=_extract_license(raw),
            owner=dataset_id.split("/")[0] if "/" in dataset_id else None,
            tags=raw.get("tags") or [],
            row_count=row_count,
            column_count=column_count,
            file_format=file_format,
            columns=columns,
            date_start=date_start,
            date_end=date_end,
            raw=raw,
        )

    async def get_preview(self, dataset_id: str, limit: int = 20) -> tuple[list[str], list[dict[str, Any]]]:
        client = await self._get_client()
        try:
            resp = await client.get(
                f"{self._ds_server}/rows",
                params={"dataset": dataset_id, "config": "default", "split": "train", "offset": 0, "length": limit},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise AdapterError("PREVIEW_UNAVAILABLE", "This dataset has no server-side preview (config/split may differ).") from exc
            raise AdapterError("API_FAILURE", "Preview request to Hugging Face failed.") from exc
        except httpx.HTTPError as exc:
            raise AdapterError("SOURCE_UNREACHABLE", "Hugging Face is unreachable.") from exc

        payload = resp.json()
        rows = [r.get("row", {}) for r in payload.get("rows", [])]
        columns = [c.get("column", {}).get("name", "") for c in payload.get("features", [])]
        if not columns and rows:
            columns = list(rows[0].keys())
        return columns, rows

    async def get_files(self, dataset_id: str) -> list[dict[str, Any]]:
        client = await self._get_client()
        files: list[dict[str, Any]] = []
        try:
            resp = await client.get(f"{self._ds_server}/parquet", params={"dataset": dataset_id})
            if resp.status_code == 200:
                for pf in resp.json().get("parquet_files", [])[:20]:
                    files.append(
                        {
                            "name": pf.get("filename") or pf.get("url", "").split("/")[-1],
                            "url": pf.get("url"),
                            "size": pf.get("size"),
                            "format": "parquet",
                        }
                    )
        except httpx.HTTPError:
            pass
        return files

    async def download(self, dataset_id: str, dest_path: str, max_bytes: int) -> dict[str, Any]:
        client = await self._get_client()
        files = await self.get_files(dataset_id)
        if not files:
            raise AdapterError("DOWNLOAD_UNAVAILABLE", "No machine-readable data files exposed for this dataset.")

        target = files[0]
        url = target.get("url")
        if not url:
            raise AdapterError("DOWNLOAD_UNAVAILABLE", "Data file has no download URL.")

        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256()
        size = 0
        try:
            async with client.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    async for chunk in resp.aiter_bytes(1024 * 512):
                        size += len(chunk)
                        if size > max_bytes:
                            raise AdapterError(
                                "DATASET_TOO_LARGE",
                                f"Dataset exceeds the {max_bytes // (1024 * 1024)} MB download limit.",
                            )
                        fh.write(chunk)
                        sha.update(chunk)
        except httpx.HTTPError as exc:
            raise AdapterError("DOWNLOAD_FAILURE", "Download from Hugging Face failed.") from exc

        return {
            "file_path": str(dest),
            "size": size,
            "format": "parquet",
            "hash": sha.hexdigest(),
            "source_file_url": url,
            "file_name": target.get("name", dest.name),
        }
