"""Hugging Face Datasets adapter.

Uses exclusively documented public APIs:
  - Catalog search:   GET https://huggingface.co/api/datasets?search=...
  - Dataset metadata: GET https://huggingface.co/api/datasets/{id}
  - Preview/statistics/parquet: https://datasets-server.huggingface.co/...
No credentials, no HTML scraping.
"""

import asyncio
import hashlib
import math
import tempfile
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pandas import Timestamp

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

    # Upstream (datasets-server) occasionally returns transient 5xx/429s; retry briefly.
    RETRY_STATUS = {429, 500, 502, 503, 504}
    PREVIEW_FALLBACK_MAX_BYTES = 64 * 1024 * 1024

    async def _request_with_retry(self, url: str, *, params: dict | None = None, attempts: int = 3) -> httpx.Response:
        """GET with short backoff on transient upstream failures."""
        client = await self._get_client()
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = await client.get(url, params=params, follow_redirects=True)
                if resp.status_code in self.RETRY_STATUS and attempt < attempts - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                return resp
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < attempts - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise
        raise last_error if last_error else AdapterError("API_FAILURE", "Request failed.")

    # Tabular file extensions we can preview/download directly from a repo.
    TABULAR_EXTENSIONS = {".csv", ".parquet", ".tsv", ".jsonl", ".json"}

    async def _list_repo_files(self, dataset_id: str) -> list[dict[str, Any]]:
        """List tabular data files in the dataset repo via the documented HF API.
        Used when the datasets-server has no (or failed) parquet conversion."""
        try:
            resp = await self._request_with_retry(f"{self._api_base}/datasets/{dataset_id}/tree/main")
        except (httpx.HTTPError, AdapterError):
            return []
        if resp.status_code != 200:
            return []
        files: list[dict[str, Any]] = []
        for entry in resp.json():
            if entry.get("type") != "file":
                continue
            path = entry.get("path", "")
            suffix = Path(path).suffix.lower()
            if suffix not in self.TABULAR_EXTENSIONS:
                continue
            files.append(
                {
                    "name": path,
                    "url": f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{path}",
                    "size": entry.get("size"),
                    "format": suffix.lstrip("."),
                }
            )
        files.sort(key=lambda f: f.get("size") or 0)
        return files

    async def _download_bytes(self, url: str, max_bytes: int, dest: Path) -> tuple[int, str]:
        """Stream a file to dest with a hard size cap. Returns (size, sha256)."""
        client = await self._get_client()
        sha = hashlib.sha256()
        size = 0
        try:
            async with client.stream("GET", url, follow_redirects=True, timeout=60.0) as stream:
                stream.raise_for_status()
                with open(dest, "wb") as fh:
                    async for chunk in stream.aiter_bytes(1024 * 512):
                        size += len(chunk)
                        if size > max_bytes:
                            raise AdapterError(
                                "DATASET_TOO_LARGE",
                                f"The file exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                            )
                        fh.write(chunk)
                        sha.update(chunk)
        except httpx.HTTPError as exc:
            raise AdapterError("DOWNLOAD_FAILURE", "Could not download the data file.") from exc
        return size, sha.hexdigest()

    def _dataframe_to_rows(self, df: pd.DataFrame, limit: int) -> tuple[list[str], list[dict[str, Any]]]:
        columns = [str(c) for c in df.columns]
        rows: list[dict[str, Any]] = []
        for record in df.head(limit).to_dict(orient="records"):
            clean: dict[str, Any] = {}
            for key, value in record.items():
                if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NaT:
                    clean[str(key)] = None
                elif isinstance(value, Timestamp):
                    clean[str(key)] = value.isoformat()
                elif hasattr(value, "item"):
                    clean[str(key)] = value.item()
                else:
                    clean[str(key)] = value
            rows.append(clean)
        return columns, rows

    async def _preview_from_file(
        self, url: str, file_format: str, limit: int
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Read real rows from a machine-readable data file (parquet or csv)."""
        with tempfile.TemporaryDirectory(prefix="automl-preview-") as tmp:
            dest = Path(tmp) / "preview-file"
            try:
                await self._download_bytes(url, self.PREVIEW_FALLBACK_MAX_BYTES, dest)
            except AdapterError as exc:
                code = "DATASET_TOO_LARGE" if exc.code == "DATASET_TOO_LARGE" else "SOURCE_UNREACHABLE"
                raise AdapterError(
                    code,
                    "The data file is too large for a server-side preview; "
                    "select the dataset to download it instead."
                    if code == "DATASET_TOO_LARGE"
                    else "Could not download the data file for preview.",
                ) from exc
            try:
                if file_format == "parquet":
                    df = pd.read_parquet(dest)
                else:
                    df = pd.read_csv(dest, nrows=limit, on_bad_lines="skip")
            except Exception as exc:
                raise AdapterError("PREVIEW_UNAVAILABLE", "The data file could not be read for preview.") from exc
            return self._dataframe_to_rows(df, limit)

    async def _preview_fallback(self, dataset_id: str, limit: int) -> tuple[list[str], list[dict[str, Any]]]:
        """Fallback chain when datasets-server rows is unavailable:
        1) converted parquet files via datasets-server, 2) original repo files."""
        try:
            resp = await self._request_with_retry(f"{self._ds_server}/parquet", params={"dataset": dataset_id})
            if resp.status_code == 200:
                files = resp.json().get("parquet_files") or []
                if files and files[0].get("url"):
                    return await self._preview_from_file(files[0]["url"], "parquet", limit)
        except AdapterError as exc:
            if exc.code in ("DATASET_TOO_LARGE",):
                raise
            # fall through to repo files
        except httpx.HTTPError:
            pass

        for repo_file in await self._list_repo_files(dataset_id):
            try:
                return await self._preview_from_file(repo_file["url"], repo_file["format"], limit)
            except AdapterError as exc:
                if exc.code == "DATASET_TOO_LARGE":
                    raise
                continue  # try the next file
        raise AdapterError(
            "PREVIEW_UNAVAILABLE",
            "No readable data file is available for preview right now; "
            "selecting the dataset will attempt a full download.",
        )

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
        try:
            resp = await self._request_with_retry(
                f"{self._ds_server}/rows",
                params={"dataset": dataset_id, "config": "default", "split": "train", "offset": 0, "length": limit},
            )
        except (httpx.HTTPError, AdapterError):
            # Datasets-server unreachable/transient — fall back to real data files.
            return await self._preview_fallback(dataset_id, limit)

        if resp.status_code != 200:
            try:
                return await self._preview_fallback(dataset_id, limit)
            except AdapterError:
                if resp.status_code == 404:
                    raise AdapterError(
                        "PREVIEW_UNAVAILABLE",
                        "This dataset has no server-side preview (config/split may differ).",
                    ) from None
                raise AdapterError(
                    "PREVIEW_UNAVAILABLE",
                    "The source preview service is temporarily unavailable; try again shortly.",
                ) from None

        payload = resp.json()
        if "error" in payload:
            return await self._preview_fallback(dataset_id, limit)
        rows = [r.get("row", {}) for r in payload.get("rows", [])]
        columns = [c.get("column", {}).get("name", "") for c in payload.get("features", [])]
        if not columns and rows:
            columns = list(rows[0].keys())
        return columns, rows

    async def get_files(self, dataset_id: str) -> list[dict[str, Any]]:
        """All machine-readable data files: datasets-server parquet conversions first,
        then original tabular files in the repo (for datasets without a conversion)."""
        client = await self._get_client()
        files: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        try:
            resp = await client.get(f"{self._ds_server}/parquet", params={"dataset": dataset_id})
            if resp.status_code == 200:
                for pf in resp.json().get("parquet_files", [])[:20]:
                    name = pf.get("filename") or pf.get("url", "").split("/")[-1]
                    if name in seen_names:
                        continue
                    seen_names.add(name)
                    files.append(
                        {
                            "name": name,
                            "url": pf.get("url"),
                            "size": pf.get("size"),
                            "format": "parquet",
                        }
                    )
        except httpx.HTTPError:
            pass
        for repo_file in await self._list_repo_files(dataset_id):
            if repo_file["name"] in seen_names:
                continue
            seen_names.add(repo_file["name"])
            files.append(repo_file)
        return files

    async def download(self, dataset_id: str, dest_path: str, max_bytes: int) -> dict[str, Any]:
        files = await self.get_files(dataset_id)
        if not files:
            raise AdapterError("DOWNLOAD_UNAVAILABLE", "No machine-readable data files exposed for this dataset.")

        target = files[0]
        url = target.get("url")
        if not url:
            raise AdapterError("DOWNLOAD_UNAVAILABLE", "Data file has no download URL.")

        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        size, digest = await self._download_bytes(url, max_bytes, dest)

        return {
            "file_path": str(dest),
            "size": size,
            "format": target.get("format") or "csv",
            "hash": digest,
            "source_file_url": url,
            "file_name": target.get("name", dest.name),
        }
