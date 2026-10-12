"""EU Open Data Portal (data.europa.eu) adapter — the second real source.

Uses the documented data.europa.eu Search API (public, no auth):
  - search:   GET /api/hub/search/search?q=...&limit=...
  - metadata: GET /api/hub/search/datasets/{id}
  - preview:  direct download of a tabular distribution (capped) read via pandas
No scraping; every URL comes from the API's own distributions.
"""

import tempfile
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from app.config import get_settings
from app.discovery.adapters.base import (
    AdapterColumn,
    AdapterError,
    AdapterMetadata,
    AdapterSearchResult,
    DatasetSourceAdapter,
)
from app.schemas import ParsedRequirements

_TABULAR_FORMATS = {"csv", "parquet", "tsv", "json"}
_PREVIEW_MAX_BYTES = 64 * 1024 * 1024


def _en(value: Any) -> Any:
    """Hub fields are multilingual dicts {'en': ..., 'fr': ...}; pick the English text."""
    if isinstance(value, dict):
        for key in ("en", "eng"):
            if isinstance(value.get(key), str):
                return value[key]
        for v in value.values():
            if isinstance(v, str):
                return v
        return None
    return value if isinstance(value, str) else None


def _first_url(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[0], str):
        return value[0]
    return None


def _format_label(fmt_raw: Any) -> str:
    """format may be a str, a dict with a label dict/str, or None."""
    if isinstance(fmt_raw, str):
        return fmt_raw.lower()
    if isinstance(fmt_raw, dict):
        label = fmt_raw.get("label")
        if isinstance(label, str):
            return label.lower()
        if isinstance(label, dict):
            return (_en(label) or "").lower()
    return ""


class EUOpenDataAdapter(DatasetSourceAdapter):
    slug = "eu-open-data"

    def __init__(self, client: httpx.AsyncClient | None = None):
        settings = get_settings()
        self._base = "https://data.europa.eu/api/hub/search"
        self._timeout = settings.http_timeout_seconds
        self._client = client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _get(self, path: str, params: dict | None = None) -> dict:
        client = await self._get_client()
        try:
            resp = await client.get(f"{self._base}{path}", params=params, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            code = "DATASET_REMOVED" if exc.response.status_code == 404 else "API_FAILURE"
            raise AdapterError(code, f"data.europa.eu returned HTTP {exc.response.status_code}.") from exc
        except httpx.HTTPError as exc:
            raise AdapterError("SOURCE_UNREACHABLE", "The EU Open Data Portal is unreachable.") from exc
        return resp.json().get("result") or {}


    def _to_search_result(self, item: dict) -> AdapterSearchResult | None:
        ds_id = item.get("id")
        if not ds_id:
            return None
        keywords = item.get("keywords") or {}
        kw_list = []
        if isinstance(keywords, dict):
            for v in keywords.values():
                if isinstance(v, str):
                    kw_list.append(v)
                elif isinstance(v, list):
                    kw_list.extend([x for x in v if isinstance(x, str)])
        dists = item.get("distributions") or []
        formats = {
            _format_label(d.get("format"))
            for d in dists
        }
        fmt = next((f for f in ("csv", "parquet", "tsv", "json") if f in formats), None)
        return AdapterSearchResult(
            source_dataset_id=ds_id,
            name=_en(item.get("title")) or ds_id,
            url=f"https://data.europa.eu/data/datasets/{ds_id}",
            description=_en(item.get("description")),
            owner=_en((item.get("publisher") or {}).get("label")) if isinstance(item.get("publisher"), dict) else None,
            tags=list(dict.fromkeys(kw_list))[:12],
            file_format=fmt,
            download_available=True,
            preview_available=True,
            raw=item,
        )

    async def search(self, requirements: ParsedRequirements) -> list[AdapterSearchResult]:
        from app.parsing.locations import location_variants

        queries: list[str] = []
        base = list(requirements.keywords)
        locs = location_variants(requirements)

        def add(parts: list[str]) -> None:
            q = " ".join(p for p in parts if p).strip()
            if q and q not in queries:
                queries.append(q)

        if base:
            add(base + locs[:1])
            add(base)
            add(base[:2])
        if requirements.domain:
            add(requirements.domain + locs[:1])
            add(requirements.domain[:1])
        for loc in locs[:2]:
            add(loc)
        if not queries:
            add(requirements.domain)

        results: list[AdapterSearchResult] = []
        seen: set[str] = set()
        for q in queries[:6]:
            try:
                result = await self._get("/search", {"q": q, "limit": 20})
            except AdapterError:
                continue
            for item in result.get("results", []):
                converted = self._to_search_result(item)
                if converted is None or converted.source_dataset_id in seen:
                    continue
                seen.add(converted.source_dataset_id)
                results.append(converted)
            if len(results) >= 12:
                break
        return results

    def _distributions(self, raw: dict) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        for d in raw.get("distributions") or []:
            fmt = _format_label(d.get("format"))
            fmt = "csv" if "csv" in fmt else ("parquet" if "parquet" in fmt else ("json" if "json" in fmt else fmt))
            url = _first_url(d.get("download_url")) or _first_url(d.get("access_url"))
            if not url:
                continue
            files.append({"name": _en(d.get("title")) or url.split("/")[-1] or "distribution", "url": url, "format": fmt})
        return files

    async def get_metadata(self, dataset_id: str) -> AdapterMetadata:
        raw = await self._get(f"/datasets/{dataset_id}")
        dists = self._distributions(raw)
        keywords = raw.get("keywords") or {}
        kw_list = []
        if isinstance(keywords, dict):
            for v in keywords.values():
                if isinstance(v, str):
                    kw_list.append(v)
                elif isinstance(v, list):
                    kw_list.extend([x for x in v if isinstance(x, str)])
        columns = [AdapterColumn(name=k, dtype="unknown") for k in kw_list[:20]]
        return AdapterMetadata(
            source_dataset_id=dataset_id,
            name=_en(raw.get("title")) or dataset_id,
            url=f"https://data.europa.eu/data/datasets/{dataset_id}",
            description=_en(raw.get("description")),
            license=_en((raw.get("licence") or {}).get("label")) if isinstance(raw.get("licence"), dict) else None,
            owner=_en((raw.get("publisher") or {}).get("label")) if isinstance(raw.get("publisher"), dict) else None,
            tags=list(dict.fromkeys(kw_list))[:12],
            file_format=dists[0]["format"] if dists else None,
            columns=columns,
            raw=raw,
        )

    async def get_preview(self, dataset_id: str, limit: int = 20) -> tuple[list[str], list[dict[str, Any]]]:
        raw = await self._get(f"/datasets/{dataset_id}")
        dists = self._distributions(raw)
        if not dists:
            raise AdapterError("PREVIEW_UNAVAILABLE", "This dataset has no downloadable distributions.")

        client = await self._get_client()
        last_error: AdapterError | None = None
        for f in dists[:5]:
            if f["format"] not in _TABULAR_FORMATS:
                continue
            try:
                with tempfile.TemporaryDirectory(prefix="automl-eu-preview-") as tmp:
                    dest = Path(tmp) / "preview-file"
                    resp = await client.get(f["url"], follow_redirects=True, timeout=60.0)
                    resp.raise_for_status()
                    if len(resp.content) > _PREVIEW_MAX_BYTES:
                        raise AdapterError(
                            "PREVIEW_UNAVAILABLE",
                            "The data file is too large for a server-side preview; select the dataset to download it.",
                        )
                    dest.write_bytes(resp.content)
                    if f["format"] == "parquet":
                        df = pd.read_parquet(dest)
                    elif f["format"] == "json":
                        df = pd.read_json(dest)
                    else:
                        df = pd.read_csv(dest, nrows=limit, on_bad_lines="skip")
                return [str(c) for c in df.columns], [
                    {str(k): self._clean_value(v) for k, v in row.items()}
                    for row in df.head(limit).to_dict(orient="records")
                ]
            except AdapterError as exc:
                last_error = exc
                continue
            except (httpx.HTTPError, pd.errors.ParserError, ValueError, UnicodeDecodeError):
                last_error = AdapterError("PREVIEW_UNAVAILABLE", "The distribution could not be read.")
                continue  # landing-page URLs and exotic formats: try the next distribution
        raise last_error or AdapterError(
            "PREVIEW_UNAVAILABLE",
            "None of the dataset's distributions could be read for preview.",
        )

    @staticmethod
    def _clean_value(value: Any) -> Any:
        if value is None or value is pd.NaT:
            return None
        if isinstance(value, float) and value != value:
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                return str(value)
        if isinstance(value, (list, tuple)):
            return [_clean_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _clean_value(v) for k, v in value.items()}
        return value

    async def get_files(self, dataset_id: str) -> list[dict[str, Any]]:
        raw = await self._get(f"/datasets/{dataset_id}")
        return self._distributions(raw)

    async def download(self, dataset_id: str, dest_path: str, max_bytes: int) -> dict[str, Any]:
        raw = await self._get(f"/datasets/{dataset_id}")
        dists = self._distributions(raw)
        target = next((f for f in dists if f["format"] in _TABULAR_FORMATS), None)
        if target is None:
            raise AdapterError(
                "DOWNLOAD_UNAVAILABLE",
                "This dataset exposes no machine-readable data files (only landing pages/documents).",
            )

        client = await self._get_client()
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        import hashlib

        digest = hashlib.sha256()
        try:
            async with client.stream("GET", target["url"], follow_redirects=True, timeout=120.0) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    async for chunk in resp.aiter_bytes(1024 * 512):
                        size += len(chunk)
                        if size > max_bytes:
                            raise AdapterError(
                                "DATASET_TOO_LARGE",
                                f"The file exceeds the {max_bytes // (1024 * 1024)} MB download limit.",
                            )
                        fh.write(chunk)
                        digest.update(chunk)
        except httpx.HTTPError as exc:
            raise AdapterError("DOWNLOAD_FAILURE", "Download from data.europa.eu failed.") from exc
        return {
            "file_path": str(dest),
            "size": size,
            "format": target["format"],
            "hash": digest.hexdigest(),
            "source_file_url": target["url"],
            "file_name": target["name"],
        }
