"""Kaggle Datasets adapter — authenticated via stored Connections.

Uses the documented Kaggle API v1 with the user's own credentials
(basic auth `username:key`, stored as an encrypted Connection secret):
  - search:   GET /api/v1/datasets/list?search=...
  - metadata: GET /api/v1/datasets/list/{owner}/{slug}
  - download: GET /api/v1/datasets/download/{owner}/{slug} (zip) — the largest
              tabular file inside is extracted and returned as csv
Kaggle has no server-side preview API; preview is served by extracting the
dataset zip's tabular file (small datasets only).
"""

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from app.config import get_settings
from app.discovery.adapters.base import (
    AdapterError,
    AdapterMetadata,
    AdapterSearchResult,
    DatasetSourceAdapter,
)
from app.schemas import ParsedRequirements

_PREVIEW_MAX_BYTES = 32 * 1024 * 1024


class KaggleAdapter(DatasetSourceAdapter):
    slug = "kaggle"

    def __init__(self, credentials: str | None = None, client: httpx.AsyncClient | None = None):
        """credentials: 'username:key' from a stored Connection (decrypted server-side)."""
        settings = get_settings()
        self._base = "https://www.kaggle.com/api/v1"
        self._timeout = settings.http_timeout_seconds
        self._client = client
        self._auth = self._parse_credentials(credentials)

    @staticmethod
    def _parse_credentials(credentials: str | None):
        if not credentials or ":" not in credentials:
            return None
        username, key = credentials.split(":", 1)
        return (username, key)

    def _require_auth(self):
        if self._auth is None:
            raise AdapterError(
                "API_KEY_REQUIRED",
                "Kaggle requires credentials. Add them under Profile → Connections (format: username:key).",
            )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, auth=self._auth)
        return self._client

    async def search(self, requirements: ParsedRequirements) -> list[AdapterSearchResult]:
        from app.parsing.locations import location_variants

        self._require_auth()
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

        client = await self._get_client()
        results: list[AdapterSearchResult] = []
        seen: set[str] = set()
        for q in queries[:5]:
            try:
                resp = await client.get(f"{self._base}/datasets/list", params={"search": q, "sortBy": "relevance"})
                resp.raise_for_status()
            except (httpx.HTTPError, AdapterError):
                continue
            for item in resp.json():
                ref = item.get("ref")
                if not ref or ref in seen:
                    continue
                seen.add(ref)
                results.append(
                    AdapterSearchResult(
                        source_dataset_id=ref,
                        name=item.get("title") or ref.split("/")[-1],
                        url=f"https://www.kaggle.com/datasets/{ref}",
                        description=item.get("subtitle") if isinstance(item.get("subtitle"), str) else None,
                        owner=ref.split("/")[0],
                        tags=[t.get("name", "") for t in item.get("tags", []) if isinstance(t, dict)],
                        file_size_bytes=item.get("totalBytes") if isinstance(item.get("totalBytes"), int) else None,
                        download_available=True,
                        preview_available=False,  # no server-side preview; preview comes from the zip
                        raw=item,
                    )
                )
            if len(results) >= 12:
                break
        return results

    async def get_metadata(self, dataset_id: str) -> AdapterMetadata:
        self._require_auth()
        client = await self._get_client()
        try:
            resp = await client.get(f"{self._base}/datasets/list/{dataset_id}")
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            code = "DATASET_REMOVED" if exc.response.status_code == 404 else "API_FAILURE"
            raise AdapterError(code, f"Kaggle metadata request failed for '{dataset_id}'.") from exc
        except httpx.HTTPError as exc:
            raise AdapterError("SOURCE_UNREACHABLE", "Kaggle is unreachable.") from exc

        raw = resp.json()
        files = self._tabular_files(raw.get("files") or [])
        return AdapterMetadata(
            source_dataset_id=dataset_id,
            name=raw.get("title") or dataset_id.split("/")[-1],
            url=f"https://www.kaggle.com/datasets/{dataset_id}",
            description=raw.get("subtitle") if isinstance(raw.get("subtitle"), str) else None,
            license=None,  # license lives on the dataset page; not in the API payload
            owner=dataset_id.split("/")[0] if "/" in dataset_id else None,
            row_count=None,
            file_size_bytes=raw.get("totalBytes") if isinstance(raw.get("totalBytes"), int) else None,
            file_format=files[0]["format"] if files else None,
            raw=raw,
        )

    @staticmethod
    def _tabular_files(files: list[dict]) -> list[dict[str, Any]]:
        out = []
        for f in files:
            name = f.get("name") or ""
            suffix = Path(name).suffix.lower()
            if suffix in (".csv", ".parquet", ".tsv", ".json"):
                out.append({"name": name, "size": f.get("totalBytes"), "format": suffix.lstrip(".")})
        out.sort(key=lambda f: f.get("size") or 0, reverse=True)  # largest = primary file
        return out

    async def _fetch_zip(self, dataset_id: str, max_bytes: int, dest: Path) -> None:
        self._require_auth()
        client = await self._get_client()
        try:
            async with client.stream(
                "GET", f"{self._base}/datasets/download/{dataset_id}", follow_redirects=True, timeout=120.0
            ) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    async for chunk in resp.aiter_bytes(1024 * 512):
                        if fh.tell() + len(chunk) > max_bytes:
                            raise AdapterError(
                                "DATASET_TOO_LARGE",
                                f"The dataset exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                            )
                        fh.write(chunk)
        except httpx.HTTPError as exc:
            raise AdapterError("DOWNLOAD_FAILURE", "Download from Kaggle failed.") from exc

    @staticmethod
    def _extract_primary_tabular(zip_path: Path, dest: Path) -> tuple[str, int]:
        with zipfile.ZipFile(zip_path) as zf:
            members = [
                m for m in zf.infolist()
                if not m.is_dir() and Path(m.filename).suffix.lower() in (".csv", ".parquet", ".tsv")
            ]
            if not members:
                raise AdapterError(
                    "DOWNLOAD_UNAVAILABLE",
                    "The Kaggle archive contains no tabular data files.",
                )
            members.sort(key=lambda m: m.file_size, reverse=True)
            primary = members[0]
            dest.write_bytes(zf.read(primary))
            return Path(primary.filename).suffix.lstrip("."), primary.file_size

    async def get_preview(self, dataset_id: str, limit: int = 20) -> tuple[list[str], list[dict[str, Any]]]:
        with tempfile.TemporaryDirectory(prefix="automl-kaggle-preview-") as tmp:
            zip_path = Path(tmp) / "dataset.zip"
            await self._fetch_zip(dataset_id, _PREVIEW_MAX_BYTES, zip_path)
            fmt, _size = self._extract_primary_tabular(zip_path, Path(tmp) / "data")
            data_path = Path(tmp) / "data"
            if fmt == "parquet":
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path, nrows=limit)
            columns = [str(c) for c in df.columns]
            rows = [
                {str(k): self._clean_value(v) for k, v in row.items()}
                for row in df.head(limit).to_dict(orient="records")
            ]
        return columns, rows

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
            return [KaggleAdapter._clean_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): KaggleAdapter._clean_value(v) for k, v in value.items()}
        return value

    async def get_files(self, dataset_id: str) -> list[dict[str, Any]]:
        self._require_auth()
        client = await self._get_client()
        try:
            resp = await client.get(f"{self._base}/datasets/list/{dataset_id}")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise AdapterError("API_FAILURE", "Kaggle file listing failed.") from exc
        return self._tabular_files(resp.json().get("files") or [])

    async def download(self, dataset_id: str, dest_path: str, max_bytes: int) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="automl-kaggle-dl-") as tmp:
            zip_path = Path(tmp) / "dataset.zip"
            await self._fetch_zip(dataset_id, max_bytes, zip_path)
            fmt, inner_size = self._extract_primary_tabular(zip_path, Path(tmp) / "data")
            (Path(tmp) / "data").rename(dest_path)

        import hashlib

        digest = hashlib.sha256(dest_path.read_bytes()).hexdigest()
        return {
            "file_path": str(dest_path),
            "size": dest_path.stat().st_size,
            "format": fmt,
            "hash": digest,
            "source_file_url": f"https://www.kaggle.com/datasets/{dataset_id}",
            "file_name": dataset_id.split("/")[-1] + f".{fmt}",
        }
