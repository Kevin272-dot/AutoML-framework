"""Dataset source adapter interface.

The discovery engine contains no source-specific logic; every source is accessed
through an adapter implementing this interface. Adapters only use documented,
official access mechanisms (official APIs / download endpoints) — never HTML
scraping.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdapterSearchResult:
    source_dataset_id: str
    name: str
    url: str
    description: str | None = None
    license: str | None = None
    owner: str | None = None
    tags: list[str] = field(default_factory=list)
    row_count: int | None = None
    column_count: int | None = None
    file_format: str | None = None
    file_size_bytes: int | None = None
    download_available: bool = False
    preview_available: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterColumn:
    name: str
    dtype: str
    nullable: bool = True
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class AdapterMetadata:
    source_dataset_id: str
    name: str
    url: str
    description: str | None = None
    license: str | None = None
    owner: str | None = None
    tags: list[str] = field(default_factory=list)
    row_count: int | None = None
    column_count: int | None = None
    file_format: str | None = None
    file_size_bytes: int | None = None
    columns: list[AdapterColumn] = field(default_factory=list)
    date_start: str | None = None
    date_end: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class AdapterError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class DatasetSourceAdapter(ABC):
    """Contract for one data source."""

    slug: str = ""

    @abstractmethod
    async def search(self, requirements) -> list[AdapterSearchResult]:
        """Search the source's catalog using parsed requirements."""

    @abstractmethod
    async def get_metadata(self, dataset_id: str) -> AdapterMetadata:
        """Full metadata for one dataset."""

    @abstractmethod
    async def get_preview(self, dataset_id: str, limit: int = 20) -> tuple[list[str], list[dict[str, Any]]]:
        """Real sample rows: (columns, rows)."""

    @abstractmethod
    async def get_files(self, dataset_id: str) -> list[dict[str, Any]]:
        """File listing: [{name, url, size, format}]."""

    @abstractmethod
    async def download(self, dataset_id: str, dest_path: str, max_bytes: int) -> dict[str, Any]:
        """Stream dataset file(s) to dest_path. Returns {file_path, size, format, hash}."""
