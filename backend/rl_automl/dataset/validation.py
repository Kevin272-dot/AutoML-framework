"""Upload validation and safe reading (spec §32).

This is the trust boundary. Anything that arrives from a user passes through here before
any other module sees it. Rules:

* extension allowlist -- no pickle/joblib/npy/executable uploads, ever
* magic-byte sniffing so a renamed pickle cannot sneak past the extension check
* size, row and column caps
* column-name sanitisation (length, dunder, duplicates, control characters)
* reads are delegated to pandas with type constraints; nothing is ``eval``-ed

Model artifacts are a separate trust domain: they are only ever loaded from paths this
process created, which ``packaging`` enforces with recorded checksums.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rl_automl.core.config import SecurityConfig
from rl_automl.core.errors import DatasetValidationError, SecurityError
from rl_automl.core.logging import get_logger

logger = get_logger("dataset.validation")

TEXT_EXTENSIONS = frozenset({".csv", ".tsv", ".txt", ".jsonl", ".json"})
BINARY_EXTENSIONS = frozenset({".parquet"})

#: Leading bytes of formats that can execute code or unpickle objects on load.
_DANGEROUS_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x80\x02", "pickle protocol 2"),
    (b"\x80\x03", "pickle protocol 3"),
    (b"\x80\x04", "pickle protocol 4"),
    (b"\x80\x05", "pickle protocol 5"),
    (b"\x89HDF", "HDF5 container"),
    (b"\x93NUM", "numpy array"),
    (b"\x00\x00\x00\x00", "unknown binary header"),
)

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass
class ValidationReport:
    """What the gate did, recorded on the run for auditability."""

    path: str
    filename: str
    extension: str
    size_bytes: int
    sha256: str
    n_rows: int = 0
    n_cols: int = 0
    original_column_names: list[str] = field(default_factory=list)
    renamed_columns: dict[str, str] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "renamed_columns": dict(self.renamed_columns),
            "dropped_columns": list(self.dropped_columns),
            "warnings": list(self.warnings),
        }


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class DatasetValidator:
    def __init__(self, config: SecurityConfig | None = None) -> None:
        self.config = config or SecurityConfig()

    # -- checks ------------------------------------------------------------------

    def check_filename(self, filename: str) -> str:
        """Validate the extension against the allowlist and return it normalised."""
        name = Path(filename or "").name
        if not name:
            raise DatasetValidationError("uploaded file has no name")

        extension = Path(name).suffix.lower()
        if not extension:
            raise DatasetValidationError(
                f"file '{name}' has no extension; expected one of "
                f"{sorted(self.config.allowed_extensions)}"
            )
        if extension not in self.config.allowed_extensions:
            raise SecurityError(
                f"file type '{extension}' is not accepted",
                filename=name,
                allowed=sorted(self.config.allowed_extensions),
            )
        return extension

    def check_size(self, path: str | Path) -> int:
        resolved = Path(path)
        if not resolved.is_file():
            raise DatasetValidationError(f"dataset file not found: {resolved}")
        size = resolved.stat().st_size
        if size == 0:
            raise DatasetValidationError("dataset file is empty")
        limit = int(self.config.max_upload_mb * 1024 * 1024)
        if size > limit:
            raise SecurityError(
                f"dataset is {size / 1024 / 1024:.1f} MB which exceeds the "
                f"{self.config.max_upload_mb:.0f} MB limit",
                size_bytes=size,
                limit_bytes=limit,
            )
        return size

    def check_content(self, path: str | Path, extension: str) -> None:
        """Sniff the header so a renamed pickled object cannot reach ``pandas.read_pickle``."""
        with Path(path).open("rb") as handle:
            header = handle.read(16)

        for magic, label in _DANGEROUS_MAGIC:
            if header.startswith(magic):
                # A legitimate UTF-8 CSV/JSON will not begin with these control bytes.
                if magic == b"\x00\x00\x00\x00" and extension not in BINARY_EXTENSIONS:
                    raise SecurityError(
                        "file begins with null bytes; refusing to parse unknown binary content",
                        detected=label,
                    )
                raise SecurityError(
                    f"refusing to load serialized content ({label}); "
                    "upload a plain data file such as CSV or Parquet",
                    detected=label,
                )

        if header.startswith(b"PK\x03\x04") and extension != ".parquet":
            raise SecurityError(
                "file is a ZIP/OOXML/Office container, not tabular data",
                detected="zip container",
            )

        if extension in TEXT_EXTENSIONS:
            try:
                header.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    header.decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    raise DatasetValidationError(
                        "text dataset is not valid UTF-8; re-encode it as UTF-8 "
                        "(or upload Parquet) before submitting"
                    ) from exc

    # -- reading -----------------------------------------------------------------

    def read(
        self, path: str | Path, filename: str | None = None
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Validate, read and sanitise a dataset. Returns the frame and the audit report."""
        resolved = Path(path)
        display_name = filename or resolved.name
        extension = self.check_filename(display_name)
        size = self.check_size(resolved)
        self.check_content(resolved, extension)

        report = ValidationReport(
            path=str(resolved),
            filename=display_name,
            extension=extension,
            size_bytes=size,
            sha256=sha256_file(resolved),
        )

        frame = self._read_by_extension(resolved, extension, report)
        if frame is None or frame.empty:
            raise DatasetValidationError("dataset contains no rows")

        if frame.shape[0] > self.config.max_rows:
            raise SecurityError(
                f"dataset has {frame.shape[0]:,} rows which exceeds the "
                f"{self.config.max_rows:,} row limit",
                n_rows=int(frame.shape[0]),
            )
        if frame.shape[1] > self.config.max_cols:
            raise SecurityError(
                f"dataset has {frame.shape[1]:,} columns which exceeds the "
                f"{self.config.max_cols:,} column limit",
                n_cols=int(frame.shape[1]),
            )

        frame = self._sanitise_columns(frame, report)
        report.n_rows, report.n_cols = int(frame.shape[0]), int(frame.shape[1])
        logger.info(
            "dataset validated",
            extra={
                "context": {"name": display_name, **{"rows": report.n_rows, "cols": report.n_cols}}
            },
        )
        return frame, report

    def _read_by_extension(
        self, path: Path, extension: str, report: ValidationReport
    ) -> pd.DataFrame | None:
        if extension == ".parquet":
            return pd.read_parquet(path)
        if extension == ".jsonl":
            return pd.read_json(path, lines=True)
        if extension == ".json":
            return pd.read_json(path)

        separator = "\t" if extension == ".tsv" else None
        if extension == ".txt":
            # Only treat .txt as delimited data; sniff a separator.
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                sample = handle.readline()
            separator = "\t" if sample.count("\t") > sample.count(",") else ","

        kwargs: dict[str, object] = {"low_memory": False}
        if separator:
            kwargs["sep"] = separator
        frame = pd.read_csv(path, **kwargs)
        if frame.shape[1] == 1 and separator is None:
            report.warnings.append(
                "only one column detected; if this file is delimited by something other "
                "than a comma, re-export it as CSV or TSV"
            )
        return frame

    def _sanitise_columns(self, frame: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
        cfg = self.config
        report.original_column_names = [str(c) for c in frame.columns]
        rename: dict[object, str] = {}
        seen: dict[str, int] = {}
        drop: list[str] = []

        for column in frame.columns:
            raw = str(column)
            cleaned = _CONTROL_CHARS.sub("", raw).strip()

            if not cleaned:
                drop.append(raw)
                report.warnings.append("dropped a column whose name was empty")
                continue
            if cfg.reject_dunder_columns and cleaned.startswith("__") and cleaned.endswith("__"):
                drop.append(raw)
                report.warnings.append(f"dropped column with reserved name '{cleaned}'")
                continue
            if len(cleaned) > cfg.max_column_name_length:
                cleaned = cleaned[: cfg.max_column_name_length]
                report.warnings.append(
                    f"truncated an over-long column name to {cfg.max_column_name_length} characters"
                )
            if cleaned in seen:
                seen[cleaned] += 1
                unique = f"{cleaned}__{seen[cleaned]}"
                report.renamed_columns[raw] = unique
                rename[column] = unique
                continue
            seen[cleaned] = 1
            if cleaned != raw:
                report.renamed_columns[raw] = cleaned
            rename[column] = cleaned

        if drop:
            frame = frame.drop(columns=[c for c in frame.columns if str(c) in drop])
        if rename:
            frame = frame.rename(columns=rename)
        report.dropped_columns = drop
        return frame


def validate_and_read(
    path: str | Path,
    filename: str | None = None,
    config: SecurityConfig | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    return DatasetValidator(config).read(path, filename=filename)


def store_upload(
    source: str | Path,
    destination_dir: str | Path,
    filename: str | None = None,
    config: SecurityConfig | None = None,
) -> tuple[Path, ValidationReport]:
    """Validate an inbound file and copy it into the run's dataset directory.

    The file is copied only after it passes validation, so the artifact tree never
    contains something the gate rejected.
    """
    validator = DatasetValidator(config)
    frame, report = validator.read(source, filename=filename)  # raises on rejection

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / Path(report.filename).name
    if target.resolve() != Path(source).resolve():
        shutil.copyfile(source, target)

    # Persist the sanitised frame so downstream code never re-parses raw user input.
    cleaned = target.with_suffix(".cleaned.csv")
    frame.to_csv(cleaned, index=False)
    report.warnings.append(f"sanitised copy written to {cleaned.name}")
    return cleaned, report


__all__ = [
    "BINARY_EXTENSIONS",
    "TEXT_EXTENSIONS",
    "DatasetValidator",
    "ValidationReport",
    "sha256_file",
    "store_upload",
    "validate_and_read",
]
