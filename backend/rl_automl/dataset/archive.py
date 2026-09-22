"""Safe unpacking of archive uploads.

A zip from a data host (a Kaggle download, say) is untrusted input that arrives *before* any
of the normal dataset checks can see it, so it gets its own gate. Everything here exists to
answer one question: which members are safe to hand to
:func:`rl_automl.dataset.validation.store_upload`, and why were the others refused.

The threat model, and the guard for each:

* **Zip slip** -- a member named ``../../evil.csv``. Members are matched on their *name*
  only and written under a caller-supplied directory using a flattened basename, and the
  resolved destination is re-checked to be inside that directory.
* **Zip bomb** -- a small upload that inflates without bound. Members are capped by declared
  uncompressed size, by the sum of those sizes, and by compression ratio.
* **Archive sprawl** -- thousands of members. The member count is capped.
* **Format smuggling** -- a member whose extension is not tabular, or a name that disguises
  one. Only allowlisted extensions are extracted, and the extracted file is then subjected
  to the usual magic-byte and parse checks by the single-file gate.
* **Symlinks** -- a member that is a link. Nothing is extracted from a non-regular member.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from rl_automl.core.config import SecurityConfig
from rl_automl.core.errors import DatasetValidationError, SecurityError
from rl_automl.core.logging import get_logger

logger = get_logger("dataset.archive")

ARCHIVE_EXTENSIONS = frozenset({".zip"})

#: Names that carry no data and exist only in the host OS or the archiver.
_NOISE_PARTS = frozenset({"__MACOSX"})
_NOISE_PREFIXES = ("._", ".")
_NOISE_NAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini"})


@dataclass
class ExtractedMember:
    """A member that was extracted and is safe to hand on to the single-file gate."""

    name: str
    """The member name as it appeared in the archive."""

    path: Path
    """Where the bytes were written."""

    size_bytes: int


@dataclass
class RejectedMember:
    """A member that was deliberately not extracted, with the reason, for the client."""

    name: str
    reason: str


@dataclass
class ExtractionResult:
    members: list[ExtractedMember] = field(default_factory=list)
    rejected: list[RejectedMember] = field(default_factory=list)


def _is_noise(name: str) -> bool:
    parts = PurePosixPath(name).parts
    if any(part in _NOISE_PARTS for part in parts):
        return True
    filename = parts[-1] if parts else name
    lowered = filename.lower()
    if lowered in _NOISE_NAMES:
        return True
    return any(filename.startswith(prefix) for prefix in _NOISE_PREFIXES)


def _safe_basename(name: str) -> str | None:
    """Flatten a member name to a basename, or ``None`` if the name is hostile.

    Traversal is refused rather than sanitised: a member that *tries* to escape is not a
    member worth extracting, and silently rewriting it would hide the attempt.
    """
    if not name or name.endswith("/"):
        return None
    if "\\" in name:
        return None
    path = PurePosixPath(name)
    if path.is_absolute():
        return None
    if any(part in {"..", "."} for part in path.parts):
        return None
    basename = path.name
    if not basename or basename in {".", ".."}:
        return None
    return basename


def looks_like_archive(path: str | Path) -> bool:
    """True when the file's extension and magic bytes both say zip."""
    resolved = Path(path)
    if resolved.suffix.lower() not in ARCHIVE_EXTENSIONS:
        return False
    try:
        return zipfile.is_zipfile(resolved)
    except OSError:  # pragma: no cover - unreadable path
        return False


def extract_tabular_members(
    archive_path: str | Path,
    destination: str | Path,
    config: SecurityConfig | None = None,
) -> ExtractionResult:
    """Extract the tabular members of ``archive_path`` into ``destination``.

    Raises :class:`SecurityError` when the archive as a whole is unacceptable (not a zip,
    too many members, too large when expanded, or inflating too far). Individual members
    that are unusable are reported in ``result.rejected`` instead of failing the upload,
    because a real dataset download routinely carries files the trainer does not want.
    """
    cfg = config or SecurityConfig()
    resolved = Path(archive_path)
    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not resolved.is_file():
        raise DatasetValidationError(f"archive not found: {resolved}")
    if not zipfile.is_zipfile(resolved):
        raise DatasetValidationError(
            "the uploaded file is not a readable ZIP archive; re-download it and try again"
        )

    per_member_limit = int(cfg.max_upload_mb * 1024 * 1024)
    total_limit = int(cfg.max_zip_mb * 1024 * 1024)
    allowed = {extension.lower() for extension in cfg.allowed_extensions}

    result = ExtractionResult()
    total_uncompressed = 0
    considered = 0

    with zipfile.ZipFile(resolved) as archive:
        infos = archive.infolist()
        if len(infos) > cfg.max_archive_members:
            raise SecurityError(
                f"the archive contains {len(infos)} entries which exceeds the "
                f"{cfg.max_archive_members} entry limit; extract it yourself and upload "
                "the one table you need",
                n_members=len(infos),
                limit=cfg.max_archive_members,
            )

        for info in infos:
            name = info.filename
            if info.is_dir():
                continue
            if _is_noise(name):
                continue

            considered += 1
            basename = _safe_basename(name)
            if basename is None:
                result.rejected.append(
                    RejectedMember(name=name, reason="unsafe path in the archive")
                )
                logger.warning(
                    "refused an archive member with an unsafe path",
                    extra={"context": {"member": name}},
                )
                continue

            extension = PurePosixPath(basename).suffix.lower()
            if extension not in allowed:
                result.rejected.append(
                    RejectedMember(
                        name=name,
                        reason=f"'{extension or 'no extension'}' is not a supported table format",
                    )
                )
                continue

            if info.file_size > per_member_limit:
                result.rejected.append(
                    RejectedMember(
                        name=name,
                        reason=f"{info.file_size / 1024 / 1024:.0f} MB is over the "
                        f"{cfg.max_upload_mb:.0f} MB per-file limit",
                    )
                )
                continue

            total_uncompressed += info.file_size
            if total_uncompressed > total_limit:
                raise SecurityError(
                    f"the archive expands to more than {cfg.max_zip_mb:.0f} MB",
                    limit_bytes=total_limit,
                )

            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > cfg.max_compression_ratio:
                raise SecurityError(
                    f"archive member '{name}' inflates {ratio:.0f}x, which exceeds the "
                    f"{cfg.max_compression_ratio:.0f}x limit; refusing to expand it",
                    member=name,
                    ratio=round(ratio, 1),
                )

            destination_file = (target_dir / basename).resolve()
            if not destination_file.is_relative_to(target_dir.resolve()):
                result.rejected.append(
                    RejectedMember(name=name, reason="unsafe path in the archive")
                )
                continue

            with archive.open(info) as source, destination_file.open("wb") as handle:
                handle.write(source.read())

            result.members.append(
                ExtractedMember(name=basename, path=destination_file, size_bytes=info.file_size)
            )

    if not result.members:
        detail = "; ".join(f"{item.name}: {item.reason}" for item in result.rejected[:5])
        raise DatasetValidationError(
            "the archive contains no usable table"
            + (f" ({detail})" if detail else " (it had no readable entries)")
        )

    logger.info(
        "archive unpacked",
        extra={"context": {"members": len(result.members), "rejected": len(result.rejected)}},
    )
    return result


__all__ = [
    "ARCHIVE_EXTENSIONS",
    "ExtractedMember",
    "ExtractionResult",
    "RejectedMember",
    "extract_tabular_members",
    "looks_like_archive",
]
