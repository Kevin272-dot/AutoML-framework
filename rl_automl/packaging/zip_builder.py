"""Archive building and verification (spec §23).

The requirement is specific: the ZIP "must be generated server-side and verified before
being presented as downloadable". Verification here is not a checksum formality. It:

1. re-opens the archive and tests every member's CRC
2. rejects zip-slip member paths (absolute paths or ``..`` traversal)
3. checks the required members exist and the archive respects the size cap
4. **extracts to a temporary directory, imports the generated ``model_loader``, loads the
   packaged model, predicts on a probe sample and compares the result against the
   in-process model**

Step 4 is the one that matters. A package that cannot be loaded by its own loader, or that
disagrees with the model that produced it, is a broken deliverable -- and that is exactly
the failure mode that a checksum alone would miss.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.errors import ArtifactIntegrityError, PackagingError
from rl_automl.core.logging import get_logger

logger = get_logger("packaging.zip_builder")

REQUIRED_MODEL_MEMBERS = (
    "metadata/feature_schema.json",
    "metadata/model_metadata.json",
    "model/model.joblib",
    "preprocessing/preprocessor.joblib",
    "inference/model_loader.py",
    "inference/predict.py",
    "requirements.txt",
)

REGRESSION_TOLERANCE = 1e-4

# Executed in a clean interpreter with the extraction directory as the working directory,
# so the artifact has to be genuinely self-contained to load. argv: package, probe, output.
_CLEAN_ROOM_SCRIPT = """
import json
import sys
from pathlib import Path

root, probe, destination = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, str(Path(root) / "inference"))

from model_loader import load_model

model = load_model(root)
predictions = list(model.predict(probe))


def plain(value):
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return str(value)
    return value


Path(destination).write_text(
    json.dumps(
        {
            "predictions": [plain(value) for value in predictions],
            "description": model.describe(),
        }
    ),
    encoding="utf-8",
)
"""


@dataclass
class ArchiveVerification:
    ok: bool
    sha256: str = ""
    n_members: int = 0
    size_bytes: int = 0
    problems: list[str] = field(default_factory=list)
    round_trip_ok: bool = False
    max_prediction_difference: float | None = None
    loaded_description: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "sha256": self.sha256,
            "n_members": self.n_members,
            "size_bytes": self.size_bytes,
            "problems": list(self.problems),
            "round_trip_ok": self.round_trip_ok,
            "max_prediction_difference": self.max_prediction_difference,
            "loaded_description": self.loaded_description,
        }


@dataclass
class BuiltArchive:
    path: Path
    sha256: str
    n_files: int
    size_bytes: int
    members: list[str] = field(default_factory=list)
    verification: ArchiveVerification | None = None

    @property
    def verified(self) -> bool:
        return bool(self.verification and self.verification.ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "n_files": self.n_files,
            "size_bytes": self.size_bytes,
            "verified": self.verified,
            "verification": self.verification.to_dict() if self.verification else None,
        }


class ZipBuilder:
    def __init__(self, config: AutoMLConfig | None = None) -> None:
        self.config = config or AutoMLConfig()

    # -- building ----------------------------------------------------------------

    def build(
        self,
        source_dir: str | Path,
        destination: str | Path,
        *,
        arcname: str | None = None,
        required_members: tuple[str, ...] = REQUIRED_MODEL_MEMBERS,
        probe_frame: pd.DataFrame | None = None,
        expected_predictions: np.ndarray | None = None,
        task_type: str = "",
        verify: bool = True,
    ) -> BuiltArchive:
        """Zip a package directory and verify the result before returning it."""
        source = Path(source_dir)
        if not source.is_dir():
            raise PackagingError(f"nothing to archive: {source} is not a directory")

        members = sorted(path for path in source.rglob("*") if path.is_file())
        if not members:
            raise PackagingError(f"nothing to archive: {source} contains no files")

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        prefix = arcname if arcname is not None else source.name

        try:
            with zipfile.ZipFile(
                target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as archive:
                for member in members:
                    archive.write(member, arcname=f"{prefix}/{member.relative_to(source)}")
        except OSError as exc:
            raise PackagingError(f"could not write archive {target}: {exc}") from exc

        built = BuiltArchive(
            path=target,
            sha256=sha256_file(target),
            n_files=len(members),
            size_bytes=target.stat().st_size,
            members=[
                f"{prefix}/{member.relative_to(source)}".replace("\\", "/") for member in members
            ],
        )

        if verify:
            built.verification = self.verify(
                target,
                required_members=required_members,
                probe_frame=probe_frame,
                expected_predictions=expected_predictions,
                task_type=task_type,
            )
            if not built.verification.ok:
                raise ArtifactIntegrityError(
                    "the generated archive failed verification and will not be offered "
                    "for download",
                    problems=built.verification.problems,
                    path=str(target),
                )

        logger.info(
            "archive built",
            extra={
                "context": {
                    "path": str(target),
                    "files": built.n_files,
                    "size_kb": round(built.size_bytes / 1024, 1),
                    "verified": built.verified,
                }
            },
        )
        return built

    def build_file_archive(
        self,
        payload: dict[str, str | bytes | Path],
        destination: str | Path,
        *,
        arcname: str = "results",
        verify: bool = True,
    ) -> BuiltArchive:
        """Archive loose files (the results bundle) with the same integrity checks."""
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)

        written: list[str] = []
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in payload.items():
                member = f"{arcname}/{name}"
                if isinstance(content, Path):
                    archive.write(content, arcname=member)
                else:
                    data = content.encode("utf-8") if isinstance(content, str) else content
                    archive.writestr(member, data)
                written.append(member)

        built = BuiltArchive(
            path=target,
            sha256=sha256_file(target),
            n_files=len(written),
            size_bytes=target.stat().st_size,
            members=written,
        )
        if verify:
            built.verification = self.verify(
                target, required_members=tuple(written), probe_frame=None, expected_predictions=None
            )
            if not built.verification.ok:
                raise ArtifactIntegrityError(
                    "the results archive failed verification",
                    problems=built.verification.problems,
                )
        return built

    # -- verification ------------------------------------------------------------

    def verify(
        self,
        archive_path: str | Path,
        *,
        required_members: tuple[str, ...] = (),
        probe_frame: pd.DataFrame | None = None,
        expected_predictions: np.ndarray | None = None,
        task_type: str = "",
    ) -> ArchiveVerification:
        """Full integrity check, including an end-to-end reload of the packaged model."""
        path = Path(archive_path)
        problems: list[str] = []

        if not path.is_file():
            return ArchiveVerification(ok=False, problems=[f"archive not found: {path}"])

        verification = ArchiveVerification(
            ok=False,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )

        try:
            with zipfile.ZipFile(path) as archive:
                verification.n_members = len(archive.namelist())

                corrupt = archive.testzip()
                if corrupt is not None:
                    problems.append(f"member failed its CRC check: {corrupt}")

                for name in archive.namelist():
                    issue = _unsafe_member(name)
                    if issue:
                        problems.append(issue)

                names = set(archive.namelist())
                for required in required_members:
                    if not any(name.endswith(required) for name in names):
                        problems.append(f"missing required member: {required}")
        except zipfile.BadZipFile as exc:
            return ArchiveVerification(
                ok=False,
                problems=[f"not a readable ZIP archive: {exc}"],
                sha256=verification.sha256,
            )

        limit_bytes = int(self.config.security.max_zip_mb * 1024 * 1024)
        if verification.size_bytes > limit_bytes:
            problems.append(
                f"archive is {verification.size_bytes / 1024 / 1024:.1f} MB, above the "
                f"{self.config.security.max_zip_mb:.0f} MB limit"
            )

        if probe_frame is not None and expected_predictions is not None:
            round_trip = self._round_trip_check(path, probe_frame, expected_predictions, task_type)
            verification.round_trip_ok = round_trip["ok"]
            verification.max_prediction_difference = round_trip.get("max_difference")
            verification.loaded_description = round_trip.get("description")
            problems.extend(round_trip["problems"])
        else:
            verification.round_trip_ok = False
            # Not a failure: some archives (the results bundle) carry no model.
            verification.loaded_description = None

        verification.problems = problems
        verification.ok = not problems
        if not verification.ok:
            logger.warning(
                "archive verification failed",
                extra={"context": {"path": str(path), "problems": problems[:5]}},
            )
        return verification

    def _round_trip_check(
        self,
        archive_path: Path,
        probe_frame: pd.DataFrame,
        expected_predictions: np.ndarray,
        task_type: str,
    ) -> dict[str, Any]:
        """Extract, then load and predict **in a clean interpreter**.

        Running the packaged loader in a subprocess whose working directory is the
        extraction directory -- and whose ``PYTHONPATH`` is cleared -- is the point of the
        check. An in-process reload would still see this project on ``sys.path`` and would
        happily load an artifact that cannot load anywhere else.
        """
        problems: list[str] = []
        result: dict[str, Any] = {"ok": False, "problems": problems, "max_difference": None}

        temporary = Path(tempfile.mkdtemp(prefix="automl_verify_"))
        try:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(temporary)

            schema_path = next(temporary.rglob("feature_schema.json"), None)
            if schema_path is None:
                problems.append("the archive does not contain a loadable model_package")
                return result
            package_root = schema_path.parent.parent

            probe_path = temporary / "probe.csv"
            output_path = temporary / "predictions.json"
            probe_frame.to_csv(probe_path, index=False)

            runner = [
                sys.executable,
                "-c",
                _CLEAN_ROOM_SCRIPT,
                str(package_root),
                str(probe_path),
                str(output_path),
            ]
            environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
            completed = subprocess.run(
                runner,
                cwd=str(temporary),
                capture_output=True,
                text=True,
                timeout=600,
                env=environment,
            )
            if completed.returncode != 0 or not output_path.is_file():
                detail = (completed.stderr or completed.stdout or "").strip().splitlines()
                problems.append(
                    "the packaged model could not be loaded in a clean environment: "
                    + (detail[-1] if detail else f"exit code {completed.returncode}")
                )
                return result

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            actual = np.asarray(payload.get("predictions", []), dtype=object)
            expected = np.asarray(expected_predictions)
            if actual.size != expected.size:
                problems.append(
                    f"the packaged model returned {actual.size} predictions "
                    f"for {expected.size} rows"
                )
                return result
            # A representation difference is not a mismatch; the comparison below handles
            # it. Only a failed cast falls back to the raw values.
            with contextlib.suppress(TypeError, ValueError):
                actual = actual.astype(expected.dtype)

            difference = _prediction_difference(actual, expected, task_type)
            result["max_difference"] = difference
            result["description"] = payload.get("description")

            if difference is None:
                problems.append("packaged predictions could not be compared with the model")
            elif difference > REGRESSION_TOLERANCE:
                problems.append(
                    f"packaged predictions differ from the in-process model by {difference:.6g} "
                    f"(tolerance {REGRESSION_TOLERANCE})"
                )
            else:
                result["ok"] = True
        except subprocess.TimeoutExpired:
            problems.append("loading the packaged model in a clean environment timed out")
        except Exception as exc:  # pragma: no cover - defensive
            problems.append(f"verification raised {type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

        return result


def _prediction_difference(
    actual: np.ndarray, expected: np.ndarray, task_type: str
) -> float | None:
    """Zero for identical classes/values; max absolute difference for continuous output.

    Class labels travel through JSON and through the packaged loader, so the two sides can
    legitimately differ in dtype while agreeing in value (``0`` vs ``"0"``). Numeric output is
    compared numerically; anything else is compared on string form before being declared a
    mismatch, so a representation difference is never mistaken for a broken artifact.
    """
    if actual.shape != expected.shape:
        try:
            actual = actual.reshape(expected.shape)
        except ValueError:
            return None

    numeric = {"i", "u", "f", "b"}
    if task_type == "regression" or (
        expected.dtype.kind in numeric and actual.dtype.kind in numeric
    ):
        try:
            return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        except (TypeError, ValueError):
            return None

    if np.array_equal(actual, expected):
        return 0.0
    try:
        equal = np.array_equal(
            np.asarray(actual).astype(str).reshape(-1),
            np.asarray(expected).astype(str).reshape(-1),
        )
    except (TypeError, ValueError):
        return None
    return 0.0 if equal else float("inf")


def _unsafe_member(name: str) -> str | None:
    """Reject zip-slip style member names."""
    normalised = name.replace("\\", "/")
    if normalised.startswith("/") or (len(normalised) > 1 and normalised[1] == ":"):
        return f"member uses an absolute path: {name}"
    if ".." in normalised.split("/"):
        return f"member escapes the archive root: {name}"
    return None


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "REQUIRED_MODEL_MEMBERS",
    "ArchiveVerification",
    "BuiltArchive",
    "ZipBuilder",
    "sha256_file",
]
