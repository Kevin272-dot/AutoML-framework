"""Curated public datasets, pinned by checksum (spec §34).

This is a **third trust tier**, and it is deliberately not the same thing as the upload
gate in :mod:`rl_automl.dataset.validation`:

* user uploads are untrusted bytes, so they are sniffed, allowlisted and sanitised;
* model artifacts are only ever loaded from paths this process created;
* these datasets are fetched from a short, hard-coded list of public URLs, and every
  download must match a **SHA-256 pinned at the time the entry was added**. A changed or
  substituted file fails closed rather than being parsed.

Because the pin guarantees identity, the parsing rules per dataset (delimiter, header,
column names, missing-value markers) can live here next to the URL instead of being
guessed at read time. Each download is normalised to a plain comma-separated CSV in the
cache, so everything downstream -- profiling, splitting, training, packaging -- sees an
ordinary frame and no special cases.

Offline by default: nothing is fetched unless ``allow_network=True`` is passed, and a
cached copy is always preferred. Licences and attribution are recorded per dataset and
written alongside the cached file as provenance.

To refresh a pin, download the file, verify it by hand, and update ``sha256`` here; the
registry validates its own shape at import, so a malformed entry fails immediately.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from rl_automl.core.errors import ConfigError, DatasetValidationError, SecurityError
from rl_automl.core.logging import get_logger
from rl_automl.core.types import TaskType, utc_now_iso

logger = get_logger("dataset.remote")

DEFAULT_CACHE_SUBDIR = "datasets/remote"
DOWNLOAD_TIMEOUT_S = 120.0
USER_AGENT = "rl-automl/0.1 (curated public dataset fetch)"

UCI_LICENSE = "CC BY 4.0 (UCI Machine Learning Repository)"
UCI_ATTRIBUTION = "Dua, D. and Graff, C. (2019). UCI Machine Learning Repository."

_ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

_AIRFOIL_COLUMNS = [
    "frequency",
    "angle_of_attack",
    "chord_length",
    "free_stream_velocity",
    "suction_side_displacement",
    "sound_pressure_level",
]


@dataclass(frozen=True)
class RemoteDataset:
    """One pinned download, with everything needed to parse and attribute it."""

    name: str
    url: str
    sha256: str
    target: str
    task_type: TaskType
    metric: str
    license: str
    attribution: str
    description: str
    read: dict[str, Any] = field(default_factory=dict)
    approx_rows: int = 0
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "sha256": self.sha256,
            "target": self.target,
            "task_type": self.task_type.value,
            "metric": self.metric,
            "license": self.license,
            "attribution": self.attribution,
            "description": self.description,
            "approx_rows": self.approx_rows,
            "tags": list(self.tags),
        }


REMOTE_DATASETS: dict[str, RemoteDataset] = {
    "open_titanic": RemoteDataset(
        name="open_titanic",
        url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        sha256="4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7",
        target="Survived",
        task_type=TaskType.CLASSIFICATION,
        metric="f1",
        license="Public domain (Titanic passenger manifest)",
        attribution="Compiled from the public Titanic passenger list; mirror datasciencedojo/datasets.",
        description=(
            "891-row passenger manifest: binary survival target with missing values and "
            "high-cardinality text columns (Name, Ticket, Cabin)"
        ),
        approx_rows=891,
        tags=("remote", "public", "mixed-dtype", "missing-values", "high-cardinality"),
    ),
    "open_adult": RemoteDataset(
        name="open_adult",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        sha256="5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
        target="income",
        task_type=TaskType.CLASSIFICATION,
        metric="f1",
        license=UCI_LICENSE,
        attribution=(
            f"{UCI_ATTRIBUTION} Donor: Ronny Kohavi and Barry Becker (1994 census extract)."
        ),
        description=(
            "32,561-row census extract: binary income target, nine categorical columns and "
            "'?' missing markers"
        ),
        read={
            "header": None,
            "names": _ADULT_COLUMNS,
            "skipinitialspace": True,
            "na_values": "?",
        },
        approx_rows=32_561,
        tags=("remote", "public", "mixed-dtype", "missing-values", "medium"),
    ),
    "open_wine_quality_red": RemoteDataset(
        name="open_wine_quality_red",
        url=(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "wine-quality/winequality-red.csv"
        ),
        sha256="4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e",
        target="quality",
        task_type=TaskType.REGRESSION,
        metric="rmse",
        license=UCI_LICENSE,
        attribution=(
            f"{UCI_ATTRIBUTION} Donor: Paulo Cortez (Cortez et al., 2009), red vinho verde."
        ),
        description="1,599-row red wine chemistry table with an ordinal quality score",
        read={"sep": ";"},
        approx_rows=1_599,
        tags=("remote", "public", "numeric", "small"),
    ),
    "open_airfoil": RemoteDataset(
        name="open_airfoil",
        url=(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/"
            "airfoil_self_noise.dat"
        ),
        sha256="74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3",
        target="sound_pressure_level",
        task_type=TaskType.REGRESSION,
        metric="rmse",
        license=UCI_LICENSE,
        attribution=f"{UCI_ATTRIBUTION} Donor: Thomas Brooks (NASA), 1989.",
        description="1,503-row NASA airfoil self-noise measurements, all numeric, tab-separated",
        read={"sep": "\t", "header": None, "names": _AIRFOIL_COLUMNS},
        approx_rows=1_503,
        tags=("remote", "public", "numeric", "small"),
    ),
}


def _validate_registry() -> None:
    """Fail at import on a malformed entry rather than at download time."""
    for key, spec in REMOTE_DATASETS.items():
        if key != spec.name:
            raise ConfigError(f"remote dataset key '{key}' does not match name '{spec.name}'")
        if not spec.url.startswith("https://"):
            raise ConfigError(f"remote dataset '{key}' must use https", url=spec.url)
        if len(spec.sha256) != 64 or any(c not in "0123456789abcdef" for c in spec.sha256):
            raise ConfigError(f"remote dataset '{key}' has no valid sha256 pin")
        for required in ("target", "license", "attribution", "description"):
            if not getattr(spec, required):
                raise ConfigError(f"remote dataset '{key}' is missing '{required}'")


_validate_registry()


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------


def default_cache_dir() -> Path:
    """Cache location used when a caller has no configuration to hand."""
    return Path("artifacts") / DEFAULT_CACHE_SUBDIR


def resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    return Path(cache_dir) if cache_dir is not None else default_cache_dir()


def cached_frame_path(name: str, cache_dir: str | Path | None = None) -> Path:
    return resolve_cache_dir(cache_dir) / f"{name}.csv"


def provenance_path(name: str, cache_dir: str | Path | None = None) -> Path:
    return resolve_cache_dir(cache_dir) / f"{name}.provenance.json"


def is_remote(name: str) -> bool:
    return name in REMOTE_DATASETS


def remote_names() -> list[str]:
    return sorted(REMOTE_DATASETS)


def is_cached(name: str, cache_dir: str | Path | None = None) -> bool:
    return cached_frame_path(name, cache_dir).is_file()


def provenance(name: str, cache_dir: str | Path | None = None) -> dict[str, Any] | None:
    path = provenance_path(name, cache_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - damaged cache entry
        return None


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------


def download(
    name: str,
    cache_dir: str | Path | None = None,
    *,
    timeout: float = DOWNLOAD_TIMEOUT_S,
) -> Path:
    """Fetch one pinned dataset, verify it, normalise it to CSV, and cache it.

    Raises rather than degrading: a checksum mismatch means the bytes are not the bytes
    that were reviewed, so there is nothing safe to fall back to.
    """
    spec = REMOTE_DATASETS.get(name)
    if spec is None:
        raise ConfigError(f"'{name}' is not a curated remote dataset", known=remote_names())

    destination = cached_frame_path(name, cache_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "downloading curated dataset",
        extra={"context": {"name": name, "url": spec.url}},
    )
    request = urllib.request.Request(spec.url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except Exception as exc:
        raise DatasetValidationError(
            f"could not download '{name}': {type(exc).__name__}: {exc}",
            url=spec.url,
        ) from exc

    digest = hashlib.sha256(payload).hexdigest()
    if digest != spec.sha256:
        raise SecurityError(
            f"downloaded '{name}' does not match its pinned checksum; refusing to parse it",
            name=name,
            url=spec.url,
            expected=spec.sha256,
            actual=digest,
        )

    try:
        frame = pd.read_csv(io.BytesIO(payload), **spec.read)
    except Exception as exc:
        raise DatasetValidationError(
            f"'{name}' downloaded correctly but could not be parsed: {type(exc).__name__}: {exc}",
            url=spec.url,
        ) from exc

    if frame.empty:
        raise DatasetValidationError(f"'{name}' downloaded as an empty table", url=spec.url)
    if spec.target not in frame.columns:
        raise DatasetValidationError(
            f"'{name}' is missing its expected target column '{spec.target}'",
            columns=[str(column) for column in frame.columns],
        )

    frame.to_csv(destination, index=False)
    provenance_path(name, cache_dir).write_text(
        json.dumps(
            {
                **spec.to_dict(),
                "downloaded_at": utc_now_iso(),
                "n_rows": int(frame.shape[0]),
                "n_cols": int(frame.shape[1]),
                "columns": [str(column) for column in frame.columns],
                "bytes": len(payload),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "curated dataset cached",
        extra={"context": {"name": name, "rows": int(frame.shape[0]), "path": str(destination)}},
    )
    return destination


def load_frame(
    name: str,
    cache_dir: str | Path | None = None,
    *,
    allow_network: bool = True,
) -> pd.DataFrame:
    """Load a curated dataset, downloading it only when it is not already cached."""
    path = cached_frame_path(name, cache_dir)
    if not path.is_file():
        if not allow_network:
            raise ConfigError(
                f"'{name}' is a curated remote dataset and is not cached; "
                "download it once with network access, or pass allow_network=True",
                name=name,
                cache_dir=str(resolve_cache_dir(cache_dir)),
            )
        path = download(name, cache_dir)
    return pd.read_csv(path)


def remote_source(
    name: str, cache_dir: str | Path | None = None
) -> Any:  # -> DatasetSource, imported lazily to avoid a cycle
    """Wrap a curated dataset as a :class:`DatasetSource` for the loader registry."""
    from rl_automl.dataset.loaders import DatasetSource

    spec = REMOTE_DATASETS.get(name)
    if spec is None:
        raise ConfigError(f"'{name}' is not a curated remote dataset", known=remote_names())

    def loader() -> pd.DataFrame:
        return load_frame(name, cache_dir, allow_network=True)

    return DatasetSource(
        name=spec.name,
        task_type=spec.task_type,
        loader=loader,
        target=spec.target,
        metric=spec.metric,
        description=f"{spec.description}. Licence: {spec.license}",
        requires_network=not is_cached(name, cache_dir),
        tags=(*spec.tags, "remote"),
    )


def remote_descriptors(cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    return [
        {**spec.to_dict(), "cached": is_cached(spec.name, cache_dir)}
        for spec in (REMOTE_DATASETS[name] for name in remote_names())
    ]


__all__ = [
    "DEFAULT_CACHE_SUBDIR",
    "REMOTE_DATASETS",
    "RemoteDataset",
    "cached_frame_path",
    "default_cache_dir",
    "download",
    "is_cached",
    "is_remote",
    "load_frame",
    "provenance",
    "remote_descriptors",
    "remote_names",
    "remote_source",
    "resolve_cache_dir",
]
