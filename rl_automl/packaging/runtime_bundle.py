"""Runtime sources shipped inside a model package.

A fitted pipeline and model are pickled, and pickle records classes by dotted path. Any
class defined in this project therefore has to travel with the artifact, or the package
only loads on a machine where AutoML happens to be installed -- which is precisely the
machine the package exists to avoid needing.

So the package carries its own copies of the modules that define those classes, under
``runtime/``, and the generated ``model_loader`` registers them under their original
dotted names before unpickling.

Every file listed here must import **nothing** from this project except other files listed
here. That is what keeps the artifact self-contained, and it is asserted by the tests.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from rl_automl.core.errors import PackagingError
from rl_automl.core.logging import get_logger

logger = get_logger("packaging.runtime_bundle")

RUNTIME_DIRNAME = "runtime"

#: dotted module name the pickle refers to -> path relative to the ``rl_automl`` package
MODULE_SOURCES: dict[str, str] = {
    "rl_automl.execution.runtime_transformers": "execution/runtime_transformers.py",
    "rl_automl.execution.torch_models": "execution/torch_models.py",
    "rl_automl.core.errors": "core/errors.py",
    "rl_automl.core.logging": "core/logging.py",
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def runtime_module_map() -> dict[str, str]:
    """Mapping the generated loader uses: dotted name -> path inside the package."""
    return {dotted: f"{RUNTIME_DIRNAME}/{relative}" for dotted, relative in MODULE_SOURCES.items()}


def write_runtime_bundle(destination: str | Path) -> list[Path]:
    """Copy the runtime sources into ``destination/runtime/``. Returns the files written."""
    root = Path(destination)
    written: list[Path] = []
    for relative in MODULE_SOURCES.values():
        source = _PROJECT_ROOT / relative
        if not source.is_file():  # pragma: no cover - only if the layout changes
            raise PackagingError(f"runtime source is missing from the installation: {source}")
        target = root / RUNTIME_DIRNAME / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        written.append(target)
    logger.info(
        "runtime bundle written",
        extra={"context": {"files": len(written), "destination": str(root)}},
    )
    return written


__all__ = ["MODULE_SOURCES", "RUNTIME_DIRNAME", "runtime_module_map", "write_runtime_bundle"]
