"""Determinism helpers.

Reproducibility is a requirement, not a nicety: two runs with the same seed and the
same approved pipelines must produce the same metrics. Anything that consumes
randomness goes through :func:`make_rng` or `seed_everything`.
"""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np

_DEFAULT_SEED = 1234


def seed_everything(seed: int | None = None, deterministic_torch: bool = True) -> int:
    """Seed every RNG this package touches and return the seed actually used."""
    if seed is None:
        seed = _DEFAULT_SEED

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is an optional extra
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Best effort: some ops have no deterministic kernel and will warn.
            with contextlib.suppress(Exception):  # pragma: no cover - torch version dependent
                torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover - torch not installed
        pass

    return seed


def make_rng(seed: int | None = None) -> np.random.Generator:
    """A fresh, independent NumPy generator. Preferred over global ``np.random``."""
    return np.random.default_rng(_DEFAULT_SEED if seed is None else seed)


def derive_seed(base_seed: int, *labels: str | int) -> int:
    """Derive a stable child seed from a base seed and a path of labels.

    Used so that each pipeline gets its own reproducible seed without the caller
    having to hand-manage a counter.
    """
    digest = abs(hash((base_seed, *labels))) % (2**31 - 1)
    return int(digest) if digest else base_seed


def torch_device(prefer_cuda: bool = True) -> str:
    """Return the best available device string. CPU is always a valid answer."""
    try:
        import torch

        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover - torch not installed
        pass
    return "cpu"


__all__ = ["derive_seed", "make_rng", "seed_everything", "torch_device"]
