"""Adaptive RL-driven AutoML.

Layering (enforced by ``rl_automl.tests.test_import_boundaries``)::

    core -> task/dataset -> search/registry -> environment -> agent
    execution -> packaging -> orchestrator -> api/cli

The RL agent decides *what to try*. The executor trains. The evaluator measures.
The packaging engine deploys. No layer may absorb another's responsibility.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
