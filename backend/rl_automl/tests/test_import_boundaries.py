from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize(
    "path",
    [*sorted((ROOT / "agent").glob("*.py")), ROOT / "environment" / "simulator.py"],
)
def test_agent_and_simulator_do_not_import_execution(path):
    forbidden = [
        name
        for name in _imports(path)
        if name == "rl_automl.execution" or name.startswith("rl_automl.execution.")
    ]
    assert not forbidden, f"{path.relative_to(ROOT)} imports execution: {forbidden}"
