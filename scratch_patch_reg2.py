"""Reorder the class-weighted preset block after the base tables (temporary script)."""

from __future__ import annotations

import pathlib

PATH = pathlib.Path("rl_automl/search/model_registry.py")
text = PATH.read_text(encoding="utf-8")

START = "#: Class-weighted variants."
END = "_XGB_PRESETS = ("
REGISTRATION = "# Registration"

start = text.index(START)
end = text.index(END)
if start > end:
    raise SystemExit("the class-weighted block is already after its dependencies")

block = text[start:end].rstrip() + "\n\n\n"
text = text[:start] + text[end:]

registration_index = text.index(REGISTRATION)
line_start = text.rindex("\n# ------", 0, registration_index)
text = text[:line_start] + "\n\n" + block + text[line_start:]

PATH.write_text(text, encoding="utf-8")

# Confirm the ordering invariant: every CLF table must follow its base table.
order = [
    "_RF_PRESETS = (",
    "_RF_PRESETS_CLF",
    "_XGB_PRESETS = (",
    "_XGB_PRESETS_CLF",
    "_LGBM_PRESETS = (",
    "_LGBM_PRESETS_CLF",
]
positions = [(marker, text.index(marker)) for marker in order]
for (left, left_at), (right, right_at) in zip(positions, positions[1:], strict=False):
    if left_at > right_at:
        raise SystemExit(f"ordering violated: {left} appears after {right}")
print("ordering verified:", [f"{m}={p}" for m, p in positions])
print("model_registry.py reordered")
