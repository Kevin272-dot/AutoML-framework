"""Move the class-weighted preset block below all base preset tables (temporary script)."""

from __future__ import annotations

import pathlib

PATH = pathlib.Path("rl_automl/search/model_registry.py")
text = PATH.read_text(encoding="utf-8")

START = "#: Class-weighted variants."
AFTER_BLOCK = "_LINEAR_PRESETS = ("
REGISTRATION = "# Registration"

start = text.index(START)
end = text.index(AFTER_BLOCK)
block = text[start:end].rstrip() + "\n\n\n"
text = text[:start] + text[end:]

registration_index = text.index(REGISTRATION)
# Back up to the start of the separator line above "# Registration".
line_start = text.rindex("\n# ------", 0, registration_index)
text = text[:line_start] + "\n\n" + block + text[line_start:]

PATH.write_text(text, encoding="utf-8")
print("moved class-weighted preset block")
