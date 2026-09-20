"""Replace non-ASCII escape sequences in source strings (temporary scratch script).

The previous pass only caught literal non-ASCII characters. Escapes such as ``\\u2248``
look like plain ASCII in the file but produce a non-ASCII character at runtime, which then
breaks printing on a cp1252 Windows console.
"""

from __future__ import annotations

import pathlib
import re

MAPPING = {
    "\\u2248": "~",       # almost equal
    "\\u2014": "-",       # em dash
    "\\u2013": "-",       # en dash
    "\\u2019": "'",       # right single quote
    "\\u201c": '"',
    "\\u201d": '"',
    "\\u2192": "->",      # right arrow
    "\\u00d7": "x",       # multiplication sign
    "\\u2265": ">=",
    "\\u2264": "<=",
    "\\u00b2": "2",
    "\\u2026": "...",
    "\\u00b1": "+/-",
    "\\u00b7": "-",       # middle dot
}

ESCAPE = re.compile(r"\\u[0-9a-fA-F]{4}")

changed = 0
unmapped: dict[str, set[str]] = {}

for path in sorted(pathlib.Path("rl_automl").rglob("*.py")):
    text = path.read_text(encoding="utf-8")
    if "\\u" not in text:
        continue

    original = text
    for escape, replacement in MAPPING.items():
        text = text.replace(escape, replacement)

    for match in ESCAPE.finditer(text):
        unmapped.setdefault(match.group(0), set()).add(str(path))

    if text != original:
        path.write_text(text, encoding="utf-8")
        changed += 1
        print("updated", path)

print(f"changed files: {changed}")
for escape, files in unmapped.items():
    print("UNMAPPED escape", escape, "in", sorted(files))

# Confirm the explanation string is now console-safe.
from rl_automl.search.recommendation import RecommendationConfig  # noqa: E402

sample = "expected f1 ~ 0.975 using the 'balanced' preset; medium cost (~120s of training)"
sample.encode("cp1252")
print("cp1252 encode check: OK")
print("imports:", RecommendationConfig.__name__)
