"""Find non-ASCII characters in package source (temporary scratch script)."""

from __future__ import annotations

import pathlib

REPLACEMENTS = {
    "\u2248": "~",       # almost equal
    "\u2014": "-",       # em dash
    "\u2013": "-",       # en dash
    "\u2019": "'",       # right single quote
    "\u201c": '"',
    "\u201d": '"',
    "\u2192": "->",      # right arrow
    "\u00d7": "x",       # multiplication sign
    "\u2265": ">=",
    "\u2264": "<=",
    "\u00b2": "2",       # superscript two
    "\u2026": "...",
    "\u00b1": "+/-",
    "\u03b1": "alpha",
    "\u03bb": "lambda",
    "\u03b3": "gamma",
    "\u03bc": "mu",
    "\u007f": "",
}

total = 0
changed_files = 0
for path in sorted(pathlib.Path("rl_automl").rglob("*.py")):
    text = path.read_text(encoding="utf-8")
    original = text
    for source, replacement in REPLACEMENTS.items():
        if source in text:
            text = text.replace(source, replacement)
    remaining = {ch for ch in text if ord(ch) > 127}
    if remaining:
        print(f"UNMAPPED in {path}: {sorted(remaining)}")
    if text != original:
        path.write_text(text, encoding="utf-8")
        changed_files += 1
        count = sum(original.count(source) for source in REPLACEMENTS)
        total += count
        print(f"fixed {count:3d} character(s) in {path}")

print(f"changed files: {changed_files}, replacements: {total}")
