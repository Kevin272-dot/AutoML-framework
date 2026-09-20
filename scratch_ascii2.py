"""List remaining non-ASCII characters with counts (temporary scratch script)."""

from __future__ import annotations

import collections
import pathlib

counts: collections.Counter[str] = collections.Counter()
where: dict[str, set[str]] = {}

for path in sorted(pathlib.Path("rl_automl").rglob("*.py")):
    for ch in path.read_text(encoding="utf-8"):
        if ord(ch) > 127:
            counts[ch] += 1
            where.setdefault(ch, set()).add(str(path))

for ch, count in counts.most_common():
    try:
        ch.encode("cp1252")
        portable = "cp1252-safe"
    except UnicodeEncodeError:
        portable = "NOT cp1252-safe"
    print(f"{ch!r} U+{ord(ch):04X} count={count:4d} {portable} files={len(where[ch])}")
