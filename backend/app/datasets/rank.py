"""Dataset relevance ranking.

score = 0.30*keyword_match + 0.20*date_match + 0.20*domain_match
      + 0.10*completeness + 0.10*size_score + 0.10*feature_quality

Every component is normalized to 0-1 and returned with the total so the UI can
explain the score. No hidden heuristics.
"""

import re

from app.parsing.locations import location_variants as _location_variants
from app.schemas import ParsedRequirements

WEIGHTS = {
    "keyword_match": 0.30,
    "date_match": 0.20,
    "domain_match": 0.20,
    "completeness": 0.10,
    "size_score": 0.10,
    "feature_quality": 0.10,
}

_ID_LIKE = re.compile(r"(^|_)(id|uuid|guid|identifier|key|code|slug)(_|$)", re.I)


def _keyword_match(cand: dict, req: ParsedRequirements) -> float:
    # Locations count as matchable terms too — "chennai" in a title/description is signal.
    terms = list(req.keywords) + [loc.lower() for loc in _location_variants(req)]
    if not terms:
        return 0.5  # nothing to match on; neutral
    haystack = " ".join(
        filter(None, [cand.get("name", ""), cand.get("description") or "", " ".join(cand.get("tags", []))])
    ).lower()
    hits = sum(1 for term in terms if term.lower() in haystack)
    return min(1.0, hits / len(terms))


def _date_match(cand: dict, req: ParsedRequirements) -> float:
    if not req.date_start and not req.date_end:
        return 0.5
    ds, de = cand.get("date_start"), cand.get("date_end")
    if ds is None and de is None:
        # Unknown temporal coverage is common on catalog sources; score it low-neutral
        # rather than zero so datasets with strong keyword/domain matches still surface.
        return 0.3
    cand_start = int(ds) if ds and str(ds).isdigit() else None
    cand_end = int(de) if de and str(de).isdigit() else None
    if cand_start and not cand_end:
        cand_end = cand_start
    req_start = int(req.date_start) if req.date_start and req.date_start.isdigit() else None
    req_end = int(req.date_end) if req.date_end and req.date_end.isdigit() else None
    if cand_start is None:
        return 0.3
    overlap_start = max(cand_start, req_start) if req_start else cand_start
    overlap_end = min(cand_end, req_end) if req_end else cand_end
    if overlap_end < overlap_start:
        return 0.0
    req_span = ((req_end or cand_end) - (req_start or cand_start)) or 1
    overlap = overlap_end - overlap_start
    return max(0.0, min(1.0, overlap / req_span))


def _domain_match(cand: dict, req: ParsedRequirements) -> float:
    if not req.domain:
        return 0.5
    cand_domains = set(cand.get("domains", [])) | {t.lower() for t in cand.get("tags", [])}
    hits = sum(1 for d in req.domain if d.lower() in cand_domains)
    if hits == 0:
        # Fall back to text mentions in tags/description.
        haystack = " ".join(cand.get("tags", []) + [cand.get("description") or ""]).lower()
        hits = sum(1 for d in req.domain if d.lower() in haystack)
    return hits / len(req.domain)


def _completeness(cand: dict) -> float:
    """How much of the normalized metadata the source actually provided."""
    fields = [
        cand.get("description"),
        cand.get("license"),
        cand.get("row_count"),
        cand.get("column_count"),
        cand.get("file_format"),
        cand.get("file_size_bytes"),
        cand.get("date_start"),
        cand.get("tags") or None,
    ]
    present = sum(1 for f in fields if f is not None)
    return present / len(fields)


def _size_score(cand: dict, req: ParsedRequirements) -> float:
    rows = cand.get("row_count")
    if rows is None:
        return 0.5
    if rows <= 0:
        return 0.0
    if req.max_rows and rows > req.max_rows:
        return 0.2
    if req.min_rows and rows < req.min_rows:
        return 0.2
    # Comfortable working range: 500 .. 5,000,000 rows on a log scale.
    import math

    lo, hi = math.log10(500), math.log10(5_000_000)
    v = (math.log10(rows) - lo) / (hi - lo)
    return max(0.0, min(1.0, v))


def _feature_quality(cand: dict) -> float:
    cols = cand.get("columns") or []
    if not cols:
        return 0.5
    n = len(cols)
    if n == 0:
        return 0.0
    id_like = sum(1 for c in cols if _ID_LIKE.search(c.get("name", "")))
    typed = sum(1 for c in cols if c.get("semantic_type") not in (None, "unknown"))
    penalty = id_like / n
    reward = typed / n
    score = 0.4 + 0.4 * reward - 0.3 * penalty
    # penalize extremely narrow tables
    if n < 4:
        score -= 0.2
    return max(0.0, min(1.0, score))


def score_candidate(cand: dict, req: ParsedRequirements) -> dict:
    components = {
        "keyword_match": round(_keyword_match(cand, req), 4),
        "date_match": round(_date_match(cand, req), 4),
        "domain_match": round(_domain_match(cand, req), 4),
        "completeness": round(_completeness(cand), 4),
        "size_score": round(_size_score(cand, req), 4),
        "feature_quality": round(_feature_quality(cand), 4),
    }
    total = sum(WEIGHTS[k] * v for k, v in components.items())
    return {"overall_score": round(total, 4), "score_components": components}


def score_candidates(candidates: list[dict], req: ParsedRequirements) -> list[dict]:
    """Score and sort in place; returns candidates sorted by overall_score desc."""
    for cand in candidates:
        result = score_candidate(cand, req)
        cand["score_total"] = result["overall_score"]
        cand["score_components"] = result["score_components"]
    candidates.sort(key=lambda c: c.get("score_total", 0.0), reverse=True)
    return candidates
