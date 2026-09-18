"""Conservative dataset deduplication.

A candidate is dropped only when it is an exact duplicate by:
  - (source, source_dataset_id), or
  - canonical URL (normalized), or
  - exact normalized name within the same owner.

Similar names alone NEVER merge unrelated datasets.
"""


def dedupe_key_for(source_slug: str, source_dataset_id: str) -> str:
    return f"{source_slug}:{source_dataset_id}"


def dedupe_candidates(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (unique, dropped). Input candidates carry source slug in 'source_slug'."""
    seen: set[str] = set()
    unique: list[dict] = []
    dropped: list[dict] = []

    for cand in candidates:
        keys = {
            f"sid:{cand.get('source_slug', '')}:{cand.get('source_dataset_id', '')}",
            f"url:{cand.get('canonical_url', '').rstrip('/').lower()}",
        }

        if keys & seen:
            dropped.append(cand)
            continue
        seen |= keys
        unique.append(cand)
    return unique, dropped
