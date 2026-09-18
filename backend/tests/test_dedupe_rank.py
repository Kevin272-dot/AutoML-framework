from app.datasets.dedupe import dedupe_candidates
from app.datasets.rank import score_candidates
from app.schemas import ParsedRequirements


def _cand(**overrides):
    base = {
        "source_slug": "huggingface",
        "source_dataset_id": "user/ds-a",
        "canonical_url": "https://huggingface.co/datasets/user/ds-a",
        "name": "Crop Yield India",
        "name_normalized": "crop yield india",
        "description": "District-wise crop yield statistics for India",
        "license": "cc-by-4.0",
        "owner": "user",
        "tags": ["agriculture"],
        "domains": ["agriculture"],
        "row_count": 120_000,
        "column_count": 18,
        "file_format": "csv",
        "file_size_bytes": 50_000_000,
        "columns": [{"name": f"col{i}", "semantic_type": "numeric"} for i in range(10)],
        "raw_metadata": {"anything": "kept"},
    }
    base.update(overrides)
    return base


def test_exact_url_duplicates_dropped():
    unique, dropped = dedupe_candidates([_cand(), _cand()])
    assert len(unique) == 1
    assert len(dropped) == 1


def test_same_id_different_source_kept():
    a = _cand(source_slug="huggingface")
    b = _cand(source_slug="kaggle", canonical_url="https://www.kaggle.com/datasets/x/ds-a",
              source_dataset_id="x/ds-a")
    unique, dropped = dedupe_candidates([a, b])
    assert len(unique) == 2
    assert dropped == []


def test_similar_names_not_merged():
    a = _cand(owner="alice", name="Weather Data", canonical_url="https://example.com/1",
              source_dataset_id="a/1")
    b = _cand(owner="bob", name="Weather Data 2", canonical_url="https://example.com/2",
              source_dataset_id="b/2")
    unique, _ = dedupe_candidates([a, b])
    assert len(unique) == 2


def test_ranking_components_sum_and_order():
    req = ParsedRequirements(keywords=["crop", "yield"], domain=["agriculture"],
                             date_start="2020", date_end="2025")
    good = _cand(date_start="2019", date_end="2025")
    poor = _cand(source_dataset_id="u/ds-b", canonical_url="https://example.com/b",
                 name="Stock Prices", name_normalized="stock prices", tags=["finance"],
                 domains=["finance"], row_count=None, description=None, license=None,
                 file_format=None, file_size_bytes=None, columns=[])
    scored = score_candidates([poor, good], req)
    assert scored[0] is good
    for cand in scored:
        comps = cand["score_components"]
        assert set(comps) == {"keyword_match", "date_match", "domain_match",
                              "completeness", "size_score", "feature_quality"}
        assert all(0.0 <= v <= 1.0 for v in comps.values())
        assert 0.0 <= cand["score_total"] <= 1.0


def test_raw_metadata_preserved_through_pipeline():
    req = ParsedRequirements(keywords=["crop"], domain=["agriculture"])
    scored = score_candidates([_cand()], req)
    assert scored[0]["raw_metadata"] == {"anything": "kept"}
