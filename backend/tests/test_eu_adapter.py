"""EU Open Data Portal adapter: search ladder, format handling, preview fallback."""

import httpx
import pytest
import respx

from app.discovery.adapters.eu import EUOpenDataAdapter
from app.discovery.adapters.base import AdapterError

EU = "https://data.europa.eu/api/hub/search"


def _search_payload(n=2):
    return {
        "result": {
            "count": n,
            "results": [
                {
                    "id": f"ds-{i}",
                    "title": {"en": f"Rainfall Dataset {i}"},
                    "description": {"en": "Rainfall measurements"},
                    "keywords": {"en": ["rainfall", "climate"]},
                    "publisher": {"label": {"en": "Met Agency"}},
                    "distributions": [
                        {"format": {"label": {"en": "CSV"}}, "download_url": ["https://example.com/data.csv"]},
                        {"format": "HTML", "access_url": ["https://example.com/landing"]},
                    ],
                }
                for i in range(n)
            ],
        }
    }


@pytest.mark.asyncio
@respx.mock
async def test_search_degrades_through_ladder_and_dedupes():
    from app.schemas import ParsedRequirements

    respx.get(f"{EU}/search").mock(return_value=httpx.Response(200, json=_search_payload(2)))
    results = await EUOpenDataAdapter().search(
        ParsedRequirements(keywords=["rainfall"], domain=["climate"], location=["chennai"], parser="deterministic")
    )
    assert len(results) == 2
    r = results[0]
    assert r.source_dataset_id == "ds-0"
    assert r.file_format == "csv"  # CSV distribution recognized, HTML ignored
    assert "rainfall" in r.tags


@pytest.mark.asyncio
@respx.mock
async def test_format_as_plain_string_is_handled():
    payload = {"result": {"results": [{
        "id": "ds-x", "title": {"en": "Weather"},
        "distributions": [{"format": "CSV", "download_url": "https://example.com/d.csv"}],
    }]}}
    respx.get(f"{EU}/search").mock(return_value=httpx.Response(200, json=payload))
    from app.schemas import ParsedRequirements

    results = await EUOpenDataAdapter().search(ParsedRequirements(keywords=["weather"], parser="deterministic"))
    assert results[0].file_format == "csv"


@pytest.mark.asyncio
@respx.mock
async def test_preview_downloads_csv_distribution():
    csv_payload = b"station,rain\na,1.5\nb,2.0\n"
    respx.get(f"{EU}/datasets/ds-x").mock(return_value=httpx.Response(200, json={
        "result": {"title": {"en": "Weather"}, "distributions": [
            {"format": {"label": {"en": "CSV"}}, "download_url": ["https://example.com/d.csv"]},
        ]}
    }))
    respx.get("https://example.com/d.csv").mock(return_value=httpx.Response(200, content=csv_payload))
    columns, rows = await EUOpenDataAdapter().get_preview("ds-x", limit=5)
    assert columns == ["station", "rain"]
    assert rows == [{"station": "a", "rain": 1.5}, {"station": "b", "rain": 2.0}]


@pytest.mark.asyncio
@respx.mock
async def test_preview_honest_error_when_only_landing_pages():
    respx.get(f"{EU}/datasets/ds-x").mock(return_value=httpx.Response(200, json={
        "result": {"title": {"en": "Weather"}, "distributions": [
            {"format": "HTML", "access_url": ["https://example.com/landing"]},
        ]}
    }))
    respx.get("https://example.com/landing").mock(return_value=httpx.Response(200, text="<html>page</html>"))
    from app.schemas import ParsedRequirements

    with pytest.raises(AdapterError) as exc_info:
        await EUOpenDataAdapter().get_preview("ds-x")
    assert exc_info.value.code == "PREVIEW_UNAVAILABLE"
