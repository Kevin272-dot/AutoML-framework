import httpx
import pytest
import respx

from app.discovery.adapters.huggingface import HuggingFaceAdapter
from app.schemas import ParsedRequirements

HF_API = "https://huggingface.co/api"
DSS = "https://datasets-server.huggingface.co"


@pytest.mark.asyncio
@respx.mock
async def test_search_parses_results_and_dedupes():
    respx.get(f"{HF_API}/datasets").mock(
        return_value=httpx.Response(200, json=[
            {"id": "user/crop-india", "tags": ["license:cc-by-4.0", "size_categories:medium"]},
            {"id": "user/crop-india", "tags": []},  # duplicate id
            {"id": "org/weather", "tags": []},
        ])
    )
    adapter = HuggingFaceAdapter()
    results = await adapter.search(ParsedRequirements(keywords=["crop"], domain=["agriculture"]))
    assert len(results) == 2
    crop = next(r for r in results if r.source_dataset_id == "user/crop-india")
    assert crop.license == "cc-by-4.0"
    assert crop.url == "https://huggingface.co/datasets/user/crop-india"
    assert crop.download_available and crop.preview_available


@pytest.mark.asyncio
@respx.mock
async def test_metadata_extracts_columns_and_rows():
    respx.get(f"{HF_API}/datasets/user/crop-india").mock(
        return_value=httpx.Response(200, json={"id": "user/crop-india", "tags": ["license:mit"]})
    )
    respx.get(f"{DSS}/parquet").mock(return_value=httpx.Response(200, json={"parquet_files": [{"url": "x"}]}))
    respx.get(f"{DSS}/statistics").mock(return_value=httpx.Response(200, json={
        "statistics": [{
            "num_rows": 1234,
            "column_statistics": [
                {"column_name": "district", "column_type": "string_label",
                 "column_statistics": {"frequencies": {"north": 10, "south": 8}}},
                {"column_name": "yield", "column_type": "float",
                 "column_statistics": {"frequencies": {}}},
            ],
        }]
    }))
    meta = await HuggingFaceAdapter().get_metadata("user/crop-india")
    assert meta.row_count == 1234
    assert meta.column_count == 2
    assert meta.file_format == "parquet"
    assert meta.license == "mit"
    assert [c.name for c in meta.columns] == ["district", "yield"]


@pytest.mark.asyncio
@respx.mock
async def test_preview_returns_rows():
    respx.get(f"{DSS}/rows").mock(return_value=httpx.Response(200, json={
        "features": [{"column": {"name": "district"}}, {"column": {"name": "yield"}}],
        "rows": [{"row": {"district": "north", "yield": 4.2}}, {"row": {"district": "south", "yield": 3.1}}],
    }))
    columns, rows = await HuggingFaceAdapter().get_preview("user/crop-india", limit=10)
    assert columns == ["district", "yield"]
    assert rows[1]["yield"] == 3.1


@pytest.mark.asyncio
@respx.mock
async def test_preview_404_raises_structured_error():
    respx.get(f"{DSS}/rows").mock(return_value=httpx.Response(404, json={}))
    respx.get(f"{DSS}/parquet").mock(return_value=httpx.Response(200, json={"parquet_files": []}))
    respx.get(f"{HF_API}/datasets/user/missing/tree/main").mock(return_value=httpx.Response(200, json=[]))
    from app.discovery.adapters.base import AdapterError

    with pytest.raises(AdapterError) as exc_info:
        await HuggingFaceAdapter().get_preview("user/missing")
    assert exc_info.value.code == "PREVIEW_UNAVAILABLE"


@pytest.mark.asyncio
@respx.mock
async def test_download_streams_and_hashes(tmp_path):
    payload = b"col1,col2\n1,2\n3,4\n"
    respx.get(f"{DSS}/parquet").mock(return_value=httpx.Response(200, json={
        "parquet_files": [{"url": "https://huggingface.co/datasets/user/crop-india/resolve/main/data.parquet",
                           "filename": "data.parquet", "size": len(payload)}]
    }))
    respx.get(f"{HF_API}/datasets/user/crop-india/tree/main").mock(return_value=httpx.Response(200, json=[]))
    respx.get("https://huggingface.co/datasets/user/crop-india/resolve/main/data.parquet").mock(
        return_value=httpx.Response(200, content=payload)
    )
    dest = tmp_path / "out.parquet"
    result = await HuggingFaceAdapter().download("user/crop-india", str(dest), max_bytes=1024 * 1024)
    assert result["size"] == len(payload)
    assert result["format"] == "parquet"
    assert len(result["hash"]) == 64
    assert dest.read_bytes() == payload


@pytest.mark.asyncio
@respx.mock
async def test_download_enforces_size_limit(tmp_path):
    respx.get(f"{DSS}/parquet").mock(return_value=httpx.Response(200, json={
        "parquet_files": [{"url": "https://example.com/big.parquet", "filename": "big.parquet"}]
    }))
    respx.get(f"{HF_API}/datasets/user/big/tree/main").mock(return_value=httpx.Response(200, json=[]))
    respx.get("https://example.com/big.parquet").mock(
        return_value=httpx.Response(200, content=b"x" * 10000)
    )
    from app.discovery.adapters.base import AdapterError

    with pytest.raises(AdapterError) as exc_info:
        await HuggingFaceAdapter().download("user/big", str(tmp_path / "b"), max_bytes=100)
    assert exc_info.value.code == "DATASET_TOO_LARGE"


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_hf_search_smoke():
    """Live smoke test: run with `pytest -m live`. Requires internet access."""
    results = await HuggingFaceAdapter().search(ParsedRequirements(keywords=["crop yield"], domain=["agriculture"]))
    assert isinstance(results, list)
