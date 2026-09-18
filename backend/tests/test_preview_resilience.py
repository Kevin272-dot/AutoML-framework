"""Preview robustness: retry on transient upstream failures + parquet fallback."""

import io

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from app.discovery.adapters.huggingface import HuggingFaceAdapter
from app.discovery.adapters.base import AdapterError

DSS = "https://datasets-server.huggingface.co"
HF_API = "https://huggingface.co/api"


@pytest.mark.asyncio
@respx.mock
async def test_preview_retries_transient_504_then_succeeds():
    route = respx.get(f"{DSS}/rows").mock(
        side_effect=[
            httpx.Response(504, json={"error": "Gateway Timeout"}),
            httpx.Response(200, json={
                "features": [{"column": {"name": "a"}}],
                "rows": [{"row": {"a": 1}}],
            }),
        ]
    )
    columns, rows = await HuggingFaceAdapter().get_preview("user/ds")
    assert route.call_count == 2
    assert columns == ["a"] and rows == [{"a": 1}]


@pytest.mark.asyncio
@respx.mock
async def test_preview_falls_back_to_parquet_when_rows_is_down():
    df = pd.DataFrame({"district": ["north", "south"], "yield": [4.2, 3.1]})
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    payload = buf.getvalue()

    respx.get(f"{DSS}/rows").mock(return_value=httpx.Response(504, json={"error": "Gateway Timeout"}))
    respx.get(f"{DSS}/parquet").mock(return_value=httpx.Response(200, json={
        "parquet_files": [{"url": "https://example.com/data.parquet", "filename": "data.parquet"}]
    }))
    respx.get("https://example.com/data.parquet").mock(return_value=httpx.Response(200, content=payload))

    columns, rows = await HuggingFaceAdapter().get_preview("user/ds", limit=10)
    assert columns == ["district", "yield"]
    assert rows == [{"district": "north", "yield": 4.2}, {"district": "south", "yield": 3.1}]


@pytest.mark.asyncio
@respx.mock
async def test_preview_fallback_surfaces_honest_error_when_no_files():
    respx.get(f"{DSS}/rows").mock(return_value=httpx.Response(504, json={"error": "Gateway Timeout"}))
    respx.get(f"{DSS}/parquet").mock(return_value=httpx.Response(200, json={"parquet_files": []}))
    respx.get(f"{HF_API}/datasets/user/empty/tree/main").mock(return_value=httpx.Response(200, json=[]))

    with pytest.raises(AdapterError) as exc_info:
        await HuggingFaceAdapter().get_preview("user/empty")
    assert exc_info.value.code == "PREVIEW_UNAVAILABLE"


@pytest.mark.asyncio
@respx.mock
async def test_rows_error_payload_falls_back_to_parquet():
    df = pd.DataFrame({"x": [1]})
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)

    respx.get(f"{DSS}/rows").mock(return_value=httpx.Response(200, json={"error": "Some upstream failure"}))
    respx.get(f"{DSS}/parquet").mock(return_value=httpx.Response(200, json={
        "parquet_files": [{"url": "https://example.com/d.parquet", "filename": "d.parquet"}]
    }))
    respx.get("https://example.com/d.parquet").mock(return_value=httpx.Response(200, content=buf.getvalue()))
    columns, rows = await HuggingFaceAdapter().get_preview("user/ds")
    assert columns == ["x"] and rows == [{"x": 1}]


@pytest.mark.asyncio
@respx.mock
async def test_preview_handles_array_and_nested_cells():
    """Cells containing numpy arrays / lists (e.g. climate_fever evidence) must not crash."""
    df = pd.DataFrame({
        "claim": ["c1"],
        "evidence": [np.array([{"text": "a"}, {"text": "b"}])],
        "labels": [np.array([1, 0])],
    })
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)

    respx.get(f"{DSS}/rows").mock(return_value=httpx.Response(500, json={"error": "boom"}))
    respx.get(f"{DSS}/parquet").mock(return_value=httpx.Response(200, json={
        "parquet_files": [{"url": "https://example.com/arr.parquet", "filename": "arr.parquet"}]
    }))
    respx.get("https://example.com/arr.parquet").mock(return_value=httpx.Response(200, content=buf.getvalue()))

    columns, rows = await HuggingFaceAdapter().get_preview("user/ds")
    assert columns == ["claim", "evidence", "labels"]
    assert rows[0]["claim"] == "c1"
    assert rows[0]["labels"] == [1, 0]
    assert rows[0]["evidence"] == [{"text": "a"}, {"text": "b"}]


@pytest.mark.asyncio
@respx.mock
async def test_preview_falls_back_to_original_repo_csv_when_conversion_failed():
    """Datasets-server conversion failed entirely -> read the original repo CSV."""
    csv_payload = b"state,crop,yield\nPuducherry,Rice,2.56\n"
    respx.get(f"{DSS}/rows").mock(return_value=httpx.Response(504, json={"error": "Gateway Timeout"}))
    respx.get(f"{DSS}/parquet").mock(return_value=httpx.Response(200, json={
        "parquet_files": [], "failed": [{"kind": "config-parquet"}]
    }))
    respx.get(f"{HF_API}/datasets/user/no-conv/tree/main").mock(return_value=httpx.Response(200, json=[
        {"type": "file", "path": ".gitattributes", "size": 100},
        {"type": "file", "path": "crop_yield.csv", "size": len(csv_payload)},
    ]))
    respx.get("https://huggingface.co/datasets/user/no-conv/resolve/main/crop_yield.csv").mock(
        return_value=httpx.Response(200, content=csv_payload)
    )

    columns, rows = await HuggingFaceAdapter().get_preview("user/no-conv", limit=5)
    assert columns == ["state", "crop", "yield"]
    assert rows == [{"state": "Puducherry", "crop": "Rice", "yield": 2.56}]
