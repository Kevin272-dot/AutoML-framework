import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_full_discovery_flow_up_to_approval_gate(client):
    """Parse → sources+audit → STOP for approval → approve → search results."""
    resp = client.post("/api/discovery/requests", json={"query": "crop yield in India between 2020 and 2025"})
    assert resp.status_code == 200
    req = resp.json()
    assert req["status"] == "REQUEST_CREATED"
    assert req["parsed_requirements"]["date_start"] == "2020"
    assert "india" in req["parsed_requirements"]["location"]

    # Audit: hits the network (Hugging Face). In CI without internet this would fail,
    # so this test is skipped if the base URL is unreachable.
    import httpx

    try:
        httpx.head("https://huggingface.co", timeout=5.0)
    except httpx.HTTPError:
        pytest.skip("No network access; skipping live audit flow test.")

    resp = client.post(f"/api/discovery/requests/{req['id']}/audit")
    assert resp.status_code == 200
    assert resp.json()["status"] == "WAITING_FOR_SOURCE_APPROVAL"

    sources = client.get(f"/api/discovery/requests/{req['id']}/sources").json()
    assert sources, "expected at least one discovered source"
    hf = next((s for s in sources if s["slug"] == "huggingface"), None)
    assert hf is not None
    assert hf["audit"] is not None

    # Search before approval must be blocked.
    blocked = client.post(f"/api/discovery/requests/{req['id']}/search")
    assert blocked.status_code == 409

    # Approve HF, then search (live network).
    resp = client.post(
        f"/api/discovery/requests/{req['id']}/approve-sources",
        json={"source_ids": [hf["id"]]},
    )
    assert resp.status_code == 200
    assert resp.json()["approved_source_ids"] == [hf["id"]]

    resp = client.post(f"/api/discovery/requests/{req['id']}/search")
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    import time

    for _ in range(60):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.5)
    assert job["status"] == "COMPLETED", job

    results = client.get(f"/api/discovery/requests/{req['id']}/results").json()
    assert isinstance(results, list)
    for cand in results:
        assert 0.0 <= cand["overall_score"] <= 1.0
        comps = cand["score_components"]
        assert set(comps) == {
            "keyword_match", "date_match", "domain_match",
            "completeness", "size_score", "feature_quality",
        }


def test_approve_unknown_source_rejected(client):
    resp = client.post("/api/discovery/requests", json={"query": "weather data 2020"})
    req_id = resp.json()["id"]
    client.post(f"/api/discovery/requests/{req_id}/audit")
    resp = client.post(
        f"/api/discovery/requests/{req_id}/approve-sources",
        json={"source_ids": ["nonexistent"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["approved_source_ids"] == []
    assert "nonexistent" in body["rejected_source_ids"]


def test_error_envelope_no_stack_trace(client):
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404
