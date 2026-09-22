from __future__ import annotations

import concurrent.futures
import io
import json
import time
import zipfile

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from rl_automl.api.app import create_app
from rl_automl.api.routes.runs import _render_event
from rl_automl.api.run_manager import RunManager
from rl_automl.core.types import RunObject, RunStatus


def _wait(client, run_id, wanted, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/runs/{run_id}").json()
        if payload["status"] in wanted:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"run did not reach {wanted}")


def test_api_upload_source_plan_sse_and_approval_race(config, monkeypatch):
    app = create_app(config)
    manager = app.state.manager
    submitted = []
    monkeypatch.setattr(manager, "submit_execution", lambda run_id, run: submitted.append(run_id))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        source = client.post("/datasets/from-source", json={"source": "iris", "name": "iris.csv"})
        assert source.status_code == 201
        dataset = source.json()
        assert dataset["n_rows"] == 150
        assert (
            client.post(
                "/datasets",
                files={"file": ("bad.exe", b"not,csv\n1,2", "application/octet-stream")},
            ).status_code
            == 400
        )

        created = client.post(
            "/runs",
            json={
                "dataset_id": dataset["dataset_id"],
                "problem_statement": "Classify iris target using measurements",
                "target": "species",
                "task_type": "classification",
                "metric": "f1",
                "use_policy": False,
            },
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]
        planned = _wait(client, run_id, {"awaiting_approval", "failed"})
        assert planned["status"] == "awaiting_approval", planned.get("error")
        recommendation = client.get(f"/runs/{run_id}/recommendations")
        assert recommendation.status_code == 200
        assert recommendation.json()["requires_user_approval"]

        def approve():
            return client.post(f"/runs/{run_id}/approve", json={"selection": "all"})

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: approve(), range(2)))
        assert sorted(response.status_code for response in responses) == [202, 409]
        assert submitted == [run_id]


def test_sse_event_rendering_is_well_formed():
    rendered = _render_event({"stage": "planning", "sequence": 2, "message": "working"})
    assert rendered.startswith("event: planning\ndata: ")
    payload = json.loads(rendered.split("data: ", 1)[1])
    assert payload["sequence"] == 2


def _csv(rows: int = 40) -> str:
    lines = ["number,category,outcome"]
    for i in range(rows):
        lines.append(f"{i},{'north' if i % 2 else 'south'},{'yes' if i % 3 else 'no'}")
    return "\n".join(lines) + "\n"


def _zip_bytes(members: dict[str, str], compress: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compress) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def test_archive_upload_registers_each_table_and_skips_the_rest(config):
    archive = _zip_bytes(
        {
            "train.csv": _csv(),
            "test.csv": _csv(),
            "notes.md": "# not a table",
            "__MACOSX/._train.csv": "junk",
        }
    )
    app = create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/datasets/archive", files={"file": ("kaggle.zip", archive, "application/zip")}
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert sorted(item["name"] for item in body["datasets"]) == ["test.csv", "train.csv"]
        assert all(item["n_rows"] == 40 for item in body["datasets"])
        reasons = {item["name"]: item["reason"] for item in body["rejected"]}
        assert "notes.md" in reasons
        # Both tables are usable, so a caller can pick train.csv to train on.
        assert len({item["dataset_id"] for item in body["datasets"]}) == 2


def test_archive_upload_refuses_path_traversal(config):
    archive = _zip_bytes({"../escaped.csv": _csv(), "good.csv": _csv()})
    app = create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/datasets/archive", files={"file": ("evil.zip", archive, "application/zip")}
        )
        assert response.status_code == 201
        body = response.json()
        assert [item["name"] for item in body["datasets"]] == ["good.csv"]
        assert any("unsafe path" in item["reason"] for item in body["rejected"])

    # Nothing may be written outside the artifact tree.
    assert not (config.artifacts_dir().parent / "escaped.csv").exists()


def test_archive_upload_refuses_a_zip_bomb(config):
    # 4 MB of one repeated byte compresses to almost nothing, so the ratio guard trips.
    archive = _zip_bytes({"huge.csv": "0" * (4 * 1024 * 1024)})
    app = create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/datasets/archive", files={"file": ("bomb.zip", archive, "application/zip")}
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["code"] == "security_error"
        assert "inflates" in detail["message"]


def test_archive_endpoint_rejects_a_non_archive(config):
    app = create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/datasets/archive", files={"file": ("data.csv", _csv(), "text/csv")}
        )
        assert response.status_code == 400
        assert "not a readable ZIP" in response.json()["detail"]["message"]


def test_dataset_sources_endpoint_lists_usable_names(config):
    app = create_app(config)
    with TestClient(app) as client:
        response = client.get("/datasets/sources")
        assert response.status_code == 200
        sources = response.json()
        names = {source["name"] for source in sources}
        # The literal path must not be swallowed by the /{dataset_id} route.
        assert "iris" in names
        assert {"name", "task_type", "metric", "requires_network"} <= set(sources[0])


def test_model_catalog_reports_runnable_task_types(config):
    app = create_app(config)
    with TestClient(app) as client:
        response = client.get("/models")
        assert response.status_code == 200
        catalog = response.json()
        # The fixture enables only supervised models, so clustering must be reported as
        # unrunnable rather than left for a client to discover by failing a run.
        assert "classification" in catalog["runnable_tasks"]
        assert "clustering" not in catalog["runnable_tasks"]
        assert catalog["by_task"]["clustering"] == []
        # Everything offered for a task must be one of the enabled keys.
        assert set(catalog["by_task"]["classification"]) <= set(catalog["enabled"])


def test_dashboard_is_served_at_root(config):
    app = create_app(config)
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "RL-AutoML" in response.text


def test_artifact_endpoint_serves_only_recorded_files_under_root(config):
    manager = RunManager(config)
    run = RunObject(run_id="run_artifact", status=RunStatus.COMPLETE)
    directory = config.artifacts_dir() / "runs" / run.run_id
    directory.mkdir(parents=True)
    model_zip = directory / "model.zip"
    model_zip.write_bytes(b"PK-test")
    run.artifacts.model_zip = str(model_zip)
    run.artifacts.model_zip_sha256 = "hash"
    manager._store(run)
    app = create_app(config, manager_factory=lambda _cfg: manager)
    with TestClient(app) as client:
        listing = client.get(f"/runs/{run.run_id}/artifacts")
        assert listing.status_code == 200
        assert listing.json()["artifacts"][0]["sha256"] == "hash"
        download = client.get(f"/runs/{run.run_id}/artifacts/model.zip")
        assert download.status_code == 200
        assert download.content == b"PK-test"

        outside = config.artifacts_dir().parent / "outside.zip"
        outside.write_bytes(b"secret")
        run.artifacts.model_zip = str(outside)
        manager._store(run)
        assert client.get(f"/runs/{run.run_id}/artifacts/model.zip").status_code == 404
