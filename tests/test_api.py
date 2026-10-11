"""Tests for the JSON HTTP API using FastAPI's TestClient."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from decision_server.api import create_api
from decision_server.registry import Registry

CATALOG = """
models:
  - id: demo
    label: Demo
    kind: local
    backend: plugin
    plugin: decision_server.plugins.uniform
    tasks: [choice, noul, score]
"""


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    path = tmp_path / "models.yaml"
    path.write_text(CATALOG, encoding="utf-8")
    return TestClient(create_api(Registry.from_file(path)))


def _choice_request(model: str = "demo") -> dict:
    return {
        "model": model,
        "state": {"x": 1},
        "questions": [
            {
                "id": "q",
                "task": "choice",
                "instructions": "pick",
                "options": [
                    {"id": "a", "description": "A"},
                    {"id": "b", "description": "B"},
                ],
            }
        ],
    }


def test_health(client: TestClient):
    assert client.get("/health").json() == {"status": "ok"}


def test_list_models(client: TestClient):
    models = client.get("/v1/models").json()
    assert [m["id"] for m in models] == ["demo"]


def test_get_model(client: TestClient):
    body = client.get("/v1/models/demo").json()
    assert body["backend"] == "plugin"
    assert body["tasks"] == ["choice", "noul", "score"]


def test_get_unknown_model(client: TestClient):
    assert client.get("/v1/models/nope").status_code == 404


def test_systemone_runs(client: TestClient):
    response = client.post("/v1/systemone", json=_choice_request())
    assert response.status_code == 200
    answers = response.json()["answers"]
    assert answers[0]["probabilities"] == {"a": 0.5, "b": 0.5}


def test_systemone_unknown_model(client: TestClient):
    response = client.post("/v1/systemone", json=_choice_request("nope"))
    assert response.status_code == 404


def test_systemone_invalid_question(client: TestClient):
    body = _choice_request()
    body["questions"][0]["options"] = [{"id": "a", "description": "A"}]  # only one option
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 422


def test_systemone_unsupported_task(client: TestClient):
    path_catalog = """
models:
  - id: choiceonly
    kind: local
    backend: plugin
    plugin: decision_server.plugins.uniform
    tasks: [choice]
"""
    import tempfile

    path = Path(tempfile.mkdtemp()) / "m.yaml"
    path.write_text(path_catalog, encoding="utf-8")
    local = TestClient(create_api(Registry.from_file(path)))
    body = _choice_request("choiceonly")
    body["questions"][0] = {
        "id": "q",
        "task": "noul",
        "instructions": "judge",
        "options": [
            {"id": "false", "description": "No"},
            {"id": "true", "description": "Yes"},
        ],
    }
    assert local.post("/v1/systemone", json=body).status_code == 422


def test_openapi_available(client: TestClient):
    schema = client.get("/openapi.json").json()
    assert "/v1/systemone" in schema["paths"]
