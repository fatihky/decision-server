"""Registry loading and service behaviour tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from decision_server.backend import BackendError
from decision_server.registry import Registry, default_registry
from decision_server.schema import DecisionRequest
from decision_server.service import run

CATALOG = """
models:
  - id: demo
    label: Demo
    kind: local
    backend: plugin
    plugin: decision_server.plugins.uniform
    tasks: [choice, noul, score]
  - id: needs-torch
    label: Needs Torch
    kind: local
    backend: plugin
    plugin: decision_server.plugins.uniform
    requires: [definitely_not_installed_module_xyz]
    tasks: [choice]
"""


@pytest.fixture()
def registry(tmp_path: Path) -> Registry:
    path = tmp_path / "models.yaml"
    path.write_text(CATALOG, encoding="utf-8")
    return Registry.from_file(path)


def _request(task: str) -> DecisionRequest:
    if task == "noul":
        options = [{"id": "false", "description": "No"}, {"id": "true", "description": "Yes"}]
    elif task == "score":
        options = [
            {"id": "a", "description": "A", "value": 1},
            {"id": "b", "description": "B", "value": 2},
        ]
    else:
        options = [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}]
    return DecisionRequest.model_validate(
        {
            "state": {},
            "questions": [
                {"id": "q", "task": task, "instructions": "decide", "options": options}
            ],
        }
    )


def test_default_registry_loads():
    assert default_registry().descriptions()


def test_availability_reflects_missing_modules(registry: Registry):
    by_id = {d.id: d for d in registry.descriptions()}
    assert by_id["demo"].available
    assert not by_id["needs-torch"].available
    assert by_id["needs-torch"].unavailable_reason
    assert "definitely_not_installed_module_xyz" in by_id["needs-torch"].unavailable_reason


def test_uniform_backend_runs(registry: Registry):
    response = run(registry, "demo", _request("choice"))
    assert response.answers[0].probabilities == {"a": 0.5, "b": 0.5}


def test_unsupported_task_rejected(registry: Registry):
    with pytest.raises(ValueError):
        run(registry, "needs-torch", _request("score"))


def test_backend_failure_is_isolated_per_question(registry: Registry, monkeypatch):
    backend = registry.get("demo")

    def boom(request):
        raise RuntimeError("inference exploded")

    monkeypatch.setattr(backend, "predict", boom)
    response = run(registry, "demo", _request("choice"))
    assert response.answers[0].error == "inference exploded"
    assert response.answers[0].probabilities is None


def test_duplicate_model_ids_rejected(tmp_path: Path):
    path = tmp_path / "models.yaml"
    path.write_text(
        """
models:
  - {id: a, kind: local, backend: unavailable, tasks: [], description: one}
  - {id: a, kind: local, backend: unavailable, tasks: [], description: two}
""",
        encoding="utf-8",
    )
    with pytest.raises(BackendError):
        Registry.from_file(path)
