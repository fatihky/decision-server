"""Cloud proxy decoding tests using a stubbed HTTP transport."""

from __future__ import annotations

import httpx
import pytest

from decision_server.backend import BackendError
from decision_server.cloud import CloudBackend
from decision_server.local import LocalHTTPBackend
from decision_server.schema import DecisionRequest


def _request(task: str):
    if task == "noul":
        options = [{"id": "false", "description": "No"}, {"id": "true", "description": "Yes"}]
    elif task == "score":
        options = [
            {"id": "low", "description": "Low", "value": 1},
            {"id": "high", "description": "High", "value": 2},
        ]
    else:
        options = [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}]
    return DecisionRequest.model_validate(
        {
            "state": {"x": 1},
            "questions": [
                {"id": "q1", "task": task, "instructions": "decide", "options": options}
            ],
        }
    )


@pytest.fixture()
def backend(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret")
    b = CloudBackend(
        "m",
        base_url="https://example.test",
        endpoint="/v1/systemone",
        api_key_env="TEST_API_KEY",
        rounding_decimals=2,
    )
    yield b
    b.close()


def _install(backend, payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    backend.client = httpx.Client(
        base_url="https://example.test", transport=httpx.MockTransport(handler)
    )


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    with pytest.raises(BackendError):
        CloudBackend(
            "m",
            base_url="https://example.test",
            endpoint="/v1/systemone",
            api_key_env="TEST_API_KEY",
        )


def test_choice_anonymous_keys_mapped_back(backend):
    _install(
        backend,
        {
            "model": "m",
            "usage": {"input_tokens": 5, "output_tokens": 1},
            "answers": {
                "q1": {
                    "type": "choice",
                    "probabilities": {"option_0": 0.25, "option_1": 0.75},
                }
            },
        },
    )
    result = backend.predict(_request("choice"))
    assert result[0].probabilities == {"a": 0.25, "b": 0.75}
    assert backend.usage().input_tokens == 5


def test_score_levels_ordered_by_value(backend):
    _install(
        backend,
        {
            "model": "m",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "answers": {"q1": {"type": "score", "probabilities": {"0": 0.4, "1": 0.6}}},
        },
    )
    result = backend.predict(_request("score"))
    assert result[0].probabilities == {"low": 0.4, "high": 0.6}


def test_noul_maps_true_probability(backend):
    _install(
        backend,
        {
            "model": "m",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "answers": {"q1": {"type": "noul", "noul": 0.8}},
        },
    )
    result = backend.predict(_request("noul"))
    assert result[0].probabilities == {"false": pytest.approx(0.2), "true": pytest.approx(0.8)}


def test_rounded_probabilities_renormalized(backend):
    _install(
        backend,
        {
            "model": "m",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "answers": {
                "q1": {
                    "type": "choice",
                    "probabilities": {"option_0": 0.33, "option_1": 0.66},
                }
            },
        },
    )
    result = backend.predict(_request("choice"))
    assert sum(result[0].probabilities.values()) == pytest.approx(1.0)


def test_http_error_raises(backend):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    backend.client = httpx.Client(
        base_url="https://example.test", transport=httpx.MockTransport(handler)
    )
    backend.retries = 1
    with pytest.raises(BackendError):
        backend.predict(_request("choice"))


def test_local_http_backend_needs_no_key(monkeypatch):
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    b = LocalHTTPBackend("m", base_url="http://127.0.0.1:8080")
    try:
        assert b.settings()["managed"] == "caller"
    finally:
        b.close()
