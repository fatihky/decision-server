"""Tests for the llama.cpp manager and Hub helpers (no downloads, no server)."""

from __future__ import annotations

from pathlib import Path

import pytest

from decision_server import llama


def test_find_free_port_is_usable():
    port = llama.find_free_port()
    assert 1 <= port <= 65535


def test_llama_binary_env_override(monkeypatch):
    monkeypatch.setenv("DECISION_SERVER_LLAMA_BIN", "/custom/llama-server")
    assert llama.llama_binary() == "/custom/llama-server"


def test_llama_binary_missing(monkeypatch):
    monkeypatch.delenv("DECISION_SERVER_LLAMA_BIN", raising=False)

    def no_which(_):
        return None

    monkeypatch.setattr(llama.shutil, "which", no_which)
    assert llama.llama_binary() is None


def test_cache_dir_honours_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DECISION_SERVER_CACHE", str(tmp_path / "c"))
    assert llama.cache_dir() == tmp_path / "c"
    assert llama.cache_dir().is_dir()


def test_llama_server_requires_binary(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DECISION_SERVER_LLAMA_BIN", "/definitely/missing/llama-server")
    # binary path is set but does not exist -> server will fail on start, not construct
    server = llama.LlamaServer(tmp_path / "m.gguf", model_name="m")
    assert server.base_url.startswith("http://127.0.0.1:")


def test_llama_server_no_binary(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("DECISION_SERVER_LLAMA_BIN", raising=False)
    monkeypatch.setattr(llama.shutil, "which", lambda _: None)
    with pytest.raises(llama.BackendError):
        llama.LlamaServer(tmp_path / "m.gguf", model_name="m")


def test_hub_search_returns_dataclass(monkeypatch):
    from decision_server import hub

    class FakeInfo:
        def __init__(self):
            self.__dict__ = {
                "id": "owner/model",
                "downloads": 10,
                "likes": 2,
                "tags": ["decision-model", "gguf"],
                "library_name": "gguf",
                "sha": "abc",
            }

    class FakeHub:
        def list_models(self, **kwargs):
            return [FakeInfo()]

    monkeypatch.setattr(hub, "_client", lambda: FakeHub())
    models = hub.search_models(limit=1)
    assert models[0].id == "owner/model"
    assert models[0].is_gguf
    assert models[0].servable_via == "gguf"


def test_download_gguf_rejects_empty(monkeypatch, tmp_path: Path):
    import decision_server.hub as hub

    monkeypatch.setattr(hub, "list_gguf_files", lambda *a, **k: [])
    with pytest.raises(llama.BackendError):
        llama.download_gguf("owner/empty")
