"""Tests for Hub discovery, catalog loading and backend selection."""

from __future__ import annotations

from pathlib import Path

from decision_server.registry import Registry, default_registry


def test_default_registry_includes_ecosystem_models():
    registry = default_registry()
    ids = {d.id for d in registry.descriptions()}
    # Curated entries
    assert "typesafe" in ids
    assert "llama-server" in ids
    # Generated GGUF entries from the ecosystem catalog
    assert "ggml-org__Kev-0.8B-GGUF" in ids
    assert "ggml-org__Laya-GGUF" in ids


def test_ecosystem_entries_are_gguf_backend():
    registry = default_registry()
    by_id = {d.id: d for d in registry.descriptions()}
    kev = by_id["ggml-org__Kev-0.8B-GGUF"]
    assert kev.backend == "gguf"
    assert kev.kind == "local"
    assert kev.source == "https://huggingface.co/ggml-org/Kev-0.8B-GGUF"


def test_gguf_availability_depends_on_llama_binary(monkeypatch):
    import decision_server.llama as llama

    monkeypatch.setattr(llama, "llama_binary", lambda: None)
    registry = default_registry()
    by_id = {d.id: d for d in registry.descriptions()}
    assert not by_id["ggml-org__Kev-0.8B-GGUF"].available
    reason = by_id["ggml-org__Kev-0.8B-GGUF"].unavailable_reason or ""
    assert "llama-server" in reason


def test_gguf_available_when_binary_present(monkeypatch):
    import decision_server.llama as llama

    monkeypatch.setattr(llama, "llama_binary", lambda: "/usr/bin/llama-server")
    registry = default_registry()
    by_id = {d.id: d for d in registry.descriptions()}
    assert by_id["ggml-org__Kev-0.8B-GGUF"].available


def test_from_files_merges_and_dedupes(tmp_path: Path):
    a = tmp_path / "a.yaml"
    a.write_text(
        "models:\n  - {id: x, kind: local, backend: plugin, plugin: p}\n",
        encoding="utf-8",
    )
    b = tmp_path / "b.yaml"
    b.write_text(
        "gguf:\n  - {id: y, label: Y, repo_id: owner/repo, gguf_file: m.gguf}\n",
        encoding="utf-8",
    )
    registry = Registry.from_files([a, b])
    ids = {d.id for d in registry.descriptions()}
    assert ids == {"x", "y"}
    by_id = {d.id: d for d in registry.descriptions()}
    assert by_id["y"].backend == "gguf"


def test_unavailable_backend_reports_reason(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "models:\n  - {id: z, kind: local, backend: unavailable, tasks: [], description: nope}\n",
        encoding="utf-8",
    )
    registry = Registry.from_file(path)
    desc = registry.descriptions()[0]
    assert not desc.available
    assert desc.unavailable_reason == "nope"


def test_cloud_requires_api_key(tmp_path: Path, monkeypatch):
    path = tmp_path / "cloud.yaml"
    path.write_text(
        "models:\n"
        "  - id: c\n"
        "    kind: cloud\n"
        "    backend: cloud\n"
        "    model: m\n"
        "    base_url: https://example.test\n"
        "    endpoint: /v1/systemone\n"
        "    api_key_env: TEST_CLOUD_KEY\n"
        "    tasks: [choice]\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TEST_CLOUD_KEY", raising=False)
    registry = Registry.from_file(path)
    desc = registry.descriptions()[0]
    assert not desc.available
    assert "TEST_CLOUD_KEY" in (desc.unavailable_reason or "")

    monkeypatch.setenv("TEST_CLOUD_KEY", "secret")
    assert Registry.from_file(path).descriptions()[0].available
