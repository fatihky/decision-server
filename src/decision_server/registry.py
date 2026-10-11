"""Model registry: a declarative catalog of every servable model.

The catalog is loaded from ``models.yaml`` shipped with the package. Each
entry declares how to construct its backend. No model weights, secrets or
dataset paths live here — only the routing information.

Entry shape::

    models:
      - id: my-local-model
        label: My Local Model
        kind: local            # local | cloud
        backend: plugin        # plugin | local-http | cloud
        tasks: [choice, noul, score]
        plugin: my_pkg.backends.my_model      # for backend: plugin
        config: {device: cuda:0}              # passed to the plugin
        requires: [torch, transformers]       # runtime extras
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .backend import Backend, BackendError
from .cloud import CloudBackend
from .local import LocalHTTPBackend, load_plugin
from .schema import ModelDescription
from .servable import AutoMapBackend, GGUFBackend


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class Registry:
    """Constructs and caches backends from the declarative catalog."""

    def __init__(self, entries: list[dict[str, Any]]):
        ids = [entry["id"] for entry in entries]
        if len(set(ids)) != len(ids):
            raise BackendError("Duplicate model ids in catalog")
        self._entries = {entry["id"]: entry for entry in entries}
        self._cache: dict[str, Backend] = {}

    @classmethod
    def from_file(cls, path: Path) -> Registry:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = data.get("models")
        if not isinstance(entries, list) or not entries:
            raise BackendError(f"No models defined in {path}")
        return cls(entries)

    @classmethod
    def from_files(cls, paths: list[Path]) -> Registry:
        """Merge several catalogs; each file may use ``models`` or ``gguf`` keys."""

        merged: list[dict[str, Any]] = []
        ids: set[str] = set()
        for path in paths:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for key in ("models", "gguf", "automap"):
                for entry in data.get(key) or []:
                    if entry["id"] in ids:
                        continue
                    ids.add(entry["id"])
                    if key == "gguf":
                        entry = {"kind": "local", "backend": "gguf", **entry}
                    elif key == "automap":
                        entry = {"kind": "local", "backend": "automap", **entry}
                    if entry.get("repo_id") and not entry.get("source"):
                        entry["source"] = f"https://huggingface.co/{entry['repo_id']}"
                    merged.append(entry)
        if not merged:
            raise BackendError("No models defined in the supplied catalogs")
        return cls(merged)

    def descriptions(self) -> list[ModelDescription]:
        out: list[ModelDescription] = []
        for entry in self._entries.values():
            requires = entry.get("requires", [])
            missing = [m for m in requires if not _has_module(m)]
            unavailable = self._unavailable(entry, missing)
            out.append(
                ModelDescription(
                    id=entry["id"],
                    label=entry.get("label", entry["id"]),
                    kind=entry["kind"],
                    backend=entry["backend"],
                    tasks=entry.get("tasks", ["choice", "noul", "score"]),
                    description=entry.get("description", ""),
                    source=entry.get("source"),
                    revision=entry.get("revision"),
                    requires=requires,
                    available=unavailable is None,
                    unavailable_reason=unavailable,
                )
            )
        return out

    def _unavailable(self, entry: dict[str, Any], missing: list[str]) -> str | None:
        backend = entry["backend"]
        if backend == "unavailable":
            return entry.get("description") or "no servable form"
        if backend == "cloud":
            key_env = entry.get("api_key_env")
            if key_env and not os.environ.get(key_env):
                return f"set {key_env} to enable this cloud model"
        if backend == "gguf":
            from .llama import llama_binary

            if llama_binary() is None:
                return "llama-server not on PATH; install llama.cpp"
        if backend == "automap":
            for module in ("torch", "transformers"):
                if not _has_module(module):
                    return "missing runtime: torch, transformers"
        if missing:
            return "missing runtime: " + ", ".join(missing)
        return None

    def get(self, model_id: str) -> Backend:
        if model_id not in self._entries:
            raise BackendError(f"Unknown model: {model_id}")
        if model_id not in self._cache:
            self._cache[model_id] = self._build(self._entries[model_id])
        return self._cache[model_id]

    def add_entry(self, entry: dict[str, Any]) -> None:
        """Register an extra model at runtime (e.g. one found on the Hub)."""

        if entry["id"] in self._entries:
            return
        if entry.get("repo_id") and not entry.get("source"):
            entry["source"] = f"https://huggingface.co/{entry['repo_id']}"
        self._entries[entry["id"]] = entry

    def add_hub_model(self, repo_id: str) -> ModelDescription:
        """Fetch Hub metadata and register the repo as a servable model."""

        from .hub import describe

        meta = describe(repo_id)
        entry: dict[str, Any] = {
            "id": repo_id.replace("/", "__"),
            "label": repo_id,
            "kind": "local",
            "revision": meta.get("sha"),
            "source": f"https://huggingface.co/{repo_id}",
        }
        if meta["is_gguf"]:
            entry.update(
                backend="gguf",
                repo_id=repo_id,
                gguf_file=meta["gguf_files"][0] if meta["gguf_files"] else None,
                tasks=["choice", "noul", "score"],
            )
        elif meta["has_custom_code"]:
            entry.update(
                backend="automap", repo_id=repo_id, tasks=["choice", "noul", "score"]
            )
        else:
            raise BackendError(
                f"{repo_id} has no GGUF and no custom auto_map interface; "
                "write a family plugin to serve it"
            )
        self.add_entry(entry)
        for desc in self.descriptions():
            if desc.id == entry["id"]:
                return desc
        raise BackendError(f"Could not register {repo_id}")

    def _build(self, entry: dict[str, Any]) -> Backend:
        backend = entry["backend"]
        config = dict(entry.get("config", {}))
        if backend == "cloud":
            return CloudBackend(
                entry["model"],
                base_url=entry["base_url"],
                endpoint=entry["endpoint"],
                api_key_env=entry["api_key_env"],
                rounding_decimals=entry.get("rounding_decimals", 4),
            )
        if backend == "local-http":
            return LocalHTTPBackend(
                entry["model"],
                base_url=entry.get("base_url", "http://127.0.0.1:8080"),
                endpoint=entry.get("endpoint", "/v1/systemone"),
                rounding_decimals=entry.get("rounding_decimals", 6),
            )
        if backend == "plugin":
            return load_plugin(entry["plugin"], config)
        if backend == "gguf":
            return GGUFBackend(
                entry["repo_id"],
                filename=entry.get("gguf_file"),
                revision=entry.get("revision"),
                gpu_layers=config.get("gpu_layers"),
                context=config.get("context"),
                parallel=config.get("parallel"),
                extra_args=config.get("extra_args"),
            )
        if backend == "automap":
            return AutoMapBackend(
                entry["repo_id"],
                revision=entry.get("revision"),
                device=config.get("device", "cpu"),
                dtype=config.get("dtype", "float32"),
                context_limit=int(config.get("context_limit", 8192)),
            )
        raise BackendError(f"Unsupported backend kind: {backend}")

    def close(self) -> None:
        for backend in self._cache.values():
            try:
                backend.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        self._cache.clear()


@lru_cache(maxsize=1)
def default_registry() -> Registry:
    """Load the curated catalog plus the generated ecosystem and family catalogs."""

    base = Path(__file__).parent
    return Registry.from_files(
        [base / "models.yaml", base / "ecosystem.yaml", base / "families.yaml"]
    )
