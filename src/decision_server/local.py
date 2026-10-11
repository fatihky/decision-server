"""Local backends: run inference in-process from a checkpoint or plugin.

Local models fall into two groups:

1. **Plugin backends** — a Python module that exposes a ``build(config)``
   function returning a :class:`~decision_server.backend.Backend`. The module
   path is given in the registry. This is how model-specific inference (which
   varies widely per family) is attached without editing the server.

2. **Caller-managed HTTP servers** — an external ``llama-server`` (or similar)
   exposing the typed-question endpoint. The local backend behaves like the
   cloud proxy but without credentials and with a local base URL.

Both produce the same typed answers as cloud backends.
"""

from __future__ import annotations

import importlib
import json
from threading import Lock

from .backend import Backend, BackendError
from .cloud import CloudBackend
from .schema import DecisionRequest, Prediction, Usage


class LocalHTTPBackend(CloudBackend):
    """Typed questions through a locally started server (no credentials)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        endpoint: str = "/v1/systemone",
        rounding_decimals: int = 6,
        timeout: float = 600,
        retries: int = 2,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.rounding_decimals = rounding_decimals
        self.retries = retries
        self._lock = Lock()
        self._resolved: str | None = None
        self._usage = Usage()
        import httpx

        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def settings(self) -> dict:
        return {
            "endpoint": f"{self.base_url}{self.endpoint}",
            "transport": "local-typed-questions",
            "managed": "caller",
        }


class PluginBackend:
    """Wrap a plugin-provided backend, adding error isolation per request."""

    def __init__(self, inner: Backend, *, plugin: str, settings: dict | None = None):
        self.inner = inner
        self.plugin = plugin
        self._settings = settings or {}

    def predict(self, request: DecisionRequest) -> list[Prediction]:
        return self.inner.predict(request)

    def revision(self) -> str | None:
        return self.inner.revision()

    def settings(self) -> dict:
        merged = {"plugin": self.plugin, "transport": "in-process"}
        merged.update(self.inner.settings())
        merged.update(self._settings)
        return merged

    def usage(self) -> Usage:
        return self.inner.usage()

    def close(self) -> None:
        self.inner.close()


def load_plugin(module_path: str, config: dict) -> Backend:
    """Import ``module_path`` and call its ``build(config)`` entry point."""

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise BackendError(
            f"Cannot import plugin {module_path!r}; install its runtime extra"
        ) from exc
    build = getattr(module, "build", None)
    if build is None:
        raise BackendError(f"Plugin {module_path!r} has no build(config) entry point")
    try:
        inner = build(dict(config))
    except Exception as exc:  # noqa: BLE001 - surface construction failures to the caller
        raise BackendError(f"Plugin {module_path!r} failed to load: {exc}") from exc
    return PluginBackend(inner, plugin=module_path)


def dumps_state(state: object) -> str:
    """Serialize state for plugins that need an exact JSON boundary."""

    return json.dumps(state, ensure_ascii=False, allow_nan=False)
