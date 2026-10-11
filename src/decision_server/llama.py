"""Drive a local ``llama-server`` for GGUF decision models.

``llama-server`` exposes ``/v1/systemone`` natively when a GGUF carries
decision metadata (``<arch>.decision.type``). This module:

1. downloads the chosen ``.gguf`` from the Hub into a local cache,
2. launches ``llama-server`` bound to a free localhost port,
3. waits for the decision endpoint to become healthy, and
4. tears the process down on close.

The binary is located from ``DECISION_SERVER_LLAMA_BIN`` or ``PATH``. When it
is absent the model is still listed, but starting it fails with a clear message.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .backend import BackendError


def llama_binary() -> str | None:
    explicit = os.environ.get("DECISION_SERVER_LLAMA_BIN")
    if explicit:
        return explicit
    for name in ("llama-server", "llama"):
        found = shutil.which(name)
        if found:
            return found
    return None


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def cache_dir() -> Path:
    override = os.environ.get("DECISION_SERVER_CACHE")
    base = Path(override) if override else Path.home() / ".cache" / "decision-server"
    base.mkdir(parents=True, exist_ok=True)
    return base


def download_gguf(
    repo_id: str,
    filename: str | None = None,
    *,
    revision: str | None = None,
) -> Path:
    """Download one GGUF (or use a cached copy) and return its local path."""

    import huggingface_hub

    if filename is None:
        from .hub import list_gguf_files

        files = list_gguf_files(repo_id, revision)
        if not files:
            raise BackendError(f"No .gguf files found in {repo_id}")
        filename = files[0]
    try:
        path = huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision
        )
    except Exception as exc:  # noqa: BLE001 - report download failures clearly
        raise BackendError(f"Could not download {repo_id}/{filename}: {exc}") from exc
    return Path(path)


@dataclass
class ServerHandle:
    process: subprocess.Popen
    port: int
    base_url: str


class LlamaServer:
    """A managed ``llama-server`` subprocess serving one GGUF."""

    def __init__(
        self,
        gguf_path: Path,
        *,
        model_name: str,
        gpu_layers: int | None = None,
        context: int | None = None,
        parallel: int | None = None,
        extra_args: list[str] | None = None,
        startup_timeout: float = 180.0,
    ):
        binary = llama_binary()
        if not binary:
            raise BackendError(
                "llama-server not found; install llama.cpp or set DECISION_SERVER_LLAMA_BIN"
            )
        self.binary = binary
        self.gguf_path = Path(gguf_path)
        self.model_name = model_name
        self.port = find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.startup_timeout = startup_timeout
        args = [binary, "-m", str(self.gguf_path), "--port", str(self.port), "--host", "127.0.0.1"]
        if gpu_layers is not None:
            args += ["-ngl", str(gpu_layers)]
        if context is not None:
            args += ["-c", str(context)]
        if parallel is not None:
            args += ["-np", str(parallel)]
        args += extra_args or []
        self._args = args
        self._handle: ServerHandle | None = None

    def start(self) -> ServerHandle:
        log = cache_dir() / f"llama-server-{self.port}.log"
        with log.open("wb") as stream:
            self._handle = ServerHandle(
                process=subprocess.Popen(self._args, stdout=stream, stderr=subprocess.STDOUT),
                port=self.port,
                base_url=self.base_url,
            )
        self._wait_ready()
        return self._handle

    def _wait_ready(self) -> None:
        import httpx

        deadline = time.monotonic() + self.startup_timeout
        last_error: str | None = None
        while time.monotonic() < deadline:
            assert self._handle is not None
            if self._handle.process.poll() is not None:
                raise BackendError(f"llama-server exited early; see logs in {cache_dir()}")
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    return
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(1.0)
        raise BackendError(f"llama-server did not become ready: {last_error or 'timeout'}")

    def stop(self) -> None:
        if self._handle is None:
            return
        process = self._handle.process
        try:
            process.terminate()
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        finally:
            self._handle = None
