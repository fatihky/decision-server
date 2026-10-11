"""Backends that auto-download and serve Hub decision models.

Two backends cover the ecosystem:

- :class:`GGUFBackend` — downloads a decision GGUF and drives ``llama-server``,
  which implements ``/v1/systemone`` natively. Works for any decision model that
  ships a GGUF (the uniform path).
- :class:`AutoMapBackend` — loads a transformer checkpoint that registers an
  ``auto_map``/custom decision interface and calls it. Used when no GGUF exists.

Both download from the Hub and cache, so no manual weight management is needed.
"""

from __future__ import annotations

from typing import Any

from .backend import BackendError
from .cloud import CloudBackend
from .llama import LlamaServer, download_gguf
from .schema import DecisionRequest, Prediction, Usage


class GGUFBackend(CloudBackend):
    """Serve a downloaded decision GGUF through a managed ``llama-server``."""

    def __init__(
        self,
        repo_id: str,
        *,
        filename: str | None = None,
        revision: str | None = None,
        gpu_layers: int | None = None,
        context: int | None = None,
        parallel: int | None = None,
        extra_args: list[str] | None = None,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.requested_revision = revision
        self.gpu_layers = gpu_layers
        self.context = context
        self.parallel = parallel
        self.extra_args = extra_args or []
        self._path = None
        self._server: LlamaServer | None = None
        self.model = repo_id
        self._started = False

    def _ensure(self) -> None:
        if self._started:
            return
        self._path = download_gguf(self.repo_id, self.filename, revision=self.requested_revision)
        self._server = LlamaServer(
            self._path,
            model_name=self.repo_id,
            gpu_layers=self.gpu_layers,
            context=self.context,
            parallel=self.parallel,
            extra_args=self.extra_args,
        )
        handle = self._server.start()
        self.base_url = handle.base_url
        self.endpoint = "/v1/systemone"
        self.rounding_decimals = 6
        self.retries = 1
        self._resolved = None
        from threading import Lock

        import httpx

        self._lock = Lock()
        self._usage = Usage()
        self.client = httpx.Client(base_url=self.base_url, timeout=600)
        self._started = True

    def predict(self, request: DecisionRequest) -> list[Prediction]:
        self._ensure()
        return super().predict(request)

    def revision(self) -> str | None:
        from .hub import resolve_revision

        try:
            return resolve_revision(self.repo_id, self.requested_revision)
        except Exception:  # noqa: BLE001 - revision is informational
            return self.requested_revision or self.repo_id

    def settings(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "gguf_file": self.filename,
            "server": "llama.cpp llama-server",
            "endpoint": "/v1/systemone",
            "gpu_layers": self.gpu_layers,
            "context": self.context,
            "managed": "decision-server",
        }

    def close(self) -> None:
        if self._started:
            try:
                super().close()
            finally:
                if self._server is not None:
                    self._server.stop()


class AutoMapBackend:
    """Serve a transformer decision model through its custom/auto_map interface.

    The checkpoint is expected to expose a callable that accepts the typed
    questions and returns probabilities per option. Because interfaces differ,
    this backend tries the documented ``AutoModel`` entry point and falls back
    to a sequence-classification head.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        trust_remote_code: bool = True,
        context_limit: int = 8192,
    ):
        import importlib

        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise BackendError("Install the transformers extra (torch + transformers)") from exc
        self.torch = torch
        self.repo_id = repo_id
        self.revision_arg = revision
        self.device = device
        self.context_limit = context_limit
        self._usage = Usage()
        self.model: Any = None
        self.tokenizer: Any = None
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                repo_id, revision=revision, trust_remote_code=trust_remote_code
            )
            self.model = transformers.AutoModel.from_pretrained(
                repo_id, revision=revision, trust_remote_code=trust_remote_code
            ).to(device).eval()
        except Exception as exc:  # noqa: BLE001 - report load failures clearly
            raise BackendError(f"Could not load {repo_id}: {exc}") from exc

    def predict(self, request: DecisionRequest) -> list[Prediction]:
        import json

        answers: list[Prediction] = []
        for q in request.questions:
            instructions = q.instructions
            context = json.dumps(request.state, ensure_ascii=False, sort_keys=True)
            pairs = [(f"{instructions}\n\nState: {context}", o.description) for o in q.options]
            encoded = self.tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            lengths = encoded["attention_mask"].sum(dim=1)
            if int(lengths.max()) > self.context_limit:
                raise BackendError(
                    f"Input requires {int(lengths.max())} tokens, exceeding {self.context_limit}"
                )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with self.torch.no_grad():
                output = self.model(**encoded)
            logits = self._decision_logits(output, len(q.options))
            probabilities = self.torch.softmax(logits, dim=-1)[0].tolist()
            result = {o.id: float(p) for o, p in zip(q.options, probabilities, strict=True)}
            answers.append(Prediction(question_id=q.id, probabilities=result))
            self._usage.requests += 1
            self._usage.input_tokens += int(encoded["input_ids"].numel())
        return answers

    def _decision_logits(self, output: Any, n_options: int):
        logits = getattr(output, "logits", None)
        if logits is None:
            raise BackendError("Model returned no logits")
        logits = logits.float()
        if logits.shape[-1] == n_options:
            return logits[0]
        # Fall back to a single scalar per option (last logit).
        return logits[:, -1].flatten()

    def revision(self) -> str | None:
        return self.revision_arg or self.repo_id

    def settings(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "device": self.device,
            "context_limit": self.context_limit,
            "transport": "in-process-transformers",
        }

    def usage(self) -> Usage:
        return self._usage.model_copy()

    def close(self) -> None:
        del self.model
