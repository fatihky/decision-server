"""Backend for cross-encoder / zero-shot reranker decision models.

Rerankers and zero-shot classifiers score a (query, passage) pair. For a
``choice`` question each option is treated as a passage against the question
and the state; the scores are softmaxed into a distribution. This covers
``gte-reranker``, ``bge-reranker``, ``mxbai-rerank``, ``zerank``,
``ModernBERT-zeroshot`` and similar encoders.

Configuration keys:

- ``repo_id`` (required): the reranker checkpoint.
- ``revision``: checkpoint revision.
- ``kind``: ``reranker`` (default) or ``zeroshot``.
- ``labels``: for ``zeroshot``, the candidate labels (defaults to the options).
- ``device``, ``dtype``, ``context_limit``, ``temperature``.
- ``instruction``: optional task instruction prefixed to the query.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

from ..backend import BackendError
from ..schema import DecisionRequest, Prediction, Usage, check_probabilities


class RerankerBackend:
    def __init__(self, config: dict):
        repo_id = config.get("repo_id")
        if not repo_id:
            raise BackendError("reranker requires a 'repo_id'")
        self.repo_id = repo_id
        self.revision_arg = config.get("revision")
        self.kind = config.get("kind", "reranker")
        self.device = config.get("device", "cpu")
        self.dtype = config.get("dtype", "float32")
        self.context_limit = int(config.get("context_limit", 8192))
        self.temperature = float(config.get("temperature", 1.0))
        self.instruction = config.get("instruction", "")
        self._usage = Usage()
        self._loaded = False
        self.model: Any = None
        self.tokenizer: Any = None
        self.pipeline: Any = None
        self.torch: Any = None

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            self.torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise BackendError("Install the transformers extra (torch + transformers)") from exc
        dtype = self.torch.bfloat16 if self.dtype == "bfloat16" else self.torch.float32
        if self.kind == "zeroshot":
            self.pipeline = transformers.pipeline(
                "zero-shot-classification",
                model=self.repo_id,
                revision=self.revision_arg,
                device=self.device,
                dtype=dtype,
            )
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.repo_id, revision=self.revision_arg
            )
            self.model = (
                transformers.AutoModelForSequenceClassification.from_pretrained(
                    self.repo_id, revision=self.revision_arg, dtype=dtype
                )
                .to(self.device)
                .eval()
            )
        self._loaded = True

    def predict(self, request: DecisionRequest) -> list[Prediction]:
        self._load()
        context = (
            request.state
            if isinstance(request.state, str)
            else json.dumps(request.state, ensure_ascii=False, sort_keys=True)
        )
        predictions: list[Prediction] = []
        for q in request.questions:
            if q.task == "noul":
                predictions.append(self._predict_noul(q, context))
            else:
                predictions.append(self._predict_rank(q, context))
            self._usage.requests += 1
        return predictions

    def _predict_rank(self, q, context: str) -> Prediction:
        options = _ordered(q)
        query = self._query(q, context)
        scores = [self._score(query, o.description) for o in options]
        probs = self._softmax(scores)
        result = {o.id: p for o, p in zip(options, probs, strict=True)}
        check_probabilities(result, [o.id for o in q.options])
        return Prediction(question_id=q.id, probabilities=result)

    def _predict_noul(self, q, context: str) -> Prediction:
        query = self._query(q, context)
        true_text = _desc(q, "true") or "Yes"
        false_text = _desc(q, "false") or "No"
        scores = [self._score(query, false_text), self._score(query, true_text)]
        probs = self._softmax(scores)
        result = {"false": probs[0], "true": probs[1]}
        check_probabilities(result, [o.id for o in q.options])
        return Prediction(question_id=q.id, probabilities=result)

    def _score(self, query: str, passage: str) -> float:
        if self.kind == "zeroshot":
            output = self.pipeline(query, candidate_labels=[passage], multi_label=False)
            return float(output["scores"][0])
        encoded = self.tokenizer(
            query, passage, return_tensors="pt", truncation=False
        )
        if encoded["input_ids"].shape[1] > self.context_limit:
            raise BackendError(
                f"Input requires {encoded['input_ids'].shape[1]} tokens, "
                f"exceeding {self.context_limit}"
            )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded).logits.float()
        if logits.shape[-1] == 1:
            return float(logits[0, 0])
        return float(logits[0, -1])

    def _query(self, q, context: str) -> str:
        prefix = f"{self.instruction}\n" if self.instruction else ""
        return f"{prefix}{context}\n\n{q.instructions}"

    def _softmax(self, scores: list[float]) -> list[float]:
        torch = self.torch
        tensor = torch.tensor(scores, dtype=torch.float32) / max(self.temperature, 1e-6)
        return [float(p) for p in torch.softmax(tensor, dim=0)]

    def revision(self) -> str | None:
        return self.revision_arg or self.repo_id

    def settings(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "kind": self.kind,
            "device": self.device,
            "transport": "cross-encoder-ranking",
        }

    def usage(self) -> Usage:
        return self._usage.model_copy()

    def close(self) -> None:
        self._loaded = False
        self.model = None
        self.pipeline = None


def _ordered(question):
    if question.task == "score":
        return sorted(question.options, key=lambda o: o.value or 0.0)
    return list(question.options)


def _desc(question, option_id: str) -> str | None:
    for option in question.options:
        if option.id == option_id:
            return option.description
    return None


def build(config: dict) -> RerankerBackend:
    return RerankerBackend(config)
