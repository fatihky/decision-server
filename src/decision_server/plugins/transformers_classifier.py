"""Local backend that scores options with a Hugging Face classifiers.

Works for any ``AutoModelForSequenceClassification`` checkpoint whose label
set can be reduced to a single positive score (for example an NLI model where
the entailment logit means "this option fits"). Each option is scored
independently against the question, and the scores are softmaxed into a
probability distribution over the declared option IDs.

Configuration keys (all optional except ``model``):

- ``model``: Hub repo id or local path of the classifier.
- ``device``: ``cpu``, ``cuda:0``, ... (default ``cpu``).
- ``dtype``: ``float32`` or ``bfloat16`` (default ``float32``).
- ``positive_label``: index of the class used as the score (default: last).
- ``context_limit``: maximum tokens; longer inputs are rejected, not truncated.
- ``revision``: checkpoint revision.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..backend import BackendError
from ..schema import DecisionRequest, Prediction, Usage, check_probabilities


def _render(state: object, question) -> str:
    import json

    parts = []
    if question.system_prompt:
        parts.append(question.system_prompt)
    parts.append(question.instructions)
    parts.append("State: " + json.dumps(state, ensure_ascii=False, sort_keys=True))
    return "\n\n".join(parts)


class TransformersClassifierBackend:
    def __init__(self, config: dict):
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise BackendError("Install the transformers extra (torch + transformers)") from exc
        self.torch = torch
        model = config.get("model")
        if not model:
            raise BackendError("transformers-classifier requires a 'model' config value")
        self.model_id = model
        self.model: Any = None
        self.tokenizer: Any = None
        self.device = config.get("device", "cpu")
        dtype_name = config.get("dtype", "float32")
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
        self.context_limit = int(config.get("context_limit", 512))
        self.model_revision = config.get("revision")
        self.positive_label = config.get("positive_label")
        self._usage = Usage()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model, revision=self.model_revision
        )
        self.model = (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                model, revision=self.model_revision, dtype=dtype
            )
            .to(self.device)
            .eval()
        )

    def predict(self, request: DecisionRequest) -> list[Prediction]:
        answers: list[Prediction] = []
        for q in request.questions:
            context = _render(request.state, q)
            pairs = [(context, o.description) for o in q.options]
            encoded = self.tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                return_tensors="pt",
                truncation=False,
                padding=True,
            )
            lengths = encoded["attention_mask"].sum(dim=1)
            if int(lengths.max()) > self.context_limit:
                raise BackendError(
                    f"Input requires {int(lengths.max())} tokens, exceeding {self.context_limit}"
                )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with self.torch.no_grad():
                logits = self.model(**encoded).logits.float()
            index = self.positive_label if self.positive_label is not None else logits.shape[-1] - 1
            scores = logits[:, index]
            probabilities = self.torch.softmax(scores, dim=0).tolist()
            result = {o.id: float(p) for o, p in zip(q.options, probabilities, strict=True)}
            check_probabilities(result, [o.id for o in q.options])
            answers.append(Prediction(question_id=q.id, probabilities=result))
            with_usage = encoded["input_ids"].numel()
            self._usage.input_tokens += int(with_usage)
            self._usage.requests += 1
        return answers

    def revision(self) -> str | None:
        return self.model_revision or self.model_id

    def settings(self) -> dict:
        return {
            "model": self.model_id,
            "device": self.device,
            "context_limit": self.context_limit,
            "strategy": "per-option-classification-softmax",
            "input_length_policy": "reject-overflow-no-truncation",
        }

    def usage(self) -> Usage:
        return self._usage.model_copy()

    def close(self) -> None:
        del self.model


def build(config: dict) -> TransformersClassifierBackend:
    return TransformersClassifierBackend(config)
