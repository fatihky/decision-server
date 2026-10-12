"""Local backend for GLiNER2 schema-extraction / decision checkpoints.

Fastino's ``gliner2`` family (``gliner2-large-v1``, the ``gliner2.5-*`` encoders
and the ``GLiNER2.5-Decide`` specialists) is not a Transformers architecture:
the checkpoints declare ``model_type: gliner2`` and an
``Gliner2ForSchemaExtraction`` architecture that ``transformers`` does not
register. Loading them through ``AutoModel`` fails with::

    The checkpoint you are trying to load has model type gliner2 but
    Transformers does not recognize this architecture.

They are instead driven by the ``gliner2`` package. Its ``AutoExtractor`` reads
the checkpoint's ``architecture`` field and dispatches to the right public
class, and ``classify_text`` scores a caller-supplied label set directly:
single-label tasks pick the argmax, while probabilities for every label are
recoverable from ``include_confidence=True``.

Each options list is scored in a single forward pass. For ``choice`` the
option descriptions are the labels; for ``noul`` the two constant labels are
``"false"`` and ``"true"``; for ``score`` the descriptions are scored and placed
on the ordinal scale by their declared ``value``.

Configuration keys:

- ``repo_id`` (required): Hub repo id of the GLiNER2 checkpoint.
- ``revision``: checkpoint revision, forwarded when the loader accepts it.
- ``device``: ``cpu``, ``cuda``, ``mps`` (default ``cpu``).
- ``threshold``: multi-label selection cutoff; probabilities are unaffected.
- ``default_label``: label text for options that declare no description.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..backend import BackendError
from ..schema import DecisionRequest, Prediction, Usage, check_probabilities

_FALSE_LABEL = "false"
_TRUE_LABEL = "true"


class Gliner2Backend:
    def __init__(self, config: dict):
        repo_id = config.get("repo_id")
        if not repo_id:
            raise BackendError("gliner2 requires a 'repo_id'")
        self.repo_id = repo_id
        self.revision_arg = config.get("revision")
        self.device = config.get("device", "cpu")
        self.threshold = float(config.get("threshold", 0.5))
        self.default_label = config.get("default_label", "option")
        self._usage = Usage()
        self.extractor: Any = None

    def _load(self) -> None:
        if self.extractor is not None:
            return
        try:
            gliner2 = importlib.import_module("gliner2")
        except ImportError as exc:
            raise BackendError(
                "Install the gliner2 extra (torch + transformers + gliner2)"
            ) from exc
        kwargs: dict[str, Any] = {"map_location": self.device}
        if self.revision_arg:
            kwargs["revision"] = self.revision_arg
        try:
            self.extractor = gliner2.AutoExtractor.from_pretrained(self.repo_id, **kwargs)
        except TypeError:
            kwargs.pop("revision", None)
            self.extractor = gliner2.AutoExtractor.from_pretrained(self.repo_id, **kwargs)
        except Exception as exc:  # noqa: BLE001 - report load failures clearly
            raise BackendError(f"Could not load {self.repo_id}: {exc}") from exc

    def predict(self, request: DecisionRequest) -> list[Prediction]:
        self._load()
        text = (
            request.state
            if isinstance(request.state, str)
            else _render_state(request.state)
        )
        predictions: list[Prediction] = []
        for q in request.questions:
            try:
                predictions.append(self._predict_question(q, text))
            except BackendError as exc:
                predictions.append(Prediction(question_id=q.id, error=str(exc)))
            self._usage.requests += 1
        return predictions

    def _predict_question(self, q, text: str) -> Prediction:
        if q.task == "noul":
            labels = [_FALSE_LABEL, _TRUE_LABEL]
            keys = {_FALSE_LABEL: "false", _TRUE_LABEL: "true"}
        elif q.task == "score":
            ordered = sorted(q.options, key=lambda o: o.value or 0.0)
            labels = [self._label(o) for o in ordered]
            keys = {self._label(o): o.id for o in ordered}
        else:
            labels = [self._label(o) for o in q.options]
            keys = {self._label(o): o.id for o in q.options}

        labels = _dedupe(labels)
        if not labels:
            raise BackendError("No labels available for this question")

        output = self.extractor.classify_text(
            text,
            {q.id: labels},
            threshold=self.threshold,
            include_confidence=True,
        )
        probabilities = _extract_probabilities(output, q.id, labels)
        result = {keys.get(label, label): prob for label, prob in probabilities.items()}
        check_probabilities(result, [o.id for o in q.options])
        return Prediction(question_id=q.id, probabilities=result)

    def _label(self, option) -> str:
        return option.description or f"{self.default_label} {option.id}"

    def _render_state(self, state: Any) -> str:
        return _render_state(state)

    def revision(self) -> str | None:
        return self.revision_arg or self.repo_id

    def settings(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "device": self.device,
            "threshold": self.threshold,
            "transport": "gliner2-schema-extraction",
        }

    def usage(self) -> Usage:
        return self._usage.model_copy()

    def close(self) -> None:
        self.extractor = None


def _render_state(state: Any) -> str:
    import json

    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def _dedupe(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            result.append(label)
    return result


def _extract_probabilities(output: Any, task: str, labels: list[str]) -> dict[str, float]:
    if not isinstance(output, dict) or task not in output:
        raise BackendError(f"No answer returned for question {task!r}")
    payload = output[task]
    if isinstance(payload, dict) and "probabilities" in payload:
        raw = {str(k): float(v) for k, v in payload["probabilities"].items()}
        raw = {label: raw.get(label, 0.0) for label in labels}
    elif isinstance(payload, dict) and "label" in payload:
        confidence = float(payload.get("confidence", 1.0))
        raw = {
            label: (confidence if label == payload["label"] else 0.0) for label in labels
        }
    elif isinstance(payload, list):
        scores: dict[str, float] = {}
        for item in payload:
            if isinstance(item, dict) and "label" in item:
                scores[str(item["label"])] = float(item.get("confidence", 0.0))
        raw = {label: scores.get(label, 0.0) for label in labels}
    else:
        raise BackendError(f"Unrecognized answer for question {task!r}: {payload!r}")
    return _normalize(raw)


def _normalize(raw: dict[str, float]) -> dict[str, float]:
    if not raw:
        return raw
    total = sum(raw.values())
    if total <= 0:
        uniform = 1.0 / len(raw)
        return {label: uniform for label in raw}
    return {label: value / total for label, value in raw.items()}


def build(config: dict) -> Gliner2Backend:
    return Gliner2Backend(config)
