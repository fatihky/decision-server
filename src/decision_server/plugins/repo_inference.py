"""Backend that uses the inference code a repository ships.

Many decision models publish a self-contained ``inference.py`` or ``model.py``
with an entry point such as ``DecisionModel.from_pretrained(...).decide(...)``
or a module-level ``predict``. This plugin downloads that code, builds the
model, and adapts its answers to the typed contract.

Configuration keys:

- ``repo_id`` (required): Hub repository containing the code and weights.
- ``revision``: checkpoint revision.
- ``code_file``: the file that holds the entry point (default ``inference.py``).
- ``code_files``: extra files the module imports (downloaded alongside).
- ``entry``: explicit entry point name, overriding auto-detection.
- ``model_class``: explicit ``DecisionModel``-style class name.
- ``device``: passed to the family constructor when it accepts one.
- ``init_kwargs``: extra keyword arguments for the constructor.

Answers are mapped by matching the family's returned question keys to ours. If
the family returns probabilities under different option keys, an ``option_map``
maps family keys to our option IDs.
"""

from __future__ import annotations

from typing import Any

from ..backend import BackendError
from ..schema import DecisionRequest, Prediction, Usage, check_probabilities
from .code_loader import find_entry, find_model_class, load_module_from, snapshot_code_files


class RepoInferenceBackend:
    def __init__(self, config: dict):
        repo_id = config.get("repo_id")
        if not repo_id:
            raise BackendError("repo-inference requires a 'repo_id'")
        self.repo_id = repo_id
        self.revision_arg = config.get("revision")
        self.code_file = config.get("code_file", "inference.py")
        self.code_files = list(config.get("code_files", [])) + [self.code_file]
        self.entry_name = config.get("entry")
        self.model_class_name = config.get("model_class")
        self.device = config.get("device")
        self.init_kwargs = dict(config.get("init_kwargs", {}))
        self.option_map: dict | None = config.get("option_map")
        self._usage = Usage()
        self.module: Any = None
        self.handler: Any = None
        self.model: Any = None

    def _load(self) -> None:
        if self.handler is not None or self.model is not None:
            return
        directory = snapshot_code_files(self.repo_id, self.revision_arg, self.code_files)
        module_name = f"decision_repo_{self.repo_id.replace('/', '_')}"
        self.module = load_module_from(directory / self.code_file, module_name)

        # Prefer a shipped model class with from_pretrained; else a function.
        try:
            cls = find_model_class(
                self.module, [self.model_class_name] if self.model_class_name else None
            )
        except BackendError:
            cls = None
        if cls is not None:
            kwargs: dict[str, Any] = {"revision": self.revision_arg, **self.init_kwargs}
            if self.device:
                kwargs.setdefault("device", self.device)
            builder: Any = cls
            try:
                self.model = builder.from_pretrained(self.repo_id, **kwargs)
            except TypeError:
                kwargs.pop("revision", None)
                self.model = builder.from_pretrained(self.repo_id, **kwargs)
            self.handler = self._call_model
        else:
            names = [self.entry_name] if self.entry_name else None
            self.handler = find_entry(self.module, names)

    def _call_model(self, state: Any, questions: dict) -> Any:
        assert self.model is not None
        for name in (self.entry_name, "decide", "system_one", "predict", "infer"):
            if name and hasattr(self.model, name):
                return getattr(self.model, name)(state, questions)
        if callable(self.model):
            return self.model(state, questions)
        raise BackendError("Loaded model exposes no usable entry point")

    def predict(self, request: DecisionRequest) -> list[Prediction]:
        self._load()
        assert self.handler is not None
        # Anonymous IDs: the family never sees option IDs unless it must.
        out = self.handler(request.state, self._questions(request))
        return self._decode(request, out)

    def _questions(self, request: DecisionRequest) -> dict:
        result: dict = {}
        for q in request.questions:
            if q.task == "noul":
                criteria: Any = {"true": _desc(q, "true"), "false": _desc(q, "false")}
            elif q.task == "score":
                ordered = sorted(q.options, key=lambda o: o.value or 0.0)
                criteria = [o.description for o in ordered]
            else:
                criteria = {o.id: o.description for o in q.options}
            result[q.id] = {
                "type": q.task,
                "instructions": q.instructions,
                "criteria": criteria,
            }
        return result

    def _decode(self, request: DecisionRequest, out: Any) -> list[Prediction]:
        answers = _extract_answers(out)
        predictions: list[Prediction] = []
        for q in request.questions:
            if q.id not in answers:
                predictions.append(Prediction(question_id=q.id, error="No answer returned"))
                continue
            prob = _answer_to_probabilities(answers[q.id], q, self.option_map)
            if prob is None:
                predictions.append(
                    Prediction(question_id=q.id, error=f"Unrecognized answer: {answers[q.id]!r}")
                )
                continue
            try:
                check_probabilities(prob, [o.id for o in q.options])
            except ValueError as exc:
                predictions.append(Prediction(question_id=q.id, error=str(exc)))
                continue
            predictions.append(Prediction(question_id=q.id, probabilities=prob))
        self._usage.requests += 1
        return predictions

    def revision(self) -> str | None:
        return self.revision_arg or self.repo_id

    def settings(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "code_file": self.code_file,
            "transport": "repo-provided-inference",
        }

    def usage(self) -> Usage:
        return self._usage.model_copy()

    def close(self) -> None:
        self.model = None
        self.module = None
        self.handler = None


def _desc(question, option_id: str) -> str | None:
    for option in question.options:
        if option.id == option_id:
            return option.description
    return None


def _extract_answers(out: Any) -> dict:
    if out is None:
        raise BackendError("Inference returned nothing")
    if isinstance(out, dict):
        if "answers" in out and isinstance(out["answers"], dict):
            return out["answers"]
        return out
    if hasattr(out, "answers"):
        return dict(out.answers)
    raise BackendError(f"Unrecognized inference output type: {type(out).__name__}")


def _answer_to_probabilities(answer: Any, question, option_map: dict | None) -> dict | None:
    """Map one family answer to probabilities keyed by our option IDs."""

    ids = [o.id for o in question.options]
    if isinstance(answer, dict):
        if isinstance(answer.get("probabilities"), dict):
            raw = {str(k): float(v) for k, v in answer["probabilities"].items()}
            return _rekey(raw, question, option_map)
        if question.task == "noul" and "noul" in answer:
            value = float(answer["noul"])
            return {"false": 1 - value, "true": value}
        if question.task == "score":
            if isinstance(answer.get("probabilities"), dict):
                return _rekey(answer["probabilities"], question, option_map)
            if "score" in answer:
                index = max(0, min(len(ids) - 1, round(float(answer["score"]))))
                ordered = sorted(question.options, key=lambda o: o.value or 0.0)
                return {o.id: (1.0 if i == index else 0.0) for i, o in enumerate(ordered)}
        if "choice" in answer:
            chosen = str(answer["choice"])
            return {oid: (1.0 if oid == chosen else 0.0) for oid in ids}
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        if question.task == "noul":
            value = float(answer)
            return {"false": 1 - value, "true": value}
    return None


def _rekey(raw: dict, question, option_map: dict | None) -> dict:
    raw = {str(k): float(v) for k, v in raw.items()}
    if option_map:
        raw = {option_map.get(k, k): v for k, v in raw.items()}
    # Score levels may come back as "0","1",... in ascending order.
    level_ids = {str(i) for i in range(len(question.options))}
    if set(raw) == level_ids and question.task == "score":
        ordered = sorted(question.options, key=lambda o: o.value or 0.0)
        return {o.id: raw[str(i)] for i, o in enumerate(ordered)}
    # Anonymous choice keys.
    anon = {f"option_{i}" for i in range(len(question.options))}
    if set(raw) == anon and question.task == "choice":
        return {o.id: raw[f"option_{i}"] for i, o in enumerate(question.options)}
    total = sum(raw.values())
    if total > 0 and abs(total - 1) > 1e-3:
        raw = {k: v / total for k, v in raw.items()}
    return raw


def build(config: dict) -> RepoInferenceBackend:
    return RepoInferenceBackend(config)
