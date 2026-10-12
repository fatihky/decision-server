"""Tests for the family plugins (pure decode logic; no downloads)."""

from __future__ import annotations

import pytest

from decision_server.plugins.repo_inference import (
    RepoInferenceBackend,
    _answer_to_probabilities,
    _extract_answers,
    _rekey,
)
from decision_server.schema import Option, Question


def _choice_question() -> Question:
    return Question(
        id="q",
        task="choice",
        instructions="pick",
        options=[Option(id="a", description="A"), Option(id="b", description="B")],
    )


def _score_question() -> Question:
    return Question(
        id="q",
        task="score",
        instructions="rate",
        options=[
            Option(id="low", description="Low", value=1),
            Option(id="mid", description="Mid", value=2),
            Option(id="high", description="High", value=3),
        ],
    )


def _noul_question() -> Question:
    return Question(
        id="q",
        task="noul",
        instructions="judge",
        options=[Option(id="false", description="No"), Option(id="true", description="Yes")],
    )


def test_extract_answers_shapes():
    assert _extract_answers({"answers": {"q": {}}}) == {"q": {}}
    assert _extract_answers({"q": {}}) == {"q": {}}


class _Container:
    answers = {"q": {"choice": "a"}}


def test_extract_answers_object():
    assert _extract_answers(_Container()) == {"q": {"choice": "a"}}


def test_choice_from_chosen_field():
    prob = _answer_to_probabilities({"choice": "a"}, _choice_question(), None)
    assert prob == {"a": 1.0, "b": 0.0}


def test_noul_from_value():
    prob = _answer_to_probabilities({"noul": 0.8}, _noul_question(), None)
    assert prob == {"false": pytest.approx(0.2), "true": pytest.approx(0.8)}


def test_score_from_index():
    prob = _answer_to_probabilities({"score": 2}, _score_question(), None)
    assert prob == {"low": 0.0, "mid": 0.0, "high": 1.0}


def test_rekey_anonymous_choice():
    raw = {"option_0": 0.3, "option_1": 0.7}
    assert _rekey(raw, _choice_question(), None) == {"a": 0.3, "b": 0.7}


def test_rekey_score_levels():
    raw = {"0": 0.2, "1": 0.3, "2": 0.5}
    assert _rekey(raw, _score_question(), None) == {"low": 0.2, "mid": 0.3, "high": 0.5}


def test_rekey_option_map():
    raw = {"yes": 0.9, "no": 0.1}
    mapped = _rekey(raw, _noul_question(), {"yes": "true", "no": "false"})
    assert mapped == {"true": 0.9, "false": 0.1}


def test_rekey_normalizes_unnormalized():
    raw = {"a": 2.0, "b": 2.0}
    out = _rekey(raw, _choice_question(), None)
    assert out == {"a": 0.5, "b": 0.5}


def test_backend_requires_repo_id():
    from decision_server.backend import BackendError

    with pytest.raises(BackendError):
        RepoInferenceBackend({})


def test_backend_decodes_with_stubbed_handler():
    backend = RepoInferenceBackend({"repo_id": "owner/repo"})
    from decision_server.schema import DecisionRequest

    request = DecisionRequest.model_validate(
        {
            "state": {"x": 1},
            "questions": [
                {
                    "id": "q",
                    "task": "choice",
                    "instructions": "pick",
                    "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
                }
            ],
        }
    )

    def handler(state, questions):
        return {"answers": {"q": {"probabilities": {"a": 0.9, "b": 0.1}}}}

    backend.handler = handler
    predictions = backend.predict(request)
    assert predictions[0].probabilities == {"a": 0.9, "b": 0.1}


def test_backend_reports_missing_answer():
    backend = RepoInferenceBackend({"repo_id": "owner/repo"})
    from decision_server.schema import DecisionRequest

    request = DecisionRequest.model_validate(
        {
            "state": "s",
            "questions": [
                {
                    "id": "q",
                    "task": "noul",
                    "instructions": "judge",
                    "options": [
                        {"id": "false", "description": "No"},
                        {"id": "true", "description": "Yes"},
                    ],
                }
            ],
        }
    )

    backend.handler = lambda state, questions: {"answers": {}}
    predictions = backend.predict(request)
    assert predictions[0].error == "No answer returned"


def test_code_loader_finds_entry(tmp_path):
    from decision_server.plugins.code_loader import find_entry, load_module_from

    module_file = tmp_path / "inference.py"
    module_file.write_text("def decide(state, questions):\n    return {'answers': {}}\n")
    module = load_module_from(module_file, "candidate_inference")
    entry = find_entry(module)
    assert entry("s", {}) == {"answers": {}}


def test_code_loader_finds_model_class(tmp_path):
    from decision_server.plugins.code_loader import find_model_class, load_module_from

    module_file = tmp_path / "model.py"
    module_file.write_text(
        "class DecisionModel:\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, repo):\n"
        "        return cls()\n"
    )
    module = load_module_from(module_file, "candidate_model")
    cls = find_model_class(module)
    assert cls.__name__ == "DecisionModel"


def test_code_loader_missing_entry(tmp_path):
    from decision_server.backend import BackendError
    from decision_server.plugins.code_loader import find_entry, load_module_from

    module_file = tmp_path / "empty.py"
    module_file.write_text("X = 1\n")
    module = load_module_from(module_file, "candidate_empty")
    with pytest.raises(BackendError):
        find_entry(module)


def test_reranker_requires_repo():
    from decision_server.backend import BackendError
    from decision_server.plugins.reranker import RerankerBackend

    with pytest.raises(BackendError):
        RerankerBackend({})


def test_peft_requires_repo():
    from decision_server.backend import BackendError
    from decision_server.plugins.peft_decision import PeftDecisionBackend

    with pytest.raises(BackendError):
        PeftDecisionBackend({})


def test_gliner2_requires_repo():
    from decision_server.backend import BackendError
    from decision_server.plugins.gliner2 import Gliner2Backend

    with pytest.raises(BackendError):
        Gliner2Backend({})


class _StubExtractor:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def classify_text(self, text, tasks, **kwargs):
        self.calls.append((text, tasks, kwargs))
        return {name: self.output for name in tasks}


def _gliner2_backend(output):
    from decision_server.plugins.gliner2 import Gliner2Backend

    backend = Gliner2Backend({"repo_id": "fastino/gliner2.5-small-v1"})
    backend.extractor = _StubExtractor(output)
    return backend


def _request(state, task, options):
    from decision_server.schema import DecisionRequest

    return DecisionRequest.model_validate(
        {
            "state": state,
            "questions": [
                {"id": "q", "task": task, "instructions": "pick", "options": options}
            ],
        }
    )


def test_gliner2_choice_from_probabilities():
    payload = {"value": "A", "confidence": 0.9, "probabilities": {"A": 0.9, "B": 0.1}}
    backend = _gliner2_backend(payload)
    request = _request(
        "state",
        "choice",
        [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
    )
    prediction = backend.predict(request)[0]
    assert prediction.probabilities == {"a": 0.9, "b": pytest.approx(0.1)}
    _, tasks, kwargs = backend.extractor.calls[0]
    assert tasks == {"q": ["A", "B"]}
    assert kwargs["include_confidence"] is True


def test_gliner2_noul_uses_constant_labels():
    payload = {"value": "true", "confidence": 0.7, "probabilities": {"false": 0.3, "true": 0.7}}
    backend = _gliner2_backend(payload)
    request = _request(
        "state",
        "noul",
        [{"id": "false", "description": "No"}, {"id": "true", "description": "Yes"}],
    )
    prediction = backend.predict(request)[0]
    assert prediction.probabilities == {"false": pytest.approx(0.3), "true": pytest.approx(0.7)}
    _, tasks, _ = backend.extractor.calls[0]
    assert tasks == {"q": ["false", "true"]}


def test_gliner2_score_maps_ordered_labels():
    payload = {
        "value": "High",
        "confidence": 0.6,
        "probabilities": {"Low": 0.1, "Mid": 0.3, "High": 0.6},
    }
    backend = _gliner2_backend(payload)
    request = _request(
        "state",
        "score",
        [
            {"id": "high", "description": "High", "value": 3},
            {"id": "low", "description": "Low", "value": 1},
            {"id": "mid", "description": "Mid", "value": 2},
        ],
    )
    prediction = backend.predict(request)[0]
    assert prediction.probabilities == {"low": 0.1, "mid": 0.3, "high": pytest.approx(0.6)}


def test_gliner2_reports_unrecognized_answer():
    backend = _gliner2_backend("not-a-dict")
    request = _request(
        "state",
        "choice",
        [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
    )
    assert backend.predict(request)[0].error is not None

