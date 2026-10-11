"""Schema validation tests for the typed-question contract."""

from __future__ import annotations

import pytest

from decision_server.schema import DecisionRequest, Option, Question, check_probabilities


def test_choice_accepts_arbitrary_state():
    request = DecisionRequest.model_validate(
        {
            "state": {"nested": [1, 2, 3], "text": "hi"},
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
    assert request.questions[0].task == "choice"


def test_noul_requires_false_true_ids():
    with pytest.raises(ValueError):
        Question(
            id="q",
            task="noul",
            instructions="judge",
            options=[Option(id="no", description="No"), Option(id="yes", description="Yes")],
        )


def test_score_requires_numeric_values():
    with pytest.raises(ValueError):
        Question(
            id="q",
            task="score",
            instructions="rate",
            options=[Option(id="a", description="A"), Option(id="b", description="B")],
        )


def test_score_requires_two_distinct_values():
    with pytest.raises(ValueError):
        Question(
            id="q",
            task="score",
            instructions="rate",
            options=[
                Option(id="a", description="A", value=1),
                Option(id="b", description="B", value=1),
            ],
        )


def test_duplicate_question_ids_rejected():
    question = {
        "id": "q",
        "task": "choice",
        "instructions": "pick",
        "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
    }
    with pytest.raises(ValueError):
        DecisionRequest.model_validate({"state": {}, "questions": [question, question]})


def test_check_probabilities_rejects_bad_sum():
    with pytest.raises(ValueError):
        check_probabilities({"a": 0.5, "b": 0.6}, ["a", "b"])


def test_check_probabilities_rejects_wrong_keys():
    with pytest.raises(ValueError):
        check_probabilities({"a": 1.0}, ["a", "b"])
