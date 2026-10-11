"""Typed-question contract shared by every backend.

The server accepts arbitrary JSON state plus any number of typed questions.
Three question types exist:

- ``choice``: pick one option; the answer is a probability per option.
- ``noul``: a yes/no judgment; the answer is one probability for ``true``.
- ``score``: place a value on an ordered numeric scale.

Nothing here depends on any particular benchmark, dataset or model. A caller
sends whatever state and options it wants and receives probabilities.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Task = Literal["choice", "noul", "score"]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Option(_Record):
    """One alternative, judgment or numeric level."""

    id: str = Field(min_length=1)
    description: str
    description_json: str | None = None
    value: float | None = None


class Question(_Record):
    """One typed decision request."""

    id: str = Field(min_length=1)
    task: Task
    instructions: str = Field(min_length=1)
    instructions_json: str | None = None
    system_prompt: str = ""
    options: list[Option] = Field(min_length=2)

    @model_validator(mode="after")
    def _check(self) -> Question:
        ids = [o.id for o in self.options]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate option IDs")
        if self.task == "noul" and set(ids) != {"false", "true"}:
            raise ValueError("A noul question requires exactly the option IDs 'false' and 'true'")
        if self.task == "score":
            values = [o.value for o in self.options]
            if any(v is None for v in values):
                raise ValueError("Every score option needs a numeric value")
            if len(set(values)) < 2:
                raise ValueError("A score scale needs at least two distinct values")
        return self


class DecisionRequest(_Record):
    """One inference request: arbitrary state plus typed questions."""

    state: Any
    state_json: str | None = None
    questions: list[Question] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> DecisionRequest:
        ids = [q.id for q in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate question IDs")
        return self


class Prediction(_Record):
    """One answer: probabilities keyed by option ID, or an error string."""

    question_id: str
    probabilities: dict[str, float] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Prediction:
        if (self.probabilities is None) == (self.error is None):
            raise ValueError("A prediction carries exactly one of probabilities or error")
        return self


class Usage(_Record):
    requests: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class DecisionResponse(_Record):
    """One inference result for the whole request."""

    model: str
    backend: str
    revision: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    answers: list[Prediction]

    @model_validator(mode="after")
    def _check(self) -> DecisionResponse:
        ids = [a.question_id for a in self.answers]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate answer IDs")
        return self


class ModelDescription(_Record):
    """Public description of one servable model."""

    id: str
    label: str
    kind: Literal["local", "cloud"]
    backend: str
    tasks: list[Task]
    description: str = ""
    source: str | None = None
    revision: str | None = None
    requires: list[str] = Field(default_factory=list)
    available: bool = True
    unavailable_reason: str | None = None


def check_probabilities(
    probabilities: dict[str, float], ids: list[str], *, abs_tol: float = 1e-5
) -> None:
    """Validate that probabilities cover exactly ``ids`` and sum to one."""

    if set(probabilities) != set(ids):
        raise ValueError("Probability keys must match the option IDs exactly")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=abs_tol):
        raise ValueError("Probabilities must sum to one")
