"""Service layer: run a request against a named model and shape the response."""

from __future__ import annotations

from .registry import Registry
from .schema import DecisionRequest, DecisionResponse, Prediction, Usage


def describe_models(registry: Registry):
    return registry.descriptions()


def run(registry: Registry, model_id: str, request: DecisionRequest) -> DecisionResponse:
    """Execute ``request`` on ``model_id`` and return a typed response.

    A backend that raises on the whole request yields per-question error
    predictions so the caller sees which questions failed, rather than a
    blanket failure that hides partial success.
    """

    backend = registry.get(model_id)
    entry = next(d for d in registry.descriptions() if d.id == model_id)
    supported = set(entry.tasks)
    unsupported = [q for q in request.questions if q.task not in supported]
    if unsupported:
        tasks = ", ".join(sorted({q.task for q in unsupported}))
        raise ValueError(f"Model {model_id!r} does not support task(s): {tasks}")

    try:
        predictions = backend.predict(request)
    except Exception as exc:  # noqa: BLE001 - present the failure per question
        message = str(exc) or exc.__class__.__name__
        predictions = [Prediction(question_id=q.id, error=message) for q in request.questions]

    by_id = {p.question_id: p for p in predictions}
    ordered = [
        by_id.get(q.id, Prediction(question_id=q.id, error="No prediction returned"))
        for q in request.questions
    ]
    return DecisionResponse(
        model=model_id,
        backend=entry.backend,
        revision=backend.revision(),
        settings=backend.settings(),
        usage=backend.usage() or Usage(),
        answers=ordered,
    )
