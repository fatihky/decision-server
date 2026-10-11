"""Uniform-probability backend for offline checks.

Returns equal probability over every option. This is a pipeline demo, never a
model measurement.
"""

from __future__ import annotations

from ..schema import DecisionRequest, Prediction, Usage


class UniformBackend:
    def predict(self, request: DecisionRequest) -> list[Prediction]:
        answers = []
        for q in request.questions:
            share = 1 / len(q.options)
            answers.append(
                Prediction(
                    question_id=q.id,
                    probabilities={o.id: share for o in q.options},
                )
            )
        return answers

    def revision(self) -> str | None:
        return "1"

    def settings(self) -> dict:
        return {"strategy": "uniform"}

    def usage(self) -> Usage:
        return Usage()

    def close(self) -> None:
        pass


def build(config: dict) -> UniformBackend:
    return UniformBackend()
