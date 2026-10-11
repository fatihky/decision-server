"""Bundled example requests shown in the UI.

Each sample is a ready-to-run :class:`DecisionRequest` covering one of the
three task types, plus a combined request. They use only generic content so
they work with any backend.
"""

from __future__ import annotations

import json

from .schema import DecisionRequest

_SAMPLES: dict[str, dict] = {
    "choice": {
        "state": {
            "message": "I've been feeling exhausted for three weeks and can't concentrate.",
        },
        "questions": [
            {
                "id": "urgency",
                "task": "choice",
                "instructions": "Choose the most appropriate handling for this message.",
                "options": [
                    {"id": "routine", "description": "Handle through the normal queue."},
                    {"id": "priority", "description": "Escalate to a priority review."},
                    {"id": "urgent", "description": "Treat as an urgent safety concern."},
                ],
            }
        ],
    },
    "noul": {
        "state": {"sentence": "The library is closed on public holidays."},
        "questions": [
            {
                "id": "entailment",
                "task": "noul",
                "instructions": "Is the statement supported by common sense reasoning?",
                "options": [
                    {"id": "false", "description": "No"},
                    {"id": "true", "description": "Yes"},
                ],
            }
        ],
    },
    "score": {
        "state": {"review": "Battery life is fine but the screen scratches easily."},
        "questions": [
            {
                "id": "rating",
                "task": "score",
                "instructions": "Assign a numeric quality score from the criteria below.",
                "options": [
                    {"id": "poor", "description": "Poor", "value": 1},
                    {"id": "fair", "description": "Fair", "value": 2},
                    {"id": "good", "description": "Good", "value": 3},
                    {"id": "excellent", "description": "Excellent", "value": 4},
                ],
            }
        ],
    },
    "mixed": {
        "state": {"text": "Vaccines are effective at preventing severe illness."},
        "questions": [
            {
                "id": "agreement",
                "task": "choice",
                "instructions": "Pick the strongest stance this text expresses.",
                "options": [
                    {"id": "support", "description": "Supports the claim."},
                    {"id": "neutral", "description": "Neither supports nor opposes."},
                    {"id": "oppose", "description": "Opposes the claim."},
                ],
            },
            {
                "id": "toxicity",
                "task": "noul",
                "instructions": "Does this text contain harassment?",
                "options": [
                    {"id": "false", "description": "No"},
                    {"id": "true", "description": "Yes"},
                ],
            },
            {
                "id": "confidence",
                "task": "score",
                "instructions": "Rate how confident the text sounds.",
                "options": [
                    {"id": "low", "description": "Low", "value": 1},
                    {"id": "mid", "description": "Medium", "value": 2},
                    {"id": "high", "description": "High", "value": 3},
                ],
            },
        ],
    },
}


def sample_names() -> list[str]:
    return list(_SAMPLES)


def sample_request(name: str) -> DecisionRequest:
    if name not in _SAMPLES:
        raise KeyError(name)
    return DecisionRequest.model_validate(_SAMPLES[name])


def sample_json(name: str) -> str:
    return json.dumps(_SAMPLES[name], indent=2, ensure_ascii=False)
