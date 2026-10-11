"""Cloud backends: forward typed requests to remote HTTP endpoints.

These models are hosted elsewhere. The server proxies the request, adding a
bearer token read from the environment. No weights are loaded locally.

The wire format is the shared typed-question contract::

    POST <base_url><endpoint>
    {"model": <id>, "state": <any>, "questions": {"<qid>": {...}}}
    -> {"model": ..., "usage": {...}, "answers": {"<qid>": {...}}}

Response answers are mapped back to the request's question IDs. Numeric
probabilities that only sum to one within a decimal rounding tolerance are
renormalized, matching servers that return rounded values.
"""

from __future__ import annotations

import json
import os
import time
from threading import Lock

import httpx

from .backend import BackendError
from .schema import DecisionRequest, Option, Prediction, Question, Usage, check_probabilities


def _questions_for_api(
    questions: list[Question], *, sort_score: bool, anonymous_choice: bool
) -> dict:
    """Encode questions into the transport shape."""

    result: dict = {}
    for q in questions:
        structured = json.loads(q.instructions_json) if q.instructions_json else q.instructions
        instructions = structured
        if q.system_prompt:
            instructions = {"system": q.system_prompt, "instruction": structured}

        def description(option: Option) -> object:
            if option.description_json is not None:
                return json.loads(option.description_json)
            return option.description

        if sort_score and q.task == "score":
            options = sorted(q.options, key=lambda o: o.value or 0.0)
        else:
            options = q.options
        criteria = (
            [description(o) for o in options]
            if q.task == "score"
            else {o.id: description(o) for o in options}
        )
        if anonymous_choice and q.task == "choice":
            criteria = {f"option_{i}": description(o) for i, o in enumerate(options)}
        result[q.id] = {"type": q.task, "instructions": instructions, "criteria": criteria}
    return result


class CloudBackend:
    """Proxy a typed request to a remote typed-question endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        endpoint: str,
        api_key_env: str,
        rounding_decimals: int = 4,
        timeout: float = 120,
        retries: int = 3,
    ):
        key = os.environ.get(api_key_env)
        if not key:
            raise BackendError(f"Set {api_key_env} to use this cloud model")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.rounding_decimals = rounding_decimals
        self.retries = retries
        self._lock = Lock()
        self._resolved: str | None = None
        self._usage = Usage()
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}"},
        )

    def predict(self, request: DecisionRequest) -> list[Prediction]:
        body = {
            "model": self.model,
            "state": request.state,
            "questions": _questions_for_api(
                request.questions, sort_score=True, anonymous_choice=True
            ),
        }
        last: httpx.Response | None = None
        for attempt in range(self.retries):
            with self._lock:
                self._usage.requests += 1
            try:
                response = self.client.post(self.endpoint, json=body)
            except httpx.TransportError as exc:  # bound retries on transient transport errors
                if attempt == self.retries - 1:
                    raise BackendError("Cloud transport failure") from exc
                time.sleep(2**attempt)
                continue
            last = response
            if response.status_code not in {429, 500, 502, 503, 504, 529}:
                break
            time.sleep(2**attempt)
        if last is None or last.is_error:
            status = last.status_code if last is not None else "unknown"
            raise BackendError(f"Cloud HTTP {status}")
        payload = last.json()
        resolved = payload.get("model")
        with self._lock:
            if self._resolved is not None and resolved != self._resolved:
                raise BackendError("Resolved model changed during the run")
            self._resolved = resolved
            usage = payload.get("usage") or {}
            self._usage.input_tokens += int(usage.get("input_tokens", 0))
            self._usage.output_tokens += int(usage.get("output_tokens", 0))
        return self._decode(request, payload["answers"])

    def _decode(self, request: DecisionRequest, answers: dict) -> list[Prediction]:
        if set(answers) != {q.id for q in request.questions}:
            raise BackendError("Returned question IDs differ from the request")
        results: list[Prediction] = []
        for q in request.questions:
            answer = answers[q.id]
            if answer.get("type") != q.task:
                raise BackendError("Returned task differs from the request")
            if q.task == "noul":
                value = float(answer["noul"])
                probabilities = {"false": 1 - value, "true": value}
            elif q.task == "score":
                raw = answer["probabilities"]
                if set(raw) != {str(i) for i in range(len(q.options))}:
                    raise BackendError("Score level IDs do not match")
                ordered = sorted(q.options, key=lambda o: o.value or 0.0)
                probabilities = {o.id: float(raw[str(i)]) for i, o in enumerate(ordered)}
            else:
                raw = {k: float(v) for k, v in answer["probabilities"].items()}
                if set(raw) == {f"option_{i}" for i in range(len(q.options))}:
                    raw = {o.id: raw[f"option_{i}"] for i, o in enumerate(q.options)}
                probabilities = raw
            probabilities = self._normalize(probabilities)
            check_probabilities(probabilities, [o.id for o in q.options])
            results.append(Prediction(question_id=q.id, probabilities=probabilities))
        return results

    def _normalize(self, probabilities: dict[str, float]) -> dict[str, float]:
        total = sum(probabilities.values())
        tolerance = 0.5 * 10 ** (-self.rounding_decimals) + 1e-6
        if total <= 0 or abs(total - 1) > len(probabilities) * tolerance:
            raise BackendError("Probability sum exceeds the allowed rounding error")
        if any(not 0 <= p <= 1 for p in probabilities.values()):
            raise BackendError("Invalid probability")
        return {k: v / total for k, v in probabilities.items()}

    def revision(self) -> str | None:
        return self._resolved or self.model

    def settings(self) -> dict:
        return {
            "endpoint": f"{self.base_url}{self.endpoint}",
            "transport": "http-typed-questions",
            "probability_normalization": f"sum-normalized-within-{self.rounding_decimals}-decimals",
        }

    def usage(self) -> Usage:
        with self._lock:
            return self._usage.model_copy()

    def close(self) -> None:
        self.client.close()
