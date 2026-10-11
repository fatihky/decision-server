"""Backend interface implemented by every servable model.

A backend receives a validated :class:`DecisionRequest` and returns one
:class:`Prediction` per question. Two concrete families exist:

- ``cloud``: forwards the request to a remote HTTP endpoint using a bearer
  token from the environment.
- ``local``: loads a checkpoint (or connects to a caller-managed server) and
  runs inference in-process.

Backends never see targets, ground truth or provenance; only the request.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schema import DecisionRequest, Prediction, Usage


class BackendError(RuntimeError):
    """Raised when a backend cannot answer a request."""


@runtime_checkable
class Backend(Protocol):
    """Minimal contract every model backend satisfies."""

    def predict(self, request: DecisionRequest) -> list[Prediction]:
        """Return one prediction per question, in request order."""
        ...

    def revision(self) -> str | None:
        """Return the resolved model revision, if known."""
        ...

    def settings(self) -> dict:
        """Return effective runtime settings recorded alongside the answer."""
        ...

    def usage(self) -> Usage:
        """Return accumulated token/request usage, if tracked."""
        ...

    def close(self) -> None:
        """Release any resources owned by the backend."""
        ...
