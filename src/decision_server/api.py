"""JSON HTTP API exposing the same typed contract as the UI.

Routes:

- ``GET  /v1/models``            list servable models and availability
- ``GET  /v1/models/{id}``       one model's description
- ``POST /v1/systemone``         run typed questions against a model
- ``POST /v1/models``            register a model from the Hub by repo id
- ``GET  /v1/hub/search``        search the Hub ``decision-model`` tag
- ``POST /v1/hub/describe``      inspect one Hub repository
- ``GET  /health``               liveness

The request and response bodies are the same Pydantic models used everywhere
else, so a client that talks to this API can talk to any backend.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .registry import Registry
from .schema import (
    DecisionRequest,
    DecisionResponse,
    ModelDescription,
    Option,
    Prediction,
    Question,
)
from .service import run


class RunRequest(DecisionRequest):
    """A decision request addressed to one model."""

    model: str = Field(min_length=1)


class AddModelRequest(BaseModel):
    repo_id: str = Field(min_length=3)


class HubModel(BaseModel):
    id: str
    servable_via: str
    downloads: int
    likes: int
    is_gguf: bool


def create_api(registry: Registry) -> FastAPI:
    app = FastAPI(
        title="decision-server",
        version="0.1.0",
        description="Serve SystemOne-compatible decision models behind one typed contract.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models", response_model=list[ModelDescription])
    def list_models() -> list[ModelDescription]:
        return registry.descriptions()

    @app.get("/v1/models/{model_id}", response_model=ModelDescription)
    def get_model(model_id: str) -> ModelDescription:
        for d in registry.descriptions():
            if d.id == model_id:
                return d
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")

    @app.post("/v1/models", response_model=ModelDescription)
    def add_model(body: AddModelRequest) -> ModelDescription:
        try:
            return registry.add_hub_model(body.repo_id)
        except Exception as exc:  # noqa: BLE001 - surface the reason to the client
            raise HTTPException(status_code=422, detail=str(exc) or exc.__class__.__name__) from exc

    @app.post("/v1/systemone", response_model=DecisionResponse)
    def systemone(body: RunRequest) -> DecisionResponse:
        from .backend import BackendError

        request = DecisionRequest(
            state=body.state, state_json=body.state_json, questions=body.questions
        )
        try:
            return run(registry, body.model, request)
        except BackendError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/hub/search", response_model=list[HubModel])
    def hub_search(
        query: str | None = None,
        gguf_only: bool = False,
        limit: int = Query(default=30, ge=1, le=200),
    ) -> list[HubModel]:
        from .hub import search_models

        try:
            results = search_models(query=query, gguf_only=gguf_only, limit=limit)
        except Exception as exc:  # noqa: BLE001 - surface Hub errors
            raise HTTPException(status_code=502, detail=str(exc) or "Hub search failed") from exc
        return [
            HubModel(
                id=m.id,
                servable_via=m.servable_via,
                downloads=m.downloads,
                likes=m.likes,
                is_gguf=m.is_gguf,
            )
            for m in results
        ]

    @app.post("/v1/hub/describe")
    def hub_describe(body: AddModelRequest) -> dict:
        from .hub import describe

        try:
            return describe(body.repo_id)
        except Exception as exc:  # noqa: BLE001 - surface Hub errors
            raise HTTPException(status_code=502, detail=str(exc) or "describe failed") from exc

    return app


__all__ = [
    "AddModelRequest",
    "HubModel",
    "Option",
    "Prediction",
    "Question",
    "RunRequest",
    "create_api",
]
