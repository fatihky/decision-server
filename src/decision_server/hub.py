"""Discover SystemOne/decision models on the Hugging Face Hub.

The Hub tags decision models with ``decision-model``. GGUF decision models are
tagged automatically from their llama.cpp metadata (``<arch>.decision.type``)
and are servable with ``llama-server``. Transformer decision models ship either
custom code (``auto_map``) or a standard classification head.

This module only reads Hub metadata; downloading happens in the backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DECISION_TAG = "decision-model"


@dataclass
class HubModel:
    """One decision model discovered on the Hub."""

    id: str
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    library: str | None = None
    is_gguf: bool = False
    gguf_files: list[str] = field(default_factory=list)
    revision: str | None = None

    @property
    def servable_via(self) -> str:
        """How this model can be served: ``gguf`` or ``transformers``."""

        return "gguf" if self.is_gguf else "transformers"


def _client():
    import huggingface_hub

    return huggingface_hub


def search_models(
    *,
    query: str | None = None,
    gguf_only: bool = False,
    limit: int = 40,
    author: str | None = None,
) -> list[HubModel]:
    """List decision models, newest first, optionally GGUF-only."""

    hub = _client()
    filters = [DECISION_TAG]
    if gguf_only:
        filters.append("gguf")
    kwargs: dict[str, Any] = {
        "filter": filters,
        "limit": limit,
        "sort": "downloads",
        "full": False,
    }
    if query:
        kwargs["search"] = query
    if author:
        kwargs["author"] = author
    results = hub.list_models(**kwargs)
    return [_to_model(info) for info in results]


def _to_model(info: Any) -> HubModel:
    info = info if isinstance(info, dict) else info.__dict__
    tags = list(info.get("tags") or [])
    return HubModel(
        id=info["id"],
        downloads=int(info.get("downloads") or 0),
        likes=int(info.get("likes") or 0),
        tags=tags,
        library=info.get("library_name"),
        is_gguf="gguf" in tags,
        revision=info.get("sha"),
    )


def list_gguf_files(repo_id: str, revision: str | None = None) -> list[str]:
    """Return the ``.gguf`` files in a repo, largest last."""

    hub = _client()
    files = hub.list_repo_files(repo_id, revision=revision)
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    return sorted(ggufs)


def resolve_revision(repo_id: str, revision: str | None = None) -> str:
    """Resolve a branch/tag to a commit SHA for reproducible downloads."""

    hub = _client()
    info = hub.model_info(repo_id, revision=revision, files_metadata=False)
    sha = getattr(info, "sha", None)
    if not sha:
        raise ValueError(f"Could not resolve a revision for {repo_id}")
    return str(sha)


def describe(repo_id: str, revision: str | None = None) -> dict[str, Any]:
    """Return Hub metadata used to choose a backend."""

    hub = _client()
    info = hub.model_info(repo_id, revision=revision)
    tags = list(info.tags or [])
    siblings = [s.rfilename for s in (info.siblings or [])]
    ggufs = sorted(f for f in siblings if f.lower().endswith(".gguf"))
    return {
        "id": info.id,
        "sha": info.sha,
        "tags": tags,
        "library": info.library_name,
        "is_gguf": "gguf" in tags or bool(ggufs),
        "gguf_files": ggufs,
        "has_custom_code": "custom-code" in tags or any(
            f == "configuration_auto.py" or f.startswith("modeling_") for f in siblings
        ),
        "files": siblings,
    }
