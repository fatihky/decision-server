"""Shared helpers for loading code that a model repository ships."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..backend import BackendError

# Ordered by likely entry name across the ecosystem.
ENTRY_NAMES = ("decide", "system_one", "predict", "infer", "judge", "score_options")


def snapshot_code_files(repo_id: str, revision: str | None, filenames: list[str]) -> Path:
    """Download the named files from a repo and return their directory."""

    import huggingface_hub

    directory: Path | None = None
    for name in filenames:
        try:
            path = huggingface_hub.hf_hub_download(
                repo_id=repo_id, filename=name, revision=revision
            )
        except Exception as exc:  # noqa: BLE001 - report which file failed
            raise BackendError(f"Could not download {repo_id}/{name}: {exc}") from exc
        directory = Path(path).parent
    if directory is None:
        raise BackendError(f"No code files requested from {repo_id}")
    return directory


def load_module_from(path: Path, module_name: str) -> types.ModuleType:
    """Import a single Python file by path without requiring it on sys.path."""

    if not path.is_file():
        raise BackendError(f"Module file not found: {path}")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise BackendError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - surface import-time failures
        raise BackendError(f"Importing {path.name} failed: {exc}") from exc
    return module


def find_entry(module: types.ModuleType, names: list[str] | None = None) -> Callable[..., Any]:
    """Return the first module-level callable matching a known entry name."""

    for name in names or ENTRY_NAMES:
        candidate = getattr(module, name, None)
        if callable(candidate) and not inspect.isclass(candidate):
            return candidate
    raise BackendError(
        f"No entry point among {names or ENTRY_NAMES} in {module.__name__}"
    )


def find_model_class(module: types.ModuleType, names: list[str] | None = None) -> type:
    """Return the first class exposing a ``from_pretrained`` constructor."""

    for name, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        if names and name not in names:
            continue
        if hasattr(obj, "from_pretrained"):
            return obj
    if names:
        for name in names:
            obj = getattr(module, name, None)
            if inspect.isclass(obj):
                return obj
    raise BackendError(f"No model class with from_pretrained in {module.__name__}")
