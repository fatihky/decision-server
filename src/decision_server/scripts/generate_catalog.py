"""Generate the decision-model catalog from a list of Hub repositories.

This script turns a set of candidate repository IDs into ``models.yaml`` entries
by asking the Hub how each one can be served:

- repositories tagged ``gguf`` become ``gguf`` entries (served by llama-server),
- repositories with custom/auto_map code become ``automap`` entries,
- the rest are recorded as ``unavailable`` with the reason.

Run it once to (re)build the catalog, or point it at your own repo list::

    python -m decision_server.scripts.generate_catalog repos.txt > models.yaml

``repos.txt`` is one repository ID per line; blank lines and ``#`` comments are
ignored. ``--include-discovered`` also pulls recent repositories from the Hub
``decision-model`` tag.

The bundled catalogs were built from the model roster of S1MB (System One
Mosaic Benchmark, MIT): https://github.com/hotchpotch/S1MB — extract the
``model`` values of its ``evaluator/upstream-models.json`` to reproduce the
input list::

    curl -s https://raw.githubusercontent.com/hotchpotch/S1MB/main/evaluator/upstream-models.json \\
      | jq -r '.[].model // empty' | sort -u > repos.txt

S1MB drives each model through a dedicated adapter against a pinned upstream
checkout; ``ecosystem.yaml`` serves the same models from their GGUF mirrors and
``families.yaml`` from in-process plugins (see the README's Provenance section).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ..hub import describe, search_models


def _entry_for(repo_id: str) -> dict:
    try:
        meta = describe(repo_id)
    except Exception as exc:  # noqa: BLE001 - report per-repo failures
        return {
            "id": repo_id.replace("/", "__"),
            "label": repo_id,
            "kind": "local",
            "backend": "unavailable",
            "source": f"https://huggingface.co/{repo_id}",
            "tasks": [],
            "description": f"Could not read Hub metadata: {exc}",
        }

    base = {
        "id": repo_id.replace("/", "__"),
        "label": repo_id,
        "kind": "local",
        "source": f"https://huggingface.co/{repo_id}",
        "revision": meta.get("sha"),
        "tasks": ["choice", "noul", "score"],
    }
    if meta["is_gguf"]:
        base.update(
            backend="gguf",
            repo_id=repo_id,
            gguf_file=meta["gguf_files"][0] if meta["gguf_files"] else None,
            description=(
                f"GGUF decision model served by llama-server "
                f"({len(meta['gguf_files'])} files)."
            ),
        )
    elif meta["has_custom_code"]:
        base.update(
            backend="automap",
            repo_id=repo_id,
            description="Transformer decision model loaded through its custom code.",
        )
    else:
        base.update(
            backend="unavailable",
            tasks=[],
            description=(
                "No GGUF and no custom auto_map interface; "
                "requires a family-specific plugin."
            ),
        )
    return base


def _discovered(limit: int) -> list[str]:
    return [m.id for m in search_models(limit=limit)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generate-catalog")
    parser.add_argument("repos", type=Path, help="File with one repository ID per line.")
    parser.add_argument("--include-discovered", action="store_true")
    parser.add_argument("--discover-limit", type=int, default=60)
    args = parser.parse_args(argv)

    repos = []
    for line in args.repos.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            repos.append(line)
    if args.include_discovered:
        repos.extend(_discovered(args.discover_limit))

    seen: set[str] = set()
    entries = []
    for repo in repos:
        if repo in seen:
            continue
        seen.add(repo)
        entries.append(_entry_for(repo))

    yaml.safe_dump({"models": entries}, sys.stdout, sort_keys=False, allow_unicode=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
