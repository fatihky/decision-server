"""Console entry point: launch the Gradio UI and/or the JSON API."""

from __future__ import annotations

import argparse
import sys

from .app import build_app
from .registry import Registry, default_registry


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="decision-server", description="Serve decision models.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default localhost).")
    parser.add_argument("--port", type=int, default=7860, help="Bind port (default 7860).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--registry", help="Path to an alternative models.yaml catalog.")
    parser.add_argument("--list", action="store_true", help="List registered models and exit.")
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Serve only the JSON API (no Gradio UI).",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Serve only the Gradio UI (no JSON API).",
    )
    parser.add_argument(
        "--discover",
        nargs="?",
        const="",
        metavar="QUERY",
        help="Search the Hub for decision models and exit.",
    )
    parser.add_argument(
        "--gguf-only",
        action="store_true",
        help="With --discover, list only GGUF-servable models.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.registry:
        from pathlib import Path

        registry = Registry.from_file(Path(args.registry))
    else:
        registry = default_registry()

    if args.list:
        descriptions = registry.descriptions()
        available = sum(d.available for d in descriptions)
        for d in descriptions:
            status = "ok" if d.available else f"unavailable: {d.unavailable_reason}"
            print(f"{d.id}\t{d.kind}/{d.backend}\t{','.join(d.tasks)}\t{status}")
        print(f"\n{available}/{len(descriptions)} models available", file=sys.stderr)
        return 0

    if args.discover is not None:
        from .hub import search_models

        models = search_models(
            query=args.discover or None, gguf_only=args.gguf_only, limit=50
        )
        for m in models:
            print(f"{m.id}\t{m.servable_via}\t{m.downloads} downloads")
        return 0

    if args.api_only:
        import uvicorn

        from .api import create_api

        uvicorn.run(create_api(registry), host=args.host, port=args.port)
        return 0

    app = build_app(registry)
    if args.no_api:
        app.launch(server_name=args.host, server_port=args.port, share=args.share)
        return 0

    # Serve the JSON API at the root and mount the Gradio UI at "/ui", so both
    # run on one port under one server process.
    import gradio as gr
    import uvicorn

    from .api import create_api

    api = create_api(registry)
    gr.mount_gradio_app(api, app, path="/ui")
    print(f"API:  http://{args.host}:{args.port}/v1/systemone")
    print(f"UI:   http://{args.host}:{args.port}/ui")
    uvicorn.run(api, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
