"""Gradio UI for exploring and calling the decision models."""

from __future__ import annotations

import json

import gradio as gr

from .registry import Registry
from .samples import sample_json, sample_names
from .schema import DecisionRequest
from .service import run


def _model_choices(registry: Registry) -> list[tuple[str, str]]:
    """Return (label, value) pairs, flagging unavailable models."""

    choices = []
    for d in registry.descriptions():
        label = d.label
        if not d.available:
            label = f"{label} — unavailable ({d.unavailable_reason})"
        choices.append((label, d.id))
    return choices


def _model_info(registry: Registry, model_id: str) -> str:
    for d in registry.descriptions():
        if d.id == model_id:
            lines = [
                f"**{d.label}**",
                f"- id: `{d.id}`",
                f"- kind: {d.kind} / backend: {d.backend}",
                f"- tasks: {', '.join(d.tasks)}",
            ]
            if d.source:
                lines.append(f"- source: {d.source}")
            if d.revision:
                lines.append(f"- revision: `{d.revision}`")
            if d.requires:
                lines.append(f"- requires: {', '.join(d.requires)}")
            if d.description:
                lines.append(f"\n{d.description}")
            if not d.available:
                lines.append(f"\n⚠️ Unavailable: {d.unavailable_reason}")
            return "\n".join(lines)
    return "_Select a model._"


def _format_response(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_app(registry: Registry) -> gr.Blocks:
    choices = _model_choices(registry)
    first = choices[0][1] if choices else None

    with gr.Blocks(title="Decision Server") as app:
        gr.Markdown(
            "# Decision Server\nServe choice, noul and score models behind one typed contract."
        )

        with gr.Row():
            with gr.Column(scale=1):
                model = gr.Dropdown(
                    choices=choices,
                    value=first,
                    label="Model",
                    interactive=True,
                )
                info = gr.Markdown(_model_info(registry, first) if first else "")
                sample = gr.Dropdown(
                    choices=["— none —", *sample_names()],
                    value="— none —",
                    label="Load sample",
                )
                load = gr.Button("Load sample into editor")
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=3):
                request_box = gr.Code(
                    label="Request JSON",
                    language="json",
                    value=sample_json(sample_names()[0]),
                    lines=24,
                )
                response_box = gr.Code(
                    label="Response JSON",
                    language="json",
                    value="",
                    lines=24,
                )

                with gr.Accordion("Search Hugging Face Hub", open=False):
                    with gr.Row():
                        hub_query = gr.Textbox(label="Query", placeholder="e.g. kev, laya, decider")
                        hub_gguf = gr.Checkbox(label="GGUF only", value=True)
                        hub_search = gr.Button("Search")
                    hub_results = gr.Dataframe(
                        headers=["repo_id", "downloads", "servable_via", "gguf_file"],
                        interactive=False,
                        wrap=True,
                    )
                    with gr.Row():
                        hub_repo = gr.Textbox(label="Repo id", placeholder="owner/repo")
                        hub_add = gr.Button("Add model")
                    hub_status = gr.Markdown("")

        sample.select(
            fn=lambda name: sample_json(name) if name in sample_names() else gr.update(),
            inputs=[sample],
            outputs=[request_box],
        )
        load.click(
            fn=lambda name: sample_json(name) if name in sample_names() else gr.update(),
            inputs=[sample],
            outputs=[request_box],
        )
        model.change(
            fn=lambda mid: _model_info(registry, mid),
            inputs=[model],
            outputs=[info],
        )

        def _run(model_id: str, raw: str) -> str:
            if not model_id:
                return _format_response({"error": "Select a model"})
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                return _format_response({"error": f"Invalid JSON: {exc}"})
            try:
                request = DecisionRequest.model_validate(data)
            except Exception as exc:  # noqa: BLE001 - report validation problems
                return _format_response({"error": f"Invalid request: {exc}"})
            try:
                response = run(registry, model_id, request)
            except Exception as exc:  # noqa: BLE001 - report backend problems
                return _format_response({"error": str(exc) or exc.__class__.__name__})
            return _format_response(response.model_dump(exclude_none=True))

        run_btn.click(fn=_run, inputs=[model, request_box], outputs=[response_box])

        def _search(query: str, gguf_only: bool):
            from .hub import search_models

            try:
                results = search_models(query=query or None, gguf_only=gguf_only, limit=30)
            except Exception as exc:  # noqa: BLE001 - surface Hub errors in the UI
                return [], f"Search failed: {exc}"
            rows = [[m.id, m.downloads, m.servable_via, ""] for m in results]
            return rows, f"Found {len(results)} models. Copy a repo id above to add it."

        hub_search.click(
            fn=_search,
            inputs=[hub_query, hub_gguf],
            outputs=[hub_results, hub_status],
        )

        def _add(repo_id: str):
            if not repo_id.strip():
                return gr.update(), "Enter a repo id first."
            try:
                desc = registry.add_hub_model(repo_id.strip())
            except Exception as exc:  # noqa: BLE001 - surface registration errors
                return gr.update(), f"Could not add: {exc}"
            return gr.update(choices=_model_choices(registry), value=desc.id), f"Added {desc.id}."

        hub_add.click(fn=_add, inputs=[hub_repo], outputs=[model, hub_status])

    return app
