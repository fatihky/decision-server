"""Gradio UI for exploring and calling the decision models."""

from __future__ import annotations

import json
from typing import Any

import gradio as gr

from .registry import Registry
from .samples import Sample, categories, sample, sample_json, sample_names, samples
from .schema import DecisionRequest, DecisionResponse, Question
from .service import run

_ALL_CATEGORIES = "All categories"
_BAR_WIDTH = 14

CSS = """
.examples-table table { font-size: 0.86rem; }
.examples-table td { padding-top: 0.35rem; padding-bottom: 0.35rem; }
.example-card { border-left: 3px solid var(--color-accent, #f97316); padding-left: 0.8rem; }
.result-summary table { font-size: 0.86rem; }
"""


def _model_choices(registry: Registry) -> list[tuple[str, str]]:
    """Return (label, value) pairs, flagging unavailable models."""

    choices = []
    for d in registry.descriptions():
        label = d.label
        if not d.available:
            label = f"{label} — unavailable ({d.unavailable_reason})"
        choices.append((label, d.id))
    return choices


def _model_tasks(registry: Registry, model_id: str | None) -> set[str]:
    for d in registry.descriptions():
        if d.id == model_id:
            return set(d.tasks)
    return set()


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


def _matches(smp: Sample, query: str) -> bool:
    haystack = " ".join(
        [
            smp.name,
            smp.label,
            smp.category,
            smp.summary,
            *smp.tasks,
            *(q["id"] for q in smp.payload["questions"]),
            *(q["instructions"] for q in smp.payload["questions"]),
        ]
    ).lower()
    return all(term in haystack for term in query.lower().split())


def filter_samples(category: str, query: str) -> list[Sample]:
    """Examples matching a category and a free-text filter, in catalog order."""

    query = (query or "").strip()
    return [
        smp
        for smp in samples()
        if category in ("", _ALL_CATEGORIES, smp.category) and (not query or _matches(smp, query))
    ]


def category_choices() -> list[tuple[str, str]]:
    """(label, value) pairs for the category filter, each labelled with its count."""

    counts: dict[str, int] = {}
    for smp in samples():
        counts[smp.category] = counts.get(smp.category, 0) + 1
    return [
        (f"{_ALL_CATEGORIES} ({len(samples())})", ""),
        *[(f"{name} ({counts[name]})", name) for name in categories()],
    ]


def example_rows(items: list[Sample]) -> list[list[str]]:
    """Table rows for the examples browser."""

    return [
        [smp.label, smp.category, " · ".join(smp.tasks), str(smp.question_count), smp.summary]
        for smp in items
    ]


def resolve_row(names: list[str], index: Any) -> str | None:
    """Map a table click to an example name; ``index`` is a row or a (row, col) pair."""

    row = index[0] if isinstance(index, (list, tuple)) else index
    if row is None:
        return None
    row = int(row)
    return names[row] if 0 <= row < len(names) else None


def example_card(registry: Registry, name: str, model_id: str | None) -> str:
    """Markdown detail for one example, including model compatibility."""

    if not name:
        return "_Pick a row above to see what the example contains._"
    smp = sample(name)
    lines = [
        f"**{smp.label}** — `{smp.name}`",
        f"{smp.category} · {smp.question_count} question"
        f"{'s' if smp.question_count != 1 else ''} · " + " ".join(f"`{t}`" for t in smp.tasks),
        "",
        smp.summary,
        "",
        "| Question | Task | Options |",
        "| --- | --- | --- |",
    ]
    for q in smp.payload["questions"]:
        options = ", ".join(
            o["id"] + (f"={o['value']:g}" if isinstance(o.get("value"), int | float) else "")
            for o in q["options"]
        )
        lines.append(f"| `{q['id']}` | {q['task']} | {options} |")

    unsupported = sorted(set(smp.tasks) - _model_tasks(registry, model_id))
    if model_id and unsupported:
        lines.append(f"\n⚠️ The selected model does not support: {', '.join(unsupported)}")
    return "\n".join(lines)


def _bar(probability: float) -> str:
    filled = round(max(0.0, min(1.0, probability)) * _BAR_WIDTH)
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def _expected_value(question: Question, probabilities: dict[str, float]) -> float | None:
    """Probability-weighted mean of a score scale, or None if values are absent."""

    total = 0.0
    for option in question.options:
        if option.value is None:
            return None
        total += probabilities.get(option.id, 0.0) * float(option.value)
    return total


def summarize_response(response: DecisionResponse, request: DecisionRequest) -> str:
    """Render answers as a table instead of raw JSON: top option, bar, spread."""

    questions = {q.id: q for q in request.questions}
    header = f"`{response.model}` · {response.backend}"
    if response.revision:
        header += f" · `{response.revision[:12]}`"
    usage = response.usage
    header += f" · {usage.requests} req · {usage.input_tokens} in / {usage.output_tokens} out tok"

    rows = ["", "| Question | Answer | Confidence | Distribution |", "| --- | --- | --- | --- |"]
    for answer in response.answers:
        if answer.error:
            rows.append(f"| `{answer.question_id}` | ⚠️ error | — | {answer.error} |")
            continue
        probabilities = answer.probabilities or {}
        ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top_p = ranked[0]
        rest = " · ".join(f"{k} {v:.3f}" for k, v in ranked[1:4])
        if len(ranked) > 4:
            rest += " · …"
        answer_cell = f"**{top_id}**"
        question = questions.get(answer.question_id)
        if question is not None and question.task == "score":
            expected = _expected_value(question, probabilities)
            if expected is not None:
                answer_cell += f"<br>≈ {expected:g}"
        rows.append(
            f"| `{answer.question_id}` | {answer_cell} | {_bar(top_p)} {top_p:.3f} | {rest} |"
        )
    return header + "\n".join(rows)


def _format_response(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_app(registry: Registry) -> gr.Blocks:
    choices = _model_choices(registry)
    first = choices[0][1] if choices else None
    initial_examples = filter_samples("", "")
    initial_example = initial_examples[0].name

    with gr.Blocks(title="Decision Server", css=CSS) as app:
        gr.Markdown(
            "# Decision Server\n"
            "Serve choice, noul and score models behind one typed contract. "
            "Pick a model, pick an example, press **Run**."
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                model = gr.Dropdown(
                    choices=choices,
                    value=first,
                    label="Model",
                    interactive=True,
                )
                info = gr.Markdown(_model_info(registry, first) if first else "")
                run_btn = gr.Button("Run request", variant="primary")

            with gr.Column(scale=3, min_width=520):
                with gr.Group():
                    gr.Markdown(f"### Examples ({len(samples())})")
                    with gr.Row():
                        category = gr.Dropdown(
                            choices=category_choices(),
                            value="",
                            label="Category",
                            scale=2,
                        )
                        example_filter = gr.Textbox(
                            label="Filter",
                            placeholder="search labels, questions, tasks…",
                            scale=3,
                        )
                    examples_table = gr.Dataframe(
                        headers=["Example", "Category", "Tasks", "Qs", "What it shows"],
                        datatype=["str", "str", "str", "str", "str"],
                        value=example_rows(initial_examples),
                        interactive=False,
                        wrap=True,
                        max_height=232,
                        elem_classes=["examples-table"],
                        label="Click a row to load it",
                    )
                    example_detail = gr.Markdown(
                        example_card(registry, initial_example, first),
                        elem_classes=["example-card"],
                    )
                    with gr.Row():
                        load_btn = gr.Button("Load into editor", scale=1)
                        run_example_btn = gr.Button("Load & run", variant="primary", scale=1)

                request_box = gr.Code(
                    label="Request JSON",
                    language="json",
                    value=sample_json(initial_example),
                    lines=22,
                )
                result_summary = gr.Markdown("", elem_classes=["result-summary"])
                with gr.Accordion("Response JSON", open=True):
                    response_box = gr.Code(
                        label="Response",
                        language="json",
                        value="",
                        lines=20,
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

        # The row order follows the current filter, so keep the visible names
        # alongside the table to resolve a clicked row back to an example.
        visible = gr.State([s.name for s in initial_examples])
        selected = gr.State(initial_example)

        def _apply_filter(cat: str, query: str, name: str, model_id: str | None):
            items = filter_samples(cat, query)
            names = [s.name for s in items]
            if not items:
                card = "_No examples match this filter._"
            elif name in names:
                card = example_card(registry, name, model_id)
            else:
                card = example_card(registry, "", model_id)
            return example_rows(items), names, card

        category.change(
            fn=_apply_filter,
            inputs=[category, example_filter, selected, model],
            outputs=[examples_table, visible, example_detail],
            api_name="filter_examples",
        )
        example_filter.change(
            fn=_apply_filter,
            inputs=[category, example_filter, selected, model],
            outputs=[examples_table, visible, example_detail],
            api_name="search_examples",
        )

        def _select(names: list[str], model_id: str | None, sel: gr.SelectData):
            name = resolve_row(names, sel.index)
            if name is None:
                return gr.update(), gr.update(), gr.update()
            return (
                sample_json(name),
                example_card(registry, name, model_id),
                name,
            )

        examples_table.select(
            fn=_select,
            inputs=[visible, model],
            outputs=[request_box, example_detail, selected],
            api_name="select_example",
        )

        def _load(name: str):
            if name not in sample_names():
                return gr.update()
            return sample_json(name)

        load_btn.click(fn=_load, inputs=[selected], outputs=[request_box], api_name="load_example")

        def _on_model(model_id: str | None, name: str):
            return _model_info(registry, model_id or ""), example_card(registry, name, model_id)

        model.change(
            fn=_on_model,
            inputs=[model, selected],
            outputs=[info, example_detail],
            api_name="describe_model",
        )

        def _execute(model_id: str | None, data: dict) -> tuple[str, str]:
            if not model_id:
                return "", _format_response({"error": "Select a model"})
            try:
                request = DecisionRequest.model_validate(data)
            except Exception as exc:  # noqa: BLE001 - report validation problems
                return "", _format_response({"error": f"Invalid request: {exc}"})
            try:
                response = run(registry, model_id, request)
            except Exception as exc:  # noqa: BLE001 - report backend problems
                return "", _format_response({"error": str(exc) or exc.__class__.__name__})
            return (
                summarize_response(response, request),
                _format_response(response.model_dump(exclude_none=True)),
            )

        def _run_raw(model_id: str | None, raw: str) -> tuple[str, str]:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                return "", _format_response({"error": f"Invalid JSON: {exc}"})
            return _execute(model_id, data)

        run_btn.click(
            fn=_run_raw,
            inputs=[model, request_box],
            outputs=[result_summary, response_box],
            scroll_to_output=True,
            api_name="run_request",
        )

        def _run_example(name: str, model_id: str | None) -> tuple[Any, str, str]:
            if name not in sample_names():
                return gr.update(), "", _format_response({"error": "Pick an example row first"})
            raw = sample_json(name)
            summary, body = _execute(model_id, json.loads(raw))
            return raw, summary, body

        run_example_btn.click(
            fn=_run_example,
            inputs=[selected, model],
            outputs=[request_box, result_summary, response_box],
            scroll_to_output=True,
            api_name="run_example",
        )

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
            api_name="search_hub",
        )

        def _add(repo_id: str):
            if not repo_id.strip():
                return gr.update(), "Enter a repo id first."
            try:
                desc = registry.add_hub_model(repo_id.strip())
            except Exception as exc:  # noqa: BLE001 - surface registration errors
                return gr.update(), f"Could not add: {exc}"
            return gr.update(choices=_model_choices(registry), value=desc.id), f"Added {desc.id}."

        hub_add.click(
            fn=_add, inputs=[hub_repo], outputs=[model, hub_status], api_name="add_hub_model"
        )

    return app
