"""Tests for the Gradio UI helpers (no server is launched)."""

from __future__ import annotations

from pathlib import Path

import pytest

from decision_server.app import (
    _model_choices,
    category_choices,
    example_card,
    example_rows,
    filter_samples,
    resolve_row,
    summarize_response,
)
from decision_server.registry import Registry
from decision_server.samples import sample_names, sample_request, samples
from decision_server.schema import DecisionResponse, Prediction
from decision_server.service import run

CATALOG = """
models:
  - id: demo
    label: Demo
    kind: local
    backend: plugin
    plugin: decision_server.plugins.uniform
    tasks: [choice, noul, score]
  - id: choiceonly
    label: Choice only
    kind: local
    backend: plugin
    plugin: decision_server.plugins.uniform
    tasks: [choice]
  - id: cloud
    label: Cloud
    kind: cloud
    backend: cloud
    model: remote
    base_url: https://example.invalid
    api_key_env: DECISION_SERVER_TEST_MISSING_KEY
    tasks: [choice, noul, score]
"""


@pytest.fixture()
def registry(tmp_path: Path) -> Registry:
    path = tmp_path / "models.yaml"
    path.write_text(CATALOG, encoding="utf-8")
    return Registry.from_file(path)


def test_build_app(registry: Registry):
    gradio = pytest.importorskip("gradio")
    from decision_server.app import build_app

    assert isinstance(build_app(registry), gradio.Blocks)


def test_model_choices_flag_unavailable(registry: Registry):
    labels = dict((value, label) for label, value in _model_choices(registry))
    assert labels["demo"] == "Demo"
    assert "unavailable" in labels["cloud"]


def test_filter_all(registry: Registry):
    assert [s.name for s in filter_samples("All categories", "")] == sample_names()
    assert filter_samples("", "") == samples()


def test_category_choices(registry: Registry):
    choices = category_choices()
    labels = dict((value, label) for label, value in choices)
    assert labels[""] == f"All categories ({len(samples())})"
    assert set(labels) == {""} | {s.category for s in samples()}
    for label, value in choices:
        if value:
            count = sum(1 for s in samples() if s.category == value)
            assert label.endswith(f"({count})")


def test_filter_by_category(registry: Registry):
    basics = filter_samples("Basics", "")
    assert basics
    assert all(s.category == "Basics" for s in basics)


def test_filter_by_query(registry: Registry):
    assert [s.name for s in filter_samples("All categories", "ticket")]
    assert [s.name for s in filter_samples("All categories", "noul batch")]
    assert filter_samples("All categories", "no-such-example") == []


def test_filter_category_and_query_combine(registry: Registry):
    assert filter_samples("Finance", "return") == []


def test_example_rows(registry: Registry):
    rows = example_rows(samples())
    assert len(rows) == len(samples())
    assert all(len(row) == 3 for row in rows)
    assert all(isinstance(cell, str) for row in rows for cell in row)


def test_resolve_row():
    names = ["a", "b", "c"]
    assert resolve_row(names, 1) == "b"
    assert resolve_row(names, [2, 4]) == "c"
    assert resolve_row(names, (0, 0)) == "a"
    assert resolve_row(names, None) is None
    assert resolve_row(names, 7) is None
    assert resolve_row([], 0) is None


def test_example_card(registry: Registry):
    card = example_card(registry, "support-thread", "demo")
    assert "**Whole conversation**" in card
    assert "`support-thread`" in card
    assert "| `team` | choice |" in card
    assert "does not support" not in card


def test_example_card_warns_about_unsupported_tasks(registry: Registry):
    card = example_card(registry, "mixed", "choiceonly")
    assert "does not support" in card
    assert "noul" in card and "score" in card


def test_example_card_without_selection(registry: Registry):
    assert "Pick a row" in example_card(registry, "", "demo")


def test_summarize_response(registry: Registry):
    request = sample_request("mixed")
    response = run(registry, "demo", request)
    summary = summarize_response(response, request)
    assert "`demo`" in summary
    assert "| Question | Answer | Confidence | Distribution |" in summary
    for question in request.questions:
        assert f"`{question.id}`" in summary


def test_summarize_response_shows_score_expectation(registry: Registry):
    request = sample_request("review-rating")
    response = run(registry, "demo", request)
    assert "≈" in summarize_response(response, request)


def test_summarize_response_renders_errors(registry: Registry):
    request = sample_request("noul")
    response = DecisionResponse(
        model="demo",
        backend="plugin",
        answers=[Prediction(question_id="entailment", error="backend exploded")],
    )
    summary = summarize_response(response, request)
    assert "error" in summary
    assert "backend exploded" in summary
