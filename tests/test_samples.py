"""Bundled sample validity tests."""

from __future__ import annotations

import json

import pytest

from decision_server.samples import (
    CATEGORY_ORDER,
    categories,
    sample,
    sample_json,
    sample_names,
    sample_request,
    samples,
)


def test_samples_present():
    assert "choice" in sample_names()
    assert "noul" in sample_names()
    assert "score" in sample_names()


def test_every_sample_validates():
    for name in sample_names():
        request = sample_request(name)
        assert request.questions


def test_noul_sample_uses_false_true():
    request = sample_request("noul")
    assert request.questions[0].task == "noul"
    assert {o.id for o in request.questions[0].options} == {"false", "true"}


def test_mixed_sample_has_all_tasks():
    tasks = {q.task for q in sample_request("mixed").questions}
    assert tasks == {"choice", "noul", "score"}


def test_sample_json_is_serializable():
    assert '"state"' in sample_json("choice")


def test_every_sample_has_metadata():
    for smp in samples():
        assert smp.label and smp.summary
        assert smp.category in CATEGORY_ORDER
        assert smp.tasks
        assert smp.question_count == len(smp.payload["questions"])


def test_names_and_labels_are_unique():
    names = sample_names()
    assert len(set(names)) == len(names)
    labels = [s.label for s in samples()]
    assert len(set(labels)) == len(labels)


def test_every_category_is_used():
    assert set(categories()) == {s.category for s in samples()}
    assert categories() == [c for c in CATEGORY_ORDER if c in categories()]


def test_catalog_covers_every_task():
    tasks = {q["task"] for smp in samples() for q in smp.payload["questions"]}
    assert tasks == {"choice", "noul", "score"}


def test_all_noul_questions_use_false_true():
    for smp in samples():
        for q in smp.payload["questions"]:
            if q["task"] == "noul":
                assert {o["id"] for o in q["options"]} == {"false", "true"}


def test_all_score_questions_have_distinct_values():
    for smp in samples():
        for q in smp.payload["questions"]:
            if q["task"] == "score":
                values = [o["value"] for o in q["options"]]
                assert all(isinstance(v, int | float) for v in values)
                assert len(set(values)) > 1


def test_sample_metadata_matches_payload():
    for smp in samples():
        request = smp.request()
        assert smp.tasks == list(dict.fromkeys(q.task for q in request.questions))
        assert smp.question_count == len(request.questions)
        assert json.loads(smp.json()) == smp.payload


def test_sample_lookup():
    assert sample("choice").name == "choice"
    with pytest.raises(KeyError):
        sample("nope")


def test_sample_json_indent():
    compact = sample_json("choice", indent=2)
    wide = sample_json("choice", indent=4)
    assert json.loads(compact) == json.loads(wide)
    assert len(wide) > len(compact)
