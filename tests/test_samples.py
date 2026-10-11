"""Bundled sample validity tests."""

from __future__ import annotations

from decision_server.samples import sample_json, sample_names, sample_request


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
