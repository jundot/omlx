# SPDX-License-Identifier: Apache-2.0
"""The /search route wiring (fake-driven): async job -> loop -> best result.

diverse_candidates and evaluate_constellation are mocked; the route's
orchestration (validate measurement, spawn the job, poll progress, serialize the
best) is tested through the real router off-cluster (AGENTS.md). The search is a
background job: POST returns a handle, GET /search/{job_id} polls the outcome.
"""

from __future__ import annotations

import time

from assistant_harness import take_measurement

from omlx.cluster.assistant import policy as assistant_policy
from omlx.cluster.assistant import search_wiring
from omlx.cluster.assistant.election import PlanCandidate
from omlx.cluster.assistant.evaluation import ConstellationResult

SEARCH_URL = "/admin/api/cluster/assistant/search"


def _candidate(order: tuple[str, ...]) -> PlanCandidate:
    return PlanCandidate(
        rank_order=order, stages=(), feasible=True, reason="estimate",
        plan_hash=order[0] * 16, context_tokens=32768,
    )


def _run_search(client, payload) -> dict:
    """POST /search, then poll the job until it completes; return the outcome."""
    response = client.post(SEARCH_URL, json=payload)
    assert response.status_code == 200, response.text
    handle = response.json()
    assert handle["state"] == "queued"
    job_id = handle["job_id"]
    assert handle["poll"].endswith(job_id)
    for _ in range(200):
        snapshot = client.get(f"{SEARCH_URL}/{job_id}").json()
        if snapshot["state"] in ("completed", "failed"):
            return snapshot
        time.sleep(0.02)
    raise AssertionError("search job did not complete in time")


def test_search_route_runs_the_loop_and_returns_the_best_measured(api, monkeypatch) -> None:
    client, _, _ = api()
    measurement_id = take_measurement(client)

    candidates = [_candidate(("a", "b")), _candidate(("a", "b", "c"))]
    monkeypatch.setattr(
        assistant_policy, "diverse_candidates", lambda *a, **k: candidates, raising=False
    )
    rates = {("a", "b"): 40.0, ("a", "b", "c"): 70.0}

    async def fake_evaluate(candidate, **kwargs):
        return ConstellationResult(
            rank_order=candidate.rank_order,
            model_path="/models/qwen",
            context_tokens=32768,
            plan_signature=candidate.plan_hash,
            decode_tok_per_s=rates[candidate.rank_order],
            feasible=True,
        )

    monkeypatch.setattr(search_wiring, "evaluate_constellation", fake_evaluate)

    body = _run_search(client, {"measurement_id": measurement_id})
    assert body["state"] == "completed"
    assert body["best"]["rank_order"] == ["a", "b", "c"]
    assert body["best"]["decode_tok_per_s"] == 70.0
    assert len(body["results"]) == 2
    assert len(body["tested"]) == 2
    assert body["done"] == 2


def test_search_route_refuses_an_unknown_measurement(api) -> None:
    client, _, _ = api()
    response = client.post(SEARCH_URL, json={"measurement_id": "ms-nope"})
    assert response.status_code == 409
    assert response.json()["code"] == "stale_measurement"


def test_search_route_no_candidates_is_a_refusal(api, monkeypatch) -> None:
    client, _, _ = api()
    measurement_id = take_measurement(client)
    monkeypatch.setattr(assistant_policy, "diverse_candidates", lambda *a, **k: [], raising=False)
    response = client.post(SEARCH_URL, json={"measurement_id": measurement_id})
    assert response.status_code == 409
    assert response.json()["code"] == "no_feasible_plan"


def test_search_status_unknown_job_is_a_refusal(api) -> None:
    client, _, _ = api()
    response = client.get(f"{SEARCH_URL}/search-nope")
    assert response.status_code == 409
    assert response.json()["code"] == "search_not_found"


def test_search_route_with_real_candidate_generation(api, monkeypatch) -> None:
    """End-to-end: real diverse_candidates from the measurement, fake evaluate."""
    client, _, _ = api()
    measurement_id = take_measurement(client)

    async def fake_evaluate(candidate, **kwargs):
        return ConstellationResult(
            rank_order=candidate.rank_order,
            model_path="/models/qwen",
            context_tokens=candidate.context_tokens,
            plan_signature=candidate.plan_hash,
            decode_tok_per_s=50.0 + len(candidate.rank_order),
            feasible=True,
        )

    monkeypatch.setattr(search_wiring, "evaluate_constellation", fake_evaluate)
    body = _run_search(client, {"measurement_id": measurement_id, "max_count": 3})
    assert body["state"] == "completed"
    assert len(body["tested"]) >= 1
    assert body["best"] is not None
    assert body["best"]["measured"] is True
    assert all(r["feasible"] for r in body["results"])