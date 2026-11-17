# SPDX-License-Identifier: Apache-2.0
"""The /tune-redeploy route (fake-driven): one deploy per config, rank real rates.

``diverse_candidates`` and ``evaluate_constellation`` are mocked; the route's
orchestration (validate the ready winner + measurement, spawn the job, measure
each config, save the winning knobs per model+node config, poll) is tested
through the real router off-cluster (AGENTS.md). A deploy per config is minutes,
so it is a background job.
"""

from __future__ import annotations

import time

from assistant_harness import MODEL_PATH, take_measurement

from omlx.cluster.assistant import policy as assistant_policy
from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant import search_wiring, tuning_wiring
from omlx.cluster.assistant.election import PlanCandidate
from omlx.cluster.assistant.evaluation import ConstellationResult
from omlx.cluster.assistant.operations import (
    KIND_START,
    STATE_READY,
    OperationEndpoint,
    OperationProgress,
)
from omlx.cluster.assistant.tuning import FineTuneResult
from omlx.cluster.assistant.tuning_store import TuningPreset

REDEPLOY_URL = "/admin/api/cluster/assistant/tune-redeploy"
TUNE_URL = "/admin/api/cluster/assistant/tune"


def _ready_winner(store, measurement_id: str):
    endpoint = OperationEndpoint(node="a", host="127.0.0.1", port=8099, ready=True)
    op = store.start(
        kind=KIND_START,
        measurement_id=measurement_id,
        coordinator_node="a",
        coordinator_local=True,
        rank_order=["a", "b"],
        context_tokens=32768,
        model_path=MODEL_PATH,
        endpoint=endpoint,
    )
    store.advance(
        op.operation_id,
        STATE_READY,
        progress=OperationProgress(phase="ready"),
        endpoint=endpoint,
    )
    return op


def _candidate(order=("a", "b"), context=32768) -> PlanCandidate:
    return PlanCandidate(
        rank_order=order,
        stages=(),
        feasible=True,
        reason="estimate",
        plan_hash="a" * 16,
        context_tokens=context,
    )


def _poll(client, job_id: str) -> dict:
    for _ in range(200):
        snapshot = client.get(f"{REDEPLOY_URL}/{job_id}").json()
        if snapshot["state"] in ("completed", "failed"):
            return snapshot
        time.sleep(0.02)
    raise AssertionError("redeploy tune job did not complete in time")


def _run(client, operation_id: str) -> dict:
    response = client.post(REDEPLOY_URL, json={"operation_id": operation_id})
    assert response.status_code == 200, response.text
    handle = response.json()
    assert handle["state"] == "queued"
    assert handle["poll"].endswith(handle["job_id"])
    return _poll(client, handle["job_id"])


def test_a_model_without_mtp_measures_off_and_saves_the_preset(api, monkeypatch) -> None:
    client, _, store = api()
    measurement_id = take_measurement(client)
    op = _ready_winner(store, measurement_id)

    monkeypatch.setattr(
        assistant_policy, "diverse_candidates", lambda *a, **k: [_candidate()], raising=False
    )

    async def fake_evaluate(candidate, **kwargs):
        return ConstellationResult(
            rank_order=candidate.rank_order,
            model_path=MODEL_PATH,
            context_tokens=candidate.context_tokens,
            plan_signature=candidate.plan_hash,
            decode_tok_per_s=42.0,
            feasible=True,
        )

    monkeypatch.setitem(search_wiring.__dict__, "evaluate_constellation", fake_evaluate)

    body = _run(client, op.operation_id)

    assert body["state"] == "completed"
    assert body["mtp_applicable"] is False
    assert [row["config"] for row in body["rows"]] == ["mtp_off"]
    assert body["best"]["config"] == "mtp_off"
    assert body["best"]["mtp"] is False
    assert body["preset_key"]

    preset = assistant_routes._tuning_store().get(MODEL_PATH, ",".join(["a", "b"]))
    assert preset is not None
    assert preset.mtp is False


def test_mtp_capable_model_ranks_the_depths_and_saves_the_winner(api, monkeypatch) -> None:
    client, _, store = api()
    measurement_id = take_measurement(client)
    op = _ready_winner(store, measurement_id)

    monkeypatch.setitem(assistant_routes.__dict__, "_tune_mtp_capable", lambda path: True)
    monkeypatch.setattr(
        assistant_policy, "diverse_candidates", lambda *a, **k: [_candidate()], raising=False
    )

    async def fake_evaluate(candidate, **kwargs):
        patch = kwargs.get("model_settings_patch") or {}
        depth = patch.get("mtp_adaptive_max_depth")
        # Depth 2 is the measured winner; MTP off is the baseline.
        rate = {None: 55.0, 1: 60.0, 2: 72.0, 3: 68.0, 4: 66.0}[depth]
        if not kwargs.get("mtp", False):
            rate = 40.0
        return ConstellationResult(
            rank_order=candidate.rank_order,
            model_path=MODEL_PATH,
            context_tokens=candidate.context_tokens,
            plan_signature=candidate.plan_hash,
            decode_tok_per_s=rate,
            feasible=True,
        )

    monkeypatch.setitem(search_wiring.__dict__, "evaluate_constellation", fake_evaluate)

    body = _run(client, op.operation_id)

    assert body["state"] == "completed"
    assert body["mtp_applicable"] is True
    assert [row["config"] for row in body["rows"]] == [
        "mtp_off",
        "mtp_on_adaptive",
        "mtp_on_depth_1",
        "mtp_on_depth_2",
        "mtp_on_depth_3",
        "mtp_on_depth_4",
    ]
    assert body["best"]["config"] == "mtp_on_depth_2"
    assert body["best"]["mtp_depth"] == 2

    preset = assistant_routes._tuning_store().get(MODEL_PATH, ",".join(["a", "b"]))
    assert preset is not None
    assert preset.mtp is True
    assert preset.mtp_depth == 2


def test_a_config_the_winner_cannot_reproduce_is_an_infeasible_row(api, monkeypatch) -> None:
    client, _, store = api()
    measurement_id = take_measurement(client)
    op = _ready_winner(store, measurement_id)

    monkeypatch.setitem(assistant_routes.__dict__, "_tune_mtp_capable", lambda path: True)
    # The winner's rank order exists for MTP off, not for the MTP-on configs.
    def candidates_for(*a, **k):
        return [_candidate()] if k.get("mtp") is False else []

    monkeypatch.setattr(assistant_policy, "diverse_candidates", candidates_for, raising=False)

    async def fake_evaluate(candidate, **kwargs):
        return ConstellationResult(
            rank_order=candidate.rank_order,
            model_path=MODEL_PATH,
            context_tokens=candidate.context_tokens,
            plan_signature=candidate.plan_hash,
            decode_tok_per_s=50.0,
            feasible=True,
        )

    monkeypatch.setitem(search_wiring.__dict__, "evaluate_constellation", fake_evaluate)

    body = _run(client, op.operation_id)
    by_label = {row["config"]: row for row in body["rows"]}
    assert by_label["mtp_on_depth_2"]["feasible"] is False
    assert body["best"]["config"] == "mtp_off"


def test_redeploy_tuning_refuses_a_non_ready_operation(api) -> None:
    client, _, store = api()
    op = store.start(kind=KIND_START, measurement_id="ms-1", coordinator_node="a")
    response = client.post(REDEPLOY_URL, json={"operation_id": op.operation_id})
    assert response.status_code == 409
    assert response.json()["code"] == "operation_not_ready"


def test_redeploy_status_unknown_job_is_a_refusal(api) -> None:
    client, _, _ = api()
    response = client.get(f"{REDEPLOY_URL}/redeploy-nope")
    assert response.status_code == 409
    assert response.json()["code"] == "redeploy_not_found"


def _fake_depth_sweep(monkeypatch, rates: dict) -> None:
    """A redeploy sweep whose measured rates are the given depth→rate map."""

    monkeypatch.setitem(assistant_routes.__dict__, "_tune_mtp_capable", lambda path: True)
    monkeypatch.setattr(
        assistant_policy, "diverse_candidates", lambda *a, **k: [_candidate()], raising=False
    )

    async def fake_evaluate(candidate, **kwargs):
        if not kwargs.get("mtp", False):
            rate = rates["off"]
        else:
            depth = (kwargs.get("model_settings_patch") or {}).get(
                "mtp_adaptive_max_depth", "adaptive"
            )
            rate = rates[depth]
        return ConstellationResult(
            rank_order=candidate.rank_order,
            model_path=MODEL_PATH,
            context_tokens=candidate.context_tokens,
            plan_signature=candidate.plan_hash,
            decode_tok_per_s=rate,
            feasible=True,
        )

    monkeypatch.setitem(search_wiring.__dict__, "evaluate_constellation", fake_evaluate)


def _fake_tune(monkeypatch, sampler: dict) -> None:
    """The sampler sweep of ``POST /tune``; it never measures the MTP depth."""

    async def fake_fine_tune(complete, *, model_id, **kwargs):
        return FineTuneResult(
            chunk=None,
            chunk_applicable=False,
            sampler=sampler,
            mtp=True,
            mtp_applicable=True,
        )

    monkeypatch.setitem(tuning_wiring.__dict__, "fine_tune_constellation", fake_fine_tune)


def test_a_tune_after_a_redeploy_keeps_both_knobs(api, monkeypatch) -> None:
    # One combined preset: the redeploy measures mtp+depth, /tune measures the
    # sampler; whichever runs second must not erase the other (the live-run notes
    # §26). This is the live regression (depth 3 was erased by a later /tune).
    client, _, store = api()
    measurement_id = take_measurement(client)
    op = _ready_winner(store, measurement_id)

    _fake_depth_sweep(
        monkeypatch,
        {"off": 40.0, "adaptive": 55.0, 1: 60.0, 2: 70.0, 3: 75.0, 4: 66.0},
    )
    body = _run(client, op.operation_id)
    assert body["best"]["mtp"] is True
    assert body["best"]["mtp_depth"] == 3

    _fake_tune(monkeypatch, {"sampler": "deterministic"})
    response = client.post(TUNE_URL, json={"operation_id": op.operation_id})
    assert response.status_code == 200, response.text

    preset = assistant_routes._tuning_store().get(MODEL_PATH, "a,b")
    assert preset is not None
    assert preset.mtp is True
    assert preset.mtp_depth == 3
    assert preset.sampler["sampler"] == "deterministic"


def test_a_redeploy_after_a_tune_keeps_both_knobs(api, monkeypatch) -> None:
    client, _, store = api()
    measurement_id = take_measurement(client)
    op = _ready_winner(store, measurement_id)

    _fake_tune(monkeypatch, {"sampler": "balanced", "temperature": 0.6})
    response = client.post(TUNE_URL, json={"operation_id": op.operation_id})
    assert response.status_code == 200, response.text

    _fake_depth_sweep(
        monkeypatch,
        {"off": 40.0, "adaptive": 55.0, 1: 60.0, 2: 70.0, 3: 75.0, 4: 66.0},
    )
    body = _run(client, op.operation_id)
    assert body["best"]["mtp_depth"] == 3

    preset = assistant_routes._tuning_store().get(MODEL_PATH, "a,b")
    assert preset is not None
    assert preset.mtp_depth == 3
    assert preset.sampler["sampler"] == "balanced"
    assert preset.sampler["temperature"] == 0.6


def test_a_redeploy_that_measures_no_fixed_depth_clears_a_stale_one(
    api, monkeypatch
) -> None:
    # The preset is authoritative: if the adaptive config wins the sweep, the
    # stale depth from an earlier sweep must not outlive that measurement. The
    # sampler is still kept (the sweep does not measure it).
    client, _, store = api()
    measurement_id = take_measurement(client)
    op = _ready_winner(store, measurement_id)
    assistant_routes._tuning_store().save(
        TuningPreset(
            model_signature=MODEL_PATH,
            hardware_key="a,b",
            mtp=True,
            mtp_depth=4,
            sampler={"sampler": "balanced"},
        )
    )

    _fake_depth_sweep(
        monkeypatch,
        {"off": 40.0, "adaptive": 99.0, 1: 60.0, 2: 70.0, 3: 75.0, 4: 66.0},
    )
    body = _run(client, op.operation_id)
    assert body["best"]["config"] == "mtp_on_adaptive"

    preset = assistant_routes._tuning_store().get(MODEL_PATH, "a,b")
    assert preset is not None
    assert preset.mtp_depth is None
    assert preset.sampler["sampler"] == "balanced"
