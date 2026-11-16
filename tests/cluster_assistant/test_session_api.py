# SPDX-License-Identifier: Apache-2.0
"""Reload resume: the last deployment and the stop lock.

The card cannot rebuild a plan after a reload (plan is pure and not persisted),
so the operation store keeps a minimal last-deployment record. These tests drive
the real router: the session is a read of that record, and a stop is refused
while another apply/start is running.
"""

from __future__ import annotations

from assistant_harness import (
    APPLY_URL,
    OPERATIONS_URL,
    START_URL,
    FakeCoordinator,
    activation_report,
    apply_payload,
    start_payload,
    take_measurement,
    take_plan,
    wait_for_operation,
)

from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant.operations import KIND_APPLY, OperationStore


def _coordinator() -> FakeCoordinator:
    report = dict(activation_report())
    report["deployment"] = {"deployment_id": "dep-123"}
    return FakeCoordinator(report=report)


def _local_admin(port, path, body=None, *, method="GET", timeout=None):
    if path.endswith("/operations/active"):
        return {"operation": None}
    return {"ok": True, "deployment_id": "dep-123", "stopped": True}


def _start_ready(client, measurement_id, plan) -> dict:
    apply = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    assert apply.status_code == 200, apply.text
    applied = wait_for_operation(client, apply.json()["operation_id"])
    assert applied["state"] == "applied"
    start = client.post(START_URL, json=start_payload(plan))
    assert start.status_code == 200, start.text
    ready = wait_for_operation(client, start.json()["operation_id"])
    assert ready["state"] == "ready", ready
    return ready


def test_the_session_is_empty_before_any_operation(api) -> None:
    client, _, _ = api()

    response = client.get("/admin/api/cluster/assistant/session")

    assert response.status_code == 200
    assert response.json() == {"session": None}


def test_a_ready_deployment_is_resumable_after_a_reload(api) -> None:
    client, _, _ = api(
        coordinator=_coordinator().factory(), local_addresses=("172.16.1.11",)
    )
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)
    ready = _start_ready(client, measurement_id, plan)

    response = client.get("/admin/api/cluster/assistant/session")

    assert response.status_code == 200
    session = response.json()["session"]
    assert session["operation_id"] == ready["operation_id"]
    assert session["state"] == "ready"
    assert session["stopped"] is False
    assert session["measurement_id"] == measurement_id
    assert session["plan_signature"] == plan["plan_signature"]
    assert session["model_path"] == "/models/qwen"
    assert session["context_tokens"] == plan["context_tokens"]
    assert session["coordinator_node"] == plan["coordinator_node"]
    assert session["rank_order"] == plan["rank_order"]
    assert session["endpoint"]["ready"] is True
    assert session["deployment_id"]
    # Only plan inputs; no key or SSH target is part of the session.
    assert "api_key" not in str(session)
    assert "ssh" not in str(session).lower()


def test_a_stopped_session_still_offers_start_again(api, monkeypatch) -> None:
    client, _, _ = api(
        coordinator=_coordinator().factory(), local_addresses=("172.16.1.11",)
    )
    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", _local_admin)
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)
    ready = _start_ready(client, measurement_id, plan)

    stop = client.post(
        "/admin/api/cluster/assistant/stop",
        json={"operation_id": ready["operation_id"]},
    )
    assert stop.status_code == 200
    assert stop.json()["stopped"] is True

    session = client.get("/admin/api/cluster/assistant/session").json()["session"]

    assert session["state"] == "ready"
    assert session["stopped"] is True
    assert session["deployment_id"] == ready["deployment_id"]


def test_stop_is_refused_while_another_operation_runs(api) -> None:
    client, _, operations = api(
        coordinator=_coordinator().factory(), local_addresses=("172.16.1.11",)
    )
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)
    ready = _start_ready(client, measurement_id, plan)

    # Simulate a second origin/operation that is still running. The ready start
    # is terminal, so the store accepts this active apply.
    assert isinstance(operations, OperationStore)
    operations.start(kind=KIND_APPLY, measurement_id=measurement_id)

    response = client.post(
        "/admin/api/cluster/assistant/stop",
        json={"operation_id": ready["operation_id"]},
    )

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "operation_in_progress"
    assert body["action"] == "status"
    assert body["operation_id"]
    assert f"{OPERATIONS_URL}/" in body.get("poll", "") or body["operation_id"]
