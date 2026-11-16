# SPDX-License-Identifier: Apache-2.0
"""``POST /admin/api/cluster/assistant/stop`` (Screen 6: state + stop).

Stopping is the coordinator's own unload route reached through the same
local/remote seam apply/start use. The operation record is the only input: it
carries the elected coordinator, the port and the deployment ID, so a stop never
re-elects from a measurement that may have gone stale.

``state`` stays ``ready`` (the frozen machine has no stopped value); the
``stopped`` boolean is what the card reads, and a reload cannot present a
stopped deployment as running.
"""

from __future__ import annotations

from typing import Any

from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant.operations import (
    KIND_APPLY,
    KIND_START,
    STATE_APPLYING,
    STATE_LOADING,
    STATE_READY,
    STATE_STARTING,
    OperationEndpoint,
    OperationProgress,
    OperationStore,
)
from omlx.cluster.assistant.start import RemoteAdminError

STOP_URL = "/admin/api/cluster/assistant/stop"
OPERATIONS_URL = "/admin/api/cluster/assistant/operations"


def _ready_record(
    operations: OperationStore,
    *,
    coordinator_local: bool = True,
    deployment_id: str = "dev-1.5b-abc123",
) -> Any:
    endpoint = OperationEndpoint(
        node="m5", host="172.16.1.11", port=8099
    )
    record = operations.start(
        kind=KIND_START,
        measurement_id="ms-stop",
        plan_signature="a" * 64,
        coordinator_node="m5",
        coordinator_local=coordinator_local,
        rank_order=("m5", "m1"),
        context_tokens=32768,
        endpoint=endpoint,
    )
    operations.advance(
        record.operation_id,
        STATE_STARTING,
        progress=OperationProgress(phase="starting", message="starting"),
    )
    operations.advance(
        record.operation_id,
        STATE_LOADING,
        progress=OperationProgress(phase="loading", message="loading"),
    )
    operations.advance(
        record.operation_id,
        STATE_READY,
        progress=OperationProgress(phase="ready", message="ready"),
        endpoint=OperationEndpoint(
            node="m5", host="172.16.1.11", port=8099, ready=True
        ),
        deployment_id=deployment_id,
    )
    return operations.get(record.operation_id)


def test_stop_unloads_on_the_local_coordinator_and_marks_the_record(
    api, monkeypatch
) -> None:
    client, _, operations = api()
    record = _ready_record(operations)
    calls: list[tuple[Any, ...]] = []

    def local_admin(port, path, body=None, *, method="GET", timeout=None):
        calls.append((port, path, method, timeout))
        return {"ok": True, "deployment_id": "dev-1.5b-abc123", "stopped": True}

    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", local_admin)

    response = client.post(STOP_URL, json={"operation_id": record.operation_id})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["stopped"] is True
    assert body["deployment_id"] == "dev-1.5b-abc123"
    assert calls == [
        (8099, "/admin/api/cluster/deployments/dev-1.5b-abc123/unload", "POST", 300.0)
    ]
    status = client.get(f"{OPERATIONS_URL}/{record.operation_id}").json()
    assert status["state"] == STATE_READY
    assert status["stopped"] is True
    assert "deployment_stopped" in [item["code"] for item in status["warnings"]]


def test_a_second_stop_does_not_relay_again(api, monkeypatch) -> None:
    client, _, operations = api()
    record = _ready_record(operations)
    calls: list[str] = []

    def local_admin(port, path, body=None, *, method="GET", timeout=None):
        calls.append(path)
        return {"ok": True, "stopped": True}

    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", local_admin)

    first = client.post(STOP_URL, json={"operation_id": record.operation_id})
    second = client.post(STOP_URL, json={"operation_id": record.operation_id})

    assert first.status_code == second.status_code == 200
    assert second.json()["stopped"] is True
    assert len(calls) == 1


def test_stop_relays_to_a_remote_coordinator(api, monkeypatch) -> None:
    client, _, operations = api()
    record = _ready_record(operations, coordinator_local=False)
    calls: list[tuple[Any, ...]] = []

    monkeypatch.setitem(
        assistant_routes.__dict__,
        "_paired_device_record",
        lambda node_id: {"node_id": node_id, "ssh_user": "andreaskolb"},
    )

    def remote_admin(target, port, path, body, *, method="POST", timeout=None):
        calls.append((target, port, path, method, timeout))
        return {"ok": True, "stopped": True}

    monkeypatch.setitem(assistant_routes.__dict__, "remote_admin", remote_admin)

    response = client.post(STOP_URL, json={"operation_id": record.operation_id})

    assert response.status_code == 200, response.text
    assert calls == [
        (
            "andreaskolb@172.16.1.11",
            8099,
            "/admin/api/cluster/assistant/operations/active",
            "GET",
            30.0,
        ),
        (
            "andreaskolb@172.16.1.11",
            8099,
            "/admin/api/cluster/deployments/dev-1.5b-abc123/unload",
            "POST",
            300.0,
        ),
    ]


def test_stop_without_a_paired_coordinator_asks_to_pair(api, monkeypatch) -> None:
    client, _, operations = api()
    record = _ready_record(operations, coordinator_local=False)
    monkeypatch.setitem(assistant_routes.__dict__, "_paired_device_record", lambda node_id: None)
    monkeypatch.setitem(
        assistant_routes.__dict__,
        "remote_admin",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("an unpaired coordinator must not be dialled")
        ),
    )

    response = client.post(STOP_URL, json={"operation_id": record.operation_id})

    assert response.status_code == 409
    assert response.json()["code"] == "node_not_paired"
    assert response.json()["action"] == "pair_mac"
    assert operations.get(record.operation_id).stopped is False


def test_stop_refuses_an_operation_that_is_not_ready(api) -> None:
    client, _, operations = api()
    record = operations.start(
        kind=KIND_APPLY,
        measurement_id="ms-stop",
        coordinator_node="m5",
        endpoint=OperationEndpoint(node="m5", host="172.16.1.11", port=8099),
    )
    operations.advance(
        record.operation_id,
        STATE_APPLYING,
        progress=OperationProgress(phase="staging", message="staging"),
    )

    response = client.post(STOP_URL, json={"operation_id": record.operation_id})

    assert response.status_code == 409
    assert response.json()["code"] == "operation_in_progress"
    assert response.json()["action"] == "status"


def test_stop_of_an_unknown_operation_is_404(api) -> None:
    client, _, _ = api()

    response = client.post(STOP_URL, json={"operation_id": "op-does-not-exist"})

    assert response.status_code == 404
    assert response.json()["code"] == "unknown_operation"
    assert response.json()["action"] == "status"


def test_a_vanished_deployment_counts_as_stopped(api, monkeypatch) -> None:
    client, _, operations = api()
    record = _ready_record(operations)

    def local_admin(port, path, body=None, *, method="GET", timeout=None):
        raise RemoteAdminError(404, "cluster deployment not found")

    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", local_admin)

    response = client.post(STOP_URL, json={"operation_id": record.operation_id})

    assert response.status_code == 200, response.text
    assert response.json()["stopped"] is True
    assert operations.get(record.operation_id).stopped is True


def test_a_coordinator_that_does_not_answer_keeps_the_record_running(
    api, monkeypatch
) -> None:
    client, _, operations = api()
    record = _ready_record(operations)

    def local_admin(port, path, body=None, *, method="GET", timeout=None):
        raise RemoteAdminError(0, "this Mac did not answer")

    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", local_admin)

    response = client.post(STOP_URL, json={"operation_id": record.operation_id})

    assert response.status_code == 409
    assert response.json()["code"] == "coordinator_unreachable"
    assert response.json()["action"] == "retry_coordinator"
    assert operations.get(record.operation_id).stopped is False
