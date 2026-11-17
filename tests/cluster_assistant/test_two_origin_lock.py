# SPDX-License-Identifier: Apache-2.0
"""The cluster-wide apply/start lock across two origins (review B5).

A remote start drives the coordinator's *raw* cluster routes, so the origin's
operation record lives only in the origin's store: the coordinator answered
``operations/active`` with ``null`` while it was working, and a second origin
could activate concurrently. CHAPTER-5 §2.2 (E3, Option A) closes the hole by
mirroring the operation onto the coordinator through the same admin seam.

These tests pin that contract end to end, fake-driven per AGENTS.md:

* origin A's apply/start registers a running mirror on the coordinator before
  it activates; origin B's busy check sees A's operation and B's apply is
  refused with ``409 operation_in_progress`` + action ``status``;
* A's terminal mirror releases the lock and B proceeds;
* a refused mirror (the race the busy read cannot see) fails A's operation
  with the same code and action, and failure paths release the lock too;
* the release is best-effort: an unreachable coordinator never fails an
  already-applied/ready operation.

The coordinator is a real ``OperationStore`` behind a faked admin seam — the
same store its own ``operations/active`` route reads — so no test touches SSH,
a model or a Mac.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest
from assistant_harness import (
    APPLY_URL,
    MODEL_PATH,
    OPERATIONS_URL,
    START_URL,
    activation_report,
    apply_payload,
    assistant_api,
    staging_snapshot,
    start_payload,
    take_measurement,
    take_plan,
    wait_for_operation,
)

from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant import start as assistant_start
from omlx.cluster.assistant.operations import (
    KIND_APPLY,
    KIND_START,
    STATE_APPLIED,
    STATE_APPLYING,
    STATE_FAILED,
    STATE_LOADING,
    STATE_READY,
    STATE_STARTING,
    OperationInProgressError,
    OperationStore,
)
from omlx.cluster.assistant.start import (
    ACTIVATION_PATH,
    PLAN_PATH,
    STAGING_PATH,
    RemoteAdminError,
    coordinator_ops,
)

MIRROR_URL = "/admin/api/cluster/assistant/operations/mirror"
SESSION_URL = "/admin/api/cluster/assistant/session"
JOB_ID = "0123456789abcdef01234567"


class CoordinatorSeam:
    """The elected coordinator's admin API, faked over a real operation store.

    The raw cluster routes answer canned placements/staging/activation (the
    same shapes ``test_start_api``'s remote-seam tests use). The two assistant
    routes — ``operations/active`` and ``operations/mirror`` — run against a
    real ``OperationStore``, exactly the store the coordinator's own server
    reads, so the lock is exercised through the seam without a Mac.
    """

    def __init__(self, base_path: Any, *, staging_status: str = "completed") -> None:
        self.store = OperationStore(base_path / "coordinator")
        self.staging_status = staging_status
        self.calls: list[tuple[str, str]] = []
        self.mirrors: list[dict[str, Any]] = []
        #: When set, activation blocks until the event is set (a long load).
        self.activation_gate: threading.Event | None = None
        #: When true, the busy read answers "free" even while the store is
        #: locked — the race the mirror itself must still refuse.
        self.hide_active = False
        #: When set, terminal mirrors raise it (an unreachable release).
        self.release_error: BaseException | None = None

    def __call__(
        self,
        target: str,
        port: int,
        path: str,
        body: Any,
        *,
        method: str = "POST",
        timeout: float = 120.0,
    ) -> Mapping[str, Any]:
        self.calls.append((path, method))
        if path == assistant_start.ACTIVE_OPERATION_PATH:
            record = None if self.hide_active else self.store.active()
            if record is None:
                return {"operation": None}
            return {
                "operation": {
                    "operation_id": record.operation_id,
                    "kind": record.kind,
                    "state": record.state,
                    "coordinator_node": record.coordinator_node,
                    "running": True,
                }
            }
        if path == MIRROR_URL:
            assert isinstance(body, Mapping)
            self.mirrors.append(dict(body))
            if self.release_error is not None and body.get("state") in (
                STATE_APPLIED,
                STATE_READY,
                STATE_FAILED,
            ):
                raise self.release_error
            try:
                record = self.store.mirror(
                    str(body["operation_id"]),
                    str(body["kind"]),
                    str(body["coordinator_node"]),
                    str(body["state"]),
                    model_path=str(body.get("model_path") or ""),
                    context_tokens=int(body.get("context_tokens") or 0),
                )
            except OperationInProgressError as exc:
                # The seam reports the coordinator's 409 body as text, exactly
                # like the real SSH snippet does.
                raise RemoteAdminError(
                    409,
                    json.dumps(
                        {
                            "code": "operation_in_progress",
                            "message": str(exc),
                            "action": "status",
                            "operation_id": exc.operation_id,
                        }
                    ),
                ) from exc
            return {"operation_id": record.operation_id, "running": record.running}
        if path == assistant_start.DEVICES_PATH:
            return {"paired": [{"node_id": "m5"}, {"node_id": "m1"}]}
        if path == assistant_start.MODELS_PATH:
            return {
                "models": [
                    {
                        "id": "qwen",
                        "model_path": MODEL_PATH,
                        "settings": {"temperature": 0.7},
                    }
                ]
            }
        if path == assistant_start.RELOAD_PATH:
            return {"status": "ok"}
        if path == PLAN_PATH:
            return {"placement_signature": "dddddddddddddddd", "assignments": []}
        if path == STAGING_PATH:
            return {"job_id": JOB_ID, "status": "queued"}
        if path.startswith("/admin/api/cluster/stage/"):
            if self.staging_status == "failed":
                return staging_snapshot("failed", failed=("m1",))
            return staging_snapshot(self.staging_status)
        if path == assistant_start.PERFORMANCE_PROBE_PATH:
            return {"profiles": [], "backend": "ring"}
        if path == ACTIVATION_PATH:
            if self.activation_gate is not None:
                assert self.activation_gate.wait(timeout=30), "activation gate timed out"
            return activation_report()
        raise AssertionError(f"unexpected admin path {path}")


def wire_origin(stack, tmp_path, monkeypatch, seam, name):
    """Enter one origin's assistant API with its own store and the real busy read.

    Each origin gets a separate base path: two origins sharing one store would
    refuse each other locally and never exercise the coordinator's lock. The
    harness answers the busy read with "unanswerable" by default; here it is
    the real ``coordinator_active_operation`` over the faked seam.
    """

    client, store, operations = stack.enter_context(
        assistant_api(
            tmp_path / name,
            monkeypatch,
            coordinator=lambda plan: coordinator_ops(plan, remote=seam),
        )
    )
    monkeypatch.setitem(
        assistant_routes.__dict__,
        "active_operation",
        lambda plan: assistant_start.coordinator_active_operation(plan, remote=seam),
    )
    return client, store, operations


def wait_until(predicate, description: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {description}")


def drain(seam: CoordinatorSeam) -> None:
    """Let any still-running fake job finish before the clients close.

    The jobs poll staging on a worker thread; a test that fails mid-flight
    must not leave a thread polling "copying" forever (it would block the
    interpreter's exit). Completing the fake staging/activation ends them.
    """

    seam.staging_status = "completed"
    if seam.activation_gate is not None:
        seam.activation_gate.set()


def wait_for_record(store, operation_id, states, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = store.get(operation_id)
        if record is not None and record.state in states:
            return record
        time.sleep(0.01)
    raise AssertionError(f"operation {operation_id} never reached {states}")


def applied_plan(client) -> dict[str, Any]:
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)
    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    assert response.status_code == 200, response.text
    assert wait_for_operation(client, response.json()["operation_id"])["state"] == (
        STATE_APPLIED
    )
    return plan


# ---------------------------------------------------------------------------
# T1: two origins, one coordinator
# ---------------------------------------------------------------------------


def test_a_second_origin_is_refused_while_the_first_applies(
    tmp_path, monkeypatch
) -> None:
    """The B5 scenario: A's running apply must lock the coordinator for B."""

    seam = CoordinatorSeam(tmp_path, staging_status="copying")
    monkeypatch.setitem(assistant_start.__dict__, "probe_peer", lambda target: True)
    stack = contextlib.ExitStack()
    try:
        client_a, _, operations_a = wire_origin(
            stack, tmp_path, monkeypatch, seam, "origin-a"
        )
        measurement_a = take_measurement(client_a)
        plan_a = take_plan(client_a, measurement_a)
        response = client_a.post(APPLY_URL, json=apply_payload(measurement_a, plan_a))
        assert response.status_code == 200, response.text
        op_a = response.json()["operation_id"]

        # A's running mirror holds the coordinator's lock before staging runs.
        wait_until(
            lambda: (seam.store.active() or SimpleNamespace(operation_id=None))
            .operation_id
            == op_a,
            "origin A's running mirror on the coordinator",
        )
        locked = seam.store.active()
        assert locked is not None
        assert locked.mirrored is True
        assert locked.kind == KIND_APPLY
        assert locked.coordinator_node == "m5"

        # Origin B: a fresh store, the same elected coordinator.
        client_b, _, operations_b = wire_origin(
            stack, tmp_path, monkeypatch, seam, "origin-b"
        )
        measurement_b = take_measurement(client_b)
        plan_b = take_plan(client_b, measurement_b)
        plans_before = sum(path == PLAN_PATH for path, _ in seam.calls)
        refused = client_b.post(APPLY_URL, json=apply_payload(measurement_b, plan_b))

        assert refused.status_code == 409, refused.text
        body = refused.json()
        assert body["code"] == "operation_in_progress"
        assert body["action"] == "status"
        assert body["operation_id"] == op_a
        # Nothing was created on B and nothing was staged for B.
        assert operations_b.active() is None
        assert operations_b.list() == ()
        assert sum(path == PLAN_PATH for path, _ in seam.calls) == plans_before

        # A finishes: the terminal mirror releases the lock.
        seam.staging_status = "completed"
        applied = wait_for_record(operations_a, op_a, (STATE_APPLIED,))
        assert applied.state == STATE_APPLIED
        wait_until(
            lambda: seam.store.active() is None,
            "the coordinator's lock to release",
        )
        mirror = seam.store.get(op_a)
        assert mirror is not None
        assert mirror.state == STATE_APPLIED
        assert mirror.mirrored is True
        # The mirror is a lock marker, not the coordinator's own session.
        assert seam.store.latest_session() is None

        # B proceeds now.
        response_b = client_b.post(APPLY_URL, json=apply_payload(measurement_b, plan_b))
        assert response_b.status_code == 200, response_b.text
        op_b = response_b.json()["operation_id"]
        assert wait_for_operation(client_b, op_b)["state"] == STATE_APPLIED
        assert op_b != op_a
    finally:
        drain(seam)
        stack.close()


def test_a_second_origin_is_refused_while_the_first_starts(
    tmp_path, monkeypatch
) -> None:
    """The same lock for a start: the mirror lands before activation."""

    seam = CoordinatorSeam(tmp_path)
    gate = threading.Event()
    seam.activation_gate = gate
    monkeypatch.setitem(assistant_start.__dict__, "probe_peer", lambda target: True)
    stack = contextlib.ExitStack()
    try:
        client_a, _, operations_a = wire_origin(
            stack, tmp_path, monkeypatch, seam, "origin-a"
        )
        plan_a = applied_plan(client_a)
        response = client_a.post(START_URL, json=start_payload(plan_a))
        assert response.status_code == 200, response.text
        op_a = response.json()["operation_id"]

        # The running start mirror holds the lock while activation is held.
        wait_until(
            lambda: (seam.store.active() or SimpleNamespace(operation_id=None))
            .operation_id
            == op_a,
            "origin A's running start mirror",
        )
        locked = seam.store.active()
        assert locked is not None
        assert locked.kind == KIND_START
        assert locked.state == STATE_STARTING
        # The mirror landed before the coordinator was asked to activate.
        wait_until(
            lambda: any(path == ACTIVATION_PATH for path, _ in seam.calls),
            "the coordinator to be asked to activate",
        )
        mirror_index = next(
            index for index, (path, _) in enumerate(seam.calls) if path == MIRROR_URL
        )
        activation_index = next(
            index
            for index, (path, _) in enumerate(seam.calls)
            if path == ACTIVATION_PATH
        )
        assert mirror_index < activation_index

        client_b, _, operations_b = wire_origin(
            stack, tmp_path, monkeypatch, seam, "origin-b"
        )
        measurement_b = take_measurement(client_b)
        plan_b = take_plan(client_b, measurement_b)
        refused = client_b.post(APPLY_URL, json=apply_payload(measurement_b, plan_b))
        assert refused.status_code == 409, refused.text
        assert refused.json()["code"] == "operation_in_progress"
        assert refused.json()["action"] == "status"
        assert refused.json()["operation_id"] == op_a
        assert operations_b.list() == ()

        # A's activation finishes; the ready mirror releases the lock.
        gate.set()
        ready = wait_for_record(operations_a, op_a, (STATE_READY,))
        assert ready.state == STATE_READY
        wait_until(
            lambda: seam.store.active() is None,
            "the coordinator's lock to release after ready",
        )
        assert seam.store.get(op_a).state == STATE_READY

        response_b = client_b.post(APPLY_URL, json=apply_payload(measurement_b, plan_b))
        assert response_b.status_code == 200, response_b.text
        assert wait_for_operation(client_b, response_b.json()["operation_id"])[
            "state"
        ] == STATE_APPLIED
    finally:
        drain(seam)
        stack.close()


def test_a_refused_mirror_fails_the_origin_operation(tmp_path, monkeypatch) -> None:
    """The race the busy read cannot see: the mirror itself is the guard.

    The busy check answers "free" (unanswerable is not a lock), but another
    origin already holds the coordinator's store. The running mirror is
    refused, and the origin's operation fails with the documented code and
    one action — it never stages.
    """

    seam = CoordinatorSeam(tmp_path)
    seam.store.mirror("op-other-origin", KIND_START, "m5", STATE_LOADING)
    seam.hide_active = True
    monkeypatch.setitem(assistant_start.__dict__, "probe_peer", lambda target: True)
    stack = contextlib.ExitStack()
    try:
        client, _, operations = wire_origin(
            stack, tmp_path, monkeypatch, seam, "origin-a"
        )
        measurement_id = take_measurement(client)
        plan = take_plan(client, measurement_id)
        response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
        assert response.status_code == 200, response.text
        record = wait_for_record(
            operations, response.json()["operation_id"], (STATE_FAILED,)
        )

        assert record.error is not None
        assert record.error.code == "operation_in_progress"
        assert record.error.action == "status"
        assert "op-other-origin" in record.error.message
        # Nothing was staged, and the other origin's lock is untouched.
        assert STAGING_PATH not in [path for path, _ in seam.calls]
        assert seam.store.active().operation_id == "op-other-origin"
    finally:
        drain(seam)
        stack.close()


def test_a_failed_apply_releases_the_coordinator_lock(tmp_path, monkeypatch) -> None:
    """Failure paths release too: a staging failure must not wedge the lock."""

    seam = CoordinatorSeam(tmp_path, staging_status="failed")
    monkeypatch.setitem(assistant_start.__dict__, "probe_peer", lambda target: True)
    stack = contextlib.ExitStack()
    try:
        client_a, _, operations_a = wire_origin(
            stack, tmp_path, monkeypatch, seam, "origin-a"
        )
        measurement_a = take_measurement(client_a)
        plan_a = take_plan(client_a, measurement_a)
        response = client_a.post(APPLY_URL, json=apply_payload(measurement_a, plan_a))
        assert response.status_code == 200, response.text
        op_a = response.json()["operation_id"]
        failed = wait_for_record(operations_a, op_a, (STATE_FAILED,))
        assert failed.error.code == "staging_failed"

        wait_until(
            lambda: seam.store.active() is None,
            "the failed apply to release the coordinator's lock",
        )
        assert seam.store.get(op_a).state == STATE_FAILED

        client_b, _, _ = wire_origin(stack, tmp_path, monkeypatch, seam, "origin-b")
        measurement_b = take_measurement(client_b)
        plan_b = take_plan(client_b, measurement_b)
        seam.staging_status = "completed"
        response_b = client_b.post(APPLY_URL, json=apply_payload(measurement_b, plan_b))
        assert response_b.status_code == 200, response_b.text
        assert wait_for_operation(client_b, response_b.json()["operation_id"])[
            "state"
        ] == STATE_APPLIED
    finally:
        drain(seam)
        stack.close()


def test_an_unreachable_release_never_fails_the_operation(
    tmp_path, monkeypatch, caplog
) -> None:
    """Release is best-effort: the cluster is applied even if the lock lingers.

    The stale-mirror TTL is the backstop for a release that never arrives; a
    coordinator that cannot be reached for the release logs a warning instead
    of failing an operation that already finished.
    """

    seam = CoordinatorSeam(tmp_path)
    seam.release_error = RemoteAdminError(0, "the coordinator did not answer")
    monkeypatch.setitem(assistant_start.__dict__, "probe_peer", lambda target: True)
    stack = contextlib.ExitStack()
    try:
        client, _, operations = wire_origin(
            stack, tmp_path, monkeypatch, seam, "origin-a"
        )
        measurement_id = take_measurement(client)
        plan = take_plan(client, measurement_id)
        with caplog.at_level(logging.WARNING):
            response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
            assert response.status_code == 200, response.text
            record = wait_for_record(
                operations, response.json()["operation_id"], (STATE_APPLIED,)
            )

        assert record.state == STATE_APPLIED
        assert record.error is None
        # The running mirror landed; the terminal one could not be delivered.
        assert seam.store.get(record.operation_id).state == STATE_APPLYING
        assert any(
            item.name == "omlx.cluster.assistant.start"
            and item.levelno == logging.WARNING
            and record.operation_id in item.getMessage()
            for item in caplog.records
        )
    finally:
        drain(seam)
        stack.close()


# ---------------------------------------------------------------------------
# The mirror route itself (in-process, real router)
# ---------------------------------------------------------------------------


def test_the_mirror_route_takes_and_releases_the_lock(api) -> None:
    client, _, operations = api()
    body = {
        "operation_id": "op-origin-a",
        "kind": "apply",
        "coordinator_node": "m5",
        "state": "applying",
        "model_path": MODEL_PATH,
        "context_tokens": 8192,
    }

    response = client.post(MIRROR_URL, json=body)
    assert response.status_code == 200, response.text
    assert response.json() == {"operation_id": "op-origin-a", "running": True}

    active = client.get(f"{OPERATIONS_URL}/active").json()["operation"]
    assert active["operation_id"] == "op-origin-a"
    assert active["kind"] == "apply"
    assert active["running"] is True

    # A second origin's running mirror gets the same 409 shape as every other
    # operation_in_progress refusal.
    second = client.post(MIRROR_URL, json={**body, "operation_id": "op-origin-b"})
    assert second.status_code == 409
    assert second.json()["code"] == "operation_in_progress"
    assert second.json()["action"] == "status"
    assert second.json()["operation_id"] == "op-origin-a"

    # Mirrors are lock markers, not this Mac's sessions.
    assert client.get(SESSION_URL).json() == {"session": None}

    # The terminal mirror releases the lock.
    done = client.post(MIRROR_URL, json={**body, "state": "applied"})
    assert done.status_code == 200, done.text
    assert done.json() == {"operation_id": "op-origin-a", "running": False}
    assert client.get(f"{OPERATIONS_URL}/active").json() == {"operation": None}
    stored = operations.get("op-origin-a")
    assert stored is not None
    assert stored.mirrored is True
    assert stored.model_path == MODEL_PATH
    assert stored.context_tokens == 8192


def test_the_mirror_route_rejects_an_invalid_body(api) -> None:
    client, _, _ = api()
    body = {
        "operation_id": "op-origin-a",
        "kind": "apply",
        "coordinator_node": "m5",
        "state": "applying",
    }

    unknown_state = client.post(MIRROR_URL, json={**body, "state": "done"})
    assert unknown_state.status_code == 400
    assert unknown_state.json()["code"] == "invalid_mirror_request"
    unknown_kind = client.post(MIRROR_URL, json={**body, "kind": "deploy"})
    assert unknown_kind.status_code == 400
    assert unknown_kind.json()["code"] == "invalid_mirror_request"
    # The schema itself is strict: missing and extra fields are a 422.
    assert client.post(MIRROR_URL, json={}).status_code == 422
    assert client.post(MIRROR_URL, json={**body, "extra": 1}).status_code == 422


def test_mirror_is_not_read_as_an_operation_id(api) -> None:
    """The mirror route is declared before /operations/{operation_id}."""

    client, _, _ = api()
    # GET still resolves "mirror" as an (unknown) operation id: 404, not 405.
    assert client.get(f"{OPERATIONS_URL}/mirror").status_code == 404
    response = client.post(
        MIRROR_URL,
        json={
            "operation_id": "op-origin-a",
            "kind": "start",
            "coordinator_node": "m5",
            "state": "starting",
        },
    )
    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# The seam wiring
# ---------------------------------------------------------------------------


def test_remote_ops_wire_the_mirror_through_the_admin_seam() -> None:
    assert assistant_start.MIRROR_PATH == MIRROR_URL
    calls: list[tuple[str, int, str, Any]] = []

    def admin(target, port, path, body, **kwargs):
        calls.append((target, port, path, body))
        return {"operation_id": body["operation_id"], "running": True}

    ops = assistant_start._remote_ops("172.16.1.11", 8099, admin)
    assert ops.mirror is not None
    body = {
        "operation_id": "op-origin-a",
        "kind": "apply",
        "coordinator_node": "m5",
        "state": "applying",
        "model_path": MODEL_PATH,
        "context_tokens": 8192,
    }
    reply = ops.mirror(body)

    assert reply == {"operation_id": "op-origin-a", "running": True}
    assert calls == [("172.16.1.11", 8099, MIRROR_URL, body)]


def test_the_local_ops_leave_the_mirror_unset() -> None:
    """A local coordinator shares the store; ``store.start`` already refuses."""

    plan = SimpleNamespace(
        endpoint=None,
        coordinator=SimpleNamespace(local=True),
        rank_order=("m5", "m1"),
        targets={},
    )
    ops = assistant_start._local_ops(plan)
    assert ops.local is True
    assert ops.mirror is None


def _ops_with_mirror(exc: Exception):
    class _Ops:
        target = "172.16.1.12"

        def mirror(self, body):
            raise exc

    return _Ops()


def test_a_coordinator_without_the_mirror_route_does_not_block_the_apply(tmp_path) -> None:
    """An older coordinator build has no lock route; a mixed-version cluster
    must still deploy. The absence is a log line, not a fake lock."""

    from omlx.cluster.assistant.operations import KIND_APPLY

    store = assistant_start.OperationStore(tmp_path / "store")
    op = store.start(kind=KIND_APPLY, measurement_id="ms-1", coordinator_node="m5")

    for status in (404, 405):
        refusal = assistant_start._mirror_running(
            store,
            op.operation_id,
            _ops_with_mirror(assistant_start.RemoteAdminError(status, "{}")),
            STATE_APPLYING,
        )
        assert refusal is None, status


def test_a_500_from_the_mirror_still_propagates(tmp_path) -> None:
    """Only a missing route is tolerated; a real coordinator failure is not."""

    from omlx.cluster.assistant.operations import KIND_APPLY

    store = assistant_start.OperationStore(tmp_path / "store")
    op = store.start(kind=KIND_APPLY, measurement_id="ms-1", coordinator_node="m5")

    with pytest.raises(assistant_start.RemoteAdminError):
        assistant_start._mirror_running(
            store,
            op.operation_id,
            _ops_with_mirror(assistant_start.RemoteAdminError(500, "boom")),
            STATE_APPLYING,
        )
