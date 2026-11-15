# SPDX-License-Identifier: Apache-2.0
"""``POST /admin/api/cluster/assistant/start`` and operation status.

The rules these pin are the ones that make an approved plan a real deployment:

* nothing starts that apply did not stage, and the placement must still be the
  one apply signed;
* the supervisor starts on the *elected coordinator*, never on the app origin,
  and a remote coordinator is driven through the existing SSH seam;
* a failed coordinator start fails the operation with one action and starts
  nothing locally — there is no fallback branch to find;
* ``endpoint.ready`` is true only after health and one passed smoke completion;
* operations are one-at-a-time, monotonic and carry no throughput rate.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from assistant_harness import (
    APPLY_URL,
    GIB,
    MODEL_PATH,
    OPERATIONS_URL,
    FakeCoordinator,
    activation_report,
    apply_payload,
    staging_snapshot,
    start_payload,
    take_measurement,
    take_plan,
    wait_for_operation,
)

from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant import start as assistant_start
from omlx.cluster.assistant.operations import (
    KIND_START,
    STATE_APPLIED,
    STATE_READY,
    STATE_STARTING,
)
from omlx.cluster.assistant.start import (
    ACTIVATION_PATH,
    PLAN_PATH,
    STAGING_PATH,
    RemoteAdminError,
    coordinator_ops,
    remote_admin,
)

START_URL = "/admin/api/cluster/assistant/start"


def applied(client, fake: FakeCoordinator, *, rank0_extra_bytes: int = 0):
    """Take a measurement, approve the plan and apply it. Returns the plan."""

    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id, rank0_extra_bytes=rank0_extra_bytes)
    response = client.post(
        APPLY_URL,
        json=apply_payload(measurement_id, plan, rank0_extra_bytes=rank0_extra_bytes),
    )
    assert response.status_code == 200, response.text
    operation = wait_for_operation(client, response.json()["operation_id"])
    assert operation["state"] == STATE_APPLIED, operation
    return plan


def test_apply_refuses_a_coordinator_that_cannot_reach_a_selected_peer(api) -> None:
    fake = FakeCoordinator()
    fake.paired_nodes = []
    client, _, _ = api(coordinator=fake.factory())
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(
        APPLY_URL, json=apply_payload(measurement_id, plan)
    )
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == "failed"
    assert operation["error"]["code"] == "node_not_paired"
    assert operation["error"]["action"] == "pair_mac"
    assert fake.prepared == []


# ---------------------------------------------------------------------------
# Only an applied plan may start
# ---------------------------------------------------------------------------


def test_start_rejects_a_plan_that_was_never_applied(api) -> None:
    client, _, _ = api()
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(START_URL, json=start_payload(plan))

    assert response.status_code == 409
    assert response.json()["code"] == "operation_not_applied"
    assert response.json()["action"] == "apply"


def test_start_rejects_a_start_operation_id(api) -> None:
    fake = FakeCoordinator()
    client, _, operations = api(coordinator=fake.factory())
    plan = applied(client, fake)
    running = operations.start(
        kind=KIND_START, measurement_id=plan["measurement_id"], now=None
    )

    response = client.post(
        START_URL, json=start_payload(plan, operation_id=running.operation_id)
    )

    assert response.status_code == 409
    assert response.json()["code"] == "operation_not_applied"


def test_start_rejects_a_different_plan_signature(api) -> None:
    fake = FakeCoordinator()
    client, _, _ = api(coordinator=fake.factory())
    plan = applied(client, fake)

    response = client.post(START_URL, json=start_payload(plan, plan_signature="f" * 64))

    assert response.status_code == 409
    assert response.json()["code"] == "operation_not_applied"


def test_start_refuses_an_expired_measurement_with_the_same_flat_409(api) -> None:
    fake = FakeCoordinator()
    client, store, _ = api(coordinator=fake.factory())
    plan = applied(client, fake)
    record = store.get(plan["measurement_id"])
    assert record is not None
    from dataclasses import replace
    from datetime import UTC, datetime, timedelta

    old = datetime.now(UTC) - timedelta(minutes=30)
    store.finish(
        record.measurement_id,
        measurement=replace(record.measurement_set(), measured_at=old.isoformat()),
        endpoints=dict(record.endpoints),
    )

    response = client.post(START_URL, json=start_payload(plan))

    assert response.status_code == 409
    assert response.json()["code"] == "stale_measurement"
    assert response.json()["action"] == "measure"
    assert response.json()["reason"] == "age"
    assert "operation_id" not in response.json()


# ---------------------------------------------------------------------------
# The elected coordinator, not the origin
# ---------------------------------------------------------------------------


def test_start_launches_on_the_elected_coordinator(api) -> None:
    fake = FakeCoordinator()
    client, _, _ = api(coordinator=fake.factory())
    plan = applied(client, fake, rank0_extra_bytes=70 * GIB)
    assert plan["coordinator_node"] == "m5"  # the app origin is the M1
    assert plan["origin_node"] == "m1"

    response = client.post(START_URL, json=start_payload(plan))

    assert response.status_code == 200, response.text
    handle = response.json()
    assert handle["state"] == "starting"

    operation = wait_for_operation(client, handle["operation_id"])

    assert operation["state"] == STATE_READY
    assert operation["coordinator_node"] == "m5"
    assert operation["rank_order"] == ["m5", "m1"]
    assert operation["endpoint"]["node"] == "m5"
    assert operation["endpoint"]["ready"] is True
    assert operation["endpoint"]["url"] == "http://172.16.1.11:8099"
    assert operation["error"] is None
    assert fake.health_checks == 1
    # The coordinator received the signed request. Rank 0 is spawned as the
    # coordinator's own local child: `ClusterDeployment` requires
    # `hosts[0].ssh == "127.0.0.1"`, and the validation cluster runs as a line,
    # so the macOS Tahoe ring workaround (rank 0 over SSH to the coordinator's
    # own LAN address) is neither allowed by the contract nor needed.
    assert len(fake.activated) == 1
    activated = fake.activated[0]
    assert activated.approved_placement == fake.placement
    assert activated.hosts[0].node_id == "m5"
    assert activated.hosts[0].ssh == "127.0.0.1"
    assert [host.node_id for host in activated.hosts] == ["m5", "m1"]
    assert [node.capacity_bytes for node in activated.nodes] == [
        int(107.5 * GIB),
        int(60.0 * GIB),
    ]
    # No rate is claimed anywhere: the field exists but stays null until a
    # real post-ready measurement fills it.
    assert "tok_per_s" not in str(operation)
    assert operation["observed"] is None


def test_start_refuses_a_placement_that_changed_since_apply(api) -> None:
    fake = FakeCoordinator(placement="aaaaaaaaaaaaaaaa")
    client, _, _ = api(coordinator=fake.factory())
    plan = applied(client, fake)

    fake.placement = "bbbbbbbbbbbbbbbb"
    response = client.post(START_URL, json=start_payload(plan))

    assert response.status_code == 200, response.text
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == "failed"
    assert operation["error"]["code"] == "placement_changed"
    assert operation["error"]["action"] == "plan"
    assert fake.activated == []


# ---------------------------------------------------------------------------
# Failure means one action and no local fallback
# ---------------------------------------------------------------------------


def test_a_failed_health_check_starts_nothing(api) -> None:
    fake = FakeCoordinator(healthy=False)
    client, _, _ = api(coordinator=fake.factory())
    plan = applied(client, fake)

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == "failed"
    assert operation["error"]["code"] == "coordinator_unreachable"
    assert operation["error"]["action"] == "retry_coordinator"
    assert set(operation["error"]) == {"code", "message", "action"}
    assert fake.activated == []
    assert operation["endpoint"]["ready"] is False


def test_a_failed_coordinator_start_never_starts_local_rank_zero(
    api, monkeypatch
) -> None:
    from omlx.cluster import routes as cluster_routes

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the origin must never run the activation itself")

    monkeypatch.setattr(cluster_routes, "activate_cluster_deployment", forbidden)
    monkeypatch.setattr(cluster_routes, "prepare_cluster_deployment", forbidden)

    fake = FakeCoordinator(activate_error=RemoteAdminError(503, "launch failed"))
    client, _, _ = api(coordinator=fake.factory())
    plan = applied(client, fake)

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == "failed"
    assert operation["error"]["code"] == "coordinator_start_failed"
    assert operation["error"]["action"] == "retry_coordinator"
    assert operation["endpoint"]["ready"] is False
    # The elected Mac is the one that was asked; the origin never appears.
    assert fake.plans[0].coordinator.local is False
    assert len(fake.activated) == 1


def test_a_remote_coordinator_without_an_admin_key_is_one_action() -> None:
    """The coordinator's own key gate is reported, never bypassed."""

    with pytest.raises(RemoteAdminError) as excinfo:
        assistant_start.remote_admin(
            "172.16.1.11",
            0,
            ACTIVATION_PATH,
            {},
        )

    assert "admin port" in str(excinfo.value)


# ---------------------------------------------------------------------------
# endpoint.ready needs health and one completion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "report",
    [
        activation_report(ok=False),
        activation_report(canary=False),
        activation_report(tokens=0),
        activation_report(state="loading"),
        activation_report(ranks_ready=False),
        {"ok": True},
    ],
)
def test_endpoint_stays_not_ready_without_a_passed_smoke(api, report) -> None:
    fake = FakeCoordinator(report=report)
    client, _, _ = api(coordinator=fake.factory())
    plan = applied(client, fake)

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == "failed"
    assert operation["error"]["code"] == "readiness_unproven"
    assert operation["error"]["action"] == "diagnostics"
    assert operation["endpoint"]["ready"] is False


def test_the_ready_operation_records_that_no_rate_was_measured(api) -> None:
    fake = FakeCoordinator()
    client, _, _ = api(coordinator=fake.factory())
    plan = applied(client, fake)

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    codes = {warning["code"] for warning in operation["warnings"]}
    assert "performance_unmeasured" in codes
    warning = next(
        item
        for item in operation["warnings"]
        if item["code"] == "performance_unmeasured"
    )
    assert "no throughput was measured" in warning["message"]
    assert operation["endpoint"]["ready"] is True


# ---------------------------------------------------------------------------
# One at a time, monotonic, and observable
# ---------------------------------------------------------------------------


def test_a_second_start_while_one_runs_is_409_operation_in_progress(api) -> None:
    fake = FakeCoordinator()
    client, _, operations = api(coordinator=fake.factory())
    plan = applied(client, fake)
    running = operations.start(kind=KIND_START, measurement_id=plan["measurement_id"])

    response = client.post(START_URL, json=start_payload(plan))

    assert response.status_code == 409
    assert response.json()["code"] == "operation_in_progress"
    assert response.json()["operation_id"] == running.operation_id
    assert response.json()["action"] == "status"


def test_the_operation_state_is_monotonic_through_a_start(api) -> None:
    fake = FakeCoordinator()
    seen: list[str] = []
    client, _, operations = api(coordinator=fake.factory())
    plan = applied(client, fake)

    response = client.post(START_URL, json=start_payload(plan))
    operation_id = response.json()["operation_id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        body = client.get(f"{OPERATIONS_URL}/{operation_id}").json()
        if not seen or seen[-1] != body["state"]:
            seen.append(body["state"])
        if body["state"] in (STATE_READY, "failed"):
            break
        time.sleep(0.005)

    order = ["queued", "starting", "loading", "ready"]
    positions = [order.index(state) for state in seen if state in order]
    assert positions == sorted(positions)
    assert seen[-1] == STATE_READY


# ---------------------------------------------------------------------------
# The remote seam
# ---------------------------------------------------------------------------


def test_a_remote_coordinator_is_driven_through_the_existing_ssh_seam(
    api, monkeypatch
) -> None:
    calls: list[tuple[str, int, str, str]] = []
    # The health gate is a real SSH round trip in production; here it answers.
    monkeypatch.setattr(assistant_start, "probe_peer", lambda target: True)

    def remote(
        target: str,
        port: int,
        path: str,
        body: Any,
        *,
        method: str = "POST",
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        calls.append((target, port, path, method))
        if path == assistant_start.DEVICES_PATH:
            return {"paired": [{"node_id": "m5"}, {"node_id": "m1"}]}
        if path == assistant_start.RELOAD_PATH:
            return {"status": "ok"}
        if path == PLAN_PATH:
            return {"placement_signature": "dddddddddddddddd", "assignments": []}
        if path == STAGING_PATH:
            return {"job_id": "0123456789abcdef01234567", "status": "queued"}
        if path.startswith("/admin/api/cluster/stage/"):
            return staging_snapshot("completed")
        if path == assistant_start.PERFORMANCE_PROBE_PATH:
            return {"profiles": [], "backend": "ring"}
        if path == ACTIVATION_PATH:
            return activation_report()
        if path == assistant_start.MODELS_PATH:
            return {"models": []}
        if path == assistant_start.MIRROR_PATH:
            return {"operation_id": body["operation_id"], "running": True}
        raise AssertionError(f"unexpected admin path {path}")

    client, _, _ = api(coordinator=lambda plan: coordinator_ops(plan, remote=remote))
    plan = applied(client, FakeCoordinator())

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_READY
    # Every call went to the elected coordinator's own address and port, never
    # to loopback on the origin.
    assert {target for target, _, _, _ in calls} == {"172.16.1.11"}
    assert {port for _, port, _, _ in calls} == {8099}
    assert [path for _, _, path, _ in calls] == [
        # Apply: mirror running, do the work, mirror terminal (applied).
        assistant_start.MIRROR_PATH,
        assistant_start.DEVICES_PATH,
        # The reload decision reads the catalogue first; the model is not
        # listed yet, so the re-read before prepare stays.
        assistant_start.MODELS_PATH,
        assistant_start.RELOAD_PATH,
        PLAN_PATH,
        STAGING_PATH,
        "/admin/api/cluster/stage/0123456789abcdef01234567",
        assistant_start.PERFORMANCE_PROBE_PATH,
        assistant_start.MIRROR_PATH,
        # Start: mirror running, activate, mirror terminal (ready).
        assistant_start.MIRROR_PATH,
        assistant_start.MODELS_PATH,
        assistant_start.RELOAD_PATH,
        PLAN_PATH,
        ACTIVATION_PATH,
        # The chat profile is read after readiness so a model with no sampling
        # of its own gets usable defaults (see ensure_chat_profile).
        assistant_start.MODELS_PATH,
        assistant_start.MIRROR_PATH,
    ]
    assert {method for _, _, _, method in calls} == {"POST", "GET"}


def test_apply_through_the_ssh_seam_stages_then_stops(api) -> None:
    calls: list[str] = []

    def remote(
        target: str,
        port: int,
        path: str,
        body: Any,
        *,
        method: str = "POST",
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        calls.append(path)
        if path == assistant_start.DEVICES_PATH:
            return {"paired": [{"node_id": "m5"}, {"node_id": "m1"}]}
        if path == assistant_start.MODELS_PATH:
            return {"models": []}
        if path == assistant_start.RELOAD_PATH:
            return {"status": "ok"}
        if path == PLAN_PATH:
            return {"placement_signature": "dddddddddddddddd", "assignments": []}
        if path == STAGING_PATH:
            return {"job_id": "0123456789abcdef01234567", "status": "queued"}
        if path.startswith("/admin/api/cluster/stage/"):
            return staging_snapshot("completed")
        if path == assistant_start.PERFORMANCE_PROBE_PATH:
            return {"profiles": [], "backend": "ring"}
        if path == assistant_start.MIRROR_PATH:
            return {"operation_id": body["operation_id"], "running": True}
        raise AssertionError(path)

    client, _, _ = api(coordinator=lambda plan: coordinator_ops(plan, remote=remote))
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_APPLIED
    assert calls == [
        # The running operation is mirrored onto the coordinator before any
        # work (the cluster-wide lock), and the terminal state releases it.
        assistant_start.MIRROR_PATH,
        assistant_start.DEVICES_PATH,
        assistant_start.MODELS_PATH,
        assistant_start.RELOAD_PATH,
        PLAN_PATH,
        STAGING_PATH,
        "/admin/api/cluster/stage/0123456789abcdef01234567",
        assistant_start.PERFORMANCE_PROBE_PATH,
        assistant_start.MIRROR_PATH,
    ]
    assert ACTIVATION_PATH not in calls


def test_remote_admin_keeps_the_admin_key_on_the_coordinator(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(ssh_target, snippet, argument, **kwargs):
        seen["ssh_target"] = ssh_target
        seen["snippet"] = snippet
        seen["argument"] = argument
        return {"ok": True, "status": 200, "body": {"ok": True}}

    monkeypatch.setattr(assistant_start, "run_remote_python", fake_run)

    reply = remote_admin("172.16.1.11", 8099, PLAN_PATH, {"model_path": "/models/qwen"})

    assert reply == {"ok": True}
    assert seen["ssh_target"] == "172.16.1.11"
    # The key is read and used on the coordinator; it is never in the payload.
    assert "get_settings().auth" in seen["snippet"]
    assert "api_key" in seen["snippet"]
    assert "api_key" not in seen["argument"]
    assert "Authorization" in seen["snippet"]


def test_the_local_coordinator_uses_the_existing_local_pipeline(
    api, monkeypatch
) -> None:
    from omlx.cluster import routes as cluster_routes

    calls: list[Any] = []

    def prepare(request):
        calls.append("prepare")
        return (
            request,
            SimpleNamespace(deployment_id="deployment-1"),
            {"placement_signature": "eeeeeeeeeeeeeeee", "assignments": []},
        )

    def stage(deployment, *, source_host, parallel):
        calls.append(("stage", source_host))
        return {"job_id": "0123456789abcdef01234567", "status": "queued"}

    def status(job_id):
        calls.append("status")
        return staging_snapshot("completed")

    async def activate(request):
        calls.append("activate")
        return activation_report()

    def devices():
        calls.append("devices")
        return {"paired": [{"node_id": "m5"}, {"node_id": "m1"}]}

    monkeypatch.setattr(cluster_routes, "prepare_cluster_deployment", prepare)
    monkeypatch.setattr(cluster_routes, "start_cluster_staging", stage)
    monkeypatch.setattr(cluster_routes, "read_cluster_staging_job", status)
    monkeypatch.setattr(cluster_routes, "activate_cluster_deployment", activate)
    monkeypatch.setattr(assistant_start, "check_peers", lambda hosts: ())

    client, _, _ = api(
        coordinator=lambda plan: replace(
            coordinator_ops(plan),
            probe=None,
            devices=devices,
            reload=lambda: calls.append("reload"),
        ),
        local_addresses=("172.16.1.11",),
    )
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id, rank0_extra_bytes=70 * GIB)
    response = client.post(
        APPLY_URL,
        json=apply_payload(measurement_id, plan, rank0_extra_bytes=70 * GIB),
    )
    apply_operation = wait_for_operation(client, response.json()["operation_id"])
    assert apply_operation["state"] == STATE_APPLIED

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_READY
    assert operation["endpoint"]["ready"] is True
    assert calls == [
        "devices",
        "reload",
        "prepare",
        ("stage", "127.0.0.1"),
        "status",
        "reload",
        "prepare",
        "activate",
    ]


def test_a_start_operation_survives_a_restart_as_failed(api) -> None:
    client, _, operations = api()
    running = operations.start(
        kind=KIND_START, measurement_id="ms-1", plan_signature="a" * 64
    )
    operations.advance(running.operation_id, STATE_STARTING)
    from omlx.cluster.assistant.operations import OperationStore

    assistant_routes.configure_assistant_operations(
        OperationStore(operations.base_path)
    )

    body = client.get(f"{OPERATIONS_URL}/{running.operation_id}").json()

    assert body["state"] == "failed"
    assert body["error"]["action"] == "start"


def test_the_remote_admin_snippet_loads_settings_on_a_bare_interpreter(
    monkeypatch, tmp_path
) -> None:
    """The coordinator's ``python -c`` starts with no settings loaded.

    Reading ``get_settings()`` there raised "Settings not initialized", so every
    remote apply/start failed on the coordinator instead of calling its admin
    route. The snippet now initializes settings from that Mac's own base path.
    """

    from omlx.cluster.assistant.start import _REMOTE_ADMIN_SNIPPET

    payload = json.dumps(
        {
            "port": 1,
            "path": "/admin/api/cluster/plan",
            "method": "POST",
            "body": {"model_path": "/models/qwen"},
            "timeout": 0.5,
        }
    )
    environment = dict(os.environ)
    environment["OMLX_BASE_PATH"] = str(tmp_path)
    environment["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parents[2])

    completed = subprocess.run(
        [sys.executable, "-c", _REMOTE_ADMIN_SNIPPET, payload],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )

    output = completed.stdout + completed.stderr
    assert "Settings not initialized" not in output
    assert "URLError" in output or "Connection refused" in output


def test_start_gives_a_model_without_sampling_a_usable_chat_profile(api) -> None:
    """A fresh cluster must not need the user to fix sampling first.

    The model's own defaults (temperature 1.0, no token cap) let one factual
    question think for a thousand tokens, and a cap small enough to stop that
    truncates the answer. This fills the documented profile once, and says so.
    """

    coordinator = FakeCoordinator(
        models=[
            {"id": "qwen", "model_path": MODEL_PATH, "settings": {"temperature": None}}
        ]
    )
    client, _, _ = api(coordinator=coordinator.factory())
    plan = applied(client, coordinator)

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_READY
    assert coordinator.profile_writes == [
        (
            "qwen",
            {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 0,
                "max_tokens": 2048,
                "force_sampling": True,
            },
        )
    ]
    codes = [warning["code"] for warning in operation["warnings"]]
    assert assistant_start.WARNING_CHAT_PROFILE_APPLIED in codes


def test_start_never_overwrites_sampling_the_user_chose(api) -> None:
    coordinator = FakeCoordinator(
        models=[
            {"id": "qwen", "model_path": MODEL_PATH, "settings": {"temperature": 0.2}}
        ]
    )
    client, _, _ = api(coordinator=coordinator.factory())
    plan = applied(client, coordinator)

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_READY
    assert coordinator.profile_writes == []
    codes = {warning["code"] for warning in operation["warnings"]}
    assert assistant_start.WARNING_CHAT_PROFILE_APPLIED not in codes


def test_a_failed_chat_profile_write_does_not_unready_the_cluster(api) -> None:
    """Readiness already happened; a settings write cannot undo it."""

    coordinator = FakeCoordinator(
        models=[
            {"id": "qwen", "model_path": MODEL_PATH, "settings": {"max_tokens": None}}
        ],
        profile_error=RuntimeError("the coordinator refused the settings write"),
    )
    client, _, _ = api(coordinator=coordinator.factory())
    plan = applied(client, coordinator)

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_READY
    assert operation["endpoint"]["ready"] is True
    codes = [warning["code"] for warning in operation["warnings"]]
    assert assistant_start.WARNING_CHAT_PROFILE_SKIPPED in codes


# ---------------------------------------------------------------------------
# The reload is conditional: it unloads every loaded model on the coordinator
# ---------------------------------------------------------------------------


def test_start_does_not_reload_a_coordinator_that_already_lists_the_model(api) -> None:
    """A chat model the user is running must survive a cluster start.

    The catalogue already lists the model, so the placement can be prepared
    without the destructive re-read.
    """

    fake = FakeCoordinator()  # the default catalogue lists MODEL_PATH
    client, _, _ = api(coordinator=fake.factory())
    plan = applied(client, fake)

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_READY
    assert fake.reloads == 0


def test_start_reloads_when_the_model_is_not_listed_yet(api) -> None:
    """A model staged after apply is on disk but not in the catalogue yet."""

    fake = FakeCoordinator(models=[])
    client, _, _ = api(coordinator=fake.factory())
    plan = applied(client, fake)
    fake.reloads = 0  # count the start's re-read, not the apply's

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_READY
    assert fake.reloads == 1


def test_the_active_operation_route_reports_what_is_running(api) -> None:
    """The coordinator's answer to "are you already running an apply/start?"."""

    fake = FakeCoordinator(snapshots=[staging_snapshot("copying")])
    client, _, _ = api(coordinator=fake.factory())
    assert client.get(f"{OPERATIONS_URL}/active").json() == {"operation": None}

    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)
    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    assert response.status_code == 200, response.text
    operation_id = response.json()["operation_id"]

    active = client.get(f"{OPERATIONS_URL}/active").json()["operation"]
    assert active["operation_id"] == operation_id
    assert active["kind"] == "apply"
    assert active["running"] is True

    # "active" must not shadow the operation-status route.
    assert client.get(f"{OPERATIONS_URL}/{operation_id}").json()["state"] == (
        "applying"
    )

    # Let the job finish so the fixture does not wait for a permanent poll.
    fake.snapshots.append(staging_snapshot("completed"))
    assert wait_for_operation(client, operation_id)["state"] == STATE_APPLIED


# ---------------------------------------------------------------------------
# A saved tuning preset (per model + node config) is replayed on Start
# ---------------------------------------------------------------------------


def test_start_applies_a_saved_tuning_preset(api) -> None:
    """Tuning is expensive; a tuned model starts with its saved knobs."""

    from omlx.cluster.assistant.tuning_store import TuningPreset

    coordinator = FakeCoordinator(
        models=[
            {
                "id": "qwen",
                "model_path": MODEL_PATH,
                # top_p/top_k set so the chat-profile rule leaves sampling
                # alone; the MTP depth and temperature are still unset.
                "settings": {"top_p": 1.0, "top_k": 0},
            }
        ]
    )
    client, _, _ = api(coordinator=coordinator.factory())
    plan = applied(client, coordinator)
    assistant_routes._tuning_store().save(
        TuningPreset(
            model_signature=MODEL_PATH,
            hardware_key=",".join(sorted(plan["selected_node_ids"])),
            mtp_depth=4,
            sampler={"temperature": 0.7, "top_p": 0.95, "top_k": 40},
        )
    )

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_READY
    assert coordinator.profile_writes == [
        (
            "qwen",
            {
                "mtp_adaptive_max_depth": 4,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
            },
        )
    ]
    codes = [warning["code"] for warning in operation["warnings"]]
    assert assistant_start.WARNING_TUNING_PRESET_APPLIED in codes


def test_start_applies_the_tuning_preset_over_a_stale_setting(api) -> None:
    """The tuned winner is authoritative over a leftover sweep value."""

    from omlx.cluster.assistant.tuning_store import TuningPreset

    coordinator = FakeCoordinator(
        models=[
            {
                "id": "qwen",
                "model_path": MODEL_PATH,
                "settings": {"temperature": 0.2, "top_p": 1.0, "top_k": 0},
            }
        ]
    )
    client, _, _ = api(coordinator=coordinator.factory())
    plan = applied(client, coordinator)
    assistant_routes._tuning_store().save(
        TuningPreset(
            model_signature=MODEL_PATH,
            hardware_key=",".join(sorted(plan["selected_node_ids"])),
            mtp_depth=4,
            sampler={"temperature": 0.7, "top_p": 0.95, "top_k": 40},
        )
    )

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_READY
    # The preset is the measured winner, so it replaces the stale 0.2/1.0/0.
    assert coordinator.profile_writes == [
        (
            "qwen",
            {
                "mtp_adaptive_max_depth": 4,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
            },
        )
    ]
    codes = [warning["code"] for warning in operation["warnings"]]
    assert assistant_start.WARNING_TUNING_PRESET_APPLIED in codes


def test_start_without_a_saved_preset_writes_no_tuning(api) -> None:
    """No preset means Start behaves exactly as before."""

    coordinator = FakeCoordinator(
        models=[
            {"id": "qwen", "model_path": MODEL_PATH, "settings": {"temperature": 0.7}}
        ]
    )
    client, _, _ = api(coordinator=coordinator.factory())
    plan = applied(client, coordinator)

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_READY
    assert coordinator.profile_writes == []
    codes = {warning["code"] for warning in operation["warnings"]}
    assert assistant_start.WARNING_TUNING_PRESET_APPLIED not in codes
    assert assistant_start.WARNING_TUNING_PRESET_SKIPPED not in codes


def test_a_memory_pressure_start_names_the_cause_and_the_fix(api) -> None:
    """'does not fit ... right now' is memory pressure, not an unreachable
    coordinator; retrying without freeing memory cannot help."""

    fake = FakeCoordinator(
        activate_error=RemoteAdminError(
            503,
            "Model does not fit the current per-node memory limits; no ranks "
            "were started. rank 0 needs 71.5 GiB but can admit 31.4 GiB right now.",
        )
    )
    client, _, _ = api(coordinator=fake.factory())
    plan = applied(client, fake)

    response = client.post(START_URL, json=start_payload(plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == "failed"
    assert operation["error"]["code"] == "coordinator_start_failed"
    assert operation["error"]["action"] == "diagnostics"
    assert "still holding memory" in operation["error"]["message"]
    assert "Start again" in operation["error"]["message"]
