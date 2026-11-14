# SPDX-License-Identifier: Apache-2.0
"""``POST /admin/api/cluster/assistant/apply``.

These drive the real router with a ``TestClient``: a measurement is taken
through ``/measure`` (or seeded directly when a test controls the facts), the
plan is built by ``/plan``, and ``apply`` is asked to stage it. Everything that
would touch another Mac is a fake, so the suite never reaches SSH, a model on
disk or the validation cluster.

They pin what apply must never get wrong:

* the approved ``plan_signature`` is re-proved by re-electing, never trusted, and
  a stale, mismatched or single-Mac plan is refused with one action;
* apply stages and starts nothing — no activation reaches the fakes;
* one apply at a time, and the refusal names the running operation;
* the operation handle is pollable, persists, and carries no throughput rate.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

from assistant_harness import (
    APPLY_URL,
    GIB,
    MODEL_PATH,
    OPERATIONS_URL,
    SSH,
    UNITS,
    FakeCoordinator,
    apply_payload,
    seed_measurement,
    staging_snapshot,
    take_measurement,
    take_plan,
    wait_for_operation,
)
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from omlx.cluster.assistant import provisioning
from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant.election import ElectionResult
from omlx.cluster.assistant.operations import (
    ACTION_APPLY,
    ACTION_FREE_DISK_SPACE,
    ACTION_RETRY_COORDINATOR,
    KIND_APPLY,
    STATE_APPLIED,
    OperationStore,
)
from omlx.cluster.assistant.state import MeasurementStore, ModelIdentity

# ---------------------------------------------------------------------------
# The measurement contract, unchanged from /plan

# ---------------------------------------------------------------------------


def test_apply_refuses_an_unknown_measurement_like_plan(api) -> None:
    client, _, _ = api()

    response = client.post(APPLY_URL, json=apply_payload("ms-missing", {}))

    assert response.status_code == 409
    refusal = response.json()
    assert refusal["code"] == "stale_measurement"
    assert refusal["action"] == "measure"
    assert refusal["reason"] == "unknown"
    assert refusal["measurement_id"] == "ms-missing"
    assert "operation_id" not in refusal
    assert "state" not in refusal


def test_apply_refuses_a_running_or_failed_measurement(api) -> None:
    client, store, _ = api()
    running = store.start(
        model=ModelIdentity(model_path=MODEL_PATH), node_ids=["m5", "m1"]
    )
    body = apply_payload(running.measurement_id, {})

    response = client.post(APPLY_URL, json=body)
    assert response.status_code == 409
    assert response.json()["reason"] == "state"
    assert response.json()["action"] == "measure"

    store.fail(running.measurement_id, code="measurement_failed", message="no")
    response = client.post(APPLY_URL, json=body)
    assert response.status_code == 409
    assert response.json()["reason"] == "state"


def test_apply_refuses_a_stale_measurement(api) -> None:
    client, store, _ = api()
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)
    record = store.get(measurement_id)
    assert record is not None
    old = datetime.now(UTC) - timedelta(minutes=30)
    store.finish(
        measurement_id,
        measurement=replace(record.measurement_set(), measured_at=old.isoformat()),
        endpoints=dict(record.endpoints),
    )

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))

    assert response.status_code == 409
    assert response.json()["code"] == "stale_measurement"
    assert response.json()["reason"] == "age"
    assert response.json()["action"] == "measure"


def test_apply_refuses_a_different_model_mtp_or_node_set(api) -> None:
    client, _, _ = api()
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    other_model = client.post(
        APPLY_URL,
        json=apply_payload(measurement_id, plan, model_path="/models/other"),
    )
    assert other_model.status_code == 409
    assert other_model.json()["reason"] == "model"

    other_mtp = client.post(
        APPLY_URL, json=apply_payload(measurement_id, plan, mtp=True)
    )
    assert other_mtp.status_code == 409
    assert other_mtp.json()["reason"] == "mtp"

    other_nodes = client.post(
        APPLY_URL,
        json=apply_payload(measurement_id, plan, rank_order=["m5", "m9"]),
    )
    assert other_nodes.status_code == 409
    assert other_nodes.json()["reason"] == "node_set"
    assert other_nodes.json()["action"] == "measure"


# ---------------------------------------------------------------------------
# The approved plan is re-proved, never trusted
# ---------------------------------------------------------------------------


def test_a_wrong_plan_signature_is_refused_with_one_action(api) -> None:
    client, _, _ = api()
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(
        APPLY_URL, json=apply_payload(measurement_id, plan, plan_signature="0" * 64)
    )

    assert response.status_code == 409
    assert response.json()["code"] == "plan_mismatch"
    assert response.json()["action"] == "plan"
    assert "plan again" in response.json()["message"]


def test_a_wrong_coordinator_or_rank_order_is_refused(api) -> None:
    client, _, _ = api()
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id, rank0_extra_bytes=70 * GIB)
    assert plan["coordinator_node"] == "m5"

    wrong_coordinator = client.post(
        APPLY_URL, json=apply_payload(measurement_id, plan, coordinator_node="m1")
    )
    assert wrong_coordinator.status_code == 409
    assert wrong_coordinator.json()["code"] == "plan_mismatch"

    wrong_order = client.post(
        APPLY_URL, json=apply_payload(measurement_id, plan, rank_order=["m1", "m5"])
    )
    assert wrong_order.status_code == 409
    assert wrong_order.json()["code"] == "plan_mismatch"


def test_a_single_mac_plan_is_refused_with_add_mac(api) -> None:
    client, _, _ = api()
    measurement_id = take_measurement(client, nodes=[UNITS[0]])
    plan = take_plan(client, measurement_id)
    assert plan["feasible"] is True
    assert plan["rank_order"] == ["m5"]

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))

    assert response.status_code == 409
    assert response.json()["code"] == "single_node_plan"
    assert response.json()["action"] == "add_mac"


def test_an_unpaired_mac_is_one_action_not_a_guess(api) -> None:
    def device(node_id: str) -> dict[str, Any] | None:
        return (
            {"last_addrs": [SSH["m5"]], "http_port": 8099} if node_id == "m5" else None
        )

    client, _, _ = api(device=device)
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id, rank0_extra_bytes=70 * GIB)

    response = client.post(
        APPLY_URL,
        json=apply_payload(measurement_id, plan, rank0_extra_bytes=70 * GIB),
    )

    assert response.status_code == 409
    assert response.json()["code"] == "node_not_paired"
    assert response.json()["action"] == "pair_mac"


def test_an_unpaired_coordinator_asks_for_the_pairing(api) -> None:
    """An unpaired Mac has no address *because* it is unpaired.

    Telling the user to bind an interface would send them to the wrong control;
    the one next step is the pairing flow (PROJECT-BRIEF §13.2).
    """

    client, _, _ = api(device=lambda node_id: None)
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))

    assert response.status_code == 409
    assert response.json()["code"] == "node_not_paired"
    assert response.json()["action"] == "pair_mac"


def test_a_paired_coordinator_without_a_dialable_address_is_configured(api) -> None:
    """Paired but no usable address is an interface problem, not a pairing one."""

    client, _, _ = api(
        device=lambda node_id: ({"last_addrs": [], "http_port": 8099})
    )
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))

    assert response.status_code == 409
    assert response.json()["code"] == "coordinator_unresolved"
    assert response.json()["action"] == "configure_endpoint"


# ---------------------------------------------------------------------------
# apply stages, and starts nothing
# ---------------------------------------------------------------------------


def test_apply_stages_and_never_starts_inference(api) -> None:
    fake = FakeCoordinator()
    client, _, _ = api(coordinator=fake.factory())
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))

    assert response.status_code == 200, response.text
    handle = response.json()
    assert handle["state"] == "applying"
    assert handle["poll"] == f"{OPERATIONS_URL}/{handle['operation_id']}"

    operation = wait_for_operation(client, handle["operation_id"])

    assert operation["state"] == STATE_APPLIED
    assert operation["kind"] == KIND_APPLY
    assert operation["coordinator_node"] == "m5"
    assert operation["rank_order"] == ["m5", "m1"]
    assert operation["progress"]["phase"] == "applied"
    assert operation["progress"]["nodes_ready"] == 2
    assert operation["error"] is None
    assert operation["placement_signature"] == fake.placement
    # No inference: the activation half of the pipeline was never called.
    assert fake.activated == []
    # The placement the coordinator signed is what staging was asked to copy.
    request, source_host, prepared = fake.staged[0]
    assert request.approved_placement == fake.placement
    assert prepared.placement_signature == fake.placement
    # The coordinator already holds the checkpoint, so the source is local to
    # it — not an assumption about the app origin.
    assert source_host == "127.0.0.1"
    # The endpoint is known but not ready: nothing has been started or proven.
    assert operation["endpoint"]["host"] == "172.16.1.11"
    assert operation["endpoint"]["port"] == 8099
    assert operation["endpoint"]["ready"] is False
    # No throughput is claimed anywhere: the field is null until a real
    # post-ready measurement fills it.
    assert operation["observed"] is None
    assert "tok_per_s" not in str(operation)


def test_apply_reports_a_placement_the_coordinator_recomputed(api) -> None:
    fake = FakeCoordinator(assignments=[{"rank": 0, "start_layer": 1, "end_layer": 9}])
    client, _, _ = api(coordinator=fake.factory())
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    handle = client.post(APPLY_URL, json=apply_payload(measurement_id, plan)).json()
    operation = wait_for_operation(client, handle["operation_id"])

    codes = {warning["code"] for warning in operation["warnings"]}
    assert codes == {"placement_recomputed"}


def test_apply_fails_with_one_action_when_staging_fails(api) -> None:
    fake = FakeCoordinator(
        snapshots=[staging_snapshot("failed", ready=("m5",), failed=("m1",))]
    )
    client, _, _ = api(coordinator=fake.factory())
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    handle = client.post(APPLY_URL, json=apply_payload(measurement_id, plan)).json()
    operation = wait_for_operation(client, handle["operation_id"])

    assert operation["state"] == "failed"
    assert operation["error"]["code"] == "staging_failed"
    assert operation["error"]["action"] == "apply"
    assert "m1" in operation["error"]["message"]
    assert set(operation["error"]) == {"code", "message", "action"}
    assert fake.activated == []


def test_apply_uses_a_peer_as_the_model_source_when_the_coordinator_lacks_it(
    api,
) -> None:
    """§9.6: the coordinator is not assumed to hold the model."""

    def device(node_id: str) -> dict[str, Any] | None:
        return {"last_addrs": [SSH[node_id]], "ssh_user": "omlx", "http_port": 8099}

    fake = FakeCoordinator()
    client, store, _ = api(coordinator=fake.factory(), device=device)
    measurement_id = seed_measurement(store, present=("m1",))
    plan = take_plan(client, measurement_id, rank0_extra_bytes=70 * GIB)
    assert plan["coordinator_node"] == "m5"

    response = client.post(
        APPLY_URL,
        json=apply_payload(measurement_id, plan, rank0_extra_bytes=70 * GIB),
    )
    assert response.status_code == 200, response.text
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_APPLIED
    _, source_host, _ = fake.staged[0]
    # The coordinator reaches the M1 through its paired SSH target.
    assert source_host == "omlx@172.16.1.14"


def test_apply_does_not_reload_a_coordinator_that_already_lists_the_model(api) -> None:
    """A reload unloads every loaded model on the coordinator.

    A chat model the user is running in parallel must not be evicted just
    because a cluster is being prepared. When the catalogue already lists the
    model, the settings write went through its admin route and prepare sees it
    without a re-read.
    """

    fake = FakeCoordinator()  # the default catalogue lists MODEL_PATH
    client, _, _ = api(coordinator=fake.factory())
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_APPLIED
    assert fake.reloads == 0


def test_apply_reloads_when_the_model_is_not_listed_yet(api) -> None:
    """A checkpoint staged outside the pool is on disk but not in the catalogue.

    Only then is the destructive re-read needed, so prepare sees the model and
    the settings file written for it.
    """

    fake = FakeCoordinator(models=[])
    client, _, _ = api(coordinator=fake.factory())
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_APPLIED
    assert fake.reloads == 1


def test_apply_reloads_when_the_catalogue_cannot_be_read(api) -> None:
    """An unreadable catalogue cannot prove the model is listed.

    The re-read then stays (the old behaviour), so prepare still sees a freshly
    staged model; a coordinator that cannot answer fails the reload honestly.
    """

    fake = FakeCoordinator(models_error=RuntimeError("the catalogue did not answer"))
    client, _, _ = api(coordinator=fake.factory())
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == STATE_APPLIED
    assert fake.reloads == 1


# ---------------------------------------------------------------------------
# Jobs, handles and refusals
# ---------------------------------------------------------------------------


def test_a_second_apply_while_one_runs_is_409_operation_in_progress(api) -> None:
    client, _, operations = api()
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)
    running = operations.start(
        kind=KIND_APPLY, measurement_id=measurement_id, plan_signature="a" * 64
    )

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))

    assert response.status_code == 409
    assert response.json()["code"] == "operation_in_progress"
    assert response.json()["operation_id"] == running.operation_id
    assert response.json()["action"] == "status"


def test_the_operation_status_route_reports_and_404s(api) -> None:
    client, _, operations = api()
    running = operations.start(
        kind=KIND_APPLY, measurement_id="ms-1", plan_signature="a" * 64
    )

    found = client.get(f"{OPERATIONS_URL}/{running.operation_id}")

    assert found.status_code == 200
    body = found.json()
    assert body["operation_id"] == running.operation_id
    assert body["state"] == "queued"
    assert body["endpoint"] is None

    missing = client.get(f"{OPERATIONS_URL}/op-does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["code"] == "operation_not_found"
    assert missing.json()["action"] == "apply"


def test_a_running_operation_is_reported_failed_after_a_restart(api) -> None:
    client, _, operations = api()
    running = operations.start(
        kind=KIND_APPLY, measurement_id="ms-1", plan_signature="a" * 64
    )
    # Simulate the process restart: the same file, a new store.
    assistant_routes.configure_assistant_operations(
        OperationStore(operations.base_path)
    )

    body = client.get(f"{OPERATIONS_URL}/{running.operation_id}").json()

    assert body["state"] == "failed"
    assert body["error"]["code"] == "operation_interrupted"
    assert body["error"]["action"] == "apply"


def test_apply_requires_the_plan_time_inputs_and_nothing_else(api) -> None:
    client, _, _ = api()
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)
    body = apply_payload(measurement_id, plan)

    without_measurement = dict(body)
    del without_measurement["measurement_id"]
    assert client.post(APPLY_URL, json=without_measurement).status_code == 422

    body["stages"] = [{"rank": 0, "start_layer": 0, "end_layer": 20}]
    assert client.post(APPLY_URL, json=body).status_code == 422


def test_apply_is_admin_gated_by_the_mount_not_by_the_router(api) -> None:
    def deny() -> None:
        raise HTTPException(401, "admin required")

    app = FastAPI()
    app.include_router(assistant_routes.router, dependencies=[Depends(deny)])

    response = TestClient(app).post(APPLY_URL, json=apply_payload("ms-1", {}))

    assert response.status_code == 401


def test_the_default_seams_read_the_addresses_discovery_reports(monkeypatch) -> None:
    """A local coordinator is proven by address, so the seam must return them.

    ``local_addr_dicts`` reports ``{"ip": …}``; reading another key made the set
    empty, so "this Mac is the coordinator" could never be true and apply
    refused the ordinary local case with ``invalid SSH target: ''``.
    """

    monkeypatch.setattr(
        "omlx.cluster.discovery.local_addr_dicts",
        lambda: [{"ip": "10.0.0.5", "if_type": "lan"}, {"ip": "10.0.0.6"}],
    )

    seams = provisioning.default_provisioning_seams()

    assert list(seams.local_addresses()) == ["10.0.0.5", "10.0.0.6"]


def test_the_origin_mac_is_dialable_when_another_mac_is_rank_zero(tmp_path) -> None:
    """The origin can be a worker, so the coordinator must be able to reach it.

    A Mac has no paired record of itself, so its own target used to stay empty
    and apply refused with ``node_not_paired`` — the brief's "the origin can be
    the smallest Mac" case (§9.5). Its measured address is the evidence.
    """

    store = MeasurementStore(tmp_path / "store")
    measurement_id = seed_measurement(store)
    record = store.get(measurement_id)
    assert record is not None
    result = ElectionResult(
        feasible=True,
        coordinator_node="m1",
        rank_order=("m1", "m5"),
        endpoint=None,
        selected=None,
        considered=(),
        rejected=(),
        policy=None,
        origin_node="m5",
        plan_signature="a" * 64,
        reasons=(),
    )
    seams = provisioning.ProvisioningSeams(
        device_record=lambda node_id: (
            None if node_id == "m5" else {"last_addrs": [SSH[node_id]]}
        ),
        local_addresses=lambda: (SSH["m5"],),
    )

    targets = provisioning.resolve_targets(record, result, seams)

    assert targets["m5"].local is True
    assert targets["m5"].ssh == SSH["m5"]
    assert targets["m1"].local is False


def test_a_local_coordinator_still_runs_over_loopback(tmp_path) -> None:
    """Rank 0 on this Mac stays local; the origin-address rule must not leak."""

    store = MeasurementStore(tmp_path / "store")
    measurement_id = seed_measurement(store)
    record = store.get(measurement_id)
    assert record is not None
    result = ElectionResult(
        feasible=True,
        coordinator_node="m5",
        rank_order=("m5", "m1"),
        endpoint=None,
        selected=None,
        considered=(),
        rejected=(),
        policy=None,
        origin_node="m5",
        plan_signature="a" * 64,
        reasons=(),
    )
    seams = provisioning.ProvisioningSeams(
        device_record=lambda node_id: (
            None if node_id == "m5" else {"last_addrs": [SSH[node_id]]}
        ),
        local_addresses=lambda: (SSH["m5"],),
    )

    targets = provisioning.resolve_targets(record, result, seams)

    assert targets["m5"].local is True
    assert targets["m5"].ssh == ""


def test_an_unpaired_discovered_mac_is_never_a_launch_target(monkeypatch) -> None:
    """Discovery is not trust: only a paired Mac may receive a rank.

    The registry also lists unpaired Macs, with whatever addresses the network
    announced — including IPv6 link-local ones. Using that row as a launch
    target made apply try to SSH to `fe80::…` (``invalid SSH target``) instead
    of asking the user to pair the Mac.
    """

    class Registry:
        def __init__(self) -> None:
            self.rows = {
                "unpaired": {"node_id": "unpaired", "last_addrs": ["fe80::1"]},
            }

        def get(self, node_id: str):
            return self.rows.get(node_id)

        def is_paired(self, node_id: str) -> bool:
            return False

    monkeypatch.setattr(
        "omlx.cluster.registry.get_device_registry", lambda: Registry()
    )

    seams = provisioning.default_provisioning_seams()

    assert seams.device_record("unpaired") is None


def test_a_link_local_address_never_wins_over_ipv4() -> None:
    assert (
        provisioning._first_address("fe80::18b2:3e41:95e1:b30b", "172.16.1.13")
        == "172.16.1.13"
    )
    assert provisioning._first_address("fe80::1") == "fe80::1"


def test_a_jaccl_capable_link_is_launched_as_a_ring(tmp_path) -> None:
    """A measured transport is capability, not the deployment's backend.

    JACCL needs a full per-host RDMA connectivity matrix, and this path has no
    pairwise path data — copying the measured ``jaccl-ring`` through made apply
    fail on a real 3-Mac cluster with "JACCL requires a full RDMA connectivity
    matrix". A ring over the same Thunderbolt bridge is the MVP transport.
    """

    store = MeasurementStore(tmp_path / "store")
    measurement_id = seed_measurement(store)
    record = store.get(measurement_id)
    assert record is not None
    nodes = []
    for node in record.nodes:
        updated = dict(node)
        updated["link"] = {
            "transport": "jaccl-ring",
            "bandwidth_gbps": None,
            "latency_ms": None,
        }
        nodes.append(updated)
    record = replace(record, nodes=tuple(nodes))

    result = ElectionResult(
        feasible=True,
        coordinator_node="m5",
        rank_order=("m5", "m1"),
        endpoint=None,
        selected=None,
        considered=(),
        rejected=(),
        policy=None,
        origin_node="m5",
        plan_signature="a" * 64,
        reasons=(),
    )
    seams = provisioning.ProvisioningSeams(
        device_record=lambda node_id: {"last_addrs": [SSH[node_id]]},
        local_addresses=lambda: (SSH["m5"],),
    )
    inputs = provisioning.PlanInputs(
        measurement_id=record.measurement_id,
        plan_signature="a" * 64,
        model=record.model,
        context_tokens=8192,
        microbatch_size=1,
    )

    targets = provisioning.resolve_targets(record, result, seams)
    request = provisioning.build_deployment_request(
        record,
        result,
        targets,
        coordinator=targets["m5"],
        source_host="127.0.0.1",
        inputs=inputs,
    )

    assert request.backend == "ring"


def test_a_closed_coordinator_asks_to_retry_the_coordinator(api) -> None:
    """The textbook ``retry_coordinator`` case: its oMLX is not answering.

    Live, a stopped coordinator app produced ``apply_failed`` + ``diagnostics``
    because the remote snippet died with a traceback instead of reporting "no
    answer". The next step for the user is to open oMLX on that Mac, not a
    diagnostics bundle — and never a local fallback.
    """

    from omlx.cluster.assistant.start import RemoteAdminError

    fake = FakeCoordinator()

    def closed(*_args: Any, **_kwargs: Any) -> Any:
        raise RemoteAdminError(
            0, "the coordinator did not answer: connection refused"
        )

    client, _, _ = api(coordinator=lambda plan: replace(fake.ops(plan), prepare=closed))
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == "failed"
    assert operation["error"]["code"] == "staging_refused"
    assert operation["error"]["action"] == ACTION_RETRY_COORDINATOR
    assert operation["endpoint"]["ready"] is False


def test_an_http_refusal_keeps_the_apply_action(api) -> None:
    """A 409 from a live coordinator is a staging refusal, not "open oMLX"."""

    from omlx.cluster.assistant.start import RemoteAdminError

    fake = FakeCoordinator()

    def refusing(*_args: Any, **_kwargs: Any) -> Any:
        raise RemoteAdminError(409, "the placement changed")

    client, _, _ = api(
        coordinator=lambda plan: replace(fake.ops(plan), prepare=refusing)
    )
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    operation = wait_for_operation(client, response.json()["operation_id"])

    assert operation["state"] == "failed"
    assert operation["error"]["code"] == "staging_refused"
    assert operation["error"]["action"] == ACTION_APPLY


def test_apply_refuses_while_the_coordinator_runs_another_origins_operation(
    api, monkeypatch
) -> None:
    """One apply/start at a time — across origins, not just within one.

    Each origin owns its store, so a second Mac driving the same coordinator
    would activate and replace the first. The coordinator is the only Mac that
    can answer whether it is busy, and its answer is a refusal with the running
    operation id, not a silent second activation.
    """

    fake = FakeCoordinator()
    client, _, operations = api(coordinator=fake.factory())
    monkeypatch.setattr(
        assistant_routes, "active_operation", lambda plan: "op-other-origin"
    )
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "operation_in_progress"
    assert body["action"] == "status"
    assert body["operation_id"] == "op-other-origin"
    # Nothing was created locally and nothing was staged on the coordinator.
    assert operations.active() is None
    assert fake.staged == []


def test_apply_proceeds_when_the_coordinator_cannot_be_asked(api, monkeypatch) -> None:
    """An unanswerable lock check must not become a fake lock."""

    def unavailable(plan):
        raise RuntimeError("the coordinator did not answer")

    fake = FakeCoordinator()
    client, _, _ = api(coordinator=fake.factory())
    monkeypatch.setattr(assistant_routes, "active_operation", unavailable)
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))

    assert response.status_code == 200, response.text
    assert wait_for_operation(client, response.json()["operation_id"])["state"] == (
        STATE_APPLIED
    )


def test_a_mac_without_room_for_the_copy_asks_to_free_space(api) -> None:
    """A copy that cannot fit is refused up front with one action.

    "Not enough disk space on Mac Studio" has a known fix — delete a model you
    no longer use there. Discovering it half way through a copy would waste the
    evening.
    """

    client, store, _ = api()
    measurement_id = take_measurement(client)
    record = store.get(measurement_id)
    assert record is not None
    layout = {
        "source": MODEL_PATH,
        "fixed_weight_bytes": 1 * GIB,
        "layer_weight_bytes": [2 * GIB] * 20,
        "tensor_count": 200,
        "supports_pipeline": True,
    }
    facts = tuple(
        replace(
            node,
            model_present=(node.node_id == "m5"),
            disk_free_gib=(5.0 if node.node_id == "m1" else node.disk_free_gib),
        )
        for node in record.measurement_set().nodes
    )
    store.finish(
        measurement_id,
        measurement=replace(record.measurement_set(), nodes=facts),
        endpoints=record.endpoints,
        layout=layout,
    )
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "insufficient_disk"
    assert body["action"] == ACTION_FREE_DISK_SPACE
    assert "GiB" in body["message"]
