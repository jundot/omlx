# SPDX-License-Identifier: Apache-2.0
"""The measurement job as the API exposes it.

These drive the real router with a ``TestClient`` and injected probes, so no
test touches SSH, a model or the validation cluster. They pin the four things
the job must never get wrong:

* it returns a handle before the first probe finishes, and never blocks the
  event loop while a probe is blocked in a worker thread;
* the stored record reaches ``ready`` with the measured facts;
* a failure ends in ``failed`` with exactly one actionable next step;
* one job at a time, and no staging, launch or registry write anywhere.

The admin gate is stubbed the way ``tests/test_cluster_ui_integration.py`` does
it: the router carries no authentication of its own, so mounting it with a
denying dependency must make every call 401. The real gates are attached in
``omlx/server.py::_register_cluster_routes``.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant.state import MeasurementStore
from omlx.cluster.node_role import NodeBudgetSuggestion
from omlx.cluster.transport import TransportInfo, TransportMatrix

GIB = 1024**3
MEASURE_URL = "/admin/api/cluster/assistant/measure"
STATUS_URL = "/admin/api/cluster/assistant/status"
PLAN_URL = "/admin/api/cluster/assistant/plan"
MEASURED_AT = "2026-11-03T12:00:00+00:00"

CAPACITY_GIB = {"m5": 107.5, "m1": 60.0}
USABLE_GIB = {"m5": 96.75, "m1": 54.0}
SSH = {"m5": "172.16.1.11", "m1": "172.16.1.14"}


def measure_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "nodes": [
            {"node_id": "m5", "ssh": SSH["m5"]},
            {"node_id": "m1", "ssh": SSH["m1"]},
        ],
        "roles": {"m5": "headless", "m1": "headless"},
        "model_size_bytes": 40 * GIB,
        "layer_count": 20,
    }
    payload.update(overrides)
    return payload


def fake_budget(
    *,
    role: str,
    ssh_target: str | None = None,
    capacity_bytes: int = 0,
    capacity_source: str | None = None,
) -> NodeBudgetSuggestion:
    """The shape ``node_role.suggest_budget`` returns, without touching a Mac."""

    return NodeBudgetSuggestion(
        capacity_bytes=capacity_bytes,
        reserve_bytes=int(capacity_bytes * 0.1),
        role=role,
        capacity_source=capacity_source or "admission_ceiling",
    )


def probes(
    *,
    gate: threading.Event | None = None,
    started: threading.Event | None = None,
    admission_error: Exception | None = None,
    transports: TransportMatrix | None = None,
    device: Callable[[str], dict[str, Any] | None] | None = None,
    budget_override: Callable[..., NodeBudgetSuggestion] | None = None,
    inventory: Callable[..., list[dict[str, Any]]] | None = None,
    capacities: Mapping[str, float] | None = None,
    model_layout: Callable[..., dict[str, Any]] | None = None,
) -> assistant_routes.MeasureProbes:
    caps = dict(capacities or CAPACITY_GIB)
    def admission_ceiling(ssh: str, python_executable: str | None = None) -> int:
        if started is not None:
            started.set()
        if gate is not None:
            assert gate.wait(10), "the test never released the probe"
        if admission_error is not None:
            raise admission_error
        node_id = next(key for key, value in SSH.items() if value == ssh)
        return int(caps[node_id] * GIB)

    def budget(**kwargs: Any) -> NodeBudgetSuggestion:
        if budget_override is not None:
            return budget_override(**kwargs)
        ssh = kwargs.get("ssh_target")
        node_id = next(key for key, value in SSH.items() if value == ssh)
        suggestion = fake_budget(**kwargs)
        return NodeBudgetSuggestion(
            capacity_bytes=suggestion.capacity_bytes,
            reserve_bytes=int((CAPACITY_GIB[node_id] - USABLE_GIB[node_id]) * GIB),
            role=suggestion.role,
            capacity_source="admission_ceiling",
        )

    return assistant_routes.MeasureProbes(
        admission_ceiling=admission_ceiling,
        budget=budget,
        transports=lambda hosts: (
            transports
            or TransportMatrix(
                transports=(
                    TransportInfo(
                        kind="thunderbolt",
                        interface="en5",
                        peer_node_id=SSH["m1"],
                        source_node_id=SSH["m5"],
                        link_speed_gbps=80,
                    ),
                ),
                backend="jaccl-ring",
            )
        ),
        model_inventory=inventory
        or (lambda ssh, python_executable=None: [
            {"model_path": "/models/qwen", "id": "qwen"}
        ]),
        model_layout=model_layout,
        disk_free=lambda path, ssh_target=None: 300 * GIB,
        device_record=device
        or (lambda node_id: {"last_addrs": [SSH[node_id]]} if node_id in SSH else None),
    )


@pytest.fixture()
def client(tmp_path):
    # One portal for the whole test: a fresh ``TestClient`` per request closes
    # its event loop, which would wait for the worker thread and hide exactly
    # the "returns before the probe finishes" behaviour these tests pin.
    stack = contextlib.ExitStack()

    def _build(
        *,
        settings: Any = None,
        **kwargs: Any,
    ) -> tuple[TestClient, MeasurementStore]:
        store = MeasurementStore(tmp_path / kwargs.pop("directory", "store"))
        assistant_routes.configure_assistant_measurements(
            store, probes=probes(**kwargs)
        )
        assistant_routes.set_assistant_getters(
            lambda: SimpleNamespace(base_path=tmp_path),
            lambda: settings or SimpleNamespace(port=8099, host="127.0.0.1"),
        )
        app = FastAPI()
        app.include_router(assistant_routes.router)
        api = stack.enter_context(TestClient(app))
        return api, store

    yield _build
    stack.close()


def wait_for_state(
    client: TestClient, measurement_id: str, *, timeout: float = 10.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(STATUS_URL, params={"measurement_id": measurement_id})
        assert response.status_code == 200, response.text
        last = response.json()
        if last["state"] != "running":
            return last
        time.sleep(0.01)
    raise AssertionError(f"measurement never finished: {last}")


# ---------------------------------------------------------------------------
# Measure returns a handle, not a result
# ---------------------------------------------------------------------------


def test_measure_returns_a_handle_before_any_probe_finishes(client) -> None:
    gate, started = threading.Event(), threading.Event()
    api, store = client(gate=gate, started=started)

    response = api.post(MEASURE_URL, json=measure_payload())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "running"
    assert body["measurement_id"].startswith("ms-")
    assert body["poll"] == f"{STATUS_URL}?measurement_id={body['measurement_id']}"
    # The probe was reached, and it has not finished: the response did not wait
    # for it.
    assert started.wait(5)
    assert not gate.is_set()
    record = store.get(body["measurement_id"])
    assert record is not None and record.state == "running"

    # ...and the event loop is still free while that probe is blocked.
    status = api.get(STATUS_URL, params={"measurement_id": body["measurement_id"]})
    assert status.status_code == 200
    assert status.json()["state"] == "running"

    gate.set()


def test_a_finished_measurement_is_ready_with_its_facts(client) -> None:
    api, _ = client()

    measurement_id = api.post(MEASURE_URL, json=measure_payload()).json()[
        "measurement_id"
    ]
    status = wait_for_state(api, measurement_id)

    assert status["state"] == "ready"
    assert status["measurement_id"] == measurement_id
    assert status["node_count"] == 2
    assert status["measured_at"]
    assert status["error"] is None
    assert [node["node_id"] for node in status["nodes"]] == ["m5", "m1"]
    assert status["nodes"][0]["admission_p05_gib"] == pytest.approx(96.75)
    assert status["nodes"][0]["endpoint_host"] == "172.16.1.11"
    assert status["nodes"][0]["endpoint_port"] == 8099
    # Measured without a performance probe: no rate was invented.
    codes = {warning["code"] for warning in status["warnings"]}
    assert codes == {"unmeasured_compute"}
    assert status["warnings"][0]["action"] == ""


def test_unreadable_free_space_is_reported_not_silently_unknown() -> None:
    from omlx.cluster.assistant.models import (
        AssistantMeasureNode,
        AssistantMeasureRequest,
    )

    warnings: list[Any] = []
    node = AssistantMeasureNode(node_id="m1", ssh=SSH["m1"])
    request = AssistantMeasureRequest(nodes=[node], model_path="/models/qwen")
    probes = SimpleNamespace(disk_free=lambda path, ssh_target=None: 0)

    assert assistant_routes._disk_free(probes, node, request, warnings) is None
    assert [warning.code for warning in warnings] == ["disk_free_unreadable"]


def test_an_unknown_interpreter_lets_the_probe_discover_one(client) -> None:
    """A Mac with no known interpreter must still report what it holds.

    Passing the caller's ``None`` through as the interpreter made every ordinary
    measurement report ``model_inventory_unreadable``: the probe refuses a
    non-absolute path instead of falling back to the peer's own. Availability
    then reads as "not on that Mac", and staging copies a model that is already
    there.
    """

    calls: list[tuple[str, dict[str, Any]]] = []

    def inventory(ssh: str, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append((ssh, kwargs))
        return [{"model_path": "/models/qwen", "id": "qwen"}]

    payload = measure_payload(
        model_path="/models/qwen",
        model_size_bytes=None,
        nodes=[
            {"node_id": "m5", "ssh": SSH["m5"]},
            {
                "node_id": "m1",
                "ssh": SSH["m1"],
                "python_executable": "/opt/omlx/bin/python",
            },
        ],
        roles={"m5": "headless", "m1": "headless"},
    )
    api, store = client(inventory=inventory)

    measurement_id = api.post(MEASURE_URL, json=payload).json()["measurement_id"]
    status = wait_for_state(api, measurement_id)

    assert status["state"] == "ready"
    by_ssh = dict(calls)
    assert "python_executable" not in by_ssh[SSH["m5"]]
    assert by_ssh[SSH["m1"]] == {"python_executable": "/opt/omlx/bin/python"}
    codes = {warning["code"] for warning in status["warnings"]}
    assert "model_inventory_unreadable" not in codes
    record = store.get(measurement_id)
    assert record is not None
    assert all(node.model_present for node in record.measurement_set().nodes)


def test_this_mac_resolves_a_dialable_endpoint_without_a_paired_record(
    client, monkeypatch
) -> None:
    """The ordinary local case must still name an endpoint.

    A Mac has no paired record of itself, so the measurement used to record an
    empty endpoint host for the app origin — and a plan whose coordinator *is*
    this Mac therefore carried no endpoint, no URL, and nothing to show when the
    cluster came up. The address a peer could dial is announced locally.
    """

    monkeypatch.setattr(
        assistant_routes, "_local_endpoint_addresses", lambda: ["172.16.1.11"]
    )

    def budget_override(**kwargs: Any) -> NodeBudgetSuggestion:
        capacity = int(CAPACITY_GIB["m5"] * GIB)
        return NodeBudgetSuggestion(
            capacity_bytes=capacity,
            reserve_bytes=int(capacity * 0.1),
            role="headless",
            capacity_source="admission_ceiling",
        )

    api, _ = client(device=lambda node_id: None, budget_override=budget_override)
    payload = measure_payload(
        nodes=[{"node_id": "m5", "ssh": "127.0.0.1", "python_executable": None}],
        roles={"m5": "headless"},
    )

    measurement_id = api.post(MEASURE_URL, json=payload).json()["measurement_id"]
    status = wait_for_state(api, measurement_id)

    assert status["state"] == "ready"
    assert status["nodes"][0]["endpoint_host"] == "172.16.1.11"
    assert status["nodes"][0]["endpoint_port"] == 8099


def test_local_endpoint_addresses_prefer_something_a_peer_can_dial(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "omlx.cluster.discovery.local_addr_dicts",
        lambda: [
            {"ip": "fe80::1437:9b93:2451:5286%bridge0"},
            {"ip": "127.0.0.1"},
            {"ip": "169.254.10.9"},
            {"ip": "192.168.178.55"},
            {"ip": "10.1.2.3"},
        ],
    )

    assert assistant_routes._local_endpoint_addresses() == [
        "192.168.178.55",
        "10.1.2.3",
        "169.254.10.9",
    ]


def test_the_job_only_reads_and_stores_a_measurement(client) -> None:
    """No staging, no launch, no registry and no deployment write."""

    api, _ = client()
    measurement_id = api.post(MEASURE_URL, json=measure_payload()).json()[
        "measurement_id"
    ]
    assert wait_for_state(api, measurement_id)["state"] == "ready"

    forbidden = {
        "ClusterDeployment",
        "get_cluster_registry",
        "launch_cluster",
        "plan_staging",
        "stage_files_from_source",
        "stage_manifest",
        "run_worker_smoke",
        "configure_device_registry",
    }
    assert forbidden & set(vars(assistant_routes)) == set()


def test_a_probe_failure_ends_the_job_with_exactly_one_action(client) -> None:
    api, store = client(admission_error=RuntimeError("ssh: connect timed out"))
    measurement_id = api.post(MEASURE_URL, json=measure_payload()).json()[
        "measurement_id"
    ]

    status = wait_for_state(api, measurement_id)

    assert status["state"] == "failed"
    assert status["node_count"] == 0
    assert status["error"]["code"] == "admission_probe_failed"
    assert "timed out" in status["error"]["message"]
    assert status["error"]["action"] == "measure"
    record = store.get(measurement_id)
    assert record is not None and record.error is not None
    assert list(record.error.to_dict()) == ["code", "message", "action"]


def test_no_mac_can_be_measured_is_a_failed_job_not_an_empty_plan(client) -> None:
    def unusable_budget(**kwargs: Any) -> NodeBudgetSuggestion:
        return NodeBudgetSuggestion(
            capacity_bytes=0, reserve_bytes=0, role="headless", capacity_source=""
        )

    api, _ = client(budget_override=unusable_budget)

    measurement_id = api.post(MEASURE_URL, json=measure_payload()).json()[
        "measurement_id"
    ]
    status = wait_for_state(api, measurement_id)

    assert status["state"] == "failed"
    assert status["error"]["code"] == "measurement_refused"
    assert status["error"]["action"] == "measure"
    assert "no Mac could be measured" in status["error"]["message"]


def test_a_second_measure_while_one_runs_is_refused(client) -> None:
    gate, started = threading.Event(), threading.Event()
    api, store = client(gate=gate, started=started)
    first = api.post(MEASURE_URL, json=measure_payload()).json()["measurement_id"]
    assert started.wait(5)

    second = api.post(MEASURE_URL, json=measure_payload())

    assert second.status_code == assistant_routes.MEASUREMENT_IN_PROGRESS_STATUS == 409
    body = second.json()
    assert body["code"] == "measurement_in_progress"
    assert body["action"] == "status"
    assert body["measurement_id"] == first
    # Refused, not joined: exactly one job exists.
    assert len([r for r in store.list() if r.state == "running"]) == 1

    gate.set()
    assert wait_for_state(api, first)["state"] == "ready"


# ---------------------------------------------------------------------------
# Request contract and status
# ---------------------------------------------------------------------------


def test_an_invalid_ssh_target_is_refused_before_a_job_starts(client) -> None:
    api, store = client()

    response = api.post(
        MEASURE_URL,
        json=measure_payload(nodes=[{"node_id": "m5", "ssh": "-oProxyCommand=evil"}]),
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_measure_request"
    assert store.list() == ()


def test_duplicate_node_ids_are_refused(client) -> None:
    api, store = client()

    response = api.post(
        MEASURE_URL,
        json=measure_payload(
            nodes=[
                {"node_id": "m5", "ssh": SSH["m5"]},
                {"node_id": "m5", "ssh": SSH["m1"]},
            ]
        ),
    )

    assert response.status_code == 400
    assert store.list() == ()


def test_a_hardware_only_measurement_is_a_third_identity(client) -> None:
    api, _ = client()

    response = api.post(MEASURE_URL, json=measure_payload(model_size_bytes=None))

    assert response.status_code == 200, response.text
    measurement_id = response.json()["measurement_id"]
    status = api.get(STATUS_URL, params={"measurement_id": measurement_id})
    assert status.status_code == 200
    assert status.json()["state"] == "ready"
    assert "hardware only" in status.json()["model"]


def test_a_model_identity_cannot_name_both_a_path_and_a_size(client, tmp_path) -> None:
    api, _ = client()

    response = api.post(
        MEASURE_URL,
        json=measure_payload(model_path=str(tmp_path), model_size_bytes=40 * GIB),
    )

    assert response.status_code == 400
    assert (
        "exactly one of model_path or model_size_bytes"
        in (response.json()["detail"]["message"])
    )


def test_unknown_request_fields_are_refused(client) -> None:
    api, _ = client()
    assert api.post(MEASURE_URL, json=measure_payload(dry_run=True)).status_code == 422


def test_a_local_mac_uses_its_own_settings_for_the_endpoint(client) -> None:
    """The owner's ``ServerSettings`` is the port source, and loopback is not
    an address another Mac can dial."""

    api, _ = client(
        budget_override=lambda **kwargs: NodeBudgetSuggestion(
            capacity_bytes=int(107.5 * GIB),
            reserve_bytes=int(10.75 * GIB),
            role=kwargs.get("role", "headless"),
            capacity_source="admission_ceiling",
        ),
        device=lambda node_id: None,
        settings=SimpleNamespace(port=9123, host="10.0.0.5"),
    )
    measurement_id = api.post(
        MEASURE_URL,
        json=measure_payload(nodes=[{"node_id": "m5", "ssh": "127.0.0.1"}]),
    ).json()["measurement_id"]

    status = wait_for_state(api, measurement_id)

    assert status["state"] == "ready"
    assert status["nodes"][0]["endpoint_host"] == "10.0.0.5"
    assert status["nodes"][0]["endpoint_port"] == 9123


def test_status_of_an_unknown_measurement_is_404_with_one_action(client) -> None:
    api, _ = client()

    response = api.get(STATUS_URL, params={"measurement_id": "ms-nope"})

    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "measurement_not_found"
    assert body["action"] == "measure"
    assert set(body) == {"code", "message", "action"}


def test_status_requires_a_measurement_id(client) -> None:
    api, _ = client()
    assert api.get(STATUS_URL).status_code == 422


def test_the_measurement_routes_are_behind_admin_authentication(client) -> None:
    """The router has no auth of its own; the server mounts it gated."""

    def deny() -> None:
        raise HTTPException(401, "admin required")

    app = FastAPI()
    app.include_router(assistant_routes.router, dependencies=[Depends(deny)])
    api = TestClient(app)

    assert api.post(MEASURE_URL, json=measure_payload()).status_code == 401
    assert api.get(STATUS_URL, params={"measurement_id": "x"}).status_code == 401
    assert api.post(PLAN_URL, json={}).status_code == 401


def test_the_measurement_remembers_what_each_mac_listed(client) -> None:
    """Presence for *any* recommended model needs the measured catalogue.

    Without it a recommendation can only say "already present" for the
    checkpoint the measurement covered, and reports every other model as a copy
    that has to be made.
    """

    def inventory(ssh: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"model_path": "/models/qwen", "id": "qwen"},
            {"model_path": "/models/glm", "id": "glm"},
        ]

    api, store = client(inventory=inventory)
    payload = measure_payload(
        model_path="/models/qwen", model_size_bytes=None
    )

    measurement_id = api.post(MEASURE_URL, json=payload).json()["measurement_id"]
    assert wait_for_state(api, measurement_id)["state"] == "ready"

    record = store.get(measurement_id)
    assert record is not None
    for node in record.measurement_set().nodes:
        assert set(node.model_paths) == {"/models/qwen", "/models/glm"}


def test_a_small_mac_is_measured_as_headless_and_says_so(client) -> None:
    """A workstation reserve larger than the Mac must not starve it.

    Live: a 32 GB Mac asked to be a workstation admitted **1.0 GiB**, so the
    recommendation refused every model with "none of the 7 candidate models
    fits". The Mac is measured as Headless instead, with one warning that names
    the reason.
    """

    seen_roles: list[str] = []

    def budget_override(**kwargs: Any) -> NodeBudgetSuggestion:
        seen_roles.append(kwargs["role"])
        capacity = int(kwargs["capacity_bytes"] or 21 * GIB)
        # What the real role does on a 21 GiB ceiling: the 32 GiB workstation
        # floor wins and leaves 1 GiB.
        reserve = (
            capacity - 1 * GIB
            if kwargs["role"] == "workstation"
            else int(capacity * 0.1)
        )
        return NodeBudgetSuggestion(
            capacity_bytes=capacity,
            reserve_bytes=reserve,
            role=kwargs["role"],
            capacity_source="admission_ceiling",
        )

    payload = measure_payload(roles={"m5": "unused", "m1": "workstation"})
    api, store = client(
        capacities={"m5": 107.5, "m1": 21.0}, budget_override=budget_override
    )
    measurement_id = api.post(MEASURE_URL, json=payload).json()["measurement_id"]
    status = wait_for_state(api, measurement_id)

    assert status["state"] == "ready"
    codes = {warning["code"]: warning for warning in status["warnings"]}
    assert assistant_routes.WARNING_WORKSTATION_TOO_LARGE in codes
    assert codes[assistant_routes.WARNING_WORKSTATION_TOO_LARGE]["node_id"] == "m1"
    # ...and the role actually used is headless, so a model can still fit.
    assert "workstation" in seen_roles  # asked
    assert seen_roles[-1] == "headless"  # recomputed


def test_a_model_that_lives_on_another_mac_is_measured_from_that_mac(client) -> None:
    """The origin cannot read a peer's checkpoint; the peer lends its shape.

    Live-relevant: a small origin with the model only on the bigger Mac used to
    fail planning with an unreadable checkpoint although the cluster could stage
    and run it. The layout is read on the Mac that holds it and stored with the
    measurement, so plan/apply/start all re-elect from the same numbers.
    """

    peer_layout = {
        "source": "/models/qwen",
        "fixed_weight_bytes": 1_000_000,
        "layer_weight_bytes": [2_000_000] * 12,
        "tensor_count": 120,
    }

    inventory_calls: list[dict[str, Any]] = []

    def inventory(ssh: str, **kwargs: Any) -> list[dict[str, Any]]:
        # An unknown interpreter must not be passed through as None: the real
        # probe refuses a non-absolute path, which silently skipped this whole
        # path once (and no layout was ever read from a peer).
        inventory_calls.append(kwargs)
        return [{"model_path": "/models/qwen", "id": "qwen"}]

    seen: list[tuple[str, str]] = []

    def layout(ssh: str, path: str) -> dict[str, Any]:
        seen.append((ssh, path))
        return peer_layout

    api, store = client(inventory=inventory, model_layout=layout)
    payload = measure_payload(model_path="/models/qwen", model_size_bytes=None)

    measurement_id = api.post(MEASURE_URL, json=payload).json()["measurement_id"]
    status = wait_for_state(api, measurement_id)

    assert status["state"] == "ready"
    record = store.get(measurement_id)
    assert record is not None
    assert record.layout == peer_layout
    assert seen and seen[0][1] == "/models/qwen"
    assert all("python_executable" not in call for call in inventory_calls)
    codes = {warning["code"] for warning in status["warnings"]}
    assert assistant_routes.WARNING_LAYOUT_FROM_PEER in codes
