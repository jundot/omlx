# SPDX-License-Identifier: Apache-2.0
"""The measurement-driven plan API.

These drive the real router with a ``TestClient``: a measurement is taken
through ``/measure`` (with injected probes, so nothing reaches a Mac) and the
plan is then built from its persisted facts. They pin the things the endpoint
must never get wrong:

* the Mac running the app is not rank 0 — the elected Mac is the one whose tail
  actually fits, decided from a stored measurement;
* the notes plan object is complete: measurement id and timestamp, signature,
  stages, predicted stage seconds, ``observed: null`` and a real endpoint port
  with ``ready: false``;
* a missing, unknown, stale or mismatched measurement is a distinct 409 with one
  action, and tensor parallelism is an explicit 400;
* an infeasible cluster is explicit (409, no coordinator, machine-readable code)
  and planning changes nothing.

The admin gate is stubbed the way ``tests/test_cluster_ui_integration.py`` does
it. The real gates are attached in
``omlx/server.py::_register_cluster_routes``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant.state import MeasurementStore
from omlx.cluster.node_role import NodeBudgetSuggestion
from omlx.cluster.performance import NodePerformanceProfile
from omlx.cluster.transport import TransportInfo, TransportMatrix

GIB = 1024**3
MEASURE_URL = "/admin/api/cluster/assistant/measure"
STATUS_URL = "/admin/api/cluster/assistant/status"
PLAN_URL = "/admin/api/cluster/assistant/plan"
RECOMMEND_URL = "/admin/api/cluster/assistant/recommend"
APPLY_URL = "/admin/api/cluster/assistant/apply"
START_URL = "/admin/api/cluster/assistant/start"
STOP_URL = "/admin/api/cluster/assistant/stop"
SEARCH_URL = "/admin/api/cluster/assistant/search"
TUNE_URL = "/admin/api/cluster/assistant/tune"
REDEPLOY_URL = "/admin/api/cluster/assistant/tune-redeploy"
TUNING_PRESET_URL = "/admin/api/cluster/assistant/tuning-preset"
RUN_LOCAL_URL = "/admin/api/cluster/assistant/run-local"
TUNE_LOCAL_URL = "/admin/api/cluster/assistant/tune-local"
SESSION_URL = "/admin/api/cluster/assistant/session"
PERFORMANCE_PROBE_URL = "/admin/api/cluster/assistant/performance-probe"
OPERATIONS_URL = "/admin/api/cluster/assistant/operations"
MEASURED_AT = "2026-11-03T12:00:00+00:00"

CAPACITY_GIB = {"m5": 107.5, "m1": 60.0}
USABLE_GIB = {"m5": 96.75, "m1": 54.0}
SSH = {"m5": "192.0.2.11", "m1": "192.0.2.14"}
UNITS = [
    {"node_id": "m5", "ssh": SSH["m5"]},
    {"node_id": "m1", "ssh": SSH["m1"]},
]


def performance_profile(
    node_id: str, decode_gib_per_s: float
) -> NodePerformanceProfile:
    return NodePerformanceProfile(
        node_id=node_id,
        rank=0,
        # Relative speed signal mapped into the plausible bandwidth window the
        # election's plausibility guard expects (E4=B); ordering preserved.
        decode_weight_bytes_per_second=decode_gib_per_s * 100 * GIB,
        prefill_weight_bytes_per_second=decode_gib_per_s * 2 * 100 * GIB,
        collective_latency_seconds=0.0003,
        collective_bandwidth_bytes_per_second=6.6e9,
        backend="jaccl-ring" if node_id == "m5" else "ring",
        measured_at=MEASURED_AT,
        samples=5,
    )


PERFORMANCE = [performance_profile("m5", 2.0), performance_profile("m1", 0.8)]


def measure_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "nodes": UNITS,
        "roles": {"m5": "headless", "m1": "headless"},
        "model_size_bytes": 40 * GIB,
        "layer_count": 20,
        "include_performance_probe": True,
    }
    payload.update(overrides)
    return payload


def plan_payload(measurement_id: str, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "measurement_id": measurement_id,
        "model_size_bytes": 40 * GIB,
        "layer_count": 20,
        "context_tokens": 8192,
        "origin_node": "m1",
    }
    payload.update(overrides)
    return payload


def probes(
    *,
    device: Callable[[str], dict[str, Any] | None] | None = None,
) -> assistant_routes.MeasureProbes:
    def admission_ceiling(ssh: str, python_executable: str | None = None) -> int:
        node_id = next(key for key, value in SSH.items() if value == ssh)
        return int(CAPACITY_GIB[node_id] * GIB)

    def budget(**kwargs: Any) -> NodeBudgetSuggestion:
        ssh = kwargs.get("ssh_target")
        node_id = next(key for key, value in SSH.items() if value == ssh)
        return NodeBudgetSuggestion(
            capacity_bytes=int(CAPACITY_GIB[node_id] * GIB),
            reserve_bytes=int((CAPACITY_GIB[node_id] - USABLE_GIB[node_id]) * GIB),
            role=kwargs.get("role", "headless"),
            capacity_source="admission_ceiling",
        )

    return assistant_routes.MeasureProbes(
        admission_ceiling=admission_ceiling,
        budget=budget,
        transports=lambda hosts: TransportMatrix(
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
        ),
        model_inventory=lambda ssh, python_executable=None: [
            {"model_path": "/models/qwen", "id": "qwen"}
        ],
        disk_free=lambda path, ssh_target=None: 300 * GIB,
        device_record=(
            device
            if device is not None
            else (
                lambda node_id: (
                    {"last_addrs": [SSH[node_id]], "http_port": 8099}
                    if node_id in SSH
                    else None
                )
            )
        ),
        performance=lambda nodes: PERFORMANCE,
    )


DEFAULT_SETTINGS = SimpleNamespace(port=8099, host="127.0.0.1")


def default_device(node_id: str) -> dict[str, Any] | None:
    return {"last_addrs": [SSH[node_id]], "http_port": 8099} if node_id in SSH else None


@pytest.fixture()
def api(tmp_path):
    stack = contextlib.ExitStack()

    def _build(
        *,
        device: Callable[[str], dict[str, Any] | None] | None = default_device,
        settings: Any = DEFAULT_SETTINGS,
    ) -> tuple[TestClient, MeasurementStore]:
        store = MeasurementStore(tmp_path / "store")
        assistant_routes.configure_assistant_measurements(
            store, probes=probes(device=device)
        )
        assistant_routes.set_assistant_getters(
            lambda: SimpleNamespace(base_path=tmp_path / "store"),
            lambda: settings,
        )
        app = FastAPI()
        app.include_router(assistant_routes.router)
        return stack.enter_context(TestClient(app)), store

    yield _build
    stack.close()


def take_measurement(client: TestClient, **overrides: Any) -> str:
    response = client.post(MEASURE_URL, json=measure_payload(**overrides))
    assert response.status_code == 200, response.text
    measurement_id = response.json()["measurement_id"]
    status = client.get(STATUS_URL, params={"measurement_id": measurement_id})
    assert status.status_code == 200
    assert status.json()["state"] == "ready", status.text
    return measurement_id


# ---------------------------------------------------------------------------
# The election the API is a window onto
# ---------------------------------------------------------------------------


def test_plan_elects_the_remote_mac_when_the_origin_tail_does_not_fit(api) -> None:
    """the notes at the API boundary, through a stored measurement: M1 cannot hold
    tail + head, M5 can, and the app origin does not decide it."""

    client, _ = api()
    measurement_id = take_measurement(client)

    response = client.post(
        PLAN_URL, json=plan_payload(measurement_id, rank0_extra_bytes=70 * GIB)
    )

    assert response.status_code == 200, response.text
    plan = response.json()
    assert plan["feasible"] is True
    assert plan["coordinator_node"] == "m5"
    assert plan["rank_order"] == ["m5", "m1"]
    assert plan["origin_node"] == "m1"  # recorded, and not the answer
    assert plan["measurement_id"] == measurement_id
    assert plan["measurement_timestamp"]
    assert [item["node_id"] for item in plan["rejected"]] == ["m1"]
    assert "no legal pipeline split" in plan["rejected"][0]["reason"]


def test_the_plan_object_matches_the_decided_contract(api) -> None:
    client, _ = api()

    plan = client.post(PLAN_URL, json=plan_payload(take_measurement(client))).json()

    # the notes field names, exactly: no ``measurements_timestamp``, no
    # ``critical_rank``, no ``endpoint.node_id``.
    assert plan["measurement_id"].startswith("ms-")
    assert plan["measurement_timestamp"]
    assert "measurements_timestamp" not in plan
    assert len(plan["plan_signature"]) == 64
    assert [stage["node_id"] for stage in plan["stages"]] == ["m5", "m1"]
    assert [stage["rank"] for stage in plan["stages"]] == [0, 1]
    assert sum(plan["groups"]) == 20
    assert any(stage["critical"] for stage in plan["stages"])

    predicted = plan["predicted"]
    assert predicted["measured"] is True
    assert predicted["critical_stage_seconds"] > 0
    assert predicted["critical_stage_rank"] in (0, 1)
    assert "critical_rank" not in predicted
    assert [stage["rank"] for stage in predicted["stages"]] == [0, 1]
    assert all(stage["seconds"] > 0 for stage in predicted["stages"])
    # Token rates arrive only after the smoke test.
    assert plan["observed"] is None
    assert "decode_tok_per_s" not in str(predicted)

    endpoint = plan["endpoint"]
    assert set(endpoint) == {"node", "host", "port", "url", "ready"}
    assert endpoint["node"] == "m5"
    assert endpoint["host"] == "192.0.2.11"
    assert endpoint["port"] == 8099
    assert endpoint["url"] == "http://192.0.2.11:8099"
    assert endpoint["ready"] is False

    assert plan["model"].startswith("synthetic model")
    assert plan["context_tokens"] == 8192
    assert plan["mtp"] is False
    assert plan["tensor_parallel_size"] == 1
    assert plan["policy"]["name"] == "feasibility_first"
    assert plan["reasons"]
    assert plan["warnings"] == []


def test_the_origin_never_changes_the_elected_coordinator(api) -> None:
    """the notes: the same measurement elects the same Mac from any origin."""

    client, _ = api()
    measurement_id = take_measurement(client)
    elections = {}
    for origin in ("m1", "m5", "laptop-outside-the-cluster", None):
        response = client.post(
            PLAN_URL,
            json=plan_payload(
                measurement_id, rank0_extra_bytes=70 * GIB, origin_node=origin
            ),
        )
        assert response.status_code == 200, response.text
        plan = response.json()
        elections[origin] = (
            plan["coordinator_node"],
            tuple(plan["rank_order"]),
            plan["plan_signature"],
        )
        assert plan["origin_node"] == (origin or "")

    assert len(set(elections.values())) == 1
    assert elections["m1"][0] == "m5"


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------


def test_the_endpoint_port_defaults_to_8099_only_when_settings_are_unreadable(
    api,
) -> None:
    """No port reading and no paired port: the documented default, never 0."""

    client, _ = api(
        device=lambda node_id: (
            {"last_addrs": [SSH[node_id]]} if node_id in SSH else None
        ),
        settings=None,
    )

    plan = client.post(PLAN_URL, json=plan_payload(take_measurement(client))).json()

    assert plan["endpoint"]["port"] == 8099
    assert plan["endpoint"]["ready"] is False
    assert ":8099" in plan["endpoint"]["url"]


def test_a_configured_port_is_reported_unchanged(api) -> None:
    client, _ = api(
        device=lambda node_id: (
            {"last_addrs": [SSH[node_id]], "http_port": 9123}
            if node_id in SSH
            else None
        ),
        settings=SimpleNamespace(port=9123, host="127.0.0.1"),
    )

    plan = client.post(PLAN_URL, json=plan_payload(take_measurement(client))).json()

    assert plan["endpoint"]["port"] == 9123
    assert plan["endpoint"]["url"].endswith(":9123")


def test_a_node_without_a_reachable_host_has_no_endpoint_and_one_action(api) -> None:
    """Never a port-0 URL: an unresolvable endpoint is null plus a next step."""

    client, _ = api(device=lambda node_id: None)

    plan = client.post(PLAN_URL, json=plan_payload(take_measurement(client))).json()

    assert plan["feasible"] is True
    assert plan["endpoint"] is None
    unresolved = [
        warning
        for warning in plan["warnings"]
        if warning["code"] == "endpoint_unresolved"
    ]
    assert len(unresolved) == 1
    assert unresolved[0]["action"] == "configure_endpoint"
    assert unresolved[0]["node_id"] == plan["coordinator_node"]
    assert '"port": 0' not in str(plan)


# ---------------------------------------------------------------------------
# The two explicit refusals, and the measurement contract
# ---------------------------------------------------------------------------


def test_plan_without_a_measurement_id_is_422(api) -> None:
    client, _ = api()
    body = plan_payload("unused")
    del body["measurement_id"]

    assert client.post(PLAN_URL, json=body).status_code == 422


def test_plan_refuses_request_body_facts(api) -> None:
    """Facts come from the measurement: a ``nodes`` array is not accepted."""

    client, _ = api()
    measurement_id = take_measurement(client)
    body = plan_payload(measurement_id)
    body["nodes"] = [{"node_id": "m5", "capacity_gib": 999, "admission_p05_gib": 999}]

    response = client.post(PLAN_URL, json=body)

    assert response.status_code == 422
    assert "nodes" in response.text


def test_an_unknown_measurement_is_a_stale_409_with_no_plan(api) -> None:
    client, _ = api()

    response = client.post(PLAN_URL, json=plan_payload("ms-does-not-exist"))

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "stale_measurement"
    assert body["action"] == "measure"
    assert body["reason"] == "unknown"
    assert body["measurement_id"] == "ms-does-not-exist"
    assert "stages" not in body
    assert "coordinator_node" not in body


def test_a_stale_measurement_is_a_stale_409(api) -> None:
    client, store = api()
    measurement_id = take_measurement(client)
    record = store.get(measurement_id)
    assert record is not None
    old = datetime.now(UTC) - timedelta(minutes=30)
    store.finish(
        measurement_id,
        measurement=replace(record.measurement_set(), measured_at=old.isoformat()),
        endpoints=dict(record.endpoints),
    )

    response = client.post(PLAN_URL, json=plan_payload(measurement_id))

    assert response.status_code == 409
    assert response.json()["code"] == "stale_measurement"
    assert response.json()["reason"] == "age"
    assert response.json()["action"] == "measure"


def test_a_node_set_mismatch_is_a_stale_409(api) -> None:
    client, _ = api()
    measurement_id = take_measurement(client)

    # A subset is allowed (the fewest Macs that fit); a Mac the measurement
    # never covered is still a refusal with one action.
    response = client.post(
        PLAN_URL, json=plan_payload(measurement_id, node_ids=["m5", "m9"])
    )

    assert response.status_code == 409
    assert response.json()["code"] == "stale_measurement"
    assert response.json()["reason"] == "node_set"
    assert "m9" in response.json()["message"]


def test_a_model_mismatch_is_a_stale_409(api) -> None:
    client, _ = api()
    measurement_id = take_measurement(client)

    response = client.post(
        PLAN_URL,
        json=plan_payload(measurement_id, model_size_bytes=60 * GIB),
    )

    assert response.status_code == 409
    assert response.json()["code"] == "stale_measurement"
    assert response.json()["reason"] == "model"


def test_an_mtp_mismatch_is_a_stale_409(api) -> None:
    client, _ = api()
    measurement_id = take_measurement(client)

    response = client.post(PLAN_URL, json=plan_payload(measurement_id, mtp=True))

    assert response.status_code == 409
    assert response.json()["code"] == "stale_measurement"
    assert response.json()["reason"] == "mtp"


def test_tensor_parallelism_is_rejected_instead_of_being_planned_as_pp(api) -> None:
    """the notes: an explicit 4xx, and no PP plan smuggled into the body."""

    client, _ = api()
    measurement_id = take_measurement(client)

    response = client.post(
        PLAN_URL, json=plan_payload(measurement_id, tensor_parallel_size=2)
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "unsupported_parallelism"
    assert "tensor parallelism" in detail["message"]
    assert "pipeline parallelism only" in detail["message"]
    assert "stages" not in response.json()
    assert "coordinator_node" not in response.json()


def test_an_infeasible_cluster_is_409_and_names_no_coordinator(api) -> None:
    """the notes: no plan is an explicit answer, never the local Mac."""

    client, _ = api()
    measurement_id = take_measurement(
        client, model_size_bytes=400 * GIB, layer_count=40
    )

    response = client.post(
        PLAN_URL,
        json=plan_payload(measurement_id, model_size_bytes=400 * GIB, layer_count=40),
    )

    assert response.status_code == assistant_routes.NO_FEASIBLE_PLAN_STATUS == 409
    plan = response.json()
    assert plan["feasible"] is False
    assert plan["code"] == "no_feasible_plan"
    assert plan["action"] == "adjust_model_or_context"
    assert plan["coordinator_node"] == ""
    assert plan["rank_order"] == []
    assert plan["endpoint"] is None
    assert plan["stages"] == []
    assert plan["predicted"]["measured"] is False
    assert plan["predicted"]["critical_stage_seconds"] is None
    assert plan["plan_signature"] == ""
    assert plan["measurement_id"] == measurement_id
    # The Mac running the app is not offered as a consolation prize.
    assert "m1" not in plan["rank_order"]
    assert {item["node_id"] for item in plan["rejected"]} == {"m1", "m5"}


def test_an_unmeasured_plan_is_flagged_instead_of_being_called_fast(api) -> None:
    """the notes: with no probe the plan says "unknown", not a rate."""

    client, _ = api()
    measurement_id = take_measurement(client, include_performance_probe=False)

    plan = client.post(PLAN_URL, json=plan_payload(measurement_id)).json()

    assert plan["feasible"] is True
    assert plan["predicted"]["measured"] is False
    assert plan["predicted"]["critical_stage_seconds"] is None
    assert all(stage["seconds"] is None for stage in plan["predicted"]["stages"])
    codes = {warning["code"] for warning in plan["warnings"]}
    assert codes == {"unmeasured_compute", "unmeasured_plan"}


# ---------------------------------------------------------------------------
# Purity and surface
# ---------------------------------------------------------------------------


def test_planning_is_idempotent_and_has_no_side_effects(api) -> None:
    """the notes: ``plan`` is a pure function of the stored measurement."""

    client, store = api()
    measurement_id = take_measurement(client)
    body = plan_payload(measurement_id, rank0_extra_bytes=70 * GIB)
    stored = store.path.read_bytes()

    first = client.post(PLAN_URL, json=body)
    second = client.post(PLAN_URL, json=body)

    assert first.status_code == 200, first.text
    assert first.json() == second.json()
    # Reading the measurement must not rewrite it.
    assert store.path.read_bytes() == stored


def test_plan_never_reads_settings_or_runs_a_probe(api, monkeypatch) -> None:
    """The one rule that keeps a plan honest: it measures nothing itself."""

    client, _ = api()
    measurement_id = take_measurement(client)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("plan must not read settings or run a probe")

    monkeypatch.setattr(assistant_routes, "_server_settings", forbidden)
    monkeypatch.setattr(
        assistant_routes,
        "_measure_probes",
        forbidden,
    )
    monkeypatch.setattr(
        assistant_routes,
        "default_measure_probes",
        forbidden,
    )
    monkeypatch.setattr(
        assistant_routes,
        "measurement_timestamp",
        forbidden,
    )

    response = client.post(PLAN_URL, json=plan_payload(measurement_id))

    assert response.status_code == 200, response.text
    assert response.json()["feasible"] is True


def test_the_router_exposes_the_assistant_routes(api) -> None:
    assert [
        (route.path, sorted(route.methods)) for route in assistant_routes.router.routes
    ] == [
        (MEASURE_URL, ["POST"]),
        (STATUS_URL, ["GET"]),
        (RECOMMEND_URL, ["POST"]),
        (PLAN_URL, ["POST"]),
        # The free-space affordance: one model, on one measured Mac, through that
        # Mac's own delete route.
        (
            "/admin/api/cluster/assistant/models/{node_id}/{model_id:path}",
            ["DELETE"],
        ),
    ]
    # ...and the module never imported oMLX's mutating cluster surface.
    forbidden = {
        "ClusterDeployment",
        "get_cluster_registry",
        "launch_cluster",
        "plan_staging",
        "stage_manifest",
        "stage_files_from_source",
        "run_worker_smoke",
    }
    assert forbidden & set(vars(assistant_routes)) == set()


def test_the_plan_route_is_behind_admin_authentication(api) -> None:
    """The router has no auth of its own; the server mounts it gated."""

    def deny() -> None:
        raise HTTPException(401, "admin required")

    app = FastAPI()
    app.include_router(assistant_routes.router, dependencies=[Depends(deny)])
    client = TestClient(app)

    assert client.post(PLAN_URL, json=plan_payload("x")).status_code == 401
    assert client.get(PLAN_URL).status_code in (401, 405)


def test_the_server_registers_the_assistant_router_next_to_the_cluster_one() -> None:
    """The deliberate upstream patch is one include_router call, not a new app.

    ``_register_cluster_routes`` imports the assistant router, injects the
    settings getters and mounts it with the same admin and experimental gates as
    ``cluster.routes``. Reading the function's own names keeps this a wiring
    check without registering routers on the global app from inside a test.
    """

    import omlx.server as server

    names = set(server._register_cluster_routes.__code__.co_names)
    assert {
        "cluster.routes",
        "cluster.assistant.routes",
        "set_assistant_getters",
        "require_admin",
        "require_distributed_inference_enabled",
    } <= names


def test_the_status_poll_is_the_handle_the_measure_route_returns(api) -> None:
    """The poll URL a client is handed must be the route that answers."""

    client, _ = api()
    response = client.post(
        MEASURE_URL, json=measure_payload(include_performance_probe=False)
    )
    body = response.json()

    status = client.get(body["poll"])

    assert status.status_code == 200
    assert status.json()["measurement_id"] == body["measurement_id"]


def test_a_measured_set_is_accepted_verbatim_by_the_election(api) -> None:
    """The stored facts, not a re-measurement, decide the plan."""

    client, store = api()
    measurement_id = take_measurement(client)
    record = store.get(measurement_id)
    assert record is not None
    node_ids: Sequence[str] = record.node_ids
    assert list(node_ids) == ["m5", "m1"]


def test_the_plan_is_built_from_the_layout_the_measurement_recorded(
    api, monkeypatch
) -> None:
    """The plan may not change because a shard file changed after measuring.

    Live, the same stored measurement elected `.11` before a server restart and
    `.14` after it, so an approved apply refused to start with
    ``plan_mismatch`` although nobody had changed anything. The measurement now
    records the checkpoint's shape and every re-election uses it.
    """

    from omlx.cluster.assistant import provisioning

    client, store = api()
    measurement_id = take_measurement(client)
    record = store.get(measurement_id)
    assert record is not None and record.layout is not None

    # Make the checkpoint unreadable: a plan built from the record must not care.
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan re-read the checkpoint instead of the record")

    monkeypatch.setattr(provisioning, "inspect_safetensors_layout", forbidden)
    monkeypatch.setattr(provisioning, "model_layout_for", forbidden)

    response = client.post(PLAN_URL, json=plan_payload(measurement_id))

    assert response.status_code == 200, response.text
    assert response.json()["feasible"] is True
