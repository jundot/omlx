# SPDX-License-Identifier: Apache-2.0
"""Shared offline harness for the Cluster Assistant API tests.

Everything here is a fake: a measured pair of Macs, a synthetic model layout, a
coordinator whose prepare/stage/activate/health calls are recorded instead of
performed, and a ``TestClient`` over the real assistant router. No test touches
SSH, a model on disk or a Mac.

It lives in its own module (not in a test file) so ``test_apply_api.py`` and
``test_start_api.py`` can share it without importing one test module from
another, which is also what keeps the fixture visible to lint.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster.assistant import provisioning
from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant import start as assistant_start
from omlx.cluster.assistant.measurement import (
    measure_budget_payloads,
    measurement_timestamp,
)
from omlx.cluster.assistant.operations import STATE_APPLIED, OperationStore
from omlx.cluster.assistant.performance_store import AssistantPerformanceStore
from omlx.cluster.assistant.provisioning import ProvisioningSeams
from omlx.cluster.assistant.start import CoordinatorOps, PreparedDeployment
from omlx.cluster.assistant.state import (
    MeasurementEndpoint,
    MeasurementStore,
    ModelIdentity,
)
from omlx.cluster.node_role import NodeBudgetSuggestion
from omlx.cluster.performance import NodePerformanceProfile
from omlx.cluster.planner import ModelLayout
from omlx.cluster.transport import TransportInfo, TransportMatrix

GIB = 1024**3
MEASURE_URL = "/admin/api/cluster/assistant/measure"
STATUS_URL = "/admin/api/cluster/assistant/status"
PLAN_URL = "/admin/api/cluster/assistant/plan"
APPLY_URL = "/admin/api/cluster/assistant/apply"
START_URL = "/admin/api/cluster/assistant/start"
OPERATIONS_URL = "/admin/api/cluster/assistant/operations"

MODEL_PATH = "/models/qwen"
MEASURED_AT = "2026-11-03T12:00:00+00:00"

CAPACITY_GIB = {"m5": 107.5, "m1": 60.0}
USABLE_GIB = {"m5": 96.75, "m1": 54.0}
SSH = {"m5": "192.0.2.11", "m1": "192.0.2.14"}
UNITS = [
    {"node_id": "m5", "ssh": SSH["m5"]},
    {"node_id": "m1", "ssh": SSH["m1"]},
]


def model_layout(path: str) -> ModelLayout:
    """A checkpoint that fits the measured pair, so the plan is feasible."""

    return ModelLayout(
        source=path,
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=(2 * GIB,) * 20,
        supports_pipeline=True,
    )


def performance_profile(node_id: str, decode_gib_per_s: float) -> dict[str, Any]:
    # decode_gib_per_s is a RELATIVE speed signal; map it into the plausible
    # memory-bandwidth window the election's plausibility guard expects (E4=B).
    # x100 turns the 0.8-2.0 calibration values into ~86-215 GB/s, preserving
    # their relative ordering.
    return NodePerformanceProfile(
        node_id=node_id,
        rank=0,
        decode_weight_bytes_per_second=decode_gib_per_s * 100 * GIB,
        prefill_weight_bytes_per_second=decode_gib_per_s * 2 * 100 * GIB,
        collective_latency_seconds=0.0003,
        collective_bandwidth_bytes_per_second=6.6e9,
        backend="jaccl-ring" if node_id == "m5" else "ring",
        measured_at=MEASURED_AT,
        samples=5,
    ).to_dict()


PERFORMANCE = [performance_profile("m5", 2.0), performance_profile("m1", 0.8)]


def default_device(node_id: str) -> dict[str, Any] | None:
    return {"last_addrs": [SSH[node_id]], "http_port": 8099} if node_id in SSH else None


def measure_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "nodes": UNITS,
        "roles": {"m5": "headless", "m1": "headless"},
        "model_path": MODEL_PATH,
        "include_performance_probe": True,
    }
    payload.update(overrides)
    return payload


def plan_payload(measurement_id: str, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "measurement_id": measurement_id,
        "model_path": MODEL_PATH,
        "context_tokens": 8192,
        "origin_node": "m1",
    }
    payload.update(overrides)
    return payload


def probes(
    *,
    device: Callable[[str], dict[str, Any] | None] = default_device,
    inventory: Callable[..., list[dict[str, Any]]] | None = None,
    model_settings: Callable[[str], Mapping[str, Any]] | None = None,
    transports: Callable[[Sequence[str]], TransportMatrix] | None = None,
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
        transports=transports
        or (
            lambda hosts: TransportMatrix(
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
        or (lambda ssh, python_executable=None: [{"model_path": MODEL_PATH, "id": "qwen"}]),
        disk_free=lambda path, ssh_target=None: 300 * GIB,
        device_record=device,
        performance=lambda nodes: PERFORMANCE,
        model_settings=model_settings,
    )


def seed_measurement(
    store: MeasurementStore,
    *,
    present: Sequence[str] = ("m5", "m1"),
    measured_at: str | None = None,
    fabric_links: Sequence[Sequence[str]] = (),
    fabric_measured: bool = False,
) -> str:
    """A ready, model-bound measurement with exactly the facts a test wants."""

    payloads = [
        {
            "node_id": node_id,
            "capacity_bytes": int(CAPACITY_GIB[node_id] * GIB),
            "reserve_bytes": int((CAPACITY_GIB[node_id] - USABLE_GIB[node_id]) * GIB),
            "usable_bytes": int(USABLE_GIB[node_id] * GIB),
            "role": "headless",
            "capacity_source": "admission_ceiling",
            "endpoint_host": SSH[node_id],
            "endpoint_port": 8099,
            "disk_free_bytes": 300 * GIB,
            "model_present": node_id in present,
            "performance": performance_profile(
                node_id, 2.0 if node_id == "m5" else 0.8
            ),
        }
        for node_id in SSH
    ]
    measurement = measure_budget_payloads(
        payloads, measured_at=measured_at or measurement_timestamp()
    )
    record = store.start(model=ModelIdentity(model_path=MODEL_PATH), node_ids=list(SSH))
    store.finish(
        record.measurement_id,
        measurement=measurement,
        endpoints={
            node_id: MeasurementEndpoint(host=SSH[node_id], port=8099)
            for node_id in SSH
        },
        fabric_links=fabric_links,
        fabric_measured=fabric_measured,
    )
    return record.measurement_id


def staging_snapshot(
    status: str = "completed",
    *,
    ready: Sequence[str] = ("m5", "m1"),
    failed: Sequence[str] = (),
) -> dict[str, Any]:
    nodes = {}
    for node_id in ("m5", "m1"):
        if node_id in failed:
            node_status = "failed"
        elif node_id in ready:
            node_status = "ready"
        else:
            node_status = "copying"
        nodes[node_id] = {"node_id": node_id, "status": node_status}
    return {
        "job_id": "0123456789abcdef01234567",
        "status": status,
        "ready": status == "completed",
        "nodes": nodes,
    }


def activation_report(
    *,
    ok: bool = True,
    canary: bool = True,
    tokens: int = 4,
    state: str = "ready",
    ranks_ready: bool = True,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "readiness": {
            "state": state,
            "canary_passed": canary,
            "canary_completion_tokens": tokens,
            "all_ranks_ready": ranks_ready,
            "ranks": [],
        },
    }


class FakeCoordinator:
    """The coordinator's pipeline, faked: no SSH, no model, no Mac.

    It records every call so a test can assert that apply stages and never
    activates, that the source is the coordinator's own copy, and that staging
    was asked to copy the placement the coordinator signed.
    """

    def __init__(
        self,
        *,
        placement: str = "abcdef0123456789",
        snapshots: Sequence[Mapping[str, Any]] | None = None,
        healthy: bool = True,
        report: Mapping[str, Any] | None = None,
        activate_error: BaseException | None = None,
        assignments: Sequence[Mapping[str, Any]] = (),
        models: Sequence[Mapping[str, Any]] | None = None,
        models_error: BaseException | None = None,
        profile_error: BaseException | None = None,
        observe_reply: Mapping[str, Any] | None = None,
        observe_error: BaseException | None = None,
        probe_reply: Mapping[str, Any] | None = None,
        probe_error: BaseException | None = None,
        stream_observe_reply: Mapping[str, Any] | None = None,
        stream_observe_error: BaseException | None = None,
    ) -> None:
        self.placement = placement
        self.snapshots = list(snapshots or [staging_snapshot("completed")])
        self.healthy = healthy
        self.report = report if report is not None else activation_report()
        self.activate_error = activate_error
        self.assignments = list(assignments)
        self.plans: list[Any] = []
        self.prepared: list[Any] = []
        self.staged: list[tuple[Any, str, PreparedDeployment]] = []
        self.activated: list[Any] = []
        self.health_checks = 0
        self._reads = 0
        # The coordinator's catalogue, and the settings writes the assistant
        # performed on it. ``models`` defaults to a model that already has
        # sampling values, so no test writes by accident. The shape is the real
        # admin catalogue's (``list_models``): sampling lives nested under
        # ``settings`` (``asdict(ModelSettings)``), never at the top level.
        self.models = (
            list(models)
            if models is not None
            else [
                {
                    "id": "qwen",
                    "model_path": MODEL_PATH,
                    "settings": {"temperature": 0.7},
                }
            ]
        )
        self.models_error = models_error
        self.profile_error = profile_error
        self.profile_writes: list[tuple[str, Mapping[str, Any]]] = []
        # The post-ready measurement completion. A test that turns the probe
        # on gets this timing reply unless it asks for a failure.
        self.observe_reply = (
            dict(observe_reply)
            if observe_reply is not None
            else {
                "usage": {
                    "time_to_first_token": 0.25,
                    "prompt_tokens_per_second": 320.0,
                    "generation_tokens_per_second": 42.0,
                }
            }
        )
        self.observe_error = observe_error
        self.observed_calls: list[str] = []
        self.probe_reply = None if probe_reply is None else dict(probe_reply)
        self.probe_error = probe_error
        self.probed: list[Any] = []
        self.reloads = 0
        self.paired_nodes: list[str] | None = None
        self.stream_observe_reply = (
            None if stream_observe_reply is None else dict(stream_observe_reply)
        )
        self.stream_observe_error = stream_observe_error
        self.stream_observed_calls: list[str] = []

    def ops(self, plan: Any) -> CoordinatorOps:
        self.plans.append(plan)

        def prepare(request: Any) -> PreparedDeployment:
            self.prepared.append(request)
            return PreparedDeployment(
                plan={
                    "placement_signature": self.placement,
                    "assignments": list(self.assignments),
                }
            )

        def stage(
            prepared: PreparedDeployment, request: Any, source_host: str
        ) -> Mapping[str, Any]:
            self.staged.append((request, source_host, prepared))
            return {"job_id": "0123456789abcdef01234567", "status": "queued"}

        def models() -> Mapping[str, Any]:
            if self.models_error is not None:
                raise self.models_error
            return {"models": list(self.models)}

        def write_model_settings(model_id: str, patch: Mapping[str, Any]) -> Any:
            if self.profile_error is not None:
                raise self.profile_error
            self.profile_writes.append((model_id, dict(patch)))
            return {"success": True, "model_id": model_id}

        def staging_status(job_id: str) -> Mapping[str, Any] | None:
            index = min(self._reads, len(self.snapshots) - 1)
            self._reads += 1
            return self.snapshots[index]

        def observe(model_id: str) -> Mapping[str, Any]:
            self.observed_calls.append(model_id)
            if self.observe_error is not None:
                raise self.observe_error
            return dict(self.observe_reply)

        def probe(request: Any) -> Mapping[str, Any]:
            self.probed.append(request)
            if self.probe_error is not None:
                raise self.probe_error
            return dict(self.probe_reply or {})

        def reload() -> Mapping[str, Any]:
            self.reloads += 1
            return {"status": "ok"}

        def devices() -> Mapping[str, Any]:
            if self.paired_nodes is None:
                paired = [
                    node_id
                    for node_id in plan.rank_order
                    if node_id != plan.coordinator.node_id
                ]
            else:
                paired = list(self.paired_nodes)
            return {"paired": [{"node_id": node_id} for node_id in paired]}

        def stream_observe(model_id: str) -> Mapping[str, Any]:
            self.stream_observed_calls.append(model_id)
            if self.stream_observe_error is not None:
                raise self.stream_observe_error
            return dict(self.stream_observe_reply or {})

        async def health() -> Mapping[str, Any]:
            self.health_checks += 1
            return {"ok": self.healthy, "message": "the elected coordinator"}

        async def activate(request: Any) -> Mapping[str, Any]:
            self.activated.append(request)
            if self.activate_error is not None:
                raise self.activate_error
            return self.report

        return CoordinatorOps(
            local=False,
            target="192.0.2.11",
            prepare=prepare,
            stage=stage,
            staging_status=staging_status,
            health=health,
            activate=activate,
            models=models,
            write_model_settings=write_model_settings,
            devices=devices,
            reload=reload,
            observe=observe,
            probe=(
                probe
                if self.probe_reply is not None or self.probe_error is not None
                else None
            ),
            stream_observe=(
                stream_observe
                if self.stream_observe_reply is not None
                or self.stream_observe_error is not None
                else None
            ),
        )

    def factory(self) -> Callable[[Any], CoordinatorOps]:
        return self.ops


def unexpected_coordinator(plan: Any) -> CoordinatorOps:
    raise AssertionError("a refused request must not reach the coordinator")


@contextlib.contextmanager
def assistant_api(
    tmp_path: Any,
    monkeypatch: Any,
    *,
    coordinator: Callable[[Any], CoordinatorOps] | None = None,
    device: Callable[[str], dict[str, Any] | None] = default_device,
    local_addresses: Sequence[str] = (),
    inventory: Callable[..., list[dict[str, Any]]] | None = None,
    performance_probe: bool = False,
    settings_writer: Callable[..., None] | None = None,
    global_settings_writer: Callable[..., None] | None = None,
    model_settings: Callable[[str], Mapping[str, Any]] | None = None,
    transports: Callable[[Sequence[str]], TransportMatrix] | None = None,
):
    """A ``TestClient`` over the real router, with every Mac-facing seam faked."""

    monkeypatch.setitem(provisioning.__dict__, "inspect_safetensors_layout", model_layout)
    # The cluster-wide "is the coordinator already running something?" read is
    # a real SSH round trip in production. Tests answer "unanswerable" (not
    # busy) so no test reaches a Mac or waits for an SSH timeout.
    monkeypatch.setitem(
        assistant_routes.__dict__, "active_operation", lambda plan: ""
    )
    # The staging poll is the only sleep in these jobs; keep the suite quick.
    monkeypatch.setitem(assistant_start.__dict__, "POLL_SECONDS", 0.01)
    store = MeasurementStore(tmp_path / "store")
    operations = OperationStore(tmp_path / "store")
    assistant_routes.configure_assistant_measurements(
        store,
        probes=probes(
            device=device,
            inventory=inventory,
            model_settings=model_settings,
            transports=transports,
        ),
    )
    assistant_routes.configure_assistant_performance(
        AssistantPerformanceStore(tmp_path / "store")
    )
    assistant_routes.configure_assistant_tuning(
        assistant_routes.TuningPresetStore(tmp_path / "store")
    )
    assistant_routes.configure_assistant_operations(
        operations,
        provisioning=ProvisioningSeams(
            device_record=device,
            local_addresses=lambda: tuple(local_addresses),
            settings_writer=settings_writer,
            global_settings_writer=global_settings_writer,
        ),
        coordinator=coordinator or unexpected_coordinator,
    )
    server_settings = SimpleNamespace(
        port=8099,
        host="127.0.0.1",
        assistant_performance_probe=performance_probe,
    )
    global_settings = SimpleNamespace(server=server_settings, save=lambda: None)
    assistant_routes.set_assistant_getters(
        lambda: SimpleNamespace(base_path=tmp_path / "store"),
        lambda: server_settings,
        lambda: global_settings,
    )
    app = FastAPI()
    app.include_router(assistant_routes.router)
    with TestClient(app) as client:
        yield client, store, operations


def take_measurement(client: TestClient, **overrides: Any) -> str:
    response = client.post(MEASURE_URL, json=measure_payload(**overrides))
    assert response.status_code == 200, response.text
    measurement_id = response.json()["measurement_id"]
    status = client.get(STATUS_URL, params={"measurement_id": measurement_id})
    assert status.status_code == 200
    assert status.json()["state"] == "ready", status.text
    return measurement_id


def take_plan(client: TestClient, measurement_id: str, **overrides: Any) -> dict:
    response = client.post(PLAN_URL, json=plan_payload(measurement_id, **overrides))
    assert response.status_code == 200, response.text
    return response.json()


def apply_payload(
    measurement_id: str, plan: Mapping[str, Any], **overrides: Any
) -> dict[str, Any]:
    policy = plan.get("resource_policy") or {}
    payload: dict[str, Any] = {
        "measurement_id": measurement_id,
        "plan_signature": plan.get("plan_signature") or "a" * 64,
        "model_path": MODEL_PATH,
        "context_tokens": policy.get("context_tokens")
        or plan.get("context_tokens", 8192),
        "target_context_tokens": policy.get("target_context_tokens")
        or plan.get("context_tokens", 8192),
        "rank0_extra_bytes": 0,
        "mtp": plan.get("mtp", False),
        "execution_profile": "balanced",
        "coordinator_node": plan.get("coordinator_node", "m5"),
        "rank_order": plan.get("rank_order", ["m5", "m1"]),
        "origin_node": "m1",
        "table_offload": policy.get("table_offload", "none"),
        "table_kind": policy.get("table_kind", ""),
        "moe_expert_offload": policy.get("moe_expert_offload", False),
        "moe_resident_fraction": policy.get("moe_resident_fraction", 1.0),
        "memory_guard_tier": policy.get("memory_guard_tier"),
        "admission": policy.get("admission", "p50"),
        # The browser spreads the frozen plan inputs into apply; the schema
        # accepts this plan-time field so that request is never a 422.
        "node_selection": "minimal",
    }
    payload.update(overrides)
    return payload


def start_payload(plan: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "measurement_id": plan["measurement_id"],
        "plan_signature": plan["plan_signature"],
        "coordinator_node": plan["coordinator_node"],
    }
    payload.update(overrides)
    return payload


def wait_for_operation(
    client: TestClient,
    operation_id: str,
    *,
    states: Sequence[str] = (STATE_APPLIED, "ready", "failed"),
    timeout: float = 30.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"{OPERATIONS_URL}/{operation_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["state"] in states:
            return body
        time.sleep(0.01)
    raise AssertionError(f"operation {operation_id} stayed {body.get('state')}")
