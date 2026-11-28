# SPDX-License-Identifier: Apache-2.0
"""Shared offline harness for the Cluster Assistant API tests.

Everything here is a fake: a measured pair of Macs, a synthetic model layout,
injected probes and a ``TestClient`` over the real assistant router. No test
touches SSH, a model on disk or a Mac.

It lives in its own module (not in a test file) so every API test module can
share it without importing one test module from another, which is also what
keeps the fixture visible to lint.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster.assistant import provisioning
from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant.measurement import (
    measure_budget_payloads,
    measurement_timestamp,
)
from omlx.cluster.assistant.performance_store import AssistantPerformanceStore
from omlx.cluster.assistant.provisioning import ProvisioningSeams
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


@contextlib.contextmanager
def assistant_api(
    tmp_path: Any,
    monkeypatch: Any,
    *,
    device: Callable[[str], dict[str, Any] | None] = default_device,
    local_addresses: Sequence[str] = (),
    inventory: Callable[..., list[dict[str, Any]]] | None = None,
    model_settings: Callable[[str], Mapping[str, Any]] | None = None,
    transports: Callable[[Sequence[str]], TransportMatrix] | None = None,
):
    """A ``TestClient`` over the real router, with every Mac-facing seam faked."""

    monkeypatch.setattr(provisioning, "inspect_safetensors_layout", model_layout)
    store = MeasurementStore(tmp_path / "store")
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
    # The delete relay resolves a Mac from the paired device record and this
    # Mac's own addresses, both faked here.
    assistant_routes.configure_assistant_seams(
        ProvisioningSeams(
            device_record=device,
            local_addresses=lambda: tuple(local_addresses),
        )
    )
    server_settings = SimpleNamespace(port=8099, host="127.0.0.1")
    assistant_routes.set_assistant_getters(
        lambda: SimpleNamespace(base_path=tmp_path / "store"),
        lambda: server_settings,
    )
    app = FastAPI()
    app.include_router(assistant_routes.router)
    with TestClient(app) as client:
        # No operation store exists in this slice; the third value keeps the
        # unpacking in the API tests identical to the run backend's harness.
        yield client, store, None


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
