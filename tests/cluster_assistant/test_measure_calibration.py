# SPDX-License-Identifier: Apache-2.0
"""The live measurement job's real-model calibration (fake-driven).

Decision 2026-11-13: ranking rests on REAL completions. The measure job runs
the small calibration model on every Mac that holds it — through the same
completion seam the post-``ready`` observe measurement uses — and folds the
result into the node ahead of the synthetic probe. These tests pin the wiring:

* a node whose seam answers gets the real profile, preferred over synthetic;
* a missing model or a failed calibration falls back to the synthetic probe
  (or unknown compute) and the measurement still reaches ``ready``;
* admission and memory are untouched — calibration measures compute only;
* the seam is the existing observe transport: local in-process for this Mac,
  the Mac's own admin seam over SSH for any other.

No real Mac, model or SSH here (AGENTS.md): the probes, the layout read and
the completion transport are fakes. Live evidence belongs in the live-run notes.
"""

from __future__ import annotations

import contextlib
import json
import time
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant import start as assistant_start
from omlx.cluster.assistant.calibration import (
    CALIBRATION_MAX_TOKENS,
    CALIBRATION_MODEL_DIR,
    CALIBRATION_PROMPT,
    CALIBRATION_WARMUP_PROMPT,
    CALIBRATION_WARMUP_TOKENS,
    calibration_body,
)
from omlx.cluster.assistant.state import MeasurementStore
from omlx.cluster.node_role import NodeBudgetSuggestion
from omlx.cluster.performance import NodePerformanceProfile
from omlx.cluster.planner import ModelLayout
from omlx.cluster.transport import TransportInfo, TransportMatrix

GIB = 1024**3
MEASURE_URL = "/admin/api/cluster/assistant/measure"
STATUS_URL = "/admin/api/cluster/assistant/status"

CAPACITY_GIB = {"m5": 107.5, "m1": 60.0}
USABLE_GIB = {"m5": 96.75, "m1": 54.0}
SSH = {"m5": "192.0.2.11", "m1": "192.0.2.14"}

# dev-1.5b is ~1 GB of weights; a round number keeps rates readable as GB/s.
WEIGHT_BYTES = 1_000_000_000
CALIBRATION_PATH = str(Path(CALIBRATION_MODEL_DIR).expanduser())
CALIBRATION_LAYOUT = {
    "source": CALIBRATION_PATH,
    "fixed_weight_bytes": 160_000_000,
    "layer_weight_bytes": [30_000_000] * 28,
    "tensor_count": 200,
}
# The real rates the fake completion reports: 500 tok/s decode, 4000 tok/s
# prefill over a 10-token prompt -> 500 / 400 GB/s in weight bytes.
REAL_DECODE_RATE = 500e9
REAL_PREFILL_RATE = 400e9
# The synthetic probe's clearly different rate (2 GiB/s), so "which one won"
# is never ambiguous.
SYNTHETIC_DECODE_RATE = 2.0 * GIB


def usage_reply(**overrides: Any) -> dict[str, Any]:
    usage = {
        "generation_tokens_per_second": 500.0,
        "prompt_tokens_per_second": 4000.0,
        "time_to_first_token": 0.02,
        "prompt_tokens": 10,
    }
    usage.update(overrides)
    return {"usage": usage}


def synthetic_profile(node_id: str) -> NodePerformanceProfile:
    return NodePerformanceProfile(
        node_id=node_id,
        rank=0,
        decode_weight_bytes_per_second=SYNTHETIC_DECODE_RATE,
        prefill_weight_bytes_per_second=2 * SYNTHETIC_DECODE_RATE,
        collective_latency_seconds=0.0003,
        collective_bandwidth_bytes_per_second=6.6e9,
        backend="ring",
        measured_at="2026-11-03T12:00:00+00:00",
        samples=5,
    )


class FakeCompletion:
    """The per-node completion seam: records calls, replays scripted replies.

    Mirrors ``start.stream_completion``'s keyword contract. A reply that is an
    ``Exception`` instance is raised instead of returned, so a failed
    completion can be faked without a real transport.
    """

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, *, port: int, body: Mapping[str, Any], target: str = ""
    ) -> Mapping[str, Any]:
        self.calls.append({"port": port, "body": dict(body), "target": target})
        reply = self.replies[len(self.calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply


def measure_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "nodes": [
            {"node_id": "m5", "ssh": SSH["m5"]},
            {"node_id": "m1", "ssh": SSH["m1"]},
        ],
        "roles": {"m5": "headless", "m1": "headless"},
        "model_path": "/models/qwen",
    }
    payload.update(overrides)
    return payload


def probes(
    *,
    inventory: Callable[..., list[dict[str, Any]]] | None = None,
    model_layout: Callable[..., dict[str, Any]] | None = None,
    performance: Callable[..., Any] | None = None,
    device: Callable[[str], dict[str, Any] | None] | None = None,
) -> assistant_routes.MeasureProbes:
    def admission_ceiling(ssh: str, python_executable: str | None = None) -> int:
        node_id = next(key for key, value in SSH.items() if value == ssh)
        return int(CAPACITY_GIB[node_id] * GIB)

    def budget(**kwargs: Any) -> NodeBudgetSuggestion:
        ssh = kwargs.get("ssh_target")
        # A loopback target is this Mac; the fake measures it as the m5 shape.
        node_id = next(
            (key for key, value in SSH.items() if value == ssh), "m5"
        )
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
        model_inventory=inventory
        or (lambda ssh, python_executable=None: [
            {"model_path": "/models/qwen", "id": "qwen"}
        ]),
        model_layout=model_layout,
        disk_free=lambda path, ssh_target=None: 300 * GIB,
        device_record=device
        or (lambda node_id: {"last_addrs": [SSH[node_id]]} if node_id in SSH else None),
        performance=performance,
    )


def calibration_inventory(ssh: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Both Macs hold the measured model and the calibration model."""

    return [
        {"model_path": "/models/qwen", "id": "qwen"},
        {"model_path": CALIBRATION_PATH, "id": "dev-1.5b"},
    ]


def calibration_layout(ssh: str, path: str) -> dict[str, Any]:
    return dict(CALIBRATION_LAYOUT)


@pytest.fixture()
def client(tmp_path):
    stack = contextlib.ExitStack()

    def _build(
        *,
        settings: Any = None,
        **kwargs: Any,
    ) -> tuple[TestClient, MeasurementStore]:
        store = MeasurementStore(tmp_path / "store")
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
    client: TestClient, measurement_id: str, *, timeout: float = 30.0
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


def run_measurement(
    api: TestClient, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    measurement_id = api.post(MEASURE_URL, json=payload or measure_payload()).json()[
        "measurement_id"
    ]
    return wait_for_state(api, measurement_id)


def facts_by_node(store: MeasurementStore, measurement_id: str) -> dict[str, Any]:
    record = store.get(measurement_id)
    assert record is not None
    return {node.node_id: node for node in record.measurement_set().nodes}


# ---------------------------------------------------------------------------
# The job folds a real calibration into each node (preferred over synthetic)
# ---------------------------------------------------------------------------


def test_the_measure_job_folds_a_real_calibration_into_each_node(
    client, monkeypatch
) -> None:
    seam = FakeCompletion(usage_reply(), usage_reply(), usage_reply(), usage_reply())
    monkeypatch.setattr(assistant_routes, "stream_completion", seam)
    api, store = client(
        inventory=calibration_inventory,
        model_layout=calibration_layout,
        performance=lambda nodes: [synthetic_profile("m5"), synthetic_profile("m1")],
    )

    status = run_measurement(
        api, measure_payload(include_performance_probe=True)
    )

    assert status["state"] == "ready"
    assert status["error"] is None
    facts = facts_by_node(store, status["measurement_id"])
    for node_id in ("m5", "m1"):
        compute = facts[node_id].compute
        assert compute is not None
        # The REAL rates, not the synthetic 2 GiB/s probe.
        assert compute.decode_weight_bytes_per_second == pytest.approx(REAL_DECODE_RATE)
        assert compute.prefill_weight_bytes_per_second == pytest.approx(
            REAL_PREFILL_RATE
        )
    # A calibrated Mac is measured: no unmeasured-compute warning.
    codes = {warning["code"] for warning in status["warnings"]}
    assert "unmeasured_compute" not in codes

    # Two completions per Mac (warmup discarded, measured second), each on the
    # right Mac's own port, with the calibration model's serving id.
    assert len(seam.calls) == 4
    for node_id, chunk in (("m5", seam.calls[:2]), ("m1", seam.calls[2:])):
        warmup, measured = chunk
        assert warmup["target"] == SSH[node_id]
        assert measured["target"] == SSH[node_id]
        assert warmup["port"] == measured["port"] == 8099
        assert warmup["body"]["messages"][0]["content"] == CALIBRATION_WARMUP_PROMPT
        assert warmup["body"]["max_tokens"] == CALIBRATION_WARMUP_TOKENS
        assert measured["body"]["model"] == "dev-1.5b"
        assert measured["body"]["messages"][0]["content"] == CALIBRATION_PROMPT
        assert measured["body"]["max_tokens"] == CALIBRATION_MAX_TOKENS
        assert measured["body"]["stream"] is True
        assert measured["body"]["stream_options"] == {"include_usage": True}


def test_calibration_alone_measures_compute_without_a_synthetic_probe(
    client, monkeypatch
) -> None:
    seam = FakeCompletion(usage_reply(), usage_reply(), usage_reply(), usage_reply())
    monkeypatch.setattr(assistant_routes, "stream_completion", seam)
    api, store = client(
        inventory=calibration_inventory, model_layout=calibration_layout
    )

    status = run_measurement(api)

    assert status["state"] == "ready"
    facts = facts_by_node(store, status["measurement_id"])
    for node_id in ("m5", "m1"):
        assert facts[node_id].compute is not None
        assert facts[node_id].compute.decode_weight_bytes_per_second == pytest.approx(
            REAL_DECODE_RATE
        )
    codes = {warning["code"] for warning in status["warnings"]}
    assert "unmeasured_compute" not in codes


# ---------------------------------------------------------------------------
# Unavailable / failed calibration: synthetic fallback, measurement still ready
# ---------------------------------------------------------------------------


def test_a_mac_without_the_calibration_model_keeps_the_synthetic_probe(
    client, monkeypatch
) -> None:
    seam = FakeCompletion()
    monkeypatch.setattr(assistant_routes, "stream_completion", seam)
    api, store = client(
        performance=lambda nodes: [synthetic_profile("m5"), synthetic_profile("m1")],
    )

    status = run_measurement(
        api, measure_payload(include_performance_probe=True)
    )

    assert status["state"] == "ready"
    facts = facts_by_node(store, status["measurement_id"])
    for node_id in ("m5", "m1"):
        compute = facts[node_id].compute
        assert compute is not None
        assert compute.decode_weight_bytes_per_second == pytest.approx(
            SYNTHETIC_DECODE_RATE
        )
    # No calibration model listed: no completion was even attempted.
    assert seam.calls == []


def test_a_failed_calibration_falls_back_to_the_synthetic_probe(
    client, monkeypatch
) -> None:
    seam = FakeCompletion(
        RuntimeError("the node did not answer"),
        RuntimeError("the node did not answer"),
    )
    monkeypatch.setattr(assistant_routes, "stream_completion", seam)
    api, store = client(
        inventory=calibration_inventory,
        model_layout=calibration_layout,
        performance=lambda nodes: [synthetic_profile("m5"), synthetic_profile("m1")],
    )

    status = run_measurement(
        api, measure_payload(include_performance_probe=True)
    )

    assert status["state"] == "ready"
    assert status["error"] is None
    facts = facts_by_node(store, status["measurement_id"])
    for node_id in ("m5", "m1"):
        assert facts[node_id].compute.decode_weight_bytes_per_second == pytest.approx(
            SYNTHETIC_DECODE_RATE
        )
    # A failed warmup cannot prove a warm measurement: one attempt per Mac.
    assert len(seam.calls) == 2


def test_a_reply_without_timing_falls_back_to_the_synthetic_probe(
    client, monkeypatch
) -> None:
    seam = FakeCompletion(usage_reply(), {}, usage_reply(), {})
    monkeypatch.setattr(assistant_routes, "stream_completion", seam)
    api, store = client(
        inventory=calibration_inventory,
        model_layout=calibration_layout,
        performance=lambda nodes: [synthetic_profile("m5"), synthetic_profile("m1")],
    )

    status = run_measurement(
        api, measure_payload(include_performance_probe=True)
    )

    assert status["state"] == "ready"
    facts = facts_by_node(store, status["measurement_id"])
    for node_id in ("m5", "m1"):
        assert facts[node_id].compute.decode_weight_bytes_per_second == pytest.approx(
            SYNTHETIC_DECODE_RATE
        )


def test_a_failed_calibration_without_a_probe_is_unknown_not_a_failure(
    client, monkeypatch
) -> None:
    seam = FakeCompletion(RuntimeError("model not loaded"))
    monkeypatch.setattr(assistant_routes, "stream_completion", seam)
    api, store = client(
        inventory=calibration_inventory, model_layout=calibration_layout
    )

    status = run_measurement(api)

    assert status["state"] == "ready"
    assert status["error"] is None
    facts = facts_by_node(store, status["measurement_id"])
    for node_id in ("m5", "m1"):
        assert facts[node_id].compute is None
        # Admission survived the failed calibration.
        assert facts[node_id].admission_p05_gib == pytest.approx(USABLE_GIB[node_id])
    codes = {warning["code"] for warning in status["warnings"]}
    assert "unmeasured_compute" in codes


def test_an_unreadable_calibration_layout_falls_back(client, monkeypatch) -> None:
    def broken_layout(ssh: str, path: str) -> dict[str, Any]:
        raise RuntimeError("the checkpoint headers could not be read")

    seam = FakeCompletion()
    monkeypatch.setattr(assistant_routes, "stream_completion", seam)
    api, store = client(
        inventory=calibration_inventory,
        model_layout=broken_layout,
        performance=lambda nodes: [synthetic_profile("m5"), synthetic_profile("m1")],
    )

    status = run_measurement(
        api, measure_payload(include_performance_probe=True)
    )

    assert status["state"] == "ready"
    facts = facts_by_node(store, status["measurement_id"])
    for node_id in ("m5", "m1"):
        assert facts[node_id].compute.decode_weight_bytes_per_second == pytest.approx(
            SYNTHETIC_DECODE_RATE
        )
    # Without weight bytes there is nothing to convert: no completion ran.
    assert seam.calls == []


# ---------------------------------------------------------------------------
# Admission / memory are untouched
# ---------------------------------------------------------------------------


def test_calibration_does_not_change_admission_or_memory(client, monkeypatch) -> None:
    seam = FakeCompletion(usage_reply(), usage_reply(), usage_reply(), usage_reply())
    monkeypatch.setattr(assistant_routes, "stream_completion", seam)
    api, store = client(
        inventory=calibration_inventory, model_layout=calibration_layout
    )
    calibrated = run_measurement(api)
    assert calibrated["state"] == "ready"

    # The same cluster measured without the calibration model present.
    plain_api, plain_store = client()
    plain = run_measurement(plain_api)
    assert plain["state"] == "ready"

    with_real = facts_by_node(store, calibrated["measurement_id"])
    without = facts_by_node(plain_store, plain["measurement_id"])
    assert set(with_real) == set(without)
    for node_id in with_real:
        real, base = with_real[node_id], without[node_id]
        assert real.capacity_gib == base.capacity_gib
        assert real.admission_p05_gib == base.admission_p05_gib
        assert real.admission_p50_gib == base.admission_p50_gib
        assert real.admission_p95_gib == base.admission_p95_gib
        assert real.memory_guard_tier == base.memory_guard_tier
        assert real.model_present == base.model_present
        # Only compute differs: real rates instead of unknown.
        assert base.compute is None
        assert real.compute is not None


# ---------------------------------------------------------------------------
# Hardware-only entry measurement and the local Mac
# ---------------------------------------------------------------------------


def test_a_hardware_only_measurement_still_calibrates(client, monkeypatch) -> None:
    """The C3 entry measurement has no model path, but ranking still needs
    real compute: the calibration reads the catalogue itself to find dev-1.5b."""

    inventory_calls: list[str] = []

    def inventory(ssh: str, **kwargs: Any) -> list[dict[str, Any]]:
        inventory_calls.append(ssh)
        return calibration_inventory(ssh)

    seam = FakeCompletion(usage_reply(), usage_reply(), usage_reply(), usage_reply())
    monkeypatch.setattr(assistant_routes, "stream_completion", seam)
    api, store = client(inventory=inventory, model_layout=calibration_layout)

    status = run_measurement(
        api, measure_payload(model_path=None, model_size_bytes=None)
    )

    assert status["state"] == "ready"
    assert "hardware only" in status["model"]
    facts = facts_by_node(store, status["measurement_id"])
    for node_id in ("m5", "m1"):
        assert facts[node_id].compute is not None
        assert facts[node_id].compute.decode_weight_bytes_per_second == pytest.approx(
            REAL_DECODE_RATE
        )
        # The calibration's catalogue read did not change the hardware-only
        # inventory semantics: model_paths stays "not measured".
        assert facts[node_id].model_paths == ()
    assert sorted(inventory_calls) == sorted([SSH["m5"], SSH["m1"]])
    assert len(seam.calls) == 4


def test_a_local_mac_calibrates_in_process(client, monkeypatch) -> None:
    """This Mac's seam is the in-process loopback call, not SSH to itself."""

    layout = ModelLayout(
        source=CALIBRATION_PATH,
        fixed_weight_bytes=CALIBRATION_LAYOUT["fixed_weight_bytes"],
        layer_weight_bytes=tuple(CALIBRATION_LAYOUT["layer_weight_bytes"]),
    )
    monkeypatch.setattr(
        assistant_routes, "inspect_safetensors_layout", lambda path: layout
    )
    seam = FakeCompletion(usage_reply(), usage_reply())
    monkeypatch.setattr(assistant_routes, "stream_completion", seam)
    api, store = client(
        inventory=calibration_inventory,
        device=lambda node_id: None,
        settings=SimpleNamespace(port=9123, host="10.0.0.5"),
    )

    status = run_measurement(
        api,
        measure_payload(nodes=[{"node_id": "m5", "ssh": "127.0.0.1"}]),
    )

    assert status["state"] == "ready"
    facts = facts_by_node(store, status["measurement_id"])
    assert facts["m5"].compute is not None
    assert facts["m5"].compute.decode_weight_bytes_per_second == pytest.approx(
        REAL_DECODE_RATE
    )
    assert len(seam.calls) == 2
    # Empty target = this Mac; the port is this Mac's own settings port.
    assert all(call["target"] == "" for call in seam.calls)
    assert all(call["port"] == 9123 for call in seam.calls)


def test_the_calibration_model_is_matched_by_folder_name_on_another_home(
    client, monkeypatch
) -> None:
    """Each Mac expands ``~`` to its own home; the folder name is the stable
    identity, and the layout is read where the model actually lives."""

    peer_path = "/Users/peer/Models/dev-1.5b"

    def inventory(ssh: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"model_path": peer_path, "id": "dev-1.5b"}]

    layout_paths: list[tuple[str, str]] = []

    def layout(ssh: str, path: str) -> dict[str, Any]:
        layout_paths.append((ssh, path))
        return dict(CALIBRATION_LAYOUT)

    seam = FakeCompletion(usage_reply(), usage_reply(), usage_reply(), usage_reply())
    monkeypatch.setattr(assistant_routes, "stream_completion", seam)
    api, store = client(inventory=inventory, model_layout=layout)

    status = run_measurement(api)

    assert status["state"] == "ready"
    facts = facts_by_node(store, status["measurement_id"])
    for node_id in ("m5", "m1"):
        assert facts[node_id].compute is not None
    # The layout was read at the peer's own path, not the origin's expansion.
    assert layout_paths == [(SSH["m5"], peer_path), (SSH["m1"], peer_path)]
    assert all(call["body"]["model"] == "dev-1.5b" for call in seam.calls)


# ---------------------------------------------------------------------------
# The seam adapter reuses the existing observe transport
# ---------------------------------------------------------------------------


def test_stream_completion_routes_local_in_process_and_remote_over_ssh(
    monkeypatch,
) -> None:
    calls: list[tuple[str, int, dict[str, Any]]] = []

    def local(port: int, model_id: str = "", *, body: Any = None) -> dict[str, Any]:
        calls.append(("local", port, dict(body)))
        return {"usage": {}}

    def remote(
        target: str, port: int, model_id: str = "", *, body: Any = None
    ) -> dict[str, Any]:
        calls.append((target, port, dict(body)))
        return {"usage": {}}

    monkeypatch.setattr(assistant_start, "_local_stream_observe", local)
    monkeypatch.setattr(assistant_start, "remote_stream_observe", remote)
    body = calibration_body("dev-1.5b", prompt="hi", max_tokens=8)

    assert assistant_start.stream_completion(port=8099, body=body) == {"usage": {}}
    assert assistant_start.stream_completion(
        port=9123, body=body, target="192.0.2.11"
    ) == {"usage": {}}
    assert calls == [("local", 8099, body), ("192.0.2.11", 9123, body)]


def test_remote_stream_observe_sends_an_explicit_body_through_the_ssh_seam(
    monkeypatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(ssh_target, snippet, argument, **kwargs):
        seen["ssh_target"] = ssh_target
        seen["argument"] = json.loads(argument)
        return {"ok": True, "usage": {"generation_tokens_per_second": 500.0}}

    monkeypatch.setattr(assistant_start, "run_remote_python", fake_run)
    body = calibration_body("dev-1.5b", prompt="hi", max_tokens=8)

    reply = assistant_start.remote_stream_observe(
        "192.0.2.11", 8099, "", body=body
    )

    assert reply == {"usage": {"generation_tokens_per_second": 500.0}}
    assert seen["ssh_target"] == "192.0.2.11"
    assert seen["argument"]["body"] == body
    assert seen["argument"]["port"] == 8099
    # The key is still read on the Mac itself, never carried in the payload.
    assert "api_key" not in seen["argument"]


def test_remote_stream_observe_without_a_body_keeps_the_observe_request(
    monkeypatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(ssh_target, snippet, argument, **kwargs):
        seen["argument"] = json.loads(argument)
        return {"ok": True, "usage": {}}

    monkeypatch.setattr(assistant_start, "run_remote_python", fake_run)

    assistant_start.remote_stream_observe("192.0.2.11", 8099, "qwen")

    assert "body" not in seen["argument"]
    assert seen["argument"]["model"] == "qwen"
    assert seen["argument"]["prompt"] == assistant_start.OBSERVE_PROMPT
    assert seen["argument"]["max_tokens"] == assistant_start.OBSERVE_MAX_TOKENS


def test_local_stream_observe_sends_an_explicit_body(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    class FakeReply:
        def __enter__(self):
            return iter(
                [
                    b'data: {"usage": {"generation_tokens_per_second": 500.0}}\n',
                    b"data: [DONE]\n",
                ]
            )

        def __exit__(self, *exc: Any) -> bool:
            return False

    def fake_urlopen(request, timeout=None):
        seen["request"] = request
        seen["timeout"] = timeout
        return FakeReply()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        "omlx.settings.get_settings",
        lambda: SimpleNamespace(auth=SimpleNamespace(api_key="")),
    )
    body = calibration_body("dev-1.5b", prompt="hi", max_tokens=8)

    reply = assistant_start._local_stream_observe(8099, "", body=body)

    assert reply == {"usage": {"generation_tokens_per_second": 500.0}}
    request = seen["request"]
    assert json.loads(request.data) == body
    assert request.full_url == "http://127.0.0.1:8099/v1/chat/completions"
    # The existing observe timeout bounds the calibration too.
    assert seen["timeout"] == assistant_start.OBSERVE_TIMEOUT_SECONDS
