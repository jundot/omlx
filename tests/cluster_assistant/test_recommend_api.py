# SPDX-License-Identifier: Apache-2.0
"""The recommendation API: which model should this measured cluster run?

These drive the real router with a ``TestClient``, a real ``/measure`` job (with
injected probes, so nothing reaches a Mac) and the candidate model layouts
injected the way ``tests/test_cluster_routes.py`` injects them. They pin:

* the answer is one model at one context, ranked from the stored hardware facts;
* the measurement's model identity is *not* compared with the candidates
  (PROJECT-BRIEF §11.1a) while ``plan`` keeps its strict check;
* a missing, unknown, stale or mismatched measurement is the same flat
  ``409 stale_measurement`` ``/plan`` returns, with no candidate list;
* no feasible candidate is ``409 no_feasible_candidate`` with the ranked near
  misses and exactly one action;
* recommendation reads no settings, runs no probe and writes nothing.

The admin gate is stubbed the way ``tests/test_cluster_ui_integration.py`` does
it. The real gates are attached in
``omlx/server.py::_register_cluster_routes``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from omlx.cluster.assistant import recommendation as recommendation_module
from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant.recommendation import ModelCandidate
from omlx.cluster.assistant.state import MeasurementStore
from omlx.cluster.node_role import NodeBudgetSuggestion
from omlx.cluster.performance import NodePerformanceProfile
from omlx.cluster.planner import ModelLayout
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
SSH = {"m5": "172.16.1.11", "m1": "172.16.1.14"}
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


def model_layout(
    *,
    total_gib: float = 40.0,
    layers: int = 20,
    supports_pipeline: bool = True,
    supports_tensor_parallel: bool = False,
) -> ModelLayout:
    """A checkpoint with a real KV shape, so the context ladder means something."""

    return ModelLayout(
        source="injected",
        fixed_weight_bytes=0,
        layer_weight_bytes=(int(total_gib * GIB) // layers,) * layers,
        supports_pipeline=supports_pipeline,
        supports_tensor_parallel=supports_tensor_parallel,
        kv_bytes_per_token_per_layer=4096,
    )


LAYOUTS = {
    "big": model_layout(total_gib=40, layers=20),
    "other": model_layout(total_gib=60, layers=20),
    "huge": model_layout(total_gib=400, layers=20),
    # Fits the measured pair, but its declared limit caps the context at 4k.
    "capped": model_layout(total_gib=120, layers=20),
    "tensor": model_layout(
        total_gib=40, supports_pipeline=False, supports_tensor_parallel=True
    ),
}


def fake_loader(
    model_path: str,
    *,
    friendly_name: str = "",
    declared_context_tokens: int | None = None,
) -> ModelCandidate:
    """The route's layout read, without touching a checkpoint.

    House style: ``tests/test_cluster_routes.py`` monkeypatches the layout
    reader the same way. A model named ``missing`` is the unreadable case; the
    real reader is exercised separately against a path that does not exist.
    """

    name = Path(model_path).name
    if name not in LAYOUTS:
        return ModelCandidate(
            model_path,
            friendly_name=friendly_name,
            declared_context_tokens=declared_context_tokens,
            layout_error="could not read the model: no such file or directory",
        )
    return ModelCandidate(
        model_path,
        friendly_name=friendly_name,
        declared_context_tokens=declared_context_tokens,
        layout=LAYOUTS[name],
    )


def recommend_payload(measurement_id: str, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "measurement_id": measurement_id,
        "candidates": [{"model_path": "/models/big", "friendly_name": "Big Model"}],
        "context_tokens": 8192,
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


@pytest.fixture()
def api(tmp_path, monkeypatch):
    stack = contextlib.ExitStack()
    monkeypatch.setitem(assistant_routes.__dict__, "load_model_candidate", fake_loader)

    def _build(
        *,
        settings: Any = DEFAULT_SETTINGS,
        fake_layouts: bool = True,
    ) -> tuple[TestClient, MeasurementStore]:
        if not fake_layouts:
            monkeypatch.setitem(
                assistant_routes.__dict__,
                "load_model_candidate",
                recommendation_module.load_model_candidate,
            )
        store = MeasurementStore(tmp_path / "store")
        assistant_routes.configure_assistant_measurements(store, probes=probes())
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
# The answer
# ---------------------------------------------------------------------------


def test_recommend_names_one_model_at_one_context(api) -> None:
    client, _ = api()
    measurement_id = take_measurement(client)

    response = client.post(RECOMMEND_URL, json=recommend_payload(measurement_id))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["feasible"] is True
    assert "code" not in body
    assert body["measurement_id"] == measurement_id
    assert body["measurement_timestamp"]
    assert body["measurement_model"].startswith("synthetic model")
    assert body["hardware_only"] is False
    assert body["target_context_tokens"] == 8192
    assert body["mtp"] is False
    assert body["execution_profile"] == "balanced"
    assert body["selected_model"] == "/models/big"
    assert body["selected_context_tokens"] == 8192
    assert len(body["reasons"]) == 1

    (model,) = body["candidates"]
    assert model["label"] == "Big Model"
    assert model["model_path"] == "/models/big"
    assert model["feasible"] is True
    assert model["pp_supported"] is True
    assert model["pipeline_stages"] == 2
    assert model["nodes_required"] == 2
    assert model["rank0_node"] == "m5"
    assert model["rank_order"] == ["m5", "m1"]
    assert model["max_context_tokens"] == 8192
    assert model["headroom_gib"] > 0
    assert model["download_gib"] == pytest.approx(40.0)
    assert model["failure_kind"] == ""
    assert model["action"] == ""
    assert model["measured_speed"] is True
    assert model["critical_stage_seconds"] > 0
    assert [stage["rank"] for stage in model["stages"]] == [0, 1]
    assert {node["node_id"] for node in model["nodes"]} == {"m5", "m1"}
    # Stage seconds only — no token rate, before or after the smoke test.
    assert "tok_per_s" not in str(body)


def test_recommend_ignores_the_measured_model_identity(api) -> None:
    """§11.1a: recommendation may evaluate a model the measurement did not cover."""

    client, _ = api()
    measurement_id = take_measurement(client)

    response = client.post(
        RECOMMEND_URL,
        json=recommend_payload(
            measurement_id, candidates=[{"model_path": "/models/other"}]
        ),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["feasible"] is True
    assert body["selected_model"] == "/models/other"
    # The hardware is what was measured; the model is not.
    assert body["measurement_model"].startswith("synthetic model")
    assert body["selected_context_tokens"] == 8192


def test_a_context_target_below_the_memory_limit_is_honoured(api) -> None:
    client, _ = api()
    measurement_id = take_measurement(client)

    body = client.post(
        RECOMMEND_URL, json=recommend_payload(measurement_id, context_tokens=4096)
    ).json()

    assert body["selected_context_tokens"] == 4096
    assert body["candidates"][0]["max_context_tokens"] == 4096


def test_the_declared_context_limit_is_respected(api) -> None:
    client, _ = api()
    measurement_id = take_measurement(client)

    body = client.post(
        RECOMMEND_URL,
        json=recommend_payload(
            measurement_id,
            candidates=[
                {
                    "model_path": "/models/big",
                    "declared_context_tokens": 4096,
                }
            ],
        ),
    ).json()

    model = body["candidates"][0]
    assert model["max_context_tokens"] == 4096
    assert model["declared_context_tokens"] == 4096


def test_a_capped_huge_model_ranks_below_the_usable_one(api) -> None:
    """B11 through the real router: usable context beats raw checkpoint size.

    The 120 GiB checkpoint fits these Macs but declares a 4k limit; the 40 GiB
    one honours the 32k target. The answer must select the usable model and
    list it first, while the capped one stays a feasible runner-up at 4k.
    """

    client, _ = api()
    measurement_id = take_measurement(client)

    response = client.post(
        RECOMMEND_URL,
        json=recommend_payload(
            measurement_id,
            context_tokens=32768,
            candidates=[
                {
                    "model_path": "/models/capped",
                    "friendly_name": "Capped Giant",
                    "declared_context_tokens": 4096,
                },
                {"model_path": "/models/big", "friendly_name": "Big Model"},
            ],
        ),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["feasible"] is True
    assert body["selected_model"] == "/models/big"
    assert body["selected_context_tokens"] == 32768
    first, second = body["candidates"]
    assert first["model_id"] == "/models/big"
    assert first["max_context_tokens"] == 32768
    assert second["model_id"] == "/models/capped"
    assert second["feasible"] is True
    assert second["max_context_tokens"] == 4096
    assert second["weight_bytes"] > first["weight_bytes"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_no_feasible_candidate_is_409_with_the_ranked_near_misses(api) -> None:
    client, _ = api()
    measurement_id = take_measurement(client)

    response = client.post(
        RECOMMEND_URL,
        json=recommend_payload(
            measurement_id, candidates=[{"model_path": "/models/huge"}]
        ),
    )

    assert response.status_code == assistant_routes.NO_FEASIBLE_CANDIDATE_STATUS == 409
    body = response.json()
    assert body["code"] == "no_feasible_candidate"
    assert body["action"] == recommendation_module.ACTION_CHOOSE_SMALLER_MODEL
    assert body["message"]
    assert body["feasible"] is False
    assert body["selected_model"] == ""
    assert body["selected_context_tokens"] is None
    # The near misses come with the refusal: no second request needed.
    (model,) = body["candidates"]
    assert model["feasible"] is False
    assert model["failure_kind"] == recommendation_module.FAILURE_MEMORY
    assert model["action"] == recommendation_module.ACTION_CHOOSE_SMALLER_MODEL
    assert body["measurement_id"] == measurement_id


def test_a_tensor_parallel_candidate_is_refused_with_its_own_action(api) -> None:
    client, _ = api()
    measurement_id = take_measurement(client)

    response = client.post(
        RECOMMEND_URL,
        json=recommend_payload(
            measurement_id, candidates=[{"model_path": "/models/tensor"}]
        ),
    )

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "no_feasible_candidate"
    assert body["action"] == recommendation_module.ACTION_CHOOSE_PIPELINE_MODEL
    (model,) = body["candidates"]
    assert model["pp_supported"] is False
    assert model["failure_kind"] == recommendation_module.FAILURE_PIPELINE_UNSUPPORTED
    assert model["action"] == recommendation_module.ACTION_CHOOSE_PIPELINE_MODEL


def test_an_unreadable_model_is_reported_by_the_real_reader(api, tmp_path) -> None:
    """The candidate is not dropped: it comes back with a reason and an action."""

    client, _ = api(fake_layouts=False)
    measurement_id = take_measurement(client)
    missing = str(tmp_path / "not-a-model")

    response = client.post(
        RECOMMEND_URL,
        json=recommend_payload(measurement_id, candidates=[{"model_path": missing}]),
    )

    assert response.status_code == 409
    body = response.json()
    (model,) = body["candidates"]
    assert model["model_id"] == missing
    assert model["feasible"] is False
    assert model["failure_kind"] == recommendation_module.FAILURE_MODEL_UNREADABLE
    assert model["action"] == recommendation_module.ACTION_CHECK_MODEL_PATH
    assert model["reasons"][0].startswith("could not read the model:")
    assert body["action"] == recommendation_module.ACTION_CHECK_MODEL_PATH


def test_an_unknown_measurement_is_the_same_stale_409_as_plan(api) -> None:
    client, _ = api()

    response = client.post(RECOMMEND_URL, json=recommend_payload("ms-does-not-exist"))

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "stale_measurement"
    assert body["action"] == "measure"
    assert body["reason"] == "unknown"
    assert body["measurement_id"] == "ms-does-not-exist"
    assert "candidates" not in body
    assert "selected_model" not in body


def test_a_stale_measurement_is_the_same_stale_409_as_plan(api) -> None:
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

    response = client.post(RECOMMEND_URL, json=recommend_payload(measurement_id))

    assert response.status_code == 409
    assert response.json()["code"] == "stale_measurement"
    assert response.json()["reason"] == "age"
    assert response.json()["action"] == "measure"
    assert "candidates" not in response.json()


# ---------------------------------------------------------------------------
# Request contract
# ---------------------------------------------------------------------------


def test_a_missing_measurement_id_is_422(api) -> None:
    client, _ = api()
    body = recommend_payload("unused")
    del body["measurement_id"]

    assert client.post(RECOMMEND_URL, json=body).status_code == 422


def test_unknown_request_fields_are_refused(api) -> None:
    client, _ = api()
    measurement_id = take_measurement(client)

    response = client.post(
        RECOMMEND_URL, json=recommend_payload(measurement_id, tensor_parallel_size=2)
    )

    assert response.status_code == 422
    assert "tensor_parallel_size" in response.text


def test_an_empty_or_oversized_candidate_list_is_422(api) -> None:
    client, _ = api()
    measurement_id = take_measurement(client)

    empty = client.post(
        RECOMMEND_URL, json=recommend_payload(measurement_id, candidates=[])
    )
    oversized = client.post(
        RECOMMEND_URL,
        json=recommend_payload(
            measurement_id,
            candidates=[
                {"model_path": "/models/big"}
                for _ in range(recommendation_module.MAX_RECOMMENDATION_CANDIDATES + 1)
            ],
        ),
    )

    assert empty.status_code == 422
    assert oversized.status_code == 422


def test_a_duplicate_model_path_is_refused(api) -> None:
    client, store = api()
    measurement_id = take_measurement(client)

    response = client.post(
        RECOMMEND_URL,
        json=recommend_payload(
            measurement_id,
            candidates=[{"model_path": "/models/big"}, {"model_path": "/models/big"}],
        ),
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_recommend_request"
    assert "more than once" in response.json()["detail"]["message"]
    # A refused request writes nothing and answers nothing.
    assert store.get(measurement_id) is not None


# ---------------------------------------------------------------------------
# Purity and surface
# ---------------------------------------------------------------------------


def test_recommending_is_idempotent_and_writes_nothing(api) -> None:
    client, store = api()
    measurement_id = take_measurement(client)
    body = recommend_payload(measurement_id)
    stored = store.path.read_bytes()

    first = client.post(RECOMMEND_URL, json=body)
    second = client.post(RECOMMEND_URL, json=body)

    assert first.status_code == 200, first.text
    assert first.json() == second.json()
    # Reading the measurement must not rewrite it.
    assert store.path.read_bytes() == stored


def test_recommend_never_reads_settings_or_runs_a_probe(api, monkeypatch) -> None:
    """The one rule that keeps a recommendation honest: it measures nothing."""

    client, _ = api()
    measurement_id = take_measurement(client)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("recommend must not read settings or run a probe")

    monkeypatch.setitem(assistant_routes.__dict__, "_server_settings", forbidden)
    monkeypatch.setitem(assistant_routes.__dict__, "_measure_probes", forbidden)
    monkeypatch.setitem(assistant_routes.__dict__, "default_measure_probes", forbidden)
    monkeypatch.setitem(assistant_routes.__dict__, "measurement_timestamp", forbidden)

    response = client.post(RECOMMEND_URL, json=recommend_payload(measurement_id))

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
        (APPLY_URL, ["POST"]),
        (START_URL, ["POST"]),
        # Screen 6: stop the one deployment a ready start launched.
        (STOP_URL, ["POST"]),
        # The automatic constellation search loop (API-first; UI later).
        (SEARCH_URL, ["POST"]),
        (SEARCH_URL + "/{job_id}", ["GET"]),
        # Fine-tune the winning constellation with real completions.
        (TUNE_URL, ["POST"]),
        # Tune the per-model knobs (MTP on/off + depth) by redeploying each
        # config once; the winner is saved per (model, node config).
        (REDEPLOY_URL, ["POST"]),
        (REDEPLOY_URL + "/{job_id}", ["GET"]),
        # Read-only: the saved tuning preset for an operation's model + node set.
        (TUNING_PRESET_URL, ["GET"]),
        # Run the measured model on this Mac alone (single-node inference, no
        # cluster): an async job, because the engine-pool load blocks.
        (RUN_LOCAL_URL, ["POST"]),
        (RUN_LOCAL_URL + "/{job_id}", ["GET"]),
        (RUN_LOCAL_URL + "/{job_id}/stop", ["POST"]),
        # Tune the model that ready local run serves: the sampler per request,
        # then every MTP config by reloading it on this Mac alone.
        (TUNE_LOCAL_URL, ["POST"]),
        (TUNE_LOCAL_URL + "/{tune_job_id}", ["GET"]),
        # Read-only: the last persisted deployment for a reloaded card.
        (PERFORMANCE_PROBE_URL, ["GET"]),
        (PERFORMANCE_PROBE_URL, ["POST"]),
        (SESSION_URL, ["GET"]),
        # Read-only: the redacted support bundle for the last measurement and
        # operation (no key, SSH target, model path, prompt or completion).
        ("/admin/api/cluster/assistant/diagnostics", ["GET"]),
        # Read-only: local interface evidence and commands; never executed.
        ("/admin/api/cluster/assistant/network-preflight", ["GET"]),
        # Read-only: lets a second origin ask the elected coordinator whether it
        # is already running an apply/start (the cluster-wide operation lock).
        (OPERATIONS_URL + "/active", ["GET"]),
        # Server-to-server: the origin mirrors its running operation onto the
        # elected coordinator so a second origin sees the cluster-wide lock.
        (OPERATIONS_URL + "/mirror", ["POST"]),
        (OPERATIONS_URL + "/{operation_id}", ["GET"]),
        # The free-space affordance: one model, on one measured Mac, through that
        # Mac's own delete route (PHASE-2B-RUN.md §9).
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


def test_the_recommend_route_is_behind_admin_authentication(api) -> None:
    """The router has no auth of its own; the server mounts it gated."""

    def deny() -> None:
        raise HTTPException(401, "admin required")

    app = FastAPI()
    app.include_router(assistant_routes.router, dependencies=[Depends(deny)])
    client = TestClient(app)

    assert client.post(RECOMMEND_URL, json=recommend_payload("x")).status_code == 401
