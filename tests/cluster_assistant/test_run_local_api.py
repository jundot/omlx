# SPDX-License-Identifier: Apache-2.0
"""The /run-local route: serve the measured model on this Mac alone.

"Run on this Mac" is its own flow, not a plan: ``plan`` keeps the PP-only
cluster contract and a single-Mac plan stays unapplicable. These tests drive the
real router with every Mac-facing seam faked (AGENTS.md): the measurement, the
local admin seam, the tuning store and the engine pool's answers are all
in-process, so no test touches a model, a server or a real endpoint.
"""

from __future__ import annotations

import time

import pytest
from assistant_harness import MODEL_PATH, SSH, take_measurement

from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant.start import RemoteAdminError
from omlx.cluster.assistant.tuning import hardware_key
from omlx.cluster.assistant.tuning_store import TuningPreset

RUN_LOCAL_URL = "/admin/api/cluster/assistant/run-local"

GIB = 1024**3


def _payload(measurement_id: str, **overrides):
    body = {
        "measurement_id": measurement_id,
        "model_path": MODEL_PATH,
        "layer_count": 20,
        "context_tokens": 32768,
        "mtp": False,
    }
    body.update(overrides)
    return body


def _poll(client, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    body: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"{RUN_LOCAL_URL}/{job_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["state"] in ("ready", "failed"):
            return body
        time.sleep(0.01)
    raise AssertionError(f"local run {job_id} stayed {body.get('state')}")


class FakeLocalAdmin:
    """This Mac's own admin seam: the catalogue, settings, load, unload, canary."""

    def __init__(
        self,
        *,
        loaded: bool = True,
        answers: bool = True,
        distributed: bool = False,
    ) -> None:
        self.loaded = loaded
        self.answers = answers
        self.distributed = distributed
        self.calls: list[tuple[str, str, object]] = []
        self.load_error: BaseException | None = None
        self.unload_error: BaseException | None = None
        self.canary_error: BaseException | None = None

    def __call__(self, port, path, body=None, *, method="GET", timeout=30.0):
        assert port == 8099
        self.calls.append((method, path, body))
        base = path.split("?", 1)[0]
        if base.endswith("/settings"):
            return {"success": True, "model_id": "qwen"}
        if base.endswith("/load"):
            if self.load_error is not None:
                raise self.load_error
            self.loaded = True
            return {"status": "ok", "model_id": "qwen"}
        if base.endswith("/unload"):
            if self.unload_error is not None:
                raise self.unload_error
            self.loaded = False
            return {"status": "ok", "model_id": "qwen"}
        if base == "/v1/chat/completions":
            if self.canary_error is not None:
                raise self.canary_error
            if not self.answers:
                return {"choices": []}
            return {"choices": [{"message": {"content": "ok"}}]}
        if base == "/admin/api/models":
            return {
                "models": [
                    {
                        "id": "qwen",
                        "model_path": MODEL_PATH,
                        "loaded": self.loaded,
                        "is_loading": False,
                        "distributed": self.distributed,
                    }
                ]
            }
        raise AssertionError(f"unexpected local admin path {path}")


@pytest.fixture()
def fast_job(monkeypatch):
    """Keep the load poll short; the real values are minutes."""

    monkeypatch.setitem(assistant_routes.__dict__, "LOCAL_RUN_POLL_SECONDS", 0.01)
    monkeypatch.setitem(assistant_routes.__dict__, "LOCAL_RUN_LOAD_TIMEOUT_SECONDS", 1.0)


@pytest.fixture(autouse=True)
def clean_local_run_jobs():
    """The job table is module-global; one test must not guard the next."""

    with assistant_routes._local_run_jobs_lock:
        assistant_routes._local_run_jobs.clear()
    yield
    with assistant_routes._local_run_jobs_lock:
        assistant_routes._local_run_jobs.clear()


def _save_preset() -> None:
    assistant_routes._tuning_store().save(
        TuningPreset(
            model_signature=MODEL_PATH,
            hardware_key=hardware_key(["m5"]),
            mtp_depth=3,
            sampler={"temperature": 0.6, "top_p": 0.95, "top_k": 0},
        )
    )


def test_run_local_applies_the_preset_loads_and_reports_this_macs_endpoint(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    measurement_id = take_measurement(client)
    _save_preset()
    admin = FakeLocalAdmin(loaded=False)
    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", admin)
    # Never SSH another Mac for this flow.
    monkeypatch.setitem(
        assistant_routes.__dict__,
        "remote_admin",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("remote admin used")),
    )

    response = client.post(RUN_LOCAL_URL, json=_payload(measurement_id))
    assert response.status_code == 200, response.text
    handle = response.json()
    assert handle["state"] == "queued"
    assert handle["job_id"].startswith("local-")
    assert handle["poll"].endswith(handle["job_id"])

    body = _poll(client, handle["job_id"])
    assert body["state"] == "ready", body
    assert body["model_id"] == "qwen"
    assert body["context_tokens"] == 32768
    assert body["mtp"] is False
    assert body["endpoint"] == {
        "host": SSH["m5"],
        "port": 8099,
        "url": f"http://{SSH['m5']}:8099",
        "ready": True,
    }
    assert body["error"] is None
    assert [warning["code"] for warning in body["warnings"]] == [
        "tuning_preset_applied"
    ]

    # The preset is written through this Mac's own settings route before the load.
    writes = [
        (method, path, payload)
        for method, path, payload in admin.calls
        if path.endswith("/settings")
    ]
    assert writes == [
        (
            "PUT",
            "/admin/api/models/qwen/settings",
            {"mtp_adaptive_max_depth": 3, "temperature": 0.6, "top_p": 0.95, "top_k": 0},
        )
    ]
    paths = [path for _, path, _ in admin.calls]
    assert paths.index("/admin/api/models/qwen/settings") < paths.index(
        "/admin/api/models/qwen/load?local_only=true"
    )
    # Every catalogue/settings/load call is this Mac's own admin API; the one
    # other call is the readiness completion on the same loopback.
    assert all(
        path.startswith("/admin/api/models") or path == "/v1/chat/completions"
        for path in paths
    )
    # A local run is always a local-only load, never a cluster deployment.
    loads = [path for _, path, _ in admin.calls if path.endswith("/load?local_only=true")]
    assert loads == ["/admin/api/models/qwen/load?local_only=true"]


def test_run_local_never_targets_a_mac_the_measurement_does_not_prove_is_local(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=("10.0.0.99",))
    measurement_id = take_measurement(client)
    admin = FakeLocalAdmin()
    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", admin)

    response = client.post(RUN_LOCAL_URL, json=_payload(measurement_id))
    assert response.status_code == 200, response.text
    body = _poll(client, response.json()["job_id"])

    assert body["state"] == "failed"
    assert body["endpoint"] is None
    assert body["error"]["code"] == "origin_not_measured"
    assert body["error"]["action"] == "measure"
    # Nothing was loaded: the flow refuses before it reaches this Mac's pool.
    assert admin.calls == []


def test_a_model_that_does_not_fit_this_mac_alone_asks_for_a_smaller_model(
    api, monkeypatch, fast_job
) -> None:
    from omlx.cluster.planner import ModelLayout

    client, _, _ = api(local_addresses=(SSH["m5"],))
    monkeypatch.setitem(
        assistant_routes.__dict__,
        "model_layout_for",
        lambda identity: ModelLayout(
            source=MODEL_PATH,
            fixed_weight_bytes=1 * GIB,
            layer_weight_bytes=(3 * GIB,) * 40,
            supports_pipeline=True,
        ),
    )
    measurement_id = take_measurement(client)
    admin = FakeLocalAdmin()
    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", admin)

    response = client.post(RUN_LOCAL_URL, json=_payload(measurement_id, layer_count=40))
    assert response.status_code == 200, response.text
    body = _poll(client, response.json()["job_id"])

    assert body["state"] == "failed"
    assert body["error"]["code"] == "insufficient_memory"
    assert body["error"]["action"] == "choose_smaller_model"
    assert admin.calls == []


def test_a_context_this_mac_cannot_hold_says_so_with_one_action(
    api, monkeypatch, fast_job
) -> None:
    from omlx.cluster.planner import ModelLayout

    client, _, _ = api(local_addresses=(SSH["m5"],))
    monkeypatch.setitem(
        assistant_routes.__dict__,
        "model_layout_for",
        lambda identity: ModelLayout(
            source=MODEL_PATH,
            fixed_weight_bytes=1 * GIB,
            layer_weight_bytes=(1 * GIB,) * 20,
            # 20 layers x 1 MiB/token: a long context is what does not fit.
            kv_bytes_per_token_per_layer=1024 * 1024,
            supports_pipeline=True,
        ),
    )
    measurement_id = take_measurement(client)
    admin = FakeLocalAdmin()
    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", admin)

    response = client.post(RUN_LOCAL_URL, json=_payload(measurement_id))
    assert response.status_code == 200, response.text
    body = _poll(client, response.json()["job_id"])

    assert body["state"] == "failed"
    assert body["error"]["code"] == "context_too_large"
    assert body["error"]["action"] == "adjust_model_or_context"
    # The refusal names the context that does fit instead of silently lowering it.
    assert "below the requested 32,768" in body["error"]["message"]
    assert admin.calls == []


def test_a_load_the_pool_refuses_is_one_action_and_never_ready(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    measurement_id = take_measurement(client)
    admin = FakeLocalAdmin()
    admin.load_error = RemoteAdminError(507, "model exceeds the wired limit")
    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", admin)

    response = client.post(RUN_LOCAL_URL, json=_payload(measurement_id))
    body = _poll(client, response.json()["job_id"])

    assert body["state"] == "failed"
    assert body["endpoint"] is None
    assert body["error"]["action"] == "adjust_model_or_context"
    # The preset still ran first: it is a preference, not a gate.
    assert [warning["code"] for warning in body["warnings"]] == [
        "tuning_preset_skipped"
    ]


def test_a_second_local_run_is_refused_while_one_is_queued(api, fast_job) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    measurement_id = take_measurement(client)
    running = assistant_routes._create_local_run_job()
    try:
        response = client.post(RUN_LOCAL_URL, json=_payload(measurement_id))
        assert response.status_code == 409
        assert response.json()["code"] == "local_run_in_progress"
        assert response.json()["action"] == "status"
        assert running in response.json()["message"]
    finally:
        assistant_routes._update_local_run_job(running, state="failed")


def test_an_unexpected_job_failure_is_reported_and_never_left_running(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    measurement_id = take_measurement(client)

    def boom(*args, **kwargs):
        raise RuntimeError("a malformed layout")

    monkeypatch.setitem(assistant_routes.__dict__, "_single_node_fit", boom)
    response = client.post(RUN_LOCAL_URL, json=_payload(measurement_id))
    body = _poll(client, response.json()["job_id"])

    assert body["state"] == "failed"
    assert body["error"]["code"] == "local_run_failed"
    assert body["error"]["action"] == "diagnostics"
    # The one-load-at-a-time guard is open again.
    assert assistant_routes._active_local_run_job() == ""


def test_the_status_route_refuses_an_unknown_job() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(assistant_routes.router)
    client = TestClient(app)
    response = client.get(f"{RUN_LOCAL_URL}/local-000000000000")
    assert response.status_code == 409
    assert response.json()["code"] == "local_run_not_found"
    assert response.json()["action"] == "status"


# ---------------------------------------------------------------------------
# local_only, table offload, canary and stop
# ---------------------------------------------------------------------------


def _table_layout():
    """A checkpoint whose lookup table the plan keeps on SSD (Qwen4 PLE shape)."""

    from omlx.cluster.planner import ModelLayout

    per_layer = 1 * GIB // 20
    return ModelLayout(
        source=MODEL_PATH,
        fixed_weight_bytes=0,
        layer_weight_bytes=(per_layer,) * 20,
        # Half of each layer is a PLE table the plan offloads by default.
        tensor_parallel_ssd_layer_bytes=(per_layer // 2,) * 20,
        table_offload_kind="ple",
        supports_pipeline=True,
    )


def test_run_local_asks_for_a_local_only_load_and_writes_the_table_offload(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    monkeypatch.setitem(assistant_routes.__dict__, "model_layout_for", lambda identity: _table_layout())
    measurement_id = take_measurement(client)
    admin = FakeLocalAdmin(loaded=False)
    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", admin)

    response = client.post(RUN_LOCAL_URL, json=_payload(measurement_id, layer_count=20))
    body = _poll(client, response.json()["job_id"])

    assert body["state"] == "ready", body
    # The fit assumed the PLE table on SSD, so the run writes that switch before
    # the load — exactly what Start does for the same lever.
    writes = [
        payload
        for method, path, payload in admin.calls
        if method == "PUT" and path.endswith("/settings")
    ]
    assert writes == [{"qwen4_ple_ssd_offload": True}]
    warnings = [warning["code"] for warning in body["warnings"]]
    assert "table_offload_enabled" in warnings
    # And the load itself is single-node by contract.
    assert "/admin/api/models/qwen/load?local_only=true" in [
        path for _, path, _ in admin.calls
    ]


def test_a_request_past_the_checkpoints_declared_context_is_named_not_clamped(
    api, monkeypatch, fast_job
) -> None:
    from omlx.cluster.planner import ModelLayout

    client, _, _ = api(local_addresses=(SSH["m5"],))
    monkeypatch.setitem(
        assistant_routes.__dict__,
        "model_layout_for",
        lambda identity: ModelLayout(
            source=MODEL_PATH,
            fixed_weight_bytes=0,
            layer_weight_bytes=(1 * GIB,) * 20,
            max_context_tokens=16384,
            supports_pipeline=True,
        ),
    )
    measurement_id = take_measurement(client)
    admin = FakeLocalAdmin()
    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", admin)

    response = client.post(RUN_LOCAL_URL, json=_payload(measurement_id))
    body = _poll(client, response.json()["job_id"])

    assert body["state"] == "failed"
    assert body["error"]["code"] == "context_too_large"
    assert "16,384" in body["error"]["message"]
    assert admin.calls == []


def test_a_distributed_engine_is_never_reported_as_a_local_run(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    measurement_id = take_measurement(client)
    admin = FakeLocalAdmin(loaded=False, distributed=True)
    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", admin)

    response = client.post(RUN_LOCAL_URL, json=_payload(measurement_id))
    body = _poll(client, response.json()["job_id"])

    assert body["state"] == "failed"
    assert body["endpoint"] is None
    assert body["error"]["code"] == "local_engine_required"
    assert body["error"]["action"] == "status"


def test_a_model_that_loads_but_does_not_answer_is_not_ready(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    measurement_id = take_measurement(client)
    admin = FakeLocalAdmin(loaded=False, answers=False)
    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", admin)

    response = client.post(RUN_LOCAL_URL, json=_payload(measurement_id))
    body = _poll(client, response.json()["job_id"])

    assert body["state"] == "failed"
    assert body["endpoint"] is None
    assert body["error"]["code"] == "local_canary_failed"
    assert body["error"]["action"] == "diagnostics"
    # The canary is one minimal completion, never a speed claim.
    canaries = [
        (path, payload)
        for method, path, payload in admin.calls
        if path == "/v1/chat/completions"
    ]
    assert len(canaries) == 1
    assert canaries[0][1]["max_tokens"] == assistant_routes.LOCAL_RUN_CANARY_MAX_TOKENS


def _ready_job(client, monkeypatch, **fake_kwargs) -> tuple[FakeLocalAdmin, str]:
    measurement_id = take_measurement(client)
    admin = FakeLocalAdmin(loaded=False, **fake_kwargs)
    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", admin)
    response = client.post(RUN_LOCAL_URL, json=_payload(measurement_id))
    body = _poll(client, response.json()["job_id"])
    assert body["state"] == "ready", body
    return admin, body["job_id"]


def test_stop_unloads_this_macs_model_and_is_idempotent(api, monkeypatch, fast_job) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    admin, job_id = _ready_job(client, monkeypatch)

    response = client.post(f"{RUN_LOCAL_URL}/{job_id}/stop")
    assert response.status_code == 200, response.text
    assert response.json() == {"job_id": job_id, "stopped": True}
    assert "/admin/api/models/qwen/unload" in [path for _, path, _ in admin.calls]

    # The ready job keeps its state (the frozen machine has no "stopped") but
    # hides its endpoint and reports ``stopped``.
    polled = client.get(f"{RUN_LOCAL_URL}/{job_id}").json()
    assert polled["state"] == "ready"
    assert polled["stopped"] is True
    assert polled["endpoint"] is None

    # A second stop is not an error: the model is already unloaded.
    again = client.post(f"{RUN_LOCAL_URL}/{job_id}/stop")
    assert again.status_code == 200
    assert again.json()["stopped"] is True


def test_stop_finishes_an_already_unloaded_model_without_error(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    admin, job_id = _ready_job(client, monkeypatch)
    admin.unload_error = RemoteAdminError(400, "Model not loaded: qwen")

    response = client.post(f"{RUN_LOCAL_URL}/{job_id}/stop")

    assert response.status_code == 200
    assert response.json() == {"job_id": job_id, "stopped": True}


def test_stop_aborts_a_queued_job_and_reopens_the_guard(api, fast_job) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    job_id = assistant_routes._create_local_run_job()
    assert assistant_routes._active_local_run_job() == job_id

    response = client.post(f"{RUN_LOCAL_URL}/{job_id}/stop")

    assert response.status_code == 200
    assert response.json() == {"job_id": job_id, "stopped": True}
    polled = client.get(f"{RUN_LOCAL_URL}/{job_id}").json()
    assert polled["state"] == "failed"
    assert polled["stopped"] is True
    assert polled["error"]["code"] == "local_run_stopped"
    assert polled["error"]["action"] == "start"
    assert assistant_routes._active_local_run_job() == ""
    # A late load step cannot turn a stopped job ready.
    assistant_routes._update_local_run_job(job_id, state="ready")
    assert client.get(f"{RUN_LOCAL_URL}/{job_id}").json()["state"] == "failed"


def test_stop_refuses_an_unknown_job() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(assistant_routes.router)
    client = TestClient(app)
    response = client.post(f"{RUN_LOCAL_URL}/local-000000000000/stop")
    assert response.status_code == 409
    assert response.json()["code"] == "local_run_not_found"
