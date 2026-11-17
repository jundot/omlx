# SPDX-License-Identifier: Apache-2.0
"""The /tune-local route: tune the model a ready local run serves.

Tuning was cluster-only (``/tune`` and ``/tune-redeploy`` need a ready *start
operation*), so a single-node run had no preset. This route tunes the one Mac
that is actually serving: the sampler is swept per request on the loaded model,
and every MTP config (off / adaptive / depth 1..4) is written, unloaded, loaded
``local_only`` and measured with ONE real completion's
``usage.generation_tokens_per_second``.

Every Mac-facing seam is faked (AGENTS.md): the measurement, this Mac's admin
seam (catalogue, settings, load, unload, completions) and the tuning store are
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
TUNE_LOCAL_URL = "/admin/api/cluster/assistant/tune-local"

# Measured MTP rates this Mac "produces": depth 3 wins, exactly the live shape
# of PHASE-2B-RUN.md §27 (MTP on with a fixed depth beats off and adaptive).
RATES = {
    (False, None): 20.0,
    (True, None): 29.0,
    (True, 1): 22.0,
    (True, 2): 30.0,
    (True, 3): 31.5,
    (True, 4): 30.5,
}


class FakeLocalAdmin:
    """This Mac's own admin seam: catalogue, settings, load, unload, completions.

    The rate a completion reports is the one for the MTP settings currently
    loaded, so a config only gets a number if it was really loaded with those
    settings (and ``load_refusals``/``max_loads`` can make a config or the final
    winner load fail honestly).
    """

    def __init__(
        self,
        *,
        rates: dict | None = None,
        settings: dict | None = None,
        load_refusals: dict | None = None,
        no_rate: tuple = (),
        max_loads: int | None = None,
        sampler_no_rate: bool = False,
        unload_queued: bool = False,
    ) -> None:
        self.settings = dict(
            settings if settings is not None else {"mtp_enabled": False}
        )
        self.rates = dict(RATES if rates is None else rates)
        self.load_refusals = dict(load_refusals or {})
        self.no_rate = set(no_rate)
        self.max_loads = max_loads
        self.sampler_no_rate = sampler_no_rate
        self.unload_queued = unload_queued
        self.loaded = True
        self.loaded_key = self._key()
        self.loads = 0
        self.calls: list[tuple[str, str, object]] = []

    def _key(self) -> tuple[bool, int | None]:
        return (
            bool(self.settings.get("mtp_enabled")),
            self.settings.get("mtp_adaptive_max_depth"),
        )

    def __call__(self, port, path, body=None, *, method="GET", timeout=30.0):
        assert port == 8099
        self.calls.append((method, path, body))
        base = path.split("?", 1)[0]
        if base.endswith("/settings"):
            assert method == "PUT"
            self.settings.update(body or {})
            return {"success": True, "model_id": "qwen"}
        if base.endswith("/load"):
            # A local tune only ever loads on this Mac alone.
            assert "local_only=true" in path, path
            key = self._key()
            if key in self.load_refusals:
                raise self.load_refusals[key]
            if self.max_loads is not None and self.loads >= self.max_loads:
                raise RemoteAdminError(507, "the wired limit is exhausted")
            self.loads += 1
            self.loaded = True
            self.loaded_key = key
            return {"status": "ok", "model_id": "qwen"}
        if base.endswith("/unload"):
            if self.unload_queued:
                # The real admin route answers 202 while it aborts the engine.
                return {"status": "unloading", "model_id": "qwen"}
            self.loaded = False
            return {"status": "ok", "model_id": "qwen"}
        if base == "/v1/chat/completions":
            return self._complete(body or {})
        if base == "/admin/api/models":
            return {
                "models": [
                    {
                        "id": "qwen",
                        "model_path": MODEL_PATH,
                        "loaded": self.loaded,
                        "is_loading": False,
                        "distributed": False,
                        "settings": dict(self.settings),
                    }
                ]
            }
        raise AssertionError(f"unexpected local admin path {path}")

    @staticmethod
    def _answer(prompt: str) -> str:
        lowered = prompt.lower()
        if "commas" in lowered:
            return "red, green, blue"
        if "17 + 25" in lowered:
            return "42"
        if "capital of france" in lowered:
            return "Paris"
        return "The ocean is wide and deep."

    def _complete(self, body: dict) -> dict:
        assert self.loaded, "a completion was measured on an unloaded model"
        prompt = str(((body.get("messages") or [{}])[0]).get("content") or "")
        answer = {"choices": [{"message": {"content": self._answer(prompt)}}]}
        if self.sampler_no_rate and "top_p" in body:
            # A server that reports no decode timing measures nothing; the
            # tuner must skip it and keep the stored sampler instead of a zero.
            return answer
        key = self.loaded_key
        rate = self.rates.get(key)
        if key in self.no_rate or rate is None:
            return answer
        temperature = float(body.get("temperature") or 0.0)
        # A deterministic sampler answers with fewer reasoning/output tokens,
        # which is the tuner's lexicographic tie-break after correctness.
        return {
            **answer,
            "usage": {
                "generation_tokens_per_second": rate,
                "completion_tokens": 8 if temperature == 0.0 else 12,
                "completion_tokens_details": {
                    "reasoning_tokens": 0 if temperature == 0.0 else 4
                },
            },
        }


@pytest.fixture(autouse=True)
def clean_local_jobs():
    """Both job tables are module-global; one test must not guard the next."""

    tables = (
        (assistant_routes._local_run_jobs, assistant_routes._local_run_jobs_lock),
        (assistant_routes._local_tune_jobs, assistant_routes._local_tune_jobs_lock),
    )
    for table, lock in tables:
        with lock:
            table.clear()
    yield
    for table, lock in tables:
        with lock:
            table.clear()


@pytest.fixture()
def fast_job(monkeypatch):
    """Keep the load/completion polls short; the real values are minutes."""

    monkeypatch.setitem(assistant_routes.__dict__, "LOCAL_RUN_POLL_SECONDS", 0.01)
    # The unload-wait and loaded-poll bound; the real value is minutes. Kept
    # short so the "the pool never drops it" test does not wait a real minute.
    monkeypatch.setitem(assistant_routes.__dict__, "LOCAL_RUN_LOAD_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setitem(assistant_routes.__dict__, "LOCAL_TUNE_POLL_SECONDS", 0.01)
    monkeypatch.setitem(assistant_routes.__dict__, "LOCAL_TUNE_TIMEOUT_SECONDS", 5.0)
    # Never SSH another Mac for this flow.
    monkeypatch.setitem(
        assistant_routes.__dict__,
        "remote_admin",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("remote admin used by a local flow")
        ),
    )


def _run_payload(measurement_id: str) -> dict:
    return {
        "measurement_id": measurement_id,
        "model_path": MODEL_PATH,
        "layer_count": 20,
        "context_tokens": 32768,
        "mtp": False,
    }


def _poll(client, url: str, job_id: str, *, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    body: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"{url}/{job_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["state"] in ("ready", "completed", "failed"):
            return body
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} stayed {body.get('state')}")


def _ready_run(
    client, monkeypatch, *, mtp_capable: bool = True, **fake_kwargs
) -> tuple[FakeLocalAdmin, str]:
    """A ready run-local job on this Mac, tuned against a fake admin seam."""

    measurement_id = take_measurement(client)
    admin = FakeLocalAdmin(**fake_kwargs)
    monkeypatch.setitem(assistant_routes.__dict__, "_local_admin", admin)
    monkeypatch.setitem(
        assistant_routes.__dict__, "_tune_mtp_capable", lambda model_path: mtp_capable
    )
    response = client.post(RUN_LOCAL_URL, json=_run_payload(measurement_id))
    assert response.status_code == 200, response.text
    body = _poll(client, RUN_LOCAL_URL, response.json()["job_id"])
    assert body["state"] == "ready", body
    return admin, body["job_id"]


def _tune(client, job_id: str) -> dict:
    response = client.post(TUNE_LOCAL_URL, json={"job_id": job_id})
    assert response.status_code == 200, response.text
    handle = response.json()
    assert handle["state"] == "queued"
    assert handle["tune_job_id"].startswith("local-tune-")
    assert handle["poll"] == f"{TUNE_LOCAL_URL}/{handle['tune_job_id']}"
    return _poll(client, TUNE_LOCAL_URL, handle["tune_job_id"])


def _settings_writes(admin: FakeLocalAdmin) -> list[dict]:
    return [
        dict(payload)
        for method, path, payload in admin.calls
        if method == "PUT" and path.endswith("/settings")
    ]


def test_tune_local_measures_the_sampler_and_every_mtp_config(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    admin, job_id = _ready_run(client, monkeypatch)

    body = _tune(client, job_id)

    assert body["state"] == "completed", body
    assert body["error"] is None
    assert body["total"] == 6
    assert body["done"] == 6
    assert [row["config"] for row in body["rows"]] == [
        "mtp_off",
        "mtp_on_adaptive",
        "mtp_on_depth_1",
        "mtp_on_depth_2",
        "mtp_on_depth_3",
        "mtp_on_depth_4",
    ]
    # Every row is a real measured rate, and the winner is the fastest.
    assert all(row["feasible"] for row in body["rows"])
    assert [row["decode_tok_per_s"] for row in body["rows"]] == [
        20.0,
        29.0,
        22.0,
        30.0,
        31.5,
        30.5,
    ]
    assert body["best"] == {
        "config": "mtp_on_depth_3",
        "mtp": True,
        "mtp_depth": 3,
    }
    # The sampler is the golden-prompt scoring, not a guess.
    assert body["sampler"]["sampler"] == "deterministic"
    assert body["sampler"]["quality"] == 1.0
    assert body["preset_key"]

    # Each config was written, unloaded and loaded locally before measuring.
    writes = _settings_writes(admin)
    assert {"mtp_enabled": False} in writes
    assert {"mtp_enabled": True, "mtp_adaptive_max_depth": None} in writes
    assert {"mtp_enabled": True, "mtp_adaptive_max_depth": 3} in writes
    # The run's own load + 6 config loads + the winner's load: the winner is
    # left serving (the run-local load happened before the tune started).
    loads = [
        path for _, path, _ in admin.calls if path.endswith("/load?local_only=true")
    ]
    assert len(loads) == 8
    assert admin.loaded is True
    assert admin.loaded_key == (True, 3)
    # The winning write carries the winner's MTP settings AND the sampler, so
    # the process serving now matches what was measured.
    assert writes[-1] == {
        "mtp_enabled": True,
        "mtp_adaptive_max_depth": 3,
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 20,
    }
    # MTP is a model setting here, never a per-request field (a per-request
    # sweep of it would measure noise — PHASE-2B-RUN §25).
    rate_bodies = [
        payload
        for _, path, payload in admin.calls
        if path == "/v1/chat/completions"
        and isinstance(payload, dict)
        and payload.get("messages", [{}])[0].get("content")
        == assistant_routes.LOCAL_TUNE_PROMPT
    ]
    assert len(rate_bodies) == 6
    assert all("mtp" not in payload for payload in rate_bodies)
    assert all(
        payload["max_tokens"] == assistant_routes.LOCAL_TUNE_MAX_TOKENS
        for payload in rate_bodies
    )


def test_tune_local_saves_one_preset_for_this_model_and_mac(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    _, job_id = _ready_run(client, monkeypatch)
    # A knobs value this Mac already had must be kept, not erased: the save
    # merges exactly like the cluster path.
    assistant_routes._tuning_store().save(
        TuningPreset(
            model_signature=MODEL_PATH,
            hardware_key=hardware_key(["m5"]),
            chunk=4096,
            mtp_depth=1,
            sampler={"sampler": "balanced", "temperature": 0.7},
        )
    )

    body = _tune(client, job_id)

    assert body["state"] == "completed", body
    preset = assistant_routes._tuning_store().get(MODEL_PATH, hardware_key(["m5"]))
    assert preset is not None
    assert preset.key == body["preset_key"]
    assert preset.hardware_key == "m5"
    assert preset.mtp is True
    assert preset.mtp_depth == 3
    assert preset.sampler["sampler"] == "deterministic"
    assert preset.measured_at
    # A knob this tuner does not measure keeps its stored value.
    assert preset.chunk == 4096


def test_a_model_without_an_mtp_head_only_measures_off(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    _, job_id = _ready_run(client, monkeypatch, mtp_capable=False)

    body = _tune(client, job_id)

    assert body["state"] == "completed", body
    assert body["mtp_applicable"] is False
    assert [row["config"] for row in body["rows"]] == ["mtp_off"]
    assert body["best"] == {"config": "mtp_off", "mtp": False, "mtp_depth": None}
    preset = assistant_routes._tuning_store().get(MODEL_PATH, hardware_key(["m5"]))
    assert preset is not None
    assert preset.mtp is False


def test_a_config_that_cannot_load_is_an_infeasible_row_not_a_rate(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    _, job_id = _ready_run(
        client,
        monkeypatch,
        load_refusals={
            (True, 3): RemoteAdminError(507, "the wired limit is exhausted")
        },
    )

    body = _tune(client, job_id)

    assert body["state"] == "completed", body
    rows = {row["config"]: row for row in body["rows"]}
    assert rows["mtp_on_depth_3"]["feasible"] is False
    assert rows["mtp_on_depth_3"]["decode_tok_per_s"] is None
    assert "did not fit this Mac's memory" in rows["mtp_on_depth_3"]["error"]
    # The rate the model never produced is never invented; the next best wins.
    assert body["best"]["config"] == "mtp_on_depth_4"
    assert body["best"]["mtp_depth"] == 4


def test_a_config_that_reports_no_rate_is_an_infeasible_row(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    _, job_id = _ready_run(client, monkeypatch, no_rate=((True, 2),))

    body = _tune(client, job_id)

    rows = {row["config"]: row for row in body["rows"]}
    assert rows["mtp_on_depth_2"]["feasible"] is False
    assert rows["mtp_on_depth_2"]["decode_tok_per_s"] is None
    assert rows["mtp_on_depth_2"]["error"] == "the completion reported no decode rate"
    assert body["best"]["config"] == "mtp_on_depth_3"


def test_a_sampler_sweep_without_rates_keeps_the_stored_sampler(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    _, job_id = _ready_run(client, monkeypatch, sampler_no_rate=True)
    assistant_routes._tuning_store().save(
        TuningPreset(
            model_signature=MODEL_PATH,
            hardware_key=hardware_key(["m5"]),
            sampler={"sampler": "loose", "temperature": 1.0},
        )
    )

    body = _tune(client, job_id)

    assert body["state"] == "completed", body
    # Nothing was measured for the sampler, so nothing is claimed.
    assert body["sampler"] is None
    assert body["sampler_rows"] == []
    preset = assistant_routes._tuning_store().get(MODEL_PATH, hardware_key(["m5"]))
    assert preset is not None
    assert preset.sampler["sampler"] == "loose"
    assert preset.mtp is True and preset.mtp_depth == 3


def test_a_winner_that_cannot_be_left_loaded_fails_the_job_with_one_action(
    api, monkeypatch, fast_job
) -> None:
    # The six config loads succeed; the winner's own reload (the seventh) is
    # refused, so the job must fail rather than claim the user is serving.
    client, _, _ = api(local_addresses=(SSH["m5"],))
    _, job_id = _ready_run(client, monkeypatch, max_loads=6)

    body = _tune(client, job_id)

    assert body["state"] == "failed"
    assert body["error"]["code"] == "insufficient_memory"
    assert body["error"]["action"] == "adjust_model_or_context"
    assert "could not be left loaded" in body["error"]["message"]
    # The measured rows are kept (they are real); no preset is saved for a
    # configuration that is not serving.
    assert len(body["rows"]) == 6
    assert body["preset_key"] == ""
    assert (
        assistant_routes._tuning_store().get(MODEL_PATH, hardware_key(["m5"])) is None
    )


def test_a_queued_unload_is_waited_for_instead_of_measuring_the_old_engine(
    api, monkeypatch, fast_job
) -> None:
    # The admin unload answers 202 "unloading" while it aborts the engine. Loading
    # immediately would hand back the engine with the PREVIOUS config's settings,
    # so no config may be measured (or labelled) until the unload is done.
    client, _, _ = api(local_addresses=(SSH["m5"],))
    _, job_id = _ready_run(client, monkeypatch, unload_queued=True)

    body = _tune(client, job_id)

    assert body["state"] == "failed"
    assert body["error"]["code"] == "local_tune_infeasible"
    assert body["error"]["action"] == "diagnostics"
    assert len(body["rows"]) == 6
    assert all(row["feasible"] is False for row in body["rows"])
    assert all(row["decode_tok_per_s"] is None for row in body["rows"])
    assert all("still unloading" in row["error"] for row in body["rows"])


def test_tuning_while_a_tune_is_running_is_refused(api, fast_job) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    running = assistant_routes._create_local_tune_job("local-x", total=2)
    try:
        response = client.post(TUNE_LOCAL_URL, json={"job_id": "local-x"})
        assert response.status_code == 409
        assert response.json()["code"] == "local_tune_in_progress"
        assert response.json()["action"] == "status"
        assert running in response.json()["message"]
    finally:
        assistant_routes._update_local_tune_job(running, working=False)


def test_a_local_run_is_refused_while_a_tune_holds_this_macs_pool(
    api, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    measurement_id = take_measurement(client)
    running = assistant_routes._create_local_tune_job("local-x", total=2)
    try:
        response = client.post(RUN_LOCAL_URL, json=_run_payload(measurement_id))
        assert response.status_code == 409
        assert response.json()["code"] == "local_tune_in_progress"
        assert response.json()["action"] == "status"
    finally:
        assistant_routes._update_local_tune_job(running, working=False)


def test_tune_local_refuses_an_unknown_run(api, fast_job) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    response = client.post(TUNE_LOCAL_URL, json={"job_id": "local-000000000000"})
    assert response.status_code == 409
    assert response.json()["code"] == "local_run_not_found"
    assert response.json()["action"] == "status"


def test_tune_local_needs_a_ready_run(api, fast_job) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    queued = assistant_routes._create_local_run_job()
    try:
        response = client.post(TUNE_LOCAL_URL, json={"job_id": queued})
        assert response.status_code == 409
        assert response.json()["code"] == "local_run_not_ready"
        assert response.json()["action"] == "status"
    finally:
        assistant_routes._update_local_run_job(queued, state="failed")


def test_a_stopped_run_is_tuned_only_after_starting_it_again(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    _, job_id = _ready_run(client, monkeypatch)
    assert client.post(f"{RUN_LOCAL_URL}/{job_id}/stop").status_code == 200

    response = client.post(TUNE_LOCAL_URL, json={"job_id": job_id})

    assert response.status_code == 409
    assert response.json()["code"] == "local_run_not_ready"
    assert response.json()["action"] == "start"


def test_an_unexpected_tune_failure_is_reported_and_reopens_the_guard(
    api, monkeypatch, fast_job
) -> None:
    client, _, _ = api(local_addresses=(SSH["m5"],))
    _, job_id = _ready_run(client, monkeypatch)

    async def boom(*args, **kwargs):
        raise RuntimeError("a malformed answer")

    # The job imports the sweep from the tuning module, so that is the seam.
    from omlx.cluster.assistant import tuning as assistant_tuning

    monkeypatch.setitem(assistant_tuning.__dict__, "sweep_samplers", boom)
    body = _tune(client, job_id)

    assert body["state"] == "failed"
    assert body["error"]["code"] == "local_tune_failed"
    assert body["error"]["action"] == "diagnostics"
    assert assistant_routes._active_local_tune_job() == ""


def test_the_status_route_refuses_an_unknown_tune_job() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(assistant_routes.router)
    client = TestClient(app)
    response = client.get(f"{TUNE_LOCAL_URL}/local-tune-000000000000")
    assert response.status_code == 409
    assert response.json()["code"] == "local_tune_not_found"
    assert response.json()["action"] == "status"
