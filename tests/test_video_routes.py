# SPDX-License-Identifier: Apache-2.0
"""Tests for the /v1/videos API routes (omlx/api/video_routes.py).

A minimal FastAPI app mounts the video router; the module-level accessors
(_get_video_manager / _get_engine_pool / _resolve_model) are monkeypatched.
get/list/delete semantics run against a REAL VideoJobManager constructed on
tmp_path with enforcer=None; only submit and the guard/venv probes are
stubbed per test. create_video also reads omlx.server._server_state
.global_settings.video inside the handler, so a settings stub is patched
onto the real ServerState instance (monkeypatch restores it afterwards).

No real model dirs, no ~/.fmlx, no worker subprocess is ever spawned.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import omlx.api.video_routes as video_routes
import omlx.server as omlx_server
from omlx.settings import VideoSettings
from omlx.video.manager import QueueFullError, VideoJob, VideoJobManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VIDEO_MODEL = "wan-t2v"
LLM_MODEL = "llama-llm"


def _video_settings(**overrides) -> VideoSettings:
    """Enabled settings with a lease big enough for the peak predictor.

    NOTE the dataclass default memory_lease_gb=28.0 is BELOW the predictor
    floor (_PEAK_BASE_GB 32 + _PEAK_MARGIN_GB 3 = 35GB), so default settings
    would 413 every request; tests pass an explicit lease.
    """
    params = dict(enabled=True, memory_lease_gb=64.0)
    params.update(overrides)
    return VideoSettings(**params)


def _make_manager(
    tmp_path: Path, settings: VideoSettings, stub_submit: bool = True
) -> VideoJobManager:
    """Real manager (real get/list_jobs/delete) with probe seams stubbed."""
    manager = VideoJobManager(
        settings=settings, base_path=tmp_path, enforcer=None
    )
    manager.guard_available = lambda: (True, "")  # type: ignore[method-assign]

    async def _probe(force: bool = False):
        return True, ""

    manager.probe_worker_venv = _probe  # type: ignore[method-assign]

    if stub_submit:
        submitted: list[VideoJob] = []

        async def _submit(job: VideoJob) -> VideoJob:
            # Record without waking the real dispatcher (no admission loop,
            # no subprocess)
            manager._jobs[job.id] = job
            submitted.append(job)
            return job

        manager.submit = _submit  # type: ignore[method-assign]
        manager.test_submitted = submitted  # type: ignore[attr-defined]
    return manager


def _seed_job(
    manager: VideoJobManager,
    job_id: str,
    created_at: float = 100.0,
    status: str = "queued",
    **kwargs,
) -> VideoJob:
    job = VideoJob(
        id=job_id,
        model_id=VIDEO_MODEL,
        model_dir="/nonexistent/model-dir",
        params={
            "prompt": "a cat",
            "width": 480,
            "height": 272,
            "frames": 49,
            "steps": 20,
            "fps": 16,
            "seed": 7,
            "seconds": 3.06,
        },
        status=status,
        created_at=created_at,
        **kwargs,
    )
    manager._jobs[job_id] = job
    return job


@pytest.fixture
def video_env(monkeypatch, tmp_path):
    """Builder returning (TestClient, manager) with accessors patched."""

    def build(
        settings: VideoSettings | None = None,
        stub_submit: bool = True,
        patch_manager_accessor: bool = True,
    ):
        vs = settings or _video_settings()
        manager = _make_manager(tmp_path, vs, stub_submit=stub_submit)

        entries = {
            VIDEO_MODEL: SimpleNamespace(
                model_path=tmp_path / "models" / "wan", model_type="video"
            ),
            LLM_MODEL: SimpleNamespace(
                model_path=tmp_path / "models" / "llama", model_type="llm"
            ),
        }
        pool = SimpleNamespace(get_entry=lambda mid: entries.get(mid))

        if patch_manager_accessor:
            monkeypatch.setattr(
                video_routes, "_get_video_manager", lambda: manager
            )
        monkeypatch.setattr(video_routes, "_get_engine_pool", lambda: pool)
        monkeypatch.setattr(video_routes, "_resolve_model", lambda m: m)
        # create_video reads _server_state.global_settings.video directly
        monkeypatch.setattr(
            omlx_server._server_state,
            "global_settings",
            SimpleNamespace(video=vs),
        )

        app = FastAPI()
        app.include_router(video_routes.router)
        return TestClient(app), manager

    return build


def _post(client: TestClient, **fields):
    body = {"model": VIDEO_MODEL, "prompt": "a cat"}
    body.update(fields)
    return client.post("/v1/videos", json=body)


# ---------------------------------------------------------------------------
# POST /v1/videos -- happy paths
# ---------------------------------------------------------------------------


class TestCreateVideo:
    def test_post_json_happy_path(self, video_env):
        client, manager = video_env()
        r = _post(client, size="480x272", seconds=3)
        assert r.status_code == 200
        body = r.json()
        assert body["id"].startswith("video_")
        assert body["object"] == "video"
        assert body["status"] == "queued"
        assert body["model"] == VIDEO_MODEL
        assert body["size"] == "480x272"
        # seconds=3 * default_fps=16 = 48 frames -> 4n+1 -> 49
        assert body["frames"] == 49
        # Derived seconds string = round(49/16, 2)
        assert body["seconds"] == "3.06"
        assert body["progress"] == 0
        assert body["error"] is None
        # Job actually reached the manager
        assert manager.get(body["id"]) is not None
        assert len(manager.test_submitted) == 1

    def test_post_multipart_all_string_fields(self, video_env):
        """openai SDK shape: multipart/form-data, every field a string."""
        client, manager = video_env()
        r = client.post(
            "/v1/videos",
            data={
                "model": VIDEO_MODEL,
                "prompt": "a cat",
                "seconds": "4",
                "steps": "10",
            },
            # File part forces multipart encoding; non-str form values are
            # filtered out by the handler
            files={"input_reference": ("ref.png", b"\x89PNG", "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "queued"
        assert body["steps"] == 10
        # "4" * fps 16 = 64 -> 4n+1 -> 65
        assert body["frames"] == 65
        assert body["seconds"] == str(round(65 / 16, 2))
        # Defaults applied when size omitted
        assert body["size"] == "480x272"

    def test_seed_and_explicit_params_pass_through(self, video_env):
        client, manager = video_env()
        r = _post(client, width=480, height=272, frames=49, seed=1234, fps=8)
        assert r.status_code == 200
        body = r.json()
        assert body["seed"] == 1234
        assert body["fps"] == 8
        job = manager.get(body["id"])
        assert job.params["seed"] == 1234


# ---------------------------------------------------------------------------
# POST /v1/videos -- model resolution errors
# ---------------------------------------------------------------------------


class TestCreateVideoModelErrors:
    def test_unknown_model_404(self, video_env):
        client, _ = video_env()
        r = _post(client, model="no-such-model")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"]

    def test_non_video_model_400(self, video_env):
        client, _ = video_env()
        r = _post(client, model=LLM_MODEL)
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "not a video generation model" in detail
        assert "model_type=llm" in detail

    def test_missing_prompt_400(self, video_env):
        client, _ = video_env()
        r = client.post("/v1/videos", json={"model": VIDEO_MODEL})
        assert r.status_code == 400

    def test_malformed_body_400(self, video_env):
        client, _ = video_env()
        r = client.post(
            "/v1/videos",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 400
        assert "Malformed request body" in r.json()["detail"]


# ---------------------------------------------------------------------------
# POST /v1/videos -- normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_dimensions_round_up_to_multiple_of_16(self, video_env):
        client, _ = video_env()
        r = _post(client, width=470, height=270)
        assert r.status_code == 200
        assert r.json()["size"] == "480x272"

    def test_frames_from_seconds_times_fps(self, video_env):
        client, _ = video_env()
        r = _post(client, seconds=3, fps=16)
        assert r.status_code == 200
        assert r.json()["frames"] == 49  # round(3*16)=48 -> 4n+1 -> 49

    def test_explicit_frames_rounded_to_4n_plus_1(self, video_env):
        client, _ = video_env()
        r = _post(client, frames=50)
        assert r.status_code == 200
        body = r.json()
        assert body["frames"] == 53  # 4*ceil(49/4)+1
        assert body["seconds"] == str(round(53 / 16, 2))

    def test_invalid_size_string_400(self, video_env):
        client, _ = video_env()
        r = _post(client, size="480by272")
        assert r.status_code == 400
        assert "Invalid size" in r.json()["detail"]

    def test_nonpositive_seconds_400(self, video_env):
        client, _ = video_env()
        r = _post(client, seconds=0)
        assert r.status_code == 400
        assert "seconds must be positive" in r.json()["detail"]


# ---------------------------------------------------------------------------
# POST /v1/videos -- static caps (400) and peak predictor (413)
# ---------------------------------------------------------------------------


class TestCapsAndPredictor:
    def test_steps_over_max_400(self, video_env):
        client, _ = video_env()  # default max_steps=50
        r = _post(client, steps=51)
        assert r.status_code == 400
        assert "max_steps" in r.json()["detail"]

    def test_pixels_over_max_400(self, video_env):
        client, _ = video_env()  # default cap 1280*720
        r = _post(client, width=1280, height=736)
        assert r.status_code == 400
        assert "max_pixels_per_frame" in r.json()["detail"]

    def test_frames_over_max_400(self, video_env):
        client, _ = video_env()  # default max_frames=121
        r = _post(client, frames=125)
        assert r.status_code == 400
        assert "max_frames" in r.json()["detail"]

    def test_peak_predictor_413_when_over_lease(self, video_env):
        # lease 36GB: predicted = 32 + 35*(80*45*21/1e6) = 34.65GB,
        # +3 margin = 37.65 > 36 -> 413
        client, _ = video_env(settings=_video_settings(memory_lease_gb=36.0))
        r = _post(client, width=1280, height=720, frames=81)
        assert r.status_code == 413
        detail = r.json()["detail"]
        assert "memory_lease_gb" in detail
        assert "Predicted memory peak" in detail

    def test_peak_predictor_small_request_fits_same_lease(self, video_env):
        # Same 36GB lease: 480x272x49 predicts 32.23 + 3 = 35.23 < 36 -> ok
        client, _ = video_env(settings=_video_settings(memory_lease_gb=36.0))
        r = _post(client, width=480, height=272, frames=49)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /v1/videos -- 503 gates
# ---------------------------------------------------------------------------


class TestServiceGates:
    def test_queue_full_503(self, video_env):
        # Real submit with max_queued_jobs=0 raises QueueFullError before
        # the dispatcher would start
        client, _ = video_env(
            settings=_video_settings(max_queued_jobs=0), stub_submit=False
        )
        r = _post(client)
        assert r.status_code == 503
        assert "queue is full" in r.json()["detail"].lower()

    def test_queue_full_error_importable_and_raised_by_submit(
        self, video_env
    ):
        _, manager = video_env(
            settings=_video_settings(max_queued_jobs=0), stub_submit=False
        )
        job = VideoJob(id="video_x", model_id="m", model_dir="d", params={})
        import asyncio

        with pytest.raises(QueueFullError):
            asyncio.run(manager.submit(job))

    def test_guard_unavailable_503(self, video_env):
        client, manager = video_env()
        manager.guard_available = lambda: (False, "guard is not running")
        r = _post(client)
        assert r.status_code == 503
        assert r.json()["detail"] == "guard is not running"

    def test_venv_probe_failure_503(self, video_env):
        client, manager = video_env()

        async def _probe(force: bool = False):
            return False, "Video worker python not found at /x"

        manager.probe_worker_venv = _probe
        r = _post(client)
        assert r.status_code == 503
        assert "worker python not found" in r.json()["detail"]

    def test_video_disabled_503(self, video_env, monkeypatch):
        # Do NOT patch _get_video_manager: the real accessor must gate on
        # settings.video.enabled via _server_state.global_settings
        client, _ = video_env(
            settings=_video_settings(enabled=False),
            patch_manager_accessor=False,
        )
        monkeypatch.setattr(
            omlx_server._server_state, "video_job_manager", None
        )
        r = _post(client)
        assert r.status_code == 503
        assert "disabled" in r.json()["detail"]
        # Every endpoint shares the gate
        assert client.get("/v1/videos").status_code == 503
        assert client.get("/v1/videos/video_x").status_code == 503
        assert client.delete("/v1/videos/video_x").status_code == 503

    def test_manager_missing_503(self, video_env, monkeypatch):
        # Enabled but lifespan never built the manager -> 503
        client, _ = video_env(patch_manager_accessor=False)
        monkeypatch.setattr(
            omlx_server._server_state, "video_job_manager", None
        )
        r = _post(client)
        assert r.status_code == 503
        assert "not initialized" in r.json()["detail"]


# ---------------------------------------------------------------------------
# GET /v1/videos/{id}
# ---------------------------------------------------------------------------


class TestGetVideo:
    def test_get_unknown_404(self, video_env):
        client, _ = video_env()
        r = client.get("/v1/videos/video_doesnotexist")
        assert r.status_code == 404

    def test_get_known_returns_wire_shape(self, video_env):
        client, manager = video_env()
        job = _seed_job(manager, "video_aaa", status="in_progress")
        job.progress = 42
        job.phase = "denoising"
        r = client.get("/v1/videos/video_aaa")
        assert r.status_code == 200
        assert r.json() == job.to_dict()
        body = r.json()
        assert body["object"] == "video"
        assert body["status"] == "in_progress"
        assert body["progress"] == 42
        assert body["phase"] == "denoising"
        assert body["size"] == "480x272"


# ---------------------------------------------------------------------------
# GET /v1/videos/{id}/content
# ---------------------------------------------------------------------------


class TestGetContent:
    def test_content_not_completed_409(self, video_env):
        client, manager = video_env()
        _seed_job(manager, "video_q", status="queued")
        r = client.get("/v1/videos/video_q/content")
        assert r.status_code == 409
        assert "queued" in r.json()["detail"]

    def test_content_unknown_404(self, video_env):
        client, _ = video_env()
        assert client.get("/v1/videos/video_nope/content").status_code == 404

    def test_content_artifact_expired_404_detail_dict(self, video_env):
        client, manager = video_env()
        job = _seed_job(manager, "video_purged", status="completed")
        job.artifact_path = None
        job.expires_at = 1750000000.5
        r = client.get("/v1/videos/video_purged/content")
        assert r.status_code == 404
        detail = r.json()["detail"]
        assert isinstance(detail, dict)
        assert detail["code"] == "artifact_expired"
        assert detail["expires_at"] == 1750000000
        assert "purged" in detail["message"]

    def test_content_completed_serves_mp4(self, video_env, tmp_path):
        client, manager = video_env()
        payload = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
        mp4 = tmp_path / "out.mp4"
        mp4.write_bytes(payload)
        job = _seed_job(manager, "video_done", status="completed")
        job.artifact_path = str(mp4)
        r = client.get("/v1/videos/video_done/content")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("video/mp4")
        assert r.content == payload
        assert "video_done.mp4" in r.headers.get("content-disposition", "")


# ---------------------------------------------------------------------------
# DELETE /v1/videos/{id}
# ---------------------------------------------------------------------------


class TestDeleteVideo:
    def test_delete_known(self, video_env):
        client, manager = video_env()
        _seed_job(manager, "video_del")
        r = client.delete("/v1/videos/video_del")
        assert r.status_code == 200
        assert r.json() == {
            "id": "video_del",
            "object": "video.deleted",
            "deleted": True,
        }
        # Record is gone afterwards
        assert manager.get("video_del") is None
        assert client.get("/v1/videos/video_del").status_code == 404

    def test_delete_unknown_404(self, video_env):
        client, _ = video_env()
        assert client.delete("/v1/videos/video_nope").status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/videos -- list envelope + pagination (real list_jobs semantics)
# ---------------------------------------------------------------------------


@pytest.fixture
def listing_env(video_env):
    client, manager = video_env()
    _seed_job(manager, "video_a", created_at=100.0)
    _seed_job(manager, "video_b", created_at=200.0)
    _seed_job(manager, "video_c", created_at=300.0)
    return client, manager


class TestListVideos:
    def test_envelope_default_desc(self, listing_env):
        client, _ = listing_env
        r = client.get("/v1/videos")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert [j["id"] for j in body["data"]] == [
            "video_c", "video_b", "video_a",
        ]
        assert body["has_more"] is False
        assert body["first_id"] == "video_c"
        assert body["last_id"] == "video_a"

    def test_limit_and_has_more(self, listing_env):
        client, _ = listing_env
        r = client.get("/v1/videos", params={"limit": 2})
        body = r.json()
        assert [j["id"] for j in body["data"]] == ["video_c", "video_b"]
        assert body["has_more"] is True
        assert body["first_id"] == "video_c"
        assert body["last_id"] == "video_b"

    def test_after_cursor(self, listing_env):
        client, _ = listing_env
        r = client.get("/v1/videos", params={"after": "video_c"})
        body = r.json()
        assert [j["id"] for j in body["data"]] == ["video_b", "video_a"]
        assert body["has_more"] is False

    def test_after_cursor_with_limit(self, listing_env):
        client, _ = listing_env
        r = client.get("/v1/videos", params={"after": "video_c", "limit": 1})
        body = r.json()
        assert [j["id"] for j in body["data"]] == ["video_b"]
        assert body["has_more"] is True

    def test_order_asc(self, listing_env):
        client, _ = listing_env
        r = client.get("/v1/videos", params={"order": "asc"})
        body = r.json()
        assert [j["id"] for j in body["data"]] == [
            "video_a", "video_b", "video_c",
        ]

    def test_bad_order_400(self, listing_env):
        client, _ = listing_env
        assert client.get(
            "/v1/videos", params={"order": "sideways"}
        ).status_code == 400

    def test_limit_clamped_to_minimum_1(self, listing_env):
        client, _ = listing_env
        r = client.get("/v1/videos", params={"limit": 0})
        body = r.json()
        assert len(body["data"]) == 1
        assert body["has_more"] is True

    def test_unknown_after_cursor_ignored(self, listing_env):
        # Manager semantics: unknown cursor falls through to the full list
        client, _ = listing_env
        r = client.get("/v1/videos", params={"after": "video_ghost"})
        body = r.json()
        assert len(body["data"]) == 3

    def test_empty_list_envelope(self, video_env):
        client, _ = video_env()
        r = client.get("/v1/videos")
        body = r.json()
        assert body == {
            "object": "list",
            "data": [],
            "has_more": False,
            "first_id": None,
            "last_id": None,
        }
