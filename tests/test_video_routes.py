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

import base64
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
I2V_MODEL = "wan-i2v-a14b"
TI2V_MODEL = "wan-ti2v-5b"
LLM_MODEL = "llama-llm"

# Full 8-byte PNG signature + padding; passes the _sniff_image magic check
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


def _video_settings(**overrides) -> VideoSettings:
    """Enabled settings; lease defaults to the dataclass default (36GB,
    P0-calibrated to admit the caps corner under the spatial-token
    predictor). Tests that exercise the 413 boundary pass a smaller
    explicit lease."""
    params = dict(enabled=True)
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
        submitted_refs: list[tuple[bytes, str] | None] = []

        async def _submit(
            job: VideoJob,
            input_reference: tuple[bytes, str] | None = None,
        ) -> VideoJob:
            # Record without waking the real dispatcher (no admission loop,
            # no subprocess); mirror the real submit's reference handling
            if input_reference is not None:
                data, suffix = input_reference
                blob_dir = manager.artifacts_dir / job.id
                blob_dir.mkdir(parents=True, exist_ok=True)
                ref_path = blob_dir / f"input_reference{suffix or '.png'}"
                ref_path.write_bytes(data)
                job.params["image_path"] = str(ref_path)
            manager._jobs[job.id] = job
            submitted.append(job)
            submitted_refs.append(input_reference)
            return job

        manager.submit = _submit  # type: ignore[method-assign]
        manager.test_submitted = submitted  # type: ignore[attr-defined]
        manager.test_submitted_refs = submitted_refs  # type: ignore[attr-defined]
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
                model_path=tmp_path / "models" / "wan",
                model_type="video",
                video_pipeline="t2v",
            ),
            I2V_MODEL: SimpleNamespace(
                model_path=tmp_path / "models" / "wan-i2v",
                model_type="video",
                video_pipeline="i2v",
            ),
            TI2V_MODEL: SimpleNamespace(
                model_path=tmp_path / "models" / "wan-ti2v",
                model_type="video",
                video_pipeline="ti2v",
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
        """openai SDK shape: multipart/form-data, every field a string,
        no file part."""
        client, manager = video_env()
        r = client.post(
            "/v1/videos",
            # (None, value) tuples force multipart encoding with plain
            # string fields and no file part
            files={
                "model": (None, VIDEO_MODEL),
                "prompt": (None, "a cat"),
                "seconds": (None, "4"),
                "steps": (None, "10"),
            },
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

    def test_post_multipart_file_to_t2v_400(self, video_env):
        """A conditioning image to a text-to-video model is rejected (the
        old behavior of silently dropping the file part is gone)."""
        client, _ = video_env()
        r = client.post(
            "/v1/videos",
            data={"model": VIDEO_MODEL, "prompt": "a cat"},
            files={"input_reference": ("ref.png", PNG_BYTES, "image/png")},
        )
        assert r.status_code == 400
        assert "text-to-video only" in r.json()["detail"]

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
        # P0-calibrated formula: predicted = 17.5 + 0.0029 * (W/16 * H/16),
        # frame-count-invariant. 1280x720 -> 3600 spatial tokens ->
        # 17.5 + 10.44 = 27.94GB, +6 margin = 33.94 > lease 30 -> 413
        client, _ = video_env(settings=_video_settings(memory_lease_gb=30.0))
        r = _post(client, width=1280, height=720, frames=81)
        assert r.status_code == 413
        detail = r.json()["detail"]
        assert "memory_lease_gb" in detail
        assert "Predicted memory peak" in detail

    def test_peak_predictor_small_request_fits_same_lease(self, video_env):
        # Same 30GB lease: 480x272 -> 510 tokens -> 17.5 + 1.48 = 18.98GB,
        # +6 margin = 24.98 < 30 -> ok
        client, _ = video_env(settings=_video_settings(memory_lease_gb=30.0))
        r = _post(client, width=480, height=272, frames=49)
        assert r.status_code == 200

    def test_peak_predictor_frame_count_invariant(self, video_env):
        # Frames do not enter the memory formula (P0: 49f == 101f peaks);
        # a long video at modest resolution must NOT 413.
        client, _ = video_env(settings=_video_settings(memory_lease_gb=30.0))
        r = _post(client, width=480, height=272, frames=121)
        assert r.status_code == 200

    def test_default_lease_admits_cap_corner(self, video_env):
        # Out-of-the-box settings must admit the caps corner (the v1 bug:
        # default lease below the predictor floor 413'd everything).
        client, _ = video_env(settings=_video_settings())
        r = _post(client, width=1280, height=720)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /v1/videos -- input_reference (I2V conditioning image)
# ---------------------------------------------------------------------------


class TestInputReference:
    def test_json_data_url_to_i2v(self, video_env):
        client, manager = video_env()
        r = _post(client, model=I2V_MODEL, input_reference=PNG_DATA_URL)
        assert r.status_code == 200
        body = r.json()
        assert body["pipeline"] == "i2v"
        assert body["has_input_reference"] is True
        job = manager.get(body["id"])
        image_path = Path(job.params["image_path"])
        assert image_path.suffix == ".png"
        assert image_path.read_bytes() == PNG_BYTES
        # The route handed (bytes, suffix) through to submit
        assert manager.test_submitted_refs == [(PNG_BYTES, ".png")]

    def test_json_raw_base64_to_i2v(self, video_env):
        client, manager = video_env()
        raw_b64 = base64.b64encode(PNG_BYTES).decode()
        r = _post(client, model=I2V_MODEL, input_reference=raw_b64)
        assert r.status_code == 200
        job = manager.get(r.json()["id"])
        assert Path(job.params["image_path"]).read_bytes() == PNG_BYTES

    def test_multipart_file_to_i2v(self, video_env):
        client, manager = video_env()
        r = client.post(
            "/v1/videos",
            data={"model": I2V_MODEL, "prompt": "a cat"},
            files={"input_reference": ("ref.png", PNG_BYTES, "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["has_input_reference"] is True
        job = manager.get(body["id"])
        assert Path(job.params["image_path"]).read_bytes() == PNG_BYTES

    def test_invalid_base64_400(self, video_env):
        client, _ = video_env()
        r = _post(client, model=I2V_MODEL, input_reference="%%%not-base64%%%")
        assert r.status_code == 400
        assert "base64" in r.json()["detail"]

    def test_bad_magic_400(self, video_env):
        # Valid base64 of bytes that are not PNG/JPEG/WebP
        client, _ = video_env()
        gif = base64.b64encode(b"GIF89a" + b"\x00" * 32).decode()
        r = _post(client, model=I2V_MODEL, input_reference=gif)
        assert r.status_code == 400
        assert "PNG, JPEG or WebP" in r.json()["detail"]

    def test_oversize_400(self, video_env, monkeypatch):
        client, _ = video_env()
        monkeypatch.setattr(video_routes, "_MAX_REFERENCE_BYTES", 32)
        r = _post(client, model=I2V_MODEL, input_reference=PNG_DATA_URL)
        assert r.status_code == 400
        assert "exceeds" in r.json()["detail"]

    def test_i2v_without_reference_or_extend_400(self, video_env):
        client, _ = video_env()
        r = _post(client, model=I2V_MODEL)
        assert r.status_code == 400
        assert "requires" in r.json()["detail"]

    def test_reference_and_extend_mutually_exclusive_400(self, video_env):
        # The exclusion fires before extend resolution, so the id can be
        # anything
        client, _ = video_env()
        r = _post(
            client,
            model=I2V_MODEL,
            input_reference=PNG_DATA_URL,
            extend_video_id="video_whatever",
        )
        assert r.status_code == 400
        assert "mutually exclusive" in r.json()["detail"]

    def test_ti2v_with_reference_ok(self, video_env):
        client, manager = video_env()
        r = _post(client, model=TI2V_MODEL, input_reference=PNG_DATA_URL)
        assert r.status_code == 200
        body = r.json()
        assert body["pipeline"] == "ti2v"
        assert body["has_input_reference"] is True
        job = manager.get(body["id"])
        assert Path(job.params["image_path"]).read_bytes() == PNG_BYTES

    def test_ti2v_without_reference_ok(self, video_env):
        # ti2v accepts an image but does not require one
        client, manager = video_env()
        r = _post(client, model=TI2V_MODEL)
        assert r.status_code == 200
        body = r.json()
        assert body["pipeline"] == "ti2v"
        assert body["has_input_reference"] is False
        assert "image_path" not in manager.get(body["id"]).params


# ---------------------------------------------------------------------------
# POST /v1/videos -- extend_video_id (video continuation)
# ---------------------------------------------------------------------------


def _seed_completed_source(
    manager: VideoJobManager, job_id: str = "video_src"
) -> VideoJob:
    """Completed source job with a real mp4 in its blob dir."""
    job = _seed_job(manager, job_id, status="completed")
    blob_dir = manager.artifacts_dir / job_id
    blob_dir.mkdir(parents=True, exist_ok=True)
    mp4 = blob_dir / "output.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    job.artifact_path = str(mp4)
    return job


class TestExtend:
    def test_extend_pins_source_geometry(self, video_env):
        # No input image: extending alone satisfies the i2v image
        # requirement (the worker conditions on the source's last frame)
        client, manager = video_env()
        _seed_completed_source(manager)
        r = _post(
            client,
            model=I2V_MODEL,
            extend_video_id="video_src",
            size="640x368",
            fps=8,
        )
        assert r.status_code == 200
        body = r.json()
        # Source geometry wins over the user-supplied size/fps
        assert body["size"] == "480x272"
        assert body["fps"] == 16
        assert body["pipeline"] == "i2v"
        assert body["extend_source_id"] == "video_src"
        job = manager.get(body["id"])
        assert job.params["width"] == 480
        assert job.params["height"] == 272
        assert job.params["fps"] == 16
        assert job.params["extend_source_id"] == "video_src"
        assert job.params["pipeline"] == "i2v"

    def test_extend_unknown_source_404(self, video_env):
        client, _ = video_env()
        r = _post(client, model=I2V_MODEL, extend_video_id="video_ghost")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"]

    @pytest.mark.parametrize("status", ["queued", "in_progress"])
    def test_extend_source_not_completed_409(self, video_env, status):
        client, manager = video_env()
        _seed_job(manager, "video_pending", status=status)
        r = _post(client, model=I2V_MODEL, extend_video_id="video_pending")
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert status in detail
        assert "completed" in detail

    def test_extend_source_artifact_purged_409(self, video_env):
        client, manager = video_env()
        src = _seed_completed_source(manager)
        Path(src.artifact_path).unlink()
        r = _post(client, model=I2V_MODEL, extend_video_id="video_src")
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert isinstance(detail, dict)
        assert detail["code"] == "artifact_expired"
        assert "purged" in detail["message"]

    def test_extend_with_t2v_model_400(self, video_env):
        # The pipeline gate fires before extend resolution
        client, manager = video_env()
        _seed_completed_source(manager)
        r = _post(client, model=VIDEO_MODEL, extend_video_id="video_src")
        assert r.status_code == 400
        assert "text-to-video only" in r.json()["detail"]


# ---------------------------------------------------------------------------
# POST /v1/videos -- upscale_resolution (SeedVR2)
# ---------------------------------------------------------------------------


class TestUpscale:
    @pytest.mark.parametrize("target", [479, 2161])
    def test_upscale_resolution_out_of_range_400(self, video_env, target):
        # Below the 480 floor or above the default max_upscale_resolution
        # 2160 cap; the range check fires before the weights check
        client, _ = video_env()
        r = _post(client, upscale_resolution=target)
        assert r.status_code == 400
        assert "out of range" in r.json()["detail"]

    def test_upscale_weights_missing_400(self, video_env, tmp_path):
        client, _ = video_env()
        r = _post(client, upscale_resolution=1080)
        assert r.status_code == 400
        detail = r.json()["detail"]
        expected_dir = (
            tmp_path / "models" / "AbstractFramework" / "seedvr2-3b-8bit"
        )
        assert str(expected_dir) in detail
        assert "Download" in detail

    def test_upscale_weights_present_ok(self, video_env, tmp_path):
        weights = tmp_path / "models" / "AbstractFramework" / "seedvr2-3b-8bit"
        (weights / "transformer").mkdir(parents=True)
        # The route requires the safetensors index, not just the dir --
        # a bare transformer/ from an aborted download must not pass
        (weights / "transformer" / "model.safetensors.index.json").write_text("{}")
        client, manager = video_env()
        r = _post(client, upscale_resolution=1080)
        assert r.status_code == 200
        body = r.json()
        assert body["upscale_resolution"] == 1080
        job = manager.get(body["id"])
        assert job.params["upscale_resolution"] == 1080
        assert job.params["upscaler_model_dir"] == str(weights)

    def test_upscale_custom_model_path_honored(self, video_env, tmp_path):
        custom = tmp_path / "custom-seedvr2"
        (custom / "transformer").mkdir(parents=True)
        (custom / "transformer" / "model.safetensors.index.json").write_text("{}")
        client, manager = video_env(
            settings=_video_settings(upscaler_model_path=str(custom))
        )
        r = _post(client, upscale_resolution=720)
        assert r.status_code == 200
        job = manager.get(r.json()["id"])
        assert job.params["upscaler_model_dir"] == str(custom)


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
