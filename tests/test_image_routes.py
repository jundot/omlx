# SPDX-License-Identifier: Apache-2.0
"""Tests for the /v1/images API routes (omlx/api/image_routes.py).

A minimal FastAPI app mounts the image router; the module-level accessors
(_get_image_manager / _image_settings / _get_engine_pool / _resolve_model)
are monkeypatched. get/list/delete/content semantics run against a REAL
MediaJobManager constructed on tmp_path with enforcer=None; submit and the
guard/venv probes are stubbed per test. Sync-path tests run a real dispatch
with a fake stdlib-only image worker script (worker_python=sys.executable,
image_worker_script constructor seam), with _memory_admission stubbed so no
enforcer is needed.

No real model dirs, no ~/.fmlx, no network, no mflux import ever.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import omlx.api.image_routes as image_routes
import omlx.server as omlx_server
from omlx.settings import ImageSettings, VideoSettings
from omlx.video.manager import MediaJob, MediaJobManager, QueueFullError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T2I_MODEL = "z-image-turbo-4bit"
T2I_SLOW_MODEL = "qwen-image-4bit"
EDIT_MODEL = "qwen-image-edit-4bit"
EDIT_PLUS_MODEL = "qwen-image-edit-2511-4bit"
LLM_MODEL = "llama-llm"

# Full 8-byte PNG signature + padding; passes the _sniff_image magic check
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()

# Fake image worker (stdlib only, runs under python -I): writes
# output-{i}.png for i < n into spec["output_dir"] plus a success manifest.
_FAKE_IMAGE_WORKER = """\
import json, os, sys

spec_path = sys.argv[sys.argv.index("--spec") + 1]
with open(spec_path) as f:
    spec = json.load(f)

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()

emit({"phase": "loading"})
emit({"phase": "loaded"})
n = int(spec.get("n") or 1)
outputs = []
for i in range(n):
    emit({"phase": "saving", "image_index": i})
    name = "output-%d.png" % i
    with open(os.path.join(spec["output_dir"], name), "wb") as f:
        f.write(b"FAKE-PNG-BYTES")
    outputs.append(name)
emit({"phase": "done"})
with open(spec["manifest_path"], "w") as f:
    json.dump({"status": "completed", "outputs": outputs}, f)
sys.exit(0)
"""


def _image_settings(**overrides) -> ImageSettings:
    params = dict(
        enabled=True,
        worker_python=sys.executable,
        sync_timeout_seconds=60,
    )
    params.update(overrides)
    return ImageSettings(**params)


def _make_manager(
    tmp_path: Path,
    settings: ImageSettings,
    stub_submit: bool = True,
    worker_body: str | None = None,
) -> MediaJobManager:
    """Real manager (real get/list_jobs/delete/wait_terminal) with the
    probe seams stubbed. With stub_submit=False the real dispatcher runs a
    fake image worker; _memory_admission is stubbed since enforcer=None."""
    image_script = None
    if worker_body is not None:
        image_script = tmp_path / "fake_image_worker.py"
        image_script.write_text(worker_body)
    manager = MediaJobManager(
        settings=VideoSettings(enabled=False),
        base_path=tmp_path,
        enforcer=None,
        image_settings=settings,
        image_worker_script=image_script,
    )
    manager.guard_available = lambda: (True, "")  # type: ignore[method-assign]

    async def _probe(force: bool = False, kind: str = "video"):
        return True, ""

    manager.probe_worker_venv = _probe  # type: ignore[method-assign]

    if stub_submit:
        submitted: list[MediaJob] = []

        async def _submit(
            job: MediaJob,
            input_reference: tuple[bytes, str] | None = None,
            input_images: list[tuple[bytes, str]] | None = None,
        ) -> MediaJob:
            # Record without waking the real dispatcher (no admission loop,
            # no subprocess); mirror the real submit's input-image handling
            if input_images:
                blob_dir = manager.artifacts_dir_for("image") / job.id
                blob_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for i, (data, suffix) in enumerate(input_images):
                    ref_path = blob_dir / f"input-{i}{suffix or '.png'}"
                    ref_path.write_bytes(data)
                    paths.append(str(ref_path))
                job.params["image_paths"] = paths
            manager._jobs[job.id] = job
            submitted.append(job)
            return job

        manager.submit = _submit  # type: ignore[method-assign]
        manager.test_submitted = submitted  # type: ignore[attr-defined]
    else:
        # The real dispatcher runs without an enforcer: bypass admission
        manager._memory_admission = lambda job: (True, "")  # type: ignore[method-assign]
    return manager


def _default_entries(tmp_path: Path) -> dict[str, SimpleNamespace]:
    def entry(model_id, pipeline, alias):
        return SimpleNamespace(
            model_id=model_id,
            model_path=tmp_path / "models" / model_id,
            model_type="image",
            image_pipeline=pipeline,
            image_alias=alias,
        )

    return {
        T2I_MODEL: entry(T2I_MODEL, "t2i", "z-image-turbo"),
        T2I_SLOW_MODEL: entry(T2I_SLOW_MODEL, "t2i", "qwen-image"),
        EDIT_MODEL: entry(EDIT_MODEL, "edit", "qwen-image-edit"),
        EDIT_PLUS_MODEL: entry(EDIT_PLUS_MODEL, "edit", "qwen-image-edit-2511"),
        LLM_MODEL: SimpleNamespace(
            model_id=LLM_MODEL,
            model_path=tmp_path / "models" / LLM_MODEL,
            model_type="llm",
        ),
    }


def _seed_image_job(
    manager: MediaJobManager,
    job_id: str,
    created_at: float = 100.0,
    status: str = "queued",
    n: int = 1,
    **kwargs,
) -> MediaJob:
    job = MediaJob(
        id=job_id,
        kind="image",
        model_id=T2I_MODEL,
        model_dir="/nonexistent/model-dir",
        params={
            "prompt": "a cat",
            "width": 512,
            "height": 512,
            "steps": 9,
            "n": n,
            "seed": 7,
            "pipeline": "t2i",
        },
        status=status,
        created_at=created_at,
        **kwargs,
    )
    manager._jobs[job_id] = job
    return job


def _seed_video_job(
    manager: MediaJobManager, job_id: str, created_at: float = 100.0
) -> MediaJob:
    job = MediaJob(
        id=job_id,
        kind="video",
        model_id="wan-t2v",
        model_dir="/nonexistent/model-dir",
        params={"prompt": "a cat", "width": 480, "height": 272},
        created_at=created_at,
    )
    manager._jobs[job_id] = job
    return job


def _seed_completed_image(
    manager: MediaJobManager, job_id: str = "img_done", n: int = 1
) -> MediaJob:
    """Completed image job with real PNG blobs on disk."""
    job = _seed_image_job(manager, job_id, status="completed", n=n)
    blob_dir = manager.artifacts_dir_for("image") / job_id
    blob_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        path = blob_dir / f"output-{i}.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([i]) * 16)
        files.append(path.name)
    job.artifact_files = files
    job.artifact_path = str(blob_dir / files[0])
    return job


@pytest.fixture
def image_env(monkeypatch, tmp_path):
    """Builder returning (TestClient, manager) with accessors patched."""

    def build(
        settings: ImageSettings | None = None,
        stub_submit: bool = True,
        patch_manager_accessor: bool = True,
        entries: dict | None = None,
        worker_body: str | None = None,
    ):
        isettings = settings or _image_settings()
        manager = _make_manager(
            tmp_path, isettings, stub_submit=stub_submit,
            worker_body=worker_body,
        )
        pool_entries = entries if entries is not None else _default_entries(
            tmp_path
        )
        pool = SimpleNamespace(
            get_entry=lambda mid: pool_entries.get(mid),
            get_model_ids=lambda: list(pool_entries.keys()),
        )

        if patch_manager_accessor:
            monkeypatch.setattr(
                image_routes, "_get_image_manager", lambda: manager
            )
        monkeypatch.setattr(image_routes, "_image_settings", lambda: isettings)
        monkeypatch.setattr(image_routes, "_get_engine_pool", lambda: pool)
        monkeypatch.setattr(image_routes, "_resolve_model", lambda m: m)
        # The real _get_image_manager reads _server_state.global_settings
        # .image.enabled, so a settings stub is patched onto the real
        # ServerState instance (monkeypatch restores it afterwards).
        monkeypatch.setattr(
            omlx_server._server_state,
            "global_settings",
            SimpleNamespace(image=isettings),
        )

        app = FastAPI()
        app.include_router(image_routes.router)
        return TestClient(app), manager

    return build


def _post(client: TestClient, **fields):
    body = {"model": T2I_MODEL, "prompt": "a cat"}
    body.update(fields)
    return client.post("/v1/images", json=body)


def _post_async(client: TestClient, **fields):
    fields.setdefault("sync", False)
    return _post(client, **fields)


# ---------------------------------------------------------------------------
# POST /v1/images -- 503 gates
# ---------------------------------------------------------------------------


class TestServiceGates:
    def test_image_disabled_503_on_every_endpoint(self, image_env, monkeypatch):
        # Do NOT patch _get_image_manager: the real accessor must gate on
        # settings.image.enabled via _server_state.global_settings
        client, _ = image_env(
            settings=_image_settings(enabled=False),
            patch_manager_accessor=False,
        )
        monkeypatch.setattr(
            omlx_server._server_state, "video_job_manager", None
        )
        r = _post(client)
        assert r.status_code == 503
        assert "disabled" in r.json()["detail"]
        # Every endpoint shares the gate
        assert client.get("/v1/images").status_code == 503
        assert client.get("/v1/images/img_x").status_code == 503
        assert client.get("/v1/images/img_x/content").status_code == 503
        assert client.delete("/v1/images/img_x").status_code == 503
        assert client.post(
            "/v1/images/generations",
            json={"model": T2I_MODEL, "prompt": "a cat"},
        ).status_code == 503

    def test_manager_missing_503(self, image_env, monkeypatch):
        # Enabled but lifespan never built the manager -> 503
        client, _ = image_env(patch_manager_accessor=False)
        monkeypatch.setattr(
            omlx_server._server_state, "video_job_manager", None
        )
        r = _post(client)
        assert r.status_code == 503
        assert "not initialized" in r.json()["detail"]

    def test_queue_full_503(self, image_env):
        # Real submit with max_queued_jobs=0 raises QueueFullError before
        # the dispatcher would start
        client, _ = image_env(
            settings=_image_settings(max_queued_jobs=0), stub_submit=False
        )
        r = _post_async(client)
        assert r.status_code == 503
        assert "queue is full" in r.json()["detail"].lower()

    def test_guard_unavailable_503(self, image_env):
        client, manager = image_env()
        manager.guard_available = lambda: (False, "guard is not running")
        r = _post_async(client)
        assert r.status_code == 503
        assert r.json()["detail"] == "guard is not running"

    def test_venv_probe_failure_503(self, image_env):
        client, manager = image_env()

        async def _probe(force: bool = False, kind: str = "video"):
            return False, "Media worker python not found at /x"

        manager.probe_worker_venv = _probe
        r = _post_async(client)
        assert r.status_code == 503
        assert "worker python not found" in r.json()["detail"]


# ---------------------------------------------------------------------------
# POST /v1/images -- async job path (sync=false)
# ---------------------------------------------------------------------------


class TestCreateImageAsync:
    def test_sync_false_returns_job_dict(self, image_env):
        client, manager = image_env()
        r = _post_async(client, size="512x512")
        assert r.status_code == 200
        body = r.json()
        assert body["id"].startswith("img_")
        assert body["object"] == "image.job"
        assert body["status"] == "queued"
        assert body["model"] == T2I_MODEL
        assert body["size"] == "512x512"
        assert body["progress"] == 0
        assert body["error"] is None
        assert body["n"] == 1
        assert body["pipeline"] == "t2i"
        assert body["has_input_image"] is False
        assert body["outputs"] == 0
        # Job actually reached the manager
        assert manager.get(body["id"]) is not None
        assert len(manager.test_submitted) == 1
        job = manager.test_submitted[0]
        # Per-alias step default applies (z-image-turbo -> 9)
        assert job.params["steps"] == 9
        assert job.kind == "image"
        # The per-alias lease landed on the job (z-image-turbo -> 12GB)
        assert job.lease_bytes == 12 * 1024**3

    def test_default_size_applied_when_omitted(self, image_env):
        client, manager = image_env()
        r = _post_async(client)
        assert r.status_code == 200
        assert r.json()["size"] == "1024x1024"
        job = manager.test_submitted[0]
        assert job.params["width"] == 1024 and job.params["height"] == 1024

    def test_unknown_model_404(self, image_env):
        client, _ = image_env()
        r = _post_async(client, model="no-such-model")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"]

    def test_non_image_model_400(self, image_env):
        client, _ = image_env()
        r = _post_async(client, model=LLM_MODEL)
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "not an image generation model" in detail
        assert "model_type=llm" in detail


# ---------------------------------------------------------------------------
# POST /v1/images -- sync path with a real fake worker
# ---------------------------------------------------------------------------


class TestSyncPath:
    def test_sync_default_returns_b64_json_for_n(self, image_env):
        client, _ = image_env(
            stub_submit=False, worker_body=_FAKE_IMAGE_WORKER
        )
        r = _post(client, n=2, steps=2)  # sync omitted -> default true
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["id"].startswith("img_")
        assert isinstance(body["created"], int) and body["created"] > 0
        assert len(body["data"]) == 2
        for item in body["data"]:
            assert base64.b64decode(item["b64_json"]) == b"FAKE-PNG-BYTES"

    def test_sync_url_response_format(self, image_env):
        client, _ = image_env(
            stub_submit=False, worker_body=_FAKE_IMAGE_WORKER
        )
        r = _post(client, response_format="url", steps=2)
        assert r.status_code == 200, r.text
        body = r.json()
        job_id = body["id"]
        # Absolute URL (request.base_url): OpenAI-style clients fetch it
        # directly; TestClient's base is http://testserver.
        assert body["data"] == [
            {"url": f"http://testserver/v1/images/{job_id}/content?index=0"}
        ]
        # The URL actually serves the artifact
        content = client.get(body["data"][0]["url"])
        assert content.status_code == 200
        assert content.content == b"FAKE-PNG-BYTES"

    def test_generations_alias_forces_sync(self, image_env):
        client, _ = image_env(
            stub_submit=False, worker_body=_FAKE_IMAGE_WORKER
        )
        r = client.post(
            "/v1/images/generations",
            json={
                "model": T2I_MODEL, "prompt": "a cat",
                "sync": False, "steps": 2,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # OpenAI image shape, NOT the job dict (sync was forced)
        assert "data" in body
        assert "status" not in body
        assert base64.b64decode(body["data"][0]["b64_json"]) == b"FAKE-PNG-BYTES"


# ---------------------------------------------------------------------------
# POST /v1/images -- auto model pick
# ---------------------------------------------------------------------------


class TestAutoPick:
    def test_pure_text_picks_turbo_t2i(self, image_env):
        client, _ = image_env()
        r = client.post(
            "/v1/images", json={"prompt": "a cat", "sync": False}
        )
        assert r.status_code == 200
        assert r.json()["model"] == T2I_MODEL

    def test_image_attached_picks_edit_model(self, image_env):
        client, _ = image_env()
        r = client.post(
            "/v1/images",
            json={"prompt": "edit it", "image": PNG_DATA_URL, "sync": False},
        )
        assert r.status_code == 200
        body = r.json()
        # First edit-pipeline candidate in sorted id order
        assert body["model"] == EDIT_PLUS_MODEL
        assert body["pipeline"] == "edit"
        assert body["has_input_image"] is True

    def test_no_image_models_404(self, image_env, tmp_path):
        entries = {
            LLM_MODEL: SimpleNamespace(
                model_id=LLM_MODEL,
                model_path=tmp_path / "models" / LLM_MODEL,
                model_type="llm",
            )
        }
        client, _ = image_env(entries=entries)
        r = client.post(
            "/v1/images", json={"prompt": "a cat", "sync": False}
        )
        assert r.status_code == 404
        assert "No suitable image model" in r.json()["detail"]

    def test_image_attached_but_no_edit_model_404(self, image_env, tmp_path):
        entries = {
            T2I_MODEL: SimpleNamespace(
                model_id=T2I_MODEL,
                model_path=tmp_path / "models" / T2I_MODEL,
                model_type="image",
                image_pipeline="t2i",
                image_alias="z-image-turbo",
            )
        }
        client, _ = image_env(entries=entries)
        r = client.post(
            "/v1/images",
            json={"prompt": "edit it", "image": PNG_DATA_URL, "sync": False},
        )
        assert r.status_code == 404
        assert "edit model" in r.json()["detail"]


# ---------------------------------------------------------------------------
# POST /v1/images -- validation 400s
# ---------------------------------------------------------------------------


class TestValidation:
    def test_edit_without_image_400(self, image_env):
        client, _ = image_env()
        r = _post_async(client, model=EDIT_MODEL)
        assert r.status_code == 400
        assert "requires at least one input image" in r.json()["detail"]

    def test_multi_image_on_t2i_400(self, image_env):
        client, _ = image_env()
        r = _post_async(
            client, model=T2I_MODEL, image=[PNG_DATA_URL, PNG_DATA_URL]
        )
        assert r.status_code == 400
        assert "at most one input image" in r.json()["detail"]

    def test_multi_image_on_plain_edit_400(self, image_env):
        # qwen-image-edit (non-2509/2511) is single-reference only
        client, _ = image_env()
        r = _post_async(
            client, model=EDIT_MODEL, image=[PNG_DATA_URL, PNG_DATA_URL]
        )
        assert r.status_code == 400
        assert "single" in r.json()["detail"]
        assert "2509/2511" in r.json()["detail"]

    def test_multi_image_on_edit_plus_ok(self, image_env):
        # Positive control: the 2511 alias takes multiple references
        client, manager = image_env()
        r = _post_async(
            client, model=EDIT_PLUS_MODEL, image=[PNG_DATA_URL, PNG_DATA_URL]
        )
        assert r.status_code == 200
        job = manager.test_submitted[0]
        assert len(job.params["image_paths"]) == 2

    def test_bad_size_string_400(self, image_env):
        client, _ = image_env()
        r = _post_async(client, size="1024by768")
        assert r.status_code == 400
        assert 'size must be "WxH"' in r.json()["detail"]

    def test_n_over_max_400(self, image_env):
        client, _ = image_env()  # default max_n=4
        r = _post_async(client, n=5)
        assert r.status_code == 400
        assert "n must be between 1 and 4" in r.json()["detail"]

    def test_steps_over_max_400(self, image_env):
        client, _ = image_env()  # default max_steps=60
        r = _post_async(client, steps=61)
        assert r.status_code == 400
        assert "steps must be between 1 and 60" in r.json()["detail"]

    def test_pixels_over_max_400(self, image_env):
        client, _ = image_env()  # default cap 2048*2048
        r = _post_async(client, size="2064x2064")
        assert r.status_code == 400
        assert "max_pixels" in r.json()["detail"]

    def test_bad_image_strength_400(self, image_env):
        client, _ = image_env()
        r = _post_async(client, image_strength=1.5)
        assert r.status_code == 400
        assert "image_strength" in r.json()["detail"]

    def test_zero_byte_image_400(self, image_env):
        # A data URL with an empty payload decodes to b""
        client, _ = image_env()
        r = _post_async(client, image="data:image/png;base64,")
        assert r.status_code == 400
        assert "zero-byte" in r.json()["detail"]

    def test_bad_base64_image_400(self, image_env):
        client, _ = image_env()
        r = _post_async(client, image="%%%not-base64%%%")
        assert r.status_code == 400
        assert "base64" in r.json()["detail"]

    def test_bad_magic_400(self, image_env):
        # Valid base64 of bytes that are not PNG/JPEG/WebP
        client, _ = image_env()
        gif = base64.b64encode(b"GIF89a" + b"\x00" * 32).decode()
        r = _post_async(client, model=EDIT_MODEL, image=gif)
        assert r.status_code == 400
        assert "PNG, JPEG or WebP" in r.json()["detail"]

    def test_missing_prompt_400(self, image_env):
        client, _ = image_env()
        r = client.post("/v1/images", json={"model": T2I_MODEL})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# POST /v1/images -- multipart submissions
# ---------------------------------------------------------------------------


class TestMultipart:
    def test_multipart_file_field_to_edit_model(self, image_env):
        client, manager = image_env()
        r = client.post(
            "/v1/images",
            data={"model": EDIT_MODEL, "prompt": "edit it", "sync": "false"},
            files={"image": ("ref.png", PNG_BYTES, "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "queued"
        assert body["has_input_image"] is True
        job = manager.test_submitted[0]
        paths = job.params["image_paths"]
        assert len(paths) == 1
        saved = Path(paths[0])
        assert saved.suffix == ".png"
        assert saved.read_bytes() == PNG_BYTES

    def test_multipart_all_string_fields_no_file(self, image_env):
        # openai SDK shape: multipart/form-data, every field a string
        client, manager = image_env()
        r = client.post(
            "/v1/images",
            files={
                "model": (None, T2I_MODEL),
                "prompt": (None, "a cat"),
                "sync": (None, "false"),
                "steps": (None, "5"),
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "queued"
        assert body["steps"] == 5
        assert manager.test_submitted[0].params["steps"] == 5


# ---------------------------------------------------------------------------
# GET /v1/images -- list envelope + pagination + kind isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def listing_env(image_env):
    client, manager = image_env()
    _seed_image_job(manager, "img_a", created_at=100.0)
    _seed_image_job(manager, "img_b", created_at=200.0)
    _seed_image_job(manager, "img_c", created_at=300.0)
    # A video job in the same manager must never appear in /v1/images
    _seed_video_job(manager, "video_x", created_at=250.0)
    return client, manager


class TestListImages:
    def test_envelope_default_desc_excludes_video(self, listing_env):
        client, _ = listing_env
        r = client.get("/v1/images")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert [j["id"] for j in body["data"]] == ["img_c", "img_b", "img_a"]
        assert body["has_more"] is False
        assert body["last_id"] == "img_a"
        assert all(j["object"] == "image.job" for j in body["data"])

    def test_limit_and_has_more(self, listing_env):
        client, _ = listing_env
        r = client.get("/v1/images", params={"limit": 2})
        body = r.json()
        assert [j["id"] for j in body["data"]] == ["img_c", "img_b"]
        assert body["has_more"] is True
        assert body["last_id"] == "img_b"

    def test_after_cursor(self, listing_env):
        client, _ = listing_env
        r = client.get("/v1/images", params={"after": "img_c"})
        body = r.json()
        assert [j["id"] for j in body["data"]] == ["img_b", "img_a"]
        assert body["has_more"] is False

    def test_order_asc(self, listing_env):
        client, _ = listing_env
        r = client.get("/v1/images", params={"order": "asc"})
        body = r.json()
        assert [j["id"] for j in body["data"]] == ["img_a", "img_b", "img_c"]

    def test_empty_list_envelope(self, image_env):
        client, _ = image_env()
        r = client.get("/v1/images")
        assert r.json() == {
            "object": "list",
            "data": [],
            "has_more": False,
            "last_id": None,
        }


# ---------------------------------------------------------------------------
# GET /v1/images/{id}
# ---------------------------------------------------------------------------


class TestGetImage:
    def test_get_unknown_404(self, image_env):
        client, _ = image_env()
        assert client.get("/v1/images/img_ghost").status_code == 404

    def test_get_video_kind_id_404(self, image_env):
        # A video job id must be invisible through the image endpoints
        client, manager = image_env()
        _seed_video_job(manager, "video_x")
        r = client.get("/v1/images/video_x")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"]

    def test_get_known_returns_wire_shape(self, image_env):
        client, manager = image_env()
        job = _seed_image_job(manager, "img_aaa", status="in_progress")
        job.progress = 42
        job.phase = "denoise"
        r = client.get("/v1/images/img_aaa")
        assert r.status_code == 200
        assert r.json() == job.to_dict()
        body = r.json()
        assert body["object"] == "image.job"
        assert body["status"] == "in_progress"
        assert body["progress"] == 42
        assert body["size"] == "512x512"


# ---------------------------------------------------------------------------
# GET /v1/images/{id}/content
# ---------------------------------------------------------------------------


class TestGetContent:
    def test_content_not_completed_409(self, image_env):
        client, manager = image_env()
        _seed_image_job(manager, "img_run", status="in_progress")
        r = client.get("/v1/images/img_run/content")
        assert r.status_code == 409
        assert "in_progress" in r.json()["detail"]

    def test_content_unknown_404(self, image_env):
        client, _ = image_env()
        assert client.get("/v1/images/img_nope/content").status_code == 404

    def test_content_index_out_of_range_404(self, image_env):
        client, manager = image_env()
        _seed_completed_image(manager, "img_one", n=1)
        r = client.get("/v1/images/img_one/content", params={"index": 1})
        assert r.status_code == 404
        assert "out of range" in r.json()["detail"]

    def test_content_artifact_expired_404_detail_dict(self, image_env):
        # Exactly the retention-purged state: files list cleared AND
        # artifact_path dropped, expires_at stamped
        client, manager = image_env()
        job = _seed_image_job(manager, "img_purged", status="completed")
        job.artifact_files = []
        job.artifact_path = None
        job.expires_at = 1750000000.5
        r = client.get("/v1/images/img_purged/content")
        assert r.status_code == 404
        detail = r.json()["detail"]
        assert isinstance(detail, dict)
        assert detail["code"] == "artifact_expired"
        assert detail["expires_at"] == 1750000000
        assert "purged" in detail["message"]

    def test_content_completed_serves_png(self, image_env):
        client, manager = image_env()
        job = _seed_completed_image(manager, "img_done", n=2)
        blob_dir = Path(job.artifact_path).parent

        r = client.get("/v1/images/img_done/content")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/png")
        assert r.content == (blob_dir / "output-0.png").read_bytes()
        assert "img_done-0.png" in r.headers.get("content-disposition", "")

        # index selects the second output
        r1 = client.get("/v1/images/img_done/content", params={"index": 1})
        assert r1.status_code == 200
        assert r1.content == (blob_dir / "output-1.png").read_bytes()
        assert "img_done-1.png" in r1.headers.get("content-disposition", "")


# ---------------------------------------------------------------------------
# DELETE /v1/images/{id}
# ---------------------------------------------------------------------------


class TestDeleteImage:
    def test_delete_known(self, image_env):
        client, manager = image_env()
        job = _seed_completed_image(manager, "img_del")
        blob_dir = Path(job.artifact_path).parent
        r = client.delete("/v1/images/img_del")
        assert r.status_code == 200
        assert r.json() == {
            "id": "img_del",
            "object": "image.deleted",
            "deleted": True,
        }
        # Record and blobs are gone afterwards
        assert manager.get("img_del") is None
        assert not blob_dir.exists()
        assert client.get("/v1/images/img_del").status_code == 404
        assert client.delete("/v1/images/img_del").status_code == 404

    def test_delete_unknown_404(self, image_env):
        client, _ = image_env()
        assert client.delete("/v1/images/img_nope").status_code == 404

    def test_delete_video_kind_id_404(self, image_env):
        client, manager = image_env()
        _seed_video_job(manager, "video_x")
        assert client.delete("/v1/images/video_x").status_code == 404
        # The video job survives -- the image endpoint never touched it
        assert manager.get("video_x") is not None


# ---------------------------------------------------------------------------
# Per-model defaults (ModelSettings.image_default_*)
# ---------------------------------------------------------------------------


class TestModelDefaults:
    """Resolution order: explicit request value > per-model settings >
    global image settings > per-alias defaults."""

    def _patch_model_settings(self, monkeypatch, **fields):
        defaults = {
            "image_default_steps": None,
            "image_default_size": None,
        }
        defaults.update(fields)
        ms = SimpleNamespace(**defaults)
        monkeypatch.setattr(
            omlx_server._server_state,
            "settings_manager",
            SimpleNamespace(get_settings=lambda mid: ms),
        )
        return ms

    def test_model_defaults_fill_unset_params(self, image_env, monkeypatch):
        client, manager = image_env()
        self._patch_model_settings(
            monkeypatch,
            image_default_steps=14,
            image_default_size="768x512",
        )
        r = _post_async(client)
        assert r.status_code == 200, r.text
        job = manager.test_submitted[0]
        assert job.params["steps"] == 14
        assert job.params["width"] == 768
        assert job.params["height"] == 512

    def test_explicit_request_beats_model_defaults(self, image_env, monkeypatch):
        client, manager = image_env()
        self._patch_model_settings(
            monkeypatch,
            image_default_steps=14,
            image_default_size="768x512",
        )
        r = _post_async(client, steps=7, size="1024x768")
        assert r.status_code == 200, r.text
        job = manager.test_submitted[0]
        assert job.params["steps"] == 7
        assert job.params["width"] == 1024
        assert job.params["height"] == 768

    def test_no_settings_manager_uses_alias_and_global_defaults(
        self, image_env, monkeypatch
    ):
        client, manager = image_env()
        monkeypatch.setattr(
            omlx_server._server_state, "settings_manager", None
        )
        r = _post_async(client)
        assert r.status_code == 200, r.text
        job = manager.test_submitted[0]
        assert job.params["steps"] == 9  # z-image-turbo alias default
        assert job.params["width"] == 1024  # global default_size
        assert job.params["height"] == 1024


# ---------------------------------------------------------------------------
# QueueFullError importability sanity (route maps it to 503)
# ---------------------------------------------------------------------------


def test_queue_full_error_raised_by_real_submit(image_env):
    import asyncio

    _, manager = image_env(
        settings=_image_settings(max_queued_jobs=0), stub_submit=False
    )
    job = MediaJob(id="img_x", kind="image", model_id="m", model_dir="d",
                   params={})
    with pytest.raises(QueueFullError):
        asyncio.run(manager.submit(job))


# ---------------------------------------------------------------------------
# Regressions pinned from the adversarial branch review
# ---------------------------------------------------------------------------


class TestReviewRegressions:
    def test_seed_negative_400(self, image_env):
        client, _ = image_env()
        assert _post_async(client, seed=-1).status_code == 400

    def test_seed_too_large_400(self, image_env):
        # Out-of-range seeds previously crashed the worker AFTER the full
        # model load (mx.random.key rejects >= 2**64; seed+i overflows).
        client, _ = image_env()
        assert _post_async(client, seed=2**63).status_code == 400

    def test_n_zero_400(self, image_env):
        client, _ = image_env()
        assert _post_async(client, n=0).status_code == 400

    def test_steps_zero_400(self, image_env):
        client, _ = image_env()
        assert _post_async(client, steps=0).status_code == 400

    def test_negative_size_400(self, image_env):
        client, _ = image_env()
        assert _post_async(client, size="-512x512").status_code == 400

    def test_too_many_input_images_400(self, image_env):
        client, _ = image_env()
        r = _post_async(
            client, model=EDIT_PLUS_MODEL, image=[PNG_DATA_URL] * 5
        )
        assert r.status_code == 400
        assert "At most" in r.json()["detail"]

    def test_lora_absolute_path_400(self, image_env):
        client, _ = image_env()
        r = _post_async(client, lora_paths=["/etc/shadow"])
        assert r.status_code == 400
        r = _post_async(client, lora_paths=["foo/../../bar.safetensors"])
        assert r.status_code == 400

    def test_lora_hf_reference_accepted(self, image_env):
        client, manager = image_env()
        r = _post_async(
            client,
            lora_paths=["lightx2v/Qwen-Image-Lightning:Qwen-8steps.safetensors"],
        )
        assert r.status_code == 200, r.text
        assert manager.test_submitted[0].params["lora_paths"] == [
            "lightx2v/Qwen-Image-Lightning:Qwen-8steps.safetensors"
        ]

    def test_text_only_with_only_edit_models_404(self, image_env, tmp_path):
        # Auto-pick must not fall back to an edit model for a text-only
        # request (it would guarantee a confusing "needs input image" 400).
        entries = {
            k: v for k, v in _default_entries(tmp_path).items()
            if getattr(v, "image_pipeline", "") == "edit"
        }
        client, _ = image_env(entries=entries)
        r = client.post("/v1/images", json={"prompt": "a cat", "sync": False})
        assert r.status_code == 404

    def test_lease_over_ceiling_413(self, image_env):
        client, manager = image_env()
        manager._enforcer = SimpleNamespace(
            is_running=True,
            get_final_ceiling=lambda: 10 * image_routes.GB,
        )
        # z-image-turbo alias lease default is 12GB > 10GB ceiling
        r = _post_async(client)
        assert r.status_code == 413
        assert "does not fit" in r.json()["detail"]

    def test_generations_timeout_cancels_job(self, image_env):
        sleeper = (
            "import json, sys, time\n"
            "spec_path = sys.argv[sys.argv.index('--spec') + 1]\n"
            "with open(spec_path) as f:\n"
            "    spec = json.load(f)\n"
            "print(json.dumps({'phase': 'loading'}), flush=True)\n"
            "time.sleep(30)\n"
        )
        client, manager = image_env(
            settings=_image_settings(sync_timeout_seconds=1),
            stub_submit=False,
            worker_body=sleeper,
        )
        r = client.post(
            "/v1/images/generations",
            json={"model": T2I_MODEL, "prompt": "a cat"},
        )
        assert r.status_code == 504
        assert "cancelled" in r.json()["detail"]
        # The orphan job was DELETEd: SDK clients auto-retry 5xx, and a
        # surviving job would hold the single-worker queue while duplicates
        # pile up behind it.
        page, _ = manager.list_jobs(kind="image")
        assert page == []

    def test_plain_sync_timeout_keeps_job(self, image_env):
        sleeper = (
            "import json, sys, time\n"
            "spec_path = sys.argv[sys.argv.index('--spec') + 1]\n"
            "with open(spec_path) as f:\n"
            "    spec = json.load(f)\n"
            "print(json.dumps({'phase': 'loading'}), flush=True)\n"
            "time.sleep(30)\n"
        )
        client, manager = image_env(
            settings=_image_settings(sync_timeout_seconds=1),
            stub_submit=False,
            worker_body=sleeper,
        )
        r = _post(client)  # sync default true, NOT the generations alias
        assert r.status_code == 504
        assert "poll GET /v1/images/" in r.json()["detail"]
        page, _ = manager.list_jobs(kind="image")
        assert len(page) == 1  # fmlx-aware callers can still poll it


class TestVideoRouteKindIsolation:
    """Image jobs must be invisible on the /v1/videos wire surface."""

    def test_image_jobs_invisible_on_video_surface(self, image_env, monkeypatch):
        import omlx.api.video_routes as video_routes

        _, manager = image_env()
        job = _seed_completed_image(manager, "img_iso")
        _seed_video_job(manager, "video_iso")

        monkeypatch.setattr(
            video_routes, "_get_video_manager", lambda: manager
        )
        app = FastAPI()
        app.include_router(video_routes.router)
        vclient = TestClient(app)

        data = vclient.get("/v1/videos").json()["data"]
        assert [j["id"] for j in data] == ["video_iso"]

        assert vclient.get("/v1/videos/img_iso").status_code == 404
        assert vclient.get("/v1/videos/img_iso/content").status_code == 404
        assert vclient.delete("/v1/videos/img_iso").status_code == 404
        # The cross-kind DELETE must not have removed the image job
        assert manager.get("img_iso") is not None
        assert job.artifact_path is not None
