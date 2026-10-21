# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from omlx import server
from omlx.api.image_routes import ImageRequest, generate_image


@pytest.fixture
def image_api(monkeypatch):
    engine = SimpleNamespace(generate=AsyncMock(return_value=(b"PNG-fixture", {})))
    pool = SimpleNamespace(
        release_engine=AsyncMock(), get_abort_requested_reason=lambda model: None
    )

    async def get(model, kind, **kwargs):
        assert kind == server.EngineType.IMAGE
        kwargs["_leased_out"].append("physical-image")
        return engine

    monkeypatch.setattr(server, "get_engine", get)
    monkeypatch.setattr(server, "get_engine_pool", lambda: pool)
    monkeypatch.setattr(server._server_state, "engine_pool", pool)
    server.app.dependency_overrides[server.verify_inference_api_key] = lambda: None
    yield TestClient(server.app), engine, pool
    server.app.dependency_overrides.pop(server.verify_inference_api_key, None)


def test_generation_shape_seed_and_lease(image_api):
    client, engine, pool = image_api
    response = client.post(
        "/v1/images/generations", json={"model": "alias", "prompt": "boat", "seed": 0}
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert len(data) == 1 and data[0]["seed"] == 0
    assert base64.b64decode(data[0]["b64_json"]) == b"PNG-fixture"
    engine.generate.assert_awaited_once_with("boat", 0)
    pool.release_engine.assert_awaited_once_with("physical-image")


@pytest.mark.parametrize(
    "extra",
    [
        {"n": 2},
        {"n": True},
        {"size": "512x512"},
        {"response_format": "url"},
        {"negative_prompt": "bad"},
        {"num_inference_steps": 4},
        {"seed": -1},
        {"seed": True},
        {"prompt": " "},
    ],
)
def test_unsupported_request_rejected_before_loading(image_api, extra):
    client, engine, pool = image_api
    response = client.post(
        "/v1/images/generations", json={"model": "image", "prompt": "boat", **extra}
    )
    assert response.status_code == 422
    engine.generate.assert_not_called()
    pool.release_engine.assert_not_called()


@pytest.mark.parametrize(
    "error,status",
    [
        (MemoryError("headroom"), 503),
        (RuntimeError("busy"), 503),
        (ValueError("prompt"), 400),
    ],
)
def test_failure_releases_lease(image_api, error, status):
    client, engine, pool = image_api
    engine.generate.side_effect = error
    response = client.post(
        "/v1/images/generations", json={"model": "image", "prompt": "boat"}
    )
    assert response.status_code == status
    pool.release_engine.assert_awaited_once_with("physical-image")


@pytest.mark.asyncio
async def test_disconnect_drains_before_releasing(image_api):
    _, engine, pool = image_api
    started = asyncio.Event()
    drained = asyncio.Event()

    async def work(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            drained.set()

    engine.generate = work

    async def disconnected():
        await started.wait()
        return True

    with pytest.raises(HTTPException) as error:
        await generate_image(
            ImageRequest(model="image", prompt="boat"),
            SimpleNamespace(is_disconnected=disconnected),
        )
    assert error.value.status_code == 499 and drained.is_set()
    pool.release_engine.assert_awaited_once_with("physical-image")


def test_route_requires_inference_auth(monkeypatch):
    monkeypatch.setattr(server._server_state, "api_key", "image-test-key")
    monkeypatch.setattr(server._server_state, "global_settings", None)
    response = TestClient(server.app).post(
        "/v1/images/generations", json={"model": "image", "prompt": "boat"}
    )
    assert response.status_code == 401


def test_valid_structural_bundle_becomes_one_image(tmp_path):
    from omlx.model_discovery import discover_models
    from tests.test_image_import import ImageImportTests

    fixture = ImageImportTests()
    fixture.setUp()
    try:
        fixture.bundle()
        result = discover_models(fixture.root)
        assert len(result) == 1
        model = next(iter(result.values()))
        assert model.model_type == model.engine_type == "image"
    finally:
        fixture.doCleanups()


def test_pool_preserves_manifest_backed_image_without_chat_config(tmp_path):
    from omlx.engine_pool import EngineEntry, EnginePool, ModelNotFoundError

    pool = EnginePool.__new__(EnginePool)
    entry = EngineEntry(
        model_id="image",
        model_path=str(tmp_path),
        model_type="image",
        engine_type="image",
        estimated_size=1,
    )
    pool._entries = {"image": entry}
    marker = tmp_path / "omlx-pipeline.json"
    marker.write_text("{}")
    pool._raise_if_model_path_missing_locked("image", entry)
    assert pool._entries["image"] is entry
    marker.unlink()
    with pytest.raises(ModelNotFoundError):
        pool._raise_if_model_path_missing_locked("image", entry)
    assert not pool._entries


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", [server.EngineType.LLM, server.EngineType.IMAGE])
async def test_wrong_task_releases_acquired_pool_lease(monkeypatch, requested):
    from unittest.mock import MagicMock

    from omlx.engine.base import BaseEngine
    from omlx.engine.image import ImageEngine
    from omlx.server import ServerState

    state = ServerState()
    engine = MagicMock(
        spec=ImageEngine if requested == server.EngineType.LLM else BaseEngine
    )
    pool = SimpleNamespace(
        resolve_model_id=lambda mid, sm: mid,
        get_engine=AsyncMock(return_value=engine),
        release_engine=AsyncMock(),
    )
    state.engine_pool = pool
    monkeypatch.setattr(server, "_server_state", state)
    with pytest.raises(HTTPException) as exc:
        await server.get_engine("physical", requested, _lease=True, _leased_out=[])
    assert exc.value.status_code == 400
    pool.release_engine.assert_awaited_once_with("physical")


@pytest.mark.asyncio
async def test_admin_lists_pipeline_once_and_never_lists_its_components(monkeypatch):
    from omlx.admin import routes
    from tests.test_image_import import ImageImportTests

    fixture = ImageImportTests()
    fixture.setUp()
    try:
        fixture.bundle()
        monkeypatch.setattr(
            routes,
            "_get_global_settings",
            lambda: SimpleNamespace(get_effective_model_dirs=lambda: [fixture.root]),
        )
        monkeypatch.setattr(routes, "_get_engine_pool", lambda: None)
        response = await routes.list_hf_models(True)
        assert len(response["models"]) == 1
        assert response["models"][0]["model_type"] == "image"
        assert response["models"][0]["structural_status"] == "valid"
    finally:
        fixture.doCleanups()


@pytest.mark.asyncio
async def test_image_profile_creation_rejected(monkeypatch):
    from omlx.admin import routes

    monkeypatch.setattr(routes, "_require_settings_manager", lambda: SimpleNamespace())
    monkeypatch.setattr(
        routes, "_require_model", lambda _: SimpleNamespace(model_type="image")
    )
    with pytest.raises(HTTPException) as exc:
        await routes.create_model_profile(
            "image", routes.CreateProfileRequest(name="chat", display_name="Chat"), True
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_image_unload_does_not_reclaim_its_already_released_reservation(
    monkeypatch,
):
    from unittest.mock import MagicMock

    from omlx.engine.image import ImageEngine
    from omlx.engine_pool import EngineEntry, EnginePool

    pool = EnginePool()
    engine = MagicMock(spec=ImageEngine)
    engine.stop = AsyncMock()
    reserved = 32 * 1024**3
    entry = EngineEntry(
        model_id="image",
        model_path="/unused",
        model_type="image",
        engine_type="image",
        estimated_size=reserved,
        engine=engine,
    )
    pool._entries = {"image": entry}
    pool._current_model_memory = reserved
    pool._metal_sync_clear_cache = AsyncMock()
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 1024)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 1024)
    await pool._stop_and_unload_engine("image")
    engine.stop.assert_awaited_once()
    pool._metal_sync_clear_cache.assert_awaited_once_with("image")
    assert pool._current_model_memory == 0 and entry.engine is None
