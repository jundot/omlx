# SPDX-License-Identifier: Apache-2.0
"""Admin endpoints for mapped weights: alignment report and realign jobs."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import mlx.core as mx
import pytest
from fastapi import HTTPException

from omlx.admin import routes as admin_routes
from omlx.model_settings import ModelSettings
from omlx.utils import mapped_weights as mw


def _write_misaligned(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = {"w": mx.arange(64, dtype=mx.uint32), "s": mx.ones((32,), mx.float16)}
    path = model_dir / "model.safetensors"
    for pad in range(8):
        mx.save_safetensors(str(path), weights, metadata={"p": "x" * pad})
        if mw.read_header(path)[0] % 8:
            break
    (model_dir / "config.json").write_text('{"model_type": "llama"}')


def _entry(model_id, path, *, loaded=False):
    return SimpleNamespace(
        model_id=model_id,
        model_path=str(path),
        engine=object() if loaded else None,
        is_loading=False,
    )


@pytest.fixture
def env(tmp_path):
    entries = {}
    pool = MagicMock()
    pool._entries = entries
    pool.get_entry.side_effect = entries.get
    settings = {}
    manager = MagicMock()
    manager.get_settings.side_effect = lambda mid: settings.setdefault(
        mid, ModelSettings()
    )
    admin_routes._realign_jobs.clear()
    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch.object(admin_routes, "_get_settings_manager", return_value=manager),
        patch.object(
            admin_routes,
            "_hf_downloader",
            SimpleNamespace(model_dir=tmp_path / "models"),
        ),
    ):
        yield SimpleNamespace(
            tmp=tmp_path, entries=entries, settings=settings, manager=manager
        )
    admin_routes._realign_jobs.clear()


async def _finish(model_id):
    for _ in range(500):
        job = admin_routes._realign_jobs[model_id]
        if job["state"] != "running":
            return job
        await asyncio.sleep(0.01)
    raise AssertionError("realign job did not finish")


@pytest.mark.asyncio
async def test_status_reports_misaligned_model(env):
    _write_misaligned(env.tmp / "models" / "m")
    env.entries["m"] = _entry("m", env.tmp / "models" / "m", loaded=True)

    status = await admin_routes.get_mapped_weights_status("m", is_admin=True)

    assert status["aligned"] is False
    assert status["loaded"] is True
    assert status["misaligned_files"] == ["model.safetensors"]
    assert status["can_realign_in_place"] and status["can_realign_copy"]
    assert status["job"] is None


@pytest.mark.asyncio
async def test_unknown_model_is_404(env):
    with pytest.raises(HTTPException) as exc:
        await admin_routes.get_mapped_weights_status("nope", is_admin=True)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_in_place_realign_enables_the_setting(env):
    path = env.tmp / "models" / "m"
    _write_misaligned(path)
    env.entries["m"] = _entry("m", path)

    response = await admin_routes.realign_model_weights(
        "m", admin_routes.RealignRequest(mode="in_place"), is_admin=True
    )
    assert response["status"] == "started"
    job = await _finish("m")

    assert job["state"] == "done", job["error"]
    assert job["done_bytes"] == job["total_bytes"] > 0
    assert mw.check_alignment(path).aligned
    assert env.settings["m"].mapped_weights_enabled is True
    env.manager.set_settings.assert_called_once()


@pytest.mark.asyncio
async def test_copy_realign_leaves_the_original_and_enables_the_copy(env):
    path = env.tmp / "models" / "m"
    _write_misaligned(path)
    original = (path / "model.safetensors").read_bytes()
    env.entries["m"] = _entry("m", path, loaded=True)  # copy works while loaded
    dest = env.tmp / "models" / "m-aligned"

    async def fake_reload():
        env.entries["m-aligned"] = _entry("m-aligned", dest)
        return True, "ok"

    with patch.object(
        admin_routes, "_reload_models", AsyncMock(side_effect=fake_reload)
    ):
        await admin_routes.realign_model_weights(
            "m", admin_routes.RealignRequest(mode="copy"), is_admin=True
        )
        job = await _finish("m")

    assert job["state"] == "done", job["error"]
    assert job["dest"] == str(dest)
    assert job["new_model_id"] == "m-aligned"
    assert (path / "model.safetensors").read_bytes() == original
    assert mw.check_alignment(dest).aligned
    assert (dest / "config.json").exists()
    assert env.settings["m-aligned"].mapped_weights_enabled is True
    assert "m" not in env.settings or not env.settings["m"].mapped_weights_enabled


@pytest.mark.asyncio
async def test_copy_destination_does_not_overwrite(env):
    path = env.tmp / "models" / "m"
    _write_misaligned(path)
    (env.tmp / "models" / "m-aligned").mkdir()
    entry = _entry("m", path)
    assert admin_routes._realign_copy_dest(entry).name == "m-aligned-2"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup,status,detail",
    [
        ("loaded", 409, "Unload"),
        ("symlinked", 400, "Hugging Face"),
        ("aligned", 400, "already aligned"),
    ],
)
async def test_in_place_refusals(env, setup, status, detail):
    path = env.tmp / "models" / "m"
    if setup == "symlinked":
        blobs = env.tmp / "blobs"
        _write_misaligned(blobs)
        path.mkdir(parents=True)
        (path / "model.safetensors").symlink_to(blobs / "model.safetensors")
    else:
        _write_misaligned(path)
    if setup == "aligned":
        mw.realign_model(path, mode="in_place")
    env.entries["m"] = _entry("m", path, loaded=setup == "loaded")

    with pytest.raises(HTTPException) as exc:
        await admin_routes.realign_model_weights(
            "m", admin_routes.RealignRequest(mode="in_place"), is_admin=True
        )
    assert exc.value.status_code == status
    assert detail in exc.value.detail
    assert "m" not in admin_routes._realign_jobs


@pytest.mark.asyncio
async def test_second_realign_while_running_is_409(env):
    path = env.tmp / "models" / "m"
    _write_misaligned(path)
    env.entries["m"] = _entry("m", path)
    admin_routes._realign_jobs["m"] = {"state": "running"}

    with pytest.raises(HTTPException) as exc:
        await admin_routes.realign_model_weights(
            "m", admin_routes.RealignRequest(mode="copy"), is_admin=True
        )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_failed_job_reports_the_error(env):
    path = env.tmp / "models" / "m"
    _write_misaligned(path)
    env.entries["m"] = _entry("m", path)

    with patch.object(mw, "realign_model", side_effect=OSError("disk full")):
        await admin_routes.realign_model_weights(
            "m", admin_routes.RealignRequest(mode="in_place"), is_admin=True
        )
        job = await _finish("m")

    assert job["state"] == "error"
    assert job["error"] == "disk full"
    assert not getattr(env.settings.get("m"), "mapped_weights_enabled", False)
