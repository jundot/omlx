# SPDX-License-Identifier: Apache-2.0
"""Image model metadata and admin model-type contracts."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from omlx.admin import routes
from omlx.engine_pool import EnginePool
from omlx.model_settings import ModelSettingsManager


@pytest.fixture
def image_admin(tmp_path, write_z_image_checkpoint):
    write_z_image_checkpoint(tmp_path / "image-model")
    pool = EnginePool()
    pool._get_final_ceiling = lambda: 0
    pool.discover_models(str(tmp_path))
    manager = ModelSettingsManager(tmp_path)
    with (
        patch.object(routes, "_get_engine_pool", return_value=pool),
        patch.object(routes, "_get_settings_manager", return_value=manager),
        patch.object(routes, "_get_server_state", return_value=None),
        patch.object(routes, "_get_global_settings", return_value=None),
    ):
        yield pool, manager


@pytest.mark.asyncio
async def test_image_checkpoint_is_listed_and_can_be_removed(image_admin, tmp_path):
    pool, _ = image_admin
    settings = SimpleNamespace(get_effective_model_dirs=lambda: [tmp_path])
    with (
        patch.object(routes, "_get_global_settings", return_value=settings),
        patch.object(routes, "_hf_downloader", None),
    ):
        result = await routes.list_hf_models(is_admin=True)
        assert [model["name"] for model in result["models"]] == ["image-model"]
        result = await routes.delete_hf_model("image-model", is_admin=True)

    assert result["success"]
    assert not (tmp_path / "image-model").exists()
    assert pool.get_entry("image-model") is None


@pytest.mark.asyncio
async def test_admin_image_override_and_auto_reset(image_admin):
    pool, manager = image_admin
    entry = pool.get_entry("image-model")
    entry.model_type = "llm"
    entry.engine_type = "batched"

    result = await routes.update_model_settings(
        "image-model",
        routes.ModelSettingsRequest(model_type_override="image"),
        is_admin=True,
    )

    assert result["success"]
    assert result["model_type"] == result["engine_type"] == "image"
    assert manager.get_settings("image-model").model_type_override == "image"

    result = await routes.update_model_settings(
        "image-model",
        routes.ModelSettingsRequest(model_type_override=None),
        is_admin=True,
    )

    assert result["model_type"] == result["engine_type"] == "image"
    assert manager.get_settings("image-model").model_type_override is None
