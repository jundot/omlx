# SPDX-License-Identifier: Apache-2.0
"""Tests for admin reload models functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import omlx.server  # noqa: F401 — ensure server module is imported first
import omlx.admin.routes as admin_routes


def _setup_mocks(
    engine_pool=None,
    settings_manager=None,
    global_settings=None,
):
    """Patch module-level getters and return originals for restoration."""
    originals = {
        "pool": admin_routes._get_engine_pool,
        "settings_manager": admin_routes._get_settings_manager,
        "global_settings": admin_routes._get_global_settings,
    }
    admin_routes._get_engine_pool = lambda: engine_pool
    admin_routes._get_settings_manager = lambda: settings_manager
    admin_routes._get_global_settings = lambda: global_settings
    return originals


def _restore_mocks(originals):
    admin_routes._get_engine_pool = originals["pool"]
    admin_routes._get_settings_manager = originals["settings_manager"]
    admin_routes._get_global_settings = originals["global_settings"]


class TestReloadModels:
    """Tests for _reload_models() helper."""

    def test_reload_success(self):
        """Successful reload: re-reads settings, re-discovers, preloads pinned."""
        pool = MagicMock()
        pool.model_count = 5
        pool.preload_pinned_models = AsyncMock()

        settings_manager = MagicMock()
        settings_manager._load = MagicMock()
        settings_manager.get_pinned_model_ids = MagicMock(return_value=[])

        global_settings = MagicMock()
        global_settings.model.model_dirs = ["/path/to/models"]
        global_settings.model.model_dir = "/path/to/models"

        originals = _setup_mocks(pool, settings_manager, global_settings)

        mock_server_state = MagicMock()
        mock_server_state.engine_pool = pool
        mock_server_state.settings_manager = settings_manager

        try:
            with patch.object(omlx.server, "_server_state", mock_server_state):
                with patch(
                    "omlx.admin.routes._apply_model_dirs_runtime",
                    new_callable=AsyncMock,
                    return_value=(True, "Re-discovered 5 models from 1 directory"),
                ) as mock_apply:
                    success, msg = asyncio.run(admin_routes._reload_models())

                    assert success is True
                    assert "5 models" in msg
                    settings_manager._load.assert_called_once()
                    mock_apply.assert_called_once_with(["/path/to/models"])
                    pool.preload_pinned_models.assert_called_once()
        finally:
            _restore_mocks(originals)

    def test_reload_engine_pool_none(self):
        """Reload fails when engine pool is not initialized."""
        mock_server_state = MagicMock()
        mock_server_state.engine_pool = None

        originals = _setup_mocks(
            engine_pool=None,
            settings_manager=None,
            global_settings=MagicMock(),
        )

        try:
            with patch.object(omlx.server, "_server_state", mock_server_state):
                success, msg = asyncio.run(admin_routes._reload_models())
                assert success is False
                assert "not initialized" in msg
        finally:
            _restore_mocks(originals)

    def test_reload_global_settings_none(self):
        """Reload fails when global settings is not available."""
        pool = MagicMock()

        mock_server_state = MagicMock()
        mock_server_state.engine_pool = pool

        originals = _setup_mocks(
            engine_pool=pool,
            settings_manager=None,
            global_settings=None,
        )

        try:
            with patch.object(omlx.server, "_server_state", mock_server_state):
                success, msg = asyncio.run(admin_routes._reload_models())
                assert success is False
                assert "not initialized" in msg
        finally:
            _restore_mocks(originals)

    def test_reload_apply_dirs_fails(self):
        """Reload propagates error from _apply_model_dirs_runtime."""
        pool = MagicMock()
        pool.preload_pinned_models = AsyncMock()

        settings_manager = MagicMock()
        settings_manager._load = MagicMock()

        global_settings = MagicMock()
        global_settings.model.model_dirs = ["/bad/path"]
        global_settings.model.model_dir = "/bad/path"

        originals = _setup_mocks(pool, settings_manager, global_settings)

        mock_server_state = MagicMock()
        mock_server_state.engine_pool = pool

        try:
            with patch.object(omlx.server, "_server_state", mock_server_state):
                with patch(
                    "omlx.admin.routes._apply_model_dirs_runtime",
                    new_callable=AsyncMock,
                    return_value=(False, "Model directory does not exist: /bad/path"),
                ):
                    success, msg = asyncio.run(admin_routes._reload_models())
                    assert success is False
                    assert "does not exist" in msg
                    pool.preload_pinned_models.assert_not_called()
        finally:
            _restore_mocks(originals)

    def test_reload_fallback_to_model_dir(self):
        """When model_dirs is empty, falls back to model_dir."""
        pool = MagicMock()
        pool.preload_pinned_models = AsyncMock()

        settings_manager = MagicMock()
        settings_manager._load = MagicMock()

        global_settings = MagicMock()
        global_settings.model.model_dirs = []
        global_settings.model.model_dir = "/fallback/path"

        originals = _setup_mocks(pool, settings_manager, global_settings)

        mock_server_state = MagicMock()
        mock_server_state.engine_pool = pool

        try:
            with patch.object(omlx.server, "_server_state", mock_server_state):
                with patch(
                    "omlx.admin.routes._apply_model_dirs_runtime",
                    new_callable=AsyncMock,
                    return_value=(True, "Re-discovered 3 models from 1 directory"),
                ) as mock_apply:
                    success, msg = asyncio.run(admin_routes._reload_models())
                    assert success is True
                    mock_apply.assert_called_once_with(["/fallback/path"])
        finally:
            _restore_mocks(originals)
