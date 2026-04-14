# SPDX-License-Identifier: Apache-2.0
"""Tests for admin API key management (validation, setup, login, settings update)."""

import asyncio
from dataclasses import fields as dataclass_fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from omlx.admin.auth import validate_api_key, verify_any_api_key, verify_api_key
from omlx.model_settings import ModelSettings
import omlx.server  # noqa: F401 — ensure server module is imported first (triggers set_admin_getters)
import omlx.admin.routes as admin_routes


class TestListModelsSettings:
    """Tests for list_models() settings completeness."""

    def test_list_models_includes_all_model_settings_fields(self):
        """Ensure list_models response includes all ModelSettings fields."""
        mock_engine_pool = MagicMock()
        mock_engine_pool.get_status.return_value = {
            "models": [
                {
                    "id": "test-model",
                    "loaded": True,
                    "estimated_size": 1000,
                    "pinned": False,
                    "engine_type": "batched",
                    "model_type": "llm",
                }
            ]
        }

        test_settings = ModelSettings(
            max_context_window=8192,
            max_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            min_p=0.05,
            presence_penalty=0.3,
            force_sampling=True,
            is_pinned=True,
            is_default=False,
            display_name="Test Model",
            description="A test model",
        )

        mock_settings_manager = MagicMock()
        mock_settings_manager.get_all_settings.return_value = {
            "test-model": test_settings
        }

        mock_server_state = MagicMock()
        mock_server_state.default_model = None

        with (
            patch.object(admin_routes, "_get_engine_pool", return_value=mock_engine_pool),
            patch.object(admin_routes, "_get_settings_manager", return_value=mock_settings_manager),
            patch.object(admin_routes, "_get_server_state", return_value=mock_server_state),
        ):
            result = asyncio.run(admin_routes.list_models(is_admin=True))

        model = result["models"][0]
        assert "settings" in model

        settings_dict = model["settings"]
        expected_fields = {f.name for f in dataclass_fields(ModelSettings)}
        actual_fields = set(settings_dict.keys())
        assert expected_fields == actual_fields, (
            f"Missing fields: {expected_fields - actual_fields}, "
            f"Extra fields: {actual_fields - expected_fields}"
        )

        # Verify specific values
        assert settings_dict["max_context_window"] == 8192
        assert settings_dict["max_tokens"] == 4096
        assert settings_dict["temperature"] == 0.7


class TestValidateApiKey:
    """Tests for validate_api_key() format validation."""

    def test_valid_key_simple(self):
        is_valid, msg = validate_api_key("abcd")
        assert is_valid is True
        assert msg == ""

    def test_valid_key_long(self):
        is_valid, msg = validate_api_key("sk-1234567890abcdef")
        assert is_valid is True

    def test_valid_key_special_chars(self):
        is_valid, msg = validate_api_key("a!@#$%^&*()-_=+[]{}|;:',.<>?/~`")
        assert is_valid is True

    def test_too_short_empty(self):
        is_valid, msg = validate_api_key("")
        assert is_valid is False
        assert "at least 4" in msg

    def test_too_short_one_char(self):
        is_valid, msg = validate_api_key("a")
        assert is_valid is False
        assert "at least 4" in msg

    def test_too_short_three_chars(self):
        is_valid, msg = validate_api_key("abc")
        assert is_valid is False
        assert "at least 4" in msg

    def test_exactly_four_chars(self):
        is_valid, msg = validate_api_key("abcd")
        assert is_valid is True

    def test_whitespace_space(self):
        is_valid, msg = validate_api_key("ab cd")
        assert is_valid is False
        assert "whitespace" in msg

    def test_whitespace_tab(self):
        is_valid, msg = validate_api_key("ab\tcd")
        assert is_valid is False
        assert "whitespace" in msg

    def test_whitespace_newline(self):
        is_valid, msg = validate_api_key("ab\ncd")
        assert is_valid is False
        assert "whitespace" in msg

    def test_whitespace_leading(self):
        is_valid, msg = validate_api_key(" abcd")
        assert is_valid is False
        assert "whitespace" in msg

    def test_whitespace_trailing(self):
        is_valid, msg = validate_api_key("abcd ")
        assert is_valid is False
        assert "whitespace" in msg

    def test_control_char_null(self):
        is_valid, msg = validate_api_key("ab\x00cd")
        assert is_valid is False
        assert "printable" in msg

    def test_control_char_bell(self):
        is_valid, msg = validate_api_key("ab\x07cd")
        assert is_valid is False
        assert "printable" in msg


class TestVerifyApiKeyAdmin:
    """Tests for verify_api_key() constant-time comparison."""

    def test_matching_keys(self):
        assert verify_api_key("secret123", "secret123") is True

    def test_non_matching_keys(self):
        assert verify_api_key("wrong", "secret123") is False

    def test_empty_api_key(self):
        assert verify_api_key("", "secret123") is False

    def test_empty_server_key(self):
        assert verify_api_key("secret123", "") is False

    def test_both_empty(self):
        assert verify_api_key("", "") is False


class TestVerifyAnyApiKey:
    """Tests for verify_any_api_key() checking main key + sub keys."""

    def test_matches_main_key(self):
        from omlx.settings import SubKeyEntry
        sub_keys = [SubKeyEntry(key="sub1"), SubKeyEntry(key="sub2")]
        assert verify_any_api_key("main-key", "main-key", sub_keys) is True

    def test_matches_sub_key(self):
        from omlx.settings import SubKeyEntry
        sub_keys = [SubKeyEntry(key="sub1"), SubKeyEntry(key="sub2")]
        assert verify_any_api_key("sub2", "main-key", sub_keys) is True

    def test_no_match(self):
        from omlx.settings import SubKeyEntry
        sub_keys = [SubKeyEntry(key="sub1")]
        assert verify_any_api_key("wrong", "main-key", sub_keys) is False

    def test_empty_api_key(self):
        from omlx.settings import SubKeyEntry
        sub_keys = [SubKeyEntry(key="sub1")]
        assert verify_any_api_key("", "main-key", sub_keys) is False

    def test_no_main_key_matches_sub(self):
        from omlx.settings import SubKeyEntry
        sub_keys = [SubKeyEntry(key="sub1")]
        assert verify_any_api_key("sub1", "", sub_keys) is True

    def test_empty_sub_keys(self):
        assert verify_any_api_key("main-key", "main-key", []) is True

    def test_no_match_empty_sub_keys(self):
        assert verify_any_api_key("wrong", "main-key", []) is False

    def test_none_main_key_no_sub_keys(self):
        assert verify_any_api_key("anything", None, []) is False

    def test_none_main_key_matches_sub(self):
        from omlx.settings import SubKeyEntry
        sub_keys = [SubKeyEntry(key="sub1")]
        assert verify_any_api_key("sub1", None, sub_keys) is True

    def test_matches_first_sub_key(self):
        from omlx.settings import SubKeyEntry
        sub_keys = [SubKeyEntry(key="sub1"), SubKeyEntry(key="sub2"), SubKeyEntry(key="sub3")]
        assert verify_any_api_key("sub1", "main-key", sub_keys) is True


class TestLoginRejectsSubKey:
    """Tests that sub keys cannot be used for admin login."""

    def test_sub_key_rejected_for_login(self):
        """Sub key should NOT grant admin login — only main key works."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key="main-key")
        mock_settings.auth.sub_keys = [
            __import__("omlx.settings", fromlist=["SubKeyEntry"]).SubKeyEntry(
                key="sub-key-1", name="Test"
            )
        ]
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.LoginRequest(api_key="sub-key-1")
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.login(request, MagicMock()))
            assert exc_info.value.status_code == 401
        finally:
            _restore_getter(original)

    def test_main_key_still_works_for_login(self):
        """Main key should still work for admin login."""
        mock_settings = _mock_global_settings(api_key="main-key")
        mock_settings.auth.sub_keys = [
            __import__("omlx.settings", fromlist=["SubKeyEntry"]).SubKeyEntry(
                key="sub-key-1", name="Test"
            )
        ]
        mock_response = MagicMock()
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.LoginRequest(api_key="main-key")
            result = asyncio.run(admin_routes.login(request, mock_response))
            assert result["success"] is True
        finally:
            _restore_getter(original)


class TestSubKeyCRUD:
    """Tests for sub key create/delete endpoints."""

    def test_create_sub_key_success(self):
        """Creating a valid sub key should succeed."""
        mock_settings = _mock_global_settings(api_key="main-key")
        mock_settings.auth.sub_keys = []
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.CreateSubKeyRequest(key="new-sub-key", name="My Sub Key")
            result = asyncio.run(admin_routes.create_sub_key(request, is_admin=True))
            assert result["success"] is True
            assert result["sub_key"]["key"] == "new-sub-key"
            assert result["sub_key"]["name"] == "My Sub Key"
            assert len(mock_settings.auth.sub_keys) == 1
            mock_settings.save.assert_called_once()
        finally:
            _restore_getter(original)

    def test_create_sub_key_duplicate_main_key(self):
        """Creating a sub key identical to the main key should fail."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key="main-key")
        mock_settings.auth.sub_keys = []
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.CreateSubKeyRequest(key="main-key")
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.create_sub_key(request, is_admin=True))
            assert exc_info.value.status_code == 400
            assert "same as the main key" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_create_sub_key_duplicate_existing(self):
        """Creating a sub key that already exists should fail."""
        from fastapi import HTTPException
        from omlx.settings import SubKeyEntry

        mock_settings = _mock_global_settings(api_key="main-key")
        mock_settings.auth.sub_keys = [SubKeyEntry(key="existing-sub")]
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.CreateSubKeyRequest(key="existing-sub")
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.create_sub_key(request, is_admin=True))
            assert exc_info.value.status_code == 400
            assert "already exists" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_create_sub_key_too_short(self):
        """Creating a sub key that's too short should fail."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key="main-key")
        mock_settings.auth.sub_keys = []
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.CreateSubKeyRequest(key="abc")
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.create_sub_key(request, is_admin=True))
            assert exc_info.value.status_code == 400
            assert "at least 4" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_delete_sub_key_success(self):
        """Deleting an existing sub key should succeed."""
        from omlx.settings import SubKeyEntry

        mock_settings = _mock_global_settings(api_key="main-key")
        mock_settings.auth.sub_keys = [SubKeyEntry(key="sub-to-delete", name="Test")]
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.DeleteSubKeyRequest(key="sub-to-delete")
            result = asyncio.run(admin_routes.delete_sub_key(request, is_admin=True))
            assert result["success"] is True
            assert len(mock_settings.auth.sub_keys) == 0
            mock_settings.save.assert_called_once()
        finally:
            _restore_getter(original)

    def test_delete_sub_key_not_found(self):
        """Deleting a non-existent sub key should return 404."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key="main-key")
        mock_settings.auth.sub_keys = []
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.DeleteSubKeyRequest(key="nonexistent")
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.delete_sub_key(request, is_admin=True))
            assert exc_info.value.status_code == 404
        finally:
            _restore_getter(original)

    def test_create_sub_key_rollback_on_save_failure(self):
        """Sub key should be rolled back if save() fails."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key="main-key")
        mock_settings.auth.sub_keys = []
        mock_settings.save.side_effect = IOError("disk full")
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.CreateSubKeyRequest(key="new-sub-key")
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.create_sub_key(request, is_admin=True))
            assert exc_info.value.status_code == 500
            # Sub key should be rolled back
            assert len(mock_settings.auth.sub_keys) == 0
        finally:
            _restore_getter(original)


def _mock_global_settings(api_key=None):
    """Create a mock GlobalSettings with the given API key."""
    mock = MagicMock()
    mock.auth.api_key = api_key
    return mock


def _patch_getter(mock_settings):
    """Replace the module-level _get_global_settings with a lambda returning mock."""
    original = admin_routes._get_global_settings
    admin_routes._get_global_settings = lambda: mock_settings
    return original


def _restore_getter(original):
    """Restore the original _get_global_settings."""
    admin_routes._get_global_settings = original


class TestSetupApiKeyEndpoint:
    """Tests for POST /admin/api/setup-api-key endpoint logic."""

    def test_setup_rejects_when_key_already_set(self):
        """Setup should fail if API key is already configured."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key="existing-key")
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.SetupApiKeyRequest(
                api_key="newkey", api_key_confirm="newkey"
            )
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.setup_api_key(request, MagicMock()))
            assert exc_info.value.status_code == 400
            assert "already configured" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_setup_rejects_mismatched_keys(self):
        """Setup should fail if api_key and api_key_confirm don't match."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.SetupApiKeyRequest(
                api_key="key1", api_key_confirm="key2"
            )
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.setup_api_key(request, MagicMock()))
            assert exc_info.value.status_code == 400
            assert "do not match" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_setup_rejects_short_key(self):
        """Setup should fail if key is too short."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.SetupApiKeyRequest(
                api_key="abc", api_key_confirm="abc"
            )
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.setup_api_key(request, MagicMock()))
            assert exc_info.value.status_code == 400
            assert "at least 4" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_setup_rejects_whitespace_key(self):
        """Setup should fail if key contains whitespace."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.SetupApiKeyRequest(
                api_key="ab cd", api_key_confirm="ab cd"
            )
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.setup_api_key(request, MagicMock()))
            assert exc_info.value.status_code == 400
            assert "whitespace" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_setup_success_saves_key(self):
        """Successful setup should save key to settings and server state."""
        from unittest.mock import patch

        mock_settings = _mock_global_settings(api_key=None)
        mock_response = MagicMock()
        mock_server_state = MagicMock()
        mock_server_state.api_key = None

        original = _patch_getter(mock_settings)
        try:
            with patch("omlx.server._server_state", mock_server_state):
                request = admin_routes.SetupApiKeyRequest(
                    api_key="validkey123", api_key_confirm="validkey123"
                )
                result = asyncio.run(
                    admin_routes.setup_api_key(request, mock_response)
                )

                assert result["success"] is True
                assert mock_settings.auth.api_key == "validkey123"
                assert mock_server_state.api_key == "validkey123"
                mock_settings.save.assert_called_once()
                mock_response.set_cookie.assert_called_once()
        finally:
            _restore_getter(original)


class TestLoginEndpoint:
    """Tests for POST /admin/api/login endpoint logic."""

    def test_login_rejects_when_no_key_configured(self):
        """Login should fail with 400 when no API key is configured."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.LoginRequest(api_key="anykey")
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.login(request, MagicMock()))
            assert exc_info.value.status_code == 400
            assert "No API key configured" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_login_rejects_invalid_key(self):
        """Login should fail with 401 for wrong API key."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key="correct-key")
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.LoginRequest(api_key="wrong-key")
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.login(request, MagicMock()))
            assert exc_info.value.status_code == 401
        finally:
            _restore_getter(original)

    def test_login_success(self):
        """Login should succeed with correct API key."""
        mock_settings = _mock_global_settings(api_key="correct-key")
        mock_response = MagicMock()
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.LoginRequest(api_key="correct-key")
            result = asyncio.run(admin_routes.login(request, mock_response))
            assert result["success"] is True
            mock_response.set_cookie.assert_called_once()
        finally:
            _restore_getter(original)


class TestStatsSecurity:
    """Tests for /admin/api/stats response hardening."""

    def test_stats_response_includes_api_key_for_admin(self):
        """The stats payload includes api_key for admin CLI snippet generation."""
        mock_settings = MagicMock()
        mock_settings.server.host = "127.0.0.1"
        mock_settings.server.port = 9981
        mock_settings.auth.api_key = "super-secret-key"
        mock_settings.claude_code.context_scaling_enabled = True
        mock_settings.claude_code.target_context_size = 200000

        mock_metrics = MagicMock()
        mock_metrics.get_snapshot.return_value = {
            "total_prompt_tokens": 0,
            "total_cached_tokens": 0,
            "cache_efficiency": 0,
            "avg_prefill_tps": 0,
            "avg_generation_tps": 0,
            "total_requests": 0,
        }

        with (
            patch.object(admin_routes, "_get_global_settings", return_value=mock_settings),
            patch("omlx.server_metrics.get_server_metrics", return_value=mock_metrics),
            patch.object(admin_routes, "_get_engine_info", return_value={}),
            patch.object(admin_routes, "_build_active_models_data", return_value={"models": []}),
            patch.object(admin_routes, "_build_runtime_cache_observability", return_value={"models": []}),
        ):
            result = asyncio.run(admin_routes.get_server_stats(is_admin=True))

        # api_key is included for admin-only CLI snippet generation in the dashboard
        assert result["api_key"] == "super-secret-key"


class TestRuntimeCacheObservability:
    """Tests for runtime cache observability robustness."""

    def test_runtime_cache_uses_model_scoped_ssd_stats(self):
        """Per-model rows should not repeat the shared SSD cache total."""
        cache_dir = Path("/tmp/omlx-cache")

        mock_settings = MagicMock()
        mock_settings.base_path = Path("/tmp/omlx-base")
        mock_settings.cache.get_ssd_cache_dir.return_value = cache_dir

        shared_ssd_stats = {
            "num_files": 999,
            "total_size_bytes": 999_999_999,
            "hot_cache_max_bytes": 0,
            "hot_cache_size_bytes": 0,
            "hot_cache_entries": 0,
        }

        manager_a = MagicMock()
        manager_a.get_stats_for_model.return_value = {
            "num_files": 3,
            "total_size_bytes": 4096,
            "hot_cache_max_bytes": 0,
            "hot_cache_size_bytes": 0,
            "hot_cache_entries": 0,
        }
        scheduler_a = MagicMock()
        scheduler_a.config.model_name = "/models/model-a"
        scheduler_a.paged_ssd_cache_manager = manager_a
        scheduler_a.get_ssd_cache_stats.return_value = {
            "block_size": 1024,
            "indexed_blocks": 12,
            "ssd_cache": shared_ssd_stats,
        }

        manager_b = MagicMock()
        manager_b.get_stats_for_model.return_value = {
            "num_files": 7,
            "total_size_bytes": 8192,
            "hot_cache_max_bytes": 0,
            "hot_cache_size_bytes": 0,
            "hot_cache_entries": 0,
        }
        scheduler_b = MagicMock()
        scheduler_b.config.model_name = "/models/model-b"
        scheduler_b.paged_ssd_cache_manager = manager_b
        scheduler_b.get_ssd_cache_stats.return_value = {
            "block_size": 2048,
            "indexed_blocks": 4,
            "ssd_cache": shared_ssd_stats,
        }

        entry_a = SimpleNamespace(
            engine=SimpleNamespace(
                _engine=SimpleNamespace(
                    engine=SimpleNamespace(scheduler=scheduler_a)
                )
            )
        )
        entry_b = SimpleNamespace(
            engine=SimpleNamespace(
                _engine=SimpleNamespace(
                    engine=SimpleNamespace(scheduler=scheduler_b)
                )
            )
        )

        engine_pool = MagicMock()
        engine_pool.get_status.return_value = {
            "models": [
                {"id": "model-a", "loaded": True},
                {"id": "model-b", "loaded": True},
            ]
        }
        engine_pool._entries = {
            "model-a": entry_a,
            "model-b": entry_b,
        }

        with patch.object(admin_routes, "_get_engine_pool", return_value=engine_pool):
            payload = admin_routes._build_runtime_cache_observability(mock_settings)

        assert payload["total_num_files"] == 10
        assert payload["total_size_bytes"] == 12288
        assert payload["effective_block_sizes"] == [1024, 2048]
        assert payload["models"] == [
            {
                "id": "model-a",
                "block_size": 1024,
                "indexed_blocks": 12,
                "indexed_blocks_display": "12",
                "has_sub_block_cache": False,
                "partial_block_skips": 0,
                "partial_tokens_skipped": 0,
                "last_partial_tokens_skipped": 0,
                "last_tokens_to_next_block": 0,
                "num_files": 3,
                "total_size_bytes": 4096,
                "hot_cache_max_bytes": 0,
                "hot_cache_size_bytes": 0,
                "hot_cache_entries": 0,
            },
            {
                "id": "model-b",
                "block_size": 2048,
                "indexed_blocks": 4,
                "indexed_blocks_display": "4",
                "has_sub_block_cache": False,
                "partial_block_skips": 0,
                "partial_tokens_skipped": 0,
                "last_partial_tokens_skipped": 0,
                "last_tokens_to_next_block": 0,
                "num_files": 7,
                "total_size_bytes": 8192,
                "hot_cache_max_bytes": 0,
                "hot_cache_size_bytes": 0,
                "hot_cache_entries": 0,
            },
        ]
        manager_a.get_stats_for_model.assert_called_once_with("/models/model-a")
        manager_b.get_stats_for_model.assert_called_once_with("/models/model-b")

    def test_runtime_cache_ignores_single_model_stats_failure(self):
        """One model failing stats collection should not break the whole payload."""
        cache_dir = Path("/tmp/omlx-cache")

        mock_settings = MagicMock()
        mock_settings.base_path = Path("/tmp/omlx-base")
        mock_settings.cache.get_ssd_cache_dir.return_value = cache_dir

        bad_scheduler = MagicMock()
        bad_scheduler.get_ssd_cache_stats.side_effect = RuntimeError("boom")
        good_scheduler = MagicMock()
        good_scheduler.get_ssd_cache_stats.return_value = {
            "block_size": 1024,
            "indexed_blocks": 12,
            "ssd_cache": {
                "num_files": 3,
                "total_size_bytes": 4096,
                "hot_cache_max_bytes": 0,
                "hot_cache_size_bytes": 0,
                "hot_cache_entries": 0,
            },
        }

        bad_entry = SimpleNamespace(
            engine=SimpleNamespace(
                _engine=SimpleNamespace(
                    engine=SimpleNamespace(scheduler=bad_scheduler)
                )
            )
        )
        good_entry = SimpleNamespace(
            engine=SimpleNamespace(
                _engine=SimpleNamespace(
                    engine=SimpleNamespace(scheduler=good_scheduler)
                )
            )
        )

        engine_pool = MagicMock()
        engine_pool.get_status.return_value = {
            "models": [
                {"id": "bad-model", "loaded": True},
                {"id": "good-model", "loaded": True},
            ]
        }
        engine_pool._entries = {
            "bad-model": bad_entry,
            "good-model": good_entry,
        }

        with patch.object(admin_routes, "_get_engine_pool", return_value=engine_pool):
            payload = admin_routes._build_runtime_cache_observability(mock_settings)

        assert [m["id"] for m in payload["models"]] == ["good-model"]
        assert payload["total_num_files"] == 3
        assert payload["total_size_bytes"] == 4096
        assert payload["effective_block_sizes"] == [1024]

    def test_runtime_cache_marks_sub_block_cached_when_indexed_blocks_zero(self):
        """Show <block_size indicator only when sub-block cache evidence exists."""
        cache_dir = Path("/tmp/omlx-cache")

        mock_settings = MagicMock()
        mock_settings.base_path = Path("/tmp/omlx-base")
        mock_settings.cache.get_ssd_cache_dir.return_value = cache_dir

        scheduler = MagicMock()
        scheduler.get_ssd_cache_stats.return_value = {
            "block_size": 1024,
            "indexed_blocks": 0,
            "ssd_cache": {
                "num_files": 0,
                "total_size_bytes": 0,
            },
            "prefix_cache": {
                "partial_block_skips": 2,
                "partial_tokens_skipped": 1200,
                "last_partial_tokens_skipped": 577,
                "last_tokens_to_next_block": 447,
            },
        }

        entry = SimpleNamespace(
            engine=SimpleNamespace(
                _engine=SimpleNamespace(
                    engine=SimpleNamespace(scheduler=scheduler)
                )
            )
        )

        engine_pool = MagicMock()
        engine_pool.get_status.return_value = {
            "models": [
                {"id": "qwen-a3b", "loaded": True},
            ]
        }
        engine_pool._entries = {"qwen-a3b": entry}

        with patch.object(admin_routes, "_get_engine_pool", return_value=engine_pool):
            payload = admin_routes._build_runtime_cache_observability(mock_settings)

        model_payload = payload["models"][0]
        assert model_payload["indexed_blocks"] == 0
        assert model_payload["has_sub_block_cache"] is True
        assert model_payload["indexed_blocks_display"] == "<1024"
        assert model_payload["last_partial_tokens_skipped"] == 577


class TestGlobalSettingsValidation:
    """Tests for stricter GlobalSettingsRequest validation."""

    def test_integrations_openclaw_tools_profile_rejects_invalid_value(self):
        with pytest.raises(ValidationError):
            admin_routes.GlobalSettingsRequest(
                integrations_openclaw_tools_profile="invalid-profile"
            )

    def test_integrations_openclaw_tools_profile_accepts_valid_values(self):
        req = admin_routes.GlobalSettingsRequest(
            integrations_openclaw_tools_profile="coding"
        )
        assert req.integrations_openclaw_tools_profile == "coding"
