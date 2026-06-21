# SPDX-License-Identifier: Apache-2.0
"""Tests for context compression feature."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# omlx.server has a transitive import chain through engine → scheduler →
# vlm_mtp → mlx_vlm. Mock the missing modules so we can import
# _apply_compression without the full ML stack installed.
for _mod in ("mlx_vlm", "mlx_vlm.speculative", "mlx_vlm.speculative.utils"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from omlx.settings import CompressionSettings, GlobalSettings  # noqa: E402, I001
from omlx.server import (  # noqa: E402, I001
    _apply_compression,
    _compression_stats,
    _server_state,
)


class TestCompressionSettings:
    """Tests for CompressionSettings dataclass."""

    def test_defaults(self):
        settings = CompressionSettings()
        assert settings.enabled is True
        assert settings.min_tokens_to_compress == 250

    def test_custom_values(self):
        settings = CompressionSettings(enabled=False, min_tokens_to_compress=500)
        assert settings.enabled is False
        assert settings.min_tokens_to_compress == 500

    def test_to_dict(self):
        settings = CompressionSettings()
        d = settings.to_dict()
        assert d == {"enabled": True, "min_tokens_to_compress": 250}

    def test_from_dict_defaults(self):
        settings = CompressionSettings.from_dict({})
        assert settings.enabled is True
        assert settings.min_tokens_to_compress == 250

    def test_from_dict_custom(self):
        settings = CompressionSettings.from_dict(
            {"enabled": False, "min_tokens_to_compress": 100}
        )
        assert settings.enabled is False
        assert settings.min_tokens_to_compress == 100

    def test_from_dict_partial(self):
        settings = CompressionSettings.from_dict({"enabled": False})
        assert settings.enabled is False
        assert settings.min_tokens_to_compress == 250

    def test_roundtrip(self):
        original = CompressionSettings(enabled=False, min_tokens_to_compress=500)
        restored = CompressionSettings.from_dict(original.to_dict())
        assert restored == original


class TestCompressionSettingsIntegration:
    """Tests for CompressionSettings in GlobalSettings."""

    def test_global_settings_has_compression(self):
        settings = GlobalSettings()
        assert hasattr(settings, "compression")
        assert settings.compression.enabled is True

    def test_global_settings_save_load(self, tmp_path):
        settings = GlobalSettings(base_path=tmp_path)
        settings.compression.enabled = False
        settings.compression.min_tokens_to_compress = 100
        settings.save()

        loaded = GlobalSettings.load(base_path=tmp_path)
        assert loaded.compression.enabled is False
        assert loaded.compression.min_tokens_to_compress == 100

    def test_cli_disable_compression(self):
        args = MagicMock()
        args.disable_compression = True
        settings = GlobalSettings()
        settings._apply_cli_overrides(args)
        assert settings.compression.enabled is False

    def test_cli_no_compression_flag(self):
        args = MagicMock()
        args.disable_compression = False
        settings = GlobalSettings()
        assert settings.compression.enabled is True

    def test_env_disable_compression(self):
        settings = GlobalSettings()
        with patch.dict("os.environ", {"OMLX_COMPRESSION_DISABLED": "true"}):
            settings._apply_env_overrides()
        assert settings.compression.enabled is False

    def test_env_enable_compression(self):
        settings = GlobalSettings()
        settings.compression.enabled = False
        with patch.dict("os.environ", {"OMLX_COMPRESSION_ENABLED": "true"}):
            settings._apply_env_overrides()
        assert settings.compression.enabled is True

    def test_env_disabled_wins_over_enabled(self):
        settings = GlobalSettings()
        with patch.dict("os.environ", {
            "OMLX_COMPRESSION_ENABLED": "true",
            "OMLX_COMPRESSION_DISABLED": "true",
        }):
            settings._apply_env_overrides()
        assert settings.compression.enabled is False

    def test_validation_min_tokens_negative(self):
        settings = GlobalSettings()
        settings.compression.min_tokens_to_compress = -1
        errors = settings.validate()
        assert any("min_tokens_to_compress" in e for e in errors)

    def test_validation_min_tokens_zero(self):
        settings = GlobalSettings()
        settings.compression.min_tokens_to_compress = 0
        errors = settings.validate()
        assert not any("min_tokens_to_compress" in e for e in errors)


class TestApplyCompression:
    """Tests for the _apply_compression function in server.py."""

    @pytest.fixture(autouse=True)
    def _reset_stats(self):
        """Reset module-level _compression_stats before each test."""
        for key in _compression_stats:
            if isinstance(_compression_stats[key], (int, float)):
                _compression_stats[key] = 0
        _compression_stats["last_compression_ratio"] = 0.0
        _compression_stats["last_compression_time"] = None
        yield
        for key in _compression_stats:
            if isinstance(_compression_stats[key], (int, float)):
                _compression_stats[key] = 0
        _compression_stats["last_compression_ratio"] = 0.0
        _compression_stats["last_compression_time"] = None

    def test_headroom_unavailable_passthrough(self):
        """When _HEADROOM_AVAILABLE is False, messages pass through unchanged."""
        messages = [{"role": "user", "content": "x" * 2000}]
        with patch("omlx.server._HEADROOM_AVAILABLE", False):
            result = _apply_compression(messages)
        assert result is messages
        assert _compression_stats["total_requests"] == 0

    def test_compression_disabled_passthrough(self):
        """When compression is disabled in settings, messages pass through."""
        settings = GlobalSettings()
        settings.compression.enabled = False
        messages = [{"role": "user", "content": "x" * 2000}]
        with patch.object(_server_state, "global_settings", settings):
            result = _apply_compression(messages)
        assert result is messages
        assert _compression_stats["total_requests"] == 0

    def test_global_settings_none_passthrough(self):
        """When global_settings is None (e.g. during shutdown), pass through."""
        messages = [{"role": "user", "content": "x" * 2000}]
        with patch.object(_server_state, "global_settings", None):
            result = _apply_compression(messages)
        assert result is messages

    def test_short_messages_counted_not_compressed(self):
        """Short messages increment total_requests but skip actual compression."""
        settings = GlobalSettings()
        settings.compression.enabled = True
        settings.compression.min_tokens_to_compress = 250
        messages = [{"role": "user", "content": "Hi"}]
        with patch.object(_server_state, "global_settings", settings):
            result = _apply_compression(messages)
        assert result is messages
        assert _compression_stats["total_requests"] == 1
        assert _compression_stats["compressed_requests"] == 0

    def test_compression_succeeds_updates_stats(self):
        """When compression saves tokens, stats updated and messages replaced."""
        settings = GlobalSettings()
        settings.compression.enabled = True
        settings.compression.min_tokens_to_compress = 250

        original = [{"role": "user", "content": "x" * 2000}]
        compressed = [{"role": "user", "content": "compressed"}]

        mock_result = MagicMock()
        mock_result.tokens_before = 500
        mock_result.tokens_after = 300
        mock_result.tokens_saved = 200
        mock_result.compression_ratio = 0.6
        mock_result.transforms_applied = ["kompress:system"]
        mock_result.messages = compressed

        with (
            patch.object(_server_state, "global_settings", settings),
            patch("omlx.server._hr_compress", return_value=mock_result),
        ):
            result = _apply_compression(original)

        assert result == compressed
        assert _compression_stats["total_requests"] == 1
        assert _compression_stats["compressed_requests"] == 1
        assert _compression_stats["total_tokens_before"] == 500
        assert _compression_stats["total_tokens_after"] == 300
        assert _compression_stats["total_tokens_saved"] == 200
        assert _compression_stats["last_compression_ratio"] == 0.6
        assert _compression_stats["last_compression_time"] is not None

    def test_zero_tokens_saved_passthrough(self):
        """When tokens_saved == 0, original messages returned, no stats update."""
        settings = GlobalSettings()
        settings.compression.enabled = True
        settings.compression.min_tokens_to_compress = 250

        messages = [{"role": "user", "content": "x" * 2000}]
        mock_result = MagicMock()
        mock_result.tokens_saved = 0
        mock_result.tokens_before = 500
        mock_result.tokens_after = 500

        with (
            patch.object(_server_state, "global_settings", settings),
            patch("omlx.server._hr_compress", return_value=mock_result),
        ):
            result = _apply_compression(messages)

        assert result is messages
        assert _compression_stats["compressed_requests"] == 0

    def test_headroom_exception_returns_original(self):
        """When headroom raises, original messages returned and warning logged."""
        settings = GlobalSettings()
        settings.compression.enabled = True
        settings.compression.min_tokens_to_compress = 250

        messages = [{"role": "user", "content": "x" * 2000}]

        with (
            patch.object(_server_state, "global_settings", settings),
            patch("omlx.server._hr_compress", side_effect=RuntimeError("boom")),
        ):
            result = _apply_compression(messages)

        assert result is messages
        assert _compression_stats["total_requests"] == 1
        assert _compression_stats["compressed_requests"] == 0


class TestCompressionStatsDerivation:
    """Tests for the avg_compression_ratio derivation logic in the admin endpoint."""

    @staticmethod
    def _derive(stats: dict) -> dict:
        """Mirror the endpoint's derivation logic."""
        stats = dict(stats)
        if stats["compressed_requests"] > 0:
            stats["avg_compression_ratio"] = (
                stats["total_tokens_saved"] / stats["total_tokens_before"]
                if stats["total_tokens_before"] > 0
                else 0.0
            )
        else:
            stats["avg_compression_ratio"] = 0.0
        return stats

    def test_no_compressed_requests(self):
        base = {
            "total_requests": 10,
            "compressed_requests": 0,
            "total_tokens_before": 0,
            "total_tokens_after": 0,
            "total_tokens_saved": 0,
            "last_compression_ratio": 0.0,
            "last_compression_time": None,
        }
        result = self._derive(base)
        assert result["avg_compression_ratio"] == 0.0

    def test_with_compressed_requests(self):
        base = {
            "total_requests": 10,
            "compressed_requests": 5,
            "total_tokens_before": 1000,
            "total_tokens_after": 600,
            "total_tokens_saved": 400,
            "last_compression_ratio": 0.6,
            "last_compression_time": 1234567890.0,
        }
        result = self._derive(base)
        assert result["avg_compression_ratio"] == 0.4

    def test_zero_before_tokens_avoids_division_by_zero(self):
        base = {
            "total_requests": 5,
            "compressed_requests": 1,
            "total_tokens_before": 0,
            "total_tokens_after": 0,
            "total_tokens_saved": 0,
            "last_compression_ratio": 0.0,
            "last_compression_time": None,
        }
        result = self._derive(base)
        assert result["avg_compression_ratio"] == 0.0
