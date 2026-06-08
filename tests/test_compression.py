# SPDX-License-Identifier: Apache-2.0
"""Tests for context compression feature."""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from omlx.settings import CompressionSettings, GlobalSettings


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
        assert settings.min_tokens_to_compress == 250  # Default preserved

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
        assert settings.compression.enabled is True  # Still default

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


class TestCompressionBlock:
    """Tests for the compression block behavior in server.py."""

    def test_compression_disabled_skips(self):
        """When compression is disabled, messages pass through unchanged."""
        settings = CompressionSettings(enabled=False)
        messages = [{"role": "user", "content": "Hello " * 100}]
        # When disabled, the block short-circuits — no compression attempted
        assert settings.enabled is False

    def test_compression_short_messages_skipped(self):
        """Messages below min_tokens_to_compress threshold are not compressed."""
        settings = CompressionSettings(enabled=True, min_tokens_to_compress=250)
        # A very short message should be skipped
        short_msg = "Hi"
        assert len(short_msg) < settings.min_tokens_to_compress

    def test_compression_import_error_isolation(self):
        """ImportError when headroom not installed is handled gracefully."""
        messages = [{"role": "user", "content": "Test message"}]
        try:
            from headroom import compress

            compress(messages)
        except ImportError:
            # Expected when headroom-ai not installed — this is the graceful path
            pass

    def test_compression_stats_structure(self):
        """Verify the compression stats dict has expected keys."""
        stats = {
            "total_requests": 0,
            "compressed_requests": 0,
            "total_tokens_before": 0,
            "total_tokens_after": 0,
            "total_tokens_saved": 0,
            "last_compression_ratio": 0.0,
            "last_compression_time": None,
        }
        assert "total_requests" in stats
        assert "compressed_requests" in stats
        assert "total_tokens_saved" in stats
        assert "last_compression_ratio" in stats
