# SPDX-License-Identifier: Apache-2.0
"""Tests for DFlash engine integration."""

import pytest

from omlx.model_settings import ModelSettings


class TestDFlashModelSettings:
    """Test DFlash fields in ModelSettings."""

    def test_default_values(self):
        settings = ModelSettings()
        assert settings.dflash_enabled is False
        assert settings.dflash_draft_model is None
        assert settings.dflash_draft_quant_bits is None

    def test_no_verify_mode_field(self):
        """verify_mode and speculative_tokens were removed in v2."""
        settings = ModelSettings()
        assert not hasattr(settings, "dflash_verify_mode")
        assert not hasattr(settings, "dflash_speculative_tokens")

    def test_to_dict_includes_dflash_fields(self):
        settings = ModelSettings(
            dflash_enabled=True,
            dflash_draft_model="z-lab/Qwen3.5-4B-DFlash",
        )
        d = settings.to_dict()
        assert d["dflash_enabled"] is True
        assert d["dflash_draft_model"] == "z-lab/Qwen3.5-4B-DFlash"

    def test_to_dict_excludes_none_dflash_fields(self):
        settings = ModelSettings(dflash_enabled=True)
        d = settings.to_dict()
        assert "dflash_draft_model" not in d
        assert "dflash_draft_quant_bits" not in d

    def test_from_dict_with_dflash_fields(self):
        data = {
            "dflash_enabled": True,
            "dflash_draft_model": "z-lab/Qwen3.5-4B-DFlash",
            "dflash_draft_quant_bits": 4,
        }
        settings = ModelSettings.from_dict(data)
        assert settings.dflash_enabled is True
        assert settings.dflash_draft_model == "z-lab/Qwen3.5-4B-DFlash"
        assert settings.dflash_draft_quant_bits == 4

    def test_from_dict_ignores_removed_fields(self):
        """Old settings with verify_mode/speculative_tokens should be ignored."""
        data = {
            "dflash_enabled": True,
            "dflash_verify_mode": "parallel-replay",
            "dflash_speculative_tokens": 16,
        }
        settings = ModelSettings.from_dict(data)
        assert settings.dflash_enabled is True

    def test_roundtrip_serialization(self):
        original = ModelSettings(
            dflash_enabled=True,
            dflash_draft_model="z-lab/Qwen3.5-4B-DFlash",
            dflash_draft_quant_bits=4,
        )
        d = original.to_dict()
        restored = ModelSettings.from_dict(d)
        assert restored.dflash_enabled == original.dflash_enabled
        assert restored.dflash_draft_model == original.dflash_draft_model
        assert restored.dflash_draft_quant_bits == original.dflash_draft_quant_bits


class TestDFlashEngineInit:
    """Test DFlashEngine initialization and configuration."""

    def test_import_without_dflash_mlx(self):
        from omlx.engine import DFlashEngine
        # Should not raise even if dflash-mlx is not installed

    def test_engine_properties(self):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            draft_quant_bits=4,
        )
        assert engine.model_name == "test-model"
        assert engine.tokenizer is None
        assert engine.model_type is None
        assert engine.has_active_requests() is False

    def test_get_stats_no_verify_mode(self):
        """Stats should not include verify_mode (removed in v2)."""
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        stats = engine.get_stats()
        assert stats["engine_type"] == "dflash"
        assert stats["model_name"] == "test-model"
        assert stats["draft_model"] == "test-draft"
        assert stats["loaded"] is False
        assert "verify_mode" not in stats

    def test_cache_stats_returns_none(self):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        assert engine.get_cache_stats() is None


class TestDFlashEnginePoolRouting:
    """Test that EnginePool routes to DFlashEngine based on settings."""

    def test_dflash_disabled_uses_batched(self):
        settings = ModelSettings(dflash_enabled=False)
        assert not getattr(settings, "dflash_enabled", False)

    def test_dflash_enabled_without_draft_model(self):
        settings = ModelSettings(dflash_enabled=True)
        draft = getattr(settings, "dflash_draft_model", None)
        assert draft is None

    def test_dflash_enabled_with_draft_model(self):
        settings = ModelSettings(
            dflash_enabled=True,
            dflash_draft_model="z-lab/Qwen3.5-4B-DFlash",
        )
        assert settings.dflash_enabled is True
        assert settings.dflash_draft_model == "z-lab/Qwen3.5-4B-DFlash"
