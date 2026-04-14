# SPDX-License-Identifier: Apache-2.0
"""Tests for DFlash speculative decoding integration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# ModelSettings DFlash fields
# ---------------------------------------------------------------------------


class TestDFlashModelSettings:
    """Test DFlash settings in ModelSettings dataclass."""

    def test_dflash_defaults(self):
        from omlx.model_settings import ModelSettings

        ms = ModelSettings()
        assert ms.dflash_enabled is False
        assert ms.dflash_draft_model is None
        assert ms.dflash_block_tokens == 16
        assert ms.dflash_quantize_kv_cache is False

    def test_dflash_enabled(self):
        from omlx.model_settings import ModelSettings

        ms = ModelSettings(dflash_enabled=True)
        assert ms.dflash_enabled is True

    def test_dflash_to_dict(self):
        from omlx.model_settings import ModelSettings

        ms = ModelSettings(dflash_enabled=True, dflash_block_tokens=8)
        d = ms.to_dict()
        assert "dflash_enabled" in d
        assert d["dflash_enabled"] is True
        assert d["dflash_block_tokens"] == 8

    def test_dflash_from_dict(self):
        from omlx.model_settings import ModelSettings

        data = {
            "dflash_enabled": True,
            "dflash_draft_model": "z-lab/Qwen3.5-9B-DFlash",
            "dflash_block_tokens": 12,
            "dflash_quantize_kv_cache": True,
        }
        ms = ModelSettings.from_dict(data)
        assert ms.dflash_enabled is True
        assert ms.dflash_draft_model == "z-lab/Qwen3.5-9B-DFlash"
        assert ms.dflash_block_tokens == 12
        assert ms.dflash_quantize_kv_cache is True

    def test_dflash_mutual_exclusion_turboquant(self):
        from omlx.model_settings import ModelSettings

        with pytest.raises(ValueError, match="dflash_enabled and turboquant_kv_enabled"):
            ModelSettings(dflash_enabled=True, turboquant_kv_enabled=True)

    def test_dflash_mutual_exclusion_planarquant(self):
        from omlx.model_settings import ModelSettings

        with pytest.raises(ValueError, match="dflash_enabled and planarquant_kv_enabled"):
            ModelSettings(dflash_enabled=True, planarquant_kv_enabled=True)

    def test_dflash_settings_persistence(self):
        from omlx.model_settings import ModelSettings, ModelSettingsManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelSettingsManager(Path(tmpdir))
            ms = ModelSettings(
                dflash_enabled=True,
                dflash_draft_model="z-lab/test-draft",
                dflash_block_tokens=8,
            )
            mgr.set_settings("test-model", ms)

            loaded = mgr.get_settings("test-model")
            assert loaded.dflash_enabled is True
            assert loaded.dflash_draft_model == "z-lab/test-draft"
            assert loaded.dflash_block_tokens == 8


# ---------------------------------------------------------------------------
# API request dflash field
# ---------------------------------------------------------------------------


class TestDFlashAPIRequest:
    """Test DFlash field in ChatCompletionRequest."""

    def test_dflash_default_none(self):
        from omlx.api.openai_models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test", messages=[{"role": "user", "content": "hi"}]
        )
        assert req.dflash is None

    def test_dflash_enabled(self):
        from omlx.api.openai_models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test", messages=[{"role": "user", "content": "hi"}],
            dflash=True,
        )
        assert req.dflash is True

    def test_dflash_disabled(self):
        from omlx.api.openai_models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test", messages=[{"role": "user", "content": "hi"}],
            dflash=False,
        )
        assert req.dflash is False


# ---------------------------------------------------------------------------
# DFlash patch module
# ---------------------------------------------------------------------------


class TestDFlashPatch:
    """Test omlx/patches/dflash.py functions."""

    def test_has_dflash_flag(self):
        from omlx.patches.dflash import _HAS_DFLASH

        # Should be True if dflash-mlx is installed, False otherwise
        assert isinstance(_HAS_DFLASH, bool)

    def test_detect_target_family_pure_attention(self):
        """_detect_target_family should return pure_attention for models without linear_attn."""
        from omlx.patches.dflash import _detect_target_family

        # Create a mock model with no linear_attn or is_linear
        mock_layer = MagicMock()
        del mock_layer.linear_attn
        del mock_layer.is_linear

        mock_inner = MagicMock()
        mock_inner.layers = [mock_layer]

        mock_wrapper = MagicMock()
        mock_wrapper.model = mock_inner

        mock_target = MagicMock()
        mock_target.model = mock_wrapper

        family = _detect_target_family(mock_target)
        assert family == "pure_attention"

    def test_detect_target_family_hybrid_gdn(self):
        """_detect_target_family should return hybrid_gdn for models with linear_attn."""
        from omlx.patches.dflash import _detect_target_family

        # When dflash-mlx is installed, detect_target_family from the
        # package is used, so we need to test the fallback by patching
        # the module-level reference to None.
        import omlx.patches.dflash as df

        # Save and replace
        original = df.detect_target_family
        try:
            df.detect_target_family = None  # Force fallback

            # Build mock chain that matches the fallback navigation:
            # target_model.model.layers → [layer with linear_attn]
            # Must use spec to prevent MagicMock auto-creating attributes
            mock_layer = MagicMock()
            mock_layer.linear_attn = MagicMock()
            del mock_layer.is_linear  # Remove is_linear so hasattr returns False

            mock_inner = MagicMock(spec=[])  # No auto-attrs
            mock_inner.layers = [mock_layer]

            mock_target = MagicMock(spec=[])  # No auto-attrs
            mock_target.model = mock_inner
            # Don't set .language_model so hasattr returns False

            family = _detect_target_family(mock_target)
            assert family == "hybrid_gdn"
        finally:
            df.detect_target_family = original

    def test_load_dflash_draft_without_package(self):
        """load_dflash_draft should return (None, None) when dflash-mlx is not available."""
        from omlx.patches import dflash as df

        if not df._HAS_DFLASH:
            draft, ref = df.load_dflash_draft("some/model")
            assert draft is None
            assert ref is None


# ---------------------------------------------------------------------------
# ServerMetrics DFlash recording
# ---------------------------------------------------------------------------


class TestDFlashMetrics:
    """Test DFlash metrics recording in ServerMetrics."""

    def test_record_dflash_cycle(self):
        from omlx.server_metrics import ServerMetrics

        metrics = ServerMetrics()
        metrics.record_dflash_cycle(
            model_id="test-model",
            accepted_from_draft=14,
            generated_tokens=16,
            draft_time_us=5000,
            verify_time_us=3000,
        )
        m = metrics._per_model.get("test-model", {})
        assert m["dflash_cycles"] == 1
        assert m["dflash_accepted_from_draft"] == 14
        assert m["dflash_generated_tokens"] == 16
        assert m["dflash_draft_time_us"] == 5000
        assert m["dflash_verify_time_us"] == 3000

    def test_dflash_metrics_multiple_cycles(self):
        from omlx.server_metrics import ServerMetrics

        metrics = ServerMetrics()
        for _ in range(3):
            metrics.record_dflash_cycle(
                model_id="test-model",
                accepted_from_draft=12,
                generated_tokens=16,
                draft_time_us=4000,
                verify_time_us=2500,
            )
        m = metrics._per_model.get("test-model", {})
        assert m["dflash_cycles"] == 3
        assert m["dflash_accepted_from_draft"] == 36
        assert m["dflash_generated_tokens"] == 48
        ratio = m["dflash_accepted_from_draft"] / m["dflash_generated_tokens"]
        assert abs(ratio - 0.75) < 0.01

    def test_dflash_metrics_per_model_isolation(self):
        from omlx.server_metrics import ServerMetrics

        metrics = ServerMetrics()
        metrics.record_dflash_cycle("model-a", 10, 16)
        metrics.record_dflash_cycle("model-b", 14, 16)

        ma = metrics._per_model.get("model-a", {})
        mb = metrics._per_model.get("model-b", {})
        assert ma["dflash_cycles"] == 1
        assert mb["dflash_cycles"] == 1
        assert ma["dflash_accepted_from_draft"] == 10
        assert mb["dflash_accepted_from_draft"] == 14


# ---------------------------------------------------------------------------
# Admin routes DFlash settings
# ---------------------------------------------------------------------------


class TestDFlashAdminRoutes:
    """Test DFlash settings in admin routes data model."""

    def test_settings_request_dflash_fields(self):
        from omlx.admin.routes import ModelSettingsRequest

        req = ModelSettingsRequest(
            dflash_enabled=True,
            dflash_draft_model="z-lab/test-draft",
            dflash_block_tokens=8,
            dflash_quantize_kv_cache=True,
        )
        assert req.dflash_enabled is True
        assert req.dflash_draft_model == "z-lab/test-draft"
        assert req.dflash_block_tokens == 8
        assert req.dflash_quantize_kv_cache is True

    def test_settings_request_dflash_defaults(self):
        from omlx.admin.routes import ModelSettingsRequest

        req = ModelSettingsRequest()
        assert req.dflash_enabled is None
        assert req.dflash_draft_model is None
        assert req.dflash_block_tokens is None
        assert req.dflash_quantize_kv_cache is None
