# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.qwen35_yarn_rope."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestComputeYarnInvFreq:
    """Test the pure-math frequency correction function."""

    def _make_inv_freq(self, dim: int, base: float = 10000.0):
        import mlx.core as mx

        return 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def test_factor_one_is_identity(self):
        """factor=1.0 should return frequencies unchanged."""
        import mlx.core as mx

        from omlx.patches.qwen35_yarn_rope import _compute_yarn_inv_freq

        dim = 64
        inv_freq = self._make_inv_freq(dim)
        result = _compute_yarn_inv_freq(
            inv_freq, dim, 10000.0, factor=1.0, orig_max_pos=32768
        )
        mx.eval(result)
        assert mx.allclose(result, inv_freq, atol=1e-6).item()

    def test_high_factor_interpolation_zone(self):
        """At high frequency indices (interpolation zone), frequencies should
        be divided by the factor."""
        import mlx.core as mx

        from omlx.patches.qwen35_yarn_rope import _compute_yarn_inv_freq

        dim = 128
        factor = 4.0
        inv_freq = self._make_inv_freq(dim)
        result = _compute_yarn_inv_freq(
            inv_freq, dim, 10000.0, factor=factor, orig_max_pos=32768
        )
        mx.eval(result)

        # The last element should be fully interpolated (divided by factor)
        last_orig = inv_freq[-1].item()
        last_yarn = result[-1].item()
        assert abs(last_yarn - last_orig / factor) < 1e-6

    def test_low_dim_extrapolation_zone(self):
        """At low frequency indices (extrapolation zone), frequencies should
        be unchanged."""
        import mlx.core as mx

        from omlx.patches.qwen35_yarn_rope import _compute_yarn_inv_freq

        dim = 128
        factor = 4.0
        inv_freq = self._make_inv_freq(dim)
        result = _compute_yarn_inv_freq(
            inv_freq, dim, 10000.0, factor=factor, orig_max_pos=32768
        )
        mx.eval(result)

        # The first element should be unchanged (extrapolation)
        assert abs(result[0].item() - inv_freq[0].item()) < 1e-6

    def test_blend_between_zones(self):
        """Mid-range indices should produce values strictly between
        original and original/factor."""
        import mlx.core as mx

        from omlx.patches.qwen35_yarn_rope import _compute_yarn_inv_freq

        dim = 128
        factor = 4.0
        inv_freq = self._make_inv_freq(dim)
        result = _compute_yarn_inv_freq(
            inv_freq, dim, 10000.0, factor=factor, orig_max_pos=32768
        )
        mx.eval(result)

        for i in range(inv_freq.shape[0]):
            orig = inv_freq[i].item()
            yarn = result[i].item()
            assert yarn <= orig + 1e-6
            assert yarn >= orig / factor - 1e-6


class TestYarnMscale:
    """Test the attention amplitude correction."""

    def test_factor_one_returns_one(self):
        from omlx.patches.qwen35_yarn_rope import _yarn_mscale

        assert _yarn_mscale(1.0) == 1.0

    def test_factor_two_default_params(self):
        from omlx.patches.qwen35_yarn_rope import _yarn_mscale

        result = _yarn_mscale(2.0, mscale=1.0, mscale_all_dim=0.0)
        expected = 0.1 * 1.0 * math.log(2.0) + 1.0
        assert abs(result - expected) < 1e-6

    def test_mscale_all_dim_nonzero(self):
        from omlx.patches.qwen35_yarn_rope import _yarn_mscale

        result = _yarn_mscale(4.0, mscale=1.0, mscale_all_dim=1.0)
        num = 0.1 * 1.0 * math.log(4.0) + 1.0
        den = 0.1 * 1.0 * math.log(4.0) + 1.0
        assert abs(result - num / den) < 1e-6
        assert abs(result - 1.0) < 1e-6


class TestSetYarnParams:
    """Test the module-level state setter."""

    def test_set_and_clear(self):
        from omlx.patches.qwen35_yarn_rope import (
            _ACTIVE_YARN_PARAMS,
            set_yarn_params,
        )

        set_yarn_params({"factor": 2.0, "orig_max_pos": 32768})
        from omlx.patches import qwen35_yarn_rope

        assert qwen35_yarn_rope._ACTIVE_YARN_PARAMS is not None
        assert qwen35_yarn_rope._ACTIVE_YARN_PARAMS["factor"] == 2.0

        set_yarn_params(None)
        assert qwen35_yarn_rope._ACTIVE_YARN_PARAMS is None


class TestIsYarnCompatible:
    """Test the dispatch helper."""

    def test_qwen3_5_vlm(self):
        from omlx.utils.model_loading import _is_yarn_compatible

        assert _is_yarn_compatible({}, "qwen3_5", is_vlm=True) is True

    def test_qwen3_5_moe_vlm(self):
        from omlx.utils.model_loading import _is_yarn_compatible

        assert _is_yarn_compatible({}, "qwen3_5_moe", is_vlm=True) is True

    def test_qwen3_5_not_vlm(self):
        from omlx.utils.model_loading import _is_yarn_compatible

        assert _is_yarn_compatible({}, "qwen3_5", is_vlm=False) is False

    def test_llama_vlm(self):
        from omlx.utils.model_loading import _is_yarn_compatible

        assert _is_yarn_compatible({}, "llama", is_vlm=True) is False

    def test_none_model_type(self):
        from omlx.utils.model_loading import _is_yarn_compatible

        assert _is_yarn_compatible({}, None, is_vlm=True) is False


class TestDispatchFromConfig:
    """Test that maybe_apply_pre_load_patches dispatches YaRN correctly."""

    def _write_config(self, tmp_dir: Path, config: dict) -> None:
        (tmp_dir / "config.json").write_text(json.dumps(config))

    def test_yarn_config_triggers_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_config(
                tmp_path,
                {
                    "model_type": "qwen3_5_moe",
                    "text_config": {
                        "max_position_embeddings": 262144,
                        "rope_parameters": {
                            "type": "yarn",
                            "factor": 4.0,
                            "original_max_position_embeddings": 262144,
                            "mrope_section": [11, 11, 10],
                            "rope_theta": 10000000,
                            "partial_rotary_factor": 0.25,
                        },
                    },
                },
            )

            with patch(
                "omlx.patches.qwen35_yarn_rope.apply_qwen35_yarn_rope_patch",
                return_value=True,
            ) as mock_apply, patch(
                "omlx.patches.qwen35_yarn_rope.set_yarn_params"
            ) as mock_set:
                from omlx.utils.model_loading import maybe_apply_pre_load_patches

                maybe_apply_pre_load_patches(str(tmp_path), for_vlm=True)

                # Should have been called twice: once with None (reset),
                # once with params (activation)
                calls = mock_set.call_args_list
                assert any(c.args == (None,) for c in calls)
                param_calls = [c for c in calls if c.args != (None,)]
                assert len(param_calls) == 1
                params = param_calls[0].args[0]
                assert params["factor"] == 4.0
                assert params["orig_max_pos"] == 262144
                mock_apply.assert_called_once()

    def test_default_type_does_not_trigger(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_config(
                tmp_path,
                {
                    "model_type": "qwen3_5_moe",
                    "text_config": {
                        "rope_parameters": {
                            "type": "default",
                            "mrope_section": [11, 11, 10],
                            "rope_theta": 10000000,
                            "partial_rotary_factor": 0.25,
                        },
                    },
                },
            )

            with patch(
                "omlx.patches.qwen35_yarn_rope.apply_qwen35_yarn_rope_patch"
            ) as mock_apply:
                from omlx.utils.model_loading import maybe_apply_pre_load_patches

                maybe_apply_pre_load_patches(str(tmp_path), for_vlm=True)
                mock_apply.assert_not_called()

    def test_non_vlm_does_not_trigger(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_config(
                tmp_path,
                {
                    "model_type": "qwen3_5_moe",
                    "text_config": {
                        "rope_parameters": {
                            "type": "yarn",
                            "factor": 2.0,
                            "mrope_section": [11, 11, 10],
                            "rope_theta": 10000000,
                            "partial_rotary_factor": 0.25,
                        },
                    },
                },
            )

            with patch(
                "omlx.patches.qwen35_yarn_rope.apply_qwen35_yarn_rope_patch"
            ) as mock_apply:
                from omlx.utils.model_loading import maybe_apply_pre_load_patches

                # for_vlm=False — should not dispatch YaRN
                maybe_apply_pre_load_patches(str(tmp_path), for_vlm=False)
                mock_apply.assert_not_called()

    def test_settings_override_enables_yarn(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_config(
                tmp_path,
                {
                    "model_type": "qwen3_5_moe",
                    "text_config": {
                        "max_position_embeddings": 262144,
                        "rope_parameters": {
                            "type": "default",
                            "mrope_section": [11, 11, 10],
                            "rope_theta": 10000000,
                            "partial_rotary_factor": 0.25,
                        },
                    },
                },
            )

            settings = MagicMock()
            settings.yarn_enabled = True
            settings.yarn_factor = 3.0

            with patch(
                "omlx.patches.qwen35_yarn_rope.apply_qwen35_yarn_rope_patch",
                return_value=True,
            ) as mock_apply, patch(
                "omlx.patches.qwen35_yarn_rope.set_yarn_params"
            ) as mock_set:
                from omlx.utils.model_loading import maybe_apply_pre_load_patches

                maybe_apply_pre_load_patches(
                    str(tmp_path), model_settings=settings, for_vlm=True
                )

                param_calls = [
                    c for c in mock_set.call_args_list if c.args != (None,)
                ]
                assert len(param_calls) == 1
                params = param_calls[0].args[0]
                assert params["factor"] == 3.0
                mock_apply.assert_called_once()
