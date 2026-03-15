# SPDX-License-Identifier: Apache-2.0
"""Tests for IndexCache monkey-patch."""

from unittest.mock import MagicMock, patch

import pytest


class TestBuildLayerPattern:
    """Test _build_layer_pattern function."""

    def test_freq_2(self):
        from omlx.patches.index_cache import _build_layer_pattern

        pattern = _build_layer_pattern(8, 2)
        assert pattern == [True, False, True, False, True, False, True, False]

    def test_freq_4(self):
        from omlx.patches.index_cache import _build_layer_pattern

        pattern = _build_layer_pattern(8, 4)
        assert pattern == [True, False, False, False, True, False, False, False]

    def test_layer_0_always_full(self):
        from omlx.patches.index_cache import _build_layer_pattern

        for freq in range(2, 10):
            pattern = _build_layer_pattern(60, freq)
            assert pattern[0] is True

    def test_freq_4_counts(self):
        from omlx.patches.index_cache import _build_layer_pattern

        pattern = _build_layer_pattern(60, 4)
        full_count = sum(pattern)
        shared_count = len(pattern) - full_count
        assert full_count == 15
        assert shared_count == 45

    def test_freq_larger_than_layers(self):
        from omlx.patches.index_cache import _build_layer_pattern

        pattern = _build_layer_pattern(3, 8)
        # Only layer 0 is Full
        assert pattern == [True, False, False]


class TestGetModelType:
    """Test _get_model_type function."""

    def test_model_type_attribute(self):
        from omlx.patches.index_cache import _get_model_type

        model = MagicMock(spec=[])
        model.model_type = "deepseek_v32"
        assert _get_model_type(model) == "deepseek_v32"

    def test_args_model_type(self):
        from omlx.patches.index_cache import _get_model_type

        model = MagicMock(spec=[])
        model.args = MagicMock(spec=[])
        model.args.model_type = "glm_moe_dsa"
        assert _get_model_type(model) == "glm_moe_dsa"

    def test_no_model_type(self):
        from omlx.patches.index_cache import _get_model_type

        model = MagicMock(spec=[])
        assert _get_model_type(model) is None


class TestApplyIndexCache:
    """Test apply_index_cache function."""

    def test_unsupported_model_returns_false(self):
        from omlx.patches.index_cache import apply_index_cache

        model = MagicMock(spec=[])
        model.model_type = "llama"
        assert apply_index_cache(model, 4) is False

    def test_no_model_type_returns_false(self):
        from omlx.patches.index_cache import apply_index_cache

        model = MagicMock(spec=[])
        assert apply_index_cache(model, 4) is False

    def test_freq_less_than_2_returns_false(self):
        from omlx.patches.index_cache import apply_index_cache

        model = MagicMock(spec=[])
        model.model_type = "deepseek_v32"
        assert apply_index_cache(model, 1) is False

    @patch("omlx.patches.index_cache._class_patch_applied", True)
    def test_applies_flags_to_layers(self):
        from omlx.patches.index_cache import apply_index_cache

        # Create a mock model with 4 layers
        model = MagicMock(spec=[])
        model.model_type = "deepseek_v32"
        model.args = MagicMock(spec=[])
        model.args.model_type = "deepseek_v32"

        layers = []
        for _ in range(4):
            layer = MagicMock(spec=[])
            layer.self_attn = MagicMock(spec=[])
            layers.append(layer)

        model.model = MagicMock(spec=[])
        model.model.layers = layers

        result = apply_index_cache(model, 2)
        assert result is True

        # Check flags on each layer
        assert layers[0].self_attn._ic_is_full is True  # layer 0: Full
        assert layers[1].self_attn._ic_is_full is False  # layer 1: Shared
        assert layers[2].self_attn._ic_is_full is True  # layer 2: Full
        assert layers[3].self_attn._ic_is_full is False  # layer 3: Shared

        # Check shared state
        assert hasattr(model.model, "_index_cache_state")
        assert model.model._index_cache_state["last_topk_indices"] is None

    @patch("omlx.patches.index_cache._class_patch_applied", True)
    def test_skips_none_layers(self):
        """None layers (pipeline parallel placeholders) should be skipped."""
        from omlx.patches.index_cache import apply_index_cache

        model = MagicMock(spec=[])
        model.model_type = "deepseek_v32"
        model.args = MagicMock(spec=[])
        model.args.model_type = "deepseek_v32"

        layer0 = MagicMock(spec=[])
        layer0.self_attn = MagicMock(spec=[])

        model.model = MagicMock(spec=[])
        model.model.layers = [layer0, None, None]

        result = apply_index_cache(model, 2)
        assert result is True
        assert layer0.self_attn._ic_is_full is True


class TestApplyPostLoadTransforms:
    """Test the centralized transform entry point."""

    def test_none_settings_returns_model(self):
        from omlx.utils.model_loading import apply_post_load_transforms

        model = MagicMock()
        result = apply_post_load_transforms(model, None)
        assert result is model

    def test_no_index_cache_freq_returns_model(self):
        from omlx.utils.model_loading import apply_post_load_transforms

        model = MagicMock()
        settings = MagicMock(spec=[])
        settings.index_cache_freq = None
        result = apply_post_load_transforms(model, settings)
        assert result is model

    @patch("omlx.patches.index_cache.apply_index_cache")
    def test_calls_apply_index_cache(self, mock_apply):
        from omlx.utils.model_loading import apply_post_load_transforms

        mock_apply.return_value = True
        model = MagicMock()
        settings = MagicMock(spec=[])
        settings.index_cache_freq = 4
        result = apply_post_load_transforms(model, settings)
        mock_apply.assert_called_once_with(model, 4)
        assert result is model

    @patch("omlx.patches.index_cache.apply_index_cache")
    def test_freq_1_skipped(self, mock_apply):
        from omlx.utils.model_loading import apply_post_load_transforms

        model = MagicMock()
        settings = MagicMock(spec=[])
        settings.index_cache_freq = 1
        result = apply_post_load_transforms(model, settings)
        mock_apply.assert_not_called()
        assert result is model
