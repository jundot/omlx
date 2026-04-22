# SPDX-License-Identifier: Apache-2.0
"""Tests for DFlash hook toggle functions.

These tests verify that _enable_dflash_hooks and _disable_dflash_hooks
correctly toggle the dflash-mlx internal attributes on a model structure.
They serve as a regression guard: if dflash-mlx renames attributes or
changes its internal API, these tests will fail immediately.

No real model is loaded — the test constructs a minimal mock that mirrors
the structure dflash-mlx expects (layers with self_attn and linear_attn).
"""

import pytest

class _MockSelfAttn:
    """Mock self-attention layer that mirrors dflash-mlx's structure."""

    def __init__(self):
        # This is the attribute split_call checks at runtime
        self._dflash_split_sdpa_enabled = False

class _MockLinearAttn:
    """Mock linear attention layer with in_proj projections."""

    def __init__(self):
        self.in_proj_b = _MockLinear()
        self.in_proj_a = _MockLinear()

class _MockLinear:
    """Mock linear layer (the raw projection inside _ExactSmallProjPad)."""

    def __init__(self):
        self.weight = None
        self.bias = None

class _MockLayer:
    """Single transformer layer with self_attn and linear_attn."""

    def __init__(self):
        self.self_attn = _MockSelfAttn()
        self.linear_attn = _MockLinearAttn()

class _MockTextModel:
    """Minimal text model with layers attribute."""

    def __init__(self, num_layers=40):
        self.layers = [_MockLayer() for _ in range(num_layers)]

class _MockTargetModel:
    """Wraps a text model the way dflash-mlx does (has .model attribute)."""

    def __init__(self, num_layers=40):
        self.model = _MockTextModel(num_layers)

class TestEnableDFlashHooks:
    """Test that _enable_dflash_hooks correctly re-enables hooks."""

    def test_sets_split_sdpa_enabled_true(self):
        """All self_attn layers should have _dflash_split_sdpa_enabled = True."""
        try:
            from omlx.engine.dflash import _enable_dflash_hooks, _get_text_model
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        model = _MockTargetModel(num_layers=40)
        text_model = _get_text_model(model)

        # Initially all False (set by _MockSelfAttn.__init__)
        for layer in text_model.layers:
            assert layer.self_attn._dflash_split_sdpa_enabled is False

        _enable_dflash_hooks(model)

        for layer in text_model.layers:
            assert layer.self_attn._dflash_split_sdpa_enabled is True

    def test_sets_hooks_installed_flag(self):
        """text_model._dflash_speculative_hooks_installed should be True."""
        try:
            from omlx.engine.dflash import _enable_dflash_hooks, _get_text_model
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        model = _MockTargetModel(num_layers=40)
        text_model = _get_text_model(model)

        assert getattr(text_model, "_dflash_speculative_hooks_installed", None) is not True

        _enable_dflash_hooks(model)

        assert text_model._dflash_speculative_hooks_installed is True

    def test_re_wraps_exact_small_proj_pad(self):
        """If projections were unwrapped, _enable should re-wrap them."""
        try:
            from omlx.engine.dflash import _enable_dflash_hooks, _get_text_model
            from dflash_mlx.runtime import _ExactSmallProjPad
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        model = _MockTargetModel(num_layers=40)
        text_model = _get_text_model(model)

        # Simulate unwrapped state: replace _ExactSmallProjPad with raw linear
        for layer in text_model.layers:
            proj_b = layer.linear_attn.in_proj_b
            proj_a = layer.linear_attn.in_proj_a
            # Wrap them first (as they would be after initial dflash init)
            layer.linear_attn.in_proj_b = _ExactSmallProjPad(proj_b)
            layer.linear_attn.in_proj_a = _ExactSmallProjPad(proj_a)

        # Now simulate _disable having unwrapped them
        for layer in text_model.layers:
            layer.linear_attn.in_proj_b = layer.linear_attn.in_proj_b.linear
            layer.linear_attn.in_proj_a = layer.linear_attn.in_proj_a.linear

        # Verify they're unwrapped
        for layer in text_model.layers:
            assert type(layer.linear_attn.in_proj_b).__name__ != "_ExactSmallProjPad"

        # Re-enable should re-wrap
        _enable_dflash_hooks(model)

        for layer in text_model.layers:
            assert type(layer.linear_attn.in_proj_b).__name__ == "_ExactSmallProjPad"
            assert type(layer.linear_attn.in_proj_a).__name__ == "_ExactSmallProjPad"

    def test_idempotent_when_already_enabled(self):
        """Calling _enable twice should be safe (no double-wrap on already-wrapped)."""
        try:
            from omlx.engine.dflash import _enable_dflash_hooks, _get_text_model
            from dflash_mlx.runtime import _ExactSmallProjPad
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        model = _MockTargetModel(num_layers=40)
        text_model = _get_text_model(model)

        # Pre-wrap projections (as dflash-mlx would do on init)
        for layer in text_model.layers:
            raw_b = layer.linear_attn.in_proj_b
            raw_a = layer.linear_attn.in_proj_a
            layer.linear_attn.in_proj_b = _ExactSmallProjPad(raw_b)
            layer.linear_attn.in_proj_a = _ExactSmallProjPad(raw_a)

        # First enable
        _enable_dflash_hooks(model)

        # Verify still single-wrapped (not double-wrapped)
        for layer in text_model.layers:
            assert type(layer.linear_attn.in_proj_b).__name__ == "_ExactSmallProjPad"
            # The inner .linear should be the raw _MockLinear, not another _ExactSmallProjPad
            assert type(layer.linear_attn.in_proj_b.linear).__name__ == "_MockLinear"

        # Second enable should be safe (no-op on already-wrapped)
        _enable_dflash_hooks(model)

        for layer in text_model.layers:
            assert type(layer.linear_attn.in_proj_b).__name__ == "_ExactSmallProjPad"
            assert type(layer.linear_attn.in_proj_b.linear).__name__ == "_MockLinear"

class TestDisableDFlashHooks:
    """Test that _disable_dflash_hooks correctly disables hooks."""

    def test_sets_split_sdpa_enabled_false(self):
        """All self_attn layers should have _dflash_split_sdpa_enabled = False."""
        try:
            from omlx.engine.dflash import _disable_dflash_hooks, _get_text_model
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        model = _MockTargetModel(num_layers=40)
        text_model = _get_text_model(model)

        # Pre-enable (as dflash-mlx would set on init)
        for layer in text_model.layers:
            layer.self_attn._dflash_split_sdpa_enabled = True

        _disable_dflash_hooks(model)

        for layer in text_model.layers:
            assert layer.self_attn._dflash_split_sdpa_enabled is False

    def test_unwraps_exact_small_proj_pad(self):
        """Projections wrapped in _ExactSmallProjPad should be unwrapped."""
        try:
            from omlx.engine.dflash import _disable_dflash_hooks, _get_text_model
            from dflash_mlx.runtime import _ExactSmallProjPad
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        model = _MockTargetModel(num_layers=40)
        text_model = _get_text_model(model)

        # Pre-wrap projections (as dflash-mlx would do on init)
        for layer in text_model.layers:
            raw_b = layer.linear_attn.in_proj_b
            raw_a = layer.linear_attn.in_proj_a
            layer.linear_attn.in_proj_b = _ExactSmallProjPad(raw_b)
            layer.linear_attn.in_proj_a = _ExactSmallProjPad(raw_a)

        _disable_dflash_hooks(model)

        for layer in text_model.layers:
            assert type(layer.linear_attn.in_proj_b).__name__ != "_ExactSmallProjPad"
            assert type(layer.linear_attn.in_proj_a).__name__ != "_ExactSmallProjPad"
            # Should be the raw linear layer
            assert hasattr(layer.linear_attn.in_proj_b, "weight")
            assert hasattr(layer.linear_attn.in_proj_a, "weight")

    def test_sets_hooks_installed_flag_false(self):
        """text_model._dflash_speculative_hooks_installed should be False."""
        try:
            from omlx.engine.dflash import _disable_dflash_hooks, _get_text_model
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        model = _MockTargetModel(num_layers=40)
        text_model = _get_text_model(model)

        # Pre-enable
        for layer in text_model.layers:
            layer.self_attn._dflash_split_sdpa_enabled = True
        text_model._dflash_speculative_hooks_installed = True

        _disable_dflash_hooks(model)

        assert text_model._dflash_speculative_hooks_installed is False

class TestToggleRoundTrip:
    """Test that disable then re-enable restores the model to working state."""

    def test_disable_then_enable_restores_all_flags(self):
        """After disable+enable, all flags should be True again."""
        try:
            from omlx.engine.dflash import (
                _enable_dflash_hooks,
                _disable_dflash_hooks,
                _get_text_model,
            )
            from dflash_mlx.runtime import _ExactSmallProjPad
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        model = _MockTargetModel(num_layers=40)
        text_model = _get_text_model(model)

        # Simulate initial dflash-mlx state
        for layer in text_model.layers:
            layer.self_attn._dflash_split_sdpa_enabled = True
            layer.linear_attn.in_proj_b = _ExactSmallProjPad(layer.linear_attn.in_proj_b)
            layer.linear_attn.in_proj_a = _ExactSmallProjPad(layer.linear_attn.in_proj_a)
        text_model._dflash_speculative_hooks_installed = True

        # Disable (as _init_fallback_engine would do)
        _disable_dflash_hooks(model)

        # Verify disabled
        assert text_model._dflash_speculative_hooks_installed is False
        for layer in text_model.layers:
            assert layer.self_attn._dflash_split_sdpa_enabled is False

        # Re-enable (as generate/stream_generate would do)
        _enable_dflash_hooks(model)

        # Verify fully restored
        assert text_model._dflash_speculative_hooks_installed is True
        for layer in text_model.layers:
            assert layer.self_attn._dflash_split_sdpa_enabled is True
            assert type(layer.linear_attn.in_proj_b).__name__ == "_ExactSmallProjPad"
            assert type(layer.linear_attn.in_proj_a).__name__ == "_ExactSmallProjPad"

    def test_multiple_disable_enable_cycles(self):
        """Multiple disable/enable cycles should be idempotent."""
        try:
            from omlx.engine.dflash import (
                _enable_dflash_hooks,
                _disable_dflash_hooks,
                _get_text_model,
            )
            from dflash_mlx.runtime import _ExactSmallProjPad
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        model = _MockTargetModel(num_layers=40)
        text_model = _get_text_model(model)

        # Simulate initial dflash-mlx state
        for layer in text_model.layers:
            layer.self_attn._dflash_split_sdpa_enabled = True
            layer.linear_attn.in_proj_b = _ExactSmallProjPad(layer.linear_attn.in_proj_b)
            layer.linear_attn.in_proj_a = _ExactSmallProjPad(layer.linear_attn.in_proj_a)
        text_model._dflash_speculative_hooks_installed = True

        for cycle in range(3):
            _disable_dflash_hooks(model)
            assert text_model._dflash_speculative_hooks_installed is False

            _enable_dflash_hooks(model)
            assert text_model._dflash_speculative_hooks_installed is True
            for layer in text_model.layers:
                assert layer.self_attn._dflash_split_sdpa_enabled is True
                assert type(layer.linear_attn.in_proj_b).__name__ == "_ExactSmallProjPad"
