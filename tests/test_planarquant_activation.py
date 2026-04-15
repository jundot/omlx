# SPDX-License-Identifier: Apache-2.0
"""Tests for the activation hook that patches make_prompt_cache."""

from __future__ import annotations

from mlx_lm.models import cache as mlx_cache

from omlx.cache.planarquant.kv_cache import PlanarQuantKVCache
from omlx.patches.planarquant_cache import (
    active_bits,
    disable_planarquant_cache,
    enable_planarquant_cache,
    is_planarquant_active,
)


def _make_fake_model():
    class _FakeLayer:
        pass

    class _FakeModel:
        layers = [_FakeLayer() for _ in range(4)]

    return _FakeModel()


def setup_function(_):
    disable_planarquant_cache()


def teardown_function(_):
    disable_planarquant_cache()


def test_disabled_by_default():
    assert not is_planarquant_active()
    assert active_bits() is None


def test_enable_is_idempotent():
    enable_planarquant_cache(3.0)
    assert is_planarquant_active()
    assert active_bits() == 3.0
    enable_planarquant_cache(3.0)
    assert active_bits() == 3.0


def test_disable_restores_factory():
    original = mlx_cache.make_prompt_cache
    enable_planarquant_cache(3.0)
    assert mlx_cache.make_prompt_cache is not original
    disable_planarquant_cache()
    assert mlx_cache.make_prompt_cache is original
    assert not is_planarquant_active()


def test_make_prompt_cache_returns_planarquant_when_active():
    model = _make_fake_model()
    baseline = mlx_cache.make_prompt_cache(model)
    assert len(baseline) == 4
    assert not any(isinstance(c, PlanarQuantKVCache) for c in baseline)

    enable_planarquant_cache(3.0)
    wrapped = mlx_cache.make_prompt_cache(model)
    assert len(wrapped) == 4
    assert all(isinstance(c, PlanarQuantKVCache) for c in wrapped)
    for c in wrapped:
        assert c.bits == 3.0


def test_quantize_v_flag_propagated():
    enable_planarquant_cache(3.0, quantize_v=False)
    model = _make_fake_model()
    wrapped = mlx_cache.make_prompt_cache(model)
    assert all(not c.quantize_v for c in wrapped)


def test_model_settings_round_trip():
    """ModelSettings round-trips PQ fields through to_dict / from_dict."""
    from omlx.model_settings import ModelSettings

    s = ModelSettings(
        planarquant_kv_enabled=True,
        planarquant_kv_bits=3,
        planarquant_quantize_v=False,
    )
    d = s.to_dict()
    assert d["planarquant_kv_enabled"] is True
    assert d["planarquant_kv_bits"] == 3
    assert d["planarquant_quantize_v"] is False

    s2 = ModelSettings.from_dict(d)
    assert s2.planarquant_kv_enabled is True
    assert s2.planarquant_kv_bits == 3
    assert s2.planarquant_quantize_v is False


def test_model_settings_defaults():
    """Default PQ fields are off / 3-bit / V-quantized."""
    from omlx.model_settings import ModelSettings

    s = ModelSettings()
    assert s.planarquant_kv_enabled is False
    assert s.planarquant_kv_bits == 3
    assert s.planarquant_quantize_v is True
