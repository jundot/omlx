# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the atomic MLX 0.32.2 upgrade."""

from __future__ import annotations

import concurrent.futures
import importlib

import mlx.core as mx
import pytest


def test_runtime_uses_exact_mlx_0322():
    assert mx.__version__ == "0.32.2"


@pytest.mark.parametrize("static", [False, True])
def test_diffusion_cache_view_preserves_decoder_extent(static):
    from mlx_vlm.models.cache import KVCache, StaticPrefixKVCache
    from mlx_vlm.models.diffusion_gemma.language import _cache_state

    cache = StaticPrefixKVCache(max_size=8) if static else KVCache()
    assert _cache_state(cache) is None
    keys = mx.arange(12, dtype=mx.float32).reshape(1, 1, 3, 4)
    values = keys + 1
    cache.update_and_fetch(keys, values)
    assert cache.keys.shape[2] > cache.offset

    actual_keys, actual_values = _cache_state(cache)
    assert actual_keys.shape[2] == (8 if static else 3)
    assert mx.array_equal(actual_keys[..., :3, :], keys).item()
    assert mx.array_equal(actual_values[..., :3, :], values).item()


def test_mlx_vlm_qwen2_array_grid_runs_after_backport():
    # Exercise the defensive early-import path as well as first-time imports.
    vision = importlib.import_module("mlx_vlm.models.qwen2_vl.vision")
    vision_model_cls = vision.VisionModel

    model = vision_model_cls.__new__(vision_model_cls)
    model.spatial_merge_size = 1
    model.rotary_pos_emb = lambda _size: mx.zeros((1, 2))
    result = model.rot_pos_emb(mx.array([[2, 1, 1]], dtype=mx.int32))
    mx.eval(result)
    assert result.shape[0] == 2


def test_mlx_vlm_grid_sample_uses_python_integer_metal_grid():

    from mlx_vlm.models.kernels import grid_sample

    values = mx.arange(4, dtype=mx.float32).reshape(1, 2, 2, 1)
    grid = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
    result = grid_sample(values, grid)
    mx.eval(result)
    assert result.shape == (1, 1, 1, 1)


def test_mlx_vlm_speculative_rng_restores_random_state_in_place():

    from mlx_vlm.speculative.common import _restore_rng_state

    original = [mx.array(value) for value in mx.random.state]
    replacement = [value + 1 for value in original]
    try:
        _restore_rng_state(replacement)
        mx.eval(*mx.random.state)
        assert all(
            bool(mx.all(actual == expected))
            for actual, expected in zip(mx.random.state, replacement)
        )
    finally:
        _restore_rng_state(original)


def _compiled_thread_call(value: int) -> int:
    @mx.compile
    def add_one(x):
        return x + 1

    return int(add_one(mx.array(value)).item())


def test_compiled_worker_cache_can_be_destroyed_repeatedly():
    """MLX #4391 must make thread-local compiled-data teardown GIL-safe."""
    for value in range(20):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(_compiled_thread_call, value).result() == value + 1
