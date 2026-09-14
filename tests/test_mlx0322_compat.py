# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the atomic MLX 0.32.2 upgrade."""

from __future__ import annotations

import concurrent.futures
import importlib
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import mlx.core as mx


def test_runtime_uses_exact_mlx_0322():
    assert mx.__version__ == "0.32.2"


def test_mlx_vlm_backport_matches_every_pinned_source():
    from omlx.patches.mlx_vlm_mlx0322_compat import (
        _MODULE_REPLACEMENTS,
        _patch_source,
    )

    package_root = Path(distribution("mlx-vlm").locate_file(""))
    for fullname, replacements in _MODULE_REPLACEMENTS.items():
        source_path = package_root / f"{fullname.replace('.', '/')}.py"
        source = source_path.read_text()
        patched = _patch_source(fullname, source)
        for replacement in replacements:
            assert patched.count(replacement.new) == replacement.count
        compile(patched, str(source_path), "exec")


def test_mlx_vlm_qwen2_array_grid_runs_after_backport():
    from omlx.patches.mlx_vlm_mlx0322_compat import (
        apply_mlx_vlm_mlx0322_compat_patch,
    )

    # Exercise the defensive early-import path as well as first-time imports.
    vision = importlib.import_module("mlx_vlm.models.qwen2_vl.vision")
    apply_mlx_vlm_mlx0322_compat_patch()
    vision_model_cls = vision.VisionModel

    model = vision_model_cls.__new__(vision_model_cls)
    model.spatial_merge_size = 1
    model.rotary_pos_emb = lambda _size: mx.zeros((1, 2))
    result = model.rot_pos_emb(mx.array([[2, 1, 1]], dtype=mx.int32))
    mx.eval(result)
    assert result.shape[0] == 2


def test_mlx_vlm_grid_sample_uses_python_integer_metal_grid():
    from omlx.patches.mlx_vlm_mlx0322_compat import (
        apply_mlx_vlm_mlx0322_compat_patch,
    )

    apply_mlx_vlm_mlx0322_compat_patch()

    from mlx_vlm.models.kernels import grid_sample

    values = mx.arange(4, dtype=mx.float32).reshape(1, 2, 2, 1)
    grid = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
    result = grid_sample(values, grid)
    mx.eval(result)
    assert result.shape == (1, 1, 1, 1)


def test_mlx_vlm_speculative_rng_restores_random_state_in_place():
    from omlx.patches.mlx_vlm_mlx0322_compat import (
        apply_mlx_vlm_mlx0322_compat_patch,
    )

    apply_mlx_vlm_mlx0322_compat_patch()

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


def test_early_mlx_vlm_import_rebinds_mtp_generation_stream():
    """The defensive reload path must not split common and MTP streams."""
    code = """
from mlx_vlm.speculative import common, mtp
old_stream = common.generation_stream
assert mtp.generation_stream is old_stream

from omlx.patches.mlx_vlm_mlx0322_compat import apply_mlx_vlm_mlx0322_compat_patch
apply_mlx_vlm_mlx0322_compat_patch()

assert common.generation_stream is not old_stream
assert mtp.generation_stream is common.generation_stream
assert mtp._mtp_rounds.__globals__[\"generation_stream\"] is common.generation_stream
assert mtp._mtp_rounds_batch.__globals__[\"generation_stream\"] is common.generation_stream
"""
    subprocess.run([sys.executable, "-c", code], check=True)


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
