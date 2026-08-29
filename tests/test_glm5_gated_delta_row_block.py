# SPDX-License-Identifier: Apache-2.0
"""Exactness tests for GLM-5's vector-gated recurrent row blocking."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mlx.core as mx
import pytest

_MODULE_PATH = (
    Path(__file__).parents[1]
    / "omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/"
    "glm5_next/gated_delta.py"
)


def _load_gated_delta_module():
    spec = importlib.util.spec_from_file_location(
        "_omlx_test_glm5_gated_delta", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gated_delta():
    if not mx.metal.is_available():
        pytest.skip("GLM-5 recurrent row blocking requires Metal")
    return _load_gated_delta_module()


def _glm_inputs(tokens: int, *, seed: int = 11):
    """Return the production GLM-5 GDN geometry (64 heads, width 128)."""
    mx.random.seed(seed)
    batch, heads, width = 1, 64, 128
    recurrent = (batch, tokens, heads, width)

    def bf16(low: float, high: float, shape):
        return mx.random.uniform(low, high, shape, dtype=mx.float32).astype(
            mx.bfloat16
        )

    return {
        "q": bf16(-0.1, 0.1, recurrent),
        "k": bf16(-0.1, 0.1, recurrent),
        "v": bf16(-0.1, 0.1, recurrent),
        "a": bf16(-1.0, 1.0, recurrent),
        "b": bf16(-1.0, 1.0, (batch, tokens, heads)),
        "A_log": mx.random.uniform(
            -2.0, 0.0, (heads, width), dtype=mx.float32
        ),
        "dt_bias": bf16(-1.0, 1.0, (heads, width)),
        "state": mx.random.uniform(
            -0.1,
            0.1,
            (batch, heads, width, width),
            dtype=mx.float32,
        ),
    }


def _assert_bitwise_equal(left, right):
    mx.eval(left, right)
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    assert bool(mx.all(left == right).item())


@pytest.mark.parametrize("tokens", [1, 5, 17])
def test_vector_gate_r4_is_bitwise_equal_to_legacy_r1(gated_delta, tokens):
    args = _glm_inputs(tokens, seed=100 + tokens)
    gate = gated_delta.compute_g_safe(
        args["A_log"], args["a"], args["dt_bias"], -5.0
    )
    beta = mx.sigmoid(args["b"])

    legacy_y, legacy_state = gated_delta._gated_delta_kernel_rows(
        args["q"],
        args["k"],
        args["v"],
        gate,
        beta,
        args["state"],
        rows_per_thread=1,
        threadgroup_y=4,
    )
    blocked_y, blocked_state = gated_delta._gated_delta_kernel_rows(
        args["q"],
        args["k"],
        args["v"],
        gate,
        beta,
        args["state"],
        rows_per_thread=4,
        threadgroup_y=2,
    )

    _assert_bitwise_equal(blocked_y, legacy_y)
    _assert_bitwise_equal(blocked_state, legacy_state)


def test_r4_preserves_masked_output_and_cache_semantics(gated_delta):
    args = _glm_inputs(5, seed=211)
    gate = gated_delta.compute_g_safe(
        args["A_log"], args["a"], args["dt_bias"], -5.0
    )
    beta = mx.sigmoid(args["b"])
    mask = mx.array([[True, False, True, False, True]])

    legacy_y, legacy_state = gated_delta._gated_delta_kernel_rows(
        args["q"],
        args["k"],
        args["v"],
        gate,
        beta,
        args["state"],
        mask,
        rows_per_thread=1,
        threadgroup_y=4,
    )
    blocked_y, blocked_state = gated_delta._gated_delta_kernel_rows(
        args["q"],
        args["k"],
        args["v"],
        gate,
        beta,
        args["state"],
        mask,
        rows_per_thread=4,
        threadgroup_y=2,
    )

    _assert_bitwise_equal(blocked_y, legacy_y)
    _assert_bitwise_equal(blocked_state, legacy_state)
    assert bool(mx.all(blocked_y[:, 1::2] == 0).item())

    all_masked = mx.zeros((1, 5), dtype=mx.bool_)
    masked_y, masked_state = gated_delta.gated_delta_kernel(
        args["q"], args["k"], args["v"], gate, beta, args["state"], all_masked
    )
    _assert_bitwise_equal(masked_y, mx.zeros_like(masked_y))
    _assert_bitwise_equal(masked_state, args["state"])


def test_update_keeps_glm_safe_lower_bound_gate(gated_delta):
    args = _glm_inputs(4, seed=307)
    gate = gated_delta.compute_g_safe(
        args["A_log"], args["a"], args["dt_bias"], -5.0
    )
    beta = mx.sigmoid(args["b"])
    expected_y, expected_state = gated_delta._gated_delta_kernel_rows(
        args["q"],
        args["k"],
        args["v"],
        gate,
        beta,
        args["state"],
        rows_per_thread=4,
        threadgroup_y=2,
    )
    actual_y, actual_state = gated_delta.gated_delta_update(
        args["q"],
        args["k"],
        args["v"],
        args["a"],
        args["b"],
        args["A_log"],
        args["dt_bias"],
        state=args["state"],
        lower_bound=-5.0,
    )

    _assert_bitwise_equal(actual_y, expected_y)
    _assert_bitwise_equal(actual_state, expected_state)
