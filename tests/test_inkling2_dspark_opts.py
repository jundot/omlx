# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Inkling 2 DSpark compiled draft ops and flash scale caching."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.inkling2 import dspark, kernels

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def test_mix_rmsnorm_matches_reference():
    mx.random.seed(0)
    b, n, taps, h = 1, 3, 4, 64
    ctx = mx.random.normal((b, n, taps * h)).astype(mx.bfloat16)
    alpha = mx.softmax(mx.random.normal((taps,)), axis=0).astype(mx.bfloat16)
    weight = mx.ones((h,), dtype=mx.bfloat16)
    eps = 1e-6

    out = dspark.mix_rmsnorm(ctx, alpha, weight, eps)
    x = ctx.reshape(b, n, taps, h).astype(mx.float32)
    ref = mx.fast.rms_norm(
        mx.sum(x * alpha.astype(mx.float32)[:, None], axis=2),
        weight.astype(mx.float32),
        eps,
    ).astype(ctx.dtype)
    mx.eval(out, ref)
    assert mx.allclose(out, ref, atol=1e-3, rtol=1e-3).item()


def test_draft_swiglu_matches_reference():
    mx.random.seed(1)
    gate = mx.random.normal((1, 4, 128)).astype(mx.bfloat16)
    up = mx.random.normal((1, 4, 128)).astype(mx.bfloat16)
    out = dspark._draft_swiglu(gate, up)
    ref = nn.silu(gate) * up
    mx.eval(out, ref)
    assert mx.allclose(out, ref, atol=1e-3, rtol=1e-3).item()


def test_markov_logit_tail_compiled_matches_reference():
    mx.random.seed(2)
    base = mx.random.normal((1, 256)).astype(mx.bfloat16)
    bias = mx.random.normal((1, 256)).astype(mx.bfloat16)
    vocab_bias = mx.where(mx.arange(256) < 200, 0.0, -mx.inf).astype(mx.float32)
    cap = 30.0

    out = dspark._markov_logit_tail_compiled(base, bias, cap, vocab_bias)
    ref = mx.tanh((base.astype(mx.float32) + bias.astype(mx.float32)) / cap) * cap
    ref = ref + vocab_bias
    mx.eval(out, ref)
    assert mx.allclose(out[:, :200], ref[:, :200], atol=1e-4, rtol=1e-4).item()
    assert mx.all(mx.isneginf(out[:, 200:])).item()


def test_flash_banded_attention_scale_cache_populated():
    kernels._FLASH_SCALE_CACHE.clear()
    mx.random.seed(3)
    b, hq, hkv, length, s, d, r = 1, 4, 2, 32, 32, 64, 32
    q = mx.random.normal((b, hq, length, d)).astype(mx.bfloat16)
    k = mx.random.normal((b, hkv, s, d)).astype(mx.bfloat16)
    v = mx.random.normal((b, hkv, s, d)).astype(mx.bfloat16)
    rel = mx.zeros((b, length, hq, r), dtype=mx.bfloat16)
    sink = mx.zeros((b, length, hq), dtype=mx.bfloat16)
    scale = d**-0.5

    out1 = kernels.flash_banded_attention(q, k, v, rel, sink, scale=scale, window=r)
    out2 = kernels.flash_banded_attention(q, k, v, rel, sink, scale=scale, window=r)
    mx.eval(out1, out2)
    assert scale in kernels._FLASH_SCALE_CACHE
    assert mx.array_equal(out1, out2).item()
