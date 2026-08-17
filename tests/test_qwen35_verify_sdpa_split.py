# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the Qwen3.5/3.6 verify-width chunked causal attention.

``_chunked_causal_sdpa`` must reproduce the per-row loop it replaces: row i
of a verify block attends ``keys[: prefix + i + 1]``. Chunks at the vector
kernel row limit ride the same kernel family as the loop, so agreement is
bit-exact at short KV and bf16 tail-ULP at long KV (2-pass reduction split).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches.qwen35_verify_sdpa_split import (
    _chunked_causal_sdpa,
    _eligible,
)

HQ, HKV, HD = 24, 4, 256


def _per_row_reference(q, k, v, scale):
    q_len = q.shape[2]
    prefix = k.shape[2] - q_len
    outs = []
    for i in range(q_len):
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q[:, :, i : i + 1, :],
                k[:, :, : prefix + i + 1, :],
                v[:, :, : prefix + i + 1, :],
                scale=scale,
                mask=None,
            )
        )
    return mx.concatenate(outs, axis=2)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("q_len", [2, 4, 5, 6, 7, 9])
@pytest.mark.parametrize("kv_len", [512, 2048])
def test_chunked_causal_matches_per_row(q_len, kv_len):
    mx.random.seed(7)
    q = mx.random.normal((1, HQ, q_len, HD)).astype(mx.bfloat16)
    k = mx.random.normal((1, HKV, kv_len, HD)).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, kv_len, HD)).astype(mx.bfloat16)
    scale = HD**-0.5
    ref = _per_row_reference(q, k, v, scale)
    got = _chunked_causal_sdpa(q, k, v, scale, limit=32 // (HQ // HKV))
    diff = mx.abs(
        ref.astype(mx.float32) - got.astype(mx.float32)
    ).max().item()
    # Same kernel family; short KV is bit-exact, long KV differs only in
    # the 2-pass reduction split (bf16 tail ULP).
    assert diff <= 3e-4, f"q_len={q_len} kv_len={kv_len} diff={diff}"


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_eligibility_gates():
    q = mx.random.normal((1, HQ, 4, HD)).astype(mx.bfloat16)
    k = mx.random.normal((1, HKV, 256, HD)).astype(mx.bfloat16)
    assert _eligible(q, k, None) > 0
    # batch > 1 is not ours
    q2 = mx.random.normal((2, HQ, 4, HD)).astype(mx.bfloat16)
    k2 = mx.random.normal((2, HKV, 256, HD)).astype(mx.bfloat16)
    assert _eligible(q2, k2, None) == 0
    # non-256 head dim is not ours
    q3 = mx.random.normal((1, HQ, 4, 128)).astype(mx.bfloat16)
    k3 = mx.random.normal((1, HKV, 256, 128)).astype(mx.bfloat16)
    assert _eligible(q3, k3, None) == 0
    # single row (plain decode) is not ours
    q4 = mx.random.normal((1, HQ, 1, HD)).astype(mx.bfloat16)
    assert _eligible(q4, k, None) == 0

    class _QuantCache:
        bits = 4

    assert _eligible(q, k, _QuantCache()) == 0


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_eligibility_gates_turboquant_proxy():
    """A turboquant-quantized KV cache hands back a proxy with .shape but no
    .ndim (mlx_vlm.turboquant._QuantizedStateProxy, kept dequantized-free on
    purpose). _eligible() must treat that as "not ours" rather than raising —
    it used to crash every verify forward once turboquant KV compression was
    active, since the .ndim check ran before the cache-type guard could rule
    the call out.
    """

    class _TurboQuantProxy:
        def __init__(self, shape):
            self.shape = shape

    q = mx.random.normal((1, HQ, 4, HD)).astype(mx.bfloat16)
    k = mx.random.normal((1, HKV, 256, HD)).astype(mx.bfloat16)
    assert _eligible(_TurboQuantProxy((1, HQ, 4, HD)), k, None) == 0
    assert _eligible(q, _TurboQuantProxy((1, HKV, 256, HD)), None) == 0
