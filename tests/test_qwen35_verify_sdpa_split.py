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
from mlx_lm.models.cache import BatchKVCache, KVCache

from omlx.patches.mlx_vlm_mtp.qwen35_verify_attention import verify_attention
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


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "batch,length,size,masked,cache_kind",
    [
        (1, 2, 32, False, "ordinary"),
        (2, 3, 193, True, "ordinary"),
        (4, 4, 513, False, "qsa"),
        (4, 4, 513, True, "qsa"),
    ],
)
@pytest.mark.parametrize("dim", [64, 96, 128, 256])
def test_matches_single_query_reductions(
    dtype, batch, length, size, dim, masked, cache_kind
):
    mx.random.seed(14)
    heads, kv_heads = 8, 2
    # Transposing also tests non-contiguous inputs.
    queries = (
        mx.random.normal((batch, length, heads, dim))
        .astype(dtype)
        .transpose(0, 2, 1, 3)
    )
    keys = (
        mx.random.normal((batch, size, kv_heads, dim))
        .astype(dtype)
        .transpose(0, 2, 1, 3)
    )
    values = (
        mx.random.normal((batch, size, kv_heads, dim))
        .astype(dtype)
        .transpose(0, 2, 1, 3)
    )
    pads = [0, 3, 7, 11][:batch]
    if cache_kind == "qsa":
        from omlx.patches.mlx_vlm_qwen4_exp_compat import (
            apply_mlx_vlm_qwen4_exp_compat_patch,
        )

        apply_mlx_vlm_qwen4_exp_compat_patch()
        from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache

        cache = BatchQSAKVCache(pads)
    else:
        cache = BatchKVCache(pads)
    mask = None
    if masked:
        mask = mx.arange(size)[None, None, None, :] % 5 != 0
        # Canonical causal masking plus holes in past context.
        mask = mask & (
            mx.arange(size)[None, None, None, :]
            < size - length + mx.arange(length)[None, None, :, None] + 1
        )
        mask = mx.broadcast_to(mask, (batch, 1, length, size))
    actual = verify_attention(
        queries, keys, values, cache=cache, scale=dim**-0.5, mask=mask
    )
    assert actual is not None
    rows = []
    for b, pad in enumerate(pads):
        tokens = []
        for t in range(length):
            end = size - length + t + 1
            tokens.append(
                mx.fast.scaled_dot_product_attention(
                    queries[b : b + 1, :, t : t + 1],
                    keys[b : b + 1, :, pad:end],
                    values[b : b + 1, :, pad:end],
                    scale=dim**-0.5,
                    mask=mask[b : b + 1, :, t : t + 1, pad:end] if masked else None,
                )
            )
        rows.append(mx.concatenate(tokens, axis=2))
    expected = mx.concatenate(rows, axis=0)
    assert mx.array_equal(actual, expected).item()


def test_future_and_left_padding_are_invisible():
    mx.random.seed(33)
    q = mx.random.normal((2, 8, 3, 128)).astype(mx.float16)
    k = mx.random.normal((2, 2, 64, 128)).astype(mx.float16)
    v = mx.random.normal((2, 2, 64, 128)).astype(mx.float16)
    cache = BatchKVCache([3, 7])
    expected = verify_attention(q, k, v, cache=cache, scale=128**-0.5, mask=None)
    k2, v2 = mx.array(k), mx.array(v)
    for row, pad in enumerate([3, 7]):
        k2[row, :, :pad] = 100
        v2[row, :, :pad] = 100
    k2[:, :, -2:] = 100
    v2[:, :, -2:] = 100
    actual = verify_attention(q, k2, v2, cache=cache, scale=128**-0.5, mask=None)
    assert mx.array_equal(actual[:, :, :1], expected[:, :, :1]).item()


@pytest.mark.parametrize(
    "case", ["single", "single_long", "long", "float32", "additive_mask", "cache"]
)
def test_unsupported_contract_returns_none(case):
    length = 1 if case == "single" else 3
    size = 8193 if case == "long" else 1024 if case == "single_long" else 64
    dtype = mx.float32 if case == "float32" else mx.float16
    batch = 1 if case == "single_long" else 2
    q = mx.zeros((batch, 8, length, 128), dtype)
    k = mx.zeros((batch, 2, size, 128), dtype)
    cache = object() if case == "cache" else BatchKVCache([0, 3][:batch])
    mask = mx.zeros((length, size)) if case == "additive_mask" else None
    assert verify_attention(q, k, k, cache=cache, scale=1.0, mask=mask) is None


def test_singleton_cache_causal_mask():
    q = mx.ones((1, 2, 3, 64), mx.float16)
    k = mx.ones((1, 1, 16, 64), mx.float16)
    result = verify_attention(q, k, k, cache=KVCache(), scale=0.125, mask="causal")
    assert mx.array_equal(result, q).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "batch,length,size,masked,cache_kind",
    [
        (2, 2, 1024, False, "ordinary"),
        (3, 3, 2048, True, "qsa"),
        (4, 4, 4096, True, "ordinary"),
        (4, 2, 8192, False, "qsa"),
    ],
)
@pytest.mark.parametrize("dim", [64, 96, 128, 256])
def test_causal_padding_and_high_precision_reference(
    dtype, batch, length, size, masked, cache_kind, dim
):
    mx.random.seed(812)
    heads, kv_heads = 6, 2
    q = (
        mx.random.normal((batch, length, heads, dim))
        .astype(dtype)
        .transpose(0, 2, 1, 3)
    )
    k = (
        mx.random.normal((batch, size, kv_heads, dim))
        .astype(dtype)
        .transpose(0, 2, 1, 3)
    )
    v = (
        mx.random.normal((batch, size, kv_heads, dim))
        .astype(dtype)
        .transpose(0, 2, 1, 3)
    )
    pads = [0, 17, size // 3, size - length - 1][:batch]
    if cache_kind == "qsa":
        from omlx.patches.mlx_vlm_qwen4_exp_compat import (
            apply_mlx_vlm_qwen4_exp_compat_patch,
        )

        apply_mlx_vlm_qwen4_exp_compat_patch()
        from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache

        cache = BatchQSAKVCache(pads)
    else:
        cache = BatchKVCache(pads)
    mask = None
    if masked:
        mask = mx.broadcast_to(
            (mx.arange(size) % 7 != 0)[None, None, None, :], (batch, 1, length, size)
        )
    actual = verify_attention(q, k, v, cache=cache, scale=dim**-0.5, mask=mask)
    assert actual is not None
    reference = []
    for row, pad in enumerate(pads):
        steps = []
        for t in range(length):
            end = size - length + t + 1
            steps.append(
                mx.fast.scaled_dot_product_attention(
                    q[row : row + 1, :, t : t + 1].astype(mx.float32),
                    k[row : row + 1, :, pad:end].astype(mx.float32),
                    v[row : row + 1, :, pad:end].astype(mx.float32),
                    scale=dim**-0.5,
                    mask=mask[row : row + 1, :, t : t + 1, pad:end] if masked else None,
                )
            )
        reference.append(mx.concatenate(steps, axis=2))
    reference = mx.concatenate(reference, axis=0)
    # A one-ULP envelope at each element, plus FP32 reduction error near zero.
    eps = 2 ** (-7 if dtype == mx.bfloat16 else -10)
    tolerance = eps * mx.abs(reference) + 2e-6
    assert mx.all(mx.abs(actual.astype(mx.float32) - reference) <= tolerance).item()
    # Change inaccessible padding and future positions; first query is invariant.
    k2, v2 = mx.array(k), mx.array(v)
    for row, pad in enumerate(pads):
        k2[row, :, :pad] = 100
        v2[row, :, :pad] = -100
    k2[:, :, -(length - 1) :] = -100
    v2[:, :, -(length - 1) :] = 100
    changed = verify_attention(q, k2, v2, cache=cache, scale=dim**-0.5, mask=mask)
    assert mx.array_equal(changed[:, :, :1], actual[:, :, :1]).item()


def test_fully_masked_rows_are_zero_and_bound_is_explicit():
    q = mx.ones((2, 6, 2, 256), mx.bfloat16)
    k = mx.ones((2, 2, 2048, 256), mx.bfloat16)
    mask = mx.zeros((2, 1, 2, 2048), mx.bool_)
    result = verify_attention(
        q, k, k, cache=BatchKVCache([0, 100]), scale=0.0625, mask=mask
    )
    assert mx.all(result == 0).item()
    long_k = mx.ones((2, 2, 8193, 256), mx.bfloat16)
    assert (
        verify_attention(
            q, long_k, long_k, cache=BatchKVCache([0, 0]), scale=0.0625, mask=None
        )
        is None
    )


def test_explicit_mask_is_preserved_when_verify_kernel_declines():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from omlx.patches.mlx_vlm_mtp.qwen35_verify_attention import apply

    original = Mock(return_value=mx.ones((2, 1, 1, 64)))
    language = SimpleNamespace(_qwen3_5_left_padded_attention=original)
    apply(language)
    q = mx.ones((2, 1, 1, 64), dtype=mx.bfloat16)
    k = mx.ones((2, 1, 4, 64), dtype=mx.bfloat16)
    mask = mx.array([True, False, True, False])[None, None, None, :]
    result = language._qwen3_5_left_padded_attention(
        q, k, k, cache=BatchKVCache([0, 0]), scale=0.125, mask=mask
    )
    assert result is None
    original.assert_not_called()
