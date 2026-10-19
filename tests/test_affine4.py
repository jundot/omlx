"""Affine4 numerics, native dispatch, and packed cache lifecycle."""

import functools

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache
from mlx_vlm.turboquant import TurboQuantMSEState

import omlx.affine4 as affine4
from omlx.affine4 import (
    Affine4Codec,
    Affine4KVCache,
    Affine8Codec,
    Affine8KVCache,
    BatchAffine4KVCache,
    BatchAffine8KVCache,
)
from omlx.patches import turboquant_attention


def random(shape, dtype=mx.float16, seed=0):
    return mx.random.normal(shape, key=mx.random.key(seed)).astype(dtype)


def close(actual, expected, atol=2e-3, rtol=3e-3):
    mx.eval(actual, expected)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )


@functools.cache
def _tensor_ops_launches() -> bool:
    """Whether this system builds the TensorOps attention kernel."""
    if not affine4._tensor_ops_available():
        return False
    probe = Affine4KVCache()
    keys = random((1, 2, 257, 64))
    probe.update_and_fetch(keys, keys)
    q = random((1, 8, 1, 64), seed=2)
    return affine4._native_attention(probe, q, *probe.state, 0.125, None) is not None


def require_tensor_ops():
    if not _tensor_ops_launches():
        pytest.skip("Metal TensorOps attention is unavailable")


def dense(cache, queries, mask=None, sinks=None, states=None):
    keys, values = cache.dequantize(*(states or (None, None)))
    return mx.fast.scaled_dot_product_attention(
        queries.astype(mx.float32),
        keys,
        values,
        scale=queries.shape[-1] ** -0.5,
        mask=mask,
        sinks=None if sinks is None else sinks.astype(mx.float32),
    ).astype(queries.dtype)


@pytest.mark.parametrize("dim", [1, 63, 64, 80, 256])
@pytest.mark.parametrize(
    "codec_class,max_error",
    [(Affine4Codec, 0.03), (Affine8Codec, 2e-4)],
)
def test_codec_orthogonal_and_quantization(dim, codec_class, max_error):
    codec = codec_class(dim, seed=17)
    vectors = random((2, 3, 5, dim), mx.float32)
    if codec.rotation is not None:
        rotation = np.array(codec.rotation)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(dim), atol=5e-7)
    close(codec._rotate_inverse(codec._rotate_forward(vectors)), vectors, 8e-3, 3e-3)
    state = codec.quantize(vectors)
    assert isinstance(state, TurboQuantMSEState)
    values_per_word = 32 // codec.bits
    assert state.indices.shape == (
        2,
        3,
        5,
        (dim + values_per_word - 1) // values_per_word,
    )
    assert state.indices.dtype == mx.uint32
    assert state.norms.dtype == mx.float32
    error = mx.mean(mx.square(codec.dequantize(state) - vectors)).item()
    assert error < max_error
    other = codec_class(dim, seed=17).quantize(vectors)
    assert mx.array_equal(state.indices, other.indices).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("magnitude", [0, 1e-8, 1000, 1e6])
@pytest.mark.parametrize("cache_class", [Affine4KVCache, Affine8KVCache])
def test_finite_scale_range(dtype, magnitude, cache_class):
    if dtype == mx.float16 and magnitude > 65504:
        pytest.skip("Magnitude exceeds the input dtype")
    vectors = mx.full((1, 2, 5, 64), magnitude, dtype=dtype)
    cache = cache_class(seed=4)
    cache.update_and_fetch(vectors, vectors)
    decoded, _ = cache.dequantize()
    assert mx.all(mx.isfinite(decoded)).item()
    if mx.max(mx.abs(vectors)).item():
        assert mx.max(mx.abs(decoded)).item() > magnitude / 2
    else:
        assert mx.all(decoded == 0).item()


@pytest.mark.parametrize("shape", [(1, 1, 1, 32), (2, 3, 7, 64), (1, 1, 2, 512)])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("cache_class", [Affine4KVCache, Affine8KVCache])
def test_actual_fused_quantize_matches_portable(shape, dtype, cache_class):
    if not mx.metal.is_available():
        pytest.skip("Fused quantization requires Metal")
    cache = cache_class(seed=13)
    keys, values = random(shape, dtype), random(shape, dtype, 1)
    cache._ensure_codecs(keys, values)
    ks, vs = cache._try_fused_kv_quantize(keys, values)
    assert ks is not None, "Eligible fused quantization unexpectedly fell back"
    for codec, source, state in (
        (cache.key_codec, keys, ks),
        (cache.value_codec, values, vs),
    ):
        reference = codec.quantize(source)
        close(state.norms, reference.norms, 1e-6, 2e-6)
        close(codec.dequantize(state), codec.dequantize(reference), 2e-5, 2e-5)


@pytest.mark.parametrize(
    "mask_kind,shape,dtype",
    [
        ("none", (1, 2, 3, 63, 79), mx.float16),
        ("causal", (2, 2, 4, 80, 96), mx.bfloat16),
        ("head", (2, 3, 17, 64, 64), mx.float32),
        ("broadcast", (1, 2, 3, 63, 79), mx.bfloat16),
        ("false", (2, 2, 4, 80, 96), mx.float32),
        ("infinity", (2, 3, 17, 64, 64), mx.float16),
        ("additive", (1, 2, 3, 63, 79), mx.float32),
    ],
)
def test_portable_attention_masks_sinks_and_dimensions(
    monkeypatch, mask_kind, shape, dtype
):
    monkeypatch.setattr(affine4, "_tensor_ops_available", lambda: False)
    batch, heads, rows, dk, dv = shape
    tokens = 23
    cache = Affine4KVCache()
    states = cache.update_and_fetch(
        random((batch, heads, tokens, dk), dtype),
        random((batch, heads, tokens, dv), dtype, 1),
    )
    q = random((batch, heads * 3, rows, dk), dtype, 2)
    mask = None
    if mask_kind == "causal":
        mask = "causal"
    elif mask_kind == "head":
        mask = random((batch, heads * 3, rows, tokens)) > 0
    elif mask_kind == "broadcast":
        mask = mx.arange(tokens)[None, None, None, :] >= 8
    elif mask_kind == "false":
        mask = mx.zeros((rows, tokens), mx.bool_)
    elif mask_kind == "infinity":
        mask = mx.full((rows, tokens), -float("inf"))
    elif mask_kind == "additive":
        mask = random((rows, tokens), mx.float32) * 0.3
    sinks = random((heads * 3,), dtype, 8)
    for sink in (None, sinks):
        actual = cache.attention(q, *states, scale=dk**-0.5, mask=mask, sinks=sink)
        assert actual.dtype == dtype
        close(
            actual,
            dense(cache, q, mask, sink),
            atol=8e-3 if dtype == mx.bfloat16 else 3e-3,
        )


@pytest.mark.parametrize(
    "batch,heads,repeats,rows,dim,tokens,dtype,cache_class,with_sinks",
    [
        (1, 1, 1, 1, 32, 257, mx.float16, Affine4KVCache, False),
        (1, 2, 3, 1, 64, 260, mx.bfloat16, Affine8KVCache, False),
        (2, 3, 4, 2, 96, 257, mx.float16, Affine8KVCache, False),
        (2, 2, 2, 4, 128, 513, mx.bfloat16, Affine4KVCache, False),
        (1, 2, 8, 4, 256, 1025, mx.float16, Affine4KVCache, False),
        (1, 2, 8, 4, 256, 150000, mx.bfloat16, Affine8KVCache, False),
        (1, 1, 32, 1, 256, 150000, mx.bfloat16, Affine4KVCache, False),
        (1, 1, 4, 1, 512, 259, mx.float16, Affine8KVCache, False),
        (1, 4, 8, 4, 512, 1025, mx.bfloat16, Affine4KVCache, True),
        (1, 4, 6, 8, 256, 1025, mx.bfloat16, Affine8KVCache, False),
        (1, 8, 8, 1, 64, 1025, mx.bfloat16, Affine8KVCache, True),
    ],
)
def test_actual_native_matches_portable(
    monkeypatch,
    batch,
    heads,
    repeats,
    rows,
    dim,
    tokens,
    dtype,
    cache_class,
    with_sinks,
):
    require_tensor_ops()
    cache = cache_class()
    cache.update_and_fetch(
        random((batch, heads, tokens, dim), dtype),
        random((batch, heads, tokens, dim), dtype, 1),
    )
    q = random((batch, heads * repeats, rows, dim), dtype, 2)
    mask = "causal" if rows > 1 else None
    # Large sinks take most of the softmax mass, so a dropped sink fails.
    sinks = mx.linspace(4, 9, heads * repeats) if with_sinks else None
    native = affine4._native_attention(cache, q, *cache.state, dim**-0.5, mask, sinks)
    assert native is not None, "Eligible native attention unexpectedly fell back"
    monkeypatch.setattr(affine4, "_tensor_ops_available", lambda: False)
    portable = cache.attention(q, scale=dim**-0.5, mask=mask, sinks=sinks)
    close(native, portable, atol=6e-3 if dtype == mx.bfloat16 else 2e-3)


@pytest.mark.parametrize(
    "cache_class,codec_class,packed_code,decoded_code,tokens",
    [
        (Affine4KVCache, Affine4Codec, 0x88888888, -8.0, 24576),
        (Affine4KVCache, Affine4Codec, 0x88888888, -8.0, 24832),
        (Affine8KVCache, Affine8Codec, 0x80808080, -128.0, 1536),
        (Affine8KVCache, Affine8Codec, 0x80808080, -128.0, 1537),
    ],
)
def test_native_long_uniform_attention_stays_finite(
    cache_class, codec_class, packed_code, decoded_code, tokens
):
    require_tensor_ops()
    heads, dim = 64, 32
    cache = cache_class()
    cache.key_codec = codec_class(dim, 0)
    cache.value_codec = codec_class(dim, 1)
    scales = mx.ones((1, heads, tokens))
    packed_width = dim // cache.key_codec.values_per_word
    keys = TurboQuantMSEState(
        scales, mx.zeros((1, heads, tokens, packed_width), mx.uint32)
    )
    values = TurboQuantMSEState(
        scales,
        mx.full((1, heads, tokens, packed_width), packed_code, mx.uint32),
    )
    queries = mx.zeros((1, heads, 1, dim), mx.bfloat16)
    output = affine4._native_attention(cache, queries, keys, values, dim**-0.5, None)
    assert output is not None
    assert mx.all(mx.isfinite(output)).item()
    expected = cache.value_codec._rotate_inverse(
        mx.full(queries.shape, decoded_code)
    )
    close(output, expected.astype(queries.dtype), atol=0.02)


@pytest.mark.parametrize(
    "dim,repeats,rows,mask,force_fused,cache_class",
    [
        (64, 4, 33, None, True, Affine4KVCache),
        (72, 4, 33, "causal", True, Affine8KVCache),
        (96, 4, 33, "causal", True, Affine4KVCache),
        (128, 4, 33, None, True, Affine8KVCache),
        (256, 6, 5, "causal", True, Affine4KVCache),
        (256, 6, 6, "causal", None, Affine8KVCache),
        (64, 8, 5, "causal", None, Affine4KVCache),
        (80, 4, 8, "causal", None, Affine8KVCache),
        (80, 4, 9, None, True, Affine4KVCache),
    ],
)
def test_bf16_prefill_forces_fused_sdpa_only_where_supported(
    monkeypatch, dim, repeats, rows, mask, force_fused, cache_class
):
    if not mx.metal.is_available():
        pytest.skip("Fused prefill requires Metal")
    cache = cache_class()
    cache.update_and_fetch(
        random((1, 2, 513, dim), mx.bfloat16),
        random((1, 2, 513, dim), mx.bfloat16, 1),
    )
    queries = random((1, 2 * repeats, rows, dim), mx.bfloat16, 2)
    # Keep decode-sized rows off the TensorOps kernel.
    monkeypatch.setattr(affine4, "_tensor_ops_available", lambda: False)
    original = mx.fast.scaled_dot_product_attention
    calls = []

    def capture(q, k, v, **kwargs):
        calls.append((q.dtype, k.dtype, kwargs.get("force_fused")))
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", capture)
    output = cache.attention(queries, scale=dim**-0.5, mask=mask)
    assert calls == [(mx.bfloat16, mx.bfloat16, force_fused)]
    close(output, dense(cache, queries, mask), atol=8e-3, rtol=8e-3)


@pytest.mark.parametrize("magnitude", [60000, 1e6])
@pytest.mark.parametrize("cache_class", [Affine4KVCache, Affine8KVCache])
def test_native_large_queries(magnitude, cache_class):
    require_tensor_ops()
    cache = cache_class()
    cache.update_and_fetch(random((1, 2, 257, 64)), random((1, 2, 257, 64), seed=1))
    q = mx.full((1, 8, 1, 64), magnitude, mx.bfloat16)
    output = affine4._native_attention(cache, q, *cache.state, 0.125, None)
    assert output is not None
    close(output, dense(cache, q), atol=0.02)


def test_supplied_strided_states():
    require_tensor_ops()
    cache = Affine4KVCache()
    cache.update_and_fetch(random((1, 2, 520, 64)), random((1, 2, 520, 64), seed=1))
    states = tuple(
        TurboQuantMSEState(s.norms[:, :, ::2], s.indices[:, :, ::2, :])
        for s in cache.state
    )
    q = random((1, 8, 1, 64), seed=2)
    close(cache.attention(q, *states, scale=0.125), dense(cache, q, states=states))


def test_from_cache_trim_and_batch_lifecycle():
    singles = []
    for length in (3, 7, 0):
        raw = KVCache()
        if length:
            raw.update_and_fetch(random((1, 2, length, 63)), random((1, 2, length, 79)))
        singles.append(Affine4KVCache.from_cache(raw, seed=11))
    clone = Affine4KVCache.from_cache(singles[1], seed=11)
    assert clone.trim(2) == 2
    assert singles[1].offset == 7
    batch = singles[0].merge(singles)
    assert type(batch) is BatchAffine4KVCache
    for i, single in enumerate(singles):
        extracted = batch.extract(i)
        assert type(extracted) is Affine4KVCache
        assert extracted.offset == single.offset
        if single.offset:
            close(extracted.dequantize()[0], single.dequantize()[0], 0, 0)
    batch.filter(mx.array([0, 2]))
    batch.update_and_fetch(random((2, 2, 1, 63)), random((2, 2, 1, 79)))
    assert batch.extract(0).offset == 4
    batch.extend(BatchAffine4KVCache.merge([clone]))
    assert batch.extract(2).offset == 5


@pytest.mark.parametrize("rows", [1, 3])
def test_batch_native_standard_mask_and_padding(monkeypatch, rows):
    require_tensor_ops()
    cache = BatchAffine4KVCache([37, 0])
    cache.update_and_fetch(random((2, 2, 260, 64)), random((2, 2, 260, 64), seed=1))
    mask = cache.make_mask(rows)
    assert mask is None or isinstance(mask, mx.array) or mask == "causal"
    cache.update_and_fetch(random((2, 2, rows, 64)), random((2, 2, rows, 64), seed=3))
    q = random((2, 6, rows, 64), seed=2)
    native = affine4._native_attention(cache, q, *cache.state, 0.125, mask)
    assert native is not None
    monkeypatch.setattr(affine4, "_tensor_ops_available", lambda: False)
    close(native, cache.attention(q, scale=0.125, mask=mask))
    arbitrary = mask & (mx.arange(260 + rows) % 3 != 0)
    close(cache.attention(q, scale=0.125, mask=arbitrary), dense(cache, q, arbitrary))


@pytest.mark.parametrize("cache_class", [BatchAffine4KVCache, BatchAffine8KVCache])
@pytest.mark.parametrize("rows", [1, 3])
def test_left_padded_batch_verify_uses_affine_attention(monkeypatch, cache_class, rows):
    """Batched decode and MTP verify must not dequantize the whole history."""
    pads = [0, 250, 590]
    cache = cache_class(pads)
    cache.update_and_fetch(random((3, 2, 600, 64)), random((3, 2, 600, 64), seed=1))
    keys, values = cache.update_and_fetch(
        random((3, 2, rows, 64), seed=3), random((3, 2, rows, 64), seed=4)
    )
    q = random((3, 6, rows, 64), seed=2)
    mask = create_causal_mask(rows, offset=600, left_padding=cache.left_padding)
    expected = dense(cache, q, mask)

    def no_dequantize(*args, **kwargs):
        raise AssertionError("affine verify dequantized the full history")

    monkeypatch.setattr(cache, "dequantize", no_dequantize)
    out = turboquant_attention._left_padded_batch_attention(
        cache, q, keys, values, 0.125, mask
    )
    assert out is not None
    close(out, expected, 4e-3, 4e-3)


def test_first_native_compile_failure_uses_portable_attention(monkeypatch):
    """Systems without TensorOps fail the first launch, not the request."""
    if not affine4._tensor_ops_available():
        pytest.skip("Native lazy failures require Metal")
    cache = Affine4KVCache()
    keys = random((1, 2, 257, 64))
    cache.update_and_fetch(keys, random((1, 2, 257, 64), seed=1))
    monkeypatch.setattr(affine4, "_NATIVE_LAUNCHABLE", {})
    kernel = mx.fast.metal_kernel(
        name="affine4_bad_attention",
        input_names=["x"],
        output_names=["y"],
        source="undefined_symbol;",
    )

    def launch(**kwargs):
        return tuple(
            kernel(
                inputs=[keys],
                output_shapes=[shape],
                output_dtypes=[mx.float32],
                grid=(1, 1, 1),
                threadgroup=(1, 1, 1),
            )[0]
            for shape in kwargs["output_shapes"]
        )

    monkeypatch.setattr(affine4, "_mpp_attention_kernel", lambda *args: launch)
    q = random((1, 8, 1, 64), seed=2)
    close(cache.attention(q, scale=0.125), dense(cache, q))
    assert False in affine4._NATIVE_LAUNCHABLE.values()


def test_first_native_command_buffer_error_propagates(monkeypatch):
    """A GPU fault during the launch check is not a missing kernel."""
    if not affine4._tensor_ops_available():
        pytest.skip("Native lazy failures require Metal")
    cache = Affine4KVCache()
    keys = random((1, 2, 257, 64))
    cache.update_and_fetch(keys, keys)
    q = random((1, 8, 1, 64), seed=2)
    mx.eval(cache.state, q)
    monkeypatch.setattr(affine4, "_NATIVE_LAUNCHABLE", {})
    real_eval = mx.eval
    calls = []

    def failing_once(*args):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("[METAL] Command buffer execution failed: test")
        return real_eval(*args)

    monkeypatch.setattr(affine4.mx, "eval", failing_once)
    with pytest.raises(RuntimeError, match="Command buffer execution failed"):
        cache.attention(q, scale=0.125)
    assert not affine4._NATIVE_LAUNCHABLE


@pytest.mark.parametrize(
    "dim,shape,dtype,codec_class",
    [
        (1, (), mx.float32, Affine4Codec),
        (7, (3,), mx.bfloat16, Affine8Codec),
        (63, (2, 3, 9), mx.float32, Affine8Codec),
        (80, (), mx.bfloat16, Affine4Codec),
        (128, (3,), mx.float32, Affine4Codec),
        (257, (2, 3, 9), mx.bfloat16, Affine8Codec),
    ],
)
def test_fused_rotated_dequantize_arbitrary_dimensions(
    dim, shape, dtype, codec_class
):
    if not mx.metal.is_available():
        pytest.skip("Fused unpack requires Metal")
    codec = codec_class(dim)
    state = codec.quantize(random((*shape, dim), mx.float32))
    reference = codec._codes(state) * state.norms[..., None]
    close(codec.dequantize_rotated(state, dtype), reference.astype(dtype), 0, 0)
    signature = (
        (dim, max(1, len(shape)), mx.float32, dtype)
        if codec.bits == 4
        else (dim, codec.bits, max(1, len(shape)), mx.float32, dtype)
    )
    assert affine4._DEQUANTIZE_LAUNCHABLE[signature] is True


def test_fused_rotated_dequantize_strides():
    codec = Affine4Codec(79)
    state = codec.quantize(random((2, 3, 19, 79)))
    indices = mx.stack((state.indices, mx.zeros_like(state.indices)), axis=-1)
    indices = indices.reshape(2, 3, 19, -1)[..., ::2]
    state = TurboQuantMSEState(state.norms[:, ::2, ::3], indices[:, ::2, ::3])
    close(
        codec.dequantize_rotated(state),
        codec._codes(state) * state.norms[..., None],
        0,
        0,
    )


def test_cpu_stream_uses_portable_operations(monkeypatch):
    def no_metal(*args, **kwargs):
        pytest.fail("CPU execution invoked a Metal kernel")

    monkeypatch.setattr(mx.fast, "metal_kernel", no_metal)
    monkeypatch.setattr(affine4, "_fused_quantize_kernel", no_metal)
    monkeypatch.setattr(affine4, "_dequantize_rotated_kernel", no_metal)
    monkeypatch.setattr(affine4, "_mpp_attention_kernel", no_metal)
    with mx.stream(mx.cpu):
        cache = Affine4KVCache()
        cache.update_and_fetch(random((1, 2, 257, 64)), random((1, 2, 257, 79)))
        queries = random((1, 6, 1, 64))
        close(cache.attention(queries, scale=0.125), dense(cache, queries))
        assert mx.all(mx.isfinite(cache.dequantize()[1])).item()


@pytest.mark.parametrize(
    "mask_kind,batched,with_sinks",
    [
        ("none", False, True),
        ("causal", True, False),
        ("head", False, False),
        ("broadcast", True, True),
        ("tokens", False, True),
        ("false", True, False),
        ("infinity", False, False),
        ("additive", True, True),
    ],
)
def test_bounded_prefill_preserves_masks_and_sinks(
    monkeypatch, mask_kind, batched, with_sinks
):
    cache = BatchAffine4KVCache([3, 0]) if batched else Affine4KVCache()
    cache.update_and_fetch(random((2, 2, 23, 63)), random((2, 2, 23, 79)))
    q = random((2, 6, 7, 63), seed=2)
    masks = {
        "none": None,
        "causal": "causal",
        "head": random((2, 6, 7, 23), seed=3) > 0,
        "broadcast": mx.arange(23)[None, None, None, :] >= 8,
        "tokens": mx.arange(23) >= 8,
        "false": mx.zeros((7, 23), mx.bool_),
        "infinity": mx.full((7, 23), -float("inf")),
        "additive": random((2, 6, 7, 23), mx.float32, seed=4),
    }
    mask = masks[mask_kind]
    sinks = random((6,), mx.float32, seed=5) if with_sinks else None
    expected = cache.attention(q, scale=63**-0.5, mask=mask, sinks=sinks)
    mx.eval(expected)
    monkeypatch.setattr(affine4, "_MAX_SCORE_ELEMENTS", 2 * 6 * 23 * 3)
    original = mx.fast.scaled_dot_product_attention
    query_lengths = []

    def record(queries, *args, **kwargs):
        query_lengths.append(queries.shape[-2])
        return original(queries, *args, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", record)
    close(cache.attention(q, scale=63**-0.5, mask=mask, sinks=sinks), expected, 3e-3)
    assert query_lengths == [3, 3, 1]


@pytest.mark.parametrize("cache_class", [Affine4KVCache, Affine8KVCache])
@pytest.mark.parametrize("rows", [1, 16])
def test_portable_attention_peak_is_bounded_across_layers(
    monkeypatch, cache_class, rows
):
    if not mx.metal.is_available():
        pytest.skip("Peak allocation tracking requires Metal")
    monkeypatch.setattr(affine4, "_tensor_ops_available", lambda: False)
    source = cache_class()
    source.update_and_fetch(
        random((1, 4, 8192, 128), mx.bfloat16),
        random((1, 4, 8192, 128), mx.bfloat16, seed=1),
    )
    mx.eval(source.state)
    caches = [cache_class.from_state(source.state, source.meta_state) for _ in range(16)]
    output = random((1, 16, rows, 128), mx.bfloat16, seed=2)
    mx.eval(output, *[part for cache in caches for state in cache.state for part in state])
    mx.synchronize()
    mx.clear_cache()
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()
    for cache in caches:
        output = cache.attention(
            output,
            scale=128**-0.5,
            mask="causal" if rows > 1 else None,
        )
    mx.eval(output)
    temporary = mx.get_peak_memory() - baseline
    assert temporary < 80 * 1024**2, (
        f"{cache_class.__name__} allocated {temporary / 1024**2:.1f} MiB "
        f"for {rows}-row attention"
    )
    assert mx.all(mx.isfinite(output)).item()


def test_bounded_prefill_peak_memory():
    if not mx.metal.is_available():
        pytest.skip("Peak allocation tracking requires Metal")
    cache = Affine4KVCache()
    cache.update_and_fetch(
        random((1, 4, 32768, 256)), random((1, 4, 32768, 256), seed=1)
    )
    queries = random((1, 24, 256, 256), seed=2)
    mx.eval(queries, cache.state)
    mx.synchronize()
    mx.clear_cache()
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()
    output = cache.attention(queries, scale=256**-0.5, mask="causal")
    mx.eval(output)
    temporary = mx.get_peak_memory() - baseline
    assert temporary < 600 * 1024**2, f"Prefill allocated {temporary / 1024**2:.1f} MiB"
    assert mx.all(mx.isfinite(output)).item()
