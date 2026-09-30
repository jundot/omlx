# SPDX-License-Identifier: Apache-2.0
"""Compiled decode quantization preserves the codec's packed cache state."""

import gc
import weakref

import mlx.core as mx
import numpy as np
import pytest
from mlx_vlm import turboquant as tq

from omlx.patches import turboquant_quantize as patch

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture
def original(monkeypatch):
    cls = tq._TurboQuantMSECodec
    call = cls.quantize
    if getattr(call, "_omlx_compiled_mse_quantize", False):
        call = call.__wrapped__
    monkeypatch.setattr(cls, "quantize", call)
    monkeypatch.setattr(patch, "_ENABLED", True)
    monkeypatch.setattr(patch, "_DISABLED", set())
    assert patch.apply_turboquant_quantize_patch()
    return call


def _same(actual, expected):
    mx.eval(actual.norms, actual.indices, expected.norms, expected.indices)
    assert mx.array_equal(
        actual.norms.view(mx.uint16), expected.norms.view(mx.uint16)
    ).item()
    assert mx.array_equal(actual.indices, expected.indices).item()


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_compiled_cache_matches_original_codec(original, dim, bits, dtype):
    for seed in (0, 1, 17):
        codec = tq._TurboQuantMSECodec(dim, bits, seed)
        for step in range(10):
            mx.random.seed(seed + step)
            vectors = (mx.random.normal((2, 4, 1, dim)) * 10 ** (step % 8 - 4)).astype(
                dtype
            )
            _same(codec.quantize(vectors), original(codec, vectors))
    assert not patch._DISABLED


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_compiled_codec_handles_zero_extreme_and_strided_vectors(original, dtype):
    codec = tq._TurboQuantMSECodec(256, 3, 42)
    mx.random.seed(33)
    values = mx.random.normal((1, 4, 1, 512)).astype(dtype)
    for vectors in (
        mx.zeros((1, 4, 1, 256), dtype),
        mx.ones((1, 4, 1, 256), dtype) * 1e-10,
        mx.ones((1, 4, 1, 256), dtype) * 1e-30,
        mx.ones((1, 4, 1, 256), dtype) * 1e20,
        values[..., ::2],
        values[..., 1::2],
    ):
        _same(codec.quantize(vectors), original(codec, vectors))


def test_signs_and_midpoints_are_live_inputs(original):
    mx.random.seed(1)
    vectors = mx.random.normal((1, 4, 1, 256)).astype(mx.float16)
    codec = tq._TurboQuantMSECodec(256, 3, 0)
    first = codec.quantize(vectors)
    codec.signs = -codec.signs
    codec._midpoints = codec._midpoints + 0.03
    changed = codec.quantize(vectors)
    _same(changed, original(codec, vectors))
    assert not mx.array_equal(first.indices, changed.indices).item()


@pytest.mark.parametrize(
    "dim,bits,shape",
    [
        (96, 3, (1, 2, 1, 96)),
        (32, 3, (1, 2, 1, 32)),
        (64, 0, (1, 2, 1, 64)),
        (64, 5, (1, 2, 1, 64)),
        (64, 3, (1, 2, 2, 64)),
        (64, 3, (2, 1, 64)),
    ],
)
def test_unsupported_layouts_keep_original_codec(
    original, monkeypatch, dim, bits, shape
):
    monkeypatch.setattr(
        patch,
        "_compiled_quantizer",
        lambda *args: pytest.fail("Unexpected compilation"),
    )
    codec = tq._TurboQuantMSECodec(dim, bits, 0)
    vectors = mx.random.normal(shape).astype(mx.float16)
    _same(codec.quantize(vectors), original(codec, vectors))


def test_instance_rotation_override_keeps_original_codec(original, monkeypatch):
    monkeypatch.setattr(
        patch,
        "_compiled_quantizer",
        lambda *args: pytest.fail("Unexpected compilation"),
    )
    codec = tq._TurboQuantMSECodec(64, 3, 0)
    codec._rotate_forward = lambda x: x
    vectors = mx.random.normal((1, 2, 1, 64)).astype(mx.float16)
    _same(codec.quantize(vectors), original(codec, vectors))


def test_compile_failure_declines_once_without_changing_state(original, monkeypatch):
    calls = []

    def fail(*args):
        calls.append(1)
        raise RuntimeError("Cannot compile this layout")

    monkeypatch.setattr(patch, "_compiled_quantizer", fail)
    codec = tq._TurboQuantMSECodec(64, 3, 0)
    vectors = mx.random.normal((1, 2, 1, 64)).astype(mx.float16)
    for _ in range(3):
        _same(codec.quantize(vectors), original(codec, vectors))
    assert calls == [1]


def test_compiled_graph_does_not_retain_codec(original):
    codec = tq._TurboQuantMSECodec(64, 3, 0)
    ref = weakref.ref(codec)
    state = codec.quantize(mx.ones((1, 2, 1, 64), mx.float16))
    mx.eval(state.norms, state.indices)
    del codec
    gc.collect()
    assert ref() is None


def test_patch_is_idempotent_and_can_be_disabled(original, monkeypatch):
    call = tq._TurboQuantMSECodec.quantize
    assert patch.apply_turboquant_quantize_patch()
    assert tq._TurboQuantMSECodec.quantize is call
    monkeypatch.setattr(patch, "_ENABLED", False)
    monkeypatch.setattr(
        patch,
        "_compiled_quantizer",
        lambda *args: pytest.fail("Unexpected compilation"),
    )
    codec = tq._TurboQuantMSECodec(64, 3, 0)
    vectors = mx.random.normal((1, 2, 1, 64)).astype(mx.float16)
    _same(codec.quantize(vectors), original(codec, vectors))


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_packed_indices_match_near_quantization_thresholds(original, dim, bits, dtype):
    for seed in (0, 1, 17):
        codec = tq._TurboQuantMSECodec(dim, bits, seed)
        rows = []
        for midpoint in codec._midpoints.tolist():
            for offset in (-3, -2, -1, 0, 1, 2, 3):
                value = np.float32(midpoint)
                for _ in range(abs(offset)):
                    value = np.nextafter(
                        value, np.float32(np.inf if offset > 0 else -np.inf)
                    )
                for scale in (1e-3, 0.125, 1, 8, 1e3):
                    row = np.zeros(dim, np.float32)
                    row[0] = value
                    row[1] = np.sqrt(np.float32(1 - value * value))
                    rows.append(row * scale)
        # Construct unit directions near each threshold in rotated coordinates,
        # then pass the inverse rotation through the actual codec.
        vectors = (
            codec._rotate_inverse(mx.array(np.asarray(rows)))
            .reshape(1, len(rows), 1, dim)
            .astype(dtype)
        )
        _same(codec.quantize(vectors), original(codec, vectors))
    assert not patch._DISABLED
