# SPDX-License-Identifier: Apache-2.0
"""The portable decode-attention path must preserve causal visibility."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.custom_kernels.decode_fast import fast


@pytest.mark.parametrize("fallback", ["missing", "unsupported", "forced"])
@pytest.mark.parametrize("query_length", [4, 16])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("device", [mx.cpu, mx.gpu])
def test_fallback_preserves_causal_visibility(
    monkeypatch, fallback, query_length, dtype, device
):
    if device == mx.gpu and not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    def rejected(*args):
        assert fallback == "unsupported", "forced fallback must bypass the extension"
        return False

    extension = (
        None
        if fallback == "missing"
        else SimpleNamespace(sdpa_decode_supported=rejected)
    )
    monkeypatch.setattr(fast, "_ext", extension)

    key_length = 19
    with mx.stream(device):
        q = mx.zeros((1, 2, query_length, 32), dtype=dtype)
        k = mx.zeros((1, 1, key_length, 32), dtype=dtype)
        values = mx.arange(key_length).astype(dtype)
        v = mx.broadcast_to(values[None, None, :, None], (1, 1, key_length, 32))
        out = fast.sdpa_decode(
            q, k, v, 32**-0.5, causal=True, force_fallback=fallback == "forced"
        )
        # Zero logits give uniform attention over the visible prefix. Queries
        # align with the last query_length keys, so each mean is last_key / 2.
        expected = mx.arange(key_length - query_length, key_length) / 2
        expected = mx.broadcast_to(expected[None, None, :, None], q.shape)
        mx.eval(out, expected)
        assert mx.allclose(out, expected, atol=0.05, rtol=0).item()


def test_fallback_keeps_explicit_noncausal_mask(monkeypatch):
    monkeypatch.setattr(fast, "_ext", None)
    q = mx.zeros((1, 1, 4, 32))
    k = mx.zeros((1, 1, 4, 32))
    v = mx.broadcast_to(mx.arange(4)[None, None, :, None], (1, 1, 4, 32)).astype(
        mx.float32
    )
    mask = mx.array([False, False, False, True])
    out = fast.sdpa_decode(q, k, v, 32**-0.5, mask=mask)
    mx.eval(out)
    assert mx.allclose(out, mx.full(q.shape, 3.0)).item()
