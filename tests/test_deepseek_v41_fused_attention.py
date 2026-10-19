"""The fused packed attention is bitwise identical to the three-kernel _mma_attention."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import fused_attention, packed_attention
from omlx.patches.deepseek_v41.quantization import pack_activation

pytestmark = pytest.mark.skipif(
    mx.default_device() != mx.gpu, reason="fused attention kernels are Metal-only"
)


def _inputs(seed, length, heads, dim, window_rows, pooled_rows, w, c):
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return mx.array(rng.normal(0, 1, shape).astype(np.float32))

    q = normal(1, length, heads, dim).astype(mx.bfloat16)
    window = pack_activation(normal(1, window_rows, dim))
    if pooled_rows:
        pooled = pack_activation(normal(1, pooled_rows, dim), 4, 16, True)
    else:
        pooled = mx.zeros((1, 0, dim // 2 + dim // 16), mx.uint8)
    # -1 and rows past the end are invalid slots; both must score -inf.
    wi = mx.array(rng.integers(-1, window_rows + 2, (1, length, w)), mx.int32)
    ci = mx.array(rng.integers(-1, pooled_rows + 2, (1, length, c)), mx.int32)
    sink = normal(heads)
    return q, window, pooled, wi, ci, sink, dim**-0.5


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize(
    "length,heads,dim,window_rows,pooled_rows,w,c",
    [
        (1, 64, 512, 128, 1024, 128, 512),  # decode: only the selected rows are decoded
        (2, 8, 128, 128, 900, 128, 449),  # gathered rows, last 64-key block partial
        (9, 16, 128, 128, 300, 128, 448),  # prefill: every cached row decoded once
        (3, 8, 32, 64, 600, 64, 513),  # D = 32: fewer dimension slices
        (4, 8, 64, 600, 0, 520, 0),  # window only, no compressed pool
    ],
)
def test_fused_attention_matches_three_kernel_reference(
    dtype, length, heads, dim, window_rows, pooled_rows, w, c
):
    """Same arithmetic in the same order as _mma_attention (see fused_attention.py). The
    float32 case keeps the bf16 query values but stores unrounded outputs, so a change of
    fp32 summation order (which bf16 outputs mostly hide) shows up as well."""
    q, *rest = _inputs(5 + length, length, heads, dim, window_rows, pooled_rows, w, c)
    args = (q.astype(dtype), *rest)
    gathered = length * (w + c) < window_rows + pooled_rows
    fused = fused_attention.fused_mma_attention(*args)
    reference = packed_attention._mma_attention(*args)
    mx.eval(fused, reference)
    assert fused.dtype == reference.dtype == dtype
    assert mx.array_equal(fused, reference).item(), f"gathered={gathered}"


def test_rounded_attention_routes_large_candidate_lists_to_the_fused_kernel(monkeypatch):
    args = _inputs(3, 2, 16, 128, 128, 700, 128, 512)
    calls = []
    real = fused_attention.fused_mma_attention

    def tracked(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(fused_attention, "fused_mma_attention", tracked)
    out = packed_attention.rounded_packed_attention(*args)
    assert calls
    monkeypatch.setattr(fused_attention, "supported", lambda q: False)   # shapes it does not take
    assert mx.array_equal(out, packed_attention.rounded_packed_attention(*args)).item()
    assert len(calls) == 1
