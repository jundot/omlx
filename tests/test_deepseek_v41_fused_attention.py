"""The fused packed attention is bitwise identical to the three-kernel matrix path it replaced."""

from functools import cache

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import fused_attention, packed_attention
from omlx.patches.deepseek_v41.quantization import pack_activation

pytestmark = pytest.mark.skipif(
    mx.default_device() != mx.gpu, reason="fused attention kernels are Metal-only"
)


# The three-kernel matrix path fused_attention replaced (scores, PV and merge as separate
# dispatches, formerly packed_attention._mma_attention): the fused kernel's bitwise oracle.
_MMA_SCORES = r"""
    const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.z, query = threadgroup_position_in_grid.y;
    const uint first_head = threadgroup_position_in_grid.x * 8;
    const int H = meta[0], W = meta[1], C = meta[2], NW = meta[3], NC = meta[4], NS = meta[5];
    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);
    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);
    threadgroup float tile[8 * D], dots[4 * 64], maxima_local[8];
    threadgroup bool valid[8];
    if (tid < 8) maxima_local[tid] = -1e30f;
    for (int begin = split * 16; begin < int(split + 1) * 16; begin += 8) {
        for (uint i = tid; i < 8 * D; i += 128) {
            const uint r = i / D, d = i % D, j = begin + r;
            const bool compressed = j >= W;
            const int row = j >= W + C ? -1 : compressed ? ci[query * C + j - W] : wi[query * W + j];
            const bool ok = row >= 0 && row < (compressed ? NC : NW);
            tile[i] = ok ? read_kv(window, pooled, row, d, D, compressed, 0.0f, 0, false) : 0.0f;
            if (d == 0) valid[r] = ok;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> score, a, b;
        score.thread_elements()[0] = 0.0f;
        score.thread_elements()[1] = 0.0f;
        for (int k = sg * (D / 4); k < (sg + 1) * (D / 4); k += 8) {
            const uint head = first_head + fm;
            a.thread_elements()[0] = head < H ? float(q[(query * H + head) * D + k + fn]) : 0.0f;
            a.thread_elements()[1] = head < H ? float(q[(query * H + head) * D + k + fn + 1]) : 0.0f;
            b.thread_elements()[0] = tile[fn * D + k + fm];
            b.thread_elements()[1] = tile[(fn + 1) * D + k + fm];
            simdgroup_multiply_accumulate(score, a, b, score);
        }
        dots[sg * 64 + fm * 8 + fn] = score.thread_elements()[0];
        dots[sg * 64 + fm * 8 + fn + 1] = score.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8 && first_head + tid < H) {
            float maximum = maxima_local[tid];
            for (uint r = 0; r < 8; ++r) {
                float sum = dots[tid * 8 + r] + dots[64 + tid * 8 + r] + dots[128 + tid * 8 + r] + dots[192 + tid * 8 + r];
                const float value = valid[r] ? sum * scalep[0] : -INFINITY;
                maximum = max(maximum, value);
                scores[(size_t(query) * H + first_head + tid) * NS * 16 + begin + r] = value;
            }
            maxima_local[tid] = maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 8 && first_head + tid < H)
        maxima[(size_t(query) * H + first_head + tid) * NS + split] = maxima_local[tid];
"""

_MMA_VALUES = r"""
    const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint block = threadgroup_position_in_grid.z, query = threadgroup_position_in_grid.y;
    const uint first_head = threadgroup_position_in_grid.x * 8;
    const int H = meta[0], W = meta[1], C = meta[2], NW = meta[3], NC = meta[4], NS = meta[5], NB = meta[6];
    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);
    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);
    threadgroup float tile[8 * D], probability[64], maxima[8], denominator[8];
    threadgroup bool valid[8];
    simdgroup_matrix<float, 8, 8> output[D / 32];
    for (int v = 0; v < D / 32; ++v) {
        output[v].thread_elements()[0] = 0.0f;
        output[v].thread_elements()[1] = 0.0f;
    }
    if (tid < 8) {
        float maximum = -1e30f;
        if (first_head + tid < H)
            for (uint b = 0; b <= min(block * 4 + 3, uint(NS - 1)); ++b)
                maximum = max(maximum, score_maxima[(size_t(query) * H + first_head + tid) * NS + b]);
        maxima[tid] = maximum;
        denominator[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int begin = int(block) * 64; begin < int(block + 1) * 64; begin += 8) {
        for (uint i = tid; i < 8 * D; i += 128) {
            const uint r = i / D, d = i % D, j = begin + r;
            const bool compressed = j >= W;
            const int row = j >= W + C ? -1 : compressed ? ci[query * C + j - W] : wi[query * W + j];
            const bool ok = row >= 0 && row < (compressed ? NC : NW);
            tile[i] = ok ? read_kv(window, pooled, row, d, D, compressed, 0.0f, 0, false) : 0.0f;
            if (d == 0) valid[r] = ok;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8) {
            const uint head = first_head + tid;
            float total = denominator[tid];
            for (uint r = 0; r < 8; ++r) {
                const int slot = begin + r;
                const float p = head < H && valid[r] ? exp(scores[((size_t(query) * H + head) * NS * 16) + slot] - maxima[tid]) : 0.0f;
                probability[tid * 8 + r] = float(bfloat16_t(p));
                total += p;
            }
            denominator[tid] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> a, b;
        a.thread_elements()[0] = probability[fm * 8 + fn];
        a.thread_elements()[1] = probability[fm * 8 + fn + 1];
        for (int v = 0; v < D / 32; ++v) {
            const uint d = sg * (D / 4) + v * 8 + fn;
            b.thread_elements()[0] = tile[fm * D + d];
            b.thread_elements()[1] = tile[fm * D + d + 1];
            simdgroup_multiply_accumulate(output[v], a, b, output[v]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint head = first_head + fm;
    if (head < H) {
        const size_t base = ((size_t(query) * H + head) * NB + block) * (D + 2);
        for (int v = 0; v < D / 32; ++v) {
            const uint d = sg * (D / 4) + v * 8 + fn;
            partial[base + d] = output[v].thread_elements()[0];
            partial[base + d + 1] = output[v].thread_elements()[1];
        }
    }
    if (tid < 8 && first_head + tid < H) {
        const size_t base = ((size_t(query) * H + first_head + tid) * NB + block) * (D + 2);
        partial[base + D] = maxima[tid];
        partial[base + D + 1] = denominator[tid];
    }
"""


@cache
def _oracle_kernel(stage):
    inputs, outputs, source = {
        "mma_scores": (
            ["q", "window", "pooled", "wi", "ci", "meta", "scalep"],
            ["scores", "maxima"],
            _MMA_SCORES,
        ),
        "mma_values": (
            ["scores", "score_maxima", "window", "pooled", "wi", "ci", "meta"],
            ["partial"],
            _MMA_VALUES,
        ),
    }[stage]
    return mx.fast.metal_kernel(
        name="test_deepseek_v41_online_" + stage,
        input_names=inputs,
        output_names=outputs,
        source=source,
        header=packed_attention._READ + "\n#include <metal_simdgroup_matrix>\n",
    )


def three_kernel_attention(q, window, pooled, wi, ci, sink, scale):
    _, length, heads, dim = q.shape
    count = wi.shape[-1] + ci.shape[-1]
    score_blocks = max(1, (count + 15) // 16)
    value_blocks = max(1, (count + 63) // 64)
    meta = mx.array(
        [
            heads,
            wi.shape[-1],
            ci.shape[-1],
            window.shape[1],
            pooled.shape[1],
            score_blocks,
            value_blocks,
        ],
        mx.int32,
    )
    window = window if window.size else mx.zeros((1,), mx.uint8)
    pooled = pooled if pooled.size else mx.zeros((1,), mx.uint8)
    wi = wi.astype(mx.int32) if wi.size else mx.zeros((1,), mx.int32)
    ci = ci.astype(mx.int32) if ci.size else mx.zeros((1,), mx.int32)
    scores, maxima = _oracle_kernel("mma_scores")(
        inputs=[q, window, pooled, wi, ci, meta, mx.array([scale])],
        template=[("D", dim)],
        grid=((heads + 7) // 8 * 128, length, score_blocks),
        threadgroup=(128, 1, 1),
        output_shapes=[
            (1, length, heads, score_blocks, 16),
            (1, length, heads, score_blocks),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    partial = _oracle_kernel("mma_values")(
        inputs=[scores, maxima, window, pooled, wi, ci, meta],
        template=[("D", dim)],
        grid=((heads + 7) // 8 * 128, length, value_blocks),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, length, heads, value_blocks, dim + 2)],
        output_dtypes=[mx.float32],
    )[0]
    return packed_attention._kernel("merge")(
        inputs=[partial, sink, mx.array([heads, value_blocks], mx.int32)],
        template=[("D", dim), ("T", q.dtype), ("CHUNK", 64)],
        grid=((heads * (dim // 32) + 3) // 4 * 128, length, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


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
    """Same arithmetic in the same order as the three-kernel path (see fused_attention.py). The
    float32 case keeps the bf16 query values but stores unrounded outputs, so a change of
    fp32 summation order (which bf16 outputs mostly hide) shows up as well."""
    q, *rest = _inputs(5 + length, length, heads, dim, window_rows, pooled_rows, w, c)
    args = (q.astype(dtype), *rest)
    gathered = length * (w + c) < window_rows + pooled_rows
    fused = fused_attention.fused_mma_attention(*args)
    reference = three_kernel_attention(*args)
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
    assert len(calls) == 1
    assert mx.array_equal(out, three_kernel_attention(*args)).item()
