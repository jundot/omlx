# SPDX-License-Identifier: Apache-2.0
"""Read selected Hy4 latent KV slots in place during short draft verification.

The GEMV and softmax reductions follow MLX 0.32.4 (Copyright Apple Inc.,
Apache-2.0), including the intermediate storage dtype.
"""

import logging
from functools import cache

import mlx.core as mx

logger = logging.getLogger(__name__)


_QK = r"""
    const uint rq = threadgroup_position_in_grid.z;
    const uint h0 = threadgroup_position_in_grid.y * SHARE;
    const uint out_slot = threadgroup_position_in_grid.x * 16
        + simdgroup_index_in_threadgroup * 4;
    const uint lane = thread_index_in_simdgroup;
    const int bank = kv_shape[2];
    const uint sub = lane * 4;
    float result[SHARE][4] = {{0}};
    for (uint k = 0; k < 512; k += 128) {
        float q[SHARE][4];
        for (uint h = 0; h < SHARE; ++h)
            for (uint i = 0; i < 4; ++i)
                q[h][i] = query[(rq * HEADS + h0 + h) * 512 + k + sub + i];
        for (uint j = 0; j < 4; ++j) {
            const uint logical = out_slot + j;
            const int address = indices[rq * WIDTH + logical];
            const uint physical = clamp(address < 0 ? address + bank : address, 0, bank - 1);
            const size_t slot = physical;
            for (uint i = 0; i < 4; ++i) {
                const float val = kv[slot * 512 + k + sub + i];
                for (uint h = 0; h < SHARE; ++h)
                    result[h][j] += val * q[h][i];
            }
        }
    }
    for (uint h = 0; h < SHARE; ++h) {
        for (uint j = 0; j < 4; ++j) {
            float v = result[h][j];
            for (ushort offset = 16; offset > 0; offset >>= 1)
                v += simd_shuffle_down(v, offset);
            if (lane == 0)
                out[(rq * HEADS + h0 + h) * WIDTH + out_slot + j] = T(v);
        }
    }
"""


_PV = r"""
    const uint rq = threadgroup_position_in_grid.z;
    const uint h0 = threadgroup_position_in_grid.y * SHARE;
    const uint lane = thread_index_in_simdgroup;
    const uint col = threadgroup_position_in_grid.x * 64
        + simdgroup_index_in_threadgroup * 16 + (lane & 3) * 4;
    const uint row_offset = (lane >> 2) * 4;
    const int bank = kv_shape[2];
    float result[SHARE][4] = {{0}};
    for (uint k = 0; k < WIDTH; k += 32) {
        for (uint i = 0; i < 4; ++i) {
            const uint logical = k + row_offset + i;
            const int address = indices[rq * WIDTH + logical];
            const uint physical = clamp(address < 0 ? address + bank : address, 0, bank - 1);
            const size_t slot = physical;
            float value[4];
            for (uint j = 0; j < 4; ++j)
                value[j] = kv[slot * 512 + col + j];
            for (uint h = 0; h < SHARE; ++h) {
                const float prob = query[(rq * HEADS + h0 + h) * WIDTH + logical];
                for (uint j = 0; j < 4; ++j)
                    result[h][j] += prob * value[j];
            }
        }
    }
    for (uint h = 0; h < SHARE; ++h) {
        for (uint j = 0; j < 4; ++j) {
            float sum = result[h][j];
            for (ushort offset = 16; offset >= 4; offset >>= 1)
                sum += simd_shuffle_down(sum, offset);
            if (lane < 4) out[(rq * HEADS + h0 + h) * 512 + col + j] = T(sum);
        }
    }
"""


@cache
def _pipeline(stage):
    logger.debug("Hy4 short verify: shared sparse KV (%s)", stage)
    return mx.fast.metal_kernel(
        name=f"hy4_gemv_shared_{stage}",
        input_names=["query", "kv", "indices"],
        output_names=["out"],
        source=_QK if stage == "qk" else _PV,
    )


def _shared_gemv(stage, q, kv, indices):
    width = indices.shape[-1]
    share = 4
    rows, heads = q.shape[0], q.shape[-3]
    template = [("T", q.dtype), ("WIDTH", width), ("HEADS", heads), ("SHARE", share)]
    channel = width if stage == "qk" else 512
    blocks = width // 16 if stage == "qk" else 512 // 64
    return _pipeline(stage)(
        inputs=[q, kv, indices],
        template=template,
        grid=(128 * blocks, heads // share, rows),
        threadgroup=(128, 1, 1),
        output_shapes=[(*q.shape[:-1], channel)],
        output_dtypes=[q.dtype],
    )[0]


_SOFTMAX = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint qh = threadgroup_position_in_grid.x;
    const uint batch = qh / HEADS;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    threadgroup float maxes[32];
    threadgroup float sums[32];
    float chunk[4];
    const uint offset = qh * WIDTH;
    for (uint j = 0; j < 4; ++j) {
      const uint col = tid * 4 + j;
      float value = -INFINITY;
      if (col == 0) {
        value = float(sinks[qh % HEADS]);
      } else if (col <= WIDTH) {
        const uint i = col - 1;
        const T mask_value = keep[batch * WIDTH + i] ? rope[offset + i]
                                                    : T(mask_min[0]);
        value = float(T(dot[offset + i] + mask_value));
      }
      chunk[j] = value;
    }
    if (simd == 0) {
      maxes[lane] = -INFINITY;
      sums[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float hi = -FLT_MAX;
    for (uint i = 0; i < 4; ++i) hi = hi < chunk[i] ? chunk[i] : hi;
    hi = simd_max(hi);
    if (lane == 0) maxes[simd] = hi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
      hi = simd_max(maxes[lane]);
      if (lane == 0) maxes[0] = hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accum = 0.0f;
    hi = maxes[0];
    for (uint i = 0; i < 4; ++i) {
      chunk[i] = fast::exp(chunk[i] - hi);
      accum += chunk[i];
    }
    accum = simd_sum(accum);
    if (lane == 0) sums[simd] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
      accum = simd_sum(sums[lane]);
      if (lane == 0) sums[0] = accum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inverse = 1.0f / sums[0];
    for (uint j = 0; j < 4; ++j) {
      const uint col = tid * 4 + j;
      if (col && col <= WIDTH) out[offset + col - 1] = T(chunk[j] * inverse);
    }
"""


@cache
def _softmax_kernel():
    return mx.fast.metal_kernel(
        name="hy4_add_sink_precise_softmax",
        input_names=["dot", "rope", "keep", "sinks", "mask_min"],
        output_names=["out"],
        source=_SOFTMAX,
    )


def fused_probs(dot, rope, keep, sinks):
    width = dot.shape[-1]
    heads = dot.shape[2]
    qh = dot.shape[0] * heads
    threads = ((width + 128) // 128) * 32
    return _softmax_kernel()(
        inputs=[
            dot,
            rope,
            keep,
            sinks,
            mx.array([mx.finfo(dot.dtype).min], mx.float32),
        ],
        template=[("T", dot.dtype), ("WIDTH", width), ("HEADS", heads)],
        grid=(threads * qh, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[dot.shape],
        output_dtypes=[dot.dtype],
    )[0]


def attend(q_latent, q_pe, latent, k_pe, topk, mask, sinks, scale):
    """Short single-sequence absorbed attention with shared KV gathers."""
    _, heads, rows, _ = q_pe.shape
    total = latent.shape[2]
    indices = topk[:, 0]
    row_ends = mx.arange(total - rows, total)
    keep = (indices < total) & (indices <= row_ends[:, None])
    safe = mx.minimum(indices, total - 1).astype(mx.int32)
    if mask is not None:
        visible = mx.broadcast_to(mask, (1, 1, rows, total))[:, 0]
        keep = keep & mx.take_along_axis(visible, safe, axis=-1)

    width = safe.shape[-1]
    rotary_keys = mx.take_along_axis(
        k_pe[:, 0], safe.reshape(1, rows * width, 1), axis=1
    )
    rotary_keys = rotary_keys.reshape(rows, 1, width, -1)
    rotary_query = q_pe.swapaxes(1, 2).reshape(rows, heads, 1, -1)
    rotary = (rotary_query * scale) @ rotary_keys.swapaxes(-1, -2)

    query = q_latent.swapaxes(1, 2).reshape(rows, heads, 1, -1)
    query = (query * mx.array(scale, dtype=query.dtype))[:, None]
    dots = _shared_gemv("qk", query, latent, safe)
    probs = fused_probs(dots, rotary, keep.reshape(rows, 1, 1, width), sinks)
    result = _shared_gemv("pv", probs, latent, safe)[:, 0]
    return result.reshape(1, rows, heads, -1).swapaxes(1, 2)
