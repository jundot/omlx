from __future__ import annotations

import logging
from functools import cache, lru_cache

import mlx.core as mx
from mlx_vlm.turboquant import (
    TurboQuantKVCache,
    TurboQuantMSEState,
    _pack_lowbit,
    _QuantizedStateProxy,
    _rht_forward,
    _rht_inverse,
    _rotation_matrix,
    _state_length,
    _unpack_lowbit,
)

from .turboquant_kv import BatchTurboQuantKVCache

logger = logging.getLogger(__name__)

AFFINE4_SCHEME = "affine4"

__all__ = [
    "AFFINE4_SCHEME",
    "Affine4Codec",
    "Affine4KVCache",
    "BatchAffine4KVCache",
]

_BITS = 4
_MIN_NATIVE_TOKENS = 256
_MAX_PADDED_ROWS = 32
# Bound each unfused float32 score matrix to 64 MiB, or one query row.
_MAX_SCORE_ELEMENTS = 16 * 1024 * 1024
_NATIVE_LAUNCHABLE = {}
_FUSED_QUANTIZE_LAUNCHABLE = {}
_DEQUANTIZE_LAUNCHABLE = {}


@cache
def _dequantize_rotated_kernel():
    if not (hasattr(mx, "metal") and mx.metal.is_available()):
        return None
    return mx.fast.metal_kernel(
        name="affine4_dequantize_rotated",
        input_names=["packed", "scales"],
        output_names=["output"],
        ensure_row_contiguous=False,
        source=r"""
            uint index = thread_position_in_grid.x;
            uint size = Dim;
            for (int axis = 0; axis < Rank; ++axis) size *= scales_shape[axis];
            if (index >= size) return;
            uint row = index / Dim;
            uint dimension = index % Dim;
            long packed_offset = long(dimension / 8) * long(packed_strides[Rank]);
            long scale_offset = 0;
            for (int axis = Rank - 1; axis >= 0; --axis) {
                uint coordinate = row % scales_shape[axis];
                row /= scales_shape[axis];
                packed_offset += long(coordinate) * long(packed_strides[axis]);
                scale_offset += long(coordinate) * long(scales_strides[axis]);
            }
            uint nibble = (packed[packed_offset] >> (4 * (dimension % 8))) & 15u;
            int code = int(nibble ^ 8u) - 8;
            output[index] = float(code) * float(scales[scale_offset]);
        """,
    )


class Affine4Codec:
    """Signed int4 codec with deterministic orthonormal rotation."""

    bits = _BITS

    def __init__(self, dim: int, seed: int = 0):
        if int(dim) != dim or dim <= 0:
            raise ValueError(f"affine4 requires a positive head dimension, got {dim}")
        self.dim = int(dim)
        self.seed = int(seed)
        dim = self.dim
        self.use_rht = (dim & (dim - 1)) == 0
        if self.use_rht:
            from mlx_vlm.turboquant import _rht_sign_vector

            self.signs = _rht_sign_vector(dim, seed)
            self.rotation = None
            self.rotation_t = None
        else:
            self.signs = None
            self.rotation = _rotation_matrix(dim, seed)
            self.rotation_t = self.rotation.transpose()

    def _rotate_forward(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        if self.use_rht:
            return _rht_forward(x, self.signs)
        return mx.matmul(x, self.rotation_t)

    def _rotate_inverse(self, x: mx.array) -> mx.array:
        if self.use_rht:
            return _rht_inverse(x, self.signs)
        return mx.matmul(x, self.rotation)

    def quantize(self, vectors: mx.array) -> TurboQuantMSEState:
        rotated = self._rotate_forward(vectors)
        scales = mx.maximum(
            mx.max(rotated, axis=-1) / 7.0,
            -mx.min(rotated, axis=-1) / 8.0,
        )
        divisor = mx.where(scales > 0, scales, 1)[..., None]
        codes = mx.clip(mx.round(rotated / divisor), -8, 7).astype(mx.int8)
        nibbles = codes.astype(mx.uint32) & 15
        if mx.default_device() == mx.cpu:
            width = (self.dim + 7) // 8
            nibbles = mx.pad(
                nibbles, [(0, 0)] * (nibbles.ndim - 1) + [(0, width * 8 - self.dim)]
            )
            words = nibbles.reshape(*nibbles.shape[:-1], width, 8)
            packed = mx.sum(words << (mx.arange(8, dtype=mx.uint32) * 4), axis=-1)
        else:
            packed = _pack_lowbit(nibbles, _BITS)
        return TurboQuantMSEState(scales, packed)

    def _codes(self, state: TurboQuantMSEState) -> mx.array:
        if mx.default_device() == mx.cpu:
            dimensions = mx.arange(self.dim, dtype=mx.uint32)
            words = mx.take(state.indices, dimensions // 8, axis=-1)
            nibbles = ((words >> ((dimensions % 8) * 4)) & 15).astype(mx.int16)
        else:
            nibbles = _unpack_lowbit(state.indices, _BITS, self.dim).astype(mx.int16)
        return ((nibbles ^ 8) - 8).astype(mx.float32)

    def dequantize_rotated(self, state: TurboQuantMSEState) -> mx.array:
        if state.norms.ndim == 0:
            expanded = TurboQuantMSEState(state.norms[None], state.indices[None])
            return self.dequantize_rotated(expanded)[0]
        signature = (self.dim, state.norms.ndim, state.norms.dtype)
        if (
            mx.default_device() == mx.gpu
            and _DEQUANTIZE_LAUNCHABLE.get(signature) is not False
            and state.norms.size
        ):
            try:
                kernel = _dequantize_rotated_kernel()
                if kernel is not None:
                    output = kernel(
                        inputs=[state.indices, state.norms],
                        template=[("Dim", self.dim), ("Rank", state.norms.ndim)],
                        output_shapes=[(*state.norms.shape, self.dim)],
                        output_dtypes=[mx.float32],
                        grid=(state.norms.size * self.dim, 1, 1),
                        threadgroup=(256, 1, 1),
                    )[0]
                    if signature not in _DEQUANTIZE_LAUNCHABLE:
                        mx.eval(output)
                        _DEQUANTIZE_LAUNCHABLE[signature] = True
                    return output
            except (RuntimeError, ValueError):
                _DEQUANTIZE_LAUNCHABLE[signature] = False
        return self._codes(state) * state.norms.astype(mx.float32)[..., None]

    def dequantize(self, state: TurboQuantMSEState) -> mx.array:
        return self._rotate_inverse(self.dequantize_rotated(state))

    def prepare_queries(self, queries: mx.array) -> mx.array:
        return self._rotate_forward(queries)


_FUSED_QUANTIZE_SOURCE = r"""
    uint d = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    uint is_value = threadgroup_position_in_grid.y;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    float input_value = is_value
        ? static_cast<float>(values[row * Dim + d])
        : static_cast<float>(keys[row * Dim + d]);
    float sign = is_value ? value_signs[d] : key_signs[d];
    threadgroup float rotated[Dim];
    rotated[d] = sign * input_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = 1; stride < Dim; stride *= 2) {
        int base = int(d) & ~stride;
        float low = rotated[base];
        float high = rotated[base | stride];
        float transformed = (int(d) & stride) ? low - high : low + high;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        rotated[d] = transformed;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float value = rotated[d] * rsqrt(float(Dim));

    float sg_positive = simd_max(value);
    float sg_negative = simd_max(-value);
    threadgroup float extrema[2 * SimdGroups];
    if (lane == 0) {
        extrema[sg] = sg_positive;
        extrema[SimdGroups + sg] = sg_negative;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float positive = (sg == 0 && lane < SimdGroups) ? extrema[lane] : 0.0f;
    float negative =
        (sg == 0 && lane < SimdGroups) ? extrema[SimdGroups + lane] : 0.0f;
    positive = simd_max(positive);
    negative = simd_max(negative);
    if (sg == 0 && lane == 0)
        extrema[0] = max(positive / 7.0f, negative / 8.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float quant_scale = extrema[0];
    if (d == 0) {
        if (is_value) value_scales[row] = quant_scale;
        else key_scales[row] = quant_scale;
    }
    float divisor = quant_scale;
    int code = divisor > 0.0f
        ? int(rint(clamp(value / divisor, -8.0f, 7.0f)))
        : 0;
    threadgroup uint codes[Dim];
    codes[d] = uint(code) & 15u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d < PackedWidth) {
        uint word = 0u;
        for (int i = 0; i < 8; ++i)
            word |= codes[d * 8 + i] << (i * 4);
        if (is_value) value_packed[row * PackedWidth + d] = word;
        else key_packed[row * PackedWidth + d] = word;
    }
"""


@cache
def _fused_quantize_kernel(dim: int):
    if (
        not hasattr(mx, "metal")
        or not mx.metal.is_available()
        or dim < 32
        or dim > 512
        or dim % 8
        or (dim & (dim - 1)) != 0
    ):
        return None
    return mx.fast.metal_kernel(
        name=f"affine4_fused_kv_quantize_d{dim}",
        input_names=["keys", "values", "key_signs", "value_signs"],
        output_names=[
            "key_scales",
            "key_packed",
            "value_scales",
            "value_packed",
        ],
        source=_FUSED_QUANTIZE_SOURCE,
    )


_MPP_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <int Dim, int Repeats, int QueryLength, int PaddedRows, int Warps,
          int NumKvHeads, typename Accumulator, typename ScalePtr>
METAL_FUNC void affine4_attention_impl(
    device uchar* keys,
    ScalePtr key_scales,
    device uchar* values,
    ScalePtr value_scales,
    device float* partials,
    device float* sums,
    device float* maxs,
    int tokens,
    int blocks,
    int valid_start,
    int valid_end,
    int key_token_stride,
    int value_token_stride,
    int key_scale_stride,
    int value_scale_stride,
    const device bool* mask,
    int mask_head_stride,
    int mask_query_stride,
    int mask_token_stride,
    bool causal,
    float attention_scale,
    uint tid,
    uint sg,
    uint3 group,
    threadgroup half* q_tile,
    threadgroup float* scores,
    threadgroup half* probabilities,
    threadgroup float* row_m,
    threadgroup float* row_l,
    threadgroup float* row_factor,
    threadgroup float* value_scale_max,
    threadgroup float* query_scales) {
    constexpr int ActiveRows = Repeats * QueryLength;
    constexpr int BK = 64;
    constexpr int OutDim = Dim / Warps;

    int bh = int(group.y) * NumKvHeads + int(group.x);
    int block = int(group.z);
    int ntiles = (tokens + BK - 1) / BK;
    int tiles_per_block = (ntiles + blocks - 1) / blocks;
    int tile_begin = block * tiles_per_block;
    int tile_end = min(ntiles, tile_begin + tiles_per_block);

    if (tile_begin >= tile_end) {
        for (int i = int(tid); i < ActiveRows * Dim; i += 32 * Warps) {
            int row = bh * ActiveRows + i / Dim;
            partials[(row * blocks + block) * Dim + i % Dim] = 0.0f;
        }
        for (int row = int(tid); row < ActiveRows; row += 32 * Warps) {
            int idx = (bh * ActiveRows + row) * blocks + block;
            sums[idx] = 0.0f;
            maxs[idx] = 0.0f;
        }
        return;
    }

    for (int row = int(tid); row < PaddedRows; row += 32 * Warps) {
        row_m[row] = -INFINITY;
        row_l[row] = 0.0f;
        row_factor[row] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    using QTensor = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;
    using KVTensor = tensor<device int4b_format, dextents<int, 2>, tensor_inline>;
    using PTensor = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;
    QTensor q_tensor(q_tile, dextents<int, 2>{Dim, PaddedRows});
    KVTensor k_tensor(keys, dextents<int, 2>{Dim, tokens},
        array<int, 2>{1, key_token_stride * 8});
    KVTensor v_tensor(values, dextents<int, 2>{Dim, tokens},
        array<int, 2>{1, value_token_stride * 8});
    PTensor p_tensor(probabilities, dextents<int, 2>{BK, PaddedRows});

    constexpr auto av_desc = matmul2d_descriptor(
        PaddedRows, OutDim, BK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<av_desc, execution_simdgroup> av_op;
    auto v_first = v_tensor.template slice<OutDim, BK>(
        int(sg) * OutDim, tile_begin * BK);
    auto output_tile = av_op.template get_destination_cooperative_tensor<
        decltype(p_tensor), decltype(v_first), Accumulator>();
    for (short i = 0; i < output_tile.get_capacity(); ++i)
        output_tile[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_scale = 0.0f;
    for (int t = tile_begin * BK + int(tid); t < min(tokens, tile_end * BK);
         t += 32 * Warps)
        local_scale = max(local_scale, float(value_scales[t * value_scale_stride]));
    local_scale = simd_max(local_scale);
    if (tid % 32 == 0) value_scale_max[sg] = local_scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float block_scale = 0.0f;
    for (int w = 0; w < Warps; ++w)
        block_scale = max(block_scale, value_scale_max[w]);
    block_scale = block_scale > 0.0f ? block_scale : 1.0f;

    for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
        int token = tile_idx * BK;
        int tile_valid_start = max(token, valid_start);
        int tile_valid_end = min(min(token + BK, tokens), valid_end);
        if (tile_valid_start >= tile_valid_end)
            continue;

        if (sg == 0) {
            constexpr auto qk_desc = matmul2d_descriptor(
                PaddedRows, BK, Dim, false, true, false);
            matmul2d<qk_desc, execution_simdgroup> qk_op;
            auto q_slice = q_tensor.template slice<Dim, PaddedRows>(0, 0);
            auto k_slice = k_tensor.template slice<Dim, BK>(0, token);
            auto score_tile = qk_op.template get_destination_cooperative_tensor<
                decltype(q_slice), decltype(k_slice), float>();
            qk_op.run(q_slice, k_slice, score_tile);
            for (short i = 0; i < score_tile.get_capacity(); ++i) {
                auto coord = score_tile.get_multidimensional_index(i);
                if (score_tile.is_valid_element(i)) {
                    int column = coord[0];
                    int row = coord[1];
                    int absolute_token = token + column;
                    int position = row % QueryLength;
                    int row_valid_end = causal
                        ? valid_end - QueryLength + position + 1 : valid_end;
                    scores[row * BK + column] = row < ActiveRows &&
                            absolute_token >= tile_valid_start &&
                            absolute_token < tile_valid_end &&
                            absolute_token < row_valid_end &&
                            mask[(row / QueryLength) * mask_head_stride +
                                 position * mask_query_stride +
                                 absolute_token * mask_token_stride]
                        ? score_tile[i] * query_scales[row] * attention_scale *
                              float(key_scales[absolute_token * key_scale_stride])
                        : -INFINITY;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (int(tid) < ActiveRows) {
            int row = int(tid);
            float tile_m = -INFINITY;
            for (int column = 0; column < BK; ++column)
                tile_m = max(tile_m, scores[row * BK + column]);
            float old_m = row_m[row];
            if (isinf(tile_m)) {
                row_factor[row] = 1.0f;
                for (int column = 0; column < BK; ++column)
                    probabilities[row * BK + column] = half(0.0f);
            } else {
                float new_m = max(old_m, tile_m);
                float factor = isinf(old_m) ? 0.0f : fast::exp(old_m - new_m);
                float l = row_l[row] * factor;
                for (int column = 0; column < BK; ++column) {
                    int absolute_token = token + column;
                    bool active = !isinf(scores[row * BK + column]);
                    float weight = active
                        ? fast::exp(scores[row * BK + column] - new_m)
                        : 0.0f;
                    l += weight;
                    probabilities[row * BK + column] = half(
                        weight * (active ? float(value_scales[absolute_token * value_scale_stride]) / block_scale : 0.0f));
                }
                row_m[row] = new_m;
                row_l[row] = l;
                row_factor[row] = factor;
            }
        }
        for (int i = int(tid) + ActiveRows * BK;
             i < PaddedRows * BK; i += 32 * Warps)
            probabilities[i] = half(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < output_tile.get_capacity(); ++i) {
            auto coord = output_tile.get_multidimensional_index(i);
            if (output_tile.is_valid_element(i))
                output_tile[i] *= row_factor[coord[1]];
        }
        auto v_slice = v_tensor.template slice<OutDim, BK>(
            int(sg) * OutDim, token);
        av_op.run(p_tensor, v_slice, output_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (int(tid) < ActiveRows) {
        int idx = (bh * ActiveRows + int(tid)) * blocks + block;
        sums[idx] = row_l[tid];
        maxs[idx] = row_m[tid];
    }
    for (short i = 0; i < output_tile.get_capacity(); ++i) {
        auto coord = output_tile.get_multidimensional_index(i);
        if (output_tile.is_valid_element(i) && coord[1] < ActiveRows) {
            int row = bh * ActiveRows + coord[1];
            int dimension = int(sg) * OutDim + coord[0];
            partials[(row * blocks + block) * Dim + dimension] = output_tile[i] * block_scale;
        }
    }
}
"""


_MPP_ATTENTION_SOURCE = r"""
    uint tid = thread_index_in_threadgroup;
    uint3 group = threadgroup_position_in_grid;
    constexpr int ActiveRows = Repeats * QueryLength;
    threadgroup half q_tile[PaddedRows * Dim];
    threadgroup float scores[PaddedRows * 64];
    threadgroup half probabilities[PaddedRows * 64];
    threadgroup float row_m[PaddedRows];
    threadgroup float row_l[PaddedRows];
    threadgroup float row_factor[PaddedRows];
    threadgroup float value_scale_max[Warps];
    threadgroup float query_scales[PaddedRows];

    int batch = int(group.y);
    int kv_head = int(group.x);
    if (int(tid) < PaddedRows) {
        int row = int(tid);
        float maximum = 0.0f;
        if (row < ActiveRows) {
            int head = kv_head * Repeats + row / QueryLength;
            int position = row % QueryLength;
            for (int d = 0; d < Dim; ++d)
                maximum = max(maximum, abs(float(queries[
                    batch * queries_strides[0] + head * queries_strides[1] +
                    position * queries_strides[2] + d])));
        }
        query_scales[row] = max(1.0f, maximum / 60000.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid); i < PaddedRows * Dim; i += 32 * Warps) {
        int row = i / Dim;
        if (row < ActiveRows) {
            int head = kv_head * Repeats + row / QueryLength;
            int position = row % QueryLength;
            q_tile[i] = half(queries[
                batch * queries_strides[0] + head * queries_strides[1] +
                position * queries_strides[2] + i % Dim] / query_scales[row]);
        } else {
            q_tile[i] = half(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int tokens = int(params[0]);
    int blocks = int(params[1]);
    float attention_scale = as_type<float>(params[2]);
    int key_word_offset = batch * int(keys_strides[0]) +
        kv_head * int(keys_strides[1]);
    int value_word_offset = batch * int(values_strides[0]) +
        kv_head * int(values_strides[1]);
    int key_scale_offset = batch * int(key_scales_strides[0]) +
        kv_head * int(key_scales_strides[1]);
    int value_scale_offset = batch * int(value_scales_strides[0]) +
        kv_head * int(value_scales_strides[1]);
    affine4_attention_impl<
        Dim, Repeats, QueryLength, PaddedRows, Warps, NumKvHeads, Accumulator>(
        (device uchar*)(keys + key_word_offset),
        key_scales + key_scale_offset,
        (device uchar*)(values + value_word_offset),
        value_scales + value_scale_offset,
        partials,
        sums,
        maxs,
        tokens,
        blocks,
        int(valid_starts[batch]),
        int(valid_ends[batch]),
        int(keys_strides[2]),
        int(values_strides[2]),
        int(key_scales_strides[2]),
        int(value_scales_strides[2]),
        mask + batch * mask_strides[0] + kv_head * Repeats * mask_strides[1],
        int(mask_strides[1]),
        int(mask_strides[2]),
        int(mask_strides[3]),
        bool(params[3]),
        attention_scale,
        tid,
        simdgroup_index_in_threadgroup,
        group,
        q_tile,
        scores,
        probabilities,
        row_m,
        row_l,
        row_factor,
        value_scale_max,
        query_scales);
"""


_MPP_REDUCE_SOURCE = r"""
    uint dimension = thread_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.x;
    int blocks = int(params[1]);
    threadgroup float global_m;
    threadgroup float global_l;
    if (dimension == 0) {
        float maximum = -INFINITY;
        for (int block = 0; block < blocks; ++block) {
            int idx = int(row) * blocks + block;
            if (sums[idx] > 0.0f)
                maximum = max(maximum, maxs[idx]);
        }
        float normalizer = 0.0f;
        for (int block = 0; block < blocks; ++block) {
            int idx = int(row) * blocks + block;
            if (sums[idx] > 0.0f)
                normalizer += sums[idx] * fast::exp(maxs[idx] - maximum);
        }
        global_m = maximum;
        global_l = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float value = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        int idx = int(row) * blocks + block;
        if (sums[idx] > 0.0f)
            value += partials[idx * Dim + int(dimension)] *
                fast::exp(maxs[idx] - global_m);
    }
    output[int(row) * Dim + int(dimension)] = global_l > 0.0f
        ? value / global_l : 0.0f;
"""


@cache
def _mpp_attention_kernel(
    dim: int,
    repeats: int,
    query_length: int,
    padded_rows: int,
    warps: int,
    kv_heads: int,
):
    return mx.fast.metal_kernel(
        name=(
            f"affine4_mpp_d{dim}_r{repeats}_l{query_length}_"
            f"m{padded_rows}_w{warps}_h{kv_heads}"
        ),
        input_names=[
            "queries",
            "keys",
            "key_scales",
            "values",
            "value_scales",
            "valid_starts",
            "valid_ends",
            "mask",
            "params",
        ],
        output_names=["partials", "sums", "maxs"],
        header=_MPP_HEADER,
        source=_MPP_ATTENTION_SOURCE,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "fast"},
    )


@cache
def _mpp_reduce_kernel(dim: int):
    return mx.fast.metal_kernel(
        name=f"affine4_mpp_reduce_d{dim}",
        input_names=["partials", "sums", "maxs", "params"],
        output_names=["output"],
        header="using namespace metal;\n",
        source=_MPP_REDUCE_SOURCE,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "fast"},
    )


def _float_bits(value: float) -> int:
    import struct

    return struct.unpack("I", struct.pack("f", value))[0]


@lru_cache(maxsize=1)
def _m5_mpp_available() -> bool:
    if not (hasattr(mx, "metal") and mx.metal.is_available()):
        return False
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return str(info.get("architecture", "")).startswith("applegpu_g17")


def _padded_rows(active_rows: int) -> int:
    return max(8, ((active_rows + 7) // 8) * 8)


def _attention_blocks(tokens: int, kv_heads: int, dim: int) -> int:
    token_tiles = (tokens + 63) // 64
    target_groups = 256 if dim <= 64 else 168
    return min(token_tiles, max(1, (target_groups + kv_heads - 1) // kv_heads))


def _attention_warps(dim: int) -> int:
    if dim <= 64:
        return 1
    if dim == 96:
        return 2
    return 4


def _native_attention(
    cache,
    queries: mx.array,
    keys_state: TurboQuantMSEState,
    values_state: TurboQuantMSEState,
    scale: float,
    mask,
) -> mx.array | None:
    if (
        mx.default_device() != mx.gpu
        or not _m5_mpp_available()
        or queries.ndim != 4
        or queries.dtype not in (mx.float16, mx.bfloat16)
    ):
        return None
    if (
        not isinstance(cache.key_codec, Affine4Codec)
        or not isinstance(cache.value_codec, Affine4Codec)
        or not isinstance(keys_state, TurboQuantMSEState)
        or not isinstance(values_state, TurboQuantMSEState)
        or keys_state.indices.ndim != 4
        or values_state.indices.ndim != 4
        or keys_state.norms.ndim != 3
        or values_state.norms.ndim != 3
    ):
        return None
    batch, query_heads, query_length, dim = queries.shape
    kv_heads = keys_state.norms.shape[1]
    if (
        dim != cache.key_codec.dim
        or dim != cache.value_codec.dim
        or kv_heads <= 0
        or keys_state.norms.shape[:2] != (batch, kv_heads)
        or values_state.norms.shape != keys_state.norms.shape
        or keys_state.indices.shape[:3] != keys_state.norms.shape
        or values_state.indices.shape != keys_state.indices.shape
        or keys_state.indices.shape[-1] != dim // 8
        or keys_state.indices.dtype != mx.uint32
        or values_state.indices.dtype != mx.uint32
        or keys_state.norms.dtype not in (mx.float16, mx.float32)
        or values_state.norms.dtype != keys_state.norms.dtype
        or dim % 32
        or dim > 512
        or query_heads % kv_heads
        or query_length < 1
        or query_length > 4
    ):
        return None
    repeats = query_heads // kv_heads
    padded_rows = _padded_rows(repeats * query_length)
    warps = _attention_warps(dim)
    if padded_rows > _MAX_PADDED_ROWS or dim % warps:
        return None
    tokens = _state_length(keys_state)
    if (
        tokens < _MIN_NATIVE_TOKENS
        or keys_state.indices.nbytes <= 4096
        or _state_length(values_state) != tokens
    ):
        return None

    is_batch = isinstance(cache, BatchAffine4KVCache)
    valid_starts = (
        cache.left_padding.astype(mx.int32)
        if is_batch
        else mx.zeros((batch,), dtype=mx.int32)
    )
    causal = isinstance(mask, str) and mask == "causal"
    if isinstance(mask, mx.array):
        if mask.dtype != mx.bool_:
            return None
    elif mask is None or causal:
        mask = mx.array(True)
    else:
        return None
    mask = mx.broadcast_to(mask, (batch, query_heads, query_length, tokens))
    valid_ends = mx.full((batch,), tokens, dtype=mx.int32)

    blocks = _attention_blocks(tokens, kv_heads, dim)
    partition_tiles = ((tokens + 63) // 64 + blocks - 1) // blocks
    # Normalized probabilities are <= 1 and signed codes have magnitude <= 8.
    # Limit half partials to 49,152, leaving headroom for accumulation rounding.
    accumulator = mx.float16 if partition_tiles <= 96 else mx.float32
    params = mx.array(
        [tokens, blocks, _float_bits(float(scale)), int(causal)], dtype=mx.uint32
    )
    grouped = queries.reshape(batch, kv_heads, repeats, query_length, dim)
    rotated = cache.key_codec.prepare_queries(grouped)
    rows = batch * query_heads * query_length
    geometry = (dim, repeats, query_length, padded_rows, warps, kv_heads)
    signature = (*geometry, keys_state.norms.dtype, accumulator)
    if _NATIVE_LAUNCHABLE.get(signature) is False:
        return None

    try:
        pass1 = _mpp_attention_kernel(*geometry)
        partials, sums, maxs = pass1(
            inputs=[
                rotated.reshape(batch, query_heads, query_length, dim),
                keys_state.indices,
                keys_state.norms,
                values_state.indices,
                values_state.norms,
                valid_starts,
                valid_ends,
                mask,
                params,
            ],
            template=[
                ("Dim", dim),
                ("Repeats", repeats),
                ("QueryLength", query_length),
                ("PaddedRows", padded_rows),
                ("Warps", warps),
                ("NumKvHeads", kv_heads),
                ("Accumulator", accumulator),
            ],
            output_shapes=[
                (rows, blocks, dim),
                (rows, blocks),
                (rows, blocks),
            ],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            grid=(kv_heads * 32, batch * warps, blocks),
            threadgroup=(32, warps, 1),
        )
        rotated_output = _mpp_reduce_kernel(dim)(
            inputs=[partials, sums, maxs, params],
            template=[("Dim", dim)],
            output_shapes=[(batch, kv_heads, repeats, query_length, dim)],
            output_dtypes=[mx.float32],
            grid=(rows * dim, 1, 1),
            threadgroup=(dim, 1, 1),
        )[0]
        output = cache.value_codec._rotate_inverse(rotated_output)
        output = output.reshape(batch, query_heads, query_length, dim).astype(
            queries.dtype
        )
        if signature not in _NATIVE_LAUNCHABLE:
            mx.eval(output)
            _NATIVE_LAUNCHABLE[signature] = True
            logger.info(
                "Affine4 native M5 attention active (d=%d, repeats=%d, rows=%d)",
                dim,
                repeats,
                query_length,
            )
        return output
    except (RuntimeError, ValueError):
        _NATIVE_LAUNCHABLE[signature] = False
        logger.warning(
            "Affine4 M5 kernel rejected geometry d=%d r=%d l=%d; using fallback",
            dim,
            repeats,
            query_length,
            exc_info=True,
        )
        return None


class Affine4KVCache(TurboQuantKVCache):
    """Rotated signed-int4 cache using TurboQuant's packed-state lifecycle.

    Float32 scales preserve the range of BF16 and float32 activations. Native
    specializations are evaluated on first use so deferred compilation failures
    can select the portable path. Warm calls stay asynchronous; execution errors
    from those calls propagate at the caller's evaluation boundary.
    """

    quantization_scheme = AFFINE4_SCHEME

    def __init__(self, bits: float = 4, seed: int = 0):
        if float(bits) != 4:
            raise ValueError(f"affine4 requires exactly 4 bits, got {bits}")
        super().__init__(bits=4, seed=int(seed))
        self._key_dim = self._value_dim = 0

    @staticmethod
    def _contiguous_packed_state(state):
        state = Affine4KVCache._unwrap(state)
        if state is None:
            return None
        return TurboQuantMSEState(state.norms, mx.contiguous(state.indices))

    @TurboQuantKVCache.state.setter
    def state(self, value):
        if value is not None:
            value = tuple(self._contiguous_packed_state(s) for s in value)
        TurboQuantKVCache.state.fset(self, value)

    @classmethod
    def from_cache(cls, cache, bits: float = 4, seed: int = 0):
        result = cls(bits=bits, seed=seed)
        if cache.empty():
            return result
        keys, values = cache.state
        if keys is None:
            return result
        if isinstance(cache, Affine4KVCache) and cache.seed == result.seed:
            result.meta_state = cache.meta_state
            result.state = (cls._unwrap(keys), cls._unwrap(values))
            result.rebuild_codecs(*result.state)
        else:
            if isinstance(cache, TurboQuantKVCache):
                keys, values = cache.dequantize(keys, values)
            result.update_and_fetch(keys, values)
        return result

    @classmethod
    def merge(cls, caches):
        return BatchAffine4KVCache.merge(caches)

    def _ensure_codecs(self, keys: mx.array, values: mx.array):
        if keys.ndim != 4 or values.ndim != 4 or keys.shape[:3] != values.shape[:3]:
            raise ValueError("affine4 keys and values must have matching B,H,T axes")
        for name, tensor, seed in (
            ("key", keys, self.seed),
            ("value", values, self.seed + 1),
        ):
            dim = tensor.shape[-1]
            codec = getattr(self, f"{name}_codec")
            if codec is None:
                stored_dim = getattr(self, f"_{name}_dim", 0)
                if stored_dim and stored_dim != dim:
                    raise ValueError(f"affine4 {name} dimension differs from metadata")
                setattr(self, f"{name}_codec", Affine4Codec(dim, seed))
                setattr(self, f"_{name}_dim", dim)
            elif codec.dim != dim:
                raise ValueError(f"affine4 {name} head dimension changed")

    @property
    def meta_state(self):
        offset = self._phys_end if isinstance(self.offset, mx.array) else self.offset
        key_dim = self.key_codec.dim if self.key_codec is not None else self._key_dim
        value_dim = (
            self.value_codec.dim if self.value_codec is not None else self._value_dim
        )
        return tuple(
            map(
                str,
                (
                    offset,
                    self.bits,
                    self.seed,
                    self.quantization_scheme,
                    key_dim,
                    value_dim,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, value):
        if len(value) != 6 or value[3] != AFFINE4_SCHEME:
            raise ValueError(
                "affine4 metadata requires scheme and key/value dimensions"
            )
        offset, bits, seed = int(value[0]), float(value[1]), int(value[2])
        key_dim, value_dim = int(value[4]), int(value[5])
        if bits != 4 or offset < 0 or min(key_dim, value_dim) < 0:
            raise ValueError("Invalid affine4 cache metadata")
        if (offset > 0 and min(key_dim, value_dim) == 0) or bool(key_dim) != bool(
            value_dim
        ):
            raise ValueError("affine4 nonempty metadata requires positive dimensions")
        self.bits, self.seed = bits, seed
        self._key_dim, self._value_dim = key_dim, value_dim
        self.key_codec = Affine4Codec(key_dim, seed) if key_dim else None
        self.value_codec = Affine4Codec(value_dim, seed + 1) if value_dim else None
        if hasattr(self, "left_padding"):
            self._phys_end = offset
            self.offset = offset - self.left_padding
        else:
            self.offset = offset
        self._cached_state = None
        self._cached_state_offset = -1
        self._shadow_keys = self._shadow_values = None

    def rebuild_codecs(self, keys_state, values_state):
        """Validate packed states against explicit dimensions before SSD reuse."""
        keys_state, values_state = self._unwrap(keys_state), self._unwrap(values_state)
        if keys_state is None and values_state is None:
            return
        dimensions = (
            self.key_codec.dim if self.key_codec is not None else self._key_dim,
            self.value_codec.dim if self.value_codec is not None else self._value_dim,
        )
        for name, state, dim in zip(
            ("key", "value"), (keys_state, values_state), dimensions
        ):
            if dim <= 0:
                raise ValueError(f"affine4 {name} dimension is missing from metadata")
            if not isinstance(state, TurboQuantMSEState):
                raise ValueError(
                    f"affine4 {name} state must contain scales and packed int4"
                )
            if (
                state.norms.ndim != 3
                or state.indices.shape != (*state.norms.shape, (dim + 7) // 8)
                or state.indices.dtype != mx.uint32
                or state.norms.dtype not in (mx.float16, mx.float32)
            ):
                raise ValueError(
                    f"affine4 {name} state shape/dtype disagrees with metadata"
                )
        if keys_state.norms.shape != values_state.norms.shape:
            raise ValueError("affine4 key/value state axes disagree")
        self._key_dim, self._value_dim = dimensions
        self.key_codec = Affine4Codec(dimensions[0], self.seed)
        self.value_codec = Affine4Codec(dimensions[1], self.seed + 1)

    def _try_fused_kv_quantize(self, keys, values):
        self._ensure_codecs(keys, values)
        if mx.default_device() != mx.gpu:
            return None, None
        dim = keys.shape[-1]
        signature = (dim, keys.dtype, values.dtype)
        if _FUSED_QUANTIZE_LAUNCHABLE.get(signature) is False:
            return None, None
        if values.shape != keys.shape:
            return None, None
        try:
            kernel = _fused_quantize_kernel(dim)
        except (RuntimeError, ValueError):
            _FUSED_QUANTIZE_LAUNCHABLE[signature] = False
            return None, None
        if kernel is None:
            return None, None
        flat_keys = keys.reshape(-1, dim)
        flat_values = values.reshape(-1, dim)
        rows = flat_keys.shape[0]
        packed_width = dim // 8
        try:
            outputs = kernel(
                inputs=[
                    flat_keys,
                    flat_values,
                    self.key_codec.signs,
                    self.value_codec.signs,
                ],
                template=[
                    ("Dim", dim),
                    ("SimdGroups", (dim + 31) // 32),
                    ("PackedWidth", packed_width),
                ],
                output_shapes=[
                    (rows,),
                    (rows, packed_width),
                    (rows,),
                    (rows, packed_width),
                ],
                output_dtypes=[
                    mx.float32,
                    mx.uint32,
                    mx.float32,
                    mx.uint32,
                ],
                grid=(dim * rows, 2, 1),
                threadgroup=(dim, 1, 1),
            )
            key_scales, key_packed, value_scales, value_packed = outputs
            if signature not in _FUSED_QUANTIZE_LAUNCHABLE:
                mx.eval(*outputs)
                _FUSED_QUANTIZE_LAUNCHABLE[signature] = True
            shape = keys.shape[:-1]
            return (
                TurboQuantMSEState(
                    key_scales.reshape(shape),
                    key_packed.reshape(*shape, packed_width),
                ),
                TurboQuantMSEState(
                    value_scales.reshape(shape),
                    value_packed.reshape(*shape, packed_width),
                ),
            )
        except (RuntimeError, ValueError):
            _FUSED_QUANTIZE_LAUNCHABLE[signature] = False
            return None, None

    def _attention_mask(self, mask, query_length, tokens):
        if isinstance(mask, str) and mask != "causal":
            raise ValueError(f"Unsupported affine4 attention mask: {mask}")
        return mask

    def _portable_attention(self, queries, keys, values, scale, mask, sinks):
        batch, heads, query_length, _ = queries.shape
        tokens = keys.shape[-2]
        step = max(1, _MAX_SCORE_ELEMENTS // max(1, batch * heads * tokens))
        sinks = None if sinks is None else sinks.astype(mx.float32)
        if query_length <= step:
            return mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=scale,
                mask=self._attention_mask(mask, query_length, tokens),
                sinks=sinks,
            )

        outputs = []
        for start in range(0, query_length, step):
            end = min(start + step, query_length)
            tile_mask = mask
            if isinstance(mask, str) and mask == "causal":
                positions = mx.arange(
                    tokens - query_length + start, tokens - query_length + end
                )
                tile_mask = positions[:, None] >= mx.arange(tokens)[None, :]
            elif isinstance(mask, mx.array) and mask.ndim >= 2 and mask.shape[-2] != 1:
                tile_mask = mask[..., start:end, :]
            output = mx.fast.scaled_dot_product_attention(
                queries[..., start:end, :],
                keys,
                values,
                scale=scale,
                mask=self._attention_mask(tile_mask, end - start, tokens),
                sinks=sinks,
            )
            # Retire each tile's score buffer before scheduling the next tile.
            mx.eval(output)
            outputs.append(output)
        return mx.concatenate(outputs, axis=-2)

    def attention(
        self,
        queries,
        keys_state=None,
        values_state=None,
        scale=1,
        mask=None,
        sinks=None,
    ):
        if keys_state is None or values_state is None:
            keys_state, values_state = self.state
        else:
            if not isinstance(keys_state, _QuantizedStateProxy):
                keys_state = self._contiguous_packed_state(keys_state)
            if not isinstance(values_state, _QuantizedStateProxy):
                values_state = self._contiguous_packed_state(values_state)
        keys_state, values_state = self._unwrap(keys_state), self._unwrap(values_state)
        if keys_state is None or values_state is None:
            raise ValueError("Cannot attend to an empty affine4 cache")
        if self.key_codec is None or self.value_codec is None:
            self.rebuild_codecs(keys_state, values_state)
        if sinks is None:
            output = _native_attention(
                self, queries, keys_state, values_state, scale, mask
            )
            if output is not None:
                return output
        rotated_keys = self.key_codec.dequantize_rotated(keys_state)
        rotated_values = self.value_codec.dequantize_rotated(values_state)
        rotated_output = self._portable_attention(
            self.key_codec.prepare_queries(queries),
            rotated_keys,
            rotated_values,
            scale,
            mask,
            sinks,
        )
        return self.value_codec._rotate_inverse(rotated_output).astype(queries.dtype)

    decode_attention = attention
    prefill_attention = attention
    quantized_attention = attention


class BatchAffine4KVCache(BatchTurboQuantKVCache, Affine4KVCache):
    """Affine4 cache with oMLX continuous-batching operations."""

    def __init__(self, left_padding, bits: float = 4, seed: int = 0):
        if not left_padding:
            raise ValueError("affine4 batch requires at least one row")
        super().__init__(left_padding, bits=bits, seed=seed)
        self.offset = -self.left_padding

    @BatchTurboQuantKVCache.state.setter
    def state(self, value):
        Affine4KVCache.state.fset(self, value)
        self._phys_end = self.offset
        self.offset = self._phys_end - self.left_padding

    @staticmethod
    def _validate_dimensions(caches):
        dimensions = {
            c.meta_state[4:]
            for c in caches
            if isinstance(c, Affine4KVCache) and c.key_codec is not None
        }
        if len(dimensions) > 1:
            raise ValueError(
                "Cannot batch affine4 caches with different key/value dimensions"
            )

    @classmethod
    def merge(cls, caches):
        cls._validate_dimensions(caches)
        return super().merge(caches)

    def extend(self, other):
        self._validate_dimensions((self, other))
        return super().extend(other)

    def _new_single_cache(self):
        return Affine4KVCache(bits=self.bits, seed=self.seed)

    def _attention_mask(self, mask, query_length, tokens):
        causal = isinstance(mask, str) and mask == "causal"
        if isinstance(mask, str) and not causal:
            raise ValueError(f"Unsupported affine4 attention mask: {mask}")
        columns = mx.arange(tokens)
        allowed = columns[None, None, None, :] >= self.left_padding[:, None, None, None]
        if causal:
            allowed = allowed & (
                columns[None, :] <= mx.arange(tokens - query_length, tokens)[:, None]
            )
        if mask is None or isinstance(mask, str):
            return allowed
        if mask.dtype == mx.bool_:
            return allowed & mask
        return mx.where(allowed, mask, -float("inf"))
