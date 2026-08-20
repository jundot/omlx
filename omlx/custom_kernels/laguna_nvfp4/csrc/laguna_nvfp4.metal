// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// NVFP4 (E4M3) decode kernels ported from Layr-Labs/mlxfast-challenge
// (Sources/MLXFastModel/LagunaRuntimeModel.swift). Kernel bodies are kept
// VERBATIM (included only the framework-injected buffer attributes and the
// resolved default flag configuration: DARKBLOOM_NVFP4_SCALE_FOLD /
// SCALE_DEFER / SCALE_CARRY / QDOT_SEED_ELIDE and E4M3_SIGN_DOMAIN on,
// nibble split 1).
//
// NVFP4 contract: U32-packed weights [N, K*4/32], U8 E4M3 group scales
// [N, K/16], group size 16, no biases. The fused gate/up plane concatenates
// gate rows 0..N-1 over up rows N..2N-1 on the row axis.
//
// The row-scale suffix (2^22) is the deferred form of the fold: the scale
// carries 2^14 of the weight magnitude, the kernel re-applies 2^22 once per
// output row before the single bf16 rounding (bit-exact per the challenge's
// closed case analysis).

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
// clang-format on

static inline float laguna_nvfp4_scale(uint8_t bits) {
    if (bits < 16u) {
        ushort fast_raw = ushort(bits) << 7;
        return float(as_type<half>(fast_raw));
    }
    ushort raw = ushort(uint(bits) << 7);
    half converted = as_type<half>(raw);
    half signed_value = converted;
    return float(signed_value);
}

static inline float laguna_nvfp4_qdot_codes_16(
    uint2 codes,
    const thread float* input,
    float scale
) {
    float accum;
    {
        const uint c = codes.x;
        const uint xe = c & 0x0F0F0F0Fu;
        const uint ge = xe | (xe << 3);
        const uint yo = c & 0xF0F0F0F0u;
        const uint go = yo | (yo >> 3);
        const uint p0 = (ge << 9) & 0x8E008E00u;
        const uint p1 = (go << 8) & 0x8E008E00u;
        const uint p2 = (ge << 1) & 0x8E008E00u;
        const uint p3 = go & 0x8E008E00u;
        const float2 v04 = float2(as_type<half2>(p0));
        const float2 v15 = float2(as_type<half2>(p1));
        const float2 v26 = float2(as_type<half2>(p2));
        const float2 v37 = float2(as_type<half2>(p3));
        accum =
            (input[0] * v04.x +
             input[1] * v15.x +
             input[2] * v26.x +
             input[3] * v37.x);
        accum +=
            (input[4] * v04.y +
             input[5] * v15.y +
             input[6] * v26.y +
             input[7] * v37.y);
    }
    {
        const uint c = codes.y;
        const uint xe = c & 0x0F0F0F0Fu;
        const uint ge = xe | (xe << 3);
        const uint yo = c & 0xF0F0F0F0u;
        const uint go = yo | (yo >> 3);
        const uint p0 = (ge << 9) & 0x8E008E00u;
        const uint p1 = (go << 8) & 0x8E008E00u;
        const uint p2 = (ge << 1) & 0x8E008E00u;
        const uint p3 = go & 0x8E008E00u;
        const float2 v04 = float2(as_type<half2>(p0));
        const float2 v15 = float2(as_type<half2>(p1));
        const float2 v26 = float2(as_type<half2>(p2));
        const float2 v37 = float2(as_type<half2>(p3));
        accum +=
            (input[8] * v04.x +
             input[9] * v15.x +
             input[10] * v26.x +
             input[11] * v37.x);
        accum +=
            (input[12] * v04.y +
             input[13] * v15.y +
             input[14] * v26.y +
             input[15] * v37.y);
    }
    return scale * accum;
}

static inline float laguna_nvfp4_qdot_16(
    const device uint8_t* weight,
    const thread float* input,
    float scale
) {
    const device uint2* packed = (const device uint2*)weight;
    return laguna_nvfp4_qdot_codes_16(packed[0], input, scale);
}

// Shared-expert fused gate/up NVFP4 QMV with in-kernel SwiGLU activation.
// (verbatim from lagunaSharedSwiGLUQMVKernel; buffer attributes injected)
//   input        [2048] bf16            — shared-expert input
//   fused_weight [1024][1024] uint8     — 512 gate rows then 512 up rows
//   fused_scales [1024][128]  uint8     — E4M3 group-16 scales (2048/16)
//   activated    [512] bf16             — silu(gate) * up
kernel void laguna_shared_nvfp4_swiglu_qmv_bf16_v1(
    const device bfloat* input [[buffer(0)]],
    const device uint8_t* fused_weight [[buffer(1)]],
    const device uint8_t* fused_scales [[buffer(2)]],
    device bfloat* activated [[buffer(3)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint input_width = 2048;
    constexpr uint output_width = 512;
    constexpr uint fused_width = 1024;
    constexpr uint packed_row_bytes = 1024;
    constexpr uint scale_row_bytes = 128;
    constexpr uint block_width = 512;
    constexpr uint values_per_lane = 16;

    uint first_row = tile * 4 + simd_group * 2;

    thread float gate_result[2] = {0.0f, 0.0f};
    thread float up_result[2] = {0.0f, 0.0f};
    thread float input_values[values_per_lane];

    for (uint block = 0; block < input_width; block += block_width) {
        const device vec<bfloat, 4>* input_vectors =
            (const device vec<bfloat, 4>*)(
                input + block + lane * values_per_lane);
        for (uint i = 0; i < values_per_lane / 4; ++i) {
            const vec<bfloat, 4> values = input_vectors[i];
            input_values[4 * i] = values[0];
            input_values[4 * i + 1] = values[1];
            input_values[4 * i + 2] = values[2];
            input_values[4 * i + 3] = values[3];
        }

        for (uint row = 0; row < 2; ++row) {
            uint gate_row = first_row + row;
            uint up_row = gate_row + output_width;
            const device uint8_t* gate_weight =
                (const device uint8_t*)fused_weight +
                gate_row * packed_row_bytes + block / 2 + lane * 8;
            const device uint8_t* up_weight =
                (const device uint8_t*)fused_weight +
                up_row * packed_row_bytes + block / 2 + lane * 8;
            const device uint8_t* gate_scale =
                fused_scales + gate_row * scale_row_bytes +
                block / 16 + lane;
            const device uint8_t* up_scale =
                fused_scales + up_row * scale_row_bytes +
                block / 16 + lane;

            gate_result[row] += laguna_nvfp4_qdot_16(
                gate_weight,
                input_values,
                laguna_nvfp4_scale(gate_scale[0]));
            up_result[row] += laguna_nvfp4_qdot_16(
                up_weight,
                input_values,
                laguna_nvfp4_scale(up_scale[0]));
        }
    }

    for (uint row = 0; row < 2; ++row) {
        gate_result[row] = simd_sum(gate_result[row]);
        up_result[row] = simd_sum(up_result[row]);
        if (lane == 0) {
            bfloat gate = bfloat(gate_result[row] * 4194304.0f);
            bfloat up = bfloat(up_result[row] * 4194304.0f);
            bfloat exp_abs = metal::exp(metal::abs(gate));
            bfloat denominator = bfloat(1) + exp_abs;
            bfloat y = bfloat(1) / denominator;
            bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
            bfloat silu = bfloat(gate * sigmoid);
            activated[first_row + row] = bfloat(silu * up);
        }
    }
}

// Shared-expert down_proj with routed + residual adds fused in one kernel.
// (verbatim from lagunaSharedDownResidualSource(halved: false))
//   activated   [512] bf16             — swiglu output (silu(gate)*up)
//   down_weight [2048][256] uint8      — NVFP4 down_proj (K=512 nibbles)
//   down_scales [2048][32]  uint8      — E4M3 group-16 scales
//   routed      [2048] bf16            — routed-expert output to add
//   residual    [2048] bf16            — decoder residual
//   output      [2048] bf16            — residual + (routed + shared)
// Grid: 256 groups x (2 simdgroups x 4 rows) = 2048 output rows.
kernel void laguna_shared_nvfp4_down_residual_bf16_v1(
    const device bfloat* activated [[buffer(0)]],
    const device uint8_t* down_weight [[buffer(1)]],
    const device uint8_t* down_scales [[buffer(2)]],
    const device bfloat* routed [[buffer(3)]],
    const device bfloat* residual [[buffer(4)]],
    device bfloat* output [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint input_width = 512;
    constexpr uint output_width = 2048;
    constexpr uint outputs_per_simd = 4;
    constexpr uint values_per_lane = 16;
    constexpr uint packed_row_bytes = 256;
    constexpr uint scale_row_bytes = 32;

    uint first_row =
        group * 2 * outputs_per_simd +
        simd_group * outputs_per_simd;

    thread float input_values[values_per_lane];
    const device vec<bfloat, 4>* input_vectors =
        (const device vec<bfloat, 4>*)(
            activated + lane * values_per_lane);
    for (uint i = 0; i < values_per_lane / 4; ++i) {
        const vec<bfloat, 4> values = input_vectors[i];
        input_values[4 * i] = values[0];
        input_values[4 * i + 1] = values[1];
        input_values[4 * i + 2] = values[2];
        input_values[4 * i + 3] = values[3];
    }

    thread float result[outputs_per_simd] = {
        0.0f, 0.0f, 0.0f, 0.0f
    };
    for (uint row = 0; row < outputs_per_simd; ++row) {
        uint output_row = first_row + row;
        const device uint8_t* weight =
            (const device uint8_t*)down_weight +
            output_row * packed_row_bytes + lane * 8;
        const device uint8_t* scale =
            down_scales + output_row * scale_row_bytes + lane;
        result[row] = laguna_nvfp4_qdot_16(
            weight,
            input_values,
            laguna_nvfp4_scale(scale[0]));
        result[row] = simd_sum(result[row]);
    }

    if (lane == 0) {
        for (uint row = 0; row < outputs_per_simd; ++row) {
            uint output_row = first_row + row;
            bfloat shared = bfloat(result[row] * 4194304.0f);
            bfloat r2 = bfloat(routed[output_row] + shared);
            output[output_row] =
                bfloat(residual[output_row] + r2);
        }
    }
}

// Routed-expert down_proj + weighted reduction over the 8 routed slots,
// fused in one kernel (verbatim from lagunaRoutedDownReduceKernel).
//   activated       [8][512] bf16         — per-slot swiglu outputs
//   down_weight     [E][2048][64] uint32  — per-expert NVFP4 down planes
//   down_scales     [128 + E*2048*16] uint8 — halved group-32 planes
//                                            (128-byte patch header)
//   indices         [8] uint32            — routed expert ids
//   router_weights  [8] fp32              — routed scores
//   routed          [2048] bf16           — sum_slots(act * w) * 2.5
// Grid: 512 tiles x (256 threads = 8 simdgroups; slot = simdgroup).
kernel void laguna_routed_nvfp4_down_reduce_bf16_v2(
    const device bfloat* activated [[buffer(0)]],
    const device uint8_t* down_weight [[buffer(1)]],
    const device uint8_t* down_scales [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    const device float* router_weights [[buffer(4)]],
    device bfloat* routed [[buffer(5)]],
    uint tile [[threadgroup_position_in_grid]],
    uint expert_slot [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint input_width = 512;
    constexpr uint output_width = 2048;
    constexpr uint experts_per_token = 8;
    constexpr uint outputs_per_simd = 4;
    constexpr uint values_per_lane = 16;
    constexpr uint packed_row_bytes = 256;
    constexpr uint scale_patch_bytes = 128;
    constexpr uint scale_row_bytes = 16;
    constexpr uint packed_expert_bytes =
        output_width * packed_row_bytes;
    constexpr uint scale_expert_bytes =
        output_width * scale_row_bytes;

    uint first_row = tile * outputs_per_simd;
    uint expert = uint(indices[expert_slot]);

    const device bfloat* expert_input =
        activated + expert_slot * input_width;
    const device uint8_t* expert_weight =
        (const device uint8_t*)down_weight +
        expert * packed_expert_bytes;
    const device uint8_t* expert_scales =
        down_scales + scale_patch_bytes + expert * scale_expert_bytes;

    thread float input_values[values_per_lane];
    const device vec<bfloat, 4>* input_vectors =
        (const device vec<bfloat, 4>*)(
            expert_input + lane * values_per_lane);
    for (uint i = 0; i < values_per_lane / 4; ++i) {
        const vec<bfloat, 4> values = input_vectors[i];
        input_values[4 * i] = values[0];
        input_values[4 * i + 1] = values[1];
        input_values[4 * i + 2] = values[2];
        input_values[4 * i + 3] = values[3];
    }

    thread float result[outputs_per_simd] = {
        0.0f, 0.0f, 0.0f, 0.0f
    };
    uint2 row_codes[outputs_per_simd];
    uint8_t row_sb[outputs_per_simd];
    for (uint row = 0; row < outputs_per_simd; ++row) {
        uint output_row = first_row + row;
        row_codes[row] = *(const device uint2*)(
            expert_weight + output_row * packed_row_bytes + lane * 8);
        row_sb[row] =
            (expert == 0 && output_row == 0 && lane == 1)
            ? down_scales[0]
            : expert_scales[output_row * scale_row_bytes + (lane >> 1)];
    }
    for (uint row = 0; row < outputs_per_simd; ++row) {
        result[row] = laguna_nvfp4_qdot_codes_16(
            row_codes[row],
            input_values,
            laguna_nvfp4_scale(row_sb[row]));
    }
    {
        const vec<float, 4> packed_rows = simd_sum(
            vec<float, 4>(result[0], result[1], result[2], result[3]));
        result[0] = packed_rows.x;
        result[1] = packed_rows.y;
        result[2] = packed_rows.z;
        result[3] = packed_rows.w;
    }

    threadgroup bfloat expert_outputs[
        experts_per_token * outputs_per_simd
    ];
    if (lane == 0) {
        for (uint row = 0; row < outputs_per_simd; ++row) {
            expert_outputs[
                expert_slot * outputs_per_simd + row
            ] = bfloat(result[row] * 4194304.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (expert_slot == 0 && lane < outputs_per_simd) {
        bfloat total = bfloat(0);
        for (uint slot = 0; slot < experts_per_token; ++slot) {
            bfloat route_weight = bfloat(router_weights[slot]);
            bfloat product = bfloat(
                expert_outputs[slot * outputs_per_simd + lane] *
                route_weight);
            total = bfloat(product + total);
        }
        routed[first_row + lane] = bfloat(total * bfloat(2.5f));
    }
}

// Full-attention QK RMSNorm + partial-RoPE with YaRN mscale, fused.
// (verbatim from lagunaFullQKNormYaRNKernel)
//   raw_queries [48*128] bf16, raw_keys [8*128] bf16
//   query_weight [128] bf16, key_weight [128] bf16 (norm weights)
//   angles [64] float32 (rotary_pairs cos + sin)
//   queries [48*128] bf16, keys [8*128] bf16
// Grid: 56 threadgroups (48 q + 8 k heads) x 32 threads.
kernel void laguna_full_qk_norm_yarn_bf16_128_v4(
    const device bfloat* raw_queries [[buffer(0)]],
    const device bfloat* raw_keys [[buffer(1)]],
    const device bfloat* query_weight [[buffer(2)]],
    const device bfloat* key_weight [[buffer(3)]],
    const device float* angles [[buffer(4)]],
    device bfloat* queries [[buffer(5)]],
    device bfloat* keys [[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint head_dim = 128;
    constexpr uint rotary_pairs = 32;
    constexpr uint query_heads = 48;
    constexpr float yarn_mscale = 1.3465735912322998f;

    const device bfloat* input;
    const device bfloat* weight;
    if (head < query_heads) {
        input = raw_queries + head * head_dim;
        weight = query_weight;
    } else {
        input = raw_keys + (head - query_heads) * head_dim;
        weight = key_weight;
    }

    uint base = lane * 4;
    thread bfloat normalized[4];
    float sum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        float value = float(input[base + i]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

    for (uint i = 0; i < 4; ++i) {
        normalized[i] =
            weight[base + i] *
            bfloat(float(input[base + i]) * inverse_rms);
    }

    thread float paired[4];
    for (uint i = 0; i < 4; ++i) {
        paired[i] = simd_shuffle(float(normalized[i]), lane ^ 8);
    }

    device bfloat* output =
        head < query_heads
        ? queries + head * head_dim
        : keys + (head - query_heads) * head_dim;
    if (lane < 8) {
        bfloat rounded_mscale = bfloat(yarn_mscale);
        for (uint i = 0; i < 4; ++i) {
            uint pair = base + i;
            float first =
                float(bfloat(normalized[i] * rounded_mscale));
            float second =
                float(bfloat(bfloat(paired[i]) * rounded_mscale));
            float cosine = angles[pair];
            float sine = angles[pair + rotary_pairs];
            output[pair] = bfloat(first * cosine - second * sine);
            output[pair + rotary_pairs] =
                bfloat(first * sine + second * cosine);
        }
    } else if (lane >= 16) {
        for (uint i = 0; i < 4; ++i) {
            output[base + i] = normalized[i];
        }
    }
}

// Sliding-attention QK RMSNorm + full RoPE, fused.
// (verbatim from lagunaSlidingQKNormRoPEKernel)
//   raw_queries [64*128] bf16, raw_keys [8*128] bf16
//   angles [128] float32 (rotary_pairs cos + sin, full 128-dim rotation)
// Grid: 72 threadgroups (64 q + 8 k heads) x 32 threads.
kernel void laguna_sliding_qk_norm_rope_bf16_128_v1(
    const device bfloat* raw_queries [[buffer(0)]],
    const device bfloat* raw_keys [[buffer(1)]],
    const device bfloat* query_weight [[buffer(2)]],
    const device bfloat* key_weight [[buffer(3)]],
    const device float* angles [[buffer(4)]],
    device bfloat* queries [[buffer(5)]],
    device bfloat* keys [[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint head_dim = 128;
    constexpr uint rotary_pairs = 64;
    constexpr uint query_heads = 64;

    const device bfloat* input;
    const device bfloat* weight;
    if (head < query_heads) {
        input = raw_queries + head * head_dim;
        weight = query_weight;
    } else {
        input = raw_keys + (head - query_heads) * head_dim;
        weight = key_weight;
    }

    uint base = lane * 4;
    thread bfloat normalized[4];
    float sum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        float value = float(input[base + i]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

    for (uint i = 0; i < 4; ++i) {
        normalized[i] =
            weight[base + i] *
            bfloat(float(input[base + i]) * inverse_rms);
    }

    thread float paired[4];
    for (uint i = 0; i < 4; ++i) {
        paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16);
    }

    device bfloat* output =
        head < query_heads
        ? queries + head * head_dim
        : keys + (head - query_heads) * head_dim;
    if (lane < 16) {
        for (uint i = 0; i < 4; ++i) {
            uint pair = base + i;
            float first = float(normalized[i]);
            float second = paired[i];
            float cosine = angles[pair];
            float sine = angles[pair + rotary_pairs];
            output[pair] = bfloat(first * cosine - second * sine);
            output[pair + rotary_pairs] =
                bfloat(first * sine + second * cosine);
        }
    }
}

// Tail NVFP4 header for the decode QKV family (resolved at the challenge's
// default flag config: DARKBLOOM_QKV_TAIL_FOLD / TAIL_NVFP4_SCALE_FOLD on).
static inline float laguna_tail_nvfp4_scale(uint8_t bits) {
    ushort raw = ushort(bits) << 7;
    return float(as_type<half>(raw));
}

static inline float laguna_tail_nvfp4_qdot(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale
) {
    float accum;
    const device uint2* wq = (const device uint2*)w;
    const uint2 codes = wq[0];
    for (int j = 0; j < 2; j++) {
        const uint32_t c = (j == 0) ? codes.x : codes.y;
        const uint32_t xe = c & 0x0F0F0F0Fu;
        const uint32_t ge = xe | (xe << 3);
        const uint32_t yo = c & 0xF0F0F0F0u;
        const uint32_t go = yo | (yo >> 3);
        const uint32_t p0 = (ge << 9) & 0x8E008E00u;
        const uint32_t p1 = (go << 8) & 0x8E008E00u;
        const uint32_t p2 = (ge << 1) & 0x8E008E00u;
        const uint32_t p3 = go & 0x8E008E00u;
        const float2 v04 = float2(as_type<half2>(p0));
        const float2 v15 = float2(as_type<half2>(p1));
        const float2 v26 = float2(as_type<half2>(p2));
        const float2 v37 = float2(as_type<half2>(p3));
        if (j == 0) {
            accum =
                (x_thread[8 * j] * v04.x +
                 x_thread[8 * j + 1] * v15.x +
                 x_thread[8 * j + 2] * v26.x +
                 x_thread[8 * j + 3] * v37.x);
        } else {
            accum +=
                (x_thread[8 * j] * v04.x +
                 x_thread[8 * j + 1] * v15.x +
                 x_thread[8 * j + 2] * v26.x +
                 x_thread[8 * j + 3] * v37.x);
        }
        accum +=
            (x_thread[8 * j + 4] * v04.y +
             x_thread[8 * j + 5] * v15.y +
             x_thread[8 * j + 6] * v26.y +
             x_thread[8 * j + 7] * v37.y);
    }
    return scale * accum;
}

// Decode QKV fused projection (R1, one row per simdgroup).
// (verbatim from lagunaDecodeNVFP4QKVR1Source(), scale-defer arm)
//   normalized   [2048] bf16
//   weight_codes [rows][1024] uint8 (rows = (heads + 2*nkv)*128)
//   weight_scales [rows][128] uint8 E4M3 group-16
//   projected    [rows] bf16
#define LAGUNA_QKV_R1_KERNEL(name, rows)                                        \
kernel void name(                                                               \
    const device bfloat* normalized [[buffer(0)]],                              \
    const device uint8_t* weight_codes [[buffer(1)]],                           \
    const device uint8_t* weight_scales [[buffer(2)]],                          \
    device bfloat* projected [[buffer(3)]],                                     \
    uint tile [[threadgroup_position_in_grid]],                                 \
    uint simd_gid [[simdgroup_index_in_threadgroup]],                           \
    uint simd_lid [[thread_index_in_simdgroup]])                                \
{                                                                               \
    constexpr uint axis_size = 2048;                                            \
    constexpr uint num_simdgroups = 2;                                          \
    constexpr uint values_per_thread = 16;                                      \
    constexpr uint block_size = 512;                                            \
    constexpr uint in_vec_size_w = axis_size / 2;                               \
    constexpr uint in_vec_size_g = axis_size / 16;                              \
                                                                                \
    uint out_row = tile * num_simdgroups + simd_gid;                            \
                                                                                \
    const device uint8_t* ws = (const device uint8_t*)weight_codes +            \
        out_row * in_vec_size_w + simd_lid * 8;                                 \
    const device uint8_t* sc = weight_scales +                                  \
        out_row * in_vec_size_g + simd_lid;                                     \
                                                                                \
    thread float x_thread[values_per_thread];                                   \
    thread float result = 0.0f;                                                 \
                                                                                \
    uint column = simd_lid * values_per_thread;                                 \
    for (uint k = 0; k < axis_size; k += block_size) {                          \
        for (uint i = 0; i < values_per_thread; ++i) {                          \
            x_thread[i] = float(normalized[column + i]);                        \
        }                                                                       \
        result += laguna_tail_nvfp4_qdot(                                       \
            ws, x_thread, laguna_tail_nvfp4_scale(sc[0]));                      \
        ws += block_size / 2;                                                   \
        sc += block_size / 16;                                                  \
        column += block_size;                                                   \
    }                                                                           \
                                                                                \
    result = simd_sum(result * 4194304.0f);                                     \
    if (simd_lid == 0) {                                                        \
        projected[out_row] = bfloat(result);                                    \
    }                                                                           \
}

LAGUNA_QKV_R1_KERNEL(
    laguna_decode_nvfp4_qkv_h48_r1_v1_se1_sd1, (48 + 2 * 8) * 128)
LAGUNA_QKV_R1_KERNEL(
    laguna_decode_nvfp4_qkv_h64_r1_v1_se1_sd1, (64 + 2 * 8) * 128)

// Gated affine o_proj, pre-activated gate variant
// (verbatim from lagunaGatedAffineOProjNVFP4Source(preActivatedGate: true),
// default flag config: sign-carry E4M3, seed elision, folded scale)
//   attention_output [in_vec] bf16
//   gate_values      [heads] bf16 (pre-activated per-head gate)
//   weight_codes     [2048][in_vec/8] uint32
//   weight_scales    [2048][in_vec/16] uint8
//   projected        [2048] bf16 = o_proj(attn * gate)
// Grid: 256 threadgroups x 64 threads (8 rows each).
#define LAGUNA_OPROJ_ACT_KERNEL(name, heads)                                   \
kernel void name(                                                               \
    const device bfloat* attention_output [[buffer(0)]],                        \
    const device bfloat* gate_values [[buffer(1)]],                            \
    const device uint8_t* weight_codes [[buffer(2)]],                          \
    const device uint8_t* weight_scales [[buffer(3)]],                         \
    device bfloat* projected [[buffer(4)]],                                    \
    uint tile [[threadgroup_position_in_grid]],                                \
    uint lid [[thread_position_in_threadgroup]],                               \
    uint simd_gid [[simdgroup_index_in_threadgroup]],                          \
    uint simd_lid [[thread_index_in_simdgroup]])                               \
{                                                                               \
    constexpr uint in_vec_size = heads * 128;                                   \
    constexpr uint out_vec_size = 2048;                                         \
    constexpr uint head_shift = 7;                                              \
    constexpr uint group_size = 16;                                             \
    constexpr uint values_per_thread = 16;                                      \
    constexpr uint codes_per_thread = values_per_thread / 8;                    \
    constexpr uint block_size = values_per_thread * 32;                         \
    constexpr uint results_per_simdgroup = 4;                                   \
    constexpr uint num_simdgroups = 2;                                          \
    constexpr uint in_vec_size_g = in_vec_size / group_size;                    \
                                                                                \
    uint out_row = tile * (num_simdgroups * results_per_simdgroup) +            \
        simd_gid * results_per_simdgroup;                                       \
    const device uint32_t* ws =                                                  \
        (const device uint32_t*)weight_codes +                                  \
        out_row * (in_vec_size / 8) + simd_lid * codes_per_thread;              \
    const device uint8_t* sc = weight_scales +                                  \
        out_row * in_vec_size_g + simd_lid;                                     \
    const device bfloat* xp = attention_output + simd_lid * values_per_thread;  \
                                                                                \
    thread float x_thread[values_per_thread];                                   \
    thread float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};      \
                                                                                \
    uint column = simd_lid * values_per_thread;                                 \
    for (uint k = 0; k < in_vec_size; k += block_size) {                        \
        float g = float(gate_values[column >> head_shift]);                     \
        for (uint i = 0; i < values_per_thread; ++i)                            \
            x_thread[i] = float(bfloat(float(xp[i]) * g));                      \
                                                                                \
        for (uint row = 0; row < results_per_simdgroup; ++row) {                \
            const device uint32_t* wl = ws + row * (in_vec_size / 8);           \
            uint8_t sbits = sc[row * in_vec_size_g];                            \
            ushort sraw = ushort(sbits) << 7;                                   \
            float scale = float(as_type<half>(sraw));                           \
            float accum;                                                        \
            for (uint j = 0; j < codes_per_thread; ++j) {                       \
                const uint c = wl[j];                                           \
                const uint xe = c & 0x0F0F0F0Fu;                                \
                const uint ge = xe | (xe << 3);                                 \
                const uint yo = c & 0xF0F0F0F0u;                                \
                const uint go = yo | (yo >> 3);                                 \
                const uint p0 = (ge << 9) & 0x8E008E00u;                        \
                const uint p1 = (go << 8) & 0x8E008E00u;                        \
                const uint p2 = (ge << 1) & 0x8E008E00u;                        \
                const uint p3 = go & 0x8E008E00u;                               \
                const float2 v04 = float2(as_type<half2>(p0));                  \
                const float2 v15 = float2(as_type<half2>(p1));                  \
                const float2 v26 = float2(as_type<half2>(p2));                  \
                const float2 v37 = float2(as_type<half2>(p3));                  \
                if (j == 0) {                                                   \
                    accum =                                                     \
                        (x_thread[8 * j] * v04.x +                              \
                         x_thread[8 * j + 1] * v15.x +                          \
                         x_thread[8 * j + 2] * v26.x +                          \
                         x_thread[8 * j + 3] * v37.x);                          \
                } else {                                                        \
                    accum +=                                                    \
                        (x_thread[8 * j] * v04.x +                              \
                         x_thread[8 * j + 1] * v15.x +                          \
                         x_thread[8 * j + 2] * v26.x +                          \
                         x_thread[8 * j + 3] * v37.x);                          \
                }                                                               \
                accum +=                                                        \
                    (x_thread[8 * j + 4] * v04.y +                              \
                     x_thread[8 * j + 5] * v15.y +                              \
                     x_thread[8 * j + 6] * v26.y +                              \
                     x_thread[8 * j + 7] * v37.y);                              \
            }                                                                   \
            result[row] += scale * accum;                                       \
        }                                                                       \
                                                                                \
        ws += block_size / 8;                                                   \
        sc += block_size / group_size;                                          \
        xp += block_size;                                                       \
        column += block_size;                                                   \
    }                                                                           \
                                                                                \
    for (uint row = 0; row < results_per_simdgroup; ++row) {                    \
        result[row] = simd_sum(result[row] * 4194304.0f);                       \
        if (simd_lid == 0) {                                                    \
            projected[out_row + row] = bfloat(result[row]);                     \
        }                                                                       \
    }                                                                           \
}

LAGUNA_OPROJ_ACT_KERNEL(
    laguna_oproj_act_h48_v1_sc1_se1, 48)
LAGUNA_OPROJ_ACT_KERNEL(
    laguna_oproj_act_h64_v1_sc1_se1, 64)

// Post-attention residual add + RMSNorm, fused.
// (verbatim from lagunaResidualRMSNormKernel)
//   residual [2048] bf16, branch [2048] bf16, weight [2048] bf16
//   summed [2048] bf16, normalized [2048] bf16
// Grid: `rows` threadgroups x 512 threads (16 simdgroups).
kernel void laguna_residual_rms_bf16_2048_v1(
    const device bfloat* residual [[buffer(0)]],
    const device bfloat* branch [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]],
    device bfloat* summed [[buffer(3)]],
    device bfloat* normalized [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    constexpr uint axis_size = 2048;
    constexpr uint n_reads = 4;
    constexpr uint simd_size = 32;

    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[simd_size];
    uint base = row * axis_size + lid * n_reads;

    thread bfloat values[n_reads];
    float acc = 0.0f;
    for (uint i = 0; i < n_reads; ++i) {
        bfloat value = bfloat(residual[base + i] + branch[base + i]);
        values[i] = value;
        summed[base + i] = value;
        float fv = float(value);
        acc += fv * fv;
    }

    acc = simd_sum(acc);
    if (simd_group == 0) {
        local_sums[simd_lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) {
        local_sums[simd_group] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        acc = simd_sum(local_sums[simd_lane]);
        if (simd_lane == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / 2048.0f + 1.0e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float laguna_inv_mean = local_inv_mean[0];

    for (uint i = 0; i < n_reads; ++i) {
        normalized[base + i] =
            weight[lid * n_reads + i] *
            bfloat(float(values[i]) * laguna_inv_mean);
    }
}

// Routed-expert fused gate/up NVFP4 QMV with in-kernel SwiGLU.
// (verbatim from lagunaRoutedSwiGLUQMVKernel)
//   input        [2048] bf16           — routed-expert input
//   fused_weight [E][1024][1024] uint8 — per-expert pair-interleaved
//                                        [gate 32; up 32] planes (E = 256)
//   fused_scales [E][1024][128]  uint8 — E4M3 group-16 scales
//   indices      [8] uint32            — top-8 routed expert ids
//   activated    [8][512] bf16         — per-slot silu(gate) * up
// Grid: 1024 groups (8 slots x 128 tiles) x (2 simdgroups x 2 rows).
kernel void laguna_routed_nvfp4_swiglu_qmv_bf16_v2(
    const device bfloat* input [[buffer(0)]],
    const device uint8_t* fused_weight [[buffer(1)]],
    const device uint8_t* fused_scales [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    device bfloat* activated [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint input_width = 2048;
    constexpr uint output_width = 512;
    constexpr uint fused_width = 1024;
    constexpr uint packed_row_bytes = 1024;
    constexpr uint scale_row_bytes = 128;
    constexpr uint packed_expert_bytes = fused_width * packed_row_bytes;
    constexpr uint scale_expert_bytes = fused_width * scale_row_bytes;
    constexpr uint block_width = 512;
    constexpr uint values_per_lane = 16;
    constexpr uint tiles_per_expert = 128;
    constexpr uint routed_experts = 8;

    uint expert_slot = group % routed_experts;
    uint tile = group / routed_experts;
    uint expert = uint(indices[expert_slot]);
    uint first_row = tile * 4 + simd_group * 2;

    const device uint8_t* expert_weight =
        (const device uint8_t*)fused_weight +
        expert * packed_expert_bytes;
    const device uint8_t* expert_scales =
        fused_scales + expert * scale_expert_bytes;

    thread float gate_result[2] = {0.0f, 0.0f};
    thread float up_result[2] = {0.0f, 0.0f};
    thread float input_values[values_per_lane];

    for (uint block = 0; block < input_width; block += block_width) {
        const device vec<bfloat, 4>* input_vectors =
            (const device vec<bfloat, 4>*)(
                input + block + lane * values_per_lane);
        for (uint i = 0; i < values_per_lane / 4; ++i) {
            const vec<bfloat, 4> values = input_vectors[i];
            input_values[4 * i] = values[0];
            input_values[4 * i + 1] = values[1];
            input_values[4 * i + 2] = values[2];
            input_values[4 * i + 3] = values[3];
        }

        for (uint row = 0; row < 2; ++row) {
            uint logical_row = first_row + row;
            uint pair_tile = logical_row / 32;
            uint gate_row = pair_tile * 64 + logical_row % 32;
            uint up_row = gate_row + 32;
            const device uint8_t* gate_weight =
                expert_weight + gate_row * packed_row_bytes +
                block / 2 + lane * 8;
            const device uint8_t* up_weight =
                expert_weight + up_row * packed_row_bytes +
                block / 2 + lane * 8;
            const device uint8_t* gate_scale =
                expert_scales + gate_row * scale_row_bytes +
                block / 16 + lane;
            const device uint8_t* up_scale =
                expert_scales + up_row * scale_row_bytes +
                block / 16 + lane;

            gate_result[row] += laguna_nvfp4_qdot_16(
                gate_weight,
                input_values,
                laguna_nvfp4_scale(gate_scale[0]));
            up_result[row] += laguna_nvfp4_qdot_16(
                up_weight,
                input_values,
                laguna_nvfp4_scale(up_scale[0]));
        }
    }

    for (uint row = 0; row < 2; ++row) {
        gate_result[row] = simd_sum(gate_result[row]);
        up_result[row] = simd_sum(up_result[row]);
        if (lane == 0) {
            bfloat gate = bfloat(gate_result[row] * 4194304.0f);
            bfloat up = bfloat(up_result[row] * 4194304.0f);
            bfloat exp_abs = metal::exp(metal::abs(gate));
            bfloat denominator = bfloat(1) + exp_abs;
            bfloat y = bfloat(1) / denominator;
            bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
            bfloat silu = bfloat(gate * sigmoid);
            activated[
                expert_slot * output_width + first_row + row
            ] = bfloat(silu * up);
        }
    }
}