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

// Batched (multi-row prefill) full-attention QK RMSNorm + partial-RoPE with
// the YaRN mscale. (verbatim from lagunaPrefillFullQKNormYaRNKernel; 4
// heads per threadgroup via the simdgroup index — the h1 twin below uses
// one head per threadgroup)
//   raw_queries [rows*48*128] bf16, raw_keys [rows*8*128] bf16
//   query_weight [128] bf16, key_weight [128] bf16
//   angles [atlas*64] float32, offsets [1] int32
//   queries [48*rows*128] bf16, keys [8*rows*128] bf16
// Grid: (14, rows) threadgroups of (128 = 4 simdgroups) threads.
kernel void laguna_prefill_full_qk_norm_yarn_bf16_128_v2(
    const device bfloat* raw_queries [[buffer(0)]],
    const device bfloat* raw_keys [[buffer(1)]],
    const device bfloat* query_weight [[buffer(2)]],
    const device bfloat* key_weight [[buffer(3)]],
    const device float* angles [[buffer(4)]],
    const device int32_t* offsets [[buffer(5)]],
    device bfloat* queries [[buffer(6)]],
    device bfloat* keys [[buffer(7)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]])
{
constexpr uint head_dim = 128;
constexpr uint rotary_pairs = 32;
constexpr uint query_heads = 48;
constexpr uint kv_heads = 8;
constexpr float yarn_mscale = 1.3465735912322998f;

uint t = threadgroup_position_in_grid.y;
uint length = threadgroups_per_grid.y;
uint head = threadgroup_position_in_grid.x * 4
    + simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

const device bfloat* input;
const device bfloat* weight;
device bfloat* output;
if (head < query_heads) {
    input = raw_queries + (t * query_heads + head) * head_dim;
    weight = query_weight;
    output = queries + (head * length + t) * head_dim;
} else {
    uint khead = head - query_heads;
    input = raw_keys + (t * kv_heads + khead) * head_dim;
    weight = key_weight;
    output = keys + (khead * length + t) * head_dim;
}

uint base = lane * 4;
thread bfloat normalized[4];
float sum = 0.0f;
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    float value = float(input[base + i]);
    sum += value * value;
}
sum = simd_sum(sum);
float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    normalized[i] =
        weight[base + i] *
        bfloat(float(input[base + i]) * inverse_rms);
}

thread float paired[4];
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    paired[i] = simd_shuffle(float(normalized[i]), lane ^ 8);
}

const device float* angle_row =
    angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
if (lane < 8) {
    bfloat rounded_mscale = bfloat(yarn_mscale);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        uint pair = base + i;
        float first =
            float(bfloat(normalized[i] * rounded_mscale));
        float second =
            float(bfloat(bfloat(paired[i]) * rounded_mscale));
        float cosine = angle_row[pair];
        float sine = angle_row[pair + rotary_pairs];
        output[pair] = bfloat(first * cosine - second * sine);
        output[pair + rotary_pairs] =
            bfloat(first * sine + second * cosine);
    }
} else if (lane >= 16) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        output[base + i] = normalized[i];
    }
}
}

// Batched full-attention QK-norm, one head per threadgroup (h1).
// (verbatim from lagunaPrefillFullQKNormYaRNH1Kernel)
// Grid: (56, rows) x (32 = 1 simdgroup) threads.
kernel void laguna_prefill_full_qk_norm_yarn_bf16_128_h1_v2(
    const device bfloat* raw_queries [[buffer(0)]],
    const device bfloat* raw_keys [[buffer(1)]],
    const device bfloat* query_weight [[buffer(2)]],
    const device bfloat* key_weight [[buffer(3)]],
    const device float* angles [[buffer(4)]],
    const device int32_t* offsets [[buffer(5)]],
    device bfloat* queries [[buffer(6)]],
    device bfloat* keys [[buffer(7)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]])
{
constexpr uint head_dim = 128;
constexpr uint rotary_pairs = 32;
constexpr uint query_heads = 48;
constexpr uint kv_heads = 8;
constexpr float yarn_mscale = 1.3465735912322998f;

uint t = threadgroup_position_in_grid.y;
uint length = threadgroups_per_grid.y;
uint head = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;

const device bfloat* input;
const device bfloat* weight;
device bfloat* output;
if (head < query_heads) {
    input = raw_queries + (t * query_heads + head) * head_dim;
    weight = query_weight;
    output = queries + (head * length + t) * head_dim;
} else {
    uint khead = head - query_heads;
    input = raw_keys + (t * kv_heads + khead) * head_dim;
    weight = key_weight;
    output = keys + (khead * length + t) * head_dim;
}

uint base = lane * 4;
thread bfloat normalized[4];
float sum = 0.0f;
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    float value = float(input[base + i]);
    sum += value * value;
}
sum = simd_sum(sum);
float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    normalized[i] =
        weight[base + i] *
        bfloat(float(input[base + i]) * inverse_rms);
}

thread float paired[4];
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    paired[i] = simd_shuffle(float(normalized[i]), lane ^ 8);
}

const device float* angle_row =
    angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
if (lane < 8) {
    bfloat rounded_mscale = bfloat(yarn_mscale);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        uint pair = base + i;
        float first =
            float(bfloat(normalized[i] * rounded_mscale));
        float second =
            float(bfloat(bfloat(paired[i]) * rounded_mscale));
        float cosine = angle_row[pair];
        float sine = angle_row[pair + rotary_pairs];
        output[pair] = bfloat(first * cosine - second * sine);
        output[pair + rotary_pairs] =
            bfloat(first * sine + second * cosine);
    }
} else if (lane >= 16) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        output[base + i] = normalized[i];
    }
}
}

// Batched (multi-row prefill) sliding-attention QK RMSNorm + full RoPE.
// (verbatim from lagunaPrefillSlidingQKNormRoPEKernel; 4 heads per tg)
//   raw_queries [rows*64*128] bf16, raw_keys [rows*8*128] bf16
//   angles [rows*128] float32, offsets [1] int32
//   queries [64*rows*128] bf16, keys [8*rows*128] bf16
// Grid: (18, rows) x (128) threads.
kernel void laguna_prefill_sliding_qk_norm_rope_bf16_128_v2(
    const device bfloat* raw_queries [[buffer(0)]],
    const device bfloat* raw_keys [[buffer(1)]],
    const device bfloat* query_weight [[buffer(2)]],
    const device bfloat* key_weight [[buffer(3)]],
    const device float* angles [[buffer(4)]],
    const device int32_t* offsets [[buffer(5)]],
    device bfloat* queries [[buffer(6)]],
    device bfloat* keys [[buffer(7)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]])
{
constexpr uint head_dim = 128;
constexpr uint rotary_pairs = 64;
constexpr uint query_heads = 64;
constexpr uint kv_heads = 8;

uint t = threadgroup_position_in_grid.y;
uint length = threadgroups_per_grid.y;
uint head = threadgroup_position_in_grid.x * 4
    + simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

const device bfloat* input;
const device bfloat* weight;
device bfloat* output;
if (head < query_heads) {
    input = raw_queries + (t * query_heads + head) * head_dim;
    weight = query_weight;
    output = queries + (head * length + t) * head_dim;
} else {
    uint khead = head - query_heads;
    input = raw_keys + (t * kv_heads + khead) * head_dim;
    weight = key_weight;
    output = keys + (khead * length + t) * head_dim;
}

uint base = lane * 4;
thread bfloat normalized[4];
float sum = 0.0f;
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    float value = float(input[base + i]);
    sum += value * value;
}
sum = simd_sum(sum);
float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    normalized[i] =
        weight[base + i] *
        bfloat(float(input[base + i]) * inverse_rms);
}

thread float paired[4];
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16);
}

const device float* angle_row =
    angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
if (lane < 16) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        uint pair = base + i;
        float first = float(normalized[i]);
        float second = paired[i];
        float cosine = angle_row[pair];
        float sine = angle_row[pair + rotary_pairs];
        output[pair] = bfloat(first * cosine - second * sine);
        output[pair + rotary_pairs] =
            bfloat(first * sine + second * cosine);
    }
}
}

// Batched sliding-attention QK-norm, one head per threadgroup (h1).
// (verbatim from lagunaPrefillSlidingQKNormRoPEH1Kernel)
// Grid: (72, rows) x (32) threads.
kernel void laguna_prefill_sliding_qk_norm_rope_bf16_128_h1_v2(
    const device bfloat* raw_queries [[buffer(0)]],
    const device bfloat* raw_keys [[buffer(1)]],
    const device bfloat* query_weight [[buffer(2)]],
    const device bfloat* key_weight [[buffer(3)]],
    const device float* angles [[buffer(4)]],
    const device int32_t* offsets [[buffer(5)]],
    device bfloat* queries [[buffer(6)]],
    device bfloat* keys [[buffer(7)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]])
{
constexpr uint head_dim = 128;
constexpr uint rotary_pairs = 64;
constexpr uint query_heads = 64;
constexpr uint kv_heads = 8;

uint t = threadgroup_position_in_grid.y;
uint length = threadgroups_per_grid.y;
uint head = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;

const device bfloat* input;
const device bfloat* weight;
device bfloat* output;
if (head < query_heads) {
    input = raw_queries + (t * query_heads + head) * head_dim;
    weight = query_weight;
    output = queries + (head * length + t) * head_dim;
} else {
    uint khead = head - query_heads;
    input = raw_keys + (t * kv_heads + khead) * head_dim;
    weight = key_weight;
    output = keys + (khead * length + t) * head_dim;
}

uint base = lane * 4;
thread bfloat normalized[4];
float sum = 0.0f;
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    float value = float(input[base + i]);
    sum += value * value;
}
sum = simd_sum(sum);
float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    normalized[i] =
        weight[base + i] *
        bfloat(float(input[base + i]) * inverse_rms);
}

thread float paired[4];
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16);
}

const device float* angle_row =
    angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
if (lane < 16) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        uint pair = base + i;
        float first = float(normalized[i]);
        float second = paired[i];
        float cosine = angle_row[pair];
        float sine = angle_row[pair + rotary_pairs];
        output[pair] = bfloat(first * cosine - second * sine);
        output[pair + rotary_pairs] =
            bfloat(first * sine + second * cosine);
    }
}
}

// Tail NVFP4 header for the decode QKV family (resolved at the challenge's
// default config: DARKBLOOM_QKV_TAIL_FOLD / TAIL_NVFP4_SCALE_FOLD on).
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

// Decode router top-8: 256-lane bitonic-sort tournament over the router
// logits (with the e-score correction bias), emitting the top-8 indices and
// scores. (verbatim from lagunaDecodeRouterTop8KernelSource)
//   logits [256] bf16, correction_bias [256] fp32
//   router_indices [8] uint32, router_scores [8] bf16
// Grid: 1 threadgroup x 256 threads.
METAL_FUNC bool laguna_router_key_before(
    float a, uint a_index, float b, uint b_index) {
    bool a_nan = metal::isnan(a);
    bool b_nan = metal::isnan(b);
    if (a_nan | b_nan) {
        if (a_nan != b_nan) {
            return !a_nan;
        }
        return a_index < b_index;
    }
    if (a < b) {
        return true;
    }
    if (b < a) {
        return false;
    }
    return a_index < b_index;
}

// Ordinal-payload helpers: `laguna_router_key_ordinal` canonicalizes both
// signed zeros and every NaN before applying the usual monotone IEEE-754 bit
// transform; `laguna_router_ordinal_before` compares two ordinals unsigned
// plus the original expert-index tie break — exactly
// `laguna_router_key_before` on the source float key. (verbatim from
// lagunaDecodeRouterOrdinalHeader.)
METAL_FUNC uint laguna_router_key_ordinal(float key) {
    uint bits = as_type<uint>(key);
    uint magnitude = bits & 0x7FFFFFFFu;
    if (magnitude > 0x7F800000u) {
        return 0xFFFFFFFFu;
    }
    if (magnitude == 0u) {
        return 0x80000000u;
    }
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

METAL_FUNC bool laguna_router_ordinal_before(
    uint a, uint a_index, uint b, uint b_index) {
    if (a < b) {
        return true;
    }
    if (b < a) {
        return false;
    }
    return a_index < b_index;
}

#define LAGUNA_ROUTER_TOP8_KERNEL(name, normalizing)                          \
kernel void name(                                                             \
    const device bfloat* logits [[buffer(0)]],                                \
    const device float* correction_bias [[buffer(1)]],                        \
    device uint32_t* router_indices [[buffer(2)]],                            \
    device bfloat* router_scores [[buffer(3)]],                               \
    uint lane [[thread_position_in_threadgroup]])                             \
{                                                                             \
    threadgroup float xchg_keys[256];                                         \
    threadgroup uint xchg_indices[256];                                       \
    threadgroup float xchg_scores[256];                                       \
                                                                              \
    float x = float(logits[lane]);                                            \
    float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));                      \
    float my_score = x < 0.0f ? y : 1.0f - y;                                 \
    float my_key = -(my_score + float(correction_bias[lane]));                \
    uint my_index = lane;                                                     \
                                                                              \
    for (uint sequence = 2; sequence <= 256; sequence <<= 1) {                \
        for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {         \
            float other_key;                                                  \
            uint other_index;                                                 \
            float other_score;                                                \
            if (stride < 32) {                                                \
                other_key = simd_shuffle_xor(my_key, ushort(stride));         \
                other_index = simd_shuffle_xor(my_index, ushort(stride));     \
                other_score = simd_shuffle_xor(my_score, ushort(stride));     \
            } else {                                                          \
                xchg_keys[lane] = my_key;                                     \
                xchg_indices[lane] = my_index;                                \
                xchg_scores[lane] = my_score;                                 \
                threadgroup_barrier(mem_flags::mem_threadgroup);              \
                uint partner = lane ^ stride;                                 \
                other_key = xchg_keys[partner];                               \
                other_index = xchg_indices[partner];                          \
                other_score = xchg_scores[partner];                           \
                threadgroup_barrier(mem_flags::mem_threadgroup);              \
            }                                                                 \
                                                                              \
            bool is_lower = (lane & stride) == 0;                             \
            float a_key = is_lower ? my_key : other_key;                      \
            uint a_index = is_lower ? my_index : other_index;                 \
            float a_score = is_lower ? my_score : other_score;                \
            float b_key = is_lower ? other_key : my_key;                      \
            uint b_index = is_lower ? other_index : my_index;                 \
            float b_score = is_lower ? other_score : my_score;                \
                                                                              \
            bool lower_wants_better = (lane & sequence) == 0;                 \
            bool b_before_a = laguna_router_key_before(                       \
                b_key, b_index, a_key, a_index);                              \
            bool a_before_b = laguna_router_key_before(                       \
                a_key, a_index, b_key, b_index);                              \
            bool swap = lower_wants_better ? b_before_a : a_before_b;         \
            if (swap) {                                                       \
                my_key = is_lower ? b_key : a_key;                            \
                my_index = is_lower ? b_index : a_index;                      \
                my_score = is_lower ? b_score : a_score;                      \
            }                                                                 \
        }                                                                     \
    }                                                                         \
                                                                              \
    if (normalizing) {                                                        \
        float total = 0.0f;                                                   \
        for (uint i = 0; i < 8; ++i) {                                        \
            total = simd_shuffle(my_score, ushort(i)) + total;                \
        }                                                                     \
        if (lane < 8) {                                                       \
            router_indices[lane] = my_index;                                  \
            router_scores[lane] = bfloat(my_score / total);                   \
        }                                                                     \
    } else {                                                                  \
        if (lane < 8) {                                                       \
            router_indices[lane] = my_index;                                  \
            router_scores[lane] = bfloat(my_score);                           \
        }                                                                     \
    }                                                                         \
}

LAGUNA_ROUTER_TOP8_KERNEL(laguna_decode_router_top8_v3, 0)
LAGUNA_ROUTER_TOP8_KERNEL(laguna_decode_router_top8_norm_v2, 1)

// Decode router top-8, ordinal payload: same 256-lane bitonic network and
// stage/stride/pair-role geometry as the accepted float-payload kernel
// above, but the live payload is `(uint ordinal, uint index)` only, sorted
// by `laguna_router_ordinal_before`. (verbatim from
// lagunaDecodeRouterOrdinalKernelSource; the original per-lane sigmoid is
// always materialized in the score table — the non-table arm recomputes the
// winner sigmoid, byte-identical to the table read.)
//   logits [256] bf16, correction_bias [256] fp32
//   router_indices [8] uint32, router_scores [8] bf16
// Grid: 1 threadgroup x 256 threads.
#define LAGUNA_ROUTER_TOP8_ORDINAL_KERNEL(name, normalizing, table)          \
kernel void name(                                                            \
    const device bfloat* logits [[buffer(0)]],                               \
    const device float* correction_bias [[buffer(1)]],                       \
    device uint32_t* router_indices [[buffer(2)]],                           \
    device bfloat* router_scores [[buffer(3)]],                              \
    uint lane [[thread_position_in_threadgroup]])                            \
{                                                                            \
    threadgroup uint xchg_ordinals[256];                                     \
    threadgroup uint xchg_indices[256];                                      \
    threadgroup float original_scores[256];                                  \
                                                                             \
    float x = float(logits[lane]);                                           \
    float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));                     \
    float score = x < 0.0f ? y : 1.0f - y;                                   \
    original_scores[lane] = score;                                           \
    float key = -(score + float(correction_bias[lane]));                     \
    uint my_ordinal = laguna_router_key_ordinal(key);                        \
    uint my_index = lane;                                                    \
                                                                             \
    for (uint sequence = 2; sequence <= 256; sequence <<= 1) {               \
        for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {        \
            uint other_ordinal;                                              \
            uint other_index;                                                \
            if (stride < 32) {                                               \
                other_ordinal = simd_shuffle_xor(my_ordinal, ushort(stride)); \
                other_index = simd_shuffle_xor(my_index, ushort(stride));    \
            } else {                                                         \
                xchg_ordinals[lane] = my_ordinal;                            \
                xchg_indices[lane] = my_index;                               \
                threadgroup_barrier(mem_flags::mem_threadgroup);             \
                uint partner = lane ^ stride;                                \
                other_ordinal = xchg_ordinals[partner];                      \
                other_index = xchg_indices[partner];                         \
                threadgroup_barrier(mem_flags::mem_threadgroup);             \
            }                                                                \
                                                                             \
            bool is_lower = (lane & stride) == 0;                            \
            bool lower_wants_better = (lane & sequence) == 0;                \
            bool want_better = lower_wants_better == is_lower;               \
            bool other_before_my = laguna_router_ordinal_before(             \
                other_ordinal, other_index, my_ordinal, my_index);           \
            bool take_other = want_better ? other_before_my : !other_before_my; \
            if (take_other) {                                                \
                my_ordinal = other_ordinal;                                  \
                my_index = other_index;                                      \
            }                                                                \
        }                                                                    \
    }                                                                        \
                                                                             \
    if (normalizing) {                                                       \
        float my_score = 0.0f;                                               \
        if (lane < 8) {                                                      \
            if (table) {                                                     \
                my_score = original_scores[my_index];                        \
            } else {                                                         \
                float winner_x = float(logits[my_index]);                    \
                float winner_y = 1.0f / (1.0f + metal::exp(metal::abs(winner_x))); \
                my_score = winner_x < 0.0f ? winner_y : 1.0f - winner_y;     \
            }                                                                \
        }                                                                    \
        float total = 0.0f;                                                  \
        for (uint i = 0; i < 8; ++i) {                                       \
            total = simd_shuffle(my_score, ushort(i)) + total;                \
        }                                                                    \
        if (lane < 8) {                                                      \
            router_indices[lane] = my_index;                                 \
            router_scores[lane] = bfloat(my_score / total);                  \
        }                                                                    \
    } else {                                                                 \
        if (lane < 8) {                                                      \
            float my_score = 0.0f;                                           \
            if (table) {                                                     \
                my_score = original_scores[my_index];                        \
            } else {                                                         \
                float winner_x = float(logits[my_index]);                    \
                float winner_y = 1.0f / (1.0f + metal::exp(metal::abs(winner_x))); \
                my_score = winner_x < 0.0f ? winner_y : 1.0f - winner_y;     \
            }                                                                \
            router_indices[lane] = my_index;                                 \
            router_scores[lane] = bfloat(my_score);                          \
        }                                                                    \
    }                                                                        \
}                                                                            \

LAGUNA_ROUTER_TOP8_ORDINAL_KERNEL(laguna_decode_router_top8_ordinal_v1, 0, 0)
LAGUNA_ROUTER_TOP8_ORDINAL_KERNEL(laguna_decode_router_top8_ordinal_norm_v1, 1, 0)
LAGUNA_ROUTER_TOP8_ORDINAL_KERNEL(laguna_decode_router_top8_ordinal_table_v1, 0, 1)
LAGUNA_ROUTER_TOP8_ORDINAL_KERNEL(laguna_decode_router_top8_ordinal_table_norm_v1, 1, 1)

// Fused sliding-attention decode (steady ring regime).
// (verbatim from lagunaSlidingFusedAttentionKernel; T_LOAD_K/T_LOAD_V/
// LAGUNA_RESCALE macros inline)
//   raw_queries [64*128] bf16, raw_keys [8*128] bf16, raw_values [8*128] bf16
//   query_weight [128] bf16, key_weight [128] bf16, angles [128] fp32
//   k_cache [8][512][128] bf16, v_cache [8][512][128] bf16
//   params [1] uint32 (write idx), scale_arr [1] fp32
//   attended [64*128] bf16
// Grid: 32 threadgroups (2 heads each) x 1024 threads.
kernel void laguna_sliding_fused_attn_ring_v1(
    const device bfloat* raw_queries [[buffer(0)]],
    const device bfloat* raw_keys [[buffer(1)]],
    const device bfloat* raw_values [[buffer(2)]],
    const device bfloat* query_weight [[buffer(3)]],
    const device bfloat* key_weight [[buffer(4)]],
    const device float* angles [[buffer(5)]],
    const device bfloat* k_cache [[buffer(6)]],
    const device bfloat* v_cache [[buffer(7)]],
    const device uint32_t* params [[buffer(8)]],
    const device float* scale_arr [[buffer(9)]],
    device bfloat* attended [[buffer(10)]],
    uint pair_tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint head_dim = 128;
    constexpr uint window = 512;
    constexpr uint gqa = 8;
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int BDP = BD + 1;
    constexpr int qk_per_thread = 4;
    constexpr int v_per_thread = 4;
    constexpr uint rotary_pairs = 64;
    constexpr int N = 512;

    typedef float U;

    uint head0 = pair_tg * 2;
    uint head1 = head0 + 1;
    uint kv_head = head0 / gqa;
    uint widx = params[0];
    float scale = scale_arr[0];

    threadgroup bfloat tg_q0[head_dim];
    threadgroup bfloat tg_q1[head_dim];
    threadgroup bfloat tg_k[head_dim];
    threadgroup bfloat tg_v[head_dim];

    if (sg < 3) {
        const device bfloat* input =
            sg == 0 ? raw_queries + head0 * head_dim
            : sg == 1 ? raw_queries + head1 * head_dim
                      : raw_keys + kv_head * head_dim;
        const device bfloat* weight =
            sg == 2 ? key_weight : query_weight;
        threadgroup bfloat* outrow =
            sg == 0 ? tg_q0 : sg == 1 ? tg_q1 : tg_k;

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
        if (lane < 16) {
            for (uint i = 0; i < 4; ++i) {
                uint pair = base + i;
                float first = float(normalized[i]);
                float second = paired[i];
                float cosine = angles[pair];
                float sine = angles[pair + rotary_pairs];
                outrow[pair] = bfloat(first * cosine - second * sine);
                outrow[pair + rotary_pairs] =
                    bfloat(first * sine + second * cosine);
            }
        }
    } else if (sg == 3) {
        const device bfloat* vin = raw_values + kv_head * head_dim;
        for (uint i = lane; i < head_dim; i += 32) {
            tg_v[i] = vin[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((head0 % gqa) == 0 && sg == 0) {
        device bfloat* kc = (device bfloat*)k_cache +
            (size_t)kv_head * (window * head_dim) +
            (size_t)widx * head_dim;
        device bfloat* vc = (device bfloat*)v_cache +
            (size_t)kv_head * (window * head_dim) +
            (size_t)widx * head_dim;
        for (uint i = lane; i < head_dim; i += 32) {
            kc[i] = tg_k[i];
            vc[i] = tg_v[i];
        }
    }

    threadgroup U outputs[4 * BN * BDP];
    threadgroup U max_scores[2 * BN];
    threadgroup U sum_exp_scores[2 * BN];

    const device bfloat* pair_keys = k_cache +
        (size_t)kv_head * (window * head_dim) +
        (size_t)sg * head_dim + lane * qk_per_thread;
    const device bfloat* pair_values = v_cache +
        (size_t)kv_head * (window * head_dim) +
        (size_t)sg * head_dim + lane * v_per_thread;
    const int inner_k_stride = BN * int(head_dim);
    const int inner_v_stride = BN * int(head_dim);

    thread U pair_q0[qk_per_thread];
    thread U pair_q1[qk_per_thread];
    thread U pair_o0[v_per_thread];
    thread U pair_o1[v_per_thread];

    for (int j = 0; j < qk_per_thread; ++j) {
        pair_q0[j] =
            static_cast<U>(scale) * tg_q0[lane * qk_per_thread + j];
        pair_q1[j] =
            static_cast<U>(scale) * tg_q1[lane * qk_per_thread + j];
    }
    for (int j = 0; j < v_per_thread; ++j) {
        pair_o0[j] = 0;
        pair_o1[j] = 0;
    }

    U pair_max0 = metal::numeric_limits<U>::lowest();
    U pair_max1 = metal::numeric_limits<U>::lowest();
    U pair_sum0 = 0;
    U pair_sum1 = 0;

    int i = sg;
    for (; i + BN < N; i += 2 * BN) {
        const device bfloat* pipe_keys_b = pair_keys + inner_k_stride;
        const device bfloat* pipe_values_b = pair_values + inner_v_stride;
        const bool sub_a = uint(i) == widx;
        const bool sub_b = uint(i + BN) == widx;
        U pipe_ka[4];
        U pipe_kb[4];
        // T_LOAD_K
        if (sub_a) {
            pipe_ka[0] = tg_k[lane * qk_per_thread + 0];
            pipe_ka[1] = tg_k[lane * qk_per_thread + 1];
            pipe_ka[2] = tg_k[lane * qk_per_thread + 2];
            pipe_ka[3] = tg_k[lane * qk_per_thread + 3];
        } else {
            const vec<bfloat, 4> v_ =
                *reinterpret_cast<const device vec<bfloat, 4>*>(pair_keys);
            pipe_ka[0] = v_.x; pipe_ka[1] = v_.y;
            pipe_ka[2] = v_.z; pipe_ka[3] = v_.w;
        }
        if (sub_b) {
            pipe_kb[0] = tg_k[lane * qk_per_thread + 0];
            pipe_kb[1] = tg_k[lane * qk_per_thread + 1];
            pipe_kb[2] = tg_k[lane * qk_per_thread + 2];
            pipe_kb[3] = tg_k[lane * qk_per_thread + 3];
        } else {
            const vec<bfloat, 4> v_ =
                *reinterpret_cast<const device vec<bfloat, 4>*>(pipe_keys_b);
            pipe_kb[0] = v_.x; pipe_kb[1] = v_.y;
            pipe_kb[2] = v_.z; pipe_kb[3] = v_.w;
        }
        bfloat pipe_va0, pipe_va1, pipe_va2, pipe_va3;
        bfloat pipe_vb0, pipe_vb1, pipe_vb2, pipe_vb3;
        // T_LOAD_V
        if (sub_a) {
            pipe_va0 = tg_v[lane * v_per_thread + 0];
            pipe_va1 = tg_v[lane * v_per_thread + 1];
            pipe_va2 = tg_v[lane * v_per_thread + 2];
            pipe_va3 = tg_v[lane * v_per_thread + 3];
        } else {
            const vec<bfloat, 4> v_ =
                *reinterpret_cast<const device vec<bfloat, 4>*>(pair_values);
            pipe_va0 = v_.x; pipe_va1 = v_.y;
            pipe_va2 = v_.z; pipe_va3 = v_.w;
        }
        if (sub_b) {
            pipe_vb0 = tg_v[lane * v_per_thread + 0];
            pipe_vb1 = tg_v[lane * v_per_thread + 1];
            pipe_vb2 = tg_v[lane * v_per_thread + 2];
            pipe_vb3 = tg_v[lane * v_per_thread + 3];
        } else {
            const vec<bfloat, 4> v_ =
                *reinterpret_cast<const device vec<bfloat, 4>*>(pipe_values_b);
            pipe_vb0 = v_.x; pipe_vb1 = v_.y;
            pipe_vb2 = v_.z; pipe_vb3 = v_.w;
        }

        U pair_score0 = 0;
        U pair_score1 = 0;
        pair_score0 += pair_q0[0] * pipe_ka[0];
        pair_score1 += pair_q1[0] * pipe_ka[0];
        pair_score0 += pair_q0[1] * pipe_ka[1];
        pair_score1 += pair_q1[1] * pipe_ka[1];
        pair_score0 += pair_q0[2] * pipe_ka[2];
        pair_score1 += pair_q1[2] * pipe_ka[2];
        pair_score0 += pair_q0[3] * pipe_ka[3];
        pair_score1 += pair_q1[3] * pipe_ka[3];
        pair_score0 = simd_sum(pair_score0);
        pair_score1 = simd_sum(pair_score1);

        U pair_new_max0 = metal::max(pair_max0, pair_score0);
        U pair_new_max1 = metal::max(pair_max1, pair_score1);
        U pair_factor0;
        U pair_factor1;
        // LAGUNA_RESCALE
        {
            const float db_delta_ = (pair_max0 - pair_new_max0);
            pair_factor0 = (as_type<uint>(db_delta_) == 0u)
                ? float(1.0f) : metal::fast::exp(db_delta_);
        }
        {
            const float db_delta_ = (pair_max1 - pair_new_max1);
            pair_factor1 = (as_type<uint>(db_delta_) == 0u)
                ? float(1.0f) : metal::fast::exp(db_delta_);
        }
        U pair_exp0 = metal::fast::exp(pair_score0 - pair_new_max0);
        U pair_exp1 = metal::fast::exp(pair_score1 - pair_new_max1);

        pair_max0 = pair_new_max0;
        pair_max1 = pair_new_max1;
        pair_sum0 = pair_sum0 * pair_factor0 + pair_exp0;
        pair_sum1 = pair_sum1 * pair_factor1 + pair_exp1;

        pair_o0[0] = pair_o0[0] * pair_factor0 + pair_exp0 * pipe_va0;
        pair_o1[0] = pair_o1[0] * pair_factor1 + pair_exp1 * pipe_va0;
        pair_o0[1] = pair_o0[1] * pair_factor0 + pair_exp0 * pipe_va1;
        pair_o1[1] = pair_o1[1] * pair_factor1 + pair_exp1 * pipe_va1;
        pair_o0[2] = pair_o0[2] * pair_factor0 + pair_exp0 * pipe_va2;
        pair_o1[2] = pair_o1[2] * pair_factor1 + pair_exp1 * pipe_va2;
        pair_o0[3] = pair_o0[3] * pair_factor0 + pair_exp0 * pipe_va3;
        pair_o1[3] = pair_o1[3] * pair_factor1 + pair_exp1 * pipe_va3;

        U pipeb_score0 = 0;
        U pipeb_score1 = 0;
        pipeb_score0 += pair_q0[0] * pipe_kb[0];
        pipeb_score1 += pair_q1[0] * pipe_kb[0];
        pipeb_score0 += pair_q0[1] * pipe_kb[1];
        pipeb_score1 += pair_q1[1] * pipe_kb[1];
        pipeb_score0 += pair_q0[2] * pipe_kb[2];
        pipeb_score1 += pair_q1[2] * pipe_kb[2];
        pipeb_score0 += pair_q0[3] * pipe_kb[3];
        pipeb_score1 += pair_q1[3] * pipe_kb[3];
        pipeb_score0 = simd_sum(pipeb_score0);
        pipeb_score1 = simd_sum(pipeb_score1);

        U pipeb_new_max0 = metal::max(pair_max0, pipeb_score0);
        U pipeb_new_max1 = metal::max(pair_max1, pipeb_score1);
        U pipeb_factor0;
        U pipeb_factor1;
        {
            const float db_delta_ = (pair_max0 - pipeb_new_max0);
            pipeb_factor0 = (as_type<uint>(db_delta_) == 0u)
                ? float(1.0f) : metal::fast::exp(db_delta_);
        }
        {
            const float db_delta_ = (pair_max1 - pipeb_new_max1);
            pipeb_factor1 = (as_type<uint>(db_delta_) == 0u)
                ? float(1.0f) : metal::fast::exp(db_delta_);
        }
        U pipeb_exp0 = metal::fast::exp(pipeb_score0 - pipeb_new_max0);
        U pipeb_exp1 = metal::fast::exp(pipeb_score1 - pipeb_new_max1);

        pair_max0 = pipeb_new_max0;
        pair_max1 = pipeb_new_max1;
        pair_sum0 = pair_sum0 * pipeb_factor0 + pipeb_exp0;
        pair_sum1 = pair_sum1 * pipeb_factor1 + pipeb_exp1;

        pair_o0[0] = pair_o0[0] * pipeb_factor0 + pipeb_exp0 * pipe_vb0;
        pair_o1[0] = pair_o1[0] * pipeb_factor1 + pipeb_exp1 * pipe_vb0;
        pair_o0[1] = pair_o0[1] * pipeb_factor0 + pipeb_exp0 * pipe_vb1;
        pair_o1[1] = pair_o1[1] * pipeb_factor1 + pipeb_exp1 * pipe_vb1;
        pair_o0[2] = pair_o0[2] * pipeb_factor0 + pipeb_exp0 * pipe_vb2;
        pair_o1[2] = pair_o1[2] * pipeb_factor1 + pipeb_exp1 * pipe_vb2;
        pair_o0[3] = pair_o0[3] * pipeb_factor0 + pipeb_exp0 * pipe_vb3;
        pair_o1[3] = pair_o1[3] * pipeb_factor1 + pipeb_exp1 * pipe_vb3;

        pair_keys += 2 * inner_k_stride;
        pair_values += 2 * inner_v_stride;
    }

    constexpr int pair_planes = 2;
    constexpr int pair_plane_size = BN * BDP;
    if (lane == 0) {
        max_scores[sg] = pair_max0;
        max_scores[BN + sg] = pair_max1;
        sum_exp_scores[sg] = pair_sum0;
        sum_exp_scores[BN + sg] = pair_sum1;
    }
    for (int p = 0; p < pair_planes; ++p) {
        outputs[p * pair_plane_size + lane * BDP + sg] = pair_o0[p];
        outputs[
            (pair_planes + p) * pair_plane_size + lane * BDP + sg] =
            pair_o1[p];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    pair_max0 = max_scores[lane];
    pair_max1 = max_scores[BN + lane];
    U pair_global_max0 = simd_max(pair_max0);
    U pair_global_max1 = simd_max(pair_max1);
    U pair_global_factor0 = metal::fast::exp(pair_max0 - pair_global_max0);
    U pair_global_factor1 = metal::fast::exp(pair_max1 - pair_global_max1);
    pair_sum0 = simd_sum(sum_exp_scores[lane] * pair_global_factor0);
    pair_sum1 = simd_sum(sum_exp_scores[BN + lane] * pair_global_factor1);

    for (int p = 0; p < pair_planes; ++p) {
        U acc0 = simd_sum(
            outputs[p * pair_plane_size + sg * BDP + lane] *
            pair_global_factor0);
        U acc1 = simd_sum(
            outputs[
                (pair_planes + p) * pair_plane_size + sg * BDP + lane] *
            pair_global_factor1);
        pair_o0[p] = pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
        pair_o1[p] = pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = 0; p < pair_planes; ++p) {
        outputs[p * pair_plane_size + lane * BDP + sg] =
            pair_o0[pair_planes + p];
        outputs[
            (pair_planes + p) * pair_plane_size + lane * BDP + sg] =
            pair_o1[pair_planes + p];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = 0; p < pair_planes; ++p) {
        U acc0 = simd_sum(
            outputs[p * pair_plane_size + sg * BDP + lane] *
            pair_global_factor0);
        U acc1 = simd_sum(
            outputs[
                (pair_planes + p) * pair_plane_size + sg * BDP + lane] *
            pair_global_factor1);
        pair_o0[pair_planes + p] =
            pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
        pair_o1[pair_planes + p] =
            pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
    }

    if (lane == 0) {
        device bfloat* pair_out0 =
            attended + head0 * head_dim + sg * v_per_thread;
        device bfloat* pair_out1 =
            attended + head1 * head_dim + sg * v_per_thread;
        for (int p = 0; p < v_per_thread; ++p) {
            pair_out0[p] = static_cast<bfloat>(pair_o0[p]);
            pair_out1[p] = static_cast<bfloat>(pair_o1[p]);
        }
    }
}

// Residual + RMSNorm + MoE router GEMV fused (rowsPerGroup 8, precomputed
// ordinal keys). (verbatim from lagunaResidualRMSNormRouterSource(8) with
// the DARKBLOOM_ROUTER_PRECOMPUTED_KEYS arm)
//   residual [2048] bf16, branch [2048] bf16, weight [2048] bf16
//   router_weight [256][2048] bf16, correction_bias [256] fp32
//   summed [2048], normalized [2048], router_logits [256], router_keys [256]
// Grid: 32 tiles x 512 threads (16 simdgroups; 8 active).

kernel void laguna_residual_rms_router_bf16_2048_rpg8(
    const device bfloat* residual [[buffer(0)]],
    const device bfloat* branch [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]],
    const device bfloat* router_weight [[buffer(3)]],
    const device float* correction_bias [[buffer(4)]],
    device bfloat* summed [[buffer(5)]],
    device bfloat* normalized [[buffer(6)]],
    device bfloat* router_logits [[buffer(7)]],
    device uint32_t* router_keys [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    constexpr uint axis_size = 2048;
    constexpr uint n_reads = 4;
    constexpr uint simd_size = 32;
    constexpr uint rows_per_group = 8;
    constexpr uint rows_per_thread = 1;
    constexpr uint active_simd_groups = 8;
    constexpr uint block_width = 128;
    constexpr uint router_blocks = axis_size / block_width;

    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[simd_size];
    threadgroup bfloat normalized_row[axis_size];
    uint base = lid * n_reads;

    thread bfloat values[n_reads];
    float acc = 0.0f;
    for (uint i = 0; i < n_reads; ++i) {
        bfloat value = bfloat(residual[base + i] + branch[base + i]);
        values[i] = value;
        if (tile == 0) {
            summed[base + i] = value;
        }
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
        bfloat value =
            weight[base + i] *
            bfloat(float(values[i]) * laguna_inv_mean);
        normalized_row[base + i] = value;
        if (tile == 0) {
            normalized[base + i] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group < active_simd_groups) {
        uint router_row = tile * rows_per_group + simd_group * rows_per_thread;
        thread float router_result[rows_per_thread] = {0.0f};
        uint column = simd_lane * n_reads;
        for (uint block = 0; block < router_blocks; block += 4) {
            vec<bfloat, 4> rw[4];
            for (uint u = 0; u < 4; ++u) {
                const device vec<bfloat, 4>* row_values =
                    (const device vec<bfloat, 4>*)(
                        router_weight + router_row * axis_size +
                            column + u * block_width);
                rw[u] = row_values[0];
            }
            for (uint u = 0; u < 4; ++u) {
                uint column_u = column + u * block_width;
                for (uint i = 0; i < n_reads; ++i) {
                    router_result[0] += float(rw[u][i]) *
                        float(normalized_row[column_u + i]);
                }
            }
            column += 4 * block_width;
        }

        for (uint r = 0; r < rows_per_thread; ++r) {
            for (ushort delta = 16; delta >= 1; delta >>= 1) {
                router_result[r] +=
                    metal::simd_shuffle_down(router_result[r], delta);
            }
        }
        if (simd_lane == 0) {
            for (uint r = 0; r < rows_per_thread; ++r) {
                bfloat logit = bfloat(router_result[r]);
                router_logits[router_row + r] = logit;
                float x = float(logit);
                float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
                float score = x < 0.0f ? y : 1.0f - y;
                router_keys[router_row + r] = laguna_router_key_ordinal(
                    -(score + float(correction_bias[router_row + r])));
            }
        }
    }
}

// Prefill MoE tail: weighted expert combine (x2.5) + shared + residual.
// (verbatim from lagunaPrefillMoETailKernel)
//   expert_outputs [rows][8][2048] bf16, router_weights [rows][8] fp32
//   shared_output [rows][2048] bf16, residual [rows][2048] bf16
//   output [rows][2048] bf16
// Grid: (512, rows) threads (4 cols per thread).
kernel void laguna_prefill_moe_tail_bf16_v1(
    const device bfloat* expert_outputs [[buffer(0)]],
    const device float* router_weights [[buffer(1)]],
    const device bfloat* shared_output [[buffer(2)]],
    const device bfloat* residual [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col0 = gid.x;

    constexpr uint hidden = 2048;
    constexpr uint experts = 8;
    constexpr uint n_cols = 4;

    uint col = col0 * n_cols;

    const device bfloat* expert_row =
        expert_outputs + (row * experts) * hidden + col;
    const device float* weight_row = router_weights + row * experts;

    bfloat expert_weights[experts];
    for (uint e = 0; e < experts; ++e) {
        expert_weights[e] = bfloat(weight_row[e]);
    }

    for (uint i = 0; i < n_cols; ++i) {
        bfloat total = bfloat(0);
        for (uint e = 0; e < experts; ++e) {
            bfloat product =
                bfloat(expert_row[e * hidden + i] * expert_weights[e]);
            total = bfloat(product + total);
        }
        bfloat scaled = bfloat(total * bfloat(2.5f));
        bfloat r2 = bfloat(scaled + shared_output[row * hidden + col + i]);
        output[row * hidden + col + i] =
            bfloat(residual[row * hidden + col + i] + r2);
    }
}

// Prefill SORTED MoE tail: weighted 8-expert combine (x2.5) + shared +
// residual with an inverse-order permutation gather over the sorted expert
// rows. (verbatim from lagunaPrefillSortedMoETailKernel)
//   sorted_expert_outputs [rows*8*2048] bf16 (sorted-regime plane)
//   inverse_order [rows*8] uint32 — sorted pos -> original pos
//   router_weights [rows*8] fp32, shared_output [rows*2048] bf16
//   residual [rows*2048] bf16, output [rows*2048] bf16
// Grid: (512, rows) threads (2048/4 cols per row) x 256-thread groups.
kernel void laguna_prefill_sorted_moe_tail_bf16_v1(
    const device bfloat* sorted_expert_outputs [[buffer(0)]],
    const device uint32_t* inverse_order [[buffer(1)]],
    const device float* router_weights [[buffer(2)]],
    const device bfloat* shared_output [[buffer(3)]],
    const device bfloat* residual [[buffer(4)]],
    device bfloat* output [[buffer(5)]],
    uint2 thread_position_in_grid [[thread_position_in_grid]])
{
constexpr uint hidden = 2048;
constexpr uint experts = 8;
constexpr uint n_cols = 4;

uint row = thread_position_in_grid.y;
uint col = thread_position_in_grid.x * n_cols;
const device float* weight_row = router_weights + row * experts;

bfloat expert_weights[experts];
uint sorted_rows[experts];
for (uint e = 0; e < experts; ++e) {
    expert_weights[e] = bfloat(weight_row[e]);
    sorted_rows[e] = inverse_order[row * experts + e];
}

for (uint i = 0; i < n_cols; ++i) {
    bfloat total = bfloat(0);
    for (uint e = 0; e < experts; ++e) {
        bfloat product = bfloat(
            sorted_expert_outputs[sorted_rows[e] * hidden + col + i] *
            expert_weights[e]);
        total = bfloat(product + total);
    }
    bfloat scaled = bfloat(total * bfloat(2.5f));
    bfloat r2 = bfloat(scaled + shared_output[row * hidden + col + i]);
    output[row * hidden + col + i] =
        bfloat(residual[row * hidden + col + i] + r2);
}
}

// Prefill router tournament: 8x 32-lane bitonic sorts then a 64-candidate
// network over the top-8s, per row. (verbatim from
// lagunaPrefillRouterTournamentKernelSource; normalizing selects the
// simd-folded normalization epilogue)
//   logits [rows*256], correction_bias [256] fp32
//   router_indices [rows*8] uint32, router_scores [rows*8] bf16
// Grid: (256, rows) threads (one threadgroup per row).
#define LAGUNA_PREFILL_ROUTER_TOURNAMENT_KERNEL(name, normalizing)            \
kernel void name(                                                            \
    const device bfloat* logits [[buffer(0)]],                               \
    const device float* correction_bias [[buffer(1)]],                       \
    device uint32_t* router_indices [[buffer(2)]],                           \
    device bfloat* router_scores [[buffer(3)]],                              \
    uint2 tlane [[thread_position_in_threadgroup]],                          \
    uint2 tgrid [[threadgroup_position_in_grid]])                            \
{                                                                            \
    uint lane = tlane.x;                                                     \
    uint row = tgrid.y;                                                      \
    threadgroup float xchg_keys[256];                                        \
    threadgroup uint xchg_indices[256];                                      \
    threadgroup float xchg_scores[256];                                      \
    threadgroup float candidate_keys[64];                                    \
    threadgroup uint candidate_indices[64];                                  \
    threadgroup float candidate_scores[64];                                  \
                                                                             \
    float x = float(logits[row * 256 + lane]);                               \
    float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));                     \
    float my_score = x < 0.0f ? y : 1.0f - y;                                \
    float my_key = -(my_score + float(correction_bias[lane]));               \
    uint my_index = lane;                                                    \
                                                                             \
    for (uint sequence = 2; sequence <= 32; sequence <<= 1) {                \
        for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {        \
            float other_key = simd_shuffle_xor(my_key, ushort(stride));      \
            uint other_index = simd_shuffle_xor(my_index, ushort(stride));   \
            float other_score = simd_shuffle_xor(my_score, ushort(stride));  \
                                                                             \
            bool is_lower = (lane & stride) == 0;                            \
            float a_key = is_lower ? my_key : other_key;                     \
            uint a_index = is_lower ? my_index : other_index;                \
            float a_score = is_lower ? my_score : other_score;               \
            float b_key = is_lower ? other_key : my_key;                     \
            uint b_index = is_lower ? other_index : my_index;                \
            float b_score = is_lower ? other_score : my_score;               \
                                                                             \
            bool lower_wants_better = (lane & sequence) == 0;                \
            bool b_before_a = laguna_router_key_before(                      \
                b_key, b_index, a_key, a_index);                             \
            bool a_before_b = laguna_router_key_before(                      \
                a_key, a_index, b_key, b_index);                             \
            bool swap = lower_wants_better ? b_before_a : a_before_b;        \
            if (swap) {                                                      \
                my_key = is_lower ? b_key : a_key;                           \
                my_index = is_lower ? b_index : a_index;                     \
                my_score = is_lower ? b_score : a_score;                     \
            }                                                                \
        }                                                                    \
    }                                                                        \
                                                                             \
    uint block = lane >> 5;                                                  \
    uint within_block = lane & 31;                                           \
    bool block_ascending = (block & 1) == 0;                                 \
    uint rank_in_block = block_ascending ? within_block : (31 - within_block); \
    bool is_local_top8 = block_ascending ? (within_block < 8) : (within_block >= 24); \
    if (is_local_top8) {                                                     \
        candidate_keys[block * 8 + rank_in_block] = my_key;                  \
        candidate_indices[block * 8 + rank_in_block] = my_index;             \
        candidate_scores[block * 8 + rank_in_block] = my_score;              \
    }                                                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
                                                                             \
    float my_key2 = candidate_keys[lane & 63];                               \
    uint my_index2 = candidate_indices[lane & 63];                           \
    float my_score2 = candidate_scores[lane & 63];                           \
    for (uint sequence = 2; sequence <= 64; sequence <<= 1) {                \
        for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {        \
            float other_key;                                                 \
            uint other_index;                                                \
            float other_score;                                               \
            if (stride < 32) {                                               \
                other_key = simd_shuffle_xor(my_key2, ushort(stride));       \
                other_index = simd_shuffle_xor(my_index2, ushort(stride));   \
                other_score = simd_shuffle_xor(my_score2, ushort(stride));   \
            } else {                                                         \
                xchg_keys[lane] = my_key2;                                   \
                xchg_indices[lane] = my_index2;                              \
                xchg_scores[lane] = my_score2;                               \
                threadgroup_barrier(mem_flags::mem_threadgroup);             \
                uint partner = lane ^ stride;                                \
                other_key = xchg_keys[partner];                              \
                other_index = xchg_indices[partner];                         \
                other_score = xchg_scores[partner];                          \
                threadgroup_barrier(mem_flags::mem_threadgroup);             \
            }                                                                \
                                                                             \
            bool is_lower = (lane & stride) == 0;                            \
            float a_key = is_lower ? my_key2 : other_key;                    \
            uint a_index = is_lower ? my_index2 : other_index;               \
            float a_score = is_lower ? my_score2 : other_score;              \
            float b_key = is_lower ? other_key : my_key2;                    \
            uint b_index = is_lower ? other_index : my_index2;               \
            float b_score = is_lower ? other_score : my_score2;              \
                                                                             \
            bool lower_wants_better = (lane & sequence) == 0;                \
            bool b_before_a = laguna_router_key_before(                      \
                b_key, b_index, a_key, a_index);                             \
            bool a_before_b = laguna_router_key_before(                      \
                a_key, a_index, b_key, b_index);                             \
            bool swap = lower_wants_better ? b_before_a : a_before_b;        \
            if (swap) {                                                      \
                my_key2 = is_lower ? b_key : a_key;                          \
                my_index2 = is_lower ? b_index : a_index;                    \
                my_score2 = is_lower ? b_score : a_score;                    \
            }                                                                \
        }                                                                    \
    }                                                                        \
                                                                             \
    if (normalizing) {                                                       \
        float total = 0.0f;                                                  \
        for (uint i = 0; i < 8; ++i) {                                       \
            total = simd_shuffle(my_score2, ushort(i)) + total;              \
        }                                                                    \
        if (lane < 8) {                                                      \
            router_indices[row * 8 + lane] = my_index2;                      \
            router_scores[row * 8 + lane] = bfloat(my_score2 / total);       \
        }                                                                    \
    } else {                                                                 \
        if (lane < 8) {                                                      \
            router_indices[row * 8 + lane] = my_index2;                      \
            router_scores[row * 8 + lane] = bfloat(my_score2);               \
        }                                                                    \
    }                                                                        \
}                                                                            \

LAGUNA_PREFILL_ROUTER_TOURNAMENT_KERNEL(laguna_prefill_router_tournament_v1, 0)
LAGUNA_PREFILL_ROUTER_TOURNAMENT_KERNEL(laguna_prefill_router_tournament_norm_v1, 1)

// Prefill router top-8: per-row O(256) per-lane predecessor count over the
// 256 choice keys (stable total order via laguna_router_key_before), the
// rank-ordered emitters plus a per-row winner score table. (verbatim from
// lagunaPrefillRouterTop8KernelSource; the normalizing epilogue folds the 8
// selected sigmoids in index order)
//   logits [rows*256], correction_bias [256] fp32
//   router_indices [rows*8] uint32, router_scores [rows*8] bf16
// Grid: (256, rows) threads (one threadgroup per row).
#define LAGUNA_PREFILL_ROUTER_TOP8_KERNEL(name, normalizing)               \
kernel void name(                                                          \
    const device bfloat* logits [[buffer(0)]],                             \
    const device float* correction_bias [[buffer(1)]],                     \
    device uint32_t* router_indices [[buffer(2)]],                         \
    device bfloat* router_scores [[buffer(3)]],                            \
    uint2 tlane [[thread_position_in_threadgroup]],                        \
    uint2 tgrid [[threadgroup_position_in_grid]])                          \
{                                                                          \
    uint lane = tlane.x;                                                   \
    uint row = tgrid.y;                                                    \
                                                                           \
    threadgroup float choice_keys[256];                                    \
    threadgroup float selected_scores[8];                                  \
                                                                           \
    float x = float(logits[row * 256 + lane]);                             \
    float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));                   \
    float score = x < 0.0f ? y : 1.0f - y;                                 \
    float corrected = score + float(correction_bias[lane]);                \
    float my_key = -corrected;                                             \
    choice_keys[lane] = my_key;                                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);                       \
                                                                           \
    uint rank = 0;                                                         \
    for (uint j = 0; j < 256; ++j) {                                       \
        rank += laguna_router_key_before(                                  \
            choice_keys[j], j, my_key, lane) ? 1 : 0;                      \
    }                                                                      \
    if (rank < 8) {                                                        \
        router_indices[row * 8 + rank] = lane;                             \
        selected_scores[rank] = score;                                     \
    }                                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                       \
                                                                           \
    if (lane < 8) {                                                        \
        if (normalizing) {                                                 \
            float total = 0.0f;                                            \
            for (uint i = 0; i < 8; ++i) {                                 \
                total = selected_scores[i] + total;                        \
            }                                                              \
            router_scores[row * 8 + lane] =                                \
                bfloat(selected_scores[lane] / total);                     \
        } else {                                                           \
            router_scores[row * 8 + lane] = bfloat(selected_scores[lane]); \
        }                                                                  \
    }                                                                      \
}                                                                          \

LAGUNA_PREFILL_ROUTER_TOP8_KERNEL(laguna_prefill_router_top8_v1, 0)
LAGUNA_PREFILL_ROUTER_TOP8_KERNEL(laguna_prefill_router_top8_norm_v1, 1)

// Prefill router tournament, ordinal payload: same two-phase 32 -> 64
// schedule and extraction geometry as the accepted float tournament above,
// but the payload is (uint ordinal, uint index) sorted by
// laguna_router_ordinal_before, with a per-row original-score table for the
// final eight indexed loads. Phase 2 runs one 64-candidate set on lanes
// 0..63 (the 32->64 cross-simdgroup merge handled explicitly). (verbatim
// from lagunaPrefillRouterTournamentOrdinalKernelSource)
//   logits [rows*256], correction_bias [256] fp32
//   router_indices [rows*8] uint32, router_scores [rows*8] bf16
// Grid: (256, rows) threads (one threadgroup per row).
#define LAGUNA_PREFILL_ROUTER_TOURNAMENT_ORDINAL_KERNEL(name, normalizing)   \
kernel void name(                                                            \
    const device bfloat* logits [[buffer(0)]],                               \
    const device float* correction_bias [[buffer(1)]],                       \
    device uint32_t* router_indices [[buffer(2)]],                           \
    device bfloat* router_scores [[buffer(3)]],                              \
    uint2 tlane [[thread_position_in_threadgroup]],                          \
    uint2 tgrid [[threadgroup_position_in_grid]])                            \
{                                                                            \
    uint lane = tlane.x;                                                     \
    uint row = tgrid.y;                                                      \
                                                                             \
    threadgroup uint xchg_ordinals[64];                                      \
    threadgroup uint xchg_indices[64];                                       \
    threadgroup uint candidate_ordinals[64];                                 \
    threadgroup uint candidate_indices[64];                                  \
    threadgroup float original_scores[256];                                  \
                                                                             \
    float x = float(logits[row * 256 + lane]);                               \
    float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));                     \
    float score = x < 0.0f ? y : 1.0f - y;                                   \
    original_scores[lane] = score;                                           \
    float key = -(score + float(correction_bias[lane]));                     \
    uint my_ordinal = laguna_router_key_ordinal(key);                        \
    uint my_index = lane;                                                    \
                                                                             \
    for (uint sequence = 2; sequence <= 32; sequence <<= 1) {                \
        for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {        \
            uint other_ordinal = simd_shuffle_xor(my_ordinal, ushort(stride)); \
            uint other_index = simd_shuffle_xor(my_index, ushort(stride));   \
                                                                             \
            bool is_lower = (lane & stride) == 0;                            \
            bool lower_wants_better = (lane & sequence) == 0;                \
            bool want_better = lower_wants_better == is_lower;               \
            bool other_before_my = laguna_router_ordinal_before(             \
                other_ordinal, other_index, my_ordinal, my_index);           \
            bool take_other = want_better ? other_before_my : !other_before_my; \
            if (take_other) {                                                \
                my_ordinal = other_ordinal;                                  \
                my_index = other_index;                                      \
            }                                                                \
        }                                                                    \
    }                                                                        \
                                                                             \
    uint block = lane >> 5;                                                  \
    uint within_block = lane & 31;                                           \
    bool block_ascending = (block & 1) == 0;                                 \
    uint rank_in_block = block_ascending ? within_block : (31 - within_block); \
    bool is_local_top8 = block_ascending ? (within_block < 8) : (within_block >= 24); \
    if (is_local_top8) {                                                     \
        candidate_ordinals[block * 8 + rank_in_block] = my_ordinal;          \
        candidate_indices[block * 8 + rank_in_block] = my_index;             \
    }                                                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
                                                                             \
    uint my_ordinal2 = 0u;                                                   \
    uint my_index2 = 0u;                                                     \
    if (lane < 64) {                                                         \
        my_ordinal2 = candidate_ordinals[lane];                              \
        my_index2 = candidate_indices[lane];                                 \
    }                                                                        \
                                                                             \
    if (lane < 64) {                                                         \
    for (uint sequence = 2; sequence <= 32; sequence <<= 1) {                \
        for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {        \
            uint other_ordinal = simd_shuffle_xor(my_ordinal2, ushort(stride)); \
            uint other_index = simd_shuffle_xor(my_index2, ushort(stride));  \
                                                                             \
            bool is_lower = (lane & stride) == 0;                            \
            bool lower_wants_better = (lane & sequence) == 0;                \
            bool want_better = lower_wants_better == is_lower;               \
            bool other_before_lane = laguna_router_ordinal_before(           \
                other_ordinal, other_index, my_ordinal2, my_index2);         \
            bool take_other = want_better ? other_before_lane : !other_before_lane; \
            if (take_other) {                                                \
                my_ordinal2 = other_ordinal;                                 \
                my_index2 = other_index;                                     \
            }                                                                \
        }                                                                    \
    }                                                                        \
    }                                                                        \
                                                                             \
    if (lane < 64) {                                                         \
        xchg_ordinals[lane] = my_ordinal2;                                   \
        xchg_indices[lane] = my_index2;                                      \
    }                                                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    if (lane < 64) {                                                         \
        uint partner = lane ^ 32u;                                           \
        uint other_ordinal = xchg_ordinals[partner];                         \
        uint other_index = xchg_indices[partner];                            \
        bool is_lower = (lane & 32u) == 0;                                   \
        bool other_before_lane = laguna_router_ordinal_before(               \
            other_ordinal, other_index, my_ordinal2, my_index2);             \
        bool take_other = is_lower ? other_before_lane : !other_before_lane; \
        if (take_other) {                                                    \
            my_ordinal2 = other_ordinal;                                     \
            my_index2 = other_index;                                         \
        }                                                                    \
        for (uint stride = 16; stride > 0; stride >>= 1) {                   \
            other_ordinal = simd_shuffle_xor(my_ordinal2, ushort(stride));   \
            other_index = simd_shuffle_xor(my_index2, ushort(stride));       \
            is_lower = (lane & stride) == 0;                                 \
            other_before_lane = laguna_router_ordinal_before(                \
                other_ordinal, other_index, my_ordinal2, my_index2);          \
            take_other = is_lower ? other_before_lane : !other_before_lane;  \
            if (take_other) {                                                \
                my_ordinal2 = other_ordinal;                                 \
                my_index2 = other_index;                                     \
            }                                                                \
        }                                                                    \
    }                                                                        \
                                                                             \
    if (normalizing) {                                                       \
        float my_score2 = lane < 8 ? original_scores[my_index2] : 0.0f;      \
        float total = 0.0f;                                                  \
        for (uint i = 0; i < 8; ++i) {                                       \
            total = simd_shuffle(my_score2, ushort(i)) + total;              \
        }                                                                    \
        if (lane < 8) {                                                      \
            router_indices[row * 8 + lane] = my_index2;                      \
            router_scores[row * 8 + lane] = bfloat(my_score2 / total);       \
        }                                                                    \
    } else {                                                                 \
        if (lane < 8) {                                                      \
            router_indices[row * 8 + lane] = my_index2;                      \
            router_scores[row * 8 + lane] =                                 \
                bfloat(original_scores[my_index2]);                          \
        }                                                                    \
    }                                                                        \
}                                                                            \

LAGUNA_PREFILL_ROUTER_TOURNAMENT_ORDINAL_KERNEL(                             \
    laguna_prefill_router_tournament_ordinal_active64_v2, 0)
LAGUNA_PREFILL_ROUTER_TOURNAMENT_ORDINAL_KERNEL(                             \
    laguna_prefill_router_tournament_ordinal_norm_active64_v2, 1)


// Inline-mask exact pass: candidate rows get the stock-exact GEMV, others
// get bfloat(coarse). (verbatim from
// laguna_lmhead_exact_inline_mask_block_delta_bf16_lane0_mask_v1)
//   coarse [V] f32, delta [V] bf16, thr [1] f32, lm_head [V*K] bf16, x [K]
//   assembled [V] bf16
// Grid: V/32 threadgroups (4 rows per simdgroup) x 256 threads (8 simdgroups).
kernel void laguna_lmhead_exact_inline_mask_block_delta_bf16_lane0_mask_v1(
    const device float* coarse [[buffer(0)]],
    const device bfloat* delta [[buffer(1)]],
    const device float* thr [[buffer(2)]],
    const device bfloat* lm_head [[buffer(3)]],
    const device bfloat* x [[buffer(4)]],
    device bfloat* assembled [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint VOCAB = 100352;
    constexpr uint K = 2048;

    uint base = tgid * 32 + sgid * 4;

    uint candidate_mask = 0;
    if (lane == 0) {
        for (uint tm = 0; tm < 4; ++tm) {
            uint r = base + tm;
            if (r < VOCAB && coarse[r] + float(delta[r]) >= thr[0]) {
                candidate_mask |= 1u << tm;
            }
        }
    }
    candidate_mask = simd_broadcast(candidate_mask, 0);

    if (candidate_mask == 0) {
        if (lane < 4 && base + lane < VOCAB) {
            assembled[base + lane] = bfloat(coarse[base + lane]);
        }
        return;
    }

    thread float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    thread bfloat inter[4];
    thread float v_coeff[4];
    uint bn = lane * 4;
    for (uint i = 0; i < 16; ++i) {
        vec<bfloat, 4> xv =
            *((const device vec<bfloat, 4>*)(x + bn));
        v_coeff[0] = float(xv.x);
        v_coeff[1] = float(xv.y);
        v_coeff[2] = float(xv.z);
        v_coeff[3] = float(xv.w);
        for (uint tm = 0; tm < 4; ++tm) {
            const device bfloat* mrow = lm_head + size_t(base + tm) * K;
            vec<bfloat, 4> mv =
                *((const device vec<bfloat, 4>*)(mrow + bn));
            inter[0] = mv.x;
            inter[1] = mv.y;
            inter[2] = mv.z;
            inter[3] = mv.w;
            result[tm] += inter[0] * v_coeff[0];
            result[tm] += inter[1] * v_coeff[1];
            result[tm] += inter[2] * v_coeff[2];
            result[tm] += inter[3] * v_coeff[3];
        }
        bn += 128;
    }
    for (uint tm = 0; tm < 4; ++tm) {
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            result[tm] += simd_shuffle_down(result[tm], sn);
        }
    }
    if (lane == 0) {
        for (uint tm = 0; tm < 4; ++tm) {
            uint r = base + tm;
            if (r < VOCAB) {
                assembled[r] = (candidate_mask & (1u << tm)) != 0
                    ? bfloat(result[tm])
                    : bfloat(coarse[r]);
            }
        }
    }
}

// Dense down_proj + residual (bf16), fused.
// (verbatim from lagunaDenseDownResidualKernel)
//   activated [8192] bf16, down_weight [2048][8192] bf16, residual [2048] bf16
//   output [2048] bf16 = residual + bf16(down)
// Grid: 2048/16=128 threadgroups x 64 threads (2 simdgroups).
kernel void laguna_dense_down_residual_bf16_v1(
    const device bfloat* activated [[buffer(0)]],
    const device bfloat* down_weight [[buffer(1)]],
    const device bfloat* residual [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint in_vec_size = 8192;
    constexpr uint rows_per_thread = 4;
    constexpr uint values_per_thread = 4;
    constexpr uint block_width = 128;
    constexpr uint blocks = in_vec_size / block_width;
    constexpr uint rows_per_group = 16;

    uint row_base = tile * rows_per_group + simd_group * rows_per_thread;

    thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
    thread float coefficients[values_per_thread];

    uint column = lane * values_per_thread;
    for (uint block = 0; block < blocks; ++block) {
        const vec<bfloat, 4> c4 =
            *((const device vec<bfloat, 4>*)(activated + column));
        for (uint i = 0; i < values_per_thread; ++i) {
            coefficients[i] = float(c4[i]);
        }
        for (uint row = 0; row < rows_per_thread; ++row) {
            const device vec<bfloat, 4>* row_values =
                (const device vec<bfloat, 4>*)(
                    down_weight + (row_base + row) * in_vec_size + column);
            const vec<bfloat, 4> w = row_values[0];
            for (uint i = 0; i < values_per_thread; ++i) {
                result[row] += float(w[i]) * coefficients[i];
            }
        }
        column += block_width;
    }

    for (uint row = 0; row < rows_per_thread; ++row) {
        for (ushort delta = 16; delta >= 1; delta >>= 1) {
            result[row] += metal::simd_shuffle_down(result[row], delta);
        }
    }
    if (lane == 0) {
        for (uint row = 0; row < rows_per_thread; ++row) {
            bfloat down = bfloat(result[row]);
            output[row_base + row] =
                bfloat(residual[row_base + row] + down);
        }
    }
}

// ── Inject probes (harness timing; verbatim) ────────────────────────────

// Empty dispatch (chaining probe).
//   control [1] u32, prev [1] u32 -> sink [256] u32
kernel void laguna_inject_empty_dispatch_v1(
    const device uint32_t* control [[buffer(0)]],
    const device uint32_t* prev [[buffer(1)]],
    device uint32_t* sink [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (control[0] == 0xFFFFFFFFu) {
        sink[gid & 255u] = gid + prev[0];
    }
}

// DRAM sweep probe.
//   pool [pool*4] u32, control [2] u32 -> sink [256] u32
kernel void laguna_inject_dram_sweep_u4_v2(
    const device uint32_t* pool [[buffer(0)]],
    const device uint32_t* control [[buffer(1)]],
    device uint32_t* sink [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    constexpr uint kThreads = 1024;
    constexpr uint kPerThread = 8;
    constexpr uint kMask = (1u << 22) - 1u;
    const device uint4* quads = (const device uint4*)pool;
    uint idx = (gid + control[0]) & kMask;
    uint passes = control[1];
    uint4 acc = uint4(0u);
    for (uint p = 0; p < passes; ++p) {
        for (uint i = 0; i < kPerThread; ++i) {
            acc ^= quads[idx];
            idx = (idx + kThreads) & kMask;
        }
    }
    uint folded = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (folded == 0xFFFFFFFFu) {
        sink[gid & 255u] = folded;
    }
}

// Decode embedding + RoPE angle atlas, fused.
// (verbatim from lagunaDecodeEmbeddingRoPEAtlasKernel)
//   tokens [1] i32, embedding_weight [V][2048] bf16
//   full_atlas [L][64] f32, sliding_atlas [L][128] f32, atlas_position i32
//   hidden [2048] bf16, full_angles [64] f32, sliding_angles [128] f32
// Grid: 512 threads.
kernel void laguna_decode_embedding_rope_atlas_bf16_2048_v2(
    const device int32_t* tokens [[buffer(0)]],
    const device bfloat* embedding_weight [[buffer(1)]],
    const device float* full_atlas [[buffer(2)]],
    const device float* sliding_atlas [[buffer(3)]],
    const device int32_t* atlas_position [[buffer(4)]],
    device bfloat* hidden [[buffer(5)]],
    device float* full_angles [[buffer(6)]],
    device float* sliding_angles [[buffer(7)]],
    uint lane [[thread_position_in_grid]])
{
    constexpr uint hidden_size = 2048;
    constexpr uint hidden_vectors = hidden_size / 4;
    constexpr uint full_width = 64;
    constexpr uint sliding_width = 128;

    uint token = uint(tokens[0]);
    uint position = uint(atlas_position[0]);

    const device vec<bfloat, 4>* embedding_vectors =
        (const device vec<bfloat, 4>*)(
            embedding_weight + token * hidden_size);
    device vec<bfloat, 4>* hidden_vectors_out =
        (device vec<bfloat, 4>*)(hidden);
    if (lane < hidden_vectors) {
        hidden_vectors_out[lane] = embedding_vectors[lane];
    }

    if (lane < full_width / 4) {
        const device vec<float, 4>* atlas_vectors =
            (const device vec<float, 4>*)(
                full_atlas + position * full_width);
        ((device vec<float, 4>*)(full_angles))[lane] =
            atlas_vectors[lane];
    }
    if (lane < sliding_width / 4) {
        const device vec<float, 4>* atlas_vectors =
            (const device vec<float, 4>*)(
                sliding_atlas + position * sliding_width);
        ((device vec<float, 4>*)(sliding_angles))[lane] =
            atlas_vectors[lane];
    }
}

// Fused full-attention decode (grow regime): QK-norm+YaRN, cache write,
// flash attention over the runtime-length N with the grow tail row.
// (verbatim from lagunaFullFusedAttentionKernel; the sliding-ring twin with
// gqa=6, partial 64-dim YaRN rotation, runtime N = params[1], capacity
// cache stride, and the pair_k tail-row path)
//   raw_queries [48*128] bf16, raw_keys [8*128] bf16, raw_values [8*128]
//   query_weight [128], key_weight [128] bf16, angles [64] fp32
//   k_cache [8][cap][128], v_cache [8][cap][128] bf16
//   params [3] u32 (widx, N, capacity), scale_arr [1] fp32
//   attended [48*128] bf16
// Grid: (24 * 1024) threads / 1024-thread groups = 24 groups (2 heads pair).
kernel void laguna_full_fused_attn_grow_v1(
    const device bfloat* raw_queries [[buffer(0)]],
    const device bfloat* raw_keys [[buffer(1)]],
    const device bfloat* raw_values [[buffer(2)]],
    const device bfloat* query_weight [[buffer(3)]],
    const device bfloat* key_weight [[buffer(4)]],
    const device float* angles [[buffer(5)]],
    const device bfloat* k_cache [[buffer(6)]],
    const device bfloat* v_cache [[buffer(7)]],
    const device uint32_t* params [[buffer(8)]],
    const device float* scale_arr [[buffer(9)]],
    device bfloat* attended [[buffer(10)]],
    uint pair_tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint head_dim = 128;
    constexpr uint gqa = 6;
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int BDP = BD + 1;
    constexpr int qk_per_thread = 4;
    constexpr int v_per_thread = 4;
    constexpr uint rotary_pairs = 32;
    constexpr float yarn_mscale = 1.3465735912322998f;

    typedef float U;

    uint head0 = pair_tg * 2;
    uint head1 = head0 + 1;
    uint kv_head = head0 / gqa;
    uint widx = params[0];
    int N = int(params[1]);
    uint capacity = params[2];
    float scale = scale_arr[0];

    threadgroup bfloat tg_q0[head_dim];
    threadgroup bfloat tg_q1[head_dim];
    threadgroup bfloat tg_k[head_dim];
    threadgroup bfloat tg_v[head_dim];

    if (sg < 3) {
        const device bfloat* input =
            sg == 0 ? raw_queries + head0 * head_dim
            : sg == 1 ? raw_queries + head1 * head_dim
                      : raw_keys + kv_head * head_dim;
        const device bfloat* weight =
            sg == 2 ? key_weight : query_weight;
        threadgroup bfloat* outrow =
            sg == 0 ? tg_q0 : sg == 1 ? tg_q1 : tg_k;

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
                outrow[pair] = bfloat(first * cosine - second * sine);
                outrow[pair + rotary_pairs] =
                    bfloat(first * sine + second * cosine);
            }
        } else if (lane >= 16) {
            for (uint i = 0; i < 4; ++i) {
                outrow[base + i] = normalized[i];
            }
        }
    } else if (sg == 3) {
        const device bfloat* vin = raw_values + kv_head * head_dim;
        for (uint i = lane; i < head_dim; i += 32) {
            tg_v[i] = vin[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((head0 % gqa) == 0 && sg == 0) {
        device bfloat* kc = (device bfloat*)k_cache +
            (size_t)kv_head * (capacity * head_dim) +
            (size_t)widx * head_dim;
        device bfloat* vc = (device bfloat*)v_cache +
            (size_t)kv_head * (capacity * head_dim) +
            (size_t)widx * head_dim;
        for (uint i = lane; i < head_dim; i += 32) {
            kc[i] = tg_k[i];
            vc[i] = tg_v[i];
        }
    }

    threadgroup U outputs[4 * BN * BDP];
    threadgroup U max_scores[2 * BN];
    threadgroup U sum_exp_scores[2 * BN];

    const device bfloat* pair_keys = k_cache +
        (size_t)kv_head * (capacity * head_dim) +
        (size_t)sg * head_dim + lane * qk_per_thread;
    const device bfloat* pair_values = v_cache +
        (size_t)kv_head * (capacity * head_dim) +
        (size_t)sg * head_dim + lane * v_per_thread;
    const int inner_k_stride = BN * int(head_dim);
    const int inner_v_stride = BN * int(head_dim);

    thread U pair_q0[qk_per_thread];
    thread U pair_q1[qk_per_thread];
    thread U pair_k[qk_per_thread];
    thread U pair_o0[v_per_thread];
    thread U pair_o1[v_per_thread];

    for (int j = 0; j < qk_per_thread; ++j) {
        pair_q0[j] =
            static_cast<U>(scale) * tg_q0[lane * qk_per_thread + j];
        pair_q1[j] =
            static_cast<U>(scale) * tg_q1[lane * qk_per_thread + j];
    }
    for (int j = 0; j < v_per_thread; ++j) {
        pair_o0[j] = 0;
        pair_o1[j] = 0;
    }

    U pair_max0 = metal::numeric_limits<U>::lowest();
    U pair_max1 = metal::numeric_limits<U>::lowest();
    U pair_sum0 = 0;
    U pair_sum1 = 0;

    int i = sg;
    for (; i + BN < N; i += 2 * BN) {
        const device bfloat* pipe_keys_b = pair_keys + inner_k_stride;
        const device bfloat* pipe_values_b = pair_values + inner_v_stride;
        const bool sub_a = uint(i) == widx;
        const bool sub_b = uint(i + BN) == widx;
        U pipe_ka[4];
        U pipe_kb[4];
        // T_LOAD_K
        if (sub_a) {
            pipe_ka[0] = tg_k[lane * qk_per_thread + 0];
            pipe_ka[1] = tg_k[lane * qk_per_thread + 1];
            pipe_ka[2] = tg_k[lane * qk_per_thread + 2];
            pipe_ka[3] = tg_k[lane * qk_per_thread + 3];
        } else {
            const vec<bfloat, 4> v_ =
                *reinterpret_cast<const device vec<bfloat, 4>*>(pair_keys);
            pipe_ka[0] = v_.x; pipe_ka[1] = v_.y;
            pipe_ka[2] = v_.z; pipe_ka[3] = v_.w;
        }
        if (sub_b) {
            pipe_kb[0] = tg_k[lane * qk_per_thread + 0];
            pipe_kb[1] = tg_k[lane * qk_per_thread + 1];
            pipe_kb[2] = tg_k[lane * qk_per_thread + 2];
            pipe_kb[3] = tg_k[lane * qk_per_thread + 3];
        } else {
            const vec<bfloat, 4> v_ =
                *reinterpret_cast<const device vec<bfloat, 4>*>(pipe_keys_b);
            pipe_kb[0] = v_.x; pipe_kb[1] = v_.y;
            pipe_kb[2] = v_.z; pipe_kb[3] = v_.w;
        }
        bfloat pipe_va0, pipe_va1, pipe_va2, pipe_va3;
        bfloat pipe_vb0, pipe_vb1, pipe_vb2, pipe_vb3;
        // T_LOAD_V
        if (sub_a) {
            pipe_va0 = tg_v[lane * v_per_thread + 0];
            pipe_va1 = tg_v[lane * v_per_thread + 1];
            pipe_va2 = tg_v[lane * v_per_thread + 2];
            pipe_va3 = tg_v[lane * v_per_thread + 3];
        } else {
            const vec<bfloat, 4> v_ =
                *reinterpret_cast<const device vec<bfloat, 4>*>(pair_values);
            pipe_va0 = v_.x; pipe_va1 = v_.y;
            pipe_va2 = v_.z; pipe_va3 = v_.w;
        }
        if (sub_b) {
            pipe_vb0 = tg_v[lane * v_per_thread + 0];
            pipe_vb1 = tg_v[lane * v_per_thread + 1];
            pipe_vb2 = tg_v[lane * v_per_thread + 2];
            pipe_vb3 = tg_v[lane * v_per_thread + 3];
        } else {
            const vec<bfloat, 4> v_ =
                *reinterpret_cast<const device vec<bfloat, 4>*>(pipe_values_b);
            pipe_vb0 = v_.x; pipe_vb1 = v_.y;
            pipe_vb2 = v_.z; pipe_vb3 = v_.w;
        }

        U pair_score0 = 0;
        U pair_score1 = 0;
        pair_score0 += pair_q0[0] * pipe_ka[0];
        pair_score1 += pair_q1[0] * pipe_ka[0];
        pair_score0 += pair_q0[1] * pipe_ka[1];
        pair_score1 += pair_q1[1] * pipe_ka[1];
        pair_score0 += pair_q0[2] * pipe_ka[2];
        pair_score1 += pair_q1[2] * pipe_ka[2];
        pair_score0 += pair_q0[3] * pipe_ka[3];
        pair_score1 += pair_q1[3] * pipe_ka[3];
        pair_score0 = simd_sum(pair_score0);
        pair_score1 = simd_sum(pair_score1);

        U pair_new_max0 = metal::max(pair_max0, pair_score0);
        U pair_new_max1 = metal::max(pair_max1, pair_score1);
        U pair_factor0;
        U pair_factor1;
        {
            const float db_delta_ = (pair_max0 - pair_new_max0);
            pair_factor0 = (as_type<uint>(db_delta_) == 0u)
                ? float(1.0f) : metal::fast::exp(db_delta_);
        }
        {
            const float db_delta_ = (pair_max1 - pair_new_max1);
            pair_factor1 = (as_type<uint>(db_delta_) == 0u)
                ? float(1.0f) : metal::fast::exp(db_delta_);
        }
        U pair_exp0 = metal::fast::exp(pair_score0 - pair_new_max0);
        U pair_exp1 = metal::fast::exp(pair_score1 - pair_new_max1);

        pair_max0 = pair_new_max0;
        pair_max1 = pair_new_max1;
        pair_sum0 = pair_sum0 * pair_factor0 + pair_exp0;
        pair_sum1 = pair_sum1 * pair_factor1 + pair_exp1;

        pair_o0[0] = pair_o0[0] * pair_factor0 + pair_exp0 * pipe_va0;
        pair_o1[0] = pair_o1[0] * pair_factor1 + pair_exp1 * pipe_va0;
        pair_o0[1] = pair_o0[1] * pair_factor0 + pair_exp0 * pipe_va1;
        pair_o1[1] = pair_o1[1] * pair_factor1 + pair_exp1 * pipe_va1;
        pair_o0[2] = pair_o0[2] * pair_factor0 + pair_exp0 * pipe_va2;
        pair_o1[2] = pair_o1[2] * pair_factor1 + pair_exp1 * pipe_va2;
        pair_o0[3] = pair_o0[3] * pair_factor0 + pair_exp0 * pipe_va3;
        pair_o1[3] = pair_o1[3] * pair_factor1 + pair_exp1 * pipe_va3;

        U pipeb_score0 = 0;
        U pipeb_score1 = 0;
        pipeb_score0 += pair_q0[0] * pipe_kb[0];
        pipeb_score1 += pair_q1[0] * pipe_kb[0];
        pipeb_score0 += pair_q0[1] * pipe_kb[1];
        pipeb_score1 += pair_q1[1] * pipe_kb[1];
        pipeb_score0 += pair_q0[2] * pipe_kb[2];
        pipeb_score1 += pair_q1[2] * pipe_kb[2];
        pipeb_score0 += pair_q0[3] * pipe_kb[3];
        pipeb_score1 += pair_q1[3] * pipe_kb[3];
        pipeb_score0 = simd_sum(pipeb_score0);
        pipeb_score1 = simd_sum(pipeb_score1);

        U pipeb_new_max0 = metal::max(pair_max0, pipeb_score0);
        U pipeb_new_max1 = metal::max(pair_max1, pipeb_score1);
        U pipeb_factor0;
        U pipeb_factor1;
        {
            const float db_delta_ = (pair_max0 - pipeb_new_max0);
            pipeb_factor0 = (as_type<uint>(db_delta_) == 0u)
                ? float(1.0f) : metal::fast::exp(db_delta_);
        }
        {
            const float db_delta_ = (pair_max1 - pipeb_new_max1);
            pipeb_factor1 = (as_type<uint>(db_delta_) == 0u)
                ? float(1.0f) : metal::fast::exp(db_delta_);
        }
        U pipeb_exp0 = metal::fast::exp(pipeb_score0 - pipeb_new_max0);
        U pipeb_exp1 = metal::fast::exp(pipeb_score1 - pipeb_new_max1);

        pair_max0 = pipeb_new_max0;
        pair_max1 = pipeb_new_max1;
        pair_sum0 = pair_sum0 * pipeb_factor0 + pipeb_exp0;
        pair_sum1 = pair_sum1 * pipeb_factor1 + pipeb_exp1;

        pair_o0[0] = pair_o0[0] * pipeb_factor0 + pipeb_exp0 * pipe_vb0;
        pair_o1[0] = pair_o1[0] * pipeb_factor1 + pipeb_exp1 * pipe_vb0;
        pair_o0[1] = pair_o0[1] * pipeb_factor0 + pipeb_exp0 * pipe_vb1;
        pair_o1[1] = pair_o1[1] * pipeb_factor1 + pipeb_exp1 * pipe_vb1;
        pair_o0[2] = pair_o0[2] * pipeb_factor0 + pipeb_exp0 * pipe_vb2;
        pair_o1[2] = pair_o1[2] * pipeb_factor1 + pipeb_exp1 * pipe_vb2;
        pair_o0[3] = pair_o0[3] * pipeb_factor0 + pipeb_exp0 * pipe_vb3;
        pair_o1[3] = pair_o1[3] * pipeb_factor1 + pipeb_exp1 * pipe_vb3;

        pair_keys += 2 * inner_k_stride;
        pair_values += 2 * inner_v_stride;
    }

    if (i < N) {
        const bool sub_t = uint(i) == widx;
        // T_LOAD_K
        if (sub_t) {
            pair_k[0] = tg_k[lane * qk_per_thread + 0];
            pair_k[1] = tg_k[lane * qk_per_thread + 1];
            pair_k[2] = tg_k[lane * qk_per_thread + 2];
            pair_k[3] = tg_k[lane * qk_per_thread + 3];
        } else {
            const vec<bfloat, 4> v_ =
                *reinterpret_cast<const device vec<bfloat, 4>*>(pair_keys);
            pair_k[0] = v_.x; pair_k[1] = v_.y;
            pair_k[2] = v_.z; pair_k[3] = v_.w;
        }
        bfloat pipe_va0, pipe_va1, pipe_va2, pipe_va3;
        // T_LOAD_V
        if (sub_t) {
            pipe_va0 = tg_v[lane * v_per_thread + 0];
            pipe_va1 = tg_v[lane * v_per_thread + 1];
            pipe_va2 = tg_v[lane * v_per_thread + 2];
            pipe_va3 = tg_v[lane * v_per_thread + 3];
        } else {
            const vec<bfloat, 4> v_ =
                *reinterpret_cast<const device vec<bfloat, 4>*>(pair_values);
            pipe_va0 = v_.x; pipe_va1 = v_.y;
            pipe_va2 = v_.z; pipe_va3 = v_.w;
        }

        U pair_score0 = 0;
        U pair_score1 = 0;
        pair_score0 += pair_q0[0] * pair_k[0];
        pair_score1 += pair_q1[0] * pair_k[0];
        pair_score0 += pair_q0[1] * pair_k[1];
        pair_score1 += pair_q1[1] * pair_k[1];
        pair_score0 += pair_q0[2] * pair_k[2];
        pair_score1 += pair_q1[2] * pair_k[2];
        pair_score0 += pair_q0[3] * pair_k[3];
        pair_score1 += pair_q1[3] * pair_k[3];
        pair_score0 = simd_sum(pair_score0);
        pair_score1 = simd_sum(pair_score1);

        U pair_new_max0 = metal::max(pair_max0, pair_score0);
        U pair_new_max1 = metal::max(pair_max1, pair_score1);
        U pair_factor0;
        U pair_factor1;
        {
            const float db_delta_ = (pair_max0 - pair_new_max0);
            pair_factor0 = (as_type<uint>(db_delta_) == 0u)
                ? float(1.0f) : metal::fast::exp(db_delta_);
        }
        {
            const float db_delta_ = (pair_max1 - pair_new_max1);
            pair_factor1 = (as_type<uint>(db_delta_) == 0u)
                ? float(1.0f) : metal::fast::exp(db_delta_);
        }
        U pair_exp0 = metal::fast::exp(pair_score0 - pair_new_max0);
        U pair_exp1 = metal::fast::exp(pair_score1 - pair_new_max1);

        pair_max0 = pair_new_max0;
        pair_max1 = pair_new_max1;
        pair_sum0 = pair_sum0 * pair_factor0 + pair_exp0;
        pair_sum1 = pair_sum1 * pair_factor1 + pair_exp1;

        pair_o0[0] = pair_o0[0] * pair_factor0 + pair_exp0 * pipe_va0;
        pair_o1[0] = pair_o1[0] * pair_factor1 + pair_exp1 * pipe_va0;
        pair_o0[1] = pair_o0[1] * pair_factor0 + pair_exp0 * pipe_va1;
        pair_o1[1] = pair_o1[1] * pair_factor1 + pair_exp1 * pipe_va1;
        pair_o0[2] = pair_o0[2] * pair_factor0 + pair_exp0 * pipe_va2;
        pair_o1[2] = pair_o1[2] * pair_factor1 + pair_exp1 * pipe_va2;
        pair_o0[3] = pair_o0[3] * pair_factor0 + pair_exp0 * pipe_va3;
        pair_o1[3] = pair_o1[3] * pair_factor1 + pair_exp1 * pipe_va3;
    }

    constexpr int pair_planes = 2;
    constexpr int pair_plane_size = BN * BDP;
    if (lane == 0) {
        max_scores[sg] = pair_max0;
        max_scores[BN + sg] = pair_max1;
        sum_exp_scores[sg] = pair_sum0;
        sum_exp_scores[BN + sg] = pair_sum1;
    }
    for (int p = 0; p < pair_planes; ++p) {
        outputs[p * pair_plane_size + lane * BDP + sg] = pair_o0[p];
        outputs[
            (pair_planes + p) * pair_plane_size + lane * BDP + sg] =
            pair_o1[p];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    pair_max0 = max_scores[lane];
    pair_max1 = max_scores[BN + lane];
    U pair_global_max0 = simd_max(pair_max0);
    U pair_global_max1 = simd_max(pair_max1);
    U pair_global_factor0 = metal::fast::exp(pair_max0 - pair_global_max0);
    U pair_global_factor1 = metal::fast::exp(pair_max1 - pair_global_max1);
    pair_sum0 = simd_sum(sum_exp_scores[lane] * pair_global_factor0);
    pair_sum1 = simd_sum(sum_exp_scores[BN + lane] * pair_global_factor1);

    for (int p = 0; p < pair_planes; ++p) {
        U acc0 = simd_sum(
            outputs[p * pair_plane_size + sg * BDP + lane] *
            pair_global_factor0);
        U acc1 = simd_sum(
            outputs[
                (pair_planes + p) * pair_plane_size + sg * BDP + lane] *
            pair_global_factor1);
        pair_o0[p] = pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
        pair_o1[p] = pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = 0; p < pair_planes; ++p) {
        outputs[p * pair_plane_size + lane * BDP + sg] =
            pair_o0[pair_planes + p];
        outputs[
            (pair_planes + p) * pair_plane_size + lane * BDP + sg] =
            pair_o1[pair_planes + p];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = 0; p < pair_planes; ++p) {
        U acc0 = simd_sum(
            outputs[p * pair_plane_size + sg * BDP + lane] *
            pair_global_factor0);
        U acc1 = simd_sum(
            outputs[
                (pair_planes + p) * pair_plane_size + sg * BDP + lane] *
            pair_global_factor1);
        pair_o0[pair_planes + p] =
            pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
        pair_o1[pair_planes + p] =
            pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
    }

    if (lane == 0) {
        device bfloat* pair_out0 =
            attended + head0 * head_dim + sg * v_per_thread;
        device bfloat* pair_out1 =
            attended + head1 * head_dim + sg * v_per_thread;
        for (int p = 0; p < v_per_thread; ++p) {
            pair_out0[p] = static_cast<bfloat>(pair_o0[p]);
            pair_out1[p] = static_cast<bfloat>(pair_o1[p]);
        }
    }
}

// ── Dense (bf16) MLP kernels ──────────────────────────────────────────

// Dense gate/up fused + SwiGLU (bf16 fused plane).
// (verbatim from lagunaDenseGateUpSwiGLUKernel)
//   input [2048] bf16, fused_weight [2*8192][2048] bf16 (gate then up)
//   activated [8192] bf16
// Grid: 8192/64=128 threadgroups x 64 threads (2 simdgroups).
kernel void laguna_dense_gate_up_swiglu_bf16_v1(
    const device bfloat* input [[buffer(0)]],
    const device bfloat* fused_weight [[buffer(1)]],
    device bfloat* activated [[buffer(2)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint in_vec_size = 2048;
    constexpr uint output_width = 8192;
    constexpr uint rows_per_thread = 4;
    constexpr uint values_per_thread = 4;
    constexpr uint block_width = 128;
    constexpr uint blocks = in_vec_size / block_width;
    constexpr uint rows_per_group = 64;

    uint row_base = tile * rows_per_group + simd_group * rows_per_thread;

    thread float gate_result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
    thread float up_result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
    thread float coefficients[values_per_thread];

    uint column = lane * values_per_thread;
    for (uint block = 0; block < blocks; ++block) {
        const vec<bfloat, 4> c4 =
            *((const device vec<bfloat, 4>*)(input + column));
        for (uint i = 0; i < values_per_thread; ++i) {
            coefficients[i] = float(c4[i]);
        }
        for (uint row = 0; row < rows_per_thread; ++row) {
            const device vec<bfloat, 4>* gate_row_values =
                (const device vec<bfloat, 4>*)(
                    fused_weight + (row_base + row) * in_vec_size + column);
            const vec<bfloat, 4> gw = gate_row_values[0];
            const device vec<bfloat, 4>* up_row_values =
                (const device vec<bfloat, 4>*)(
                    fused_weight +
                    (output_width + row_base + row) * in_vec_size + column);
            const vec<bfloat, 4> uw = up_row_values[0];
            for (uint i = 0; i < values_per_thread; ++i) {
                gate_result[row] += float(gw[i]) * coefficients[i];
                up_result[row] += float(uw[i]) * coefficients[i];
            }
        }
        column += block_width;
    }

    for (uint row = 0; row < rows_per_thread; ++row) {
        for (ushort delta = 16; delta >= 1; delta >>= 1) {
            gate_result[row] +=
                metal::simd_shuffle_down(gate_result[row], delta);
            up_result[row] +=
                metal::simd_shuffle_down(up_result[row], delta);
        }
    }
    if (lane == 0) {
        for (uint row = 0; row < rows_per_thread; ++row) {
            bfloat gate = bfloat(gate_result[row]);
            bfloat up = bfloat(up_result[row]);
            bfloat exp_abs = metal::exp(metal::abs(gate));
            bfloat denominator = bfloat(1) + exp_abs;
            bfloat y = bfloat(1) / denominator;
            bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
            bfloat silu = bfloat(gate * sigmoid);
            activated[row_base + row] = bfloat(silu * up);
        }
    }
}

// ── LM-head int5 prune family (LagunaLmHeadPrune.swift) ──────────────

static inline float laguna_e8m0_decode(uint8_t b) {
    if (b == 0u) {
        return as_type<float>(0x00400000u);
    }
    return as_type<float>(uint(b) << 23);
}

// Int5 coarse pass: fused coarse (lower bound) + delta (certified bound,
// bf16-up). (verbatim from lagunaLmHeadInt5CoarseRatioBoundDeltaBF16Kernel)
//   x [2048] bf16, codes_lo [V][1024] uint8, codes_hi [V][256] uint8,
//   scales [V][64] uint8
//   coarse [V] f32, delta [V] bf16
// Grid: V/16 threadgroups (16 rows) x 32 threads (1 simdgroup per row).
kernel void laguna_lmhead_int5_coarse_ratio_bound_delta_bf16_v6(
    const device bfloat* x [[buffer(0)]],
    const device uint8_t* codes_lo [[buffer(1)]],
    const device uint8_t* codes_hi [[buffer(2)]],
    const device uint8_t* scales [[buffer(3)]],
    device float* coarse [[buffer(4)]],
    device bfloat* delta [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr float GAMMA = 0x1p-15f;
    uint row = tg * 16 + sg;
    const device uint8_t* lorow = codes_lo + size_t(row) * 1024;
    const device uint8_t* hirow = codes_hi + size_t(row) * 256;
    const device uint8_t* srow = scales + size_t(row) * 64;

    float c_acc = 0.0f;
    float d_acc = 0.0f;
    for (uint gg = 0; gg < 2; ++gg) {
        uint g = 2 * lane + gg;
        float sd = laguna_e8m0_decode(srow[g]);
        uint4 lo4 = ((const device uint4*)(lorow + g * 16))[0];
        uint hb = ((const device uint*)(hirow + g * 4))[0];
        const device ushort4* xrow = (const device ushort4*)(x + g * 32);
        float cg = 0.0f;
        float ag = 0.0f;
        for (uint w = 0; w < 4; ++w) {
            uint lw = lo4[w];
            uint hw = hb >> (8u * w);
            uint4 ne = (uint4(lw) >> uint4(0u, 8u, 16u, 24u)) & 15u;
            uint4 no = (uint4(lw) >> uint4(4u, 12u, 20u, 28u)) & 15u;
            uint4 he = (uint4(hw) >> uint4(0u, 2u, 4u, 6u)) & 1u;
            uint4 ho = (uint4(hw) >> uint4(1u, 3u, 5u, 7u)) & 1u;
            float4 ve = float4((ne << 1u) | he) - 16.0f;
            float4 vo = float4((no << 1u) | ho) - 16.0f;
            float4 xa = as_type<float4>(uint4(xrow[2 * w]) << 16);
            float4 xb = as_type<float4>(uint4(xrow[2 * w + 1]) << 16);
            float4 xe = float4(xa.x, xa.z, xb.x, xb.z);
            float4 xo = float4(xa.y, xa.w, xb.y, xb.w);
            float4 axe = metal::abs(xe);
            float4 axo = metal::abs(xo);
            for (uint k = 0; k < 4; ++k) {
                cg += xe[k] * ve[k];
                cg += xo[k] * vo[k];
                ag += axe[k];
                ag += axo[k];
            }
        }
        c_acc += sd * cg;
        d_acc += (0.5f * sd) * ag;
    }
    c_acc = simd_sum(c_acc);
    d_acc = simd_sum(d_acc);
    if (lane == 0) {
        coarse[row] = c_acc;
        float d_up = d_acc * (1.0f + 61.0f * GAMMA);
        uint dbits = as_type<uint>(d_up);
        uint dtrunc = dbits & 0xFFFF0000u;
        if (dtrunc != dbits) {
            dtrunc += 0x00010000u;
        }
        delta[row] = as_type<bfloat>(ushort(dtrunc >> 16));
    }
}

// Argmax stage 1 over the coarse plane: 128 partial (max, idx) rows.
// (verbatim from laguna_lmhead_coarse_argmax_stage1_v5)
//   coarse [V] f32 -> partial_max [128] f32, partial_idx [128] uint32
// Grid: 224 threadgroups x 128 threads (784 active, 7 simdgroups).
kernel void laguna_lmhead_coarse_argmax_stage1_v5(
    const device float* coarse [[buffer(0)]],
    device float* partial_max [[buffer(1)]],
    device uint32_t* partial_idx [[buffer(2)]],
    uint2 tgrid [[threadgroup_position_in_grid]],
    uint2 tlid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    uint row = tgrid.y;
    uint lid = tlid.x;
    constexpr uint ROW_SIZE = 784;
    constexpr uint READS = 4;
    constexpr uint ACTIVE_THREADS = ROW_SIZE / READS;
    constexpr uint SIMD_GROUPS = 7;
    threadgroup float shared_vals[32];
    threadgroup uint shared_idxs[32];

    float best = -metal::numeric_limits<float>::infinity();
    uint best_idx = 0xFFFFFFFFu;
    if (lid < ACTIVE_THREADS) {
        uint base = row * ROW_SIZE + lid * READS;
        for (uint i = 0; i < READS; ++i) {
            float v = coarse[base + i];
            if (v > best || (v == best && base + i < best_idx)) {
                best = v;
                best_idx = base + i;
            }
        }
    }

    for (ushort sn = 16; sn >= 1; sn >>= 1) {
        float ov = simd_shuffle_down(best, sn);
        uint oi = simd_shuffle_down(best_idx, sn);
        if (ov > best || (ov == best && oi < best_idx)) {
            best = ov;
            best_idx = oi;
        }
    }
    if (simd_lane == 0) {
        shared_vals[simd_group] = best;
        shared_idxs[simd_group] = best_idx;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    best = lid < SIMD_GROUPS
        ? shared_vals[lid]
        : -metal::numeric_limits<float>::infinity();
    best_idx = lid < SIMD_GROUPS ? shared_idxs[lid] : 0xFFFFFFFFu;
    for (ushort sn = 16; sn >= 1; sn >>= 1) {
        float ov = simd_shuffle_down(best, sn);
        uint oi = simd_shuffle_down(best_idx, sn);
        if (ov > best || (ov == best && oi < best_idx)) {
            best = ov;
            best_idx = oi;
        }
    }
    if (lid == 0) {
        partial_max[row] = best;
        partial_idx[row] = best_idx;
    }
}

// Exact-winner threshold: argmax over the 128 partials, the winner row's
// stock GEMV, and the bf16 predecessor midpoint. (verbatim)
//   partial_max [128] f32, partial_idx [128] u32, lm_head [V*K] bf16,
//   x [K] bf16 -> threshold [1] f32
// Grid: 32 threads.
kernel void laguna_lmhead_exact_winner_bf16_midpoint_threshold_v1(
    const device float* partial_max [[buffer(0)]],
    const device uint32_t* partial_idx [[buffer(1)]],
    const device bfloat* lm_head [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device float* threshold [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]])
{
    constexpr uint VOCAB = 100352;
    constexpr uint K = 2048;
    constexpr uint READS = 4;
    threadgroup uint winner_row[1];

    float best = -metal::numeric_limits<float>::infinity();
    uint best_idx = 0xFFFFFFFFu;
    uint base = lid * READS;
    for (uint i = 0; i < READS; ++i) {
        float v = partial_max[base + i];
        uint idx = partial_idx[base + i];
        if (v > best || (v == best && idx < best_idx)) {
            best = v;
            best_idx = idx;
        }
    }
    for (ushort sn = 16; sn >= 1; sn >>= 1) {
        float ov = simd_shuffle_down(best, sn);
        uint oi = simd_shuffle_down(best_idx, sn);
        if (ov > best || (ov == best && oi < best_idx)) {
            best = ov;
            best_idx = oi;
        }
    }
    if (lid == 0) {
        winner_row[0] = metal::min(best_idx, uint(VOCAB - 1));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = winner_row[0];

    float result = 0.0f;
    thread bfloat inter[4];
    thread float v_coeff[4];
    uint bn = lid * 4;
    const device bfloat* mrow = lm_head + size_t(r) * K;
    for (uint i = 0; i < 16; ++i) {
        vec<bfloat, 4> xv =
            *((const device vec<bfloat, 4>*)(x + bn));
        v_coeff[0] = float(xv.x);
        v_coeff[1] = float(xv.y);
        v_coeff[2] = float(xv.z);
        v_coeff[3] = float(xv.w);
        vec<bfloat, 4> mv =
            *((const device vec<bfloat, 4>*)(mrow + bn));
        inter[0] = mv.x;
        inter[1] = mv.y;
        inter[2] = mv.z;
        inter[3] = mv.w;
        result += inter[0] * v_coeff[0];
        result += inter[1] * v_coeff[1];
        result += inter[2] * v_coeff[2];
        result += inter[3] * v_coeff[3];
        bn += 128;
    }
    for (ushort sn = 16; sn >= 1; sn >>= 1) {
        result += simd_shuffle_down(result, sn);
    }
    if (lid == 0) {
        bfloat rounded = bfloat(result);
        ushort bits = ushort(as_type<uint>(float(rounded)) >> 16);
        ushort magnitude = bits & 0x7FFFu;
        ushort predecessor_bits;
        if (magnitude == 0u) {
            predecessor_bits = 0x8001u;
        } else if ((bits & 0x8000u) == 0u) {
            predecessor_bits = bits - 1u;
        } else {
            predecessor_bits = bits + 1u;
        }
        float predecessor =
            as_type<float>(uint(predecessor_bits) << 16);
        float rounded_value = as_type<float>(uint(bits) << 16);
        threshold[0] = predecessor + (rounded_value - predecessor) * 0.5f;
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
// One-output-row scheduling twin of laguna_shared_nvfp4_swiglu_qmv_bf16_v1:
// each simdgroup owns one output row (tile * 2 + simd_group); arithmetic is
// textually identical per row. (verbatim from
// lagunaSharedSwiGLUQMVRows1Source(halved: false))
//   input        [2048] bf16            — shared-expert input
//   fused_weight [1024][1024] uint8     — 512 gate rows then 512 up rows
//   fused_scales [1024][128]  uint8     — E4M3 group-16 scales
//   activated    [512] bf16             — silu(gate) * up
// Grid: 256 groups x 64 threads (2 simdgroups; 2 rows per group).
kernel void laguna_shared_nvfp4_swiglu_qmv_rows1_bf16_v1(
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
    constexpr uint packed_row_bytes = 1024;
    constexpr uint scale_row_bytes = 128;
    constexpr uint block_width = 512;
    constexpr uint values_per_lane = 16;

    uint row = tile * 2 + simd_group;

    const device uint8_t* gate_row_weight =
        (const device uint8_t*)fused_weight +
        row * packed_row_bytes + lane * 8;
    const device uint8_t* up_row_weight =
        (const device uint8_t*)fused_weight +
        (row + output_width) * packed_row_bytes + lane * 8;
    const device uint8_t* gate_row_scale =
        fused_scales + row * scale_row_bytes + lane;
    const device uint8_t* up_row_scale =
        fused_scales + (row + output_width) * scale_row_bytes + lane;

    thread float gate_result = 0.0f;
    thread float up_result = 0.0f;
    thread float input_values[values_per_lane];

    for (uint block = 0; block < input_width; block += block_width) {
        const device vec<bfloat, 4>* input_vectors =
            (const device vec<bfloat, 4>*) (
                input + block + lane * values_per_lane);
        for (uint i = 0; i < values_per_lane / 4; ++i) {
            const vec<bfloat, 4> values = input_vectors[i];
            input_values[4 * i] = values[0];
            input_values[4 * i + 1] = values[1];
            input_values[4 * i + 2] = values[2];
            input_values[4 * i + 3] = values[3];
        }

        gate_result += laguna_nvfp4_qdot_16(
            gate_row_weight + block / 2,
            input_values,
            laguna_nvfp4_scale(gate_row_scale[block / 16]));
        up_result += laguna_nvfp4_qdot_16(
            up_row_weight + block / 2,
            input_values,
            laguna_nvfp4_scale(up_row_scale[block / 16]));
    }

    gate_result = simd_sum(gate_result);
    up_result = simd_sum(up_result);
    if (lane == 0) {
        bfloat gate = bfloat(gate_result * 4194304.0f);
        bfloat up = bfloat(up_result * 4194304.0f);
        bfloat exp_abs = metal::exp(metal::abs(gate));
        bfloat denominator = bfloat(1) + exp_abs;
        bfloat y = bfloat(1) / denominator;
        bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
        bfloat silu = bfloat(gate * sigmoid);
        activated[row] = bfloat(silu * up);
    }
}

// Group-32 halved scale-plane twin of the R1 shared kernel: one byte per 32
// weights behind the 128-byte patch header, so a simdgroup's 32 lanes read
// 16 contiguous bytes in place of 32. The fused plane concatenates gate over
// up, so the only two pairs the quantizer can leave unequal are gate row 0
// and up row 0, carried in header slots 0 and 1. (verbatim from
// lagunaSharedSwiGLUQMVRows1Source(halved: true))
//   fused_scales [128 + 1024*64] uint8 — halved group-32 plane
// Grid: 256 groups x 64 threads.
kernel void laguna_shared_nvfp4_swiglu_qmv_rows1_halved_bf16_v1(
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
    constexpr uint packed_row_bytes = 1024;
    constexpr uint scale_row_bytes = 64;
    constexpr uint scale_patch_bytes = 128;
    constexpr uint block_width = 512;
    constexpr uint values_per_lane = 16;

    uint row = tile * 2 + simd_group;

    const device uint8_t* gate_row_weight =
        (const device uint8_t*)fused_weight +
        row * packed_row_bytes + lane * 8;
    const device uint8_t* up_row_weight =
        (const device uint8_t*)fused_weight +
        (row + output_width) * packed_row_bytes + lane * 8;
    const device uint8_t* gate_row_scale =
        fused_scales + scale_patch_bytes + row * scale_row_bytes + (lane >> 1);
    const device uint8_t* up_row_scale =
        fused_scales + scale_patch_bytes
        + (row + output_width) * scale_row_bytes + (lane >> 1);

    thread float gate_result = 0.0f;
    thread float up_result = 0.0f;
    thread float input_values[values_per_lane];

    for (uint block = 0; block < input_width; block += block_width) {
        const device vec<bfloat, 4>* input_vectors =
            (const device vec<bfloat, 4>*) (
                input + block + lane * values_per_lane);
        for (uint i = 0; i < values_per_lane / 4; ++i) {
            const vec<bfloat, 4> values = input_vectors[i];
            input_values[4 * i] = values[0];
            input_values[4 * i + 1] = values[1];
            input_values[4 * i + 2] = values[2];
            input_values[4 * i + 3] = values[3];
        }

        gate_result += laguna_nvfp4_qdot_16(
            gate_row_weight + block / 2,
            input_values,
            laguna_nvfp4_scale(
                (row == 0 && block == 0 && lane == 1)
                ? fused_scales[0] : gate_row_scale[block / 32]));
        up_result += laguna_nvfp4_qdot_16(
            up_row_weight + block / 2,
            input_values,
            laguna_nvfp4_scale(
                (row == 0 && block == 0 && lane == 1)
                ? fused_scales[1] : up_row_scale[block / 32]));
    }

    gate_result = simd_sum(gate_result);
    up_result = simd_sum(up_result);
    if (lane == 0) {
        bfloat gate = bfloat(gate_result * 4194304.0f);
        bfloat up = bfloat(up_result * 4194304.0f);
        bfloat exp_abs = metal::exp(metal::abs(gate));
        bfloat denominator = bfloat(1) + exp_abs;
        bfloat y = bfloat(1) / denominator;
        bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
        bfloat silu = bfloat(gate * sigmoid);
        activated[row] = bfloat(silu * up);
    }
}

// Wide-codes twin of the halved R1 kernel: two adjacent groups per lane, one
// uint4 code load and one shared scale byte per pair, two K iterations
// instead of four. (verbatim from lagunaSharedSwiGLUQMVRows1WideKernel)
//   fused_scales [128 + 1024*64] uint8 — halved group-32 plane
// Grid: 256 groups x 64 threads.
kernel void laguna_shared_nvfp4_swiglu_qmv_rows1_halved_wide_bf16_v1(
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
    constexpr uint packed_row_bytes = 1024;
    constexpr uint scale_row_bytes = 64;
    constexpr uint scale_patch_bytes = 128;
    constexpr uint slab_width = 1024;
    constexpr uint values_per_lane = 32;

    uint row = tile * 2 + simd_group;

    const device uint8_t* gate_row_weight =
        (const device uint8_t*)fused_weight +
        row * packed_row_bytes + lane * 16;
    const device uint8_t* up_row_weight =
        (const device uint8_t*)fused_weight +
        (row + output_width) * packed_row_bytes + lane * 16;
    const device uint8_t* gate_row_scale =
        fused_scales + scale_patch_bytes + row * scale_row_bytes + lane;
    const device uint8_t* up_row_scale =
        fused_scales + scale_patch_bytes
        + (row + output_width) * scale_row_bytes + lane;

    thread float gate_result = 0.0f;
    thread float up_result = 0.0f;
    thread float input_values[values_per_lane];

    for (uint slab = 0; slab < input_width; slab += slab_width) {
        const device vec<bfloat, 4>* input_vectors =
            (const device vec<bfloat, 4>*) (
                input + slab + lane * values_per_lane);
        for (uint i = 0; i < values_per_lane / 4; ++i) {
            const vec<bfloat, 4> values = input_vectors[i];
            input_values[4 * i] = values[0];
            input_values[4 * i + 1] = values[1];
            input_values[4 * i + 2] = values[2];
            input_values[4 * i + 3] = values[3];
        }

        // The pair's halved byte covers both groups; the header byte restores
        // the one quantizer exception per plane (gate row 0 / up row 0,
        // pair 0, whose odd member is the second group of lane 0's first
        // pair).
        uint8_t gate_sb = gate_row_scale[slab / 32];
        uint8_t up_sb = up_row_scale[slab / 32];
        bool patch = row == 0 && slab == 0 && lane == 0;
        uint8_t gate_sb_b = patch ? fused_scales[0] : gate_sb;
        uint8_t up_sb_b = patch ? fused_scales[1] : up_sb;

        const uint4 gate_codes =
            *(const device uint4*)(gate_row_weight + slab / 2);
        const uint4 up_codes =
            *(const device uint4*)(up_row_weight + slab / 2);

        gate_result += laguna_nvfp4_qdot_codes_16(
            gate_codes.xy, input_values, laguna_nvfp4_scale(gate_sb));
        gate_result += laguna_nvfp4_qdot_codes_16(
            gate_codes.zw, input_values + 16, laguna_nvfp4_scale(gate_sb_b));
        up_result += laguna_nvfp4_qdot_codes_16(
            up_codes.xy, input_values, laguna_nvfp4_scale(up_sb));
        up_result += laguna_nvfp4_qdot_codes_16(
            up_codes.zw, input_values + 16, laguna_nvfp4_scale(up_sb_b));
    }

    gate_result = simd_sum(gate_result);
    up_result = simd_sum(up_result);
    if (lane == 0) {
        bfloat gate = bfloat(gate_result * 4194304.0f);
        bfloat up = bfloat(up_result * 4194304.0f);
        bfloat exp_abs = metal::exp(metal::abs(gate));
        bfloat denominator = bfloat(1) + exp_abs;
        bfloat y = bfloat(1) / denominator;
        bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
        bfloat silu = bfloat(gate * sigmoid);
        activated[row] = bfloat(silu * up);
    }
}

// Shared-expert down_proj + routed + residual adds, halved scale-plane twin
// of laguna_shared_nvfp4_down_residual_bf16_v1: one byte per 32 weights
// behind the 128-byte patch header; the flat pair 0 (output row 0, groups
// 0/1) is the only pair the quantizer can leave unequal, carried in header
// slot 0. (verbatim from lagunaSharedDownResidualSource(halved: true))
//   down_scales [128 + 2048*16] uint8 — halved group-32 down plane
// Grid: 256 groups x 64 threads (2 simdgroups x 4 rows = 8 rows/group).
kernel void laguna_shared_nvfp4_down_residual_halved_bf16_v1(
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
    constexpr uint scale_row_bytes = 16;
    constexpr uint scale_patch_bytes = 128;

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
            down_scales + scale_patch_bytes +
            output_row * scale_row_bytes + (lane >> 1);
        result[row] = laguna_nvfp4_qdot_16(
            weight,
            input_values,
            laguna_nvfp4_scale(
                (output_row == 0 && lane == 1)
                ? down_scales[0] : scale[0]));
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


// Simd-shuffle-only comparator-minimum extraction; lane `l` owns experts
// `l + 32j`, `mask` bit `j` marks extracted. Each routed slot performs only
// the rounds it needs and never waits on a cross-threadgroup selector.
METAL_FUNC uint laguna_router_top8_extract_round(
    thread const uint* keys, thread uint& mask, uint lane) {
    uint best_ordinal = 0xFFFFFFFFu;
    uint best_index = 256u;
    for (uint j = 0; j < 8; ++j) {
        if ((mask & (1u << j)) != 0u) continue;
        uint e = lane + 32u * j;
        uint o = keys[j];
        if (laguna_router_ordinal_before(o, e, best_ordinal, best_index)) {
            best_ordinal = o;
            best_index = e;
        }
    }
    // Transport the comparator's (ordinal, expert-index) state as one uint2
    // through each butterfly step. simd_shuffle_xor moves both components
    // bit-for-bit from the same source lane; comparator order is unchanged.
    uint2 best_pair = uint2(best_ordinal, best_index);
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        const uint2 other_pair = simd_shuffle_xor(best_pair, offset);
        if (laguna_router_ordinal_before(
            other_pair.x, other_pair.y, best_pair.x, best_pair.y)) {
            best_pair = other_pair;
        }
    }
    best_index = best_pair.y;
    if ((best_index & 31u) == lane) {
        mask |= 1u << (best_index >> 5u);
    }
    return best_index;
}

// R1 scheduling twin of laguna_routed_nvfp4_swiglu_qmv_bf16_v2: each
// simdgroup owns one output row rather than two; two simdgroups per 64-thread
// group and 256 tiles cover all 512 expert rows exactly once. (verbatim from
// lagunaRoutedSwiGLUQMVRows1Kernel)
//   input        [2048] bf16            — routed-expert input
//   fused_weight [E][1024][1024] uint8  — per-expert pair-interleaved
//                                         [gate; up] planes (E = 256)
//   fused_scales [E][1024][128]  uint8  — E4M3 group-16 scales
//   indices      [8] uint32             — top-8 routed expert ids
//   activated    [8][512] bf16          — per-slot silu(gate) * up
// Grid: 2048 groups (8 slots x 256 tiles) x 64 threads.
kernel void laguna_routed_nvfp4_swiglu_qmv_rows1_bf16_v1(
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
    constexpr uint tiles_per_expert = 256;
    constexpr uint routed_experts = 8;

    uint expert_slot = group % routed_experts;
    uint tile = group / routed_experts;
    uint expert = uint(indices[expert_slot]);
    uint logical_row = tile * 2 + simd_group;

    const device uint8_t* expert_weight =
        (const device uint8_t*)fused_weight +
        expert * packed_expert_bytes;
    const device uint8_t* expert_scales =
        fused_scales + expert * scale_expert_bytes;

    uint pair_tile = logical_row / 32;
    uint gate_row = pair_tile * 64 + logical_row % 32;
    uint up_row = gate_row + 32;
    const device uint8_t* gate_row_weight =
        expert_weight + gate_row * packed_row_bytes + lane * 8;
    const device uint8_t* up_row_weight =
        expert_weight + up_row * packed_row_bytes + lane * 8;
    const device uint8_t* gate_row_scale =
        expert_scales + gate_row * scale_row_bytes + lane;
    const device uint8_t* up_row_scale =
        expert_scales + up_row * scale_row_bytes + lane;

    thread float gate_result = 0.0f;
    thread float up_result = 0.0f;
    thread float input_values[values_per_lane];

    for (uint block = 0; block < input_width; block += block_width) {
        const device vec<bfloat, 4>* input_vectors =
            (const device vec<bfloat, 4>*) (
                input + block + lane * values_per_lane);
        for (uint i = 0; i < values_per_lane / 4; ++i) {
            const vec<bfloat, 4> values = input_vectors[i];
            input_values[4 * i] = values[0];
            input_values[4 * i + 1] = values[1];
            input_values[4 * i + 2] = values[2];
            input_values[4 * i + 3] = values[3];
        }

        const device uint8_t* gate_weight =
            gate_row_weight + block / 2;
        const device uint8_t* up_weight =
            up_row_weight + block / 2;
        const device uint8_t* gate_scale =
            gate_row_scale + block / 16;
        const device uint8_t* up_scale =
            up_row_scale + block / 16;

        gate_result += laguna_nvfp4_qdot_16(
            gate_weight,
            input_values,
            laguna_nvfp4_scale(gate_scale[0]));
        up_result += laguna_nvfp4_qdot_16(
            up_weight,
            input_values,
            laguna_nvfp4_scale(up_scale[0]));
    }

    gate_result = simd_sum(gate_result);
    up_result = simd_sum(up_result);
    if (lane == 0) {
        bfloat gate = bfloat(gate_result * 4194304.0f);
        bfloat up = bfloat(up_result * 4194304.0f);
        bfloat exp_abs = metal::exp(metal::abs(gate));
        bfloat denominator = bfloat(1) + exp_abs;
        bfloat y = bfloat(1) / denominator;
        bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
        bfloat silu = bfloat(gate * sigmoid);
        activated[expert_slot * output_width + logical_row] =
            bfloat(silu * up);
    }
}

// DARKBLOOM_PACKED_SCALES twin of laguna_routed_nvfp4_swiglu_qmv_bf16_v2
// consuming the walk-order scale side bank built by
// preparePackedRoutedGateUpBank: per expert `[tile 128][k-block 4][sub 8]
// [16 scale bytes]` behind the 128-byte patch header. (verbatim from
// lagunaRoutedSwiGLUQMVPackedKernel)
//   input         [2048] bf16            — routed-expert input
//   fused_weight  [E][1024][256] uint32  — per-expert fused code planes
//   packed_scales [128 + E*65536] uint8  — packed halved group-32 bank
//   indices       [8] uint32             — top-8 routed expert ids
//   activated     [8][512] bf16          — per-slot silu(gate) * up
// Grid: 1024 groups (8 experts x 128 tiles) x 64 threads.
kernel void laguna_routed_nvfp4_swiglu_qmv_packed_bf16_v1(
    const device bfloat* input [[buffer(0)]],
    const device uint8_t* fused_weight [[buffer(1)]],
    const device uint8_t* packed_scales [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    device bfloat* activated [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint input_width = 2048;
    constexpr uint output_width = 512;
    constexpr uint block_width = 512;
    constexpr uint values_per_lane = 16;
    constexpr uint routed_experts = 8;
    constexpr uint fused_row_bytes = 1024;
    constexpr uint fused_expert_bytes = 1024 * fused_row_bytes;
    constexpr uint scale_patch_bytes = 128;
    constexpr uint scale_row_bytes = 16;
    constexpr uint scale_sub_bytes = 8 * scale_row_bytes;
    constexpr uint scale_kblock_bytes = scale_sub_bytes;
    constexpr uint scale_tile_bytes = 4 * scale_kblock_bytes;
    constexpr uint packed_expert_bytes = 128 * scale_tile_bytes;

    uint expert_slot = group % routed_experts;
    uint tile = group / routed_experts;
    uint expert = uint(indices[expert_slot]);
    uint first_row = tile * 4 + simd_group * 2;

    const device uint8_t* expert_weight =
        (const device uint8_t*)fused_weight +
        expert * fused_expert_bytes;
    const device uint8_t* tile_scales =
        packed_scales + scale_patch_bytes
        + expert * packed_expert_bytes
        + tile * scale_tile_bytes;

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

        const device uint8_t* block_scales =
            tile_scales + (block / block_width) * scale_kblock_bytes;
        for (uint row = 0; row < 2; ++row) {
            uint logical_row = tile * 4 + simd_group * 2 + row;
            uint gate_row = (logical_row / 32) * 64 + logical_row % 32;
            uint up_row = gate_row + 32;
            uint sub = simd_group * 2 + row;
            const device uint8_t* gate_scale =
                block_scales + sub * 2 * scale_row_bytes + (lane >> 1);
            const device uint8_t* up_scale =
                gate_scale + scale_row_bytes;
            bool patch_lane = expert == 0 && logical_row == 0
                && block == 0 && lane == 1;
            uint8_t gate_sb =
                patch_lane ? packed_scales[0] : gate_scale[0];
            uint8_t up_sb = patch_lane ? packed_scales[1] : up_scale[0];
            const device uint8_t* gate_weight =
                expert_weight + gate_row * fused_row_bytes
                + block / 2 + lane * 8;
            const device uint8_t* up_weight =
                expert_weight + up_row * fused_row_bytes
                + block / 2 + lane * 8;

            gate_result[row] += laguna_nvfp4_qdot_16(
                gate_weight,
                input_values,
                laguna_nvfp4_scale(gate_sb));
            up_result[row] += laguna_nvfp4_qdot_16(
                up_weight,
                input_values,
                laguna_nvfp4_scale(up_sb));
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

// Packed routed QMV with an alternate expert-selection prologue: the expert
// is selected per slot by the simd-shuffle top-8 ordinal extraction over
// router_keys instead of the indices buffer. The gate/up body is identical
// to laguna_routed_nvfp4_swiglu_qmv_packed_bf16_v1. (verbatim from
// lagunaRoutedSwiGLUQMVPackedSelectedSource with the top-8 precomputed
// prelude)
//   router_keys [256] uint32 — per-expert ordinal keys from the router
// Grid: 1024 groups (8 experts x 128 tiles) x 64 threads.
kernel void laguna_routed_nvfp4_swiglu_qmv_packed_top8keys_bf16_v1(
    const device bfloat* input [[buffer(0)]],
    const device uint8_t* fused_weight [[buffer(1)]],
    const device uint8_t* packed_scales [[buffer(2)]],
    const device uint32_t* router_keys [[buffer(3)]],
    device bfloat* activated [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint input_width = 2048;
    constexpr uint output_width = 512;
    constexpr uint block_width = 512;
    constexpr uint values_per_lane = 16;
    constexpr uint routed_experts = 8;
    constexpr uint fused_row_bytes = 1024;
    constexpr uint fused_expert_bytes = 1024 * fused_row_bytes;
    constexpr uint scale_patch_bytes = 128;
    constexpr uint scale_row_bytes = 16;
    constexpr uint scale_sub_bytes = 8 * scale_row_bytes;
    constexpr uint scale_kblock_bytes = scale_sub_bytes;
    constexpr uint scale_tile_bytes = 4 * scale_kblock_bytes;
    constexpr uint packed_expert_bytes = 128 * scale_tile_bytes;

    uint expert_slot = group % routed_experts;
    uint tile = group / routed_experts;
    uint first_row = tile * 4 + simd_group * 2;
    thread uint top8_keys[8];
    for (uint j = 0; j < 8; ++j) {
        top8_keys[j] = router_keys[lane + 32u * j];
    }
    uint top8_mask = 0u;
    uint top8_winner = 0u;
    for (uint r = 0; r <= expert_slot; ++r) {
        top8_winner = laguna_router_top8_extract_round(
            top8_keys, top8_mask, lane);
    }
    uint expert = top8_winner;

    const device uint8_t* expert_weight =
        (const device uint8_t*)fused_weight + expert * fused_expert_bytes;
    const device uint8_t* tile_scales =
        packed_scales + scale_patch_bytes + expert * packed_expert_bytes
        + tile * scale_tile_bytes;

    thread float gate_result[2] = {0.0f, 0.0f};
    thread float up_result[2] = {0.0f, 0.0f};
    thread float input_values[values_per_lane];

    for (uint block = 0; block < input_width; block += block_width) {
        const device vec<bfloat, 4>* input_vectors =
            (const device vec<bfloat, 4>*) (
                input + block + lane * values_per_lane);
        for (uint i = 0; i < values_per_lane / 4; ++i) {
            const vec<bfloat, 4> values = input_vectors[i];
            input_values[4 * i] = values[0];
            input_values[4 * i + 1] = values[1];
            input_values[4 * i + 2] = values[2];
            input_values[4 * i + 3] = values[3];
        }

        const device uint8_t* block_scales =
            tile_scales + (block / block_width) * scale_kblock_bytes;
        for (uint row = 0; row < 2; ++row) {
            uint logical_row = tile * 4 + simd_group * 2 + row;
            uint gate_row = (logical_row / 32) * 64 + logical_row % 32;
            uint up_row = gate_row + 32;
            uint sub = simd_group * 2 + row;
            const device uint8_t* gate_scale =
                block_scales + sub * 2 * scale_row_bytes + (lane >> 1);
            const device uint8_t* up_scale = gate_scale + scale_row_bytes;
            const device uint8_t* gate_weight =
                expert_weight + gate_row * fused_row_bytes
                + block / 2 + lane * 8;
            const device uint8_t* up_weight =
                expert_weight + up_row * fused_row_bytes
                + block / 2 + lane * 8;

            bool patch_lane =
                expert == 0 && logical_row == 0 && block == 0 && lane == 1;
            uint8_t gate_sb = patch_lane ? packed_scales[0] : gate_scale[0];
            uint8_t up_sb = patch_lane ? packed_scales[1] : up_scale[0];

            gate_result[row] += laguna_nvfp4_qdot_16(
                gate_weight, input_values,
                laguna_nvfp4_scale(gate_sb));
            up_result[row] += laguna_nvfp4_qdot_16(
                up_weight, input_values,
                laguna_nvfp4_scale(up_sb));
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
            activated[expert_slot * output_width + first_row + row] =
                bfloat(silu * up);
        }
    }
}

// DARKBLOOM_ROUTED_GATEUP_R1 twin of the top-8 packed QMV: one output row
// per simdgroup (`bank_tile = logical_row / 4`, `sub = logical_row % 4`),
// with twice the threadgroups. (verbatim from
// lagunaRoutedSwiGLUQMVPackedTop8R1Kernel)
//   router_keys [256] uint32 — per-expert ordinal keys from the router
// Grid: 2048 groups (8 experts x 256 tiles) x 64 threads.
kernel void laguna_routed_nvfp4_swiglu_qmv_packed_top8keys_r1_bf16_v2(
    const device bfloat* input [[buffer(0)]],
    const device uint8_t* fused_weight [[buffer(1)]],
    const device uint8_t* packed_scales [[buffer(2)]],
    const device uint32_t* router_keys [[buffer(3)]],
    device bfloat* activated [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint input_width = 2048;
    constexpr uint output_width = 512;
    constexpr uint block_width = 512;
    constexpr uint values_per_lane = 16;
    constexpr uint routed_experts = 8;
    constexpr uint fused_row_bytes = 1024;
    constexpr uint fused_expert_bytes = 1024 * fused_row_bytes;
    constexpr uint scale_patch_bytes = 128;
    constexpr uint scale_row_bytes = 16;
    constexpr uint scale_sub_bytes = 8 * scale_row_bytes;
    constexpr uint scale_kblock_bytes = scale_sub_bytes;
    constexpr uint scale_tile_bytes = 4 * scale_kblock_bytes;
    constexpr uint packed_expert_bytes = 128 * scale_tile_bytes;

    uint expert_slot = group % routed_experts;
    uint tile = group / routed_experts;
    uint logical_row = tile * 2 + simd_group;
    thread uint top8_keys[8];
    for (uint j = 0; j < 8; ++j) {
        top8_keys[j] = router_keys[lane + 32u * j];
    }
    uint top8_mask = 0u;
    uint top8_winner = 0u;
    for (uint r = 0; r <= expert_slot; ++r) {
        top8_winner = laguna_router_top8_extract_round(
            top8_keys, top8_mask, lane);
    }
    uint expert = top8_winner;

    const device uint8_t* expert_weight =
        (const device uint8_t*)fused_weight + expert * fused_expert_bytes;
    const device uint8_t* row_scales =
        packed_scales + scale_patch_bytes + expert * packed_expert_bytes
        + (logical_row / 4) * scale_tile_bytes;
    uint sub = logical_row % 4;
    uint gate_row = (logical_row / 32) * 64 + logical_row % 32;
    uint up_row = gate_row + 32;

    thread float gate_result = 0.0f;
    thread float up_result = 0.0f;
    thread float input_values[values_per_lane];

    uint2 gate_codes;
    uint2 up_codes;
    uint8_t gate_sb;
    uint8_t up_sb;
    {
        const device uint8_t* first_scales =
            row_scales + sub * 2 * scale_row_bytes + (lane >> 1);
        bool patch_lane = expert == 0 && logical_row == 0 && lane == 1;
        gate_sb = patch_lane ? packed_scales[0] : first_scales[0];
        up_sb = patch_lane ? packed_scales[1] : first_scales[scale_row_bytes];
        gate_codes = *(const device uint2*)(
            expert_weight + gate_row * fused_row_bytes + lane * 8);
        up_codes = *(const device uint2*)(
            expert_weight + up_row * fused_row_bytes + lane * 8);
    }

    for (uint block = 0; block < input_width; block += block_width) {
        const device vec<bfloat, 4>* input_vectors =
            (const device vec<bfloat, 4>*) (
                input + block + lane * values_per_lane);
        for (uint i = 0; i < values_per_lane / 4; ++i) {
            const vec<bfloat, 4> values = input_vectors[i];
            input_values[4 * i] = values[0];
            input_values[4 * i + 1] = values[1];
            input_values[4 * i + 2] = values[2];
            input_values[4 * i + 3] = values[3];
        }

        const uint2 cur_gate_codes = gate_codes;
        const uint2 cur_up_codes = up_codes;
        const uint8_t cur_gate_sb = gate_sb;
        const uint8_t cur_up_sb = up_sb;
        const uint next_block = block + block_width;
        if (next_block < input_width) {
            const device uint8_t* next_scales =
                row_scales + (next_block / block_width) * scale_kblock_bytes
                + sub * 2 * scale_row_bytes + (lane >> 1);
            gate_sb = next_scales[0];
            up_sb = next_scales[scale_row_bytes];
            gate_codes = *(const device uint2*)(
                expert_weight + gate_row * fused_row_bytes
                + next_block / 2 + lane * 8);
            up_codes = *(const device uint2*)(
                expert_weight + up_row * fused_row_bytes
                + next_block / 2 + lane * 8);
        }

        gate_result += laguna_nvfp4_qdot_codes_16(
            cur_gate_codes, input_values,
            laguna_nvfp4_scale(cur_gate_sb));
        up_result += laguna_nvfp4_qdot_codes_16(
            cur_up_codes, input_values,
            laguna_nvfp4_scale(cur_up_sb));
    }

    gate_result = simd_sum(gate_result);
    up_result = simd_sum(up_result);
    if (lane == 0) {
        bfloat gate = bfloat(gate_result * 4194304.0f);
        bfloat up = bfloat(up_result * 4194304.0f);
        bfloat exp_abs = metal::exp(metal::abs(gate));
        bfloat denominator = bfloat(1) + exp_abs;
        bfloat y = bfloat(1) / denominator;
        bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
        bfloat silu = bfloat(gate * sigmoid);
        activated[expert_slot * output_width + logical_row] =
            bfloat(silu * up);
    }
}
