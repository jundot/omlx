# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exact multi-row QMV for DeepSeek DSpark target verification.

The Metal body preserves MLX ``qmv_fast``'s per-row arithmetic and 32-lane
reduction order.  It only changes the outer schedule: a simdgroup reuses each
weight byte across all verify rows before advancing K.  That makes M=2..6
bitwise equivalent to M independent decode calls without streaming the weight
matrix M times.  Six rows cover DSpark's five-draft verification block.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

_HEADER = r"""
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

struct OMLXFP8E4M3 {
    operator float16_t() {
        uint16_t value = (bits & 127) << 7;
        half converted = as_type<half>(value);
        converted *= 256.0;
        auto sign = bits & 128;
        return sign ? -converted : converted;
    }
    operator float() {
        return static_cast<float>(this->operator float16_t());
    }
    uint8_t bits;
};

struct OMLXFP8E8M0 {
    operator float() {
        uint32_t value = bits == 0
            ? 0x400000
            : (static_cast<uint32_t>(bits) << 23);
        return as_type<float>(value);
    }
    uint8_t bits;
};

inline float omlx_fp8_weight(uint8_t value) {
    return float(*(thread OMLXFP8E4M3*)(&value));
}

inline float omlx_fp8_scale(uint8_t value) {
    return float(*(thread OMLXFP8E8M0*)(&value));
}

template <int V>
inline float omlx_mxfp8_dot(
    const device uint8_t* weight,
    const thread float* input,
    float scale) {
    float accum = 0.0f;
    for (int idx = 0; idx < V; ++idx) {
        accum += input[idx] * omlx_fp8_weight(weight[idx]);
    }
    return scale * accum;
}
"""

_SOURCE = r"""
    // Matches MLX qmv_fast for 8-bit modes: two uint32 packs (8 values)
    // per lane, 32 lanes, two simdgroups, four output rows per simdgroup.
    constexpr int VALUES_PER_THREAD = 8;
    constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
    constexpr int RESULTS_PER_SIMDGROUP = 4;
    constexpr int OUTPUTS_PER_THREADGROUP = 8;

    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int output_base =
        int(threadgroup_position_in_grid.y) * OUTPUTS_PER_THREADGROUP
        + int(simd_group) * RESULTS_PER_SIMDGROUP;
    if (output_base >= N) {
        return;
    }

    const device uint8_t* weight_ptr = (const device uint8_t*)weight
        + output_base * K + int(lane) * VALUES_PER_THREAD;
    const device T* input_ptr = (const device T*)input
        + int(lane) * VALUES_PER_THREAD;
    device T* output_ptr = (device T*)output + output_base;

    float accum[M][RESULTS_PER_SIMDGROUP] = {{0}};

    if constexpr (MXFP8) {
        const device uint8_t* scale_ptr = (const device uint8_t*)scales
            + output_base * (K / GS)
            + int(lane) / (GS / VALUES_PER_THREAD);

        for (int k = 0; k < K; k += BLOCK_SIZE) {
            float input_values[M][VALUES_PER_THREAD];
            for (int row = 0; row < M; ++row) {
                for (int idx = 0; idx < VALUES_PER_THREAD; ++idx) {
                    input_values[row][idx] = float(input_ptr[row * K + idx]);
                }
            }
            for (int result = 0; result < RESULTS_PER_SIMDGROUP; ++result) {
                const device uint8_t* row_weight = weight_ptr + result * K;
                const device uint8_t* row_scale =
                    scale_ptr + result * (K / GS);
                const float scale = omlx_fp8_scale(row_scale[0]);
                for (int row = 0; row < M; ++row) {
                    accum[row][result] += omlx_mxfp8_dot<VALUES_PER_THREAD>(
                        row_weight, input_values[row], scale);
                }
            }
            weight_ptr += BLOCK_SIZE;
            scale_ptr += BLOCK_SIZE / GS;
            input_ptr += BLOCK_SIZE;
        }
    } else {
        const device T* scale_ptr = (const device T*)scales
            + output_base * (K / GS)
            + int(lane) / (GS / VALUES_PER_THREAD);
        const device T* bias_ptr = (const device T*)biases
            + output_base * (K / GS)
            + int(lane) / (GS / VALUES_PER_THREAD);

        for (int k = 0; k < K; k += BLOCK_SIZE) {
            float input_values[M][VALUES_PER_THREAD];
            float input_sums[M] = {0};
            for (int row = 0; row < M; ++row) {
                for (int idx = 0; idx < VALUES_PER_THREAD; ++idx) {
                    const float value = float(input_ptr[row * K + idx]);
                    input_values[row][idx] = value;
                    input_sums[row] += value;
                }
            }
            for (int result = 0; result < RESULTS_PER_SIMDGROUP; ++result) {
                const device uint8_t* row_weight = weight_ptr + result * K;
                const device T* row_scale = scale_ptr + result * (K / GS);
                const device T* row_bias = bias_ptr + result * (K / GS);
                const float scale = float(row_scale[0]);
                const float bias = float(row_bias[0]);
                for (int row = 0; row < M; ++row) {
                    float dot = 0.0f;
                    for (int idx = 0; idx < VALUES_PER_THREAD; ++idx) {
                        dot += input_values[row][idx] * float(row_weight[idx]);
                    }
                    accum[row][result] +=
                        scale * dot + input_sums[row] * bias;
                }
            }
            weight_ptr += BLOCK_SIZE;
            scale_ptr += BLOCK_SIZE / GS;
            bias_ptr += BLOCK_SIZE / GS;
            input_ptr += BLOCK_SIZE;
        }
    }

    for (int row = 0; row < M; ++row) {
        for (int result = 0; result < RESULTS_PER_SIMDGROUP; ++result) {
            const float value = simd_sum(accum[row][result]);
            if (lane == 0) {
                output_ptr[row * N + result] = T(value);
            }
        }
    }
"""

_MULTI_SOURCE = r"""
    constexpr int VALUES_PER_THREAD = 8;
    constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
    constexpr int RESULTS_PER_SIMDGROUP = 4;
    constexpr int OUTPUTS_PER_THREADGROUP = 8;
    constexpr int BLOCKS_PER_GROUP = N / OUTPUTS_PER_THREADGROUP;

    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int flat_block = int(threadgroup_position_in_grid.y);
    const int group = flat_block / BLOCKS_PER_GROUP;
    const int group_block = flat_block - group * BLOCKS_PER_GROUP;
    const int output_base =
        group_block * OUTPUTS_PER_THREADGROUP
        + int(simd_group) * RESULTS_PER_SIMDGROUP;

    const device uint8_t* weight_ptr = (const device uint8_t*)weight
        + (group * N + output_base) * K
        + int(lane) * VALUES_PER_THREAD;
    const device T* input_ptr = (const device T*)input
        + group * M * K + int(lane) * VALUES_PER_THREAD;
    device T* output_ptr = (device T*)output
        + group * M * N + output_base;
    const device uint8_t* scale_ptr = (const device uint8_t*)scales
        + (group * N + output_base) * (K / GS)
        + int(lane) / (GS / VALUES_PER_THREAD);

    float accum[M][RESULTS_PER_SIMDGROUP] = {{0}};
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        float input_values[M][VALUES_PER_THREAD];
        for (int row = 0; row < M; ++row) {
            for (int idx = 0; idx < VALUES_PER_THREAD; ++idx) {
                input_values[row][idx] = float(input_ptr[row * K + idx]);
            }
        }
        for (int result = 0; result < RESULTS_PER_SIMDGROUP; ++result) {
            const device uint8_t* row_weight = weight_ptr + result * K;
            const device uint8_t* row_scale =
                scale_ptr + result * (K / GS);
            const float scale = omlx_fp8_scale(row_scale[0]);
            for (int row = 0; row < M; ++row) {
                accum[row][result] += omlx_mxfp8_dot<VALUES_PER_THREAD>(
                    row_weight, input_values[row], scale);
            }
        }
        weight_ptr += BLOCK_SIZE;
        scale_ptr += BLOCK_SIZE / GS;
        input_ptr += BLOCK_SIZE;
    }

    for (int row = 0; row < M; ++row) {
        for (int result = 0; result < RESULTS_PER_SIMDGROUP; ++result) {
            const float value = simd_sum(accum[row][result]);
            if (lane == 0) {
                output_ptr[row * N + result] = T(value);
            }
        }
    }
"""

_DENSE_SOURCE = r"""
    // Matches MLX GEMVKernel<T, 8, 1, 1, 32, 4, 4>: eight simdgroups,
    // four output rows per simdgroup, and four K values per lane. The only
    // added dimension is M independent accumulators sharing each weight load.
    constexpr int THREAD_ROWS = 4;
    constexpr int THREAD_COLS = 4;
    constexpr int OUTPUTS_PER_THREADGROUP = 32;
    constexpr int K_BLOCK = 128;

    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int output_base =
        int(threadgroup_position_in_grid.x) * OUTPUTS_PER_THREADGROUP
        + int(simd_group) * THREAD_ROWS;
    if (output_base >= N) {
        return;
    }

    int k_base = int(lane) * THREAD_COLS;
    float accum[M][THREAD_ROWS] = {{0}};
    for (int block = 0; block < K; block += K_BLOCK) {
        float input_values[M][THREAD_COLS];
        for (int row = 0; row < M; ++row) {
            for (int col = 0; col < THREAD_COLS; ++col) {
                input_values[row][col] =
                    float(input[row * K + k_base + col]);
            }
        }

        for (int result = 0; result < THREAD_ROWS; ++result) {
            const device T* row_weight =
                weight + (output_base + result) * K + k_base;
            float weight_values[THREAD_COLS];
            for (int col = 0; col < THREAD_COLS; ++col) {
                weight_values[col] = float(row_weight[col]);
            }
            for (int row = 0; row < M; ++row) {
                for (int col = 0; col < THREAD_COLS; ++col) {
                    accum[row][result] +=
                        weight_values[col] * input_values[row][col];
                }
            }
        }
        k_base += K_BLOCK;
    }

    for (int row = 0; row < M; ++row) {
        for (int result = 0; result < THREAD_ROWS; ++result) {
            for (ushort step = 16; step >= 1; step >>= 1) {
                accum[row][result] +=
                    simd_shuffle_down(accum[row][result], step);
            }
            if (lane == 0) {
                output[row * N + output_base + result] =
                    T(accum[row][result]);
            }
        }
    }
"""

# DSpark's reference head promotes the checkpoint's BF16 values to FP32 before
# the projection. Reuse the same GEMV schedule while retaining BF16 storage;
# only the output store differs from the decode-exact BF16 path above.
_DENSE_FP32_SOURCE = _DENSE_SOURCE.replace(
    "T(accum[row][result])", "accum[row][result]"
)


@lru_cache(maxsize=3)
def _kernel(results_per_simdgroup: int = 4):
    if results_per_simdgroup not in (1, 2, 4):
        raise ValueError("verify QMV result tile must be 1, 2, or 4")
    source = _SOURCE
    if results_per_simdgroup != 4:
        source = source.replace(
            "constexpr int RESULTS_PER_SIMDGROUP = 4;",
            f"constexpr int RESULTS_PER_SIMDGROUP = {results_per_simdgroup};",
        ).replace(
            "constexpr int OUTPUTS_PER_THREADGROUP = 8;",
            "constexpr int OUTPUTS_PER_THREADGROUP = "
            f"{2 * results_per_simdgroup};",
        )
    return mx.fast.metal_kernel(
        name=f"omlx_deepseek_verify_qmv_exact_r{results_per_simdgroup}",
        input_names=["input", "weight", "scales", "biases"],
        output_names=["output"],
        header=_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _dense_kernel():
    return mx.fast.metal_kernel(
        name="omlx_deepseek_verify_gemv_exact",
        input_names=["input", "weight"],
        output_names=["output"],
        source=_DENSE_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _dense_fp32_kernel():
    return mx.fast.metal_kernel(
        name="omlx_deepseek_dspark_head_fp32",
        input_names=["input", "weight"],
        output_names=["output"],
        source=_DENSE_FP32_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=3)
def _multi_kernel(results_per_simdgroup: int = 4):
    if results_per_simdgroup not in (1, 2, 4):
        raise ValueError("verify grouped QMV result tile must be 1, 2, or 4")
    source = _MULTI_SOURCE
    if results_per_simdgroup != 4:
        source = source.replace(
            "constexpr int RESULTS_PER_SIMDGROUP = 4;",
            f"constexpr int RESULTS_PER_SIMDGROUP = {results_per_simdgroup};",
        ).replace(
            "constexpr int OUTPUTS_PER_THREADGROUP = 8;",
            "constexpr int OUTPUTS_PER_THREADGROUP = "
            f"{2 * results_per_simdgroup};",
        )
    return mx.fast.metal_kernel(
        name=f"omlx_deepseek_verify_multi_qmv_exact_r{results_per_simdgroup}",
        input_names=["input", "weight", "scales"],
        output_names=["output"],
        header=_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def eligible(module, inputs: mx.array) -> bool:
    """Return whether ``module(inputs)`` has an exact fast-kernel layout."""
    if inputs.ndim < 2 or module.weight.ndim != 2:
        return False
    input_dims = int(inputs.shape[-1])
    rows = int(inputs.size) // input_dims
    output_dims = int(module.scales.shape[0])
    mode = getattr(module, "mode", "affine")
    return (
        2 <= rows <= 6
        and int(module.bits) == 8
        and mode in ("mxfp8", "affine")
        and int(module.group_size) in (32, 64, 128)
        and inputs.dtype in (mx.bfloat16, mx.float16)
        and input_dims % 512 == 0
        and output_dims >= 2048
        and output_dims % 8 == 0
    )


def exact_verify_qmv(module, inputs: mx.array) -> mx.array:
    """Apply an eligible quantized linear with decode-identical reductions."""
    input_dims = int(inputs.shape[-1])
    rows = int(inputs.size) // input_dims
    output_dims = int(module.scales.shape[0])
    flat = inputs.reshape(rows, input_dims)
    mode = getattr(module, "mode", "affine")
    biases = module.get("biases")
    if biases is None:
        biases = module.scales

    results_per_simdgroup = 4
    if input_dims == 8192 and output_dims == 4096 and rows == 6:
        try:
            device_name = str(mx.device_info().get("device_name", ""))
            if device_name == "Apple M3 Ultra":
                results_per_simdgroup = 1
            elif device_name == "Apple M5 Max":
                results_per_simdgroup = 2
        except Exception:
            pass
    elif input_dims == 1024 and rows == 6 and output_dims in (12288, 20480):
        try:
            if str(mx.device_info().get("device_name", "")) == "Apple M3 Ultra":
                results_per_simdgroup = 1
        except Exception:
            pass

    (output,) = _kernel(results_per_simdgroup)(
        inputs=[flat, module.weight, module.scales, biases],
        template=[
            ("T", inputs.dtype),
            ("M", rows),
            ("K", input_dims),
            ("N", output_dims),
            ("GS", int(module.group_size)),
            ("MXFP8", mode == "mxfp8"),
        ],
        # mx.fast uses thread-grid dimensions. This is one (32, 2, 1)
        # threadgroup for each eight output rows.
        grid=(32, output_dims // results_per_simdgroup, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, output_dims)],
        output_dtypes=[inputs.dtype],
    )
    output = output.reshape((*inputs.shape[:-1], output_dims))
    if "bias" in module:
        output = output + module.bias
    return output


def exact_verify_mxfp8_bank(
    weight: mx.array,
    scales: mx.array,
    inputs: mx.array,
) -> mx.array:
    """Apply the prepared DS4 Q-A/raw-KV bank with target-row reductions."""

    input_dims = int(inputs.shape[-1])
    rows = int(inputs.size) // input_dims
    output_dims = int(scales.shape[0])
    if (
        weight.ndim != 2
        or scales.ndim != 2
        or weight.dtype != mx.uint32
        or scales.dtype != mx.uint8
        or inputs.dtype not in (mx.bfloat16, mx.float16)
        or not 2 <= rows <= 6
        or input_dims != 4096
        or output_dims != 1536
        or int(weight.shape[0]) != output_dims
        or int(weight.shape[1]) * 4 != input_dims
        or int(scales.shape[1]) * 32 != input_dims
    ):
        raise ValueError("DS4 verify MXFP8 bank has an unsupported shape")
    flat = inputs.reshape(rows, input_dims)
    results_per_simdgroup = 1
    (output,) = _kernel(results_per_simdgroup)(
        inputs=[flat, weight, scales, scales],
        template=[
            ("T", inputs.dtype),
            ("M", rows),
            ("K", input_dims),
            ("N", output_dims),
            ("GS", 32),
            ("MXFP8", True),
        ],
        grid=(32, output_dims // results_per_simdgroup, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, output_dims)],
        output_dtypes=[inputs.dtype],
    )
    return output.reshape((*inputs.shape[:-1], output_dims))


def pair_eligible(module_a, module_b, inputs: mx.array) -> bool:
    """Return whether two MXFP8 linears can share the exact native grid."""
    required = ("bits", "group_size", "mode", "weight", "scales")
    if any(
        not all(hasattr(module, name) for name in required)
        for module in (module_a, module_b)
    ):
        return False
    if inputs.ndim < 2 or module_a.weight.ndim != 2 or module_b.weight.ndim != 2:
        return False
    input_dims = int(inputs.shape[-1])
    rows = int(inputs.size) // input_dims
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast

        native = fast.has_symbol("dspark_exact_mxfp8_qmv_pair")
    except Exception:
        native = False
    return (
        native
        and 2 <= rows <= 6
        and inputs.dtype in (mx.bfloat16, mx.float16)
        and input_dims % 512 == 0
        and int(module_a.bits) == int(module_b.bits) == 8
        and int(module_a.group_size) == int(module_b.group_size) == 32
        and getattr(module_a, "mode", "affine") == "mxfp8"
        and getattr(module_b, "mode", "affine") == "mxfp8"
        and module_a.weight.shape == module_b.weight.shape
        and module_a.scales.shape == module_b.scales.shape
        and module_a.weight.dtype == module_b.weight.dtype == mx.uint32
        and module_a.scales.dtype == module_b.scales.dtype == mx.uint8
        and int(module_a.scales.shape[1]) * 32 == input_dims
        and int(module_a.scales.shape[0]) % 8 == 0
    )


def exact_verify_qmv_pair(module_a, module_b, inputs: mx.array):
    """Apply two MXFP8 linears with independent M=1-identical reductions."""
    from omlx.custom_kernels.glm_moe_dsa import fast

    input_dims = int(inputs.shape[-1])
    rows = int(inputs.size) // input_dims
    output_dims = int(module_a.scales.shape[0])
    flat = mx.contiguous(inputs.reshape(rows, input_dims))
    paired = fast.dspark_exact_mxfp8_qmv_pair(
        flat,
        module_a.weight,
        module_a.scales,
        module_b.weight,
        module_b.scales,
    )
    paired = paired.reshape((2, *inputs.shape[:-1], output_dims))
    output_a, output_b = paired[0], paired[1]
    if "bias" in module_a:
        output_a = output_a + module_a.bias
    if "bias" in module_b:
        output_b = output_b + module_b.bias
    return output_a, output_b


def multi_eligible(module, inputs: mx.array) -> bool:
    """Return whether a grouped MultiLinear has the exact shared-row path."""
    if inputs.ndim < 3 or module.weight.ndim != 3:
        return False
    groups = int(module.scales.shape[0])
    input_dims = int(inputs.shape[-1])
    rows = int(inputs.size) // (groups * input_dims)
    output_dims = int(module.scales.shape[1])
    return (
        2 <= rows <= 6
        and int(module.bits) == 8
        and getattr(module, "mode", "affine") == "mxfp8"
        and int(module.group_size) == 32
        and inputs.dtype in (mx.bfloat16, mx.float16)
        and input_dims % 512 == 0
        and output_dims % 8 == 0
    )


def exact_verify_multi_qmv(module, inputs: mx.array) -> mx.array:
    """Apply grouped mxfp8 projection with decode-identical reductions."""
    groups = int(module.scales.shape[0])
    input_dims = int(inputs.shape[-1])
    rows = int(inputs.size) // (groups * input_dims)
    output_dims = int(module.scales.shape[1])
    grouped = inputs.reshape(groups, rows, input_dims)
    results_per_simdgroup = 4
    if groups == 8 and rows == 6 and output_dims == 1024 and input_dims in (
        1536,
        2560,
    ):
        try:
            device_name = str(mx.device_info().get("device_name", ""))
            if device_name == "Apple M3 Ultra":
                results_per_simdgroup = 1
            elif device_name == "Apple M5 Max":
                results_per_simdgroup = 2
        except Exception:
            pass
    (output,) = _multi_kernel(results_per_simdgroup)(
        inputs=[grouped, module.weight, module.scales],
        template=[
            ("T", inputs.dtype),
            ("M", rows),
            ("K", input_dims),
            ("N", output_dims),
            ("GS", int(module.group_size)),
        ],
        grid=(32, groups * output_dims // results_per_simdgroup, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(groups, rows, output_dims)],
        output_dtypes=[inputs.dtype],
    )
    return output


def dense_eligible(module, inputs: mx.array) -> bool:
    """Return whether a dense linear has MLX's large-output GEMV geometry."""
    if inputs.ndim < 2 or module.weight.ndim != 2:
        return False
    input_dims = int(inputs.shape[-1])
    rows = int(inputs.size) // input_dims
    output_dims = int(module.weight.shape[0])
    return (
        2 <= rows <= 6
        and inputs.dtype in (mx.bfloat16, mx.float16)
        and module.weight.dtype == inputs.dtype
        and input_dims % 128 == 0
        and output_dims >= 4096
        and output_dims % 32 == 0
    )


def exact_verify_gemv(module, inputs: mx.array) -> mx.array:
    """Apply a dense linear with MLX M=1 GEMV's exact reduction order."""
    input_dims = int(inputs.shape[-1])
    rows = int(inputs.size) // input_dims
    output_dims = int(module.weight.shape[0])
    flat = inputs.reshape(rows, input_dims)
    (output,) = _dense_kernel()(
        inputs=[flat, module.weight],
        template=[
            ("T", inputs.dtype),
            ("M", rows),
            ("K", input_dims),
            ("N", output_dims),
        ],
        grid=((output_dims // 32) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, output_dims)],
        output_dtypes=[inputs.dtype],
    )
    output = output.reshape((*inputs.shape[:-1], output_dims))
    if "bias" in module:
        output = output + module.bias
    return output


def dspark_head_gemv(module, inputs: mx.array) -> mx.array:
    """Project a DSpark head with BF16 storage and FP32 arithmetic/output."""
    input_dims = int(inputs.shape[-1])
    rows = int(inputs.size) // input_dims
    output_dims = int(module.weight.shape[0])
    if not (
        1 <= rows <= 6
        and inputs.dtype in (mx.bfloat16, mx.float16)
        and module.weight.dtype == inputs.dtype
        and input_dims % 128 == 0
        and output_dims >= 4096
        and output_dims % 32 == 0
    ):
        return module(inputs.astype(mx.float32))

    flat = inputs.reshape(rows, input_dims)
    (output,) = _dense_fp32_kernel()(
        inputs=[flat, module.weight],
        template=[
            ("T", inputs.dtype),
            ("M", rows),
            ("K", input_dims),
            ("N", output_dims),
        ],
        grid=((output_dims // 32) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, output_dims)],
        output_dtypes=[mx.float32],
    )
    output = output.reshape((*inputs.shape[:-1], output_dims))
    if "bias" in module:
        output = output + module.bias.astype(mx.float32)
    gather = getattr(module, "_gather_logits", None)
    if callable(gather):
        output = gather(output)
    return output


__all__ = [
    "dense_eligible",
    "dspark_head_gemv",
    "eligible",
    "exact_verify_qmv_pair",
    "exact_verify_gemv",
    "exact_verify_multi_qmv",
    "exact_verify_qmv",
    "multi_eligible",
    "pair_eligible",
]
