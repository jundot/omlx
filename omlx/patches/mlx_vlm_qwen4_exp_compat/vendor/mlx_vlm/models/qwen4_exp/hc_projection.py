# SPDX-License-Identifier: Apache-2.0

"""Lossless one-dispatch Qwen4 hyperconnection input projection.

The Metal arithmetic in this module is transcribed from MLX core at pinned
commit ``ceab91938`` (``mlx/backend/metal/kernels/quantized.h``): its first
320 rows use MLX's literal ``qmv_fast`` traversal and its final four rows use
the literal standalone-N=4 general ``qmv`` traversal.  This preserves both raw
projection outputs while avoiding the two-dispatch canonical implementation.

MLX is Copyright © 2023 Apple Inc. and licensed under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_HC_COUNT = 4
_HIDDEN_SIZE = 2560
_STREAM_WIDTH = _HC_COUNT * _HIDDEN_SIZE
_LOW_RANK = 320
_GROUP_SIZE = 64
_SUPPORTED_BITS = (4, 5, 6, 8)
_DISABLED = os.environ.get("OMLX_QWEN4_HC_HYBRID", "1").strip().lower() in {
    "0",
    "false",
    "off",
    "no",
}
_KERNEL = None
_RUNTIME_FAILED = False
_FAILURE_LOGGED = False


_HEADER = r"""
using namespace metal;

template <int bits>
inline constexpr short hc_pack_factor() {
    return bits == 5 ? 8 : (bits == 6 ? 4 : 32 / bits);
}

template <int bits>
inline constexpr short hc_bytes_per_pack() {
    constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
    return power_of_2_bits ? 4 : (bits == 5 ? 5 : 3);
}

template <typename T, int N, int bits>
inline float hc_load_vector(const device T* x, thread float* xt) {
    float sum = 0.0f;
    if (bits == 4) {
        for (int i = 0; i < N; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 16.0f;
            xt[i + 2] = x[i + 2] / 256.0f;
            xt[i + 3] = x[i + 3] / 4096.0f;
        }
    } else if (bits == 5) {
        for (int i = 0; i < N; i += 8) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3]
                + x[i + 4] + x[i + 5] + x[i + 6] + x[i + 7];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 32.0f;
            xt[i + 2] = x[i + 2] / 4.0f;
            xt[i + 3] = x[i + 3] / 128.0f;
            xt[i + 4] = x[i + 4] / 16.0f;
            xt[i + 5] = x[i + 5] / 2.0f;
            xt[i + 6] = x[i + 6] / 64.0f;
            xt[i + 7] = x[i + 7] / 8.0f;
        }
    } else if (bits == 6) {
        for (int i = 0; i < N; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 64.0f;
            xt[i + 2] = x[i + 2] / 16.0f;
            xt[i + 3] = x[i + 3] / 4.0f;
        }
    } else if (bits == 8) {
        for (int i = 0; i < N; ++i) {
            sum += x[i];
            xt[i] = x[i];
        }
    }
    return sum;
}

template <int N, int bits>
inline float hc_qdot(
    const device uint8_t* w,
    const thread float* xt,
    float scale,
    float bias,
    float sum) {
    float accum = 0.0f;
    if (bits == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < N / 4; ++i) {
            accum +=
                (xt[4 * i] * (ws[i] & 0x000f)
                 + xt[4 * i + 1] * (ws[i] & 0x00f0)
                 + xt[4 * i + 2] * (ws[i] & 0x0f00)
                 + xt[4 * i + 3] * (ws[i] & 0xf000));
        }
    } else if (bits == 5) {
        for (int i = 0; i < N / 8; ++i) {
            xt += 8 * i;
            w += 5 * i;
            accum += (w[0] & 0x1f) * xt[0];
            accum += (w[0] & 0xe0) * xt[1];
            accum += (w[1] & 0x3) * (xt[1] * 256.0f);
            accum += (w[1] & 0x7c) * xt[2];
            accum += (w[1] & 0x80) * xt[3];
            accum += (w[2] & 0xf) * (xt[3] * 256.0f);
            accum += (w[2] & 0xf0) * xt[4];
            accum += (w[3] & 0x1) * (xt[4] * 256.0f);
            accum += (w[3] & 0x3e) * xt[5];
            accum += (w[3] & 0xc0) * xt[6];
            accum += (w[4] & 0x7) * (xt[6] * 256.0f);
            accum += (w[4] & 0xf8) * xt[7];
        }
    } else if (bits == 6) {
        for (int i = 0; i < N / 4; ++i) {
            xt += 4 * i;
            w += 3 * i;
            accum += (w[0] & 0x3f) * xt[0];
            accum += (w[0] & 0xc0) * xt[1];
            accum += (w[1] & 0x0f) * (xt[1] * 256.0f);
            accum += (w[1] & 0xf0) * xt[2];
            accum += (w[2] & 0x03) * (xt[2] * 256.0f);
            accum += (w[2] & 0xfc) * xt[3];
        }
    } else if (bits == 8) {
        for (int i = 0; i < N; ++i) accum += xt[i] * w[i];
    }
    return scale * accum + sum * bias;
}

template <int N, int bits>
inline float hc_qdot_safe(
    const device uint8_t* w,
    const thread float* xt,
    float scale,
    float bias,
    float sum) {
    float accum = 0.0f;
    if (bits == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < N / 4; ++i) {
            accum +=
                (xt[4 * i] * (ws[i] & 0x000f)
                 + xt[4 * i + 1] * (ws[i] & 0x00f0)
                 + xt[4 * i + 2] * (ws[i] & 0x0f00)
                 + xt[4 * i + 3] * (ws[i] & 0xf000));
        }
    } else if (bits == 5) {
        for (int i = 0; i < N / 8; ++i) {
            xt += 8 * i;
            w += 5 * i;
            accum += (w[0] & 0x1f) * xt[0];
            accum += (w[0] & 0xe0) * xt[1];
            accum += (w[1] & 0x3) * (xt[1] * 256.0f);
            accum += (w[1] & 0x7c) * xt[2];
            accum += (w[1] & 0x80) * xt[3];
            accum += (w[2] & 0xf) * (xt[3] * 256.0f);
            accum += (w[2] & 0xf0) * xt[4];
            accum += (w[3] & 0x1) * (xt[4] * 256.0f);
            accum += (w[3] & 0x3e) * xt[5];
            accum += (w[3] & 0xc0) * xt[6];
            accum += (w[4] & 0x7) * (xt[6] * 256.0f);
            accum += (w[4] & 0xf8) * xt[7];
        }
    } else if (bits == 6) {
        for (int i = 0; i < N / 4; ++i) {
            xt += 4 * i;
            w += 3 * i;
            accum += (w[0] & 0x3f) * xt[0];
            accum += (w[0] & 0xc0) * xt[1];
            accum += (w[1] & 0x0f) * (xt[1] * 256.0f);
            accum += (w[1] & 0xf0) * xt[2];
            accum += (w[2] & 0x03) * (xt[2] * 256.0f);
            accum += (w[2] & 0xfc) * xt[3];
        }
    } else if (bits == 8) {
        for (int i = 0; i < N; ++i) accum += xt[i] * w[i];
    }
    return scale * accum + sum * bias;
}
"""


_SOURCE = r"""
    const uint tg = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr int PF = hc_pack_factor<BITS>();
    constexpr int BP = hc_bytes_per_pack<BITS>();
    constexpr int GROUPS = K / 64;
    constexpr int ROW_BYTES = K * BP / PF;

    if (tg < 40) {
        constexpr int PPT = 2;
        constexpr int VPT = PF * PPT;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = 64 / VPT;
        const int out_row = int(tg) * 8 + int(sg) * 4;
        const device uint8_t* wp = (const device uint8_t*)down_w
            + out_row * ROW_BYTES + int(lane) * PPT * BP;
        const device T* sp = down_s + out_row * GROUPS
            + int(lane) / SCALE_STEP;
        const device T* bp = down_b + out_row * GROUPS
            + int(lane) / SCALE_STEP;
        const device T* xp = x + int(lane) * VPT;
        float result[4] = {0.0f};
        float xv[VPT];
        for (int k = 0; k < K; k += BLOCK) {
            float sum = hc_load_vector<T, VPT, BITS>(xp, xv);
            for (int row = 0; row < 4; ++row) {
                result[row] += hc_qdot<VPT, BITS>(
                    wp + row * ROW_BYTES,
                    xv,
                    float(sp[row * GROUPS]),
                    float(bp[row * GROUPS]),
                    sum);
            }
            wp += BLOCK * BP / PF;
            sp += BLOCK / 64;
            bp += BLOCK / 64;
            xp += BLOCK;
        }
        for (int row = 0; row < 4; ++row) {
            result[row] = simd_sum(result[row]);
            if (lane == 0) combined[out_row + row] = T(result[row]);
        }
        return;
    }

    if (sg != 0) return;
    constexpr int PPT = 1;
    constexpr int VPT = PF;
    constexpr int BLOCK = VPT * 32;
    constexpr int SCALE_STEP = 64 / VPT;
    const device uint8_t* wp = (const device uint8_t*)inject_w
        + int(lane) * BP;
    const device T* sp = inject_s + int(lane) / SCALE_STEP;
    const device T* bp = inject_b + int(lane) / SCALE_STEP;
    const device T* xp = x + int(lane) * VPT;
    float result[4] = {0.0f};
    float xv[VPT];
    int k = 0;
    for (; k < K - BLOCK; k += BLOCK) {
        float sum = hc_load_vector<T, VPT, BITS>(xp, xv);
        for (int row = 0; row < 4; ++row) {
            result[row] += hc_qdot<VPT, BITS>(
                wp + row * ROW_BYTES,
                xv,
                float(sp[row * GROUPS]),
                float(bp[row * GROUPS]),
                sum);
        }
        wp += BLOCK * BP / PF;
        sp += BLOCK / 64;
        bp += BLOCK / 64;
        xp += BLOCK;
    }
    float sum = hc_load_vector<T, VPT, BITS>(xp, xv);
    for (int row = 0; row < 4; ++row) {
        result[row] += hc_qdot_safe<VPT, BITS>(
            wp + row * ROW_BYTES,
            xv,
            float(sp[row * GROUPS]),
            float(bp[row * GROUPS]),
            sum);
        result[row] = simd_sum(result[row]);
        if (lane == 0) combined[320 + row] = T(result[row]);
    }
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen4_hc_hybrid_qmv",
            input_names=[
                "x",
                "down_w",
                "down_s",
                "down_b",
                "inject_w",
                "inject_s",
                "inject_b",
            ],
            output_names=["combined"],
            header=_HEADER,
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
    return _KERNEL


def compatible_projections(down, injection) -> bool:
    """Whether two raw Qwen4 projection banks match the native contract."""
    if _DISABLED:
        return False
    if not (
        type(down) is nn.QuantizedLinear
        and type(injection) is nn.QuantizedLinear
        and getattr(down, "group_size", None)
        == getattr(injection, "group_size", None)
        == _GROUP_SIZE
        and getattr(down, "bits", None) == getattr(injection, "bits", None)
        and getattr(down, "bits", None) in _SUPPORTED_BITS
        and getattr(down, "mode", None)
        == getattr(injection, "mode", None)
        == "affine"
        and "bias" not in down
        and "bias" not in injection
    ):
        return False
    tensors = tuple(
        getattr(projection, name, None)
        for projection in (down, injection)
        for name in ("weight", "scales", "biases")
    )
    if not all(isinstance(value, mx.array) for value in tensors):
        return False
    packed_width = _STREAM_WIDTH * down.bits // 32
    return bool(
        down.weight.shape == (_LOW_RANK, packed_width)
        and injection.weight.shape == (_HC_COUNT, packed_width)
        and down.weight.dtype == injection.weight.dtype == mx.uint32
        and down.scales.shape == (_LOW_RANK, _STREAM_WIDTH // _GROUP_SIZE)
        and injection.scales.shape
        == (_HC_COUNT, _STREAM_WIDTH // _GROUP_SIZE)
        and down.biases.shape == down.scales.shape
        and injection.biases.shape == injection.scales.shape
        and down.scales.dtype == down.biases.dtype == mx.bfloat16
        and injection.scales.dtype == injection.biases.dtype == mx.bfloat16
    )


def hybrid_projection(
    x: mx.array,
    down,
    injection,
) -> mx.array | None:
    """Return exact concatenated ``down, inject`` output or fail closed."""
    global _RUNTIME_FAILED, _FAILURE_LOGGED

    if not (
        not _RUNTIME_FAILED
        and isinstance(x, mx.array)
        and x.shape == (1, 1, _STREAM_WIDTH)
        and x.dtype == mx.bfloat16
        and compatible_projections(down, injection)
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        return None
    try:
        return _kernel()(
            inputs=[
                x,
                down.weight,
                down.scales,
                down.biases,
                injection.weight,
                injection.scales,
                injection.biases,
            ],
            template=[
                ("T", x.dtype),
                ("BITS", down.bits),
                ("K", _STREAM_WIDTH),
            ],
            grid=(32, 82, 1),
            threadgroup=(32, 2, 1),
            output_shapes=[(1, 1, _LOW_RANK + _HC_COUNT)],
            output_dtypes=[x.dtype],
        )[0]
    except Exception as exc:  # noqa: BLE001 - optional native path
        _RUNTIME_FAILED = True
        if not _FAILURE_LOGGED:
            _FAILURE_LOGGED = True
            logger.warning(
                "Qwen4 exact hybrid HC projection failed closed; using "
                "canonical split projections: %s",
                exc,
            )
        return None


__all__ = ["compatible_projections", "hybrid_projection"]
