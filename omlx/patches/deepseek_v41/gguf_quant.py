# SPDX-License-Identifier: MIT
"""Read GGML blocks in place; no affine requantization or FP8 activation rounding."""

from functools import cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .gguf import FORMATS, row_bytes


@cache
def header():
    return Path(__file__).with_suffix(".h").read_text()


@cache
def _kernel(qtype, dequant=False):
    block, size, name = FORMATS[qtype]
    decode = f"dequantize_{name}((device const block_{name} *)(w + offset), (k % {block}) / 16, v);"
    if dequant:
        source = f"""
            uint i = thread_position_in_grid.x;
            if (i >= COUNT / 16) return;
            size_t k = size_t(i) * 16;
            size_t offset = (k / {block}) * {size};
            float4x4 v;
            {decode}
            for (uint j=0; j<16; ++j) out[k+j] = O(v[j/4][j%4]);
        """
        inputs = ["w"]
    else:
        source = f"""
            uint lane = thread_index_in_simdgroup;
            uint row0 = threadgroup_position_in_grid.x * 32 + simdgroup_index_in_threadgroup * 4;
            uint route = threadgroup_position_in_grid.y;
            uint expert = ids[route];
            float acc[4] = {{0,0,0,0}};
            for (uint k=lane*8; k<K; k+=256) {{
                float xv[8];
                for (uint j=0; j<8; ++j) xv[j] = x[size_t(route)*K+k+j];
                for (uint r=0; r<4; ++r) {{
                    if (row0+r >= N) continue;
                    size_t offset = ((size_t(expert)*N + row0+r)*(K/{block}) + k/{block}) * {size};
                    float v[8];
                    gguf_decode8<{qtype}>(w + offset, k % {block}, v);
                    for (uint j=0; j<8; ++j) acc[r] += v[j] * xv[j];
                }}
            }}
            for (uint r=0; r<4; ++r) acc[r] = simd_sum(acc[r]);
            if (lane<4 && row0+lane<N) out[size_t(route)*N+row0+lane] = O(acc[lane]);
        """
        inputs = ["x", "w", "ids"]
    return mx.fast.metal_kernel(
        name=f"v41_gguf_{name}_{'dequant' if dequant else 'matvec'}",
        input_names=inputs,
        output_names=["out"],
        header=header(),
        source=source,
    )


def dequantize(weight, qtype, shape, dtype=mx.bfloat16):
    """Materialize only tensors whose consumers require an ordinary dense weight."""
    count = 1
    for dim in shape:
        count *= dim
    return _kernel(qtype, True)(
        inputs=[weight],
        template=[("COUNT", count), ("O", dtype)],
        grid=((count // 16 + 255) // 256 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[tuple(shape)],
        output_dtypes=[dtype],
    )[0]


@cache
def _matrix_kernel(qtype, gathered):
    block, size, name = FORMATS[qtype]
    select = (
        """
        const uint bi = threadgroup_position_in_grid.y;
        if (bi >= uint(count[0])) return;
        const uint start = meta[bi*3], expert = meta[bi*3+1], valid = meta[bi*3+2];
        """
        if gathered
        else """
        const uint start = threadgroup_position_in_grid.y * 8;
        const uint expert = 0, valid = min(8u, uint(M) - start);
        """
    )
    source = select + f"""
        const uint lane = thread_index_in_simdgroup;
        const uint r0 = threadgroup_position_in_grid.x * 32 + simdgroup_index_in_threadgroup * 8;
        const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);
        const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);
        simdgroup_matrix<float, 8, 8> a, b, acc;
        acc.thread_elements()[0] = 0.0f;
        acc.thread_elements()[1] = 0.0f;
        for (uint k = 0; k < K; k += 32) {{
            float v[8] = {{0,0,0,0,0,0,0,0}};
            const uint kq = k + 4 * fn;
            if (r0 + fm < N) {{
                const size_t offset = ((size_t(expert) * N + r0 + fm) * (K/{block}) + kq/{block}) * {size};
                gguf_decode8<{qtype}>(w + offset, kq % {block}, v);
            }}
            for (uint j = 0; j < 4; ++j) {{
                a.thread_elements()[0] = v[j];
                a.thread_elements()[1] = v[j + 4];
                b.thread_elements()[0] = fn < valid ? float(x[size_t(start + fn) * K + k + 4 * fm + j]) : 0.0f;
                b.thread_elements()[1] = fn + 1 < valid ? float(x[size_t(start + fn + 1) * K + k + 4 * fm + j]) : 0.0f;
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }}
        }}
        if (r0 + fm < N) {{
            if (fn < valid) out[size_t(start + fn) * N + r0 + fm] = O(acc.thread_elements()[0]);
            if (fn + 1 < valid) out[size_t(start + fn + 1) * N + r0 + fm] = O(acc.thread_elements()[1]);
        }}
    """
    return mx.fast.metal_kernel(
        name=f"v41_gguf_{name}_matrix_{int(gathered)}",
        input_names=["x", "w", "meta", "count"],
        output_names=["out"],
        source=source,
        header="#include <metal_simdgroup_matrix>\n" + header(),
    )


@cache
def _rows_kernel(qtype):
    block, size, name = FORMATS[qtype]
    return mx.fast.metal_kernel(
        name=f"v41_gguf_{name}_rows",
        input_names=["x", "w"],
        output_names=["out"],
        header=header(),
        source=f"""
            const uint lane = thread_index_in_simdgroup;
            const uint row0 = (threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup) * ROWS;
            if (row0 >= N) return;
            float acc[ROWS * NB];
            gguf_rows_nb<K, ROWS, NB, {qtype}, {block}, {size}>(w, row0, x, K, lane, acc);
            if (lane < ROWS)
                for (uint b = 0; b < NB; ++b) out[size_t(b) * N + row0 + lane] = O(acc[lane * NB + b]);
        """,
    )


class GGUFProjection(nn.Module):
    """Packed dense or expert projection with the same gather contract as SwitchLinear."""

    def __init__(self, weight, *, qtype, input_dims):
        super().__init__()
        if weight.dtype != mx.uint8 or weight.ndim not in (2, 3):
            raise ValueError(
                "GGUF projections require rank-two/three packed U8 weights"
            )
        if weight.shape[-1] != row_bytes(input_dims, qtype):
            raise ValueError("Invalid GGUF packed row length")
        self.weight = weight
        self.mode = "gguf"
        self.qtype, self.input_dims = qtype, input_dims
        self.output_dims = weight.shape[-2]
        self.quantize_input = False
        if weight.ndim == 3:
            self.num_experts = weight.shape[0]

    def __call__(self, x, indices=None, sorted_indices=False, block_plan=None):
        if x.shape[-1] != self.input_dims:
            raise ValueError("GGUF projection input width mismatch")
        if indices is None:
            if self.weight.ndim != 2:
                raise ValueError("Expert GGUF projection requires routing indices")
            shape = (*x.shape[:-1], self.output_dims)
            rows = x.size // self.input_dims
            if 2 <= rows <= 16 and self.output_dims >= 8192 and self.input_dims <= 2048:
                per_lane = 4 if self.output_dims % 4 == 0 else 1
                return _rows_kernel(self.qtype)(
                    inputs=[
                        mx.contiguous(x.reshape(rows, self.input_dims)),
                        self.weight,
                    ],
                    template=[
                        ("K", self.input_dims),
                        ("N", self.output_dims),
                        ("NB", rows),
                        ("ROWS", per_lane),
                        ("O", x.dtype),
                    ],
                    grid=(
                        (self.output_dims + 8 * per_lane - 1) // (8 * per_lane) * 256,
                        1,
                        1,
                    ),
                    threadgroup=(256, 1, 1),
                    output_shapes=[shape],
                    output_dtypes=[x.dtype],
                )[0]
            if x.size // self.input_dims >= 64:
                # One temporary projection, not a resident dequantized model.
                # FP32 preserves GGUF scales/codes without another quantization.
                dense = dequantize(
                    self.weight,
                    self.qtype,
                    (self.output_dims, self.input_dims),
                    mx.float32,
                )
                return (x.astype(mx.float32) @ dense.T).astype(x.dtype)
            ids = mx.zeros((x.size // self.input_dims,), mx.uint32)
        else:
            if self.weight.ndim != 3:
                raise ValueError("Dense GGUF projection cannot receive routing indices")
            shape = (*indices.shape, 1, self.output_dims)
            x = mx.broadcast_to(x, (*indices.shape, 1, self.input_dims))
            ids = indices.reshape(-1).astype(mx.uint32)
        x = mx.contiguous(x.reshape(-1, self.input_dims))
        # Eight-column expert tiles waste work below four routes per expert.
        # This average assumes ordinary routing; a per-block choice can handle skew.
        if (indices is None and ids.size >= 32) or (
            indices is not None
            and sorted_indices
            and ids.size >= max(64, 4 * self.num_experts)
        ):
            if indices is not None and block_plan is None:
                from ..deepseek_v4.switch_layers import _build_mxfp4_blocks

                block_plan = _build_mxfp4_blocks(ids, self.num_experts, 8)
            meta, count = block_plan if block_plan is not None else (ids, ids)
            return _matrix_kernel(self.qtype, indices is not None)(
                inputs=[x, self.weight, meta, count],
                template=[
                    ("K", self.input_dims),
                    ("N", self.output_dims),
                    ("M", ids.size),
                    ("O", x.dtype),
                ],
                grid=(
                    (self.output_dims + 31) // 32 * 128,
                    meta.shape[0] if indices is not None else (ids.size + 7) // 8,
                    1,
                ),
                threadgroup=(128, 1, 1),
                output_shapes=[shape],
                output_dtypes=[x.dtype],
            )[0]
        return _kernel(self.qtype)(
            inputs=[x, self.weight, ids],
            template=[("K", self.input_dims), ("N", self.output_dims), ("O", x.dtype)],
            grid=((self.output_dims + 31) // 32 * 256, ids.size, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[shape],
            output_dtypes=[x.dtype],
        )[0]
