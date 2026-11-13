# SPDX-License-Identifier: MIT
"""Read GGML blocks in place; no affine requantization or FP8 activation rounding."""

from functools import cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .gguf import FORMATS, row_bytes


@cache
def _kernel(qtype, dequant=False):
    block, size, name = FORMATS[qtype]
    header = Path(__file__).with_suffix(".h").read_text()
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
        # Four rows share each activation load. Prefill currently repeats GEMV
        # per routed row; a tiled gather GEMM can replace it without repacking.
        source = f"""
            uint lane = thread_index_in_simdgroup;
            uint row0 = threadgroup_position_in_grid.x * 32 + simdgroup_index_in_threadgroup * 4;
            uint route = threadgroup_position_in_grid.y;
            uint expert = ids[route];
            float acc[4] = {{0,0,0,0}};
            for (uint k=lane*16; k<K; k+=512) {{
                float xv[16];
                for (uint j=0; j<16; ++j) xv[j] = x[size_t(route)*K+k+j];
                for (uint r=0; r<4; ++r) {{
                    if (row0+r >= N) continue;
                    size_t offset = ((size_t(expert)*N + row0+r)*(K/{block}) + k/{block}) * {size};
                    float4x4 v;
                    {decode}
                    for (uint j=0; j<16; ++j) acc[r] = fma(v[j/4][j%4], xv[j], acc[r]);
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
        header=header,
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
        self.qtype, self.input_dims = qtype, input_dims
        self.output_dims = weight.shape[-2]
        self.quantize_input = False
        if weight.ndim == 3:
            self.num_experts = weight.shape[0]

    def __call__(self, x, indices=None, sorted_indices=False):
        if x.shape[-1] != self.input_dims:
            raise ValueError("GGUF projection input width mismatch")
        if indices is None:
            if self.weight.ndim != 2:
                raise ValueError("Expert GGUF projection requires routing indices")
            shape = (*x.shape[:-1], self.output_dims)
            ids = mx.zeros((x.size // self.input_dims,), mx.uint32)
        else:
            if self.weight.ndim != 3:
                raise ValueError("Dense GGUF projection cannot receive routing indices")
            shape = (*indices.shape, 1, self.output_dims)
            x = mx.broadcast_to(x, (*indices.shape, 1, self.input_dims))
            ids = indices.reshape(-1).astype(mx.uint32)
        x = mx.contiguous(x.reshape(-1, self.input_dims))
        return _kernel(self.qtype)(
            inputs=[x, self.weight, ids],
            template=[("K", self.input_dims), ("N", self.output_dims), ("O", x.dtype)],
            grid=((self.output_dims + 31) // 32 * 256, ids.size, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[shape],
            output_dtypes=[x.dtype],
        )[0]
