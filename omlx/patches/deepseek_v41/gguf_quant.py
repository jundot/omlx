# SPDX-License-Identifier: MIT
"""Read GGML blocks in place; no affine requantization or FP8 activation rounding."""

from functools import cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from omlx.custom_kernels.nax import is_nax_available
from omlx.custom_kernels.nax_tiles import NAX_TILE_HEADER

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
        const uint expert = 0, valid = min(8u, uint(x_shape[0]) - start);
        """
    )
    source = (
        select
        + f"""
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
    )
    return mx.fast.metal_kernel(
        name=f"v41_gguf_{name}_matrix_{int(gathered)}",
        input_names=["x", "w", "meta", "count"],
        output_names=["out"],
        source=source,
        header="#include <metal_simdgroup_matrix>\n" + header(),
    )


@cache
def _nax_kernel(qt, rows, gathered=True):
    bm, bn, bk = (32, 64, 64) if rows == 32 else (64, 32, 128)
    block, size, name = FORMATS[qt]
    two = qt in (10, 13, 14)
    bk = min(block, bk)
    select = (
        """
            const uint bi = threadgroup_position_in_grid.y;
            if (bi >= uint(count[0])) return;
            const uint start = meta[bi*3], expert = meta[bi*3+1], valid = meta[bi*3+2];
        """
        if gathered
        else f"""
            const uint start = threadgroup_position_in_grid.y * {bm};
            const uint expert = 0, valid = min({bm}u, uint(x_shape[0]) - start);
        """
    )
    return mx.fast.metal_kernel(
        name=f"v41_gguf_block_{name}_{bm}_{bn}_{bk}" + ("" if gathered else "_dense"),
        input_names=["x", "w", "meta", "count"],
        output_names=["out"],
        header="#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        + header(),
        source=select
        + f"""
            const uint tid = thread_index_in_threadgroup;
            const uint n0 = threadgroup_position_in_grid.x * {bn};
            threadgroup half ws[{bn * bk}], us[{bn * bk if two else 1}];
            threadgroup float ds[{bn}], ms[{bn if two else 1}];
            // MPP requires an unqualified tensor element type; run only reads x.
            auto tx = tensor(const_cast<device O*>(x) + size_t(start) * K,
                             dextents<int, 2>(K, valid), array<int, 2>{{1, K}});
            auto tw = tensor(ws, dextents<int, 2>({bk}, {bn}));
            constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
                {bm}, {bn}, {bk}, false, true, false,
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
            mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
            auto acc = op.template get_destination_cooperative_tensor<decltype(tx), decltype(tw), float>();
            for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
            for (uint base = 0; base < K; base += {block}) {{
                auto part = acc, other = acc;
                for (uint i = 0; i < part.get_capacity(); ++i) {{ part[i] = 0.0f; other[i] = 0.0f; }}
                for (uint kk = 0; kk < {block}; kk += {bk}) {{
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint z = tid * 8; z < {bn * bk}; z += 128 * 8) {{
                        const uint row = z / {bk}, k = z % {bk};
                        float v[8] = {{0}}, u[8] = {{0}}, f[2] = {{0}};
                        if (n0 + row < N && base + kk + k < K) {{
                            const size_t offset = ((size_t(expert) * N + n0 + row) * (K/{block})
                                                   + (base + kk + k)/{block}) * {size};
                            gguf_block8<{qt}>(w + offset, (kk + k) % {block}, v, u, f);
                        }}
                        for (uint j = 0; j < 8; ++j) {{
                            ws[row * {bk} + k + j] = half(v[j]);
                            if constexpr ({str(two).lower()}) us[row * {bk} + k + j] = half(u[j]);
                        }}
                        if (kk == 0 && k == 0) {{
                            ds[row] = f[0];
                            if constexpr ({str(two).lower()}) ms[row] = f[1];
                        }}
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    auto left = tx.slice(base + kk, 0);
                    op.run(left, tw, part);
                    if constexpr ({str(two).lower()}) {{
                        auto tu = tensor(us, dextents<int, 2>({bk}, {bn}));
                        op.run(left, tu, other);
                    }}
                }}
                for (uint i = 0; i < acc.get_capacity(); ++i) {{
                    auto xy = acc.get_multidimensional_index(i);
                    acc[i] += part[i] * ds[xy[0]];
                    if constexpr ({str(two).lower()}) acc[i] += other[i] * ms[xy[0]];
                }}
            }}
            for (uint i = 0; i < acc.get_capacity(); ++i) {{
                auto xy = acc.get_multidimensional_index(i);
                if (xy[1] < valid && n0 + xy[0] < N)
                    out[size_t(start + xy[1]) * N + n0 + xy[0]] = O(acc[i]);
            }}
        """,
    )


@cache
def _nax_register_kernel(qt, rows, gathered=True):
    """Decode IQ coefficients in SIMD registers; apply base scales in FP32."""
    _, size, name = FORMATS[qt]
    mm, sm, cols = (16, 16, 64) if rows == 32 else (32, 64, 128)
    scale = 0.25 if qt == 18 else 0.125
    # Group row tiles on grid x to reuse activations across output columns.
    swizzled = gathered and rows == 64
    tile = (
        "(threadgroup_position_in_grid.y / ((N + 127) / 128)) * 8 + threadgroup_position_in_grid.x"
        if swizzled
        else "threadgroup_position_in_grid.y"
    )
    column = (
        "threadgroup_position_in_grid.y % ((N + 127) / 128)"
        if swizzled
        else "threadgroup_position_in_grid.x"
    )
    select = (
        f"""
        const uint bi = {tile};
        if (bi >= uint(count[0])) return;
        const uint start = meta[bi*3], expert = meta[bi*3+1], valid = meta[bi*3+2];
        """
        if gathered
        else f"""
        const uint start = threadgroup_position_in_grid.y * {rows};
        const uint expert = 0, valid = min({rows}u, uint(x_shape[0]) - start);
        """
    )
    ctype = "op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>, metal::remove_addrspace_t<decltype(b)>, float>()"
    totals = "\n".join(
        f"auto t{m} = {ctype}; for(uint i=0;i<{mm};++i) t{m}[i]=0.0f;"
        for m in range(sm // mm)
    )
    parts = "\n".join(
        f"auto p{m} = {ctype}; for(uint i=0;i<{mm};++i) p{m}[i]=0.0f;"
        for m in range(sm // mm)
    )
    run = "\n".join(
        f"""
        if ({m * mm} < nr) {{
            _Pragma("clang loop unroll(full)")
            for (uint h=0;h<{mm // 4};++h) {{
                const uint row=min({m * mm}+sc.y+(h&1)*8+(h/4)*16,nr-1);
                const auto v=*reinterpret_cast<const device vec<O,4>*>(x+size_t(start+m0+row)*K+base+kk+sc.x+((h/2)&1)*16);
                _Pragma("clang loop unroll(full)")
                for(uint j=0;j<4;++j) a[h*4+j]=v[j];
            }}
            op.run(a,b,p{m});
        }}"""
        for m in range(sm // mm)
    )
    accumulate = "\n".join(f"t{m}[i] += p{m}[i] * d;" for m in range(sm // mm))
    output = "\n".join(
        f"if ({m * mm}+xy[1]<nr && n0+xy[0]<N) out[size_t(start+m0+{m * mm}+xy[1])*N+n0+xy[0]]=O(t{m}[i]);"
        for m in range(sm // mm)
    )
    return mx.fast.metal_kernel(
        name=f"v41_gguf_register_{name}_{rows}_{int(gathered)}",
        input_names=["x", "w", "meta", "count"],
        output_names=["out"],
        header=NAX_TILE_HEADER + header(),
        source=select
        + f"""
        const uint sg=simdgroup_index_in_threadgroup, lane=thread_index_in_simdgroup;
        const uint m0=(sg/({cols}/32))*{sm}, n0=({column})*{cols}+(sg%({cols}/32))*32;
        if(m0>=valid || n0>=N) return;
        const uint nr=min({sm}u,valid-m0);
        const short2 sc=omlx_nax::BaseNAXFrag::get_coord();
        constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor({mm},32,32,false,true,false,mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc,execution_simdgroup> op;
        auto a=op.template get_left_input_cooperative_tensor<O,half,float>();
        auto b=op.template get_right_input_cooperative_tensor<O,half,float>();
        {totals}
        for(uint base=0;base<K;base+=256) {{
            GGUFQuarter packed[4];
            _Pragma("clang loop unroll(full)")
            for(uint q=0;q<4;++q) {{
                const uint n=n0+sc.y+(q&1)*8+(q>>1)*16;
                packed[q]={{uint4(0),uint2(0)}};
                if(n<N) {{
                    const size_t at=((size_t(expert)*N+n)*(K/256)+base/256)*{size};
                    packed[q]=gguf_quarter<{qt}>(w+at,sc.x/4);
                }}
            }}
            {parts}
            for(uint kk=0;kk<256;kk+=32) {{
                _Pragma("clang loop unroll(full)")
                for(uint q=0;q<8;++q) {{
                    const half4 v=gguf_quarter_coeff4<{qt}>(packed[q&3],kk+sc.x+(q/4)*16,lane);
                    _Pragma("clang loop unroll(full)")
                    for(uint j=0;j<4;++j) b[q*4+j]=v[j];
                }}
                {run}
            }}
            _Pragma("clang loop unroll(full)")
            for(uint i=0;i<{mm};++i) {{
                const auto xy=t0.get_multidimensional_index(i);
                if(n0+xy[0]>=N) continue;
                const size_t at=((size_t(expert)*N+n0+xy[0])*(K/256)+base/256)*{size};
                const float d=float(*reinterpret_cast<const device half*>(w+at))*{scale}f;
                {accumulate}
            }}
        }}
        _Pragma("clang loop unroll(full)")
        for(uint i=0;i<{mm};++i) {{
            const auto xy=t0.get_multidimensional_index(i);
            {output}
        }}
        """,
    )


def prefill_plan(x, indices, num_experts):
    from ..deepseek_v4.switch_layers import _build_mxfp4_blocks

    # Average occupancy: wider tiles lose to the SIMD path on sparse/CED tails.
    # Per-block tile selection is the upgrade path for severely skewed routing.
    rows = (
        (64 if indices.size >= 64 * num_experts else 32)
        if indices.size >= max(64, 8 * num_experts)
        and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and mx.default_device() != mx.cpu
        and is_nax_available()
        else 8
    )
    meta, count = _build_mxfp4_blocks(indices, num_experts, rows)
    return meta, count, rows


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
            if rows >= 64 and not (
                x.dtype in (mx.bfloat16, mx.float16, mx.float32)
                and mx.default_device() != mx.cpu
                and is_nax_available()
            ):
                # Non-NAX fallback: one temporary projection, with FP32
                # preserving GGUF scales/codes without another quantization.
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
                block_plan = prefill_plan(x, ids, self.num_experts)
            meta, count, rows = (
                block_plan
                if block_plan is not None
                else (ids, ids, 64 if ids.size >= 64 else 8)
            )
            nax = rows >= 32
            registers = (
                nax
                and self.qtype in (16, 17, 18, 19, 22)
                and x.dtype in (mx.bfloat16, mx.float16)
            )
            cols = 128 if registers and rows == 64 else 64 if rows == 32 else 32
            if registers:
                kernel = _nax_register_kernel(self.qtype, rows, indices is not None)
            elif nax:
                kernel = _nax_kernel(self.qtype, rows, indices is not None)
            else:
                kernel = _matrix_kernel(self.qtype, indices is not None)
            grid = (
                (
                    8 * 128,
                    (meta.shape[0] + 7) // 8 * ((self.output_dims + 127) // 128),
                    1,
                )
                if registers and indices is not None and rows == 64
                else (
                    (self.output_dims + cols - 1) // cols * 128,
                    meta.shape[0]
                    if indices is not None
                    else (ids.size + rows - 1) // rows,
                    1,
                )
            )
            return kernel(
                inputs=[x, self.weight, meta, count],
                template=[
                    ("K", self.input_dims),
                    ("N", self.output_dims),
                    ("O", x.dtype),
                ],
                grid=grid,
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
