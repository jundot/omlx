# SPDX-License-Identifier: Apache-2.0
"""Register-neutral Q4 gate/up qmm that emits SwiGLU directly.

The dense gate/up weights are physically concatenated by qwen35_dense_gate_up.
Pairing two gate and two up columns keeps the same four-column accumulator and
threadgroup geometry as the normal split-K kernel, while avoiding the full
2N intermediate and the following SwiGLU dispatch.
"""

from __future__ import annotations

import mlx.core as mx

from .qwen35_verify_qmm import _is_armed, _is_exact_armed

_KERNELS: dict = {}


def _pack_block(m: int) -> str:
    lines = ["int kb = pack * 8;", "int gi = kb / GS;"]
    for r in range(m):
        lines.append(f"Vec8 v{r} = xv[({r} * K + kb) / 8];")
    for j in range(2):
        lines.extend([
            f"uint32_t pg{j} = w_q[(n0 + {j}) * K_by_p + pack];",
            f"uint32_t pu{j} = w_q[(NH + n0 + {j}) * K_by_p + pack];",
            f"float sg{j} = float(scales[(n0 + {j}) * K_by_gs + gi]);",
            f"float bg{j} = float(biases[(n0 + {j}) * K_by_gs + gi]);",
            f"float su{j} = float(scales[(NH + n0 + {j}) * K_by_gs + gi]);",
            f"float bu{j} = float(biases[(NH + n0 + {j}) * K_by_gs + gi]);",
        ])
    for j in range(2):
        lines.extend([
            "{",
            f"    uint32_t pg = pg{j}; uint32_t pu = pu{j};",
            f"    float sg = sg{j}; float bg = bg{j};",
            f"    float su = su{j}; float bu = bu{j};",
            "    for (int ki = 0; ki < 8; ++ki) {",
            "        float wg = float((pg >> (ki * 4)) & 0xFu) * sg + bg;",
            "        float wu = float((pu >> (ki * 4)) & 0xFu) * su + bu;",
        ])
        for r in range(m):
            lines.append(f"        acc[{j * m + r}] += float(v{r}[ki]) * wg;")
            lines.append(f"        acc[{(2 + j) * m + r}] += float(v{r}[ki]) * wu;")
        lines.extend(["    }", "}"])
    return "\n            ".join(lines)


def _kernel(m: int, group_size: int, dtype):
    key = (m, group_size, dtype)
    if key in _KERNELS:
        return _KERNELS[key]
    n_acc = 4 * m
    source = f"""
        using namespace metal;
        constexpr int GS = {group_size};
        constexpr int K_PARTS = 2;
        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        int K = int(K_size);
        int NH = int(N_half);
        int K_by_p = K / 8;
        int K_by_gs = K / GS;
        int per_part = K_by_p / K_PARTS;
        int n0 = int(threadgroup_position_in_grid.y) * 2;
        int p_start = int(part) * per_part;
        int p_end = (int(part) == 1) ? K_by_p : p_start + per_part;
        float acc[{n_acc}];
        for (int i = 0; i < {n_acc}; ++i) {{ acc[i] = 0.0f; }}
        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;
        for (int pack = p_start + int(lane); pack < p_end; pack += 32) {{
            {_pack_block(m)}
        }}
        for (int i = 0; i < {n_acc}; ++i) {{ acc[i] = simd_sum(acc[i]); }}
        threadgroup float partials[K_PARTS * {n_acc}];
        if (lane == 0) {{
            for (int i = 0; i < {n_acc}; ++i) {{
                partials[int(part) * {n_acc} + i] = acc[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part == 0 && lane < {2 * m}) {{
            float gate_total = partials[int(lane)] + partials[{n_acc} + int(lane)];
            float up_total = partials[{2 * m} + int(lane)]
                + partials[{n_acc + 2 * m} + int(lane)];
            int j = int(lane) / {m};
            int row = int(lane) - j * {m};
            T gate = T(gate_total);
            T up = T(up_total);
            T sy = T(1) / (T(1) + metal::exp(metal::abs(gate)));
            T act = gate * ((gate < T(0)) ? sy : T(1) - sy);
            y[row * NH + n0 + j] = act * up;
        }}
    """
    tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    value = mx.fast.metal_kernel(
        name=f"omlx_vk_swiglu2_m{m}_gs{group_size}_{tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_half"],
        output_names=["y"],
        source=source,
    )
    _KERNELS[key] = value
    return value


def try_fast_swiglu(linear, x, target_verify: bool):
    if not target_verify or not _is_armed() or _is_exact_armed():
        return None
    if x.ndim != 3 or x.shape[0] != 1 or x.dtype not in (mx.bfloat16, mx.float16):
        return None
    meta = getattr(linear, "_omlx_fast_swiglu_meta", None)
    if meta is None:
        bits = int(getattr(linear, "bits", 0))
        group_size = int(getattr(linear, "group_size", 0))
        k_dim = int(linear.weight.shape[1]) * 32 // max(bits, 1)
        n_dim = int(linear.weight.shape[0])
        supported = (
            bits == 4
            and getattr(linear, "mode", "affine") == "affine"
            and group_size in (32, 64, 128)
            and k_dim % 64 == 0
            and n_dim % 8 == 0
            and "bias" not in linear
        )
        meta = (supported, k_dim, n_dim // 2, group_size)
        linear._omlx_fast_swiglu_meta = meta
    supported, k_dim, n_half, group_size = meta
    rows = int(x.shape[1])
    if not supported or not (3 <= rows <= 6) or int(x.shape[2]) != k_dim:
        return None
    (out,) = _kernel(rows, group_size, x.dtype)(
        inputs=[
            x.reshape(rows, k_dim), linear.weight, linear.scales, linear.biases,
            k_dim, n_half,
        ],
        template=[("T", x.dtype)],
        grid=(64, n_half // 2, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(rows, n_half)],
        output_dtypes=[x.dtype],
    )
    return out.reshape(1, rows, n_half)


__all__ = ["try_fast_swiglu"]
