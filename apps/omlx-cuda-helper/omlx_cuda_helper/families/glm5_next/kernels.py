# SPDX-License-Identifier: Apache-2.0
"""CUDA kernels for the hot spots of GLM-5.3 prefill.

Every entry point returns None when its kernel does not cover the call or
failed its self-check; the caller then runs the MLX op chain, the reference.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from . import sources

logger = logging.getLogger("omlx_cuda_helper.glm5_next")

_FAILED: set[str] = set()
_KERNELS: dict[str, Any] = {}

# Threads of an expert tile: 8 warps, each 32 rows by 32 columns of the tile.
_TILE_THREADS = 256
# Two staged 64 x 128 row tiles and one 128 x 128 weight tile, in bf16.
_TILE_SHARED = 2 * 64 * 128 * 2 + 128 * 128 * 2
_TILE_STAGES, _TILE_PREFETCH = 2, 4
# Upper bound of expert tiles launched at once; more only idle.
_TILE_BUDGET = 16384


# Bytes of index scores held at once; longer chunks select in row batches.
_SCORE_BYTES = 256 << 20
# Tiles of 64 windows one score block walks.
_SCORE_LOOP = 8


def active() -> bool:
    return mx.cuda.is_available()


def _on(name: str) -> bool:
    return active() and name not in _FAILED


def _kernel(name: str, inputs: list[str], outputs: list[str], source: str, **kw):
    kernel = _KERNELS.get(name)
    if kernel is None:
        kernel = mx.fast.cuda_kernel(
            name=name, input_names=inputs, output_names=outputs, source=source, **kw
        )
        _KERNELS[name] = kernel
    return kernel


def _differs(got: mx.array, want: mx.array) -> float:
    """Relative L2 distance of ``got`` from ``want``."""
    got, want = got.astype(mx.float32), want.astype(mx.float32)
    return (mx.linalg.norm(got - want) / mx.maximum(mx.linalg.norm(want), 1e-30)).item()


def reject(name: str, why: str) -> None:
    _FAILED.add(name)
    logger.warning("GLM-5.3 %s kernel %s; the MLX op chain runs instead", name, why)


def verify(name: str, run, want: mx.array, bound: float) -> None:
    """Turn a kernel off unless ``run()`` reproduces ``want`` on a sample."""
    try:
        got = run()
        if got is None:
            return
        error = _differs(got, want)
    except RuntimeError as failure:
        reject(name, f"does not run ({failure})")
        return
    if not error <= bound:
        reject(name, f"differs from its reference by {error:.2e} (over {bound:.0e})")


def _tile_kernel(name: str, inputs: list[str], outputs: list[str], body: str):
    return _kernel(
        name,
        inputs,
        outputs,
        sources.borrowed("mlx-serve", body),
        header=sources.borrowed("mlx-serve", "moe_prefill_header.cu"),
        shared_memory=_TILE_SHARED,
    )


def reads(weight: mx.array, scales, biases, quant) -> bool:
    """Whether the tile kernels multiply by this packed bank of expert weights:
    [count, out, in] affine codes of 4 or 8 bits, bf16 scales, groups of 64."""
    quant = quant or {}
    return (
        scales is not None
        and biases is not None
        and scales.dtype == mx.bfloat16
        and quant.get("mode", "affine") == "affine"
        and quant.get("bits") in (4, 8)
        and quant.get("group_size") == 64
        and scales.shape[-1] % 2 == 0
        and weight.ndim == 3
        and weight.shape[1] % 128 == 0
    )


def _bank(x, weight, scales, biases, index, bits: int, count: int, splits: int):
    """Rows ``x`` [R, in] times the packed weight ``index`` names, [R, out]."""
    out = weight.shape[-2]
    return _tile_kernel(
        "glm5_bank", ["x", "w", "s", "b", "inds"], ["y"], "moe_prefill_down.cu"
    )(
        inputs=[x, weight, scales, biases, index],
        template=[
            ("T", mx.bfloat16),
            ("STG", _TILE_STAGES),
            ("BDEP", _TILE_PREFETCH),
            ("BITS", bits),
            ("K", x.shape[1]),
            ("N", out),
        ],
        grid=(_TILE_THREADS * (out // 128), count, splits),
        threadgroup=(_TILE_THREADS, 1, 1),
        output_shapes=[(x.shape[0], out)],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ]


def _moe_grid(tiles: int, count: int, routes: int) -> tuple[int, int, int]:
    # Each launch covers every 64-row tile of an expert in `splits` blocks.
    splits = min(max(_TILE_BUDGET // (tiles * count), 1), 8, -(-routes // 64))
    return (_TILE_THREADS * tiles, count, splits)


def moe_order(experts: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """(route order by expert, experts in that order, inverse order)."""
    flat = experts.reshape(-1).astype(mx.uint32)
    order = mx.argsort(flat)
    return order, flat[order], mx.argsort(order).astype(mx.uint32)


def moe_activation(moe, x: mx.array, experts: mx.array):
    """Clamped SwiGLU of each route's gate and up projection, or None.

    Returns [routes, inner] in expert order (see ``moe_order``). The expert
    weights are rounded once from float, like the tiles of a Metal quantized
    matmul.
    """
    if not _on("moe") or x.dtype != mx.bfloat16 or x.shape[0] != 1:
        return None
    banks = (moe.gate, moe.up, moe.down)
    if not all(reads(b.weight, b.scales, b.biases, b.quant) for b in banks):
        return None
    count, inner, _ = moe.gate.weight.shape
    out = moe.down.weight.shape[1]
    tokens, hidden = x.shape[1], x.shape[2]
    k = experts.shape[-1]
    bits = moe.gate.quant["bits"]
    if (
        moe.up.quant["bits"] != bits
        or hidden % 128
        or inner % 128
        or out % 128
        or out // 8 > 1024
    ):
        return None
    order, inds, _ = moe_order(experts)
    limit = mx.array([moe.limit or 0.0], dtype=mx.float32)
    return _tile_kernel(
        "glm5_moe_gate_up",
        ["x", "rows", "inds", "gw", "gs", "gb", "uw", "us", "ub", "limit"],
        ["act"],
        "moe_prefill_gate_up.cu",
    )(
        inputs=[
            x.reshape(tokens, hidden),
            (order // k).astype(mx.uint32),
            inds,
            moe.gate.weight,
            moe.gate.scales,
            moe.gate.biases,
            moe.up.weight,
            moe.up.scales,
            moe.up.biases,
            limit,
        ],
        template=[
            ("T", mx.bfloat16),
            ("STG", _TILE_STAGES),
            ("BDEP", _TILE_PREFETCH),
            ("BITS", bits),
            ("K", hidden),
            ("I", inner),
            ("CLAMP", moe.limit is not None),
        ],
        grid=_moe_grid(inner // 64, count, tokens * k),
        threadgroup=(_TILE_THREADS, 1, 1),
        output_shapes=[(tokens * k, inner)],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ]


def moe_down(moe, act: mx.array, inds: mx.array) -> mx.array:
    """Routes ``act`` [routes, inner] through their expert's down projection."""
    count, out = moe.down.weight.shape[:2]
    return _bank(
        act,
        moe.down.weight,
        moe.down.scales,
        moe.down.biases,
        inds,
        moe.down.quant["bits"],
        count,
        _moe_grid(out // 128, count, act.shape[0])[2],
    )


def moe_sum(down: mx.array, inverse: mx.array, weights: mx.array) -> mx.array:
    """Each token's routes ``down`` [routes, out] weighted and summed."""
    tokens, k = weights.shape[-2:]
    out = down.shape[1]
    return _kernel(
        "glm5_moe_sum", ["down", "inverse", "weights"], ["y"], sources.MOE_SUM
    )(
        inputs=[down, inverse, weights.reshape(-1)],
        template=[("TOPK", k), ("N", out)],
        grid=(tokens * (out // 8), 1, 1),
        threadgroup=(out // 8, 1, 1),
        output_shapes=[(1, tokens, out)],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ]


def moe_routed(moe, x: mx.array, experts: mx.array, weights: mx.array):
    """Weighted sum of each token's routed experts, [1, L, out], or None."""
    act = moe_activation(moe, x, experts)
    if act is None:
        return None
    _, inds, inverse = moe_order(experts)
    return moe_sum(moe_down(moe, act, inds), inverse, weights)


def hc_pre(connection, x: mx.array):
    """(collapsed, post, comb) of streams ``x`` [1, L, streams, D], or None."""
    streams = connection.streams
    if (
        not _on("hc read")
        or x.dtype != mx.bfloat16
        or x.ndim != 4
        or x.shape[0] != 1
        or x.shape[2] != streams
        or streams > 4
        or x.shape[3] % 8
    ):
        return None
    rows, width = x.shape[1], x.shape[3]
    eps = mx.array([connection.norm_eps, connection.eps], dtype=mx.float32)
    # The products with the mix weight are one float32 GEMM over all rows.
    dots = x.reshape(rows, streams * width).astype(mx.float32) @ connection.fn_t
    collapsed, post, comb = _kernel(
        "glm5_hc_pre",
        ["x", "dots", "base", "scale", "eps"],
        ["collapsed", "post", "comb"],
        sources.HC_PRE,
    )(
        inputs=[x[0], dots, connection.base, connection.scale, eps],
        template=[("HC", streams), ("D", width), ("ITERS", max(connection.iters, 1))],
        grid=(256 * rows, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width), (rows, streams), (rows, streams, streams)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
    )
    return collapsed[None], post[None], comb[None]


def hc_expand(branch: mx.array, residual: mx.array, post: mx.array, comb: mx.array):
    """The streams after a branch [1, L, D] is written back, or None."""
    if (
        not _on("hc write")
        or branch.dtype != mx.bfloat16
        or residual.dtype != mx.bfloat16
        or post.dtype != mx.float32
        or comb.dtype != mx.float32
        or branch.shape[0] != 1
        or residual.shape[2] > 4
        or branch.shape[2] % 8
    ):
        return None
    rows, width = branch.shape[1], branch.shape[2]
    streams = residual.shape[2]
    return _kernel(
        "glm5_hc_expand",
        ["branch", "residual", "post", "comb"],
        ["out"],
        sources.HC_EXPAND,
    )(
        inputs=[branch[0], residual[0], post[0], comb[0]],
        template=[("HC", streams), ("D", width)],
        grid=(rows * (width // 8), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, streams, width)],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ][
        None
    ]


def _kda_fits(kda, *arrays: mx.array) -> bool:
    return (
        kda.dim == 128
        and kda.lower_bound is not None
        and all(a.dtype == mx.bfloat16 and a.shape[0] == 1 for a in arrays)
    )


def kda_prework(kda, proj: mx.array, conv: mx.array):
    """(q, k, v, next conv state) of a chunk's fused projection, or None."""
    if (
        not _on("kda prework")
        or not _kda_fits(kda, proj, conv)
        or kda.conv.dtype != mx.bfloat16
    ):
        return None
    rows, heads, taps = proj.shape[1], kda.heads, kda.kernel
    qkv = 3 * heads * 128
    shape = (rows, heads, 128)
    q, k, v = _kernel(
        "glm5_kda_prework",
        ["proj", "state", "w", "qscale"],
        ["q", "k", "v"],
        sources.KDA_PREWORK,
        header=sources.FUNCTORS,
    )(
        inputs=[
            proj[0],
            conv[0],
            kda.conv.reshape(qkv, taps),
            mx.array([128**-0.5], dtype=mx.float32),
        ],
        template=[("H", heads), ("TAPS", taps)],
        grid=(rows * 3 * heads * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape, shape, shape],
        output_dtypes=[mx.bfloat16] * 3,
    )
    window = mx.concatenate([conv, proj[..., :qkv]], axis=1)
    return q[None], k[None], v[None], mx.contiguous(window[:, -(taps - 1) :, :])


def kda_decay(kda, a: mx.array):
    """Forget factors [1, L, heads, 128] in float32, or None."""
    if not _on("kda decay") or not _kda_fits(kda, a):
        return None
    return _kernel(
        "glm5_kda_decay", ["a", "rate", "bias", "bound"], ["g"], sources.KDA_DECAY
    )(
        inputs=[
            a[0],
            mx.exp(kda.a_log),
            kda.dt_bias,
            mx.array([kda.lower_bound], dtype=mx.float32),
        ],
        template=[("H", kda.heads)],
        grid=(a.size // 4, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[a.shape[1:]],
        output_dtypes=[mx.float32],
    )[
        0
    ][
        None
    ]


def kda_norm_gate(kda, out: mx.array, gate: mx.array):
    """The gated, normalized KDA output [1, L, heads, 128], or None."""
    if (
        not _on("kda norm")
        or not _kda_fits(kda, out, gate)
        or kda.o_norm.dtype != mx.bfloat16
    ):
        return None
    return _kernel(
        "glm5_kda_norm_gate", ["y", "gate", "w", "eps"], ["out"], sources.KDA_NORM_GATE
    )(
        inputs=[
            out[0],
            gate[0],
            kda.o_norm,
            mx.array([kda.norm_eps], dtype=mx.float32),
        ],
        template=[("H", kda.heads)],
        grid=(out.shape[1] * kda.heads * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[out.shape[1:]],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ][
        None
    ]


def window_scores(q, pool, weights, position: int, kpool: int) -> mx.array:
    """Index scores [rows, windows] in bf16, masked windows at -1e30."""
    rows, windows = q.shape[0], pool.shape[0]
    tiles = -(-windows // 64)
    return _kernel(
        "glm5_index_scores",
        ["qi", "wts", "pk", "pos"],
        ["out"],
        sources.borrowed("TensorFold", "index_scores.cu"),
        header=sources.borrowed("TensorFold", "index_header.cu"),
    )(
        inputs=[q, weights, pool, mx.array([position], dtype=mx.int32)],
        template=[("KP", kpool), ("LOOP", _SCORE_LOOP)],
        grid=(256 * -(-rows // 8), -(-tiles // _SCORE_LOOP), 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, windows)],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ]


def top_windows(scores: mx.array, count: int) -> mx.array:
    """The ``count`` best windows of each row of bf16 scores, [rows, count]."""
    return _kernel("glm5_top_windows", ["scores"], ["out"], sources.TOP_WINDOWS)(
        inputs=[scores],
        template=[("COUNT", count)],
        grid=(256 * scores.shape[0], 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(scores.shape[0], count)],
        output_dtypes=[mx.int32],
    )[0]


def select_windows(q, pool, weights, position: int, kpool: int, count: int):
    """Windows [rows, count] the indexer keeps for each query, or None.

    ``q`` [rows, heads, 128], ``weights`` [rows, heads] and ``pool``
    [windows, 128] are bf16; ``position`` is the position of the first query.
    """
    if (
        not _on("index")
        or q.dtype != mx.bfloat16
        or pool.dtype != mx.bfloat16
        or weights.dtype != mx.bfloat16
        or q.shape[2] != 128
        or q.shape[1] > 32
        or count > pool.shape[0]
    ):
        return None
    rows = q.shape[0]
    step = max(8, _SCORE_BYTES // (2 * pool.shape[0]) // 8 * 8)
    picked = []
    for a in range(0, rows, step):
        scores = window_scores(
            q[a : a + step], pool, weights[a : a + step], position + a, kpool
        )
        picked.append(top_windows(scores, count))
    return picked[0] if len(picked) == 1 else mx.concatenate(picked, axis=0)


def sparse_attention(q_latent, kv, tokens, scale: float):
    """Latent attention [1, 64, rows, 512] over each row's keys, or None.

    ``q_latent`` [1, 64, rows, 512] and ``kv`` [1, 1, positions, 512] are
    bf16; ``tokens`` [rows, width] names the keys of each row, -1 for none.
    """
    if (
        not _on("attention")
        or q_latent.dtype != mx.bfloat16
        or kv.dtype != mx.bfloat16
        or q_latent.shape[:2] != (1, 64)
        or q_latent.shape[3] != 512
        or kv.shape[3] != 512
    ):
        return None
    rows, width = tokens.shape
    tiles = -(-width // 32)
    return _kernel(
        f"glm5_sparse_attention_{width}",
        ["q", "kv", "tok", "scale"],
        ["out"],
        sources.SPARSE_ATTENTION,
        header=sources.FRAGMENTS,
        shared_memory=32 * 512 * 2 + 256 * 16 * 4 + tiles * 32 * 4,
    )(
        inputs=[q_latent[0], kv[0, 0], tokens, mx.array([scale], dtype=mx.float32)],
        template=[("W", width)],
        grid=(256 * rows, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(64, rows, 512)],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ][
        None
    ]
