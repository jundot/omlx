# SPDX-License-Identifier: Apache-2.0
"""Self-checks of the GLM-5.3 CUDA kernels against their MLX op chains."""

from __future__ import annotations

import mlx.core as mx

from . import kernels, model

# Largest relative error of a kernel whose float32 sums only differ in order
# from the reference: a small share of 16-bit outputs moves by one unit.
_ROUNDING = 5e-4
# The expert chain multiplies that through two projections.
_EXPERTS = 3e-3


def _joined(parts) -> mx.array | None:
    """Several outputs as one float vector."""
    if parts is None:
        return None
    return mx.concatenate([part.astype(mx.float32).flatten() for part in parts])


def _experts(moe: model.Moe) -> None:
    """The expert kernels against the MLX gather over the same routes."""
    count = moe.gate.weight.shape[0]
    x = mx.random.normal((1, 96, moe.router_t.shape[0])).astype(mx.bfloat16)
    experts = mx.random.randint(0, count, (1, 96, moe.top_k))
    weights = mx.random.uniform(shape=experts.shape)
    banks = (moe.gate, moe.up, moe.down)
    want = model.routed(banks, x, experts, weights, moe.limit)

    def run():
        return kernels.moe_routed(moe, x, experts, weights)

    kernels.verify("moe", run, want, _EXPERTS)


def _hyper_connection(connection: model.HyperConnection) -> None:
    width = connection.fn_t.shape[0] // connection.streams
    x = mx.random.normal((1, 96, connection.streams, width)).astype(mx.bfloat16)
    want = connection.reference(x)
    branch = mx.random.normal((1, 96, width)).astype(mx.bfloat16)
    _, post, comb = want
    kernels.verify(
        "hc read",
        lambda: _joined(kernels.hc_pre(connection, x)),
        _joined(want),
        _ROUNDING,
    )
    kernels.verify(
        "hc write",
        lambda: kernels.hc_expand(branch, x, post, comb),
        model.expand_reference(branch, x, post, comb),
        _ROUNDING,
    )


def _kda(kda: model.Kda) -> None:
    heads, dim = kda.heads, kda.dim
    qkv = 3 * heads * dim
    proj = mx.random.normal((1, 96, qkv + 2 * kda.gate_dim + heads))
    proj = proj.astype(mx.bfloat16)
    conv = mx.random.normal((1, kda.kernel - 1, qkv)).astype(mx.bfloat16)
    want = kda.prework(proj, conv)
    a = mx.random.normal((1, 96, heads, dim)).astype(mx.bfloat16)
    kernels.verify(
        "kda prework",
        lambda: _joined(kernels.kda_prework(kda, proj, conv)),
        _joined(want),
        _ROUNDING,
    )
    kernels.verify(
        "kda decay",
        lambda: kernels.kda_decay(kda, a),
        kda.decay_reference(a),
        _ROUNDING,
    )
    kernels.verify(
        "kda norm",
        lambda: kernels.kda_norm_gate(kda, want[0], a),
        kda.norm_gate(want[0], a),
        _ROUNDING,
    )


def _kept(got: mx.array, want: mx.array) -> float:
    """Mean share of each row's wanted members that were also chosen."""
    same = (got[:, :, None] == want[:, None, :]).any(axis=1)
    return same.astype(mx.float32).mean().item()


def _sparse(attention: model.SparseAttention) -> None:
    config = attention.config
    kp, heads, dim = config.index_kpool, config.index_heads, config.index_dim
    rows, windows, first = 64, 1024, 3200
    count = min(config.index_topk // kp, windows)
    q = mx.random.normal((rows, heads, dim)).astype(mx.bfloat16)
    pool = mx.random.normal((windows, dim)).astype(mx.bfloat16)
    weights = mx.random.normal((rows, heads)).astype(mx.bfloat16)
    try:
        picked = kernels.select_windows(q, pool, weights, first, kp, count)
        if picked is not None:
            scores = attention.window_scores(q, pool, weights, first)
            share = _kept(picked, attention.top(scores, count))
            if not share >= 0.995:
                kernels.reject(
                    "index", f"keeps {100 * share:.2f}% of the reference windows"
                )
    except RuntimeError as failure:
        kernels.reject("index", f"does not run ({failure})")
    width = count * kp + kp - 1
    q_latent = mx.random.normal((1, config.heads, 24, config.kv_rank))
    q_latent = q_latent.astype(mx.bfloat16)
    kv = mx.random.normal((1, 1, 4096, config.kv_rank)).astype(mx.bfloat16)
    tokens = mx.random.randint(-64, 4096, (24, width)).astype(mx.int32)
    kernels.verify(
        "attention",
        lambda: kernels.sparse_attention(q_latent, kv, tokens, attention.scale),
        attention.attend_reference(q_latent, kv, tokens),
        2 * _ROUNDING,
    )


def layer(layer: model.Layer) -> None:
    """Check every kernel this layer can use."""
    _hyper_connection(layer.attn_hc)
    (_kda if layer.linear else _sparse)(layer.attn)
    if isinstance(layer.mlp, model.Moe):
        _experts(layer.mlp)
