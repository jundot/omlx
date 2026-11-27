"""Standalone MLX forward for one DFly2 draft stage.

The layer consumes the target-selected latent rows supplied by
:class:`DFly2LayerContext` and produces the collapsed hidden stream consumed by
the next standalone runtime stage.  Target history is already in the target
MLA latent representation expected by DFly2; the draft block's own latents are
projected and RoPE'd here.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import mlx.core as mx

from .dfly2_context import DFly2LayerContext


def _get(config, *names):
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    raise AttributeError(f"DFly2 config needs one of {names}")


def _weight(weights: Mapping[str, mx.array], suffix: str) -> mx.array:
    if suffix in weights:
        return weights[suffix]
    matches = [value for name, value in weights.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"expected exactly one DFly2 weight ending in {suffix!r}")
    return matches[0]


def _linear(x: mx.array, weight: mx.array) -> mx.array:
    return x @ weight.T


def _rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps) * weight


def _rope(x: mx.array, positions: mx.array, rope_dim: int, theta: float, *, inverse: bool = False) -> mx.array:
    """Adjacent-pair RoPE on the final ``rope_dim`` channels."""
    if rope_dim == 0:
        return x
    inv_freq = theta ** (-mx.arange(0, rope_dim, 2, dtype=mx.float32) / rope_dim)
    angles = positions.astype(mx.float32)[..., None] * inv_freq
    cos, sin = mx.cos(angles), mx.sin(angles)
    if inverse:
        sin = -sin
    if x.ndim == positions.ndim + 2:
        cos, sin = cos[..., None, :], sin[..., None, :]
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    tail = tail.reshape(*tail.shape[:-1], rope_dim // 2, 2)
    real, imag = tail[..., 0], tail[..., 1]
    rotated = mx.stack((real * cos - imag * sin, real * sin + imag * cos), axis=-1).reshape(*tail.shape[:-2], rope_dim)
    return mx.concatenate((head, rotated.astype(x.dtype)), axis=-1)


def _hc_mixes(h: mx.array, fn: mx.array, scale: mx.array, base: mx.array, *, eps: float, norm_eps: float, iters: int):
    n = h.shape[-2]
    flat = h.flatten(-2).astype(mx.float32)
    mixes = (flat @ fn.astype(mx.float32).T) * mx.rsqrt(mx.mean(flat * flat, axis=-1, keepdims=True) + norm_eps)
    pre = mx.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + eps
    post = 2 * mx.sigmoid(mixes[..., n : 2 * n] * scale[1] + base[n : 2 * n])
    comb = (mixes[..., 2 * n :] * scale[2] + base[2 * n :]).reshape(*mixes.shape[:-1], n, n)
    comb = mx.softmax(comb, axis=-1) + eps
    comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _hc_pre(h: mx.array, pre: mx.array) -> mx.array:
    return mx.sum(h.astype(mx.float32) * pre[..., None], axis=-2).astype(h.dtype)


def _hc_post(x: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    mixed = mx.einsum("btij,btid->btjd", comb, residual.astype(mx.float32))
    return (post[..., None] * x.astype(mx.float32)[..., None, :] + mixed).astype(x.dtype)


def _attention(x: mx.array, start: int, history: DFly2LayerContext, weights: Mapping[str, mx.array], config) -> mx.array:
    heads = int(_get(config, "n_heads", "num_attention_heads"))
    head_dim = int(_get(config, "head_dim"))
    rope_dim = int(_get(config, "rope_head_dim", "qk_rope_head_dim"))
    groups = int(_get(config, "o_groups"))
    o_rank = int(_get(config, "o_lora_rank"))
    eps = float(_get(config, "norm_eps", "rms_norm_eps"))
    theta = float(_get(config, "rope_theta"))
    batch, width, _ = x.shape
    positions = mx.arange(start, start + width, dtype=mx.int32)[None]

    q = _linear(_rms_norm(_linear(x, _weight(weights, "wq_a.weight")), _weight(weights, "q_norm.weight"), eps), _weight(weights, "wq_b.weight"))
    q = _rope(q.reshape(batch, width, heads, head_dim), positions, rope_dim, theta)
    block = _rope(_rms_norm(_linear(x, _weight(weights, "wkv.weight")), _weight(weights, "kv_norm.weight"), eps), positions, rope_dim, theta)

    history_keys = mx.broadcast_to(history.latents[:, None, :, None, :], (batch, width, history.latents.shape[1], heads, head_dim))
    block_keys = mx.broadcast_to(block[:, None, :, None, :], (batch, width, width, heads, head_dim))
    keys = mx.concatenate((history_keys, block_keys), axis=2)
    values = keys
    valid_history = mx.broadcast_to(history.valid[:, None, :], (batch, width, history.valid.shape[1]))
    # DFly2's serving configuration sets dflash_causal_attn=false: every draft
    # slot reads the complete B-slot block, not only its causal prefix.
    valid = mx.concatenate((valid_history, mx.ones((batch, width, width), dtype=mx.bool_)), axis=-1)

    scores = mx.sum(q[:, :, None] * keys, axis=-1) * (head_dim ** -0.5)
    scores = mx.where(valid[..., None], scores, mx.array(-float("inf"), scores.dtype))
    sink = mx.broadcast_to(_weight(weights, "attn_sink").reshape(1, 1, heads, 1).astype(scores.dtype), (batch, width, heads, 1))
    probs = mx.softmax(mx.concatenate((scores.transpose(0, 1, 3, 2), sink), axis=-1), axis=-1)[..., :-1]
    out = mx.sum(probs.transpose(0, 1, 3, 2)[..., None] * values, axis=2)
    out = _rope(out, positions, rope_dim, theta, inverse=True)
    grouped = out.reshape(batch, width, groups, heads * head_dim // groups)
    wo_a = _weight(weights, "wo_a.weight").reshape(groups, o_rank, heads * head_dim // groups)
    projected = mx.einsum("btgd,grd->btgr", grouped, wo_a).reshape(batch, width, groups * o_rank)
    return _linear(projected, _weight(weights, "wo_b.weight"))


def _swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit > 0:
        gate, up = mx.minimum(gate, limit), mx.clip(up, -limit, limit)
    return (gate * mx.sigmoid(gate)) * up


def _moe(x: mx.array, weights: Mapping[str, mx.array], config) -> mx.array:
    experts = int(_get(config, "n_routed_experts", "num_experts"))
    topk = int(_get(config, "n_experts_per_tok", "num_experts_per_tok"))
    route_norm = bool(_get(config, "route_norm"))
    route_scale = float(_get(config, "routed_scaling_factor", "router_scaling_factor"))
    score_func = str(_get(config, "moe_router_score_func"))
    swiglu_limit = float(_get(config, "swiglu_limit"))
    router = _linear(x.astype(mx.float32), _weight(weights, "ffn.gate.weight").astype(mx.float32)) + _weight(weights, "ffn.gate.bias").astype(mx.float32)
    if score_func == "sigmoid":
        scores = mx.sigmoid(router)
    elif score_func == "softmax":
        scores = mx.softmax(router, axis=-1)
    elif score_func == "sqrtsoftplus":
        scores = mx.sqrt(mx.logaddexp(mx.zeros_like(router), router))
    else:
        raise ValueError(f"unsupported DFly2 MoE router score function: {score_func}")
    picked = mx.argsort(scores + _weight(weights, "ffn.gate.bias_vl").astype(mx.float32), axis=-1)[..., -topk:]
    picked_scores = mx.take_along_axis(scores, picked, axis=-1)
    if route_norm:
        picked_scores = picked_scores / mx.sum(picked_scores, axis=-1, keepdims=True)
    # Dispatch only selected (token, expert) pairs.  The route slots are sorted
    # by expert so every expert invocation receives only its assigned tokens;
    # empty experts never run.  Results are scattered into an expert-major
    # buffer before summation so the reduction order still matches the former
    # ascending-expert loop.
    flat_x = x.reshape(-1, x.shape[-1])
    flat_picked = picked.reshape(-1)
    flat_scores = picked_scores.astype(x.dtype).reshape(-1)
    token_ids = mx.repeat(mx.arange(flat_x.shape[0], dtype=mx.uint32), topk)
    route_order = mx.argsort(flat_picked)
    sorted_experts = flat_picked[route_order]
    selected_experts, selected_counts = mx.unique(
        sorted_experts, experts, return_counts=True, fill_value=0
    )
    mx.eval(selected_experts, selected_counts)

    route_offsets = 0
    route_indices = []
    routed_values = []
    for expert, count in zip(selected_experts.tolist(), selected_counts.tolist()):
        if count == 0:
            break
        routes = route_order[route_offsets : route_offsets + count]
        expert_x = flat_x[token_ids[routes]]
        gate = _linear(expert_x, _weight(weights, f"ffn.experts.{expert}.w1.weight"))
        up = _linear(expert_x, _weight(weights, f"ffn.experts.{expert}.w3.weight"))
        value = _linear(_swiglu(gate, up, swiglu_limit), _weight(weights, f"ffn.experts.{expert}.w2.weight"))
        route_indices.append(token_ids[routes].astype(mx.uint32) * experts + flat_picked[routes].astype(mx.uint32))
        routed_values.append(value * flat_scores[routes, None])
        route_offsets += count

    dense_routes = mx.zeros((flat_x.shape[0] * experts, x.shape[-1]), dtype=x.dtype)
    dense_routes = mx.put_along_axis(
        dense_routes,
        mx.concatenate(route_indices)[:, None],
        mx.concatenate(routed_values),
        axis=0,
    )
    out = mx.sum(dense_routes.reshape(*x.shape[:-1], experts, x.shape[-1]), axis=-2)
    shared_gate = _linear(x, _weight(weights, "ffn.shared_experts.w1.weight"))
    shared_up = _linear(x, _weight(weights, "ffn.shared_experts.w3.weight"))
    shared = _linear(_swiglu(shared_gate, shared_up, swiglu_limit), _weight(weights, "ffn.shared_experts.w2.weight"))
    return out * route_scale + shared


def make_draft_layer(weights: Mapping[str, mx.array], config) -> Callable[[mx.array, int, DFly2LayerContext], mx.array]:
    """Bind one official DFly2 stage into ``(x, start_position, history) -> x``.

    The callable returns the mHC-collapsed FFN stream.  This is the standalone
    stage boundary used by :class:`DFly2Runtime`; mHC residual copies are rebuilt
    from that stream at each stage boundary.
    """
    hc_mult = int(_get(config, "hc_mult"))
    norm_eps = float(_get(config, "norm_eps", "rms_norm_eps"))
    hc_eps = float(_get(config, "hc_eps"))
    hc_iters = int(_get(config, "hc_sinkhorn_iters"))

    def layer(x: mx.array, start_position: int, history: DFly2LayerContext) -> mx.array:
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError("DFly2 draft layer expects x [1,T,D]")
        if history.latents.shape[0] != x.shape[0] or history.latents.shape[-1] != int(_get(config, "head_dim")):
            raise ValueError("DFly2 history must match batch and target MLA latent dimension")
        if hc_mult == 1:
            h = x + _attention(_rms_norm(x, _weight(weights, "attn_norm.weight"), norm_eps), int(start_position), history, weights, config)
            return h + _moe(_rms_norm(h, _weight(weights, "ffn_norm.weight"), norm_eps), weights, config)
        h = mx.broadcast_to(x[:, :, None, :], (*x.shape[:2], hc_mult, x.shape[-1]))
        ap, ao, ac = _hc_mixes(h, _weight(weights, "hc_attn_fn"), _weight(weights, "hc_attn_scale"), _weight(weights, "hc_attn_base"), eps=hc_eps, norm_eps=norm_eps, iters=hc_iters)
        h = _hc_post(_attention(_rms_norm(_hc_pre(h, ap), _weight(weights, "attn_norm.weight"), norm_eps), int(start_position), history, weights, config), h, ao, ac)
        fp, fo, fc = _hc_mixes(h, _weight(weights, "hc_ffn_fn"), _weight(weights, "hc_ffn_scale"), _weight(weights, "hc_ffn_base"), eps=hc_eps, norm_eps=norm_eps, iters=hc_iters)
        h = _hc_post(_moe(_rms_norm(_hc_pre(h, fp), _weight(weights, "ffn_norm.weight"), norm_eps), weights, config), h, fo, fc)
        return _hc_pre(h, fp)

    return layer


def attentive_correction(
    block_hidden: mx.array,
    slot: int,
    previous_token: mx.array,
    history: DFly2LayerContext,
    weights: Mapping[str, mx.array],
    config,
    *,
    start_position: int = 0,
) -> mx.array:
    """Apply DFly2 Eq. (11) to one block slot.

    ``previous_token`` is the predecessor-embedding row for every available
    block slot (`[1,T,D]`); a single `[1,D]` row is accepted for the serving
    fallback and is repeated.  The exact gated prefix read needs the former.
    """
    if block_hidden.ndim != 3 or not 0 <= slot < block_hidden.shape[1]:
        raise ValueError("block_hidden must be [1,T,D] and slot must select a block row")
    dim = block_hidden.shape[-1]
    width = block_hidden.shape[1]
    heads = int(_get(config, "n_heads", "num_attention_heads"))
    head_dim = int(_get(config, "head_dim"))
    rope_dim = int(_get(config, "rope_head_dim", "qk_rope_head_dim"))
    groups = int(_get(config, "o_groups"))
    rank = int(_get(config, "o_lora_rank"))
    eps = float(_get(config, "norm_eps", "rms_norm_eps"))
    theta = float(_get(config, "rope_theta"))
    prefix = "hidden_correction."
    hidden_norm = _rms_norm(block_hidden, _weight(weights, prefix + "hidden_norm.weight"), eps)
    if previous_token.ndim == 2:
        previous_token = mx.broadcast_to(previous_token[:, None, :], block_hidden.shape)
    if previous_token.shape != block_hidden.shape:
        raise ValueError("previous_token must be [1,D] or [1,T,D] predecessor embeddings")
    embed_norm = _rms_norm(previous_token.astype(block_hidden.dtype), _weight(weights, prefix + "embed_norm.weight"), eps)
    inputs = mx.concatenate((hidden_norm, embed_norm), axis=-1)
    dense_gate = _linear(inputs[:, slot], _weight(weights, prefix + "gate_proj.weight"))
    inherited = _linear((dense_gate * mx.sigmoid(dense_gate)) * _linear(inputs[:, slot], _weight(weights, prefix + "up_proj.weight")), _weight(weights, prefix + "down_proj.weight"))

    attention = prefix + "attention."
    positions = mx.arange(start_position, start_position + width, dtype=mx.int32)[None]
    q = _linear(_rms_norm(_linear(inputs, _weight(weights, attention + "q_in.weight")), _weight(weights, attention + "q_norm.weight"), eps), _weight(weights, attention + "wq_b.weight")).reshape(1, width, heads, head_dim)
    q = _rope(q, positions, rope_dim, theta)[:, slot]
    # DeepSeek-V4.1's reference takes target context MLA rows directly: they
    # are both key and value latents, with values read from their leading D dims.
    ctx_logits = mx.sum(q[:, None] * history.latents[:, :, None, :], axis=-1) * (head_dim ** -0.5)
    ctx_logits = mx.where(history.valid[:, :, None], ctx_logits, mx.array(-float("inf"), ctx_logits.dtype))
    ctx_sink = mx.broadcast_to(_weight(weights, attention + "attn_sink").reshape(1, heads, 1), (1, heads, 1))
    ctx_probs = mx.softmax(mx.concatenate((ctx_logits.transpose(0, 2, 1), ctx_sink), axis=-1), axis=-1)[..., :-1].transpose(0, 2, 1)
    context_read = mx.sum(ctx_probs[..., None] * history.latents[:, :, None, :], axis=1)

    block_latent = _rope(_rms_norm(_linear(inputs, _weight(weights, attention + "kv_in.weight")), _weight(weights, attention + "kv_norm.weight"), eps), positions, rope_dim, theta)
    keys = block_latent[:, : slot + 1]
    block_logits = mx.sum(q[:, None] * keys[:, :, None, :], axis=-1).transpose(0, 2, 1) * (head_dim ** -0.5)
    back = slot - mx.arange(slot + 1, dtype=mx.int32)
    rel = _weight(weights, attention + "block_rel_bias")[:, back][None]
    block_logits = block_logits + rel
    block_sink = mx.broadcast_to(_weight(weights, attention + "block_sink").reshape(1, heads, 1), (1, heads, 1))
    block_probs = mx.softmax(mx.concatenate((block_logits, block_sink), axis=-1), axis=-1)[..., :-1]
    block_read = mx.sum(block_probs.transpose(0, 2, 1)[..., None] * keys[:, :, None, :], axis=1)
    gate_bias = _weight(weights, attention + "block_gate_bias").reshape(heads, -1)[:, slot]
    block_gate = mx.tanh(_linear(inputs[:, slot], _weight(weights, attention + "block_gate.weight")) + gate_bias)
    read = context_read + block_read * block_gate[..., None]
    position = positions[:, slot : slot + 1]
    read = _rope(read[:, None], position, rope_dim, theta, inverse=True)[:, 0]
    grouped = read.reshape(1, groups, heads * head_dim // groups)
    wo_a = _weight(weights, attention + "wo_a.weight").reshape(groups, rank, heads * head_dim // groups)
    attentive = _linear(mx.einsum("bgd,grd->bgr", grouped, wo_a).reshape(1, groups * rank), _weight(weights, attention + "wo_b.weight"))
    return block_hidden[:, slot] + inherited + attentive
