# SPDX-License-Identifier: Apache-2.0
"""Prefill forward of the Inkling 2 layers, batch one and text only.

Each block follows the path the oMLX host takes for a prefill chunk, op by op
and in the same dtypes, so 16-bit activations round as they do there. The
blocks port ``omlx/patches/inkling2/model.py``; the CUDA kernels that replace
some of them are in ``kernels``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from ... import checkpoint
from ...tensors import Linear
from . import kernels

# ModelOpt NVFP4 tensor scale: amax / (FP8 E4M3 max * FP4 E2M1 max).
NVFP4_DENOMINATOR = 448.0 * 6.0
_FP_SPECS = {
    "nvfp4": {"mode": "nvfp4", "group_size": 16, "bits": 4},
    "mxfp8": {"mode": "mxfp8", "group_size": 32, "bits": 8},
}


def expert_spec(fmt) -> dict:
    """A carrier spec ("nvfp4", "mxfp8" or an oQ dict), as the host reads it."""
    if isinstance(fmt, dict):
        mode = fmt.get("mode", "affine")
        if mode in _FP_SPECS:
            return dict(_FP_SPECS[mode])
        return {
            "mode": mode,
            "group_size": int(fmt.get("group_size", 64)),
            "bits": int(fmt.get("bits", 4)),
        }
    return dict(_FP_SPECS[fmt])


def expert_specs(config: dict, cfg: Config, i: int, w: dict) -> tuple[dict, dict]:
    """Layer ``i``'s (w13, w2) carriers: a converted checkpoint declares them in
    ``quantization``; the published one is NVFP4 where a bank has a tensor
    scale and MXFP8 elsewhere."""
    quant = config.get("quantization")
    if quant:
        default = {k: quant[k] for k in ("group_size", "bits", "mode") if k in quant}
        specs = []
        for which in ("w13", "w2"):
            entry = quant.get(f"model.layers.{i}.mlp.experts.{which}")
            specs.append(expert_spec(entry if isinstance(entry, dict) else default))
        return specs[0], specs[1]
    if cfg.expert_formats:
        return expert_spec(cfg.expert_formats[i]), expert_spec(cfg.expert_formats[i])
    fmt = "nvfp4" if "mlp.experts.w13.global_scale" in w else "mxfp8"
    return expert_spec(fmt), expert_spec(fmt)


# Upper bound on one additive mask block of the plain attention (elements).
_MASK_BUDGET = 1 << 27


@dataclass(frozen=True)
class Config:
    """The text settings the prefill forward reads, with the model defaults."""

    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    swa_heads: int
    swa_kv_heads: int
    swa_head_dim: int
    window: int
    d_rel: int
    rel_extent: int
    log_floor: int
    log_alpha: float
    eps: float
    qk_clip: float
    rel_clip: float
    sliding: tuple[bool, ...]
    conv_k: int
    block: int
    experts: int
    top_k: int
    inter: int
    gate_cap: float
    up_cap: float
    route_scale: float
    vocab: int
    expert_formats: tuple

    @classmethod
    def from_text(cls, text: dict[str, Any]) -> Config:
        layers = text.get("num_hidden_layers", 48)
        kinds = text.get("layer_types")
        local = text.get("local_layer_ids")
        if kinds is not None:
            sliding = tuple(kind == "hybrid_sliding" for kind in kinds)
        elif local is not None:
            sliding = tuple(i in set(local) for i in range(layers))
        else:
            sliding = tuple(bool((i + 1) % 6) for i in range(layers))
        heads = text.get("num_attention_heads", 48)
        kv_heads = text.get("num_key_value_heads", 8)
        head_dim = text.get("head_dim", 128)
        return cls(
            hidden=text.get("hidden_size", 4608),
            layers=layers,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            swa_heads=text.get("swa_num_attention_heads", heads),
            swa_kv_heads=text.get("swa_num_key_value_heads", kv_heads),
            swa_head_dim=text.get("swa_head_dim", head_dim),
            window=text.get("sliding_window_size", 1024),
            d_rel=text.get("d_rel", 16),
            rel_extent=text.get("rel_extent", 1024),
            log_floor=text.get("log_scaling_n_floor", 128000),
            log_alpha=text.get("log_scaling_alpha", 0.1),
            eps=text.get("rms_norm_eps", 1e-6),
            qk_clip=text.get("qk_l2_clip_threshold", 22.627416997969522),
            rel_clip=text.get("rel_l2_clip_threshold", 8.0),
            sliding=sliding,
            conv_k=text.get("conv_kernel_size", text.get("sconv_kernel_size", 4)),
            block=text.get("n_layers_per_attn_res_block", 6),
            experts=text.get("n_routed_experts", 384),
            top_k=text.get("num_experts_per_tok", 8),
            inter=text.get(
                "moe_intermediate_size", text.get("intermediate_size", 2304)
            ),
            gate_cap=text.get("mlp_gate_softcap", 4.0),
            up_cap=text.get("mlp_up_softcap", 25.0),
            route_scale=text.get("route_scale", 8.0),
            vocab=text.get("vocab_size", 201024),
            expert_formats=tuple(text.get("expert_formats") or ()),
        )


def source_layer(raw: dict[str, mx.array], i: int, cfg: Config) -> dict[str, mx.array]:
    """Layer ``i`` of the published checkpoint in the runtime names, as the
    host's loader stacks and renames it (without the ``layers.<i>.`` root)."""
    d = {
        n[len(f"layers.{i}.") :]: v
        for n, v in raw.items()
        if n.startswith(f"layers.{i}.")
    }
    hidden, k = cfg.hidden, cfg.conv_k
    out = {}
    if cfg.sliding[i]:
        parts = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj")
        out["attn.in_proj.weight"] = mx.concatenate(
            [d[f"attn.{p}.weight"] for p in parts], axis=0
        )
        out["attn.k_conv"] = d["attn.k_sconv.weight"].reshape(-1, k)
        out["attn.v_conv"] = d["attn.v_sconv.weight"].reshape(-1, k)
        out["attn_norm.weight"] = d["attn_norm.weight"]
    else:
        parts = ("wq_du", "wr_du", "gate_attn", "attn_sink_proj")
        out["attn.q_proj.weight"] = mx.concatenate(
            [d[f"attn.{p}.weight"] for p in parts], axis=0
        )
        for n in "kv":
            out[f"attn.{n}_proj.weight"] = mx.concatenate(
                [d[f"attn.w{n}_dv.weight"], d[f"attn.{n}_sconv.weight_proj"]], axis=0
            )
            out[f"attn.{n}_static"] = d[f"attn.{n}_sconv.static_w"]
        for n in "qkv":
            out[f"{n}_input_norm.weight"] = d[f"{n}_input_norm.weight"]
    out["attn.o_proj.weight"] = d["attn.wo_ud.weight"]
    out["attn.q_scale"] = d["attn.q_l2_clip.scale"]
    out["attn.r_scale"] = d["attn.rel_l2_clip.scale"]
    out["attn.rel_proj"] = d["attn.rel_logits_proj.proj"]
    out["attn_conv"] = d["attn_sconv.weight"].reshape(hidden, k)
    out["mlp_conv"] = d["mlp_sconv.weight"].reshape(hidden, k)
    out["mlp_norm.weight"] = d["mlp_norm.weight"]
    out["mlp.router_weight"] = d["mlp.gate.weight"].astype(mx.float32)
    out["mlp.router_bias"] = d["mlp.gate.bias"].astype(mx.float32)
    for which in ("w13", "w2"):
        base = f"mlp.experts.{which}"
        out[f"{base}.weight"] = d[f"{base}_weight"].view(mx.uint32)
        out[f"{base}.scales"] = d[f"{base}_weight_scale"]
        amax = d.get(f"{base}_weight_amax")
        if amax is not None:
            denom = d[f"{base}_weight_outer_scale_denominator"]
            out[f"{base}.global_scale"] = amax.astype(mx.float32) * (
                NVFP4_DENOMINATOR / denom.astype(mx.float32)
            )
    if i > 0:
        query = raw[f"attn_res.attn_queries.{i - 1}"]
        out["attn_input.query"] = query.reshape(-1, hidden).astype(mx.float32)
    query = raw[f"attn_res.mlp_queries.{i}"]
    out["mlp_input.query"] = query.reshape(-1, hidden).astype(mx.float32)
    return out


def unit_rms(x: mx.array) -> mx.array:
    return mx.fast.rms_norm(x.astype(mx.float32), None, 1e-6)


def l2_clip(x: mx.array, scale: mx.array | None, threshold: float) -> mx.array:
    xf = x.astype(mx.float32)
    if scale is not None:
        xf = xf * scale
    norm = mx.sqrt(mx.sum(xf * xf, axis=-1, keepdims=True))
    limit = mx.array(threshold, dtype=mx.float32)
    return (xf * mx.minimum(limit / (norm + 1e-6), 1.0)).astype(x.dtype)


def static_conv(x, w, state, scale):
    """Depthwise causal conv; ``w[:, K-1]`` taps the current token. Returns
    the output and the next state, the last K - 1 rows of ``[state, x]``."""
    if kernels.enabled("conv"):
        return kernels.static_conv(x, w, state, scale)
    return plain_static_conv(x, w, state, scale)


def plain_static_conv(x, w, state, scale):
    xf = x.astype(mx.float32)
    k = w.shape[-1]
    xp = mx.concatenate([state, xf], axis=1)
    length = xf.shape[1]
    out = w[:, 0] * xp[:, 0:length]
    for j in range(1, k):
        out = out + w[:, j] * xp[:, j : j + length]
    # The host rounds the conv to the weight dtype before the residual add.
    out = (out.astype(w.dtype).astype(mx.float32) + xf) * scale
    # A copy, so the state does not pin the chunk's window.
    return out.astype(x.dtype), xp[:, -(k - 1) :] * 1.0


def dynamic_conv(x, dyn, static_w, state, heads, scale):
    """Per-token taps ``static_w[lag] + dyn[head, lag]`` over k/v channels."""
    xf = x.astype(mx.float32)
    b, length, c = xf.shape
    k = static_w.shape[0]
    xp = mx.concatenate([state, xf], axis=1)
    dyn = dyn.astype(mx.float32).reshape(b, length, heads, 1, k)
    out = None
    for lag in range(k):
        start = k - 1 - lag
        window = xp[:, start : start + length].reshape(b, length, heads, c // heads)
        tap = static_w[lag].reshape(heads, c // heads) + dyn[..., lag]
        term = window * tap
        out = term if out is None else out + term
    out = (out.reshape(b, length, c) + xf) * scale
    return out.astype(x.dtype), xp[:, -(k - 1) :] * 1.0


def mix(blocks: list, partial: mx.array, query: mx.array) -> mx.array:
    """Softmax over depth sources with learned queries: [1, L, Q, H] float32."""
    scale = partial.shape[-1] ** -0.5
    if kernels.enabled("mix") and len(blocks) + 1 <= kernels.MIX_SOURCES:
        return kernels.mix(blocks, partial, query, scale)
    src = mx.stack([*blocks, partial], axis=-2)
    logits = (src @ query.T) * scale
    w = mx.softmax(logits, axis=-2, precise=True)
    return w.swapaxes(-1, -2) @ src


def masked_attention(q, k, v, rel, sink, scale, window):
    """The plain attention: SDPA over an additive mask that carries the bias
    band and a zero key/value row for the sink. q [1, Hq, L, D], k/v
    [1, Hkv, S, D], rel [1, L, Hq, R], sink [1, L, Hq]; the queries are the
    last L keys."""
    b, hq, length, d = q.shape
    s, r = k.shape[2], rel.shape[-1]
    lo = max(0, s - length - window + 1) if window else 0
    zero = mx.zeros((b, k.shape[1], 1, d), dtype=k.dtype)
    kz = mx.concatenate([zero, k[:, :, lo:]], axis=2)
    vz = mx.concatenate([zero, v[:, :, lo:]], axis=2)
    nk = s - lo
    qb = max(16, min(length, _MASK_BUDGET // max(1, b * hq * (nk + 1))))
    neg = mx.array(-mx.inf, dtype=q.dtype)
    outs = []
    for i0 in range(0, length, qb):
        i1 = min(length, i0 + qb)
        nq = i1 - i0
        a0 = s - length + i0
        hi = s - length + i1
        f = max(lo, a0 - r + 1)
        near = hi - f
        dist = (a0 - f) + mx.arange(nq)[:, None] - mx.arange(near)[None, :]
        relq = rel[:, i0:i1].transpose(0, 2, 1, 3)
        gidx = mx.broadcast_to(mx.clip(dist, 0, r - 1), (b, hq, nq, near))
        bias = mx.take_along_axis(relq, gidx, axis=-1)
        vis = dist >= 0
        if window:
            vis = vis & (dist < window)
        near_mask = mx.where(vis, mx.where(dist < r, bias, 0), neg)
        parts = [sink[:, i0:i1].transpose(0, 2, 1)[..., None].astype(q.dtype)]
        if f > lo:
            fill = neg if window else mx.array(0, dtype=q.dtype)
            parts.append(mx.broadcast_to(fill, (b, hq, nq, f - lo)))
        parts.append(near_mask.astype(q.dtype))
        if hi < s:
            parts.append(mx.broadcast_to(neg, (b, hq, nq, s - hi)))
        mask = mx.concatenate(parts, axis=-1)
        out = mx.fast.scaled_dot_product_attention(
            q[:, :, i0:i1], kz, vz, scale=scale, mask=mask
        )
        mx.eval(out)
        outs.append(out)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


@dataclass
class Cache:
    """One layer's K/V rows [1, kv heads, S, D] and its four conv windows."""

    keys: mx.array | None = None
    values: mx.array | None = None
    conv: list = field(default_factory=lambda: [None] * 4)
    # (keys, values) of the last chunk.
    appended: tuple = ()

    @property
    def offset(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def append(self, keys: mx.array, values: mx.array) -> None:
        self.appended = (keys, values)
        if self.keys is not None:
            keys = mx.concatenate([self.keys, keys], axis=2)
            values = mx.concatenate([self.values, values], axis=2)
        self.keys, self.values = keys, values

    def keep(self, rows: int) -> None:
        """Drop all but the last ``rows`` K/V rows."""
        if self.keys is not None and self.keys.shape[2] > rows:
            self.keys = self.keys[:, :, -rows:] * 1
            self.values = self.values[:, :, -rows:] * 1

    def arrays(self) -> list[mx.array]:
        held = [self.keys, self.values, *self.conv]
        return [a for a in held if a is not None]


class Experts:
    """Routed experts on one stacked carrier per projection. ``w13`` keeps the
    checkpoint's interleaved [gate0, up0, gate1, ...] rows."""

    class Bank:
        def __init__(self, w: dict, name: str, quant: dict):
            self.weight = w[f"{name}.weight"]
            self.scales = w[f"{name}.scales"]
            self.biases = w.get(f"{name}.biases")
            global_scale = w.get(f"{name}.global_scale")
            self.quant = quant
            # The NVFP4 tensor scale applies after the product, in float32.
            self.post = None
            if quant["mode"] == "nvfp4":
                if global_scale is None:
                    global_scale = mx.full(
                        (self.weight.shape[0],), NVFP4_DENOMINATOR, mx.float32
                    )
                self.post = global_scale.astype(mx.float32) * (1.0 / NVFP4_DENOMINATOR)

        @property
        def spec(self) -> dict:
            return self.quant

        @property
        def bits(self) -> int:
            return self.quant["bits"]

        def parameters(self) -> list[mx.array]:
            own = (self.weight, self.scales, self.biases, self.post)
            return [a for a in own if a is not None]

        def __call__(self, x: mx.array, experts: mx.array) -> mx.array:
            q = self.quant
            return mx.gather_qmm(
                x,
                self.weight,
                self.scales,
                self.biases,
                rhs_indices=experts,
                transpose=True,
                group_size=q["group_size"],
                bits=q["bits"],
                mode=q["mode"],
                sorted_indices=True,
            )

    def __init__(self, w: dict, cfg: Config, specs: tuple = ()):
        self.cfg = cfg
        w13, w2 = specs or (expert_spec("nvfp4"),) * 2
        self.w13 = Experts.Bank(w, "mlp.experts.w13", w13)
        self.w2 = Experts.Bank(w, "mlp.experts.w2", w2)
        self.router = w["mlp.router_weight"]
        self.bias = w["mlp.router_bias"]
        # The router weight as [K, experts], the layout the router kernel reads.
        self.columns = mx.contiguous(self.router.T)
        self.caps = (
            mx.array(cfg.gate_cap, dtype=mx.float32),
            mx.array(cfg.up_cap, dtype=mx.float32),
        )

    def parameters(self) -> list[mx.array]:
        return [
            *self.w13.parameters(),
            *self.w2.parameters(),
            self.router,
            self.bias,
            self.columns,
        ]

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Experts [N, K] and their float32 weights, the down post-scale in."""
        cfg = self.cfg
        if kernels.enabled("router_logits") and x.dtype == mx.bfloat16:
            logits = kernels.router_logits(x, self.columns)
        else:
            logits = x.astype(mx.float32) @ self.router.T
        if kernels.enabled("route"):
            inds, weights = kernels.route_topk(
                logits, self.bias, cfg.top_k, cfg.route_scale
            )
        else:
            scores = mx.sigmoid(logits)
            inds = mx.argpartition(-(scores + self.bias), kth=cfg.top_k - 1, axis=-1)
            inds = inds[..., : cfg.top_k]
            weights = mx.take_along_axis(scores, inds, axis=-1)
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
            weights = weights * cfg.route_scale
        if self.w2.post is not None:
            weights = weights * self.w2.post[inds]
        return inds, weights

    def activation(self, y: mx.array, experts: mx.array) -> mx.array:
        """The NVFP4 post-scale (rounded as the product's dtype) and the
        softcapped SwiGLU of the even (gate) and odd (up) columns."""
        if kernels.enabled("activation") and y.dtype == mx.bfloat16:
            return kernels.activation(y, experts, self.w13.post, self.caps)
        return self.plain_activation(y, experts)

    def plain_activation(self, y: mx.array, experts: mx.array) -> mx.array:
        if self.w13.post is not None:
            y = y * self.w13.post[experts][:, None].astype(y.dtype)
        gate = y[..., 0::2].astype(mx.float32)
        up = y[..., 1::2].astype(mx.float32)
        gcap, ucap = self.caps
        out = (
            mx.tanh(gate / gcap) * gcap * mx.sigmoid(gate) * (mx.tanh(up / ucap) * ucap)
        )
        return out.astype(mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        inds, weights = self.route(rows)
        each = inds.shape[-1]
        flat = inds.reshape(-1)
        order = mx.argsort(flat)
        source, experts, inverse = order // each, flat[order], mx.argsort(order)
        routed = rows[source]
        # Each bank takes an expert GEMM if one covers its carrier; the
        # activation is BF16 whatever the row dtype.
        if kernels.covers_bank(self.w13, routed.dtype):
            y = kernels.gather(routed, self.w13, experts)
        else:
            y = self.w13(routed[:, None, :], experts)[:, 0]
        act = self.activation(y, experts)
        if kernels.covers_bank(self.w2, act.dtype):
            down = kernels.gather(act, self.w2, experts)
        else:
            down = self.w2(act[:, None, :], experts)[:, 0]
        if x.dtype == mx.bfloat16 and kernels.enabled("weighted_sum"):
            out = kernels.weighted_sum(down, inverse, weights)
        else:
            down = down[inverse].reshape(*weights.shape, -1)
            out = mx.sum(down.astype(mx.float32) * weights[..., None], axis=-2)
            out = out.astype(x.dtype)
        return out.reshape(shape)


def _linear(w: dict, name: str, config: dict, path: str) -> Linear:
    return Linear(w, name, checkpoint.quantization(config, f"{path}{name}"))


class Layer:
    def __init__(self, w: dict, cfg: Config, i: int, config: dict, path: str):
        self.cfg, self.index = cfg, i
        self.sliding = cfg.sliding[i]
        self.boundary = i % cfg.block == 0
        if self.sliding:
            heads, kv, dim = cfg.swa_heads, cfg.swa_kv_heads, cfg.swa_head_dim
            self.window, self.extent, self.log_floor = cfg.window, cfg.window, None
        else:
            heads, kv, dim = cfg.heads, cfg.kv_heads, cfg.head_dim
            self.window, self.extent = 0, cfg.rel_extent
            self.log_floor = cfg.log_floor
        self.heads, self.kv_heads, self.dim = heads, kv, dim
        self.dkv = kv * dim
        dq, dr = heads * dim, heads * cfg.d_rel
        if self.sliding:
            self.in_proj = _linear(w, "attn.in_proj", config, path)
            self.k_conv, self.v_conv = w["attn.k_conv"], w["attn.v_conv"]
            self.attn_norm = w["attn_norm.weight"]
            splits = (dq, self.dkv, self.dkv, dr, dq)
        else:
            self.q_proj = _linear(w, "attn.q_proj", config, path)
            self.k_proj = _linear(w, "attn.k_proj", config, path)
            self.v_proj = _linear(w, "attn.v_proj", config, path)
            self.k_static, self.v_static = w["attn.k_static"], w["attn.v_static"]
            self.input_norms = [w[f"{n}_input_norm.weight"] for n in "qkv"]
            splits = (dq, dr, dq)
        self.cuts = [sum(splits[: j + 1]) for j in range(len(splits))]
        self.o_proj = _linear(w, "attn.o_proj", config, path)
        self.q_scale, self.r_scale = w["attn.q_scale"], w["attn.r_scale"]
        self.rel_proj = w["attn.rel_proj"]
        self.attn_conv, self.mlp_conv = w["attn_conv"], w["mlp_conv"]
        self.mlp_norm = w["mlp_norm.weight"]
        self.attn_query = w.get("attn_input.query")
        self.mlp_query = w["mlp_input.query"]
        self.experts = Experts(w, cfg, expert_specs(config, cfg, i, w))
        self.scale = dim**-0.5
        self.sconv_scale = 1.0 / math.sqrt(cfg.conv_k + 1)
        self.dconv_scale = 1.0 / math.sqrt(2 * cfg.conv_k + 1)

    def parameters(self) -> list[mx.array]:
        if self.sliding:
            own = [*self.in_proj.parameters(), self.k_conv, self.v_conv, self.attn_norm]
        else:
            own = [
                *self.q_proj.parameters(),
                *self.k_proj.parameters(),
                *self.v_proj.parameters(),
                self.k_static,
                self.v_static,
                *self.input_norms,
            ]
        own += [*self.o_proj.parameters(), self.q_scale, self.r_scale, self.rel_proj]
        own += [self.attn_conv, self.mlp_conv, self.mlp_norm, self.mlp_query]
        if self.attn_query is not None:
            own.append(self.attn_query)
        return own + self.experts.parameters()

    def _tau(self, offset: int, length: int) -> mx.array | None:
        """Query and relative-state scaling of positions past the floor."""
        if self.log_floor is None or offset + length <= self.log_floor:
            return None
        pos = mx.arange(length) + offset
        tau = 1.0 + self.cfg.log_alpha * mx.log(
            mx.maximum((pos + 1).astype(mx.float32) / self.log_floor, 1.0)
        )
        return tau.reshape(1, length, 1, 1)

    def _project(self, xq, xk, xv, cache: Cache):
        """Head-split q/k/v, the relative bias [1, L, H, R], gate and sink."""
        cfg = self.cfg
        length = xq.shape[1]
        states = [
            (
                c
                if c is not None
                else mx.zeros((1, cfg.conv_k - 1, width), dtype=mx.float32)
            )
            for c, width in zip(cache.conv[:2], (self.dkv, self.dkv))
        ]
        if self.sliding:
            q, k, v, r, gate, sink = mx.split(self.in_proj(xq), self.cuts, axis=-1)
            k, ks = static_conv(k, self.k_conv, states[0], self.sconv_scale)
            v, vs = static_conv(v, self.v_conv, states[1], self.sconv_scale)
        else:
            q, r, gate, sink = mx.split(self.q_proj(xq), self.cuts, axis=-1)
            kp, vp = self.k_proj(xk), self.v_proj(xv)
            dkv, kv = self.dkv, self.kv_heads
            k, ks = dynamic_conv(
                kp[..., :dkv],
                kp[..., dkv:],
                self.k_static,
                states[0],
                kv,
                self.dconv_scale,
            )
            v, vs = dynamic_conv(
                vp[..., :dkv],
                vp[..., dkv:],
                self.v_static,
                states[1],
                kv,
                self.dconv_scale,
            )
        cache.conv[0], cache.conv[1] = ks, vs
        q = l2_clip(
            q.reshape(1, length, self.heads, self.dim), self.q_scale, cfg.qk_clip
        )
        k = l2_clip(k.reshape(1, length, self.kv_heads, self.dim), None, cfg.qk_clip)
        v = unit_rms(v.reshape(1, length, self.kv_heads, self.dim)).astype(xq.dtype)
        r = l2_clip(
            r.reshape(1, length, self.heads, cfg.d_rel), self.r_scale, cfg.rel_clip
        )
        tau = self._tau(cache.offset, length)
        if tau is not None:
            q = (q.astype(mx.float32) * tau).astype(q.dtype)
            r = (r.astype(mx.float32) * tau).astype(r.dtype)
        rel = r @ self.rel_proj.astype(r.dtype)
        return q, k, v, rel, gate, sink

    def _attend(self, q, k, v, rel, sink, cache: Cache) -> mx.array:
        """[1, L, H, D] attention output; appends k/v to the cache."""
        cache.append(k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3))
        out = self._attention(q, rel, sink, cache)
        if self.window:
            # The next chunk's first query reads the last window - 1 keys.
            cache.keep(self.window)
        return out

    def _attention(self, q, rel, sink, cache: Cache) -> mx.array:
        if kernels.enabled("attention") and kernels.covers_attention(q, self.extent):
            return kernels.attention(
                q, cache.keys, cache.values, rel, sink, self.scale, self.window
            )
        out = masked_attention(
            q.transpose(0, 2, 1, 3),
            cache.keys,
            cache.values,
            rel,
            sink,
            self.scale,
            self.window,
        )
        return out.transpose(0, 2, 1, 3)

    def __call__(self, h: mx.array, blocks: list, cache: Cache) -> mx.array:
        """``h``: the partial block sum [1, L, H]. ``blocks`` (unit-RMS float32
        completed blocks) gains a block at a boundary layer."""
        cfg, dtype = self.cfg, h.dtype
        length, hidden = h.shape[1], h.shape[2]
        h_n = unit_rms(h)
        if self.index == 0:
            xq = xk = xv = h
        else:
            mixed = mix(blocks, h_n, self.attn_query)
            xq = mixed[:, :, 0]
            if not self.sliding:
                xk, xv = mixed[:, :, 1], mixed[:, :, 2]
        if self.sliding:
            xq = mx.fast.rms_norm(xq, self.attn_norm, cfg.eps).astype(dtype)
            xk = xv = None
        else:
            xq, xk, xv = (
                mx.fast.rms_norm(x, weight, cfg.eps).astype(dtype)
                for x, weight in zip((xq, xk, xv), self.input_norms)
            )
        q, k, v, rel, gate, sink = self._project(xq, xk, xv, cache)
        out = self._attend(q, k, v, rel, sink, cache)
        if self.boundary:
            blocks.append(h_n)
        out = out.reshape(1, length, -1).astype(mx.float32)
        out = (out * mx.sigmoid(gate.astype(mx.float32))).astype(gate.dtype)
        a = self.o_proj(out)
        states = [
            c if c is not None else mx.zeros((1, cfg.conv_k - 1, hidden), mx.float32)
            for c in cache.conv[2:]
        ]
        a, s2 = static_conv(a, self.attn_conv, states[0], self.sconv_scale)
        if self.boundary:
            residual = a.astype(mx.float32)
        else:
            residual = h.astype(mx.float32) + a.astype(mx.float32)
        y = mix(blocks, unit_rms(residual), self.mlp_query)[:, :, 0]
        m = self.experts(mx.fast.rms_norm(y, self.mlp_norm, cfg.eps).astype(dtype))
        m, s3 = static_conv(m, self.mlp_conv, states[1], self.sconv_scale)
        cache.conv[2], cache.conv[3] = s2, s3
        return (residual + m.astype(mx.float32)).astype(dtype)
