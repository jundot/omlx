# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 model (mlx-lm module registered as ``inkling2``).

- Attention residuals: layers exchange a partial block sum, which becomes a
  completed block every ``n_layers_per_attn_res_block`` layers. Attention and
  MLP inputs are learned softmax mixtures over the RMS-normalized blocks.
- No RoPE: position comes from short causal convolutions and a relative bias
  over the last ``rel_extent`` keys. Every attention row has a sink logit.
- Routed experts keep the checkpoint's NVFP4 and MXFP8 codes in MLX's native
  carriers; the NVFP4 tensor scale is applied after the matmul.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

from . import kernels
from .loader import sanitize_weights
from .vision import VisionArgs, VisionTower

_NVFP4 = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
_MXFP8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
# ModelOpt NVFP4 tensor scale: amax / (FP8 E4M3 max * FP4 E2M1 max).
_NVFP4_DENOMINATOR = 448.0 * 6.0


@dataclass
class TextArgs:
    hidden_size: int = 4608
    num_hidden_layers: int = 48
    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 48
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 128
    sliding_window_size: int = 1024
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int = 128000
    log_scaling_alpha: float = 0.1
    rms_norm_eps: float = 1e-6
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    local_layer_ids: list | None = None
    layer_types: list | None = None
    sconv_kernel_size: int = 4
    n_layers_per_attn_res_block: int = 6
    n_routed_experts: int = 384
    num_experts_per_tok: int = 8
    intermediate_size: int = 2304
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    route_scale: float = 8.0
    final_logit_softcapping: float = 30.0
    # Per-layer expert carrier: "nvfp4", "mxfp8" or "affine" (oQ output).
    expert_formats: list | None = None

    @classmethod
    def from_dict(cls, params: dict) -> TextArgs:
        names = cls.__dataclass_fields__
        args = cls(**{k: v for k, v in params.items() if k in names})
        if "moe_intermediate_size" in params:
            args.intermediate_size = int(params["moe_intermediate_size"])
        if "conv_kernel_size" in params:
            args.sconv_kernel_size = int(params["conv_kernel_size"])
        return args

    def is_sliding(self, i: int) -> bool:
        if self.layer_types is not None:
            return self.layer_types[i] == "hybrid_sliding"
        if self.local_layer_ids is not None:
            return i in set(self.local_layer_ids)
        return bool((i + 1) % 6)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "inkling2"
    text_config: dict = field(default_factory=dict)
    vision_config: dict | None = None
    mtp_config: dict | None = None
    eos_token_id: Any = 200006
    image_token_id: int = 200054
    quantization: dict | None = None

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config or {})
        self.vision = (
            VisionArgs.from_dict(self.vision_config) if self.vision_config else None
        )


def _unit_rms(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, None, eps)


def _softcap(x: mx.array, cap: float) -> mx.array:
    return mx.tanh(x / cap) * cap


@mx.compile
def _l2_clip(x: mx.array, scale: mx.array, threshold: mx.array) -> mx.array:
    xf = x.astype(mx.float32) * scale
    norm = mx.sqrt(mx.sum(xf * xf, axis=-1, keepdims=True))
    return (xf * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(x.dtype)


@mx.compile
def _softcapped_swiglu(gate: mx.array, up: mx.array, gcap: mx.array, ucap: mx.array):
    dtype = gate.dtype
    gate = gate.astype(mx.float32)
    up = up.astype(mx.float32)
    out = mx.tanh(gate / gcap) * gcap * mx.sigmoid(gate) * (mx.tanh(up / ucap) * ucap)
    return out.astype(dtype)


def _gather_sort(x, indices):
    *_, m = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // m], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape):
    return mx.unflatten(x[inv_order], 0, shape)


def expert_quant_spec(fmt) -> dict:
    """Normalize an expert carrier spec ("nvfp4", "mxfp8" or an oQ dict)."""
    if isinstance(fmt, dict):
        mode = fmt.get("mode", "affine")
        if mode in ("nvfp4", "mxfp8"):
            return expert_quant_spec(mode)
        return {
            "mode": mode,
            "group_size": int(fmt.get("group_size", 64)),
            "bits": int(fmt.get("bits", 4)),
        }
    if fmt == "nvfp4":
        return dict(_NVFP4)
    if fmt == "mxfp8":
        return dict(_MXFP8)
    raise ValueError(f"Unsupported Inkling 2 expert format: {fmt!r}")


def _expert_specs(args: TextArgs, quantization: dict | None, idx: int):
    """Carrier specs for layer ``idx``'s (w13, w2).

    A converted (oQ) checkpoint declares them in ``quantization``; the source
    checkpoint is NVFP4 except the last MXFP8 layers, and ``sanitize``
    re-checks that against the tensors it receives.
    """
    if quantization:
        default = {
            k: quantization[k]
            for k in ("group_size", "bits", "mode")
            if k in quantization
        }
        specs = []
        for which in ("w13", "w2"):
            entry = quantization.get(f"model.layers.{idx}.mlp.experts.{which}")
            specs.append(
                expert_quant_spec(entry if isinstance(entry, dict) else default)
            )
        return tuple(specs)
    if args.expert_formats:
        fmt = args.expert_formats[idx]
    else:
        fmt = "mxfp8" if idx >= args.num_hidden_layers - 8 else "nvfp4"
    return expert_quant_spec(fmt), expert_quant_spec(fmt)


class ExpertProj(nn.Module):
    """Stacked expert projection on an MLX quantized carrier.

    ``global_scale`` (NVFP4 only) is the per-expert tensor amax in MLX's
    convention; the source scale ``global_scale / (6 * 448)`` is applied after
    the matmul so the packed codes and E4M3 block scales stay unchanged.
    """

    def __init__(self, experts: int, out_dims: int, in_dims: int, spec: dict):
        super().__init__()
        self.mode = spec["mode"]
        self.bits = spec["bits"]
        self.group_size = spec["group_size"]
        per = 32 // self.bits
        groups = in_dims // self.group_size
        self.weight = mx.zeros((experts, out_dims, in_dims // per), dtype=mx.uint32)
        if self.mode == "affine":
            self.scales = mx.zeros((experts, out_dims, groups), dtype=mx.bfloat16)
            self.biases = mx.zeros((experts, out_dims, groups), dtype=mx.bfloat16)
        else:
            self.scales = mx.zeros((experts, out_dims, groups), dtype=mx.uint8)
        if self.mode == "nvfp4":
            self.global_scale = mx.full((experts,), _NVFP4_DENOMINATOR, mx.float32)

    def to_quantized(self, **kwargs):
        # Already a quantized carrier; mlx-lm offers the per-path spec of a
        # converted checkpoint here, which __init__ has applied.
        return self

    def post_scale(self, idx):
        if self.mode != "nvfp4":
            return None
        return self.global_scale[idx] * (1.0 / _NVFP4_DENOMINATOR)

    def __call__(self, x, idx, sorted_indices):
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.get("biases"),
            rhs_indices=idx,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


class Inkling2Experts(nn.Module):
    """Routed experts. ``w13`` keeps the checkpoint's interleaved
    [gate0, up0, gate1, up1, ...] rows; the activation reads the even and odd
    columns."""

    def __init__(self, args: TextArgs, w13_spec: dict, w2_spec: dict):
        super().__init__()
        e, h, i = args.n_routed_experts, args.hidden_size, args.intermediate_size
        self.w13 = ExpertProj(e, 2 * i, h, w13_spec)
        self.w2 = ExpertProj(e, h, i, w2_spec)
        self._gcap = mx.array(args.mlp_gate_softcap, dtype=mx.float32)
        self._ucap = mx.array(args.mlp_up_softcap, dtype=mx.float32)

    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:
        xe = mx.expand_dims(x, (-2, -3))
        do_sort = inds.size >= 64
        idx = inds
        inv = None
        if do_sort:
            xe, idx, inv = _gather_sort(xe, inds)
        y = self.w13(xe, idx, do_sort)
        s13 = self.w13.post_scale(idx)
        if s13 is not None:
            y = y * s13[..., None, None].astype(y.dtype)
        act = _softcapped_swiglu(y[..., 0::2], y[..., 1::2], self._gcap, self._ucap)
        z = self.w2(act, idx, do_sort)
        if do_sort:
            z = _scatter_unsort(z, inv, inds.shape)
        z = z.squeeze(-2)
        s2 = self.w2.post_scale(inds)
        if s2 is not None:
            weights = weights * s2
        out = mx.sum(z.astype(mx.float32) * weights[..., None], axis=-2)
        return out.astype(x.dtype)


class Inkling2MoE(nn.Module):
    def __init__(self, args: TextArgs, w13_spec: dict, w2_spec: dict):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.route_scale = args.route_scale
        self.router_weight = mx.zeros(
            (args.n_routed_experts, args.hidden_size), dtype=mx.float32
        )
        self.router_bias = mx.zeros((args.n_routed_experts,), dtype=mx.float32)
        self.experts = Inkling2Experts(args, w13_spec, w2_spec)

    def route(self, x: mx.array):
        logits = x.astype(mx.float32) @ self.router_weight.T
        return kernels.route_topk(
            logits, self.router_bias, self.top_k, self.route_scale
        )

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        inds, w = self.route(flat)
        return self.experts(flat, inds, w).reshape(shape)


class AttnResQuery(nn.Module):
    """Softmax over depth sources with learned queries (attention residual)."""

    def __init__(self, hidden: int, num_queries: int = 1):
        super().__init__()
        self.query = mx.zeros((num_queries, hidden), dtype=mx.float32)
        self.scale = hidden**-0.5

    def __call__(self, blocks: list, partial_n: mx.array) -> mx.array:
        """``blocks``/``partial_n`` are unit-RMS FP32 sources [B, L, H].

        Returns [B, L, Q, H] FP32 mixtures.
        """
        return kernels.attn_res_mix(blocks, partial_n, self.query, self.scale)


def _static_conv(x, w, state, scale):
    """Depthwise causal conv (taps: ``w[:, K-1]`` is the current token).

    Returns the output and the padded input window ``[state, x]``; its last
    ``K - 1`` rows are the next state.
    """
    xf = x.astype(mx.float32)
    k = w.shape[-1]
    xp = mx.concatenate([state, xf], axis=1)
    length = xf.shape[1]
    out = w[:, 0] * xp[:, 0:length]
    for j in range(1, k):
        out = out + w[:, j] * xp[:, j : j + length]
    # The reference rounds the depthwise conv to the weight dtype before the
    # residual add.
    out = (out.astype(w.dtype).astype(mx.float32) + xf) * scale
    return out.astype(x.dtype), xp


def _dynamic_conv(x, dyn, static_w, state, heads, scale):
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
    return out.astype(x.dtype), xp


class Inkling2Attention(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.sliding = args.is_sliding(layer_idx)
        self.head_dim = args.swa_head_dim if self.sliding else args.head_dim
        self.n_heads = (
            args.swa_num_attention_heads if self.sliding else args.num_attention_heads
        )
        self.n_kv = (
            args.swa_num_key_value_heads if self.sliding else args.num_key_value_heads
        )
        self.window = args.sliding_window_size if self.sliding else 0
        self.rel_extent = args.sliding_window_size if self.sliding else args.rel_extent
        self.d_rel = args.d_rel
        self.scale = self.head_dim**-0.5
        self.log_floor = None if self.sliding else args.log_scaling_n_floor
        self.log_alpha = args.log_scaling_alpha
        self.conv_k = args.sconv_kernel_size
        h = args.hidden_size
        dq = self.n_heads * self.head_dim
        self.dkv = self.n_kv * self.head_dim
        dr = self.n_heads * self.d_rel
        if self.sliding:
            # q | k | v | r | gate | sink share one input.
            splits = (dq, self.dkv, self.dkv, dr, dq, self.n_heads)
            self.in_proj = nn.Linear(h, sum(splits), bias=False)
            self.k_conv = mx.zeros((self.dkv, self.conv_k))
            self.v_conv = mx.zeros((self.dkv, self.conv_k))
        else:
            # q | r | gate | sink from the q input; k and v (plus their dynamic
            # conv taps) from their own attention-residual inputs.
            splits = (dq, dr, dq, self.n_heads)
            self.q_proj = nn.Linear(h, sum(splits), bias=False)
            self.k_proj = nn.Linear(h, self.dkv + self.n_kv * self.conv_k, bias=False)
            self.v_proj = nn.Linear(h, self.dkv + self.n_kv * self.conv_k, bias=False)
            self.k_static = mx.zeros((self.conv_k, self.dkv))
            self.v_static = mx.zeros((self.conv_k, self.dkv))
        self._cuts = list(_cumsum(splits[:-1]))
        self.o_proj = nn.Linear(dq, h, bias=False)
        self.q_scale = mx.ones((self.n_heads, self.head_dim))
        self.r_scale = mx.ones((self.n_heads, self.d_rel))
        self.rel_proj = mx.zeros((self.d_rel, self.rel_extent))
        self._qk_clip = mx.array(args.qk_l2_clip_threshold, dtype=mx.float32)
        self._r_clip = mx.array(args.rel_l2_clip_threshold, dtype=mx.float32)
        self._one = mx.array(1.0, dtype=mx.float32)
        self._sconv_scale = 1.0 / math.sqrt(self.conv_k + 1)
        self._dconv_scale = 1.0 / math.sqrt(2 * self.conv_k + 1)

    def log_tau(self, kv, length: int):
        """Query/relative-state log scaling for long positions (full layers)."""
        if self.log_floor is None:
            return None
        offset = kv.offset if kv is not None else 0
        if isinstance(offset, mx.array):
            offset = offset.reshape(-1, 1)
        pos = mx.arange(length) + offset
        tau = 1.0 + self.log_alpha * mx.log(
            mx.maximum((pos + 1).astype(mx.float32) / self.log_floor, 1.0)
        )
        return tau.reshape(-1, length, 1, 1)

    def project(self, xq, xk, xv, ks, vs, tau):
        """Projections, short convs and clips. Returns head-major q/k/v, the
        relative bias [B, L, H, R], gate, sink and the next conv states."""
        b, length, _ = xq.shape
        dkv = self.dkv
        if self.sliding:
            q, k, v, r, gate, sink = mx.split(self.in_proj(xq), self._cuts, axis=-1)
            k, ks = _static_conv(k, self.k_conv, ks, self._sconv_scale)
            v, vs = _static_conv(v, self.v_conv, vs, self._sconv_scale)
        else:
            q, r, gate, sink = mx.split(self.q_proj(xq), self._cuts, axis=-1)
            kp = self.k_proj(xk)
            vp = self.v_proj(xv)
            k, ks = _dynamic_conv(
                kp[..., :dkv],
                kp[..., dkv:],
                self.k_static,
                ks,
                self.n_kv,
                self._dconv_scale,
            )
            v, vs = _dynamic_conv(
                vp[..., :dkv],
                vp[..., dkv:],
                self.v_static,
                vs,
                self.n_kv,
                self._dconv_scale,
            )
        q = _l2_clip(
            q.reshape(b, length, self.n_heads, self.head_dim),
            self.q_scale,
            self._qk_clip,
        )
        k = _l2_clip(
            k.reshape(b, length, self.n_kv, self.head_dim), self._one, self._qk_clip
        )
        v = _unit_rms(v.reshape(b, length, self.n_kv, self.head_dim).astype(mx.float32))
        v = v.astype(xq.dtype)
        r = _l2_clip(
            r.reshape(b, length, self.n_heads, self.d_rel), self.r_scale, self._r_clip
        )
        if tau is not None:
            q = (q.astype(mx.float32) * tau).astype(q.dtype)
            r = (r.astype(mx.float32) * tau).astype(r.dtype)
        rel = r @ self.rel_proj.astype(r.dtype)
        return (
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            rel,
            gate,
            sink,
            ks,
            vs,
        )

    def attend(self, q, k, v, rel, sink, kv):
        if kv is not None:
            k, v = kv.update_and_fetch(k, v)
        left_padding = getattr(kv, "left_padding", None) if kv is not None else None
        return kernels.banded_sink_attention(
            q,
            k,
            v,
            rel,
            sink,
            scale=self.scale,
            window=self.window,
            left_padding=left_padding,
        )

    def output(self, out, gate):
        b, _, length, _ = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        out = (out.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(
            gate.dtype
        )
        return self.o_proj(out)


def _cumsum(xs):
    total = 0
    for x in xs:
        total += x
        yield total


# Inputs up to this many tokens per row (decode and speculative verify) run
# the layer through compiled stages; longer prefill chunks stay eager, since
# every new chunk length would otherwise retrace all layers.
_COMPILED_MAX_TOKENS = 16


class Inkling2Layer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int, w13_spec: dict, w2_spec: dict):
        super().__init__()
        self.layer_idx = layer_idx
        self.sliding = args.is_sliding(layer_idx)
        self.is_boundary = layer_idx % args.n_layers_per_attn_res_block == 0
        eps = args.rms_norm_eps
        h = args.hidden_size
        if layer_idx > 0:
            self.attn_input = AttnResQuery(h, 1 if self.sliding else 3)
        self.mlp_input = AttnResQuery(h, 1)
        self.attn = Inkling2Attention(args, layer_idx)
        self.mlp = Inkling2MoE(args, w13_spec, w2_spec)
        if self.sliding:
            self.attn_norm = nn.RMSNorm(h, eps=eps)
        else:
            self.q_input_norm = nn.RMSNorm(h, eps=eps)
            self.k_input_norm = nn.RMSNorm(h, eps=eps)
            self.v_input_norm = nn.RMSNorm(h, eps=eps)
        self.mlp_norm = nn.RMSNorm(h, eps=eps)
        self.attn_conv = mx.zeros((h, args.sconv_kernel_size))
        self.mlp_conv = mx.zeros((h, args.sconv_kernel_size))
        self._conv_scale = 1.0 / math.sqrt(args.sconv_kernel_size + 1)
        self._conv_k = args.sconv_kernel_size
        # A weak proxy: compiled bound methods would make each layer a
        # reference cycle, so a released model would wait for the cyclic GC.
        layer = weakref.proxy(self)
        self._pre_compiled = mx.compile(lambda *a: Inkling2Layer._pre(layer, *a))
        self._post_compiled = mx.compile(lambda *a: Inkling2Layer._post(layer, *a))

    def _pre(self, h, blocks, ks, vs, tau=None):
        dtype = h.dtype
        h_n = _unit_rms(h.astype(mx.float32))
        if self.layer_idx == 0:
            xq = xk = xv = h
        else:
            mixed = self.attn_input(blocks, h_n)
            if self.sliding:
                xq = mixed[:, :, 0]
            else:
                xq, xk, xv = mixed[:, :, 0], mixed[:, :, 1], mixed[:, :, 2]
        if self.sliding:
            xq = self.attn_norm(xq).astype(dtype)
            xk = xv = None
        else:
            xq = self.q_input_norm(xq).astype(dtype)
            xk = self.k_input_norm(xk).astype(dtype)
            xv = self.v_input_norm(xv).astype(dtype)
        return (*self.attn.project(xq, xk, xv, ks, vs, tau), h_n)

    def _post(self, out, gate, h, mlp_blocks, s2, s3):
        a = self.attn.output(out, gate)
        a, s2 = _static_conv(a, self.attn_conv, s2, self._conv_scale)
        if self.is_boundary:
            residual = a.astype(mx.float32)
        else:
            residual = h.astype(mx.float32) + a.astype(mx.float32)
        y = self.mlp_input(mlp_blocks, _unit_rms(residual))[:, :, 0]
        m = self.mlp(self.mlp_norm(y).astype(h.dtype))
        m, s3 = _static_conv(m, self.mlp_conv, s3, self._conv_scale)
        return (residual + m.astype(mx.float32)).astype(h.dtype), s2, s3

    def __call__(self, h, blocks, cache=None):
        """``h``: partial block sum [B, L, H] bf16. ``blocks``: list of unit-RMS
        FP32 completed blocks; extended in place at a block boundary."""
        b, length, hidden = h.shape
        kv = cache[0] if cache is not None else None
        conv = cache[1] if cache is not None else None
        dkv = self.attn.dkv
        states = []
        for slot, width in ((0, dkv), (1, dkv), (2, hidden), (3, hidden)):
            s = conv[slot] if conv is not None else None
            if s is None:
                s = mx.zeros((b, self._conv_k - 1, width), dtype=mx.float32)
            states.append(s)
        tau = self.attn.log_tau(kv, length)
        compiled = length <= _COMPILED_MAX_TOKENS
        pre = self._pre_compiled if compiled else self._pre
        post = self._post_compiled if compiled else self._post
        pre_args = (h, blocks, states[0], states[1])
        if tau is not None:
            pre_args = (*pre_args, tau)
        q, k, v, rel, gate, sink, ks, vs, h_n = pre(*pre_args)
        out = self.attn.attend(q, k, v, rel, sink, kv)
        if self.is_boundary:
            blocks.append(h_n)
        h, s2, s3 = post(out, gate, h, blocks, states[2], states[3])
        if conv is not None:
            windows = (ks, vs, s2, s3)
            if 1 < length <= _COMPILED_MAX_TOKENS:
                # Speculative verify rollback re-cuts the states from these.
                conv._omlx_verify_xp = dict(enumerate(windows))
            keep = self._conv_k - 1
            for slot, window in enumerate(windows):
                # A small copy, so the state does not pin the chunk's window.
                conv[slot] = window[:, -keep:] * 1.0
        return h


class Inkling2TextModel(nn.Module):
    def __init__(self, args: TextArgs, quant: dict | None = None):
        super().__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layers = [
            Inkling2Layer(args, i, *_expert_specs(args, quant, i))
            for i in range(args.num_hidden_layers)
        ]
        self.final_input = AttnResQuery(args.hidden_size, 1)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, input_embeddings=None, capture=None):
        """``capture`` maps draft feature ids to fill: ``i`` < num layers is
        the partial block sum entering layer ``i``; ``num layers`` is the
        final-norm output."""
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_norm(self.embed(inputs))
        if cache is None:
            cache = [None] * len(self.layers)
        blocks: list = []
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            if capture is not None and i in capture:
                capture[i] = h
            h = layer(h, blocks, cache=c)
        x = self.final_input(blocks, _unit_rms(h.astype(mx.float32)))[:, :, 0]
        out = self.norm(x).astype(h.dtype)
        if capture is not None and len(self.layers) in capture:
            capture[len(self.layers)] = out
        return out


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        quant = args.quantization
        self.model = Inkling2TextModel(args.text, quant)
        self.lm_head = nn.Linear(
            args.text.hidden_size, args.text.vocab_size, bias=False
        )
        if args.vision is not None:
            self.vision_tower = VisionTower(args.vision, args.text.hidden_size)
        self._softcap = args.text.final_logit_softcapping
        vocab = args.text.vocab_size
        unpadded = args.text.unpadded_vocab_size
        self._vocab_bias = None
        if unpadded and unpadded < vocab:
            self._vocab_bias = mx.where(
                mx.arange(vocab) < unpadded, 0.0, -mx.inf
            ).astype(mx.float32)

    def logits_from_hidden(self, h):
        logits = self.lm_head(h).astype(mx.float32)
        logits = _softcap(logits, self._softcap)
        if self._vocab_bias is not None:
            logits = logits + self._vocab_bias
        return logits

    def encode_image(self, pixel_values: mx.array, image_grid_thw) -> mx.array:
        """Image-token embeddings for the processor's pixel blocks."""
        if getattr(self, "vision_tower", None) is None:
            raise ValueError("This Inkling 2 checkpoint has no vision tower.")
        return self.vision_tower(pixel_values, image_grid_thw)

    def embed_inputs(
        self, inputs, pixel_values=None, image_grid_thw=None, features=None
    ):
        """Normalized token embeddings with the image features at the image
        tokens (the features themselves are not normalized)."""
        embeds = self.model.embed_norm(self.model.embed(inputs))
        if pixel_values is None and features is None:
            return embeds
        if features is None:
            features = self.encode_image(pixel_values, image_grid_thw)
        mask = inputs == self.args.image_token_id
        count = int(mask.sum().item())
        if count != features.shape[0]:
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens={count}, features={features.shape[0]}"
            )
        # The feature row of an image token: the image tokens before it.
        rows = mx.cumsum(mask.flatten()).reshape(mask.shape) - 1
        rows = features[mx.maximum(rows, 0)].astype(embeds.dtype)
        return mx.where(mask[..., None], rows, embeds)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        pixel_values=None,
        image_grid_thw=None,
        **kwargs,
    ):
        if pixel_values is not None:
            input_embeddings = self.embed_inputs(inputs, pixel_values, image_grid_thw)
        return self.forward(
            inputs,
            cache,
            input_embeddings,
            drafter=getattr(self, "_omlx_drafter", None),
            **kwargs,
        )

    def forward(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        *,
        drafter=None,
        return_hidden: bool = False,
        capture_layer_ids=None,
        **kwargs,
    ):
        """Text forward; ``drafter`` is the block drafter attached to whichever
        object the generator calls (this model or its VLM language model)."""
        scope = getattr(drafter, "scope_uids", None)
        # An ordinary decode step keeps the committed token in the drafter
        # context.
        observe = bool(
            capture_layer_ids is None
            and not return_hidden
            and scope
            and inputs is not None
            and inputs.ndim == 2
            and tuple(inputs.shape) == (len(scope), 1)
        )
        if observe:
            capture_layer_ids = list(drafter.target_layer_ids)
        if not capture_layer_ids and not return_hidden:
            return self.logits_from_hidden(self.model(inputs, cache, input_embeddings))
        capture = {int(i): None for i in capture_layer_ids or ()}
        h = self.model(inputs, cache, input_embeddings, capture=capture)
        logits = self.logits_from_hidden(h)
        captures = [capture[int(i)] for i in capture_layer_ids or ()]
        if observe:
            drafter.observe(scope, captures)
            return logits
        if return_hidden:
            return SimpleNamespace(
                logits=logits, hidden_states=captures + [h], gdn_states=None
            )
        return SimpleNamespace(logits=logits, hidden_states=captures)

    # Rows that accept different counts would need a ragged conv rollback.
    _omlx_drafter_single_row = True

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Drop the rejected drafts of a ``num_drafts + 1``-token verify."""
        rejected = int(num_drafts) - int(accepted)
        if rejected <= 0:
            return True
        keep = int(accepted) + 1
        width = self.args.text.sconv_kernel_size - 1
        for entry in cache:
            kv, conv = entry[0], entry[1]
            windows = getattr(conv, "_omlx_verify_xp", None)
            if windows is None or kv.trim(rejected) != rejected:
                return False
            for slot, window in windows.items():
                conv[slot] = window[:, keep : keep + width] * 1.0
            conv._omlx_verify_xp = None
        return True

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        accepted = int(accepted)
        if not self.mtp_partial_rollback(caches, accepted, int(block_size) - 1):
            raise RuntimeError("Inkling 2 verify rollback found no conv windows")
        return accepted

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [CacheList(KVCache(), ArraysCache(4)) for _ in self.model.layers]

    def sanitize(self, weights: dict) -> dict:
        out, formats = sanitize_weights(self, weights)
        args = self.args.text
        hidden, inter = args.hidden_size, args.intermediate_size
        experts = args.n_routed_experts
        for idx, fmt in formats.items():
            mod = self.model.layers[idx].mlp.experts
            spec = expert_quant_spec(fmt)
            if mod.w13.mode != spec["mode"]:
                mod.w13 = ExpertProj(experts, 2 * inter, hidden, spec)
                mod.w2 = ExpertProj(experts, hidden, inter, spec)
        return out

    @property
    def cast_predicate(self):
        def predicate(key: str) -> bool:
            # Attention-residual queries, router and NVFP4 tensor scales are
            # FP32 in the source checkpoint and stay FP32.
            return not (
                key.endswith(".query")
                or key.endswith("router_weight")
                or key.endswith("router_bias")
                or key.endswith(".global_scale")
            )

        return predicate
