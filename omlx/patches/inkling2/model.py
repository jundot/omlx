# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 model (mlx-lm module registered as ``inkling2``).

- Attention residuals: layers exchange a partial block sum, which becomes a
  completed block every ``n_layers_per_attn_res_block`` layers. Attention and
  MLP inputs are learned softmax mixtures over the RMS-normalized blocks.
- No RoPE: position comes from short causal convolutions and a relative bias
  over the last ``rel_extent`` keys. Every attention row has a sink logit.
- Routed experts keep the checkpoint's NVFP4 and MXFP8 codes in MLX's native
  carriers; the NVFP4 tensor scale is applied after the matmul.
- Looped checkpoints (``loop_config``) run the core layers several times per
  token. Every later pass restarts from the prelude output and reads the
  previous pass's blocks only through gated messages. Each layer visit owns
  its cache state, so caches and draft taps count visits, not layers.
- A sliding visit caches only the keys its window can still reach
  (``cache.py``); a full-attention visit caches every key.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import ArraysCache, KVCache

from ..qwen35_verify_qmm import is_row_exact_armed
from . import cache as windows
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
    loop_config: dict | None = None

    @classmethod
    def from_dict(cls, params: dict) -> TextArgs:
        names = cls.__dataclass_fields__
        args = cls(**{k: v for k, v in params.items() if k in names})
        if "moe_intermediate_size" in params:
            args.intermediate_size = int(params["moe_intermediate_size"])
        if "conv_kernel_size" in params:
            args.sconv_kernel_size = int(params["conv_kernel_size"])
        if args.loop_config and "loop_num_passes" in params:
            args.loop_config = {
                **args.loop_config,
                "num_passes": int(params["loop_num_passes"]),
            }
        return args

    def is_sliding(self, i: int) -> bool:
        if self.layer_types is not None:
            return self.layer_types[i] == "hybrid_sliding"
        if self.local_layer_ids is not None:
            return i in set(self.local_layer_ids)
        return bool((i + 1) % 6)

    @property
    def loop_core(self) -> range:
        loop = self.loop_config
        return range(loop["core_start"], loop["core_end"]) if loop else range(0)

    def layer_schedule(self) -> list[tuple[int, int]]:
        """``(layer, pass)`` for every layer visit in execution order."""
        n = self.num_hidden_layers
        if not self.loop_config:
            return [(i, 0) for i in range(n)]
        core = self.loop_core
        block = self.n_layers_per_attn_res_block
        passes = int(self.loop_config["num_passes"])
        if (
            not 0 < core.start < core.stop <= n
            or core.start % block
            or core.stop % block
            or passes < 1
        ):
            # The message bank is the blocks one pass writes, so the core must
            # start and end on attention-residual block boundaries.
            raise ValueError(f"Unsupported Inkling 2 loop_config: {self.loop_config}")
        return (
            [(i, 0) for i in range(core.start)]
            + [(i, p) for p in range(passes) for i in core]
            + [(i, 0) for i in range(core.stop, n)]
        )


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


_UNIT_EPS = 1e-6


def _unit_rms(x: mx.array, eps: float = _UNIT_EPS) -> mx.array:
    return mx.fast.rms_norm(x, None, eps)


def _softcap(x: mx.array, cap: float) -> mx.array:
    return mx.tanh(x / cap) * cap


def _logit_tail(logits: mx.array, cap: float, bias: mx.array | None) -> mx.array:
    logits = _softcap(logits.astype(mx.float32), cap)
    return logits if bias is None else logits + bias


_logit_tail_compiled = mx.compile(_logit_tail)


def _keep_fp32_weight(norm: nn.Module) -> None:
    """Keep an FP32 copy of an RMSNorm weight for ``_norm``.

    The copy lives outside the parameter tree, so it is never saved, loaded
    or quantized, and is rebuilt when the weight is replaced. Call it outside
    compiled traces: the copy must be a materialized array.
    """
    if type(norm) is not nn.RMSNorm:
        return
    weight = norm.weight
    held = norm.__dict__.get("_omlx_fp32_weight")
    if held is None or held[0] is not weight:
        copy = weight.astype(mx.float32)
        mx.eval(copy)
        object.__setattr__(norm, "_omlx_fp32_weight", (weight, copy))


def _fp32_norm_stack(owner: nn.Module, norms, tag: str) -> mx.array | None:
    """The FP32 weight copies of ``norms`` stacked [N, H], or None. Built outside
    compiled traces and cached on ``owner``."""
    copies = []
    for norm in norms:
        held = norm.__dict__.get("_omlx_fp32_weight")
        if held is None or held[0] is not norm.weight:
            return None
        copies.append(held[1])
    attr = f"_omlx_norm_stack_{tag}"
    cached = owner.__dict__.get(attr)
    if cached is None or any(a is not b for a, b in zip(cached[0], copies)):
        stacked = mx.stack(copies)
        mx.eval(stacked)
        cached = (tuple(copies), stacked)
        object.__setattr__(owner, attr, cached)
    return cached[1]


def _norm(norm: nn.Module, x: mx.array) -> mx.array:
    """``norm(x)``. An FP32 input takes the weight's FP32 copy: rms_norm would
    cast a BF16 weight to FP32 on every call, a kernel each time, for the
    same values. Other inputs keep the weight's dtype in the arithmetic."""
    held = norm.__dict__.get("_omlx_fp32_weight")
    if x.dtype == mx.float32 and held is not None and held[0] is norm.weight:
        return mx.fast.rms_norm(x, held[1], norm.eps)
    return norm(x)


@mx.compile
def _l2_clip(x: mx.array, scale: mx.array, threshold: mx.array) -> mx.array:
    xf = x.astype(mx.float32) * scale
    norm = mx.sqrt(mx.sum(xf * xf, axis=-1, keepdims=True))
    return (xf * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(x.dtype)


@mx.compile
def _softcapped_swiglu(
    gate: mx.array,
    up: mx.array,
    gcap: mx.array,
    ucap: mx.array,
    scale: mx.array | None = None,
):
    dtype = gate.dtype
    if scale is not None:
        # The NVFP4 tensor scale, in the matmul output's dtype; applying it
        # here fuses it into this kernel.
        gate = gate * scale
        up = up * scale
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


_LHS_INDICES: dict = {}


def _lhs_indices(batch_shape: tuple, rhs_shape: tuple) -> mx.array:
    """The lhs indices ``gather_qmm`` defaults to (an arange over the batch of
    ``x``, broadcast against the rhs indices) as a constant array; left to
    ``gather_qmm`` they cost an arange kernel per call.

    Built from host data, so a compiled trace may keep it. The shapes are
    those of unsorted (decode and verify) rows, so the cache stays small.
    """
    key = (tuple(batch_shape), tuple(rhs_shape))
    lhs = _LHS_INDICES.get(key)
    if lhs is None:
        rows = np.arange(math.prod(batch_shape), dtype=np.uint32)
        shape = np.broadcast_shapes(tuple(batch_shape), tuple(rhs_shape))
        rows = np.broadcast_to(rows.reshape(batch_shape), shape)
        lhs = _LHS_INDICES[key] = mx.array(np.ascontiguousarray(rows))
    return lhs


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
    the matmul so the packed codes and E4M3 block scales stay unchanged. The
    unscaled output is ~1e4 times the real one and overflows float16, so
    float16 activations pass ``global_scale`` to ``gather_qmm`` instead.
    """

    def __init__(self, experts: int, out_dims: int, in_dims: int, spec: dict):
        super().__init__()
        self.mode = spec["mode"]
        self.bits = spec["bits"]
        self.group_size = spec["group_size"]
        packed_dims = in_dims * self.bits // 32
        groups = in_dims // self.group_size
        self.weight = mx.zeros((experts, out_dims, packed_dims), dtype=mx.uint32)
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

    def scales_in_kernel(self, dtype) -> bool:
        return self.mode == "nvfp4" and dtype == mx.float16

    def post_scale(self, idx, dtype):
        if self.mode != "nvfp4" or self.scales_in_kernel(dtype):
            return None
        return self.global_scale[idx] * (1.0 / _NVFP4_DENOMINATOR)

    def __call__(self, x, idx, sorted_indices):
        extra = {}
        if self.scales_in_kernel(x.dtype):
            extra["global_scale"] = self.global_scale
        if not sorted_indices:
            extra["lhs_indices"] = _lhs_indices(x.shape[:-2], idx.shape)
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
            **extra,
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
        """FP32 sum of the routed expert outputs; the caller rounds it to the
        activation dtype (the MLP conv kernel does, on read)."""
        return moe_sum(*self.unreduced(x, inds, weights))

    def unreduced(self, x: mx.array, inds: mx.array, weights: mx.array):
        """The routed expert outputs [N, K, H] and their FP32 weights [N, K]
        (NVFP4 tensor scale included), before ``moe_sum``."""
        xe = mx.expand_dims(x, (-2, -3))
        # Sorting regroups the pairs by expert and changes their arithmetic;
        # row-exact verify rows keep the per-pair gather of a decode step.
        do_sort = inds.size >= 64 and not is_row_exact_armed()
        idx = inds
        inv = None
        if do_sort:
            xe, idx, inv = _gather_sort(xe, inds)
        y = self.w13(xe, idx, do_sort)
        s13 = self.w13.post_scale(idx, xe.dtype)
        if s13 is not None:
            s13 = s13[..., None, None].astype(y.dtype)
        act = _softcapped_swiglu(
            y[..., 0::2], y[..., 1::2], self._gcap, self._ucap, s13
        )
        z = self.w2(act, idx, do_sort)
        if do_sort:
            z = _scatter_unsort(z, inv, inds.shape)
        z = z.squeeze(-2)
        s2 = self.w2.post_scale(inds, act.dtype)
        if s2 is not None:
            weights = weights * s2
        return z, weights


def moe_sum(z: mx.array, weights: mx.array) -> mx.array:
    """FP32 weighted sum of expert outputs over their axis (-2)."""
    return mx.sum(z.astype(mx.float32) * weights[..., None], axis=-2)


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

    def route(self, x: mx.array, x32: mx.array | None = None):
        if x32 is None:
            x32 = x.astype(mx.float32)
        logits = _row_matmul(x32, self.router_weight.T)
        return kernels.route_topk(
            logits, self.router_bias, self.top_k, self.route_scale
        )

    def __call__(self, x: mx.array, x32: mx.array | None = None) -> mx.array:
        """FP32 output, not yet rounded to ``x``'s dtype (see ``Inkling2Experts``).
        ``x32``, when given, is ``x.astype(float32)`` already computed."""
        return moe_sum(*self.unreduced(x, x32))

    def unreduced(self, x: mx.array, x32: mx.array | None = None):
        """Expert outputs [..., K, H] and weights [..., K] before their sum."""
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if x32 is not None:
            x32 = x32.reshape(-1, shape[-1])
        inds, w = self.route(flat, x32)
        z, w = self.experts.unreduced(flat, inds, w)
        return z.reshape(*shape[:-1], *z.shape[-2:]), w.reshape(*shape[:-1], -1)


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


def _mix_messages(x, message, down, gain_forget):
    """Loop controllers: ``(1 - tanh(forget)) * x + tanh(gain) * message``.

    ``x`` and ``message`` are [N, B, L, H] FP32, one row per controlled input.
    ``down`` is [N, 2, H, r] (the x and message halves) and ``gain_forget``
    [N, r, 2H], both FP32 copies of BF16 weights.
    """
    n, b, length, hidden = x.shape
    rows = b * length
    # Like the reference, the down projection takes BF16-rounded inputs,
    # accumulates in FP32 and rounds once. It runs as a batched matmul over
    # the halves: a single [rows, 2H] x [2H, r] matmul gets a row-count
    # dependent kernel, so prefill chunking would change the result.
    z = mx.stack([_unit_rms(x), _unit_rms(message)], axis=1)
    z = z.astype(mx.bfloat16).astype(mx.float32).reshape(2 * n, rows, hidden)
    low = _row_matmul(z, down.reshape(2 * n, hidden, -1))
    low = low.reshape(n, 2, rows, -1).sum(axis=1)
    low = low.astype(mx.bfloat16).astype(mx.float32)
    gates = _row_matmul(low, gain_forget).reshape(n, b, length, 2 * hidden)
    gain = mx.tanh(gates[..., :hidden])
    forget = mx.tanh(gates[..., hidden:])
    return (1.0 - forget) * x + gain * message


def _part(x: mx.array, i: int) -> mx.array:
    """``x[i]`` as a slice view: integer indexing of an axis runs a gather."""
    return x[i : i + 1].squeeze(0)


def _mix_messages_normed(
    mixed, messages, m0, n, down, gain_forget, norm_w, norm_eps, dtype, want_f32=False
):
    """``_mix_messages`` and the following RMSNorm and cast, with the steps
    around its matmuls in kernels; None to fall back. Input i is query i of
    ``mixed`` (or query 0) and message ``m0 + i``. Returns [n, B, L, H]."""
    if norm_w is None or not kernels.loop_messages_supported(mixed, messages):
        return None
    hidden = mixed.shape[-1]
    z = kernels.loop_message_inputs(mixed, messages, m0, n)
    low = _row_matmul(z, down.reshape(2 * n, hidden, -1))
    gates = _row_matmul(kernels.loop_message_low(low), gain_forget)
    return kernels.loop_message_outputs(
        gates, mixed, messages, m0, n, norm_w, norm_eps, dtype, want_f32
    )


def _row_matmul(x: mx.array, w: mx.array) -> mx.array:
    """``x @ w``. Row-exact verify rows take one matmul each, as a decode step
    does: a multi-row matmul switches kernels with the row count."""
    rows = x.shape[-2]
    if rows > 1 and is_row_exact_armed():
        # One single-row product per row in a batch: with a stride-0 weight
        # batch MLX keeps the one-row kernel instead of folding the rows into
        # a multi-row matmul.
        batch = mx.broadcast_shapes(x.shape[:-2], w.shape[:-2])
        wb = mx.broadcast_to(w[..., None, :, :], (*batch, rows, *w.shape[-2:]))
        return (x[..., :, None, :] @ wb).squeeze(-2)
    return x @ w


def _project_slice(
    proj: nn.Module, x: mx.array, start: int, end: int | None
) -> mx.array:
    """Output rows ``[start:end]`` of a bias-free Linear or QuantizedLinear."""
    if type(proj) is nn.QuantizedLinear:
        return _row_slice(proj, start, end)(x)
    return x @ proj.weight[start:end].T


def _row_slice(proj: nn.QuantizedLinear, start: int, end: int | None):
    """A QuantizedLinear over output rows ``[start:end]`` of ``proj``.

    Calling it rather than ``quantized_matmul`` lets the row-exact verify
    patch of ``QuantizedLinear`` serve the slice. Cached per weight arrays.
    """
    source = (proj["weight"], proj["scales"])
    cache = proj.__dict__.get("_omlx_row_slices")
    if cache is None or any(a is not b for a, b in zip(cache[0], source)):
        cache = (source, {})
        object.__setattr__(proj, "_omlx_row_slices", cache)
    sub = cache[1].get((start, end))
    if sub is None:
        sub = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
        nn.Module.__init__(sub)
        sub.group_size, sub.bits, sub.mode = proj.group_size, proj.bits, proj.mode
        sub.weight = proj.weight[start:end]
        sub.scales = proj.scales[start:end]
        if proj.get("biases") is not None:
            sub.biases = proj.biases[start:end]
        cache[1][(start, end)] = sub
    return sub


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
            self._tail_cuts = [dr, dr + dq]
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
            if xk is None:
                q, k, v, r, gate, sink = mx.split(self.in_proj(xq), self._cuts, axis=-1)
            elif type(self.in_proj) in (nn.Linear, nn.QuantizedLinear):
                # Looped passes give q, k and v their own inputs; project only
                # the row slices of ``in_proj`` that each input uses.
                c0, c1, c2 = self._cuts[:3]
                q = _project_slice(self.in_proj, xq, 0, c0)
                k = _project_slice(self.in_proj, xk, c0, c1)
                v = _project_slice(self.in_proj, xv, c1, c2)
                tail = _project_slice(self.in_proj, xq, c2, None)
                r, gate, sink = mx.split(tail, self._tail_cuts, axis=-1)
            else:
                # Wrapped projections (e.g. LoRALinear) fall back to the full
                # stacked call so the wrapper's forward is preserved.
                pq, pk, pv = mx.split(
                    self.in_proj(mx.concatenate([xq, xk, xv], axis=0)), 3, axis=0
                )
                q, _, _, r, gate, sink = mx.split(pq, self._cuts, axis=-1)
                k = pk[..., self._cuts[0] : self._cuts[1]]
                v = pv[..., self._cuts[1] : self._cuts[2]]
            k, ks = kernels.short_conv(k, self.k_conv, ks, self._sconv_scale)
            v, vs = kernels.short_conv(
                v, self.v_conv, vs, self._sconv_scale, out_dtype=mx.float32
            )
        else:
            q, r, gate, sink = mx.split(self.q_proj(xq), self._cuts, axis=-1)
            kp = self.k_proj(xk)
            vp = self.v_proj(xv)
            k, ks = kernels.dynamic_conv(
                kp[..., :dkv],
                kp[..., dkv:],
                self.k_static,
                ks,
                self.n_kv,
                self._dconv_scale,
            )
            v, vs = kernels.dynamic_conv(
                vp[..., :dkv],
                vp[..., dkv:],
                self.v_static,
                vs,
                self.n_kv,
                self._dconv_scale,
                out_dtype=mx.float32,
            )
        fused = kernels.qkv_post(
            q,
            k,
            v,
            r,
            self.q_scale,
            self.r_scale,
            self._qk_clip,
            self._r_clip,
            tau,
            self.n_heads,
            self.n_kv,
        )
        if fused is not None:
            q, k, v, r = fused
            return self._head_major(q, k, v, r, gate, sink, ks, vs)
        q = _l2_clip(
            q.reshape(b, length, self.n_heads, self.head_dim),
            self.q_scale,
            self._qk_clip,
        )
        k = _l2_clip(
            k.reshape(b, length, self.n_kv, self.head_dim), self._one, self._qk_clip
        )
        # The convs hand v over in FP32 (rounded to the activation dtype first).
        v = _unit_rms(v.reshape(b, length, self.n_kv, self.head_dim))
        v = v.astype(xq.dtype)
        r = _l2_clip(
            r.reshape(b, length, self.n_heads, self.d_rel), self.r_scale, self._r_clip
        )
        if tau is not None:
            q = (q.astype(mx.float32) * tau).astype(q.dtype)
            r = (r.astype(mx.float32) * tau).astype(r.dtype)
        return self._head_major(q, k, v, r, gate, sink, ks, vs)

    def _head_major(self, q, k, v, r, gate, sink, ks, vs):
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

    def attend(self, q, k, v, rel, sink, kv, state=None):
        left_padding = None
        if kv is not None:
            k, v = kv.update_and_fetch(k, v)
            left_padding = getattr(kv, "left_padding", None)
        elif state is not None:
            k, v, left_padding = windows.append(state, k, v)
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
        if layer_idx in args.loop_core:
            # Message queries and controllers for the q, k, v and MLP inputs,
            # in that order. Only passes after the first use them.
            rank = int(args.loop_config["gate_rank"])
            self.loop_message = AttnResQuery(h, 4)
            self.loop_down = mx.zeros((4, 2, h, rank), dtype=mx.float32)
            self.loop_gain_forget = mx.zeros((4, rank, 2 * h), dtype=mx.float32)
        self.attn = Inkling2Attention(args, layer_idx)
        self.mlp = Inkling2MoE(args, w13_spec, w2_spec)
        if self.sliding:
            self.attn_norm = nn.RMSNorm(h, eps=eps)
        else:
            self.q_input_norm = nn.RMSNorm(h, eps=eps)
            self.k_input_norm = nn.RMSNorm(h, eps=eps)
            self.v_input_norm = nn.RMSNorm(h, eps=eps)
        self.mlp_norm = nn.RMSNorm(h, eps=eps)
        self._norm_eps = eps
        self.attn_conv = mx.zeros((h, args.sconv_kernel_size))
        self.mlp_conv = mx.zeros((h, args.sconv_kernel_size))
        self._conv_scale = 1.0 / math.sqrt(args.sconv_kernel_size + 1)
        self._conv_k = args.sconv_kernel_size
        # A weak proxy: compiled bound methods would make each layer a
        # reference cycle, so a released model would wait for the cyclic GC.
        layer = weakref.proxy(self)
        self._pre_compiled = mx.compile(lambda *a: Inkling2Layer._pre(layer, *a))
        self._post_compiled = mx.compile(lambda *a: Inkling2Layer._post(layer, *a))
        # The row-exact flag is read while tracing, so its rows get their own
        # traces instead of reusing those of an unflagged forward of that width.
        self._pre_exact = mx.compile(lambda *a: Inkling2Layer._pre(layer, *a))
        self._post_exact = mx.compile(lambda *a: Inkling2Layer._post(layer, *a))

    def _pre(self, h, blocks, ks, vs, tau=None, messages=None):
        dtype = h.dtype
        attn_w = self._omlx_norm_w[0]
        fused = None
        if self.layer_idx > 0:
            fused = kernels.attn_res_fused(
                blocks,
                h,
                self.attn_input.query,
                self.attn_input.scale,
                unit_eps=_UNIT_EPS,
                norm_weight=attn_w if messages is None else None,
                norm_eps=self._norm_eps,
                out_dtype=dtype,
                want_partial=True,
            )
        if fused is not None:
            mixed, h_n = fused
            if messages is None and attn_w is not None:
                parts = [_part(mixed, i) for i in range(mixed.shape[0])]
                xq, xk, xv = (parts[0], None, None) if self.sliding else parts
                return (*self.attn.project(xq, xk, xv, ks, vs, tau), h_n)
        else:
            h_n = _unit_rms(h.astype(mx.float32))
            if self.layer_idx > 0:
                mixed = self.attn_input(blocks, h_n)
        if messages is not None:
            normed = _mix_messages_normed(
                mixed[:, :, : 1 if self.sliding else 3],
                messages,
                0,
                3,
                self.loop_down[:3],
                self.loop_gain_forget[:3],
                attn_w,
                self._norm_eps,
                dtype,
            )
            if normed is not None:
                xq, xk, xv = (_part(normed[0], i) for i in range(3))
                return (*self.attn.project(xq, xk, xv, ks, vs, tau), h_n)
        if self.layer_idx == 0:
            xq = xk = xv = h
        elif self.sliding:
            xq = xk = xv = mixed[:, :, 0]
        else:
            xq, xk, xv = mixed[:, :, 0], mixed[:, :, 1], mixed[:, :, 2]
        if messages is not None:
            xq, xk, xv = _mix_messages(
                mx.stack([xq, xk, xv]),
                messages[:, :, :3].transpose(2, 0, 1, 3),
                self.loop_down[:3],
                self.loop_gain_forget[:3],
            )
        if self.sliding and messages is None:
            xq = _norm(self.attn_norm, xq).astype(dtype)
            xk = xv = None
        elif self.sliding:
            xq, xk, xv = (_norm(self.attn_norm, x).astype(dtype) for x in (xq, xk, xv))
        else:
            xq = _norm(self.q_input_norm, xq).astype(dtype)
            xk = _norm(self.k_input_norm, xk).astype(dtype)
            xv = _norm(self.v_input_norm, xv).astype(dtype)
        return (*self.attn.project(xq, xk, xv, ks, vs, tau), h_n)

    def _post(self, out, gate, h, mlp_blocks, s2, s3, messages=None):
        a = self.attn.output(out, gate)
        # The conv kernels also do the FP32 residual adds: the attention
        # output joins h (off a block boundary), the MLP output the residual.
        residual, s2 = kernels.short_conv(
            a,
            self.attn_conv,
            s2,
            self._conv_scale,
            residual=None if self.is_boundary else h,
            out_dtype=mx.float32,
        )
        norm_w = None if messages is not None else self._omlx_norm_w[1]
        fused = kernels.attn_res_fused(
            mlp_blocks,
            residual,
            self.mlp_input.query,
            self.mlp_input.scale,
            unit_eps=_UNIT_EPS,
            norm_weight=norm_w,
            norm_eps=self.mlp_norm.eps,
            out_dtype=h.dtype,
            want_f32=True,
        )
        if fused is not None and norm_w is not None:
            x, x32 = _part(fused[0], 0), _part(fused[1], 0)
        else:
            if fused is not None:
                y = fused[0][:, :, 0]
            else:
                y = self.mlp_input(mlp_blocks, _unit_rms(residual))[:, :, 0]
            normed = None
            if messages is not None:
                normed = _mix_messages_normed(
                    fused[0] if fused is not None else y[:, :, None],
                    messages,
                    3,
                    1,
                    self.loop_down[3:],
                    self.loop_gain_forget[3:],
                    self._omlx_norm_w[1],
                    self.mlp_norm.eps,
                    h.dtype,
                    want_f32=True,
                )
            if normed is not None:
                x, x32 = _part(normed[0], 0), _part(normed[1], 0)
            else:
                if messages is not None:
                    y = _mix_messages(
                        y[None],
                        messages[None, :, :, 3],
                        self.loop_down[3:],
                        self.loop_gain_forget[3:],
                    ).squeeze(0)
                x, x32 = _norm(self.mlp_norm, y).astype(h.dtype), None
        z, zw = self.mlp.unreduced(x, x32)
        conv = kernels.moe_short_conv(
            z, zw, self.mlp_conv, s3, self._conv_scale, residual, h.dtype
        )
        if conv is None:
            conv = kernels.short_conv(
                moe_sum(z, zw),
                self.mlp_conv,
                s3,
                self._conv_scale,
                residual=residual,
                dtype=h.dtype,
            )
        h, s3 = conv
        return h, s2, s3

    def __call__(self, h, blocks, cache=None, bank=None, tau=None):
        """``h``: partial block sum [B, L, H] bf16. ``blocks``: list of unit-RMS
        FP32 completed blocks; extended in place at a block boundary.
        ``bank``: the previous loop pass's ``(blocks, partial_n)`` messages.
        ``tau``: a full-attention visit's log scaling of these positions, when
        the caller shares one across visits."""
        b, length, hidden = h.shape
        kv, conv = cache if cache is not None else (None, None)
        # A sliding visit's state starts with its key and value windows.
        base = windows.CONV if kv is None else 0
        dkv = self.attn.dkv
        states = []
        for slot, width in ((0, dkv), (1, dkv), (2, hidden), (3, hidden)):
            s = conv[base + slot] if conv is not None else None
            if s is None:
                s = mx.zeros((b, self._conv_k - 1, width), dtype=mx.float32)
            states.append(s)
        if tau is None:
            tau = self.attn.log_tau(kv, length)
        if self.sliding:
            norms = (self.attn_norm, self.mlp_norm)
        else:
            norms = (self.q_input_norm, self.k_input_norm, self.v_input_norm)
            norms += (self.mlp_norm,)
        for norm in norms:
            _keep_fp32_weight(norm)
        object.__setattr__(
            self,
            "_omlx_norm_w",
            (
                _fp32_norm_stack(self, norms[:-1], "attn"),
                _fp32_norm_stack(self, norms[-1:], "mlp"),
            ),
        )
        compiled = length <= _COMPILED_MAX_TOKENS
        if compiled and length > 1 and is_row_exact_armed():
            pre, post = self._pre_exact, self._post_exact
        elif compiled:
            pre, post = self._pre_compiled, self._post_compiled
        else:
            pre, post = self._pre, self._post
        messages = self.loop_message(*bank) if bank is not None else None
        q, k, v, rel, gate, sink, ks, vs, h_n = pre(
            h, blocks, states[0], states[1], tau, messages
        )
        out = self.attn.attend(q, k, v, rel, sink, kv, conv)
        if self.is_boundary:
            blocks.append(h_n)
        h, s2, s3 = post(out, gate, h, blocks, states[2], states[3], messages)
        if conv is not None:
            padded = (ks, vs, s2, s3)
            if 1 < length <= _COMPILED_MAX_TOKENS:
                # Speculative verify rollback re-cuts the states from these.
                conv._omlx_verify_xp = dict(enumerate(padded))
            keep = self._conv_k - 1
            for slot, window in enumerate(padded):
                tail = window[:, -keep:]
                # A prefill chunk's tail is copied, so the state does not pin
                # the chunk's window; a decode or verify window is a few rows.
                conv[base + slot] = tail if compiled else tail * 1.0
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
        self._schedule = tuple(args.layer_schedule())
        self._core_start = args.loop_core.start if args.loop_config else None

    @property
    def num_visits(self) -> int:
        if self._core_start is None:
            # The prefill helper keeps only the first layers of a plain model.
            return len(self.layers)
        return len(self._schedule)

    def cache_layout(self) -> list[tuple[int | None, int]]:
        """Where each layer visit's ``(KVCache, state)`` sit in a cache list.

        A sliding visit has no ``KVCache``; its one entry holds everything.
        """
        layout, entry = [], 0
        for idx, _ in self._schedule[: self.num_visits]:
            if self.layers[idx].sliding:
                layout.append((None, entry))
                entry += 1
            else:
                layout.append((entry, entry + 1))
                entry += 2
        return layout

    def visit_caches(self, cache: list) -> list[tuple]:
        """``(KVCache or None, state)`` of every layer visit in ``cache``."""
        layout = self.cache_layout()
        entries = layout[-1][1] + 1 if layout else 0
        if len(cache) != entries:
            # A per-layer cache would silently skip the later loop passes.
            raise ValueError(
                f"Inkling 2 needs {entries} cache entries for its "
                f"{len(layout)} layer visits, got {len(cache)}"
            )
        window = self.args.sliding_window_size
        return [
            (None, windows.ArraysCache.adopt(cache[state], window))
            if kv is None
            else (cache[kv], cache[state])
            for kv, state in layout
        ]

    def __call__(self, inputs, cache=None, input_embeddings=None, capture=None):
        """``capture`` maps draft feature ids to fill: ``i`` < num visits is
        the partial block sum entering layer visit ``i``; ``num visits`` is
        the final-norm output."""
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_norm(self.embed(inputs))
        visits = [None] * self.num_visits if cache is None else self.visit_caches(cache)
        blocks: list = []
        prelude = bank = tau = None
        for visit, ((idx, loop_pass), c) in enumerate(zip(self._schedule, visits)):
            layer = self.layers[idx]
            if idx == self._core_start and prelude is None:
                prelude = (h, list(blocks))
            elif idx == self._core_start:
                # The next pass restarts from the prelude output; the blocks
                # the last pass wrote become its message bank.
                bank = (
                    blocks[len(prelude[1]) + 1 :],
                    _unit_rms(h.astype(mx.float32)),
                )
                h, blocks = prelude[0], list(prelude[1])
            if capture is not None and visit in capture:
                capture[visit] = h
            if not layer.sliding and tau is None:
                # The full-attention visits hold the same positions, so they
                # share one log scaling.
                tau = layer.attn.log_tau(c[0] if c is not None else None, h.shape[1])
            h = layer(
                h,
                blocks,
                cache=c,
                bank=bank if loop_pass else None,
                tau=None if layer.sliding else tau,
            )
        _keep_fp32_weight(self.norm)
        norm_w = _fp32_norm_stack(self, (self.norm,), "final")
        fused = None
        if norm_w is not None:
            fused = kernels.attn_res_fused(
                blocks,
                h,
                self.final_input.query,
                self.final_input.scale,
                unit_eps=_UNIT_EPS,
                norm_weight=norm_w,
                norm_eps=self.norm.eps,
                out_dtype=h.dtype,
            )
        if fused is not None:
            out = _part(fused[0], 0)
        else:
            x = self.final_input(blocks, _unit_rms(h.astype(mx.float32)))[:, :, 0]
            out = _norm(self.norm, x).astype(h.dtype)
        if capture is not None and self.num_visits in capture:
            capture[self.num_visits] = out
        return out


class Model(nn.Module):
    # ``attend`` reads the keys and values a cache returns as dense arrays; a
    # packed (TurboQuant or affine) cache hands back codes it cannot use.
    dense_kv_attention = True
    # Verify windows up to the compiled width keep the compiled forward, and
    # mtp_partial_rollback recuts the conv tails for any rejected count.
    _omlx_mtp_max_verify_rows = _COMPILED_MAX_TOKENS

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

    @property
    def _omlx_mtp_row_exact_verify(self) -> bool:
        """Verify rows reproduce the serial decode rows only when every
        projection they take is a QuantizedLinear (``row_exact_qmv``); a dense
        multi-row matmul has its own arithmetic."""
        linears = [self.lm_head]
        for layer in self.model.layers:
            attn = layer.attn
            names = ("in_proj",) if attn.sliding else ("q_proj", "k_proj", "v_proj")
            linears += [getattr(attn, name) for name in names] + [attn.o_proj]
        return all(type(m) is nn.QuantizedLinear for m in linears)

    def logits_from_hidden(self, h):
        # Decode and verify rows run the cast, softcap and vocab bias as one
        # kernel; longer prefill chunks stay eager, like the layers.
        compiled = h.shape[-2] <= _COMPILED_MAX_TOKENS
        tail = _logit_tail_compiled if compiled else _logit_tail
        return tail(self.lm_head(h), self._softcap, self._vocab_bias)

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
        for kv, conv in self.model.visit_caches(cache):
            padded = getattr(conv, "_omlx_verify_xp", None)
            if padded is None:
                return False
            if kv is None:
                if not windows.trim(conv, rejected):
                    return False
            elif kv.trim(rejected) != rejected:
                return False
            base = windows.CONV if kv is None else 0
            for slot, xp in padded.items():
                # A verify window is a few rows, so the state may stay a view.
                conv[base + slot] = xp[:, keep : keep + width]
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
        # Every layer visit, each loop pass included, keeps its own state: a
        # sliding visit one entry with its key window and conv states, a
        # full-attention visit a KVCache and its conv states.
        window = self.args.text.sliding_window_size
        cache = []
        for kv, _ in self.model.cache_layout():
            if kv is None:
                cache.append(windows.ArraysCache(window=window))
            else:
                cache += [KVCache(), ArraysCache(4)]
        return cache

    def restore_cache(self, cache: list) -> list:
        """Prefix-cache hook: restored sliding states become joinable again."""
        self.model.visit_caches(cache)
        return cache

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
            # Attention-residual and loop message queries, router and NVFP4
            # tensor scales are FP32 in the source checkpoint and stay FP32.
            # The loop controller weights are widened to FP32 at load.
            return not (
                key.endswith(".query")
                or key.endswith("router_weight")
                or key.endswith("router_bias")
                or key.endswith(".global_scale")
                or key.endswith("loop_down")
                or key.endswith("loop_gain_forget")
            )

        return predicate
