# SPDX-License-Identifier: MIT
# ruff: noqa
# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_kernel, gated_delta_ops
from .mla import MultiLinear
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    n_group: int
    topk_group: int
    first_k_dense_replace: int
    layer_group_size: int
    group_norm_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    routed_scaling_factor: float
    head_dim: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    q_lora_rank: Optional[int] = None
    rope_interleave: bool = True
    partial_rotary_factor: float = 0.5
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    use_qkv_bias: bool = False
    use_bias: bool = False
    use_qk_norm: bool = True
    score_function: str = "sigmoid"
    norm_topk_prob: bool = True
    moe_router_enable_expert_bias: bool = True
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 0
    gated_attention_proj_granularity_type: Optional[str] = None
    no_kda_lora: bool = True
    kda_safe_gate: bool = False
    kda_lower_bound: Optional[float] = None
    short_conv_kernel_size: int = 4
    quantization_config: Optional[Dict[str, Any]] = None
    # Ling is trained with a clamped SwiGLU on its late layers; the limits
    # are per-layer and differ for routed vs shared experts. 0 means "no
    # clamp on this layer".
    expert_swiglu_limit_list: Optional[list] = None
    share_expert_swiglu_limit_list: Optional[list] = None


def recurrent_gla(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    scale: float,
    h: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    L = q.shape[2]
    in_dtype = q.dtype
    # Keep the recurrent state in float32; precision loss here compounds over
    # long prompts through repeated multiplication by exp(g) < 1.
    exp_g = mx.exp(g)[:, None, None].astype(mx.float32)
    q = (q * scale).astype(mx.float32)
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    if h is not None:
        h = h.astype(mx.float32)
    outputs = []
    for t in range(L):
        q_t = q[:, :, t : t + 1]
        k_t = k[:, :, t : t + 1]
        v_t = v[:, :, t : t + 1]
        h_up = k_t.transpose(0, 1, 3, 2) @ v_t
        h = h_up if h is None else h * exp_g + h_up
        outputs.append(q_t @ h)
    return mx.concatenate(outputs, axis=2).astype(in_dtype), h


class GroupRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5, groups: int = 1):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.groups = groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.unflatten(x, axis=-1, shape=(self.groups, -1))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * mx.flatten(x, -2)


# Marks this build as implementing Ling's SwiGLU clamp in-source, so
# omlx.patches.bailing_hybrid.swiglu_clamp leaves it alone.
_omlx_swiglu_clamp_native = True


def layer_swiglu_limit(limit_list, layer_idx: int) -> Optional[float]:
    """Per-layer SwiGLU clamp, or None when absent or zero."""
    if not limit_list or layer_idx >= len(limit_list):
        return None
    limit = limit_list[layer_idx]
    if limit in (None, 0):
        return None
    return float(limit)


def clamped_swiglu(gate: mx.array, x: mx.array, limit: float) -> mx.array:
    """SwiGLU with Ling's trained clamp.

    ``silu(gate).clamp(max=limit) * x.clamp(-limit, limit)``, matching
    vLLM's ``SwigluStepAndMul`` (see BailingMoeV3MLP in inclusionAI's vLLM
    port). Without it the late layers run unclamped and activations there
    can run away.
    """
    return mx.minimum(nn.silu(gate), limit) * mx.clip(x, -limit, limit)


class ClampedSwiGLU(nn.Module):
    """SwitchGLU-signature activation: ``__call__(x_up, x_gate)``."""

    def __init__(self, limit: float):
        super().__init__()
        self.limit = float(limit)

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return clamped_swiglu(gate, x, self.limit)


class MLP(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        intermediate_size: Optional[int] = None,
        swiglu_limit: Optional[float] = None,
    ):
        super().__init__()
        dim = intermediate_size if intermediate_size is not None else args.intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size, dim, bias=args.use_bias)
        self.up_proj = nn.Linear(args.hidden_size, dim, bias=args.use_bias)
        self.down_proj = nn.Linear(dim, args.hidden_size, bias=args.use_bias)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.swiglu_limit:
            return self.down_proj(clamped_swiglu(gate, up, self.swiglu_limit))
        return self.down_proj(swiglu(gate, up))


class MultiLatentAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.q_lora_rank = args.q_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = args.v_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.gated_attention_proj_granularity_type = (
            args.gated_attention_proj_granularity_type
        )

        self.scale = self.qk_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                args.hidden_size, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                args.hidden_size, self.q_lora_rank, bias=args.use_qkv_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.use_qkv_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)

        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )

        self.dense = nn.Linear(
            self.num_heads * self.v_head_dim,
            args.hidden_size,
            bias=args.use_qkv_bias,
        )
        if args.gated_attention_proj_granularity_type == "head_wise":
            self.g_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        elif args.gated_attention_proj_granularity_type == "element_wise":
            self.g_proj = nn.Linear(
                args.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
            )
        else:
            self.g_proj = None

        if args.rope_scaling is not None:
            mscale_all_dim = args.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = args.rope_scaling.get("factor", 1)
            if mscale_all_dim and scaling_factor > 1:
                s = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                self.scale = self.scale * s * s

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=args.rope_interleave,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_heads, self.qk_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        if self.g_proj is not None:
            gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
            if self.gated_attention_proj_granularity_type == "head_wise":
                output = output * gate.transpose(0, 2, 1)[..., None]
            else:
                output = output * gate.reshape(
                    B, L, self.num_heads, self.v_head_dim
                ).transpose(0, 2, 1, 3)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


class DepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        scale = math.sqrt(1.0 / channels)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(channels, 1, kernel_size),
        )

    def __call__(
        self,
        x: mx.array,
        cache: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        lengths: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        batch, length, channels = x.shape
        kernel_size = self.weight.shape[-1]
        if cache is None:
            cache = mx.zeros((batch, channels, kernel_size), dtype=x.dtype)

        if mask is not None:
            x = mx.where(mask[..., None], x, 0)

        history = cache[:, :, -kernel_size + 1 :].transpose(0, 2, 1)
        conv_input = mx.concatenate([history, x], axis=1)
        weight = self.weight.moveaxis(2, 1).astype(x.dtype)
        output = nn.silu(mx.conv_general(conv_input, weight, groups=channels))

        cache_input = mx.concatenate([cache, x.transpose(0, 2, 1)], axis=2)
        if lengths is not None:
            ends = mx.clip(lengths, 0, length)
            positions = (ends[:, None] + mx.arange(kernel_size))[..., None]
            next_cache = mx.take_along_axis(
                cache_input.transpose(0, 2, 1), positions, axis=1
            ).transpose(0, 2, 1)
        else:
            next_cache = mx.contiguous(cache_input[:, :, -kernel_size:])
        return output, mx.contiguous(next_cache)


class GatedRMSNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float):
        super().__init__()
        self.weight = mx.ones((head_dim,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        x = mx.fast.rms_norm(x, weight=self.weight, eps=self.eps)
        return x * mx.sigmoid(gate.astype(mx.float32)).astype(x.dtype)


def recurrent_kda(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
    h: Optional[mx.array] = None,
    safe_gate: bool = True,
    lower_bound: Optional[float] = -5.0,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Run Ling's KDA recurrence through mlx-lm's fused gated-delta kernel."""
    batch, length, heads, head_dim = q.shape
    if h is None:
        state = mx.zeros((batch, heads, head_dim, head_dim), dtype=mx.float32)
    else:
        # Ling stores the reference state as [key_dim, value_dim], while
        # mlx-lm's fused kernel uses [value_dim, key_dim].
        state = h.swapaxes(-1, -2).astype(mx.float32)

    inv_scale = head_dim**-0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    gate_input = g.astype(mx.float32) + dt_bias.reshape(
        1, 1, heads, head_dim
    ).astype(mx.float32)
    a_log = a_log.reshape(1, 1, heads, 1).astype(mx.float32)
    if safe_gate and lower_bound is not None:
        log_decay = lower_bound * mx.sigmoid(mx.exp(a_log) * gate_input)
    else:
        log_decay = -mx.exp(a_log) * nn.softplus(gate_input)
    decay = mx.exp(log_decay)
    beta = mx.sigmoid(beta)

    if (
        head_dim % 32 == 0
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        output, state = gated_delta_kernel(q, k, v, decay, beta, state, mask)
    else:
        output, state = gated_delta_ops(q, k, v, decay, beta, state, mask)

    return output, state.swapaxes(-1, -2)


class LinearAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.safe_gate = args.kda_safe_gate
        self.lower_bound = args.kda_lower_bound

        projection_size = self.num_attention_heads * self.head_dim
        self.q_proj = nn.Linear(args.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, projection_size, bias=False)
        self.q_conv1d = DepthwiseConv1d(
            projection_size, args.short_conv_kernel_size
        )
        self.k_conv1d = DepthwiseConv1d(
            projection_size, args.short_conv_kernel_size
        )
        self.v_conv1d = DepthwiseConv1d(
            projection_size, args.short_conv_kernel_size
        )
        self.A_log = mx.zeros((self.num_attention_heads,), dtype=mx.float32)
        self.dt_bias = mx.zeros((projection_size,), dtype=mx.float32)
        self.f_proj = nn.Linear(args.hidden_size, projection_size, bias=False)
        self.b_proj = nn.Linear(args.hidden_size, self.num_attention_heads, bias=False)
        self.g_proj = nn.Linear(args.hidden_size, projection_size, bias=False)
        self.o_norm = GatedRMSNorm(self.head_dim, args.rms_norm_eps)
        self.o_proj = nn.Linear(projection_size, args.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        offset: int = 0,
    ) -> mx.array:
        B, L, _ = x.shape

        # Keep each recurrent tensor in its own ArraysCache slot. The source
        # branch stores one tuple in one slot, but pinned mlx-lm batches by
        # indexing/merging slots individually; a tuple there breaks every
        # BatchGenerator prompt-to-decode transition. Prefixes created by the
        # source branch may still restore as a one-slot cache, so migrate that
        # legacy layout in place before external prefill reuses it.
        state = None
        if cache is not None:
            slots = list(cache.state)
            if len(slots) == 1:
                legacy_state = slots[0]
                if legacy_state is None:
                    slots = [None] * 4
                elif isinstance(legacy_state, (list, tuple)) and len(legacy_state) == 4:
                    slots = list(legacy_state)
                else:
                    raise ValueError(
                        "Invalid bailing_hybrid recurrent cache: expected a "
                        "four-tensor legacy state"
                    )
                cache.state = slots
            if len(slots) != 4:
                raise ValueError(
                    "Invalid bailing_hybrid recurrent cache: expected 4 slots, "
                    f"got {len(slots)}"
                )
            state = tuple(slots)
        if state is None or all(value is None for value in state):
            recurrent_state = conv_q = conv_k = conv_v = None
        else:
            recurrent_state, conv_q, conv_k, conv_v = state

        lengths = cache.lengths if cache is not None else None
        if lengths is not None:
            # ``ArraysCache.merge`` initializes ``left_padding`` even for an
            # empty cache, and its generic ``make_mask`` checks that field
            # before ``lengths``. During right-padded prompt batches the
            # resulting all-true mask would advance the recurrent state over
            # padding. The current chunk lengths are authoritative here.
            mask = mx.arange(L)[None, :] < lengths[:, None]
        q, conv_q = self.q_conv1d(self.q_proj(x), conv_q, mask, lengths)
        k, conv_k = self.k_conv1d(self.k_proj(x), conv_k, mask, lengths)
        v, conv_v = self.v_conv1d(self.v_proj(x), conv_v, mask, lengths)
        q = q.reshape(B, L, self.num_attention_heads, self.head_dim)
        k = k.reshape(B, L, self.num_attention_heads, self.head_dim)
        v = v.reshape(B, L, self.num_attention_heads, self.head_dim)
        f = self.f_proj(x).reshape(
            B, L, self.num_attention_heads, self.head_dim
        )
        beta = self.b_proj(x)

        output, recurrent_state = recurrent_kda(
            q,
            k,
            v,
            f,
            beta,
            self.A_log,
            self.dt_bias,
            recurrent_state,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            mask=mask,
        )
        if cache is not None:
            cache[0] = recurrent_state
            cache[1] = conv_q
            cache[2] = conv_k
            cache[3] = conv_v
            cache.advance(L)

        gate = self.g_proj(x).reshape(
            B, L, self.num_attention_heads, self.head_dim
        )
        output = self.o_norm(output, gate)
        return self.o_proj(output.reshape(B, L, -1))


def group_expert_select(
    gates: mx.array,
    e_score_correction_bias: Optional[mx.array],
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    score_function: str,
) -> Tuple[mx.array, mx.array]:
    in_type = gates.dtype
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates.astype(mx.float32))
    else:
        scores = mx.softmax(gates.astype(mx.float32), axis=-1)
    orig_scores = scores
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
    scores = scores * routed_scaling_factor
    return inds, scores.astype(in_type)


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.routed_scaling_factor
        self.score_function = args.score_function

        self.gate_proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = (
            mx.zeros((args.num_experts,))
            if args.moe_router_enable_expert_bias
            else None
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        return group_expert_select(
            self.gate_proj(x),
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.score_function,
        )


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )
        expert_limit = layer_swiglu_limit(args.expert_swiglu_limit_list, layer_idx)
        if expert_limit:
            self.switch_mlp.activation = ClampedSwiGLU(expert_limit)
        self.gate = Gate(args)
        shared_limit = layer_swiglu_limit(
            args.share_expert_swiglu_limit_list, layer_idx
        )
        self.shared_experts = (
            MLP(
                args,
                intermediate_size=args.moe_intermediate_size * args.num_shared_experts,
                swiglu_limit=shared_limit,
            )
            if args.num_shared_experts > 0
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        topk_idx, topk_weight = self.gate(x)
        out = self.switch_mlp(x, topk_idx)
        out = (out * topk_weight[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        n_layers = args.num_hidden_layers
        group = args.layer_group_size
        self.is_global = (
            (layer_idx + 1) % group == 0 or layer_idx >= (n_layers // group) * group
        )

        if self.is_global:
            self.attention = MultiLatentAttention(args)
        else:
            self.attention = LinearAttention(args, layer_idx=layer_idx)

        if args.num_experts is not None and layer_idx >= args.first_k_dense_replace:
            self.mlp = SparseMoeBlock(args, layer_idx=layer_idx)
        else:
            self.mlp = MLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        offset: int = 0,
    ) -> mx.array:
        if self.is_global:
            r = self.attention(self.input_layernorm(x), mask, cache)
        else:
            r = self.attention(self.input_layernorm(x), mask, cache, offset=offset)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LanguageModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Find a representative attention layer index for offset/mask sizing.
        self._attn_idx = next(
            (i for i, l in enumerate(self.layers) if l.is_global), 0
        )
        self._gla_idx = next(
            (i for i, l in enumerate(self.layers) if not l.is_global), 0
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.word_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        attn_mask = create_attention_mask(h, cache[self._attn_idx], return_array=True)
        gla_mask = create_ssm_mask(h, cache[self._gla_idx])
        offset = (
            cache[self._attn_idx].offset if cache[self._attn_idx] is not None else 0
        )
        if hasattr(offset, "dtype"):
            offset = offset + 0

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.is_global else gla_mask
            h = layer(h, mask, c, offset=offset)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        n_layers = self.args.num_hidden_layers

        # Drop MTP and any extra non-base layers (Ling 2.6 has 1 MTP layer
        # appended after num_hidden_layers; deletes weights for those).
        weights = {
            k: v
            for k, v in weights.items()
            if not (
                k.startswith("model.layers.")
                and k.split(".")[2].isdigit()
                and int(k.split(".")[2]) >= n_layers
            )
        }

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(n_layers):
            prefix = f"model.layers.{l}"

            # MoE expert stacking + gate remap
            if l >= self.args.first_k_dense_replace:
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases", "weight_scale_inv"]:
                        if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                            stacked = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                                for e in range(self.args.num_experts)
                            ]
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(
                                stacked
                            )

                for suffix in (
                    "weight",
                    "bias",
                    "scales",
                    "biases",
                    "weight_scale_inv",
                ):
                    source = f"{prefix}.mlp.gate.{suffix}"
                    if source in weights:
                        weights[f"{prefix}.mlp.gate.gate_proj.{suffix}"] = weights.pop(
                            source
                        )

            # MLA kv_b_proj split for global attention layers.
            kv_b_key = f"{prefix}.attention.kv_b_proj.weight"
            if kv_b_key in weights:
                v = weights.pop(kv_b_key)
                head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim
                num_heads = self.args.num_attention_heads
                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(
                    v[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
                )
                wv = mx.contiguous(v[:, self.args.qk_nope_head_dim :, :])
                weights[f"{prefix}.attention.embed_q.weight"] = wk
                weights[f"{prefix}.attention.unembed_out.weight"] = wv

        return self._convert_fp8_block_weights(weights)

    def _convert_fp8_block_weights(self, weights):
        """Convert Ling's published FP8 and MXFP4 tensor layouts for MLX.

        Published FP8 checkpoints use a float32 ``weight_scale_inv`` grid,
        normally with 128x128 blocks. Metal cannot multiply that layout
        directly. Converting one stacked tensor at a time keeps peak memory
        bounded while preserving the checkpoint's compact runtime footprint.

        The mixed FP4 checkpoint stores routed experts as two packed E2M1
        values per int8 byte with one E8M0 scale per 32 logical values. That is
        MLX's native MXFP4 layout, so only reinterpret the packed bytes and
        rename the scale sidecar instead of dequantizing the experts.
        """
        quantization_config = self.args.quantization_config or {}
        block_size = quantization_config.get("weight_block_size", (128, 128))
        if not isinstance(block_size, (list, tuple)) or len(block_size) != 2:
            block_size = (128, 128)
        block_rows, block_cols = (int(block_size[0]), int(block_size[1]))

        scale_keys = [key for key in weights if key.endswith(".weight_scale_inv")]
        for scale_key in scale_keys:
            weight_key = scale_key[: -len("_scale_inv")]
            if weight_key not in weights:
                continue

            source_weight = weights[weight_key]
            is_routed_mxfp4 = (
                quantization_config.get("routed_experts_quant_method") == "mxfp4"
                and ".mlp.switch_mlp." in weight_key
                and source_weight.dtype == mx.int8
            )
            if is_routed_mxfp4:
                scale = weights.pop(scale_key)
                packed = weights.pop(weight_key).view(mx.uint32)
                base = weight_key[: -len("weight")]
                weights[weight_key] = packed
                weights[f"{base}scales"] = scale
                mx.eval(packed, scale)
                continue

            if source_weight.dtype != mx.uint8:
                continue

            scale = weights.pop(scale_key)
            if scale.dtype == mx.uint8:
                scale = mx.power(
                    mx.array(2.0, dtype=mx.float32),
                    scale.astype(mx.float32) - 127.0,
                )
            else:
                scale = scale.astype(mx.float32)
            weight = mx.from_fp8(weights.pop(weight_key), dtype=mx.float32)
            out_dim, in_dim = weight.shape[-2:]
            target_out = scale.shape[-2] * block_rows
            target_in = scale.shape[-1] * block_cols
            if target_out < out_dim or target_in < in_dim:
                raise ValueError(
                    f"Invalid FP8 block scale for {weight_key}: weight "
                    f"{weight.shape}, scale {scale.shape}, block {block_size}"
                )

            pad_out = target_out - out_dim
            pad_in = target_in - in_dim
            if pad_out or pad_in:
                padding = [(0, 0)] * (weight.ndim - 2)
                padding.extend(((0, pad_out), (0, pad_in)))
                weight = mx.pad(weight, padding)

            lead = weight.shape[:-2]
            weight = weight.reshape(
                *lead,
                scale.shape[-2],
                block_rows,
                scale.shape[-1],
                block_cols,
            )
            weight = weight * scale[..., :, None, :, None]
            weight = weight.reshape(*lead, target_out, target_in)
            weight = weight[..., :out_dim, :in_dim].astype(mx.bfloat16)

            quantized, scales, biases = mx.quantize(weight, group_size=64, bits=8)
            weights[weight_key] = quantized
            base = weight_key[: -len("weight")]
            weights[f"{base}scales"] = scales
            weights[f"{base}biases"] = biases
            mx.eval(quantized, scales, biases)
            mx.clear_cache()

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate.gate_proj"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.is_global:
                caches.append(KVCache())
            else:
                caches.append(ArraysCache(size=4))
        return caches
