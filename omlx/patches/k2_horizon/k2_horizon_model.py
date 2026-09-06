# SPDX-License-Identifier: Apache-2.0
"""Implement dense, MoE, and MoVA K2 Horizon language models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear

# MoVA source numerics: docs.sglang.io/cookbook/autoregressive/IFM/K2-Horizon#2-configuration-tips
SOURCE_ROUTER_GEMM_PARTITIONS = 2
_LN2 = math.log(2.0)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    layernorm_num_groups: int
    mlp_only_layers: list[int]
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    num_shared_experts: int
    moe_gate_bias: bool
    norm_topk_prob: bool
    router_score_func: str
    router_scaling_factor: float | None
    query_key_norm: bool
    rope_parameters: dict[str, Any]
    mova_num_experts: int = 0
    mova_num_experts_per_tok: int = 0
    attention_gate_func: str | None = None
    rope_theta: float | None = None
    hidden_act: str = "silu"
    decoder_sparse_step: int = 1
    attention_bias: bool = False
    rope_head_dim: int | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = None
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 524288

    def __post_init__(self):
        self.rope_parameters = dict(self.rope_parameters)
        self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
        self.rope_head_dim = self.rope_head_dim or self.head_dim
        if self.router_scaling_factor is None:
            self.router_scaling_factor = 1.0
        if (
            self.hidden_act != "silu"
            or self.query_key_norm
            or self.use_sliding_window
            or self.sliding_window is not None
            or self.attention_gate_func not in (None, "softplus")
            or self.rope_parameters.get("rope_type", "default")
            not in ("default", "yarn")
        ):
            raise ValueError(
                "Unsupported K2 activation, attention or RoPE configuration"
            )
        if self.num_experts and (
            self.router_score_func != "sigmoid"
            or not self.norm_topk_prob
            or not self.moe_gate_bias
            or self.num_shared_experts != 1
        ):
            raise ValueError("Unsupported K2 expert routing configuration")

    def is_sparse_layer(self, layer_idx: int) -> bool:
        return (
            self.num_experts > 0
            and layer_idx not in self.mlp_only_layers
            and (layer_idx + 1) % self.decoder_sparse_step == 0
        )


@lru_cache(None)
def _yarn_rotation(dims):
    def rotate(x, cos, sin):
        if cos.ndim == 3:
            cos, sin = cos[:, None], sin[:, None]
        first, second = x[..., : dims // 2], x[..., dims // 2 : dims]
        rotated = mx.concatenate(
            [first * cos - second * sin, second * cos + first * sin], axis=-1
        )
        return mx.concatenate([rotated, x[..., dims:]], axis=-1)

    return mx.compile(rotate)


class YarnRoPE(nn.Module):
    """Apply the released YaRN frequencies and BF16 cos/sin rounding."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dims = dims = args.rope_head_dim
        params = args.rope_parameters
        self.attention_factor = params["attention_factor"]
        original = params["original_max_position_embeddings"]

        def correction(rotations):
            return (
                dims
                * math.log(original / (rotations * 2 * math.pi))
                / (2 * math.log(args.rope_theta))
            )

        low, high = correction(params["beta_fast"]), correction(params["beta_slow"])
        if params.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dims - 1)
        if low == high:
            high += 0.001
        ramp = mx.clip(
            (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low), 0, 1
        )
        freq = args.rope_theta ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        self._inv_freq = ramp / (params["factor"] * freq) + (1 - ramp) / freq

    def __call__(self, x, offset=0):
        positions = (
            mx.arange(x.shape[-2], dtype=mx.float32) + mx.array(offset)[..., None]
        )
        angles = positions[..., None] * self._inv_freq
        cos = (mx.cos(angles) * self.attention_factor).astype(x.dtype)
        sin = (mx.sin(angles) * self.attention_factor).astype(x.dtype)
        # Keep trigonometry outside compilation to preserve BF16 rounding.
        return _yarn_rotation(self.dims)(x, cos, sin)


@lru_cache(None)
def _grouped_norm(groups, eps):
    def normalize(x, weight):
        grouped = mx.unflatten(x.astype(mx.float32), -1, (groups, -1))
        normed = mx.flatten(mx.fast.rms_norm(grouped, None, eps), start_axis=-2)
        return (weight * normed).astype(x.dtype)

    return mx.compile(normalize, shapeless=True)


class GroupedRMSNorm(nn.Module):
    """RMSNorm whose statistics are computed per contiguous feature group."""

    def __init__(self, dims: int, groups: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.groups = groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _grouped_norm(self.groups, self.eps)(x, self.weight)


def router_logits(
    x: mx.array, weight: mx.array, partitions: int = SOURCE_ROUTER_GEMM_PARTITIONS
) -> mx.array:
    """Respect each sparse release's router rounding contract."""
    if x.dtype != mx.bfloat16 or weight.dtype != mx.bfloat16:
        raise ValueError(
            "K2 Horizon routers require BF16 activations and weights; got "
            f"{x.dtype} and {weight.dtype}"
        )
    if partitions == 1:
        return (x @ weight.T).astype(mx.float32)
    if partitions != SOURCE_ROUTER_GEMM_PARTITIONS:
        raise ValueError("Unsupported K2 router partition count")
    x_parts = mx.split(x, partitions, axis=-1)
    w_parts = mx.split(weight, partitions, axis=-1)
    logits = (x_parts[0] @ w_parts[0].T).astype(mx.float32)
    for x_part, w_part in zip(x_parts[1:], w_parts[1:]):
        logits = logits + (x_part @ w_part.T).astype(mx.float32)
    return logits


@lru_cache(None)
def _expert_selection(top_k, scaling_factor):
    def select(logits, bias):
        scores = mx.sigmoid(logits)
        selection = scores + bias.astype(mx.float32)
        inds = mx.argpartition(-selection, kth=top_k - 1, axis=-1)[..., :top_k]
        weights = mx.take_along_axis(scores, inds, axis=-1)
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        return inds, weights * scaling_factor

    return mx.compile(select)


def route(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    top_k: int,
    scaling_factor: float,
    *,
    partitions: int = SOURCE_ROUTER_GEMM_PARTITIONS,
) -> tuple[mx.array, mx.array]:
    """Return selected expert indices and their normalized, scaled FP32 weights."""
    return _expert_selection(top_k, scaling_factor)(
        router_logits(x, weight, partitions), bias
    )


def softplus_beta_ln2(x: mx.array) -> mx.array:
    """PyTorch ``softplus(x, beta=ln 2)`` computed in FP32 without overflow."""
    x32 = x.astype(mx.float32)
    return (mx.logaddexp(x32 * _LN2, 0.0) / _LN2).astype(x.dtype)


def _project(layer, x, lora_mask):
    conditional = getattr(layer, "conditional_forward", None)
    return conditional(x, lora_mask) if conditional is not None else layer(x)


class PartialRoPE(nn.Module):
    """Rotate leading pairs in the checkpoint's split-half head layout."""

    def __init__(self, args):
        super().__init__()
        self.half = args.head_dim // 2
        self.rotated_half = args.rope_head_dim // 2
        self.rope = nn.RoPE(args.rope_head_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x, offset=0):
        half, rotated = self.half, self.rotated_half
        paired = mx.concatenate(
            [x[..., :rotated], x[..., half : half + rotated]], axis=-1
        )
        result = self.rope(paired, offset=offset)
        return mx.concatenate(
            [
                result[..., :rotated],
                x[..., rotated:half],
                result[..., rotated:],
                x[..., half + rotated :],
            ],
            axis=-1,
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, mova: bool):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.mova = mova
        self.top_k = args.mova_num_experts_per_tok
        self.scaling_factor = args.router_scaling_factor

        q_dims = self.n_heads * self.head_dim
        kv_dims = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(args.hidden_size, q_dims, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, kv_dims, bias=args.attention_bias)
        self.o_proj = nn.Linear(q_dims, args.hidden_size, bias=args.attention_bias)
        if args.attention_gate_func is not None:
            self.gate_proj = nn.Linear(args.hidden_size, q_dims, bias=False)
        if mova:
            self.v_router = nn.Linear(
                args.hidden_size, args.mova_num_experts, bias=False
            )
            self.v_expert_bias = mx.zeros((args.mova_num_experts,))
            self.v_experts = SwitchLinear(
                args.hidden_size, kv_dims, args.mova_num_experts, bias=False
            )
        else:
            self.v_proj = nn.Linear(args.hidden_size, kv_dims, bias=args.attention_bias)
        self.rope = (
            YarnRoPE(args)
            if args.rope_parameters.get("rope_type") == "yarn"
            else (
                PartialRoPE(args)
                if args.rope_head_dim < args.head_dim
                else nn.RoPE(args.head_dim, traditional=False, base=args.rope_theta)
            )
        )

    def _values(self, x: mx.array, lora_mask=None) -> mx.array:
        if not self.mova:
            return _project(self.v_proj, x, lora_mask)
        inds, weights = route(
            x, self.v_router.weight, self.v_expert_bias, self.top_k, self.scaling_factor
        )
        routed = self.v_experts(mx.expand_dims(x, (-2, -3)), inds).squeeze(-2)
        routed = nn.silu(routed) * weights.astype(routed.dtype)[..., None]
        return routed.sum(axis=-2)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any = None,
        lora_mask=None,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries = (
            _project(self.q_proj, x, lora_mask)
            .reshape(batch, length, self.n_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            _project(self.k_proj, x, lora_mask)
            .reshape(batch, length, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self._values(x, lora_mask)
            .reshape(batch, length, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3)
        if "gate_proj" in self:
            gate = softplus_beta_ln2(self.gate_proj(x)).reshape(
                batch, length, self.n_heads, -1
            )
            output = output * gate
        return _project(self.o_proj, output.reshape(batch, length, -1), lora_mask)


class MLP(nn.Module):
    def __init__(self, dims: int, hidden_dims: int):
        super().__init__()
        self.gate_proj = nn.Linear(dims, hidden_dims, bias=False)
        self.up_proj = nn.Linear(dims, hidden_dims, bias=False)
        self.down_proj = nn.Linear(hidden_dims, dims, bias=False)

    def __call__(self, x: mx.array, lora_mask=None) -> mx.array:
        h = swiglu(
            _project(self.gate_proj, x, lora_mask), _project(self.up_proj, x, lora_mask)
        )
        return _project(self.down_proj, h, lora_mask)


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.scaling_factor = args.router_scaling_factor
        self.router_partitions = (
            SOURCE_ROUTER_GEMM_PARTITIONS if args.mova_num_experts else 1
        )
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,))
        self.experts = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_experts = MLP(
            args.hidden_size, args.moe_intermediate_size * args.num_shared_experts
        )

    def __call__(self, x: mx.array, lora_mask=None) -> mx.array:
        if lora_mask is not None:
            raise ValueError("Uno adapters require a dense K2 base")
        inds, weights = route(
            x,
            self.gate.weight,
            self.expert_bias,
            self.top_k,
            self.scaling_factor,
            partitions=self.router_partitions,
        )
        routed = self.experts(x, inds) * weights.astype(x.dtype)[..., None]
        return routed.sum(axis=-2) + self.shared_experts(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        sparse = args.is_sparse_layer(layer_idx)
        self.self_attn = Attention(args, mova=sparse and args.mova_num_experts > 0)
        if sparse:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = GroupedRMSNorm(
            args.hidden_size, args.layernorm_num_groups, args.rms_norm_eps
        )
        self.post_attention_layernorm = GroupedRMSNorm(
            args.hidden_size, args.layernorm_num_groups, args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any = None,
        lora_mask=None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache, lora_mask)
        return h + self.mlp(self.post_attention_layernorm(h), lora_mask=lora_mask)


class K2HorizonModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = GroupedRMSNorm(
            args.hidden_size, args.layernorm_num_groups, args.rms_norm_eps
        )

    def __call__(self, inputs: mx.array, cache: Any = None, lora_mask=None) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, lora_mask)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = K2HorizonModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Any = None, lora_mask=None) -> mx.array:
        out = self.model(inputs, cache, lora_mask)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        for layer_idx in range(self.args.num_hidden_layers):
            if not self.args.is_sparse_layer(layer_idx):
                continue
            prefix = f"model.layers.{layer_idx}"
            for name in ("gate_proj", "up_proj", "down_proj"):
                _stack_experts(
                    weights,
                    f"{prefix}.mlp.experts.{name}.weight",
                    [
                        f"{prefix}.mlp.experts.{e}.{name}.weight"
                        for e in range(self.args.num_experts)
                    ],
                )
            if self.args.mova_num_experts:
                _stack_experts(
                    weights,
                    f"{prefix}.self_attn.v_experts.weight",
                    [
                        f"{prefix}.self_attn.v_experts.{e}.weight"
                        for e in range(self.args.mova_num_experts)
                    ],
                )
            _rename(weights, f"{prefix}.mlp.gate.bias", f"{prefix}.mlp.expert_bias")
            _rename(
                weights,
                f"{prefix}.self_attn.v_router.bias",
                f"{prefix}.self_attn.v_expert_bias",
            )
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            return not (
                path.endswith("mlp.gate") or path.endswith("self_attn.v_router")
            )

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate


def _stack_experts(
    weights: dict[str, mx.array], stacked_key: str, expert_keys: list[str]
) -> None:
    present = [k for k in expert_keys if k in weights]
    if stacked_key in weights:
        if present:
            raise ValueError(
                f"{stacked_key} is present alongside per-expert tensors such as {present[0]}"
            )
        return
    missing = [k for k in expert_keys if k not in weights]
    if missing:
        raise ValueError(
            f"Cannot stack {stacked_key}: {len(missing)} expert tensors missing, "
            f"first {missing[0]}"
        )
    weights[stacked_key] = mx.stack([weights.pop(k) for k in expert_keys])


def _rename(weights: dict[str, mx.array], old: str, new: str) -> None:
    if old not in weights:
        return
    if new in weights:
        raise ValueError(f"Both {old} and {new} are present")
    weights[new] = weights.pop(old)
