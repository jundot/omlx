"""MLX-LM adapter for IFM K2-Horizon-MoVA-36B-A4B.

The model is not a standard Llama/Mixtral checkpoint: sparse decoder layers
use both sigmoid-routed feed-forward experts and routed value experts in
attention (MoVA).  This module mirrors the upstream K2 Horizon implementation
without substituting its grouped RMSNorm, routing rules, or attention gate.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache
from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    moe_intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    mova_num_experts: int
    mova_num_experts_per_tok: int
    num_shared_experts: int
    decoder_sparse_step: int
    mlp_only_layers: List[int]
    rms_norm_eps: float
    vocab_size: int
    head_dim: int
    max_position_embeddings: int
    norm_topk_prob: bool
    router_score_func: str
    router_scaling_factor: float
    tie_word_embeddings: bool
    layernorm_num_groups: int = 2
    rope_theta: float = 10_000_000.0
    rope_head_dim: Optional[int] = None
    attention_bias: bool = False
    moe_gate_bias: bool = True
    attention_gate_func: Optional[str] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None


class GroupRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float, groups: int):
        super().__init__()
        if dims % groups:
            raise ValueError(f"hidden size {dims} is not divisible by {groups} groups")
        self.weight = mx.ones((dims,))
        self.groups = groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.unflatten(x, axis=-1, shape=(self.groups, -1))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * mx.flatten(x, -2)


def _router(x: mx.array, gate: nn.Linear, top_k: int, *, normalize: bool, scale: float):
    """K2 router: bias affects top-k selection but not sigmoid scores."""
    # Calling the module (rather than reading ``weight`` directly) also works
    # after it has become a QuantizedLinear.  K2 applies the router bias only
    # when choosing experts, so remove it from the score logits first.
    logits = gate(x)
    if "bias" in gate:
        logits = logits - gate.bias
    scores = mx.sigmoid(logits.astype(mx.float32))
    choice_scores = scores + gate.bias.astype(scores.dtype) if "bias" in gate else scores
    indices = mx.stop_gradient(mx.argpartition(choice_scores, kth=-top_k, axis=-1)[..., -top_k:])
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if normalize:
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    return weights.astype(x.dtype) * scale, indices


def _attention_gate(x: mx.array, projection: Optional[nn.Linear], func: Optional[str], heads: int, head_dim: int):
    if projection is None:
        return None
    gate = projection(x).reshape(*x.shape[:-1], heads, head_dim)
    if func == "silu":
        return nn.silu(gate)
    if func == "softplus":
        beta = math.log(2.0)
        return mx.log1p(mx.exp(gate * beta)) / beta
    raise ValueError(f"unsupported attention gate: {func}")


class DenseAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.attention_bias)
        self.gate_proj = (
            nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
            if args.attention_gate_func is not None else None
        )
        self.gate_func = args.attention_gate_func
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        b, length, _ = x.shape
        q = self.q_proj(x).reshape(b, length, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, length, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, length, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        gate = _attention_gate(x, self.gate_proj, self.gate_func, self.n_heads, self.head_dim)
        if gate is not None:
            out = out * gate.transpose(0, 2, 1, 3)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, length, -1))


class MoVAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        self.top_k = args.mova_num_experts_per_tok
        self.router_scale = args.router_scaling_factor
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.attention_bias)
        self.v_router = nn.Linear(dim, args.mova_num_experts, bias=args.moe_gate_bias)
        self.v_experts = SwitchLinear(dim, self.n_kv_heads * self.head_dim, args.mova_num_experts, bias=False)
        self.gate_proj = (
            nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
            if args.attention_gate_func is not None else None
        )
        self.gate_func = args.attention_gate_func
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        b, length, _ = x.shape
        flat = x.reshape(-1, x.shape[-1])
        weights, indices = _router(flat, self.v_router, self.top_k, normalize=True, scale=self.router_scale)
        routed = self.v_experts(mx.expand_dims(flat, (-2, -3)), indices).squeeze(-2)
        values = (nn.silu(routed) * weights[..., None]).sum(axis=-2)
        v = values.reshape(b, length, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.q_proj(x).reshape(b, length, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, length, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        gate = _attention_gate(x, self.gate_proj, self.gate_func, self.n_heads, self.head_dim)
        if gate is not None:
            out = out * gate.transpose(0, 2, 1, 3)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, length, -1))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.router_scale = args.router_scaling_factor
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=args.moe_gate_bias)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.num_experts, bias=False)
        self.shared_experts = MLP(args.hidden_size, args.moe_intermediate_size * args.num_shared_experts)

    def __call__(self, x: mx.array) -> mx.array:
        weights, indices = _router(x, self.gate, self.top_k, normalize=True, scale=self.router_scale)
        y = self.switch_mlp(x, indices)
        y = (y * weights[..., None]).sum(axis=-2).astype(x.dtype)
        return y + self.shared_experts(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, index: int):
        super().__init__()
        sparse = index not in args.mlp_only_layers and args.num_experts > 0 and (index + 1) % args.decoder_sparse_step == 0
        self.self_attn = MoVAAttention(args) if sparse and args.mova_num_experts > 0 else DenseAttention(args)
        self.mlp = SparseMoE(args) if sparse else MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = GroupRMSNorm(args.hidden_size, args.rms_norm_eps, args.layernorm_num_groups)
        self.post_attention_layernorm = GroupRMSNorm(args.hidden_size, args.rms_norm_eps, args.layernorm_num_groups)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class K2HorizonModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = GroupRMSNorm(args.hidden_size, args.rms_norm_eps, args.layernorm_num_groups)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None) -> mx.array:
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, state in zip(self.layers, cache):
            h = layer(h, mask, state)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = K2HorizonModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None) -> mx.array:
        return self.lm_head(self.model(inputs, cache, input_embeddings))

    def sanitize(self, weights):
        weights.pop("model.rotary_emb.inv_freq", None)
        for layer in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            if f"{prefix}.mlp.experts.0.up_proj.weight" in weights:
                for name in ("up_proj", "down_proj", "gate_proj"):
                    expert_weights = [weights.pop(f"{prefix}.mlp.experts.{expert}.{name}.weight") for expert in range(self.args.num_experts)]
                    weights[f"{prefix}.mlp.switch_mlp.{name}.weight"] = mx.stack(expert_weights)
            if f"{prefix}.self_attn.v_experts.0.weight" in weights:
                expert_weights = [weights.pop(f"{prefix}.self_attn.v_experts.{expert}.weight") for expert in range(self.args.mova_num_experts)]
                weights[f"{prefix}.self_attn.v_experts.weight"] = mx.stack(expert_weights)
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # Routing weights are numerically sensitive and remain at 8-bit.
            if path.endswith("mlp.gate") or path.endswith("self_attn.v_router"):
                return {"group_size": 64, "bits": 8}
            return True
        return predicate

    @property
    def layers(self):
        return self.model.layers
