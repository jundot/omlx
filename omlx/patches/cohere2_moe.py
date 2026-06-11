# SPDX-License-Identifier: Apache-2.0
"""Runtime compatibility patch for Cohere2 MoE text models in mlx-lm.

The pinned mlx-lm version used by oMLX does not include ``cohere2_moe`` yet.
Register a local model module under ``mlx_lm.models.cohere2_moe`` so standard
mlx-lm loading can handle North Mini Code style checkpoints without changing
the engine path.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU

logger = logging.getLogger(__name__)

_MLX_LM_MODULE_NAME = "mlx_lm.models.cohere2_moe"
_APPLIED = False


def _upstream_module_available() -> bool:
    try:
        importlib.import_module(_MLX_LM_MODULE_NAME)
    except ImportError:
        return False
    return True


def apply_cohere2_moe_patch() -> bool:
    """Register ``cohere2_moe`` with the live mlx-lm import machinery."""
    global _APPLIED
    if _APPLIED:
        return False

    if not _upstream_module_available():
        sys.modules[_MLX_LM_MODULE_NAME] = sys.modules[__name__]
        try:
            import mlx_lm.models as models_pkg

            models_pkg.cohere2_moe = sys.modules[__name__]
        except Exception as e:
            logger.debug("Could not attach Cohere2 MoE module to mlx_lm.models: %s", e)

    try:
        import mlx_lm.utils as mlx_lm_utils

        mlx_lm_utils.MODEL_REMAPPING.setdefault("cohere2_vision", "cohere2_moe")
    except Exception as e:
        logger.debug("Could not update mlx-lm Cohere2 remapping: %s", e)

    _APPLIED = True
    return True


def is_applied() -> bool:
    return _APPLIED


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "cohere2_moe"
    hidden_size: int = 4096
    head_dim: int = 128
    num_hidden_layers: int = 32
    intermediate_size: int = 4096
    num_attention_heads: int = 128
    num_key_value_heads: int = 8
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 4
    shared_expert_combination_strategy: str = "average"
    expert_selection_fn: str = "sigmoid"
    norm_topk_prob: bool = True
    rope_theta: float = 50000.0
    vocab_size: int = 262144
    layer_norm_eps: float = 1e-5
    logit_scale: float = 1.0
    attention_bias: bool = False
    sliding_window: int = 4096
    sliding_window_pattern: int = 4
    use_parallel_block: bool = True
    use_qk_norm: bool = False
    use_embedding_sharing: bool = True
    first_k_dense_replace: int = 0
    prefix_dense_intermediate_size: int = 16384
    prefix_dense_sliding_window_pattern: int = 1
    layer_types: list[str] | None = None

    def __post_init__(self):
        if self.layer_types is None:
            pattern = ["sliding_attention"] * (self.sliding_window_pattern - 1) + [
                "full_attention"
            ]
            self.layer_types = (pattern * (self.num_hidden_layers // len(pattern) + 1))[
                : self.num_hidden_layers
            ]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        if head_dim * n_heads != dim:
            raise ValueError(
                f"hidden_size must equal num_attention_heads * head_dim "
                f"(got hidden_size={dim}, heads={n_heads}, head_dim={head_dim})."
            )
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.rope = nn.RoPE(head_dim, traditional=True, base=args.rope_theta)
        self.use_sliding_window = args.layer_types[layer_idx] == "sliding_attention"

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(batch, length, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(batch, length, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        if self.use_sliding_window:
            if cache is None:
                queries = self.rope(queries)
                keys = self.rope(keys)
            else:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        sdpa_type = mx.float32 if queries.dtype == mx.float16 else queries.dtype
        output = scaled_dot_product_attention(
            queries.astype(sdpa_type),
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        ).astype(queries.dtype)
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """Dense MLP used for shared experts and prefix dense layers."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Cohere2MoeSparseMoeBlock(nn.Module):
    """Sparse MoE block with sigmoid routing and optional shared experts."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.intermediate_size

        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.expert_selection_fn = args.expert_selection_fn
        self.gate = nn.Linear(dim, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            dim,
            intermediate_size,
            args.num_experts,
            bias=False,
        )

        self.num_shared_experts = args.num_shared_experts
        self.shared_expert_combination_strategy = (
            args.shared_expert_combination_strategy
        )
        if self.num_shared_experts > 0:
            shared_intermediate = intermediate_size * self.num_shared_experts
            self.shared_experts = MLP(dim, shared_intermediate)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        if self.expert_selection_fn == "sigmoid":
            gates = mx.sigmoid(gates)
        else:
            gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        if self.num_shared_experts > 0:
            shared_out = self.shared_experts(x)
            if self.shared_expert_combination_strategy == "average":
                y = (y + shared_out) / 2
            else:
                y = y + shared_out

        return y


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Attention(args, layer_idx)

        if layer_idx >= args.first_k_dense_replace:
            self.mlp = Cohere2MoeSparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.prefix_dense_intermediate_size)

        self.input_layernorm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layer_norm_eps,
            bias=False,
        )
        self.use_parallel_block = args.use_parallel_block

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        h = self.input_layernorm(x)
        attn_h = self.self_attn(h, mask, cache)

        if self.use_parallel_block:
            ff_h = self.mlp(h)
            return attn_h + ff_h + x

        h = attn_h + x
        ff_h = self.mlp(self.input_layernorm(h))
        return ff_h + h


class Cohere2MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.window_size = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layer_norm_eps,
            bias=False,
        )

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        full_cache = None
        swa_cache = None
        for i, c in enumerate(cache):
            layer_type = self.args.layer_types[i]
            if layer_type == "full_attention" and full_cache is None:
                full_cache = c
            elif layer_type == "sliding_attention" and swa_cache is None:
                swa_cache = c

        full_mask = create_attention_mask(h, full_cache)
        swa_mask = create_attention_mask(h, swa_cache, window_size=self.window_size)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_full = self.args.layer_types[i] == "full_attention"
            h = layer(h, full_mask if is_full else swa_mask, c)

        return self.norm(h)


def _clean_weight_key(key: str) -> str:
    replacements = (
        ("model.language_model.model.", "model."),
        ("language_model.model.", "model."),
        ("model.language_model.", ""),
        ("language_model.", ""),
    )
    for prefix, replacement in replacements:
        if key.startswith(prefix):
            return replacement + key[len(prefix) :]
    return key


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = Cohere2MoeModel(args)
        self.args = args
        if not args.use_embedding_sharing:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        if self.args.use_embedding_sharing:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out * self.args.logit_scale

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if any(
                marker in key
                for marker in (
                    "vision_tower",
                    "multi_modal_projector",
                    "image_newline",
                    "rotary_emb.inv_freq",
                )
            ):
                continue
            sanitized[_clean_weight_key(key)] = value

        weights = sanitized
        if self.args.use_embedding_sharing:
            weights.pop("lm_head.weight", None)

        has_stacked_experts = any(
            ".mlp.switch_mlp.up_proj.weight" in key
            or ".mlp.switch_mlp.up_proj.weight_packed" in key
            for key in weights
        )
        if not has_stacked_experts:
            self._stack_raw_expert_weights(weights)

        return weights

    def _stack_raw_expert_weights(self, weights):
        for layer_idx in range(
            self.args.first_k_dense_replace,
            self.args.num_hidden_layers,
        ):
            prefix = f"model.layers.{layer_idx}.mlp"
            for suffix in ("weight", "weight_packed"):
                expert_key = f"{prefix}.experts.0.gate_proj.{suffix}"
                if expert_key not in weights:
                    continue
                for name in ("up_proj", "down_proj", "gate_proj"):
                    to_join = [
                        weights.pop(f"{prefix}.experts.{e}.{name}.{suffix}")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.switch_mlp.{name}.{suffix}"] = mx.stack(to_join)
                    for extra in (
                        "weight_scale",
                        "weight_global_scale",
                        "input_global_scale",
                        "bias",
                    ):
                        extra_key = f"{prefix}.experts.0.{name}.{extra}"
                        if extra_key not in weights:
                            continue
                        to_join_extra = [
                            weights.pop(f"{prefix}.experts.{e}.{name}.{extra}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.switch_mlp.{name}.{extra}"] = mx.stack(
                            to_join_extra
                        )

    @property
    def quant_predicate(self):
        """Keep router gates and attention at higher precision when quantizing."""

        def predicate(path, _):
            if "mlp.gate" in path:
                return {"group_size": 64, "bits": 8}
            if "self_attn" in path:
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        caches = []
        for i in range(self.args.num_hidden_layers):
            if self.args.layer_types[i] == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
        return caches

    @property
    def layers(self):
        return self.model.layers


__all__ = ["Model", "ModelArgs", "apply_cohere2_moe_patch", "is_applied"]
