"""Plain-KV Llama, Qwen2 and Qwen3 contracts."""

from __future__ import annotations

from typing import Any

from ..geometry import PrefillModelGeometry, get_argument, make_geometry, positive_int
from .common import local_weight_shape


def geometry_from_args(arguments: Any, **options) -> PrefillModelGeometry | None:
    model_type = get_argument(arguments, "model_type")
    attention_heads = get_argument(arguments, "num_attention_heads")
    hidden_size = get_argument(arguments, "hidden_size")
    if not positive_int(attention_heads) or not positive_int(hidden_size):
        return None

    kv_heads = get_argument(arguments, "num_key_value_heads")
    if kv_heads is None and model_type == "llama":
        kv_heads = attention_heads

    head_dim = get_argument(arguments, "head_dim")
    if (
        model_type == "qwen2"
        and head_dim is not None
        and (
            hidden_size % attention_heads or head_dim != hidden_size // attention_heads
        )
    ):
        return None
    if head_dim is None:
        if model_type in ("qwen3",) or hidden_size % attention_heads != 0:
            return None
        head_dim = hidden_size // attention_heads

    return make_geometry(arguments, head_dim=head_dim, kv_heads=kv_heads, **options)


def validate(model: Any, module: Any, geometry: PrefillModelGeometry) -> str | None:
    """Verify live local dimensions; shard() preserves the outer model type."""
    import mlx.nn as nn
    from mlx_lm.models import rope_utils

    backbone = getattr(model, "model", None)
    backbone_type = {
        "llama": "LlamaModel",
        "qwen2": "Qwen2Model",
        "qwen3": "Qwen3Model",
    }[module.__name__.rsplit(".", 1)[-1]]
    if type(backbone) is not getattr(module, backbone_type):
        return "custom_execution"
    if len(model.layers) != geometry.num_layers:
        return "unsupported_geometry"
    if local_weight_shape(backbone.embed_tokens, embedding=True) != (
        geometry.vocab_size,
        geometry.hidden_size,
    ):
        return "unsupported_geometry"
    if not model.args.tie_word_embeddings and local_weight_shape(model.lm_head) != (
        geometry.vocab_size,
        geometry.hidden_size,
    ):
        return "unsupported_geometry"
    if type(backbone.norm) is not nn.RMSNorm:
        return "custom_execution"

    rope_types = (
        nn.RoPE,
        rope_utils.Llama3RoPE,
        rope_utils.YarnRoPE,
        rope_utils.SuScaledRoPE,
        rope_utils.ProportionalRoPE,
    )
    for layer in model.layers:
        if (
            type(layer) is not module.TransformerBlock
            or type(layer.self_attn) is not module.Attention
            or type(layer.mlp) is not module.MLP
            or type(layer.input_layernorm) is not nn.RMSNorm
            or type(layer.post_attention_layernorm) is not nn.RMSNorm
        ):
            return "custom_execution"
        attention = layer.self_attn
        if (
            attention.n_heads != geometry.num_attention_heads
            or attention.n_kv_heads != geometry.num_kv_heads
        ):
            return "unsupported_geometry"
        if getattr(layer, "use_sliding", False):
            return "unsupported_attention"
        if type(attention.rope) not in rope_types:
            return "unsupported_position_state"
        if any(
            type(getattr(attention, name)) is not nn.RMSNorm
            for name in ("q_norm", "k_norm")
            if hasattr(attention, name)
        ):
            return "custom_execution"
        projections = (
            (
                attention.q_proj,
                geometry.num_attention_heads * geometry.head_dim,
                geometry.hidden_size,
            ),
            (
                attention.k_proj,
                geometry.num_kv_heads * geometry.head_dim,
                geometry.hidden_size,
            ),
            (
                attention.v_proj,
                geometry.num_kv_heads * geometry.head_dim,
                geometry.hidden_size,
            ),
            (
                attention.o_proj,
                geometry.hidden_size,
                geometry.num_attention_heads * geometry.head_dim,
            ),
            (layer.mlp.gate_proj, geometry.intermediate_size, geometry.hidden_size),
            (layer.mlp.up_proj, geometry.intermediate_size, geometry.hidden_size),
            (layer.mlp.down_proj, geometry.hidden_size, geometry.intermediate_size),
        )
        if any(
            local_weight_shape(projection) != (output_size, input_size)
            for projection, output_size, input_size in projections
        ):
            return "unsupported_geometry"
    return None
