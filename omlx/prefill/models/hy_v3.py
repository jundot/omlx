"""HyV3 plain-KV execution and routed/shared expert geometry (Hy-MT2)."""

from __future__ import annotations

from typing import Any

from ..geometry import (
    PrefillExpertGeometry,
    PrefillModelGeometry,
    get_argument,
    make_geometry,
    nonnegative_int,
    positive_int,
)
from .common import execution_fallback, local_weight_shape


def geometry_from_args(arguments: Any, **options) -> PrefillModelGeometry | None:
    shared = get_argument(arguments, "num_shared_experts")
    expert_width = get_argument(arguments, "expert_hidden_dim")
    dense_layers = get_argument(arguments, "first_k_dense_replace")
    num_layers = get_argument(arguments, "num_hidden_layers")
    if (
        not nonnegative_int(shared)
        or not positive_int(expert_width)
        or not positive_int(num_layers)
        or not nonnegative_int(dense_layers)
        or dense_layers >= num_layers
    ):
        return None
    experts = PrefillExpertGeometry(
        num_experts=get_argument(arguments, "num_experts"),
        experts_per_token=get_argument(arguments, "num_experts_per_tok"),
        intermediate_size=expert_width,
        shared_intermediate_size=shared * expert_width,
    )
    return make_geometry(
        arguments,
        head_dim=get_argument(arguments, "head_dim"),
        kv_heads=get_argument(arguments, "num_key_value_heads"),
        experts=experts,
        **options,
    )


def _mlp_matches(mlp: Any, hidden_size: int, intermediate_size: int) -> bool:
    return all(
        local_weight_shape(getattr(mlp, name, None)) == shape
        for name, shape in (
            ("gate_proj", (intermediate_size, hidden_size)),
            ("up_proj", (intermediate_size, hidden_size)),
            ("down_proj", (hidden_size, intermediate_size)),
        )
    )


def validate(model: Any, module: Any, geometry: PrefillModelGeometry) -> str | None:
    """Verify the plain-KV HyV3 contract, including the normal fused MoE path."""
    import mlx.nn as nn
    from mlx_lm.models import rope_utils
    from mlx_lm.models.switch_layers import SwitchGLU

    args, backbone, experts = model.args, model.model, geometry.experts
    if experts is None:
        return "unsupported_geometry"
    if type(backbone) is not module.HYV3Model or type(backbone.norm) is not nn.RMSNorm:
        return "custom_execution"
    if len(model.layers) != geometry.num_layers:
        return "unsupported_geometry"
    embedding_shape = (geometry.vocab_size, geometry.hidden_size)
    if local_weight_shape(backbone.embed_tokens, embedding=True) != embedding_shape:
        return "unsupported_geometry"
    if (
        not args.tie_word_embeddings
        and local_weight_shape(model.lm_head) != embedding_shape
    ):
        return "unsupported_geometry"
    rope_types = (
        nn.RoPE,
        rope_utils.Llama3RoPE,
        rope_utils.YarnRoPE,
        rope_utils.SuScaledRoPE,
        rope_utils.ProportionalRoPE,
    )
    for index, layer in enumerate(model.layers):
        if (
            type(layer) is not module.DecoderLayer
            or type(layer.input_layernorm) is not nn.RMSNorm
            or type(layer.post_attention_layernorm) is not nn.RMSNorm
            or type(layer.self_attn) is not module.Attention
        ):
            return "custom_execution"
        attention, mlp = layer.self_attn, layer.mlp
        reason = execution_fallback((layer, attention, mlp))
        if reason:
            return reason
        if (
            attention.n_heads != geometry.num_attention_heads
            or attention.n_kv_heads != geometry.num_kv_heads
            or attention.head_dim != geometry.head_dim
        ):
            return "unsupported_geometry"
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
        )
        if any(
            local_weight_shape(projection) != (n, k) for projection, n, k in projections
        ):
            return "unsupported_geometry"
        if index < args.first_k_dense_replace:
            if type(mlp) is not module.MLP or not _mlp_matches(
                mlp, geometry.hidden_size, geometry.intermediate_size
            ):
                return "unsupported_geometry"
            continue
        if (
            type(mlp) is not module.MoE
            or type(mlp.router) is not module.MoEGate
            or type(mlp.switch_mlp) is not SwitchGLU
        ):
            return "custom_execution"
        if (
            mlp.num_experts_per_tok != experts.experts_per_token
            or mlp.router.top_k != experts.experts_per_token
            or local_weight_shape(mlp.router.gate)
            != (experts.num_experts, geometry.hidden_size)
            or mlp.router.expert_bias.shape != (experts.num_experts,)
        ):
            return "unsupported_geometry"
        expert_projections = (
            ("gate_proj", (experts.intermediate_size, geometry.hidden_size)),
            ("up_proj", (experts.intermediate_size, geometry.hidden_size)),
            ("down_proj", (geometry.hidden_size, experts.intermediate_size)),
        )
        if hasattr(mlp.switch_mlp, "gate_up_proj"):
            if not getattr(SwitchGLU, "_omlx_gate_up_fused_call", False) or any(
                hasattr(mlp.switch_mlp, name) for name in ("gate_proj", "up_proj")
            ):
                return "custom_execution"
            expert_projections = (
                ("gate_up_proj", (2 * experts.intermediate_size, geometry.hidden_size)),
                ("down_proj", (geometry.hidden_size, experts.intermediate_size)),
            )
        if any(
            local_weight_shape(
                getattr(mlp.switch_mlp, name), experts=experts.num_experts
            )
            != shape
            for name, shape in expert_projections
        ):
            return "unsupported_geometry"
        if experts.shared_intermediate_size:
            if type(mlp.shared_mlp) is not module.MLP or not _mlp_matches(
                mlp.shared_mlp, geometry.hidden_size, experts.shared_intermediate_size
            ):
                return "unsupported_geometry"
        elif mlp.shared_mlp is not None:
            return "unsupported_geometry"
    return None
