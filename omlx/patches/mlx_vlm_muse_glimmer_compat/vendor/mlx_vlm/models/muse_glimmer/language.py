from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache

# oMLX: the pinned mlx-vlm's rope_utils has no initialize_rope (only MRoPE
# machinery); source it from mlx-lm, whose signature matches upstream usage.
from mlx_lm.models.rope_utils import initialize_rope

from .config import TextConfig


class RMSNormNoScale(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class CenteredRMSNorm(nn.Module):
    """RMSNorm whose checkpoint scale is centered at zero (effective scale 1+w)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


# oMLX: QuantizedNormedEmbedding + NormedEmbedding.to_quantized are ported
# from mlx-vlm PR #1839. nn.Embedding.to_quantized returns a plain
# QuantizedEmbedding, which silently drops the weightless embed_norm and
# feeds unnormalized embeddings into the residual stream (logits collapse
# onto generic tokens while the model keeps generating fluent text).
class QuantizedNormedEmbedding(nn.QuantizedEmbedding):
    """Quantized NormedEmbedding that keeps the weightless embedding norm."""

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.embed_norm(super().__call__(inputs))


class NormedEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, hidden_size: int, eps: float):
        super().__init__(vocab_size, hidden_size)
        self.embed_norm = RMSNormNoScale(eps)

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.embed_norm(super().__call__(inputs))

    def to_quantized(
        self,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        **kwargs,
    ) -> "QuantizedNormedEmbedding":
        quantized = QuantizedNormedEmbedding.from_embedding(
            self, group_size, bits, mode=mode
        )
        quantized.embed_norm = self.embed_norm
        return quantized


class MLP(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Attention(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.qk_scale_factor = args.qk_scale_factor
        self.use_rope = bool(args.layer_rope_theta[layer_idx])
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

        dim = args.hidden_size
        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )
        self.qk_norm = RMSNormNoScale(args.rms_norm_eps)

        theta = (
            float(args.layer_rope_theta[layer_idx])
            if self.use_rope
            else float(args.rope_parameters.get("rope_theta", 500000.0))
        )
        self.rope = initialize_rope(
            self.head_dim,
            base=theta,
            traditional=False,
            scaling_config={"rope_type": "default", "rope_theta": theta},
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        keys = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)

        queries = (self.qk_norm(queries) * self.qk_scale_factor).transpose(0, 2, 1, 3)
        keys = self.qk_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        output = output * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = CenteredRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps
        )
        self.pre_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.rms_norm_eps
        )
        self.post_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps
        )
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        x = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        x = residual + self.post_attention_layernorm(x)

        residual = x
        x = self.mlp(self.pre_feedforward_layernorm(x))
        return residual + self.post_feedforward_layernorm(x)


class TextModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = NormedEmbedding(
            args.vocab_size, args.hidden_size, args.rms_norm_eps
        )
        self.layers = [DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window

        self.full_attention_idx = self.layer_types.index("full_attention")
        self.sliding_attention_idx = (
            self.layer_types.index("sliding_attention")
            if "sliding_attention" in self.layer_types
            else None
        )

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = (
            self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        )
        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(hidden_states, cache[self.full_attention_idx])
        sliding_mask = None
        if self.sliding_attention_idx is not None:
            sliding_mask = create_attention_mask(
                hidden_states,
                cache[self.sliding_attention_idx],
                window_size=self.sliding_window,
            )

        for layer, layer_cache in zip(self.layers, cache):
            mask = sliding_mask if layer.is_sliding else full_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
        return self.norm(hidden_states)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = TextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.final_logit_softcapping = args.final_logit_softcapping
        self.output_multiplier = args.output_multiplier

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        hidden_states = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        logits = self.lm_head(hidden_states) * self.output_multiplier
        softcap = self.final_logit_softcapping
        logits = mx.tanh(logits / softcap) * softcap
        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    @property
    def quant_predicate(self):
        def predicate(_, module):
            # NormedEmbedding has a custom forward pass which applies an RMS
            # normalization. Replacing it with QuantizedEmbedding would drop
            # that normalization and corrupt the language model's activations.
            return not isinstance(module, NormedEmbedding)

        return predicate

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window)
                if layer.is_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
