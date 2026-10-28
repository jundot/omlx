# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT
# ruff: noqa
"""``mlx_lm.models.lfm3``: LiquidAI LFM3, vendored from mlx-lm PR #2016.

Pinned mlx-lm ships LFM2 but no LFM3, so ``mlx_lm.utils._get_classes`` raises
``Model type lfm3 not supported`` while resolving the architecture, before a
single weight is read. The implementation is Apple's, unchanged apart from
absolute imports; see ``LICENSE.mlx-lm`` next to this file.

LFM3 mixes short-convolution, sliding-window and full-attention layers with
per-layer-type RoPE settings and oMLX's sandwich norms, so it cannot be aliased
to ``lfm2``: the parameter names (``gate_proj``, ``post_*_norm``,
``embedding_norm``) and the head gate do not exist there.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_lm.models.lfm2 import ShortConv


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    norm_eps: float
    conv_bias: bool
    conv_L_cache: int
    layer_types: List[str]
    sliding_window: int
    rope_parameters: Dict[str, Dict[str, Any]]
    use_sandwich_norm: bool = True
    tie_word_embeddings: bool = True


def _rope_freqs(head_dim: int, base: float, partial_rotary_factor: float = 1.0):
    # Partial RoPE keeps the frequency progression of the full head dimension
    # and leaves the tail unrotated. mx.fast.rope takes wavelengths, so an
    # effectively infinite wavelength is a rotation by zero.
    n_pairs = head_dim // 2
    rope_angles = int(partial_rotary_factor * head_dim) // 2
    exponents = mx.arange(0, 2 * rope_angles, 2, dtype=mx.float32) / head_dim
    freqs = base**exponents
    if rope_angles < n_pairs:
        freqs = mx.concatenate([freqs, mx.full((n_pairs - rope_angles,), 1e30)])
    return freqs


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, n_heads, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.norm_eps)

        rope = args.rope_parameters[layer_type]
        self._rope_freqs = _rope_freqs(
            head_dim, rope["rope_theta"], rope.get("partial_rotary_factor", 1.0)
        )

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x,
            self.head_dim,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._rope_freqs,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self._rope(queries, offset)
        keys = self._rope(keys, offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, mask=mask, scale=self.scale
        )

        # Sigmoid output gate, one scalar per head
        gate = mx.sigmoid(self.gate_proj(x))
        output = output.transpose(0, 2, 1, 3) * gate[..., None]
        return self.o_proj(output.reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w3 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class Lfm3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.is_attention_layer = self.layer_type != "conv"
        self.use_sandwich_norm = args.use_sandwich_norm

        if self.is_attention_layer:
            self.self_attn = Attention(args, self.layer_type)
        else:
            self.conv = ShortConv(args, layer_idx)
        self.feed_forward = MLP(args)

        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        if self.use_sandwich_norm:
            self.post_operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
            self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_attention_layer:
            r = self.self_attn(self.operator_norm(x), mask=mask, cache=cache)
        else:
            r = self.conv(self.operator_norm(x), mask=mask, cache=cache)
        if self.use_sandwich_norm:
            r = self.post_operator_norm(r)
        h = x + r

        r = self.feed_forward(self.ffn_norm(h))
        if self.use_sandwich_norm:
            r = self.post_ffn_norm(r)
        return h + r


class Lfm3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Lfm3DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

        layer_types = args.layer_types
        self.fa_idx = layer_types.index("full_attention")
        self.swa_idx = layer_types.index("sliding_attention")
        self.conv_idx = layer_types.index("conv")

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        masks = {
            "full_attention": create_attention_mask(h, cache[self.fa_idx]),
            "sliding_attention": create_attention_mask(
                h, cache[self.swa_idx], window_size=self.args.sliding_window
            ),
            "conv": create_ssm_mask(h, cache[self.conv_idx]),
        }

        for layer, c in zip(self.layers, cache):
            h = layer(h, masks[layer.layer_type], cache=c)

        return self.embedding_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        sanitized_weights = {}
        for name, param in weights.items():
            # Converted checkpoints keep the VL-style prefix
            name = name.removeprefix("language_model.")
            if "conv.weight" in name:
                if param.shape[-1] > param.shape[1]:
                    param = param.transpose(0, 2, 1)
            sanitized_weights[name] = param
        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.layer_type == "conv":
                caches.append(ArraysCache(size=1))
            elif layer.layer_type == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            else:
                caches.append(KVCache())
        return caches
