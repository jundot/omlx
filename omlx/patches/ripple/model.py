# SPDX-License-Identifier: Apache-2.0
# Adapted from OrcaRouter's MLX port of Ricursive's Ripple implementation.
# Source: orcarouter/Ripple-65B-abliterated-MLX @ f05d1fd49e6ff40388db462f1379140dedd2768a
# Modified for oMLX registration, DFlash2 hidden-state capture and cache rollback.

# Ripple-65B (model_type "ricursive") for mlx-lm.
#
# A transcription of the reference modeling_ricursive.py. What differs from a Llama block:
#   - PeriNorm: RMSNorm before AND after both the attention and the FFN sub-blocks
#   - 64 sliding-window layers (window 1024, RoPE) and 16 global layers (full attention, no RoPE),
#     per config.layer_types
#   - per-head RMSNorm on q and k
#   - gated attention: the attention output is multiplied by sigmoid(g_proj(x)), sigmoid in fp32
#   - value residual: v_proj reads lerp(scaled embeddings, x, sigmoid(value_res_lambda))
#   - the FFN clamps gate from above and up on both sides at ffn_clamp_threshold
#   - the embeddings are multiplied by embed_scale; the RMSNorm weight multiplies directly (no 1+w)

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache


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
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    sliding_window: int = 1024
    layer_types: list[str] | None = None
    hybrid_layer_pattern: list[int] | None = None
    embed_scale: float | None = None
    ffn_clamp_threshold: float | None = None
    use_gated_attention: bool = True
    gate_full_precision: bool = True
    use_valres: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            # hybrid_layer_pattern: 1 = sliding, 0 = global
            pattern = self.hybrid_layer_pattern or [1] * self.num_hidden_layers
            self.layer_types = [
                "sliding_attention" if p else "full_attention" for p in pattern
            ]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.gated = args.use_gated_attention
        self.gate_fp32 = args.gate_full_precision

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        if self.gated:
            self.g_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        if args.use_valres:
            self.value_res_lambda = mx.zeros((1,))

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        if self.is_sliding:
            self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
        value_residual: mx.array | None = None,
        verify_rows: bool = False,
    ) -> mx.array:
        batch_size, length, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        v_in = x
        if value_residual is not None:
            blend = mx.sigmoid(self.value_res_lambda[0]).astype(x.dtype)
            v_in = value_residual + blend * (x - value_residual)
        values = self.v_proj(v_in)

        queries = self.q_norm(
            queries.reshape(batch_size, length, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(batch_size, length, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, length, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        if verify_rows and cache is not None and length > 1:
            # Keep one-token attention arithmetic and physical ring order while
            # the heavy projections still share one pass over their weights.
            outputs = []
            for row in range(length):
                q = queries[:, :, row : row + 1]
                k = keys[:, :, row : row + 1]
                v = values[:, :, row : row + 1]
                if self.is_sliding:
                    q = self.rope(q, offset=cache.offset)
                    k = self.rope(k, offset=cache.offset)
                row_mask = create_attention_mask(x[:, row : row + 1], cache)
                k, v = cache.update_and_fetch(k, v)
                outputs.append(
                    scaled_dot_product_attention(
                        q, k, v, cache=cache, scale=self.scale, mask=row_mask
                    )
                )
            output = mx.concatenate(outputs, axis=2)
        else:
            if self.is_sliding:
                offset = cache.offset if cache is not None else 0
                queries = self.rope(queries, offset=offset)
                keys = self.rope(keys, offset=offset)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, length, -1)
        if self.gated:
            gate = self.g_proj(x)
            if self.gate_fp32:
                gate = gate.astype(mx.float32)
            output = output * mx.sigmoid(gate).astype(output.dtype)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.clamp = args.ffn_clamp_threshold
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.clamp:
            gate = mx.minimum(gate, self.clamp)
            up = mx.clip(up, -self.clamp, self.clamp)
        return self.down_proj(nn.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        eps = args.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)

    def __call__(
        self, x, mask=None, cache=None, value_residual=None, verify_rows=False
    ):
        r = self.self_attn(
            self.input_layernorm(x), mask, cache, value_residual, verify_rows
        )
        h = x + self.post_attention_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        return h + self.post_feedforward_layernorm(r)


class RicursiveModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        types = args.layer_types
        self.first_global = (
            types.index("full_attention") if "full_attention" in types else None
        )
        self.first_sliding = (
            types.index("sliding_attention") if "sliding_attention" in types else None
        )

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        capture_layer_ids=(),
        capture=None,
        verify_rows=False,
    ):
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )
        if self.args.embed_scale is not None:
            # in fp32, as torch does for a bf16 tensor times a python float
            h = (h.astype(mx.float32) * self.args.embed_scale).astype(h.dtype)
        value_residual = h if self.args.use_valres else None

        if cache is None:
            cache = [None] * len(self.layers)
        global_mask = sliding_mask = None
        if self.first_global is not None:
            global_mask = create_attention_mask(h, cache[self.first_global])
        if self.first_sliding is not None:
            sliding_mask = create_attention_mask(
                h, cache[self.first_sliding], window_size=self.args.sliding_window
            )
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            mask = sliding_mask if layer.self_attn.is_sliding else global_mask
            h = layer(h, mask, c, value_residual, verify_rows)
            if capture is not None and i in capture_layer_ids:
                capture[i] = h
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = RicursiveModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    _omlx_drafter_single_row = True
    _omlx_mtp_row_exact_verify = True

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        *,
        capture_layer_ids=None,
        return_hidden=False,
        n_confirmed=0,
    ):
        drafter = getattr(self, "_omlx_drafter", None)
        scope = getattr(drafter, "scope_uids", None)
        observe = (
            scope
            and capture_layer_ids is None
            and not return_hidden
            and inputs is not None
            and inputs.shape == (len(scope), 1)
        )
        ids = drafter.target_layer_ids if observe else (capture_layer_ids or ())
        capture = {} if ids else None
        out = self.model(
            inputs, cache, input_embeddings, ids, capture, bool(n_confirmed)
        )
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)
        if capture is not None:
            states = [capture[i] for i in ids]
            if observe:
                drafter.observe(scope, states)
            else:
                if return_hidden:
                    states.append(out)
                return SimpleNamespace(
                    logits=logits, hidden_states=states, gdn_states=None
                )
        return (logits, out) if return_hidden else logits

    def mtp_partial_rollback(self, caches, accepted, num_drafts):
        count = int(num_drafts) - int(accepted)
        if not count:
            return True
        if count < 0 or not all(c.is_trimmable() for c in caches):
            return False
        for cache in caches:
            if cache.trim(count) != count:
                raise RuntimeError("Ripple cache failed speculative trim")
        return True

    def rollback_speculative_cache(self, caches, state, accepted, block_size):
        if not self.mtp_partial_rollback(caches, accepted, block_size - 1):
            raise RuntimeError("Ripple caches cannot roll back this draft")
        return accepted

    def sanitize(self, weights):
        return {k: v for k, v in weights.items() if "rotary_emb" not in k}

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window)
                if t == "sliding_attention"
                else KVCache()
            )
            for t in self.args.layer_types
        ]
