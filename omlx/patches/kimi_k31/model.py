# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 Pro (``kimi_k31``) on the pinned mlx-lm.

Kimi K3 with DeepSeek sparse attention on its full-attention layers: an indexer
picks the top keys for each query, and K3's output gate applies to the result.
Everything else is mlx-lm's ``kimi_k3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import kimi_k3
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import CacheList, KVCache
from mlx_lm.models.deepseek_v32 import Indexer
from mlx_lm.models.pipeline import PipelineMixin

SUPPORTS_PIPELINE = True
HONORS_PIPELINE_ASSIGNMENT = True


@dataclass
class TextArgs(kimi_k3.TextArgs):
    dsa_config: dict[str, Any] | None = None

    def __post_init__(self):
        super().__post_init__()
        dsa = self.dsa_config
        if dsa is None:
            raise ValueError("A kimi_k31 config needs dsa_config")
        if dsa.get("index_rope_dim", 0) != 0:
            raise ValueError("Only index_rope_dim=0 is supported")
        if dsa.get("index_topk_freq", 1) != 1:
            raise ValueError("Indexer sharing (index_topk_freq > 1) is not supported")
        if dsa.get("index_k_norm_type", "layernorm") != "layernorm":
            raise ValueError("Only a LayerNorm indexer key norm is supported")
        if self.q_lora_rank is None:
            raise ValueError("The indexer queries need q_lora_rank")
        # Both lists are 1-based.
        if set(self.linear_attn_config["kda_layers"]) & set(dsa["dsa_layers"]):
            raise ValueError("dsa_layers must be full attention layers")


@dataclass
class ModelArgs(kimi_k3.ModelArgs):
    model_type: str = "kimi_k31"

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextArgs.from_dict(self.text_config)


class KimiK31Indexer(Indexer):
    """The DeepSeek-V3.2 indexer without RoPE."""

    def __init__(self, args: TextArgs):
        # The parent constructor reads the flat DeepSeek config and builds RoPE.
        nn.Module.__init__(self)
        dsa = args.dsa_config
        self.dim = args.hidden_size
        self.n_heads = dsa["index_num_heads"]
        self.head_dim = dsa["index_head_dim"]
        self.index_topk = dsa["index_topk"]
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=args.rms_norm_eps)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

    def rope(self, x, offset=0):
        return x


class KimiK31MLAAttention(kimi_k3.KimiK3MLAAttention):
    """Kimi K3 gated NoPE MLA over the indexer's top keys."""

    def __init__(self, args: TextArgs):
        super().__init__(args)
        self.indexer = KimiK31Indexer(args)

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)
        q = q.reshape(batch, length, self.num_heads, self.q_head_dim).transpose(
            0, 2, 1, 3
        )
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(batch, length, 1, self.qk_rope_head_dim).transpose(
            0, 2, 1, 3
        )
        kv_latent = mx.expand_dims(self.kv_a_layernorm(compressed_kv), axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        topk = self.indexer(x, qr, mask, cache=cache[1])
        if cache[0] is not None:
            # The indexer cache must be evaluated when its indices are unused.
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        if topk is not None:
            topk = mx.stop_gradient(topk)
            if length == 1:
                idx = topk[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk, axis=-1)
            else:
                shape = list(topk.shape)
                shape[-1] = kv_latent.shape[2]
                selected = mx.put_along_axis(
                    mx.zeros(shape, dtype=mx.bool_), topk, mx.array(True), axis=-1
                )
                mask = selected if mask is None else selected & mask

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if length == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache[0], scale=self.scale, mask=pe_scores
        )
        if length == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if self.use_gate:
            output = output * mx.sigmoid(self.g_proj(x))
        return self.o_proj(output)


class _LatentCacheView:
    # The K3 forward takes the attention mask and the pipeline send dependency
    # from a layer's cache. A sparse layer keeps both in its latent cache.

    def __init__(self, caches: CacheList):
        self._caches = caches

    def __getitem__(self, index):
        return self._caches[index]

    def make_mask(self, *args, **kwargs):
        return self._caches[0].make_mask(*args, **kwargs)

    @property
    def keys(self):
        return self._caches[0].keys

    @keys.setter
    def keys(self, value):
        self._caches[0].keys = value


class KimiK31TextModel(kimi_k3.KimiK3TextModel):
    def __init__(self, args: TextArgs):
        super().__init__(args)
        dsa_layers = args.dsa_config["dsa_layers"]
        for index, layer in enumerate(self.layers):
            layer.is_dsa = (index + 1) in dsa_layers
            if layer.is_dsa:
                layer.self_attn = KimiK31MLAAttention(args)

    def pipeline(self, group, split=None):
        PipelineMixin.pipeline(self, group, split)
        self.num_layers = self.end_idx - self.start_idx
        if self.use_attn_res:
            block_size = self.args.attn_res_block_size
            self.in_blocks = (self.start_idx + block_size - 1) // block_size
        self._set_cache_indices()

    def __call__(self, inputs, cache=None):
        if cache is not None:
            cache = [
                _LatentCacheView(entry) if isinstance(entry, CacheList) else entry
                for entry in cache
            ]
        return super().__call__(inputs, cache)


class LanguageModel(kimi_k3.LanguageModel):
    def __init__(self, args: TextArgs):
        super().__init__(args)
        self.model = KimiK31TextModel(args)

    def make_cache(self):
        return [
            CacheList(cache, KVCache()) if layer.is_dsa else cache
            for layer, cache in zip(self.layers, super().make_cache())
        ]


class Model(kimi_k3.Model):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.language_model = LanguageModel(args.text_config)
