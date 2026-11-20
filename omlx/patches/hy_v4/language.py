# SPDX-License-Identifier: Apache-2.0
"""Hy4 / Hy4-flash text decoder for oMLX's mlx-vlm runtime."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import CacheList, KVCache, RotatingKVCache
from mlx_vlm.models.base import LanguageModelOutput, create_attention_mask

from .config import TextConfig
from .flash import Engram
from .layers import (
    MLP,
    Attention,
    HyperConnection,
    HyperHead,
    MoE,
    hc_expand,
)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer: int):
        super().__init__()
        self.self_attn = Attention(args, layer)
        if args.mlp_layer_types[layer] == "sparse":
            self.mlp = MoE(args)
        else:
            self.mlp = MLP(args, args.intermediate_size, post_norm=args.is_flash)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.hc_attn_layer = HyperConnection(args)
        self.hc_mlp_layer = HyperConnection(args)
        if args.engram_layer_ids and layer + 1 in args.engram_layer_ids:
            self.engram_layer = Engram(args, layer + 1)

    def __call__(self, streams, mask, cache, prev_topk, engram_inputs):
        if "engram_layer" in self and engram_inputs is not None:
            streams = streams + self.engram_layer(streams, *engram_inputs)
        x, post = self.hc_attn_layer(streams)
        x, topk = self.self_attn(self.input_layernorm(x), mask, cache, prev_topk)
        streams = hc_expand(x, streams, post)
        x, post = self.hc_mlp_layer(streams)
        x = self.mlp(self.post_attention_layernorm(x))
        return hc_expand(x, streams, post), topk


class TextModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hc_head = HyperHead(args)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        kinds = args.attention_kinds
        self._first = {k: kinds.index(k) for k in ("sparse", "sliding") if k in kinds}

    def _masks(self, h, cache):
        masks = {}
        batched = any(
            c is not None and hasattr(_latent_cache(c), "left_padding") for c in cache
        )
        if h.shape[1] == 1 and not batched:
            return masks
        for kind, layer in self._first.items():
            c = _latent_cache(cache[layer])
            window = self.args.sliding_window if kind == "sliding" else None
            masks[kind] = create_attention_mask(
                h, c, window_size=window, return_array=True
            )
        return masks

    def __call__(self, inputs, cache=None, inputs_embeds=None, engram_inputs=None):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        masks = self._masks(h, cache)
        b, length, d = h.shape
        streams = mx.contiguous(
            mx.broadcast_to(h[:, :, None, :], (b, length, self.args.hc_mult, d))
        )
        prev_topk = None
        kinds = self.args.attention_kinds
        for layer, c, kind in zip(self.layers, cache, kinds):
            streams, topk = layer(streams, masks.get(kind), c, prev_topk, engram_inputs)
            if kind == "sparse":
                prev_topk = topk
        return self.norm(self.hc_head(streams))


def _latent_cache(cache):
    if isinstance(cache, CacheList):
        return cache[0]
    return cache


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.args = config
        self.config = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.media_token_ids = (120120, 120124)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for i, layer in enumerate(self.layers):
            attention = layer.self_attn
            if attention.kind == "sliding":
                caches.append(RotatingKVCache(max_size=attention.window, keep=0))
                continue
            entries = [KVCache()]
            if attention.index_module is not None:
                entries.append(KVCache())
            if i == 0 and self.config.engram_layer_ids:
                if len(entries) != 2:
                    raise ValueError("Engram history needs an indexed first layer")
                # Token ids for the Engram n-grams, trimmed with the KV caches.
                entries.append(KVCache())
            caches.append(CacheList(*entries))
        return caches

    def _engram_inputs(self, ids: mx.array, cache) -> tuple[mx.array, mx.array]:
        args = self.config
        pad = args.engram_pad_id
        width = args.engram_max_ngram_size - 1
        b, length = ids.shape
        history = None
        if cache is not None and isinstance(cache[0], CacheList):
            entries = cache[0].caches
            history = entries[2] if len(entries) > 2 else None
        total = length
        parts = []
        if history is not None:
            tokens, _ = history.update_and_fetch(
                ids.astype(mx.int32)[:, None, :, None],
                mx.zeros((b, 1, length, 0), dtype=mx.int32),
            )
            tokens = tokens[:, 0, :, 0]
            total = tokens.shape[1]
            lead = tokens[:, max(0, total - length - width) : total - length]
            if lead.shape[1]:
                parts.append(lead)
        # Positions before the start of the sequence read as padding.
        missing = width - sum(p.shape[1] for p in parts)
        if missing:
            parts.insert(0, mx.full((b, missing), pad, dtype=mx.int32))
        parts.append(ids.astype(mx.int32))
        window = mx.concatenate(parts, axis=1)
        left = getattr(history, "left_padding", None)
        if left is not None:
            slots = mx.arange(total - length - width, total)[None]
            window = mx.where(slots < left[:, None], pad, window)
        ngrams = mx.stack(
            [window[:, width - j : width - j + length] for j in range(width + 1)],
            axis=-1,
        )
        image, video = self.media_token_ids
        keep = (ids != image) & (ids != video) & (ids != pad)
        return ngrams, keep

    def project(self, hidden: mx.array) -> mx.array:
        head = (
            self.model.embed_tokens.as_linear
            if self.config.tie_word_embeddings
            else self.lm_head
        )
        if self.config.enable_lm_head_fp32:
            return head(hidden.astype(mx.float32))
        return head(hidden)

    def __call__(
        self,
        inputs: mx.array | None = None,
        cache: Any | None = None,
        inputs_embeds: mx.array | None = None,
        mask: mx.array | None = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        engram_inputs = None
        if self.config.engram_layer_ids:
            engram_inputs = self._engram_inputs(inputs, cache)
        hidden = self.model(inputs, cache, inputs_embeds, engram_inputs)
        logits = None if kwargs.get("skip_logits") else self.project(hidden)
        return LanguageModelOutput(
            logits=logits,
            hidden_states=[hidden] if kwargs.get("return_hidden") else None,
        )
