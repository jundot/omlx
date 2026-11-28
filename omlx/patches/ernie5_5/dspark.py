# SPDX-License-Identifier: Apache-2.0
"""The DSpark draft that ERNIE 5.5 Flash Preview bundles under ``model.mtp``.

Three ERNIE decoder stages without compression read the target's layer taps
through a committed-only sliding context and draft one block in parallel; a
rank-256 Markov head then biases each slot by the previous token.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.models.deepseek_v4.hyper_connection import HyperHead

from ..deepseek_v41.head import project_logits
from ..mlx_lm_mtp.deepseek_v4_dspark import DSparkContextCache
from .language import _DECODE_ROWS, DecoderLayer, GatedAttention, LatentMoE, _linear


class DraftAttention(GatedAttention):
    def append_context(self, main_x, lengths, caches, start_offset):
        kv = self.kv_norm(_linear(self.wkv, main_x))[:, None]
        at = 0
        for length, cache in zip(lengths, caches):
            offset = cache.offset if start_offset is None else start_offset
            keys = self.rope(kv[:, :, at : at + length], offset)
            cache.append(keys, start_offset=start_offset)
            at += length

    def __call__(self, x, caches, width):
        qr, q, kv = self._project(x)
        out = [
            self._attend(q[b : b + 1, :, :width], kv[b : b + 1], cache)
            for b, cache in enumerate(caches)
        ]
        return self._output(mx.concatenate(out), qr[:, :width])

    def _attend(self, q, kv, cache):
        offset = cache.offset
        q, kv = self.rope(q, offset), self.rope(kv, offset)
        kv = mx.concatenate([cache.keys, kv], axis=2)
        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=None,
            scale=self.scale,
            mask=None,
            sinks=self.attn_sink.astype(q.dtype),
        )
        return self.rope(out, offset, inverse=True)


class MarkovHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.markov_w1 = nn.Embedding(config.vocab_size, config.dspark_markov_rank)
        self.markov_w2 = nn.Linear(
            config.dspark_markov_rank, config.vocab_size, bias=False
        )

    def __call__(self, ids):
        embedding = self.markov_w1(ids)
        return project_logits(embedding, self.markov_w2.weight), embedding


class ConfidenceHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(
            config.hidden_size + config.dspark_markov_rank, 1, bias=False
        )

    def __call__(self, head, embedding):
        joined = mx.concatenate([head, embedding], axis=-1).astype(mx.float32)
        return mx.sigmoid(joined @ self.projection.weight.astype(mx.float32).T)[..., 0]


class DSpark(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        taps = len(config.dspark_target_layer_ids)
        self.layers = [
            DecoderLayer(config, DraftAttention(config, i), LatentMoE(config))
            for i in range(taps)
        ]
        self.feature_projection = nn.Linear(
            taps * config.hidden_size, config.hidden_size, bias=False
        )
        self.feature_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mhc_contract = HyperHead(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = MarkovHead(config)
        self.confidence_head = ConfidenceHead(config)

    def append_context(self, taps, caches, start_offset=None):
        lengths = [t.shape[1] for t in taps]
        main_x = self.feature_norm(
            _linear(self.feature_projection, mx.concatenate(taps, axis=1))
        )
        for s, layer in enumerate(self.layers):
            rings = [cache[s] for cache in caches]
            layer.attn.append_context(main_x, lengths, rings, start_offset)

    def project_taps(self, context):
        if not context.taps:
            return None
        taps, ends = zip(*context.taps)
        context.taps.clear()
        ends = mx.concatenate([end.reshape(1) for end in ends]).tolist()
        first = len(ends) - 1
        while first and ends[first - 1] == ends[first] - 1:
            first -= 1
        start = ends[first] - 1
        if context[0].offset != start:
            context[:] = [DSparkContextCache(ring.max_size) for ring in context]
        rows = mx.concatenate(taps[first:], axis=1)
        for at in range(0, rows.shape[1], _DECODE_ROWS):
            self.append_context(
                [rows[:, at : at + _DECODE_ROWS]], [context], start + at
            )
        return ends[-1]

    def block(self, embed_tokens, anchors, caches, width):
        config = self.config
        noise = mx.full(
            (anchors.shape[0], config.dspark_block_size - 1),
            config.dspark_noise_token_id,
            dtype=anchors.dtype,
        )
        h = embed_tokens(mx.concatenate([anchors.reshape(-1, 1), noise], axis=1))
        h = mx.broadcast_to(h[:, :, None], (*h.shape[:2], config.hc_mult, h.shape[-1]))
        for s, layer in enumerate(self.layers):
            rings = [cache[s] for cache in caches]
            stage_width = width if s == len(self.layers) - 1 else h.shape[1]
            mixed, mixes = layer.attn_hc.collapse(h, layer.attn_norm)
            out = layer.attn(mixed, rings, stage_width)
            h = layer.attn_hc.expand(out, h[:, :stage_width], mixes[:, :stage_width])
            mixed, mixes = layer.ffn_hc.collapse(h, layer.ffn_norm)
            h = layer.ffn_hc.expand(layer.ffn(mixed), h, mixes)
        return self.mhc_contract(h)

    def logits(self, head):
        return self.head(self.norm(head))
