# SPDX-License-Identifier: Apache-2.0
"""Hy4-flash only parts: the compressed indexer and the Engram memory."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import exact, indexing
from .config import TextConfig
from .engram_ssd import open_embedding
from .rope import TrailingRoPE, rope_at, rotate, rotate_pairs

_ROW_CHUNK = 256
_FP8_MAX = 448.0
# The Engram norms use PyTorch's default eps, the fp32 machine epsilon.
_ENGRAM_EPS = float(np.finfo(np.float32).eps)


class CompressedIndexer(nn.Module):
    """Pick which ``ratio``-token groups each query of a sparse layer attends to.

    Every complete group of token keys is pooled into one key (fp32 value and
    gate projections, ``intra_group_ape``, ``key_norm``) and rotated at its
    first token's position. A query scores the groups that closed before its
    own and keeps the best ``index_topk / ratio - 1`` of them plus its own
    group. The result is token slots, not groups.

    Cache layout (one KVCache): keys hold every token key; values hold each
    complete group's pooled key in the group's last slot and zeros elsewhere.
    Trimming the cache therefore drops exactly the groups that are reopened.

    Groups follow each row's own positions (slot minus left padding), so a
    batched row selects what it would alone. The returned indices are cache
    slots.
    """

    def __init__(self, args: TextConfig):
        super().__init__()
        h = args.compressed_indexer_num_attention_heads
        d = args.compressed_indexer_head_dim
        self.n_heads = h
        self.head_dim = d
        self.ratio = args.compressed_indexer_compress_ratio
        self.topk = args.index_topk
        self.group_topk = self.topk // self.ratio
        self.rope_dims = args.qk_rope_head_dim
        self.rope_base = args.rope_theta("sparse")
        eps = args.rms_norm_eps
        self.query_projection = nn.Linear(args.q_lora_rank, h * d, bias=False)
        self.head_weight_projection = nn.Linear(args.hidden_size, h, bias=False)
        self.token_key_projection = nn.Linear(args.hidden_size, d, bias=False)
        self.token_key_norm = nn.RMSNorm(d, eps=eps)
        self.key_value_projection = nn.Linear(d, d, bias=False)
        self.key_gate_projection = nn.Linear(d, d, bias=False)
        self.key_norm = nn.RMSNorm(d, eps=eps)
        self.intra_group_ape = mx.zeros((self.ratio, d), dtype=mx.float32)
        self.rope = TrailingRoPE(self.rope_dims, self.rope_base)
        self._weight_scale = h**-0.5 * d**-0.5

    def _group_planes(self, start, length, positions):
        """[3, B, length] rotary positions of the first slot of the group that
        ends at each new slot, from the slot position history.

        Slots before the group's span clamp to slot 0; they never close a group.
        """
        slot = mx.maximum(start + mx.arange(length) - (self.ratio - 1), 0)
        history = positions.history
        if history is None:
            if positions.general:
                raise ValueError("Image input needs the slot position history")
            return mx.broadcast_to(
                positions.offset[None, :, None] + slot[None, None] - start,
                (3, positions.offset.shape[0], length),
            )
        return history[:, :, slot]

    def _encode(self, tokens, cache, start, slots, positions=None):
        """Pooled key of every group that ends at one of the new slots.

        Each new slot pools the ``ratio`` token keys ending at it; only slots
        that close a group of their row keep the result.
        """
        b, length, d = tokens.shape
        r = self.ratio
        prior = min(r - 1, start)
        parts = []
        if prior < r - 1:
            parts.append(mx.zeros((b, r - 1 - prior, d), dtype=tokens.dtype))
        if prior:
            parts.append(cache.keys[:, 0, start - prior : start])
        seq = mx.concatenate(parts + [tokens], axis=1).astype(mx.float32)
        values = seq @ self.key_value_projection.weight.astype(mx.float32).T
        gates = seq @ self.key_gate_projection.weight.astype(mx.float32).T
        ape = self.intra_group_ape.astype(mx.float32)
        v = mx.stack([values[:, j : j + length] for j in range(r)], axis=2)
        g = mx.stack([gates[:, j : j + length] + ape[j] for j in range(r)], axis=2)
        pooled = (v * mx.softmax(g, axis=2, precise=True)).sum(axis=2)
        pooled = self.key_norm(pooled).astype(tokens.dtype)
        if positions is None:
            pooled = rope_at(pooled, slots - (r - 1), self.rope_dims, self.rope_base)
        else:
            planes = self._group_planes(start, length, positions)
            cos, sin = positions.tables_of(planes, self.rope_base, self.rope_dims)
            pooled = rotate_pairs(pooled, cos, sin, self.rope_dims)
        closes = (slots >= r - 1) & (slots % r == r - 1)
        return mx.where(closes[..., None], pooled, mx.zeros_like(pooled))

    def __call__(
        self, x, qr, mask, cache, offset=0, positions=None
    ) -> mx.array | list | None:
        if exact.block(x):
            # Per-row selections (None below the top-k) of a verify block.
            return [
                self(
                    x[:, i : i + 1],
                    qr[:, i : i + 1],
                    None,
                    cache,
                    positions=None if positions is None else positions.row(i),
                )
                for i in range(x.shape[1])
            ]
        b, length, _ = x.shape
        start, left = cache_layout(cache, b)
        slots = start + mx.arange(length)[None] - left[:, None]
        tokens = self.token_key_norm(self.token_key_projection(x)).astype(x.dtype)
        pooled = self._encode(tokens, cache, start, slots, positions)
        if cache is not None:
            _, pooled = cache.update_and_fetch(tokens[:, None], pooled[:, None])
            pooled = pooled[:, 0]
        total = start + length
        if total <= self.topk:
            return None
        count = total // self.ratio
        ends = left[:, None] + (self.ratio - 1) + self.ratio * mx.arange(count)[None]
        valid = ends < total
        groups = mx.take_along_axis(
            pooled, mx.minimum(ends, total - 1)[..., None], axis=1
        ).astype(mx.float32)
        q = self.query_projection(qr).reshape(b, length, self.n_heads, self.head_dim)
        q = rotate(
            self.rope,
            q.swapaxes(1, 2),
            start - left if b > 1 else start - left[0],
            positions,
        )
        w = self.head_weight_projection(x).astype(mx.float32) * self._weight_scale
        own = slots // self.ratio
        keep = min(self.group_topk - 1, count)
        expand = mx.arange(self.ratio)
        native = self._native_select(q, groups, w, own, valid, keep)
        if native is not None:
            chosen, picked = native
            picks = mx.where(picked, left[:, None, None] + chosen * self.ratio, total)
            own_slots = left[:, None, None] + own[..., None] * self.ratio
            picks = mx.concatenate([picks, own_slots], axis=-1).astype(mx.int32)
            indices = (picks[..., None] + expand).flatten(-2)
            return indices[:, None].astype(mx.int32)
        q = q.astype(mx.float32)
        w = w.swapaxes(-1, -2)[..., None]
        keys_t = groups[:, None].swapaxes(-1, -2)
        out = []
        for lo in range(0, length, _ROW_CHUNK):
            hi = min(length, lo + _ROW_CHUNK)
            scores = mx.maximum(q[:, :, lo:hi] @ keys_t, 0)
            logits = (scores * w[:, :, lo:hi]).sum(axis=1)
            own_rows = own[:, lo:hi]
            visible = (mx.arange(count)[None, None] < own_rows[..., None]) & valid[
                :, None
            ]
            logits = mx.where(visible, logits, -mx.inf)
            chosen = mx.argpartition(-logits, kth=keep - 1, axis=-1)[..., :keep]
            # A row with fewer closed groups than keep also picks masked groups,
            # possibly its own one again; point those past the cache instead.
            picked = mx.take_along_axis(logits, chosen, axis=-1) > -mx.inf
            picks = mx.where(picked, left[:, None, None] + chosen * self.ratio, total)
            own_slots = left[:, None, None] + own_rows[..., None] * self.ratio
            picks = mx.concatenate([picks, own_slots], axis=-1).astype(mx.int32)
            out.append((picks[..., None] + expand).flatten(-2))
        indices = out[0] if len(out) == 1 else mx.concatenate(out, axis=1)
        return indices[:, None].astype(mx.int32)

    def _native_select(self, q, groups, w, own, valid, keep):
        """(groups [B, L, keep], usable mask) on the sparse_index kernels.

        The native top-k takes 512 columns, one more than the closed groups a
        row keeps, so the weakest of the 512 is dropped.
        """
        if keep + 1 not in indexing.NATIVE_TOPK or groups.shape[1] < keep + 1:
            return None
        scores = indexing.scores(q, groups[:, None], w, causal=False)
        if scores is None:
            return None
        count = groups.shape[1]
        visible = (mx.arange(count)[None, None] < own[..., None]) & valid[:, None]
        scores = mx.where(visible[:, None], scores, -mx.inf)
        chosen = indexing.topk(scores, keep + 1)
        if chosen is None:
            return None
        picked = mx.take_along_axis(scores, chosen, axis=-1)[:, 0]
        best = mx.argpartition(-picked, kth=keep - 1, axis=-1)[..., :keep]
        chosen = mx.take_along_axis(chosen[:, 0], best, axis=-1)
        picked = mx.take_along_axis(picked, best, axis=-1)
        return chosen, picked > -mx.inf


def cache_layout(cache, batch: int) -> tuple[int, mx.array]:
    """(physical start slot, per-row left padding) of an indexer cache."""
    zeros = mx.zeros((batch,), dtype=mx.int32)
    if cache is None:
        return 0, zeros
    left = getattr(cache, "left_padding", None)
    if left is None:
        return int(cache.offset), zeros
    return int(cache._idx), left.astype(mx.int32)


def fp8_round_trip(x: mx.array) -> mx.array:
    """Per-vector dynamic E4M3 round trip, as the Engram tables were trained."""
    scale = mx.maximum(mx.max(mx.abs(x), axis=-1, keepdims=True), 1e-12) / _FP8_MAX
    codes = mx.to_fp8(mx.clip(x / scale, -_FP8_MAX, _FP8_MAX))
    return mx.from_fp8(codes, dtype=x.dtype) * scale


def _rms(x: mx.array, weight: mx.array | None) -> mx.array:
    """PyTorch nn.RMSNorm on half inputs: fp32 math, gain applied before the cast."""
    w = None if weight is None else weight.astype(mx.float32)
    return mx.fast.rms_norm(x.astype(mx.float32), w, _ENGRAM_EPS).astype(x.dtype)


class _Norm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = mx.ones((width,))


class _Table(nn.Module):
    def __init__(self, layer: int, rows: int, width: int):
        super().__init__()
        disk_embedding = open_embedding(layer)
        self.embedding = (
            nn.Embedding(rows, width) if disk_embedding is None else disk_embedding
        )


class Engram(nn.Module):
    """Hashed n-gram memory added to the hidden streams before attention.

    The hash runs on the GPU in int64; the trained tables are only readable
    with the multipliers drawn from the loaded token map, so they are drawn on
    first use (one host read of the map's maximum).
    """

    def __init__(self, args: TextConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.hc_mult = args.hc_mult
        self.hidden_size = args.hidden_size
        self.max_order = args.engram_max_ngram_size
        self.heads_per_order = args.engram_n_head_per_ngram
        self.pad_id = args.engram_pad_id
        self.fp8 = args.engram_fp8_embedding
        self.seed = args.engram_seed
        orders = args.engram_layer_head_vocab_sizes[str(layer_id)]
        if len(orders) != self.max_order - 1 or any(
            len(o) != self.heads_per_order for o in orders
        ):
            raise ValueError(f"Invalid Engram head layout for layer {layer_id}")
        moduli = [int(p) for order in orders for p in order]
        self._moduli = mx.array(moduli, dtype=mx.int64)
        self._offsets = mx.array(
            np.cumsum([0] + moduli[:-1], dtype=np.int64), dtype=mx.int64
        )
        self.num_heads = len(moduli)
        self.head_dim = args.engram_n_embed_per_ngram // self.heads_per_order
        width = (self.max_order - 1) * args.engram_n_embed_per_ngram
        d = args.hidden_size
        self.lookup_table = mx.zeros((args.engram_lookup_table_size,), dtype=mx.int32)
        self.multi_head_embedding = _Table(layer_id, sum(moduli), self.head_dim)
        self.lookup_per_head_norms = [_Norm(self.head_dim) for _ in moduli]
        self.value_proj = nn.Linear(width, d)
        self.key_projs = [nn.Linear(width, d) for _ in range(self.hc_mult)]
        self.norm_key = [_Norm(d) for _ in range(self.hc_mult)]
        self.norm_query = [_Norm(d) for _ in range(self.hc_mult)]
        self.unified_pre_gamma = mx.ones((d,))
        self.unified_per_hc_scales = [mx.zeros((d,)) for _ in range(self.hc_mult)]
        self._multipliers = None
        self._multipliers_source = None

    def multipliers(self) -> mx.array:
        table = self.lookup_table
        if self._multipliers is None or self._multipliers_source is not table:
            compressed = int(mx.max(table).item()) + 1
            bound = max(1, int(np.iinfo(np.int64).max) // compressed // 2)
            rng = np.random.default_rng(self.seed + 10007 * self.layer_id)
            drawn = rng.integers(0, bound, size=self.max_order, dtype=np.int64)
            self._multipliers = mx.array(drawn * 2 + 1, dtype=mx.int64)
            self._multipliers_source = table
        return self._multipliers

    def hash_rows(self, ngrams: mx.array) -> mx.array:
        """[B, L, max_order] token ids -> [B, L, heads] table rows."""
        table = self.lookup_table
        size = table.shape[0]
        ids = ngrams.astype(mx.int32)
        compressed = table[mx.minimum(ids, size - 1)]
        compressed = mx.where(ids < size, compressed, table[self.pad_id])
        mixed = compressed.astype(mx.int64) * self.multipliers()
        heads = []
        for order in range(2, self.max_order + 1):
            mix = mixed[..., 0]
            for j in range(1, order):
                mix = mx.bitwise_xor(mix, mixed[..., j])
            lo = (order - 2) * self.heads_per_order
            moduli = self._moduli[lo : lo + self.heads_per_order]
            heads.append(mix[..., None] % moduli)
        rows = mx.concatenate(heads, axis=-1) + self._offsets
        return rows.astype(mx.int32)

    def lookup(self, rows: mx.array) -> mx.array:
        # A module call, so an oQ-quantized table dequantizes only these rows.
        return self.multi_head_embedding.embedding(rows)

    def __call__(self, streams: mx.array, ngrams: mx.array, keep: mx.array) -> mx.array:
        dtype = streams.dtype
        x = self.lookup(self.hash_rows(ngrams)).astype(dtype)
        if self.fp8:
            x = fp8_round_trip(x)
        gains = mx.stack([n.weight for n in self.lookup_per_head_norms]).astype(
            mx.float32
        )
        x = (mx.fast.rms_norm(x.astype(mx.float32), None, _ENGRAM_EPS) * gains).astype(
            dtype
        )
        x = x.reshape(*x.shape[:2], -1)
        gates = []
        for i in range(self.hc_mult):
            key = _rms(exact.linear(self.key_projs[i], x), self.norm_key[i].weight)
            query = _rms(streams[:, :, i], self.norm_query[i].weight)
            # Round each gate intermediate to the activation dtype, as in PyTorch.
            dot = (key * query).astype(mx.float32).sum(axis=-1).astype(dtype)
            dot = (dot.astype(mx.float32) * self.hidden_size**-0.5).astype(dtype)
            dot = mx.sqrt(mx.maximum(mx.abs(dot), 1e-6)) * mx.sign(dot)
            gates.append(mx.sigmoid(dot.astype(mx.float32)).astype(dtype))
        gates = mx.stack(gates, axis=2)[..., None]
        main = self.unified_pre_gamma * _rms(exact.linear(self.value_proj, x), None)
        main = main.astype(dtype)[:, :, None, :]
        scales = mx.stack(self.unified_per_hc_scales).astype(dtype)
        out = gates * main + scales * main
        return mx.where(keep[:, :, None, None], out, mx.zeros_like(out))
