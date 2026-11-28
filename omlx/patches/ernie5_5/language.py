# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Neywa Labs Inc.
# mlx-vlm: adapted from mlx_vlm/models/ernie5_5_moe/language.py at commit 36494b42.

import re
import threading
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_vlm.models.cache import CacheList, KVCache, PoolingCache, RotatingKVCache
from mlx_vlm.models.deepseek_v4.hyper_connection import HyperHead
from mlx_vlm.models.deepseek_v4.language import (
    Compressor,
    DeepseekV4RoPE,
    LocalAttention,
    _align_local_mask,
    _extend_mask,
)
from mlx_vlm.models.mla import MultiLinear
from mlx_vlm.models.switch_layers import SwitchGLU

from ...custom_kernels.attention import fast as attention_fast
from ...custom_kernels.nax import is_nax_available
from ...custom_kernels.quant_matmul import fast as quant_fast
from ...custom_kernels.sparse_index import fast as index_fast
from .. import fp4_expert_rows
from ..deepseek_v4.sparse_attention_nax import sparse_topk_attention_nax
from ..deepseek_v4.verify_attention import rowwise_gemm
from ..deepseek_v4.verify_qmv import exact_verify_multi_qmv
from ..deepseek_v4.wsdpa_attention import wsdpa_prefill
from ..deepseek_v41.gemv import mxfp8_gemv_each
from ..deepseek_v41.head import project_logits
from ..glm_moe_dsa.indexer_nax import indexer_scores_nax, max_rows_per_call
from ..glm_moe_dsa.sparse_mla_nax import sparse_mla_attention_nax
from ..mimo_v2.decode_fast import combine_rms
from ..mlx_lm_mtp.cache_rollback import stage_functional_rotating_update
from ..moe_routes import sort_routes
from . import kernels, sparse_prefill_available
from .config import ModelConfig
from .mtp import DSparkHost

_DECODE_ROWS = 8
_VERIFY = threading.local()


def set_dspark_verify_armed(flag: bool) -> None:
    """The MTP generator arms this around draft verify forwards."""
    _VERIFY.armed = flag


def _request_caches(cache, batch, length):
    if isinstance(cache, list):
        return cache
    armed = getattr(_VERIFY, "armed", False)
    if cache is not None and armed and batch == 1 and 1 < length <= _DECODE_ROWS:
        return [cache]
    return None


def _fields(cache):
    # mlx: a new handle keeps the current value when setitem rebinds the old one.
    return {
        k: (
            list(v)
            if isinstance(v, list)
            else mx.array(v) if isinstance(v, mx.array) else v
        )
        for k, v in vars(cache).items()
        if k != "_ernie_verify_undo"
    }


def _states(cache):
    return [c.state for c in cache] if isinstance(cache, list) else cache.state


def _offset(cache):
    offset = cache.offset if cache is not None else 0
    return mx.array(offset) if isinstance(offset, mx.array) else offset


def _window_rows(local, kv):
    size, rows = local.max_size, kv.shape[2]
    batched = hasattr(local, "rotated")
    filled = local._offset if batched else local.offset
    empty = mx.zeros((*kv.shape[:-1], 0), dtype=kv.dtype)
    if filled < size or local.keys.shape[2] != size:
        windows, masks = [], []
        for r in range(rows):
            mask = local.make_mask(1, window_size=size, return_array=True)
            keys, _ = local.update_and_fetch(
                kv[:, :, r : r + 1], empty[:, :, r : r + 1]
            )
            windows.append(keys[0, 0])
            masks.append(None if mask is None else mask.reshape(-1)[-keys.shape[2] :])
        return windows, masks
    stage_functional_rotating_update(local, kv, empty)
    idx, rotated, writes = local._idx, batched and local.rotated, 0
    slots, positions, drops = [], [], []
    for _ in range(rows):
        if batched:
            wraps = rotated or idx >= size
            shift = idx % size + 1 if wraps else 0
            positions.append([(s - shift) % size for s in range(size)])
            drops.append(writes + wraps)
        if idx == size:
            idx, rotated = (0, True) if batched else (local.keep, False)
        writes += rotated
        slots.append(idx)
        idx += 1
    source = mx.concatenate([local.keys, kv], axis=2)
    plan = _ring_plan(size, tuple(slots))
    local.keys = mx.take(source, plan[-1], axis=2)
    local._idx = idx
    local.offset = local.offset + rows
    windows = list(mx.take(source[0, 0], plan, axis=0))
    if not batched:
        return windows, [None] * rows
    visible = mx.array(positions) >= (local.left_padding - mx.array(drops))[:, None]
    local._offset += rows
    local.rotated = rotated
    local.left_padding = local.left_padding - writes
    local.keys = mx.depends(local.keys, (local.left_padding, local.offset))
    return windows, list(visible)


def _pool_size(pool):
    if pool.pooled is None:
        return 0, 0
    extent = pool.pooled.shape[1]
    return extent, getattr(pool, "_pool_lengths", [extent])[0]


@lru_cache(maxsize=256)
def _ring_plan(size, slots):
    rows, current = [], list(range(size))
    for update, slot in enumerate(slots):
        current[slot] = size + update
        rows.append(list(current))
    return mx.array(rows, dtype=mx.uint32)


def _rows_attention(q, keys, masks, scale, sinks, bias=None):
    q = q * mx.array(scale, dtype=q.dtype)
    out, start = [], 0
    while start < len(keys):
        stop = start + 1
        while (
            stop < len(keys)
            and keys[stop].shape == keys[start].shape
            and (masks[stop] is None) == (masks[start] is None)
        ):
            stop += 1
        group = mx.stack(keys[start:stop])
        scores = rowwise_gemm(q[start:stop], group, True)
        if masks[start] is not None:
            visible = mx.stack(masks[start:stop])[:, None]
            scores = mx.where(visible, scores, mx.array(-mx.inf, dtype=scores.dtype))
        if bias is not None:
            scores = scores + bias(start, stop)
        sink = mx.broadcast_to(
            sinks.astype(scores.dtype)[None, :, None], (*scores.shape[:-1], 1)
        )
        weights = mx.softmax(
            mx.concatenate([sink, scores], axis=-1), axis=-1, precise=True
        )[..., 1:]
        out.append(rowwise_gemm(weights, group, False))
        start = stop
    return mx.concatenate(out)


def _per_row(module, x):
    return (x[..., None, :] @ module.weight.swapaxes(-1, -2)[:, None])[..., 0, :]


def _linear(module, x):
    width = x.shape[-1]
    rows = x.size // width
    if rows > _DECODE_ROWS:
        return module(x)
    flat = x.reshape(rows, width)
    if getattr(module, "mode", None) == "mxfp8":
        y = mxfp8_gemv_each(flat, module.weight, module.scales)
        if y is not None:
            return y.reshape(*x.shape[:-1], -1)
    if rows == 1 or "scales" in module:
        return module(x)
    y = mx.concatenate([module(flat[r : r + 1]) for r in range(rows)])
    return y.reshape(*x.shape[:-1], -1)


@mx.compile
def _situ(up, gate, beta, linear_beta):
    dtype = gate.dtype
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    gate = beta * mx.tanh(gate / beta) * mx.sigmoid(gate)
    up = linear_beta * mx.tanh(up / linear_beta)
    return (gate * up).astype(dtype)


class SiTU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta

    def __call__(self, up, gate):
        return _situ(up, gate, self.beta, self.linear_beta)


class MLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation = SiTU(config)

    def __call__(self, x):
        up, gate = _linear(self.up_proj, x), _linear(self.gate_proj, x)
        return _linear(self.down_proj, self.activation(up, gate))


class MultiRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.weight_1 = mx.zeros_like(self.weight)
        self.routed_scaling_factor_param = mx.full(
            (config.n_routed_experts,), config.routed_scaling_factor
        )
        self.top_k = config.num_experts_per_tok
        self.normalize = config.norm_topk_prob

    def __call__(self, x, bias):
        x = x.astype(mx.float32)
        scores = mx.sigmoid(x @ self.weight.astype(mx.float32).T)
        scores += mx.sigmoid(x @ self.weight_1.astype(mx.float32).T)
        ids = mx.argpartition(-(scores + bias), kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        ids = mx.stop_gradient(ids)
        weights = mx.take_along_axis(scores, ids, axis=-1)
        if self.normalize:
            weights /= mx.maximum(weights.sum(axis=-1, keepdims=True), 1e-20)
        return ids, weights * self.routed_scaling_factor_param[ids]


class LatentMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = MultiRouter(config)
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))
        self.fc1_latent_proj = nn.Linear(
            config.hidden_size, config.moe_latent_size, bias=False
        )
        self.switch_mlp = SwitchGLU(
            config.moe_latent_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=SiTU(config),
        )
        self.latent_norm = nn.RMSNorm(config.moe_latent_size, eps=config.rms_norm_eps)
        self.fc2_latent_proj = nn.Linear(
            config.moe_latent_size, config.hidden_size, bias=False
        )
        self.shared_experts = MLP(
            config, config.n_shared_experts * config.moe_intermediate_size
        )
        self._situ = mx.array(
            [config.activation_situ_beta, config.activation_situ_linear_beta]
        )

    def __call__(self, x):
        routed = None
        if x.size // x.shape[-1] <= _DECODE_ROWS:
            routed = self._routed_rows(x)
        if routed is None:
            ids, weights = self.gate(x, self.e_score_correction_bias)
            routed = self.latent_norm(
                self._routed_sorted(self.fc1_latent_proj(x), ids, weights)
            )
        return self.shared_experts(x) + _linear(self.fc2_latent_proj, routed)

    def _routed_sorted(self, latent, ids, weights):
        experts = self.switch_mlp
        x, row_map, idx, inv_order = sort_routes(mx.expand_dims(latent, (-2, -3)), ids)
        x = x[row_map]
        act = experts.activation(
            experts.up_proj(x, idx, sorted_indices=True),
            experts.gate_proj(x, idx, sorted_indices=True),
        )
        out = experts.down_proj(act, idx, sorted_indices=True)
        if (
            out.dtype in (mx.bfloat16, mx.float16)
            and ids.shape[-1] in quant_fast.moe_weighted_sum_top_ks()
        ):
            return quant_fast.moe_weighted_sum(out, inv_order, weights)
        out = out[inv_order].reshape(*ids.shape, -1).astype(mx.float32)
        return (out * weights[..., None]).sum(axis=-2).astype(latent.dtype)

    def _routed_rows(self, x):
        rows = x.reshape(-1, x.shape[-1])
        gate = self.gate
        routing = kernels.route(
            rows,
            gate.weight,
            gate.weight_1,
            self.e_score_correction_bias,
            gate.routed_scaling_factor_param,
            gate.top_k,
            gate.normalize,
        )
        if routing is None:
            return None
        ids, weights = routing
        latent = _linear(self.fc1_latent_proj, rows)
        experts = self.switch_mlp
        down = experts.down_proj
        projections = (experts.gate_proj, experts.up_proj, down)
        if all(getattr(p, "mode", None) == "mxfp4" for p in projections):
            act = fp4_expert_rows.gate_up(
                latent,
                ids,
                experts.gate_proj.weight,
                experts.gate_proj.scales,
                experts.up_proj.weight,
                experts.up_proj.scales,
                n_out=experts.gate_proj.weight.shape[1],
                scale_group_size=down.group_size,
                activation=kernels.SITU,
                extra=(self._situ,),
            )
            out = fp4_expert_rows.down(
                act, ids, down.weight, down.scales, scale_group_size=down.group_size
            )
        else:
            pairs = mx.expand_dims(latent, (-2, -3))
            act = experts.activation(
                experts.up_proj(pairs, ids), experts.gate_proj(pairs, ids)
            )
            out = down(act, ids)
        out = out.reshape(*ids.shape, -1)
        _, normed = combine_rms(
            mx.zeros_like(latent),
            out,
            weights,
            self.latent_norm.weight,
            self.latent_norm.eps,
        )
        return normed.reshape(*x.shape[:-1], -1)


class HyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iters = config.hc_sinkhorn_iters
        count = (2 + config.hc_mult) * config.hc_mult
        self.fn = mx.zeros(
            (count, config.hc_mult * config.hidden_size), dtype=mx.float32
        )
        self.base = mx.zeros((count,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)
        self._eps = mx.array([config.rms_norm_eps, config.hc_eps])

    def collapse(self, x, norm):
        return kernels.hc_pre(x, self.fn, self.scale, self.base, norm.weight, self._eps)

    def expand(self, y, x, mixes):
        return kernels.hc_expand(
            mixes, y, x, self.scale, self.base, self._eps, self.iters
        )


class VirtualHeads(nn.Module):
    def __init__(self, heads, rank):
        super().__init__()
        self.U = mx.zeros((heads, rank))
        self.V = mx.zeros((heads, rank))

    def __call__(self, o, groups):
        mix = (
            mx.eye(self.U.shape[0])
            + self.V.astype(mx.float32) @ self.U.astype(mx.float32).T
        )
        return kernels.vha(o, mix.astype(o.dtype), groups)


class HCACompressor(Compressor):
    def project(self, x):
        x32 = x.astype(mx.float32)
        return project_logits(x32, self.wkv.weight), project_logits(
            x32, self.wgate.weight
        )

    def consume(self, kv, gate, pool_cache, offset, dtype):
        B = kv.shape[0]
        if pool_cache is None:
            usable = kv.shape[1] // self.compress_ratio * self.compress_ratio
            kv, gate, base = kv[:, :usable], gate[:, :usable], offset
        else:
            kv, gate, base = pool_cache.accumulate_windows(kv, gate, offset)
        if kv.shape[1] == 0:
            pooled = mx.zeros((B, 0, self.head_dim), dtype=dtype)
        else:
            kv = kv.reshape(B, -1, self.compress_ratio, self.head_dim)
            gate = gate.reshape(B, -1, self.compress_ratio, self.head_dim)
            weights = mx.softmax(
                gate + self.ape.astype(mx.float32), axis=-2, precise=True
            )
            pooled = self.norm((kv * weights).sum(axis=-2))
            pooled = self.rope(pooled[:, None], offset=base).squeeze(1).astype(dtype)
        return pool_cache.update_and_fetch(pooled) if pool_cache is not None else pooled


class GatedAttention(LocalAttention):
    """Query, value-head mix and gated grouped output shared by HCA and DSpark."""

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.o_gate = nn.Linear(
            config.q_lora_rank, config.o_groups * config.o_lora_rank, bias=False
        )
        self.vha = VirtualHeads(self.n_heads, config.vha_rank)

    def _project(self, x):
        qr = self.q_norm(_linear(self.wq_a, x))
        q = mx.fast.rms_norm(self._queries(qr), None, self.config.rms_norm_eps)
        return qr, q, self.kv_norm(_linear(self.wkv, x))[:, None]

    def _output(self, out, qr):
        B, _, L, _ = out.shape
        out = self.vha(out, self.o_groups)
        out = self._project_groups(out, B, L)
        gate = _linear(self.o_gate, qr).reshape(out.shape)
        return _linear(self.wo_b, (out * mx.sigmoid(gate)).reshape(B, L, -1))

    def _queries(self, qr):
        batch, length, width = qr.shape
        wq_b = self.wq_b
        if batch * length <= _DECODE_ROWS or "scales" not in wq_b:
            q = _linear(wq_b, qr).reshape(batch, length, self.n_heads, -1)
            return q.transpose(0, 2, 1, 3)
        return mx.quantized_matmul(
            qr[:, None],
            wq_b.weight.reshape(self.n_heads, self.head_dim, -1),
            wq_b.scales.reshape(self.n_heads, self.head_dim, -1),
            transpose=True,
            group_size=wq_b.group_size,
            bits=wq_b.bits,
            mode=wq_b.mode,
        )

    def _project_groups(self, grouped, batch, length):
        groups, rows, width = grouped.shape
        wo_a = self.wo_a
        if rows > _DECODE_ROWS:
            out = wo_a(grouped.reshape(groups, batch, length, width).swapaxes(0, 1))
            return out.swapaxes(1, 2)
        if getattr(wo_a, "mode", None) == "mxfp8" and width % 512 == 0:
            out = exact_verify_multi_qmv(wo_a, grouped)
        else:
            out = mx.concatenate(
                [wo_a(grouped[:, r : r + 1]) for r in range(rows)], axis=1
            )
        return out.swapaxes(0, 1).reshape(batch, length, groups, -1)


class HCAAttention(GatedAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = HCACompressor(config, self.compress_ratio, self.head_dim)

    def __call__(self, x, cache=None):
        B, L, _ = x.shape
        qr, q, kv = self._project(x)
        projected = self.compressor.project(x)
        requests = _request_caches(cache, B, L)
        if requests is None:
            return self._output(self._attend(x, q, kv, projected, cache), qr)
        queries, keys, masks, offsets = [], [], [], []
        for b, c in enumerate(requests):
            offset = _offset(c[0])
            queries.append(self.rope(q[b : b + 1], offset)[0].transpose(1, 0, 2))
            request_keys, request_masks = self._verify_keys(
                x[b : b + 1],
                kv[b : b + 1],
                [p[b : b + 1] for p in projected],
                c,
                offset,
            )
            keys.extend(request_keys)
            masks.extend(request_masks)
            offsets.append(offset)
        out = _rows_attention(
            mx.concatenate(queries),
            keys,
            masks,
            self.scale,
            self.attn_sink.astype(q.dtype),
        )
        out = out.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        out = [
            self.rope(out[b : b + 1], o, inverse=True) for b, o in enumerate(offsets)
        ]
        return self._output(mx.concatenate(out), qr)

    def _verify_keys(self, x, kv, projected, cache, offset):
        local, pool = cache[0], cache[1]
        pool._ernie_verify_undo = (_fields(pool), *projected, offset, x.dtype)
        kv = self.rope(kv, offset)
        windows, masks = _window_rows(local, kv)
        remainder = pool.remainder
        remainder = remainder[0] if isinstance(remainder, list) else remainder
        before = _pool_size(pool)
        pooled = self.compressor.consume(*projected, pool, offset, x.dtype)[0]
        after = _pool_size(pool)
        first = self.compress_ratio - 1 - remainder
        keys, key_masks = [], []
        for r, (window, mask) in enumerate(zip(windows, masks)):
            extent, length = after if r >= first else before
            keys.append(mx.concatenate([window, pooled[:extent]]))
            if mask is None and length == extent:
                key_masks.append(None)
                continue
            visible = mx.arange(extent) < length
            mask = mx.ones(window.shape[:1], mx.bool_) if mask is None else mask
            key_masks.append(mx.concatenate([mask, visible]))
        return keys, key_masks

    def rollback(self, cache, kept, rejected):
        pool = cache[1]
        fields, kv, gate, offset, dtype = pool._ernie_verify_undo
        cache[0].trim(rejected)
        vars(pool).update(fields)
        pool._ernie_verify_undo = None
        self.compressor.consume(kv[:, :kept], gate[:, :kept], pool, offset, dtype)

    def _attend(self, x, q, kv, projected, cache):
        B, L, _ = x.shape
        local = cache[0] if cache is not None else None
        pool = cache[1] if cache is not None else None
        offset = _offset(local)
        mask = create_attention_mask(
            x, local, window_size=self.config.sliding_window, return_array=True
        )
        q, kv = self.rope(q, offset), self.rope(kv, offset)
        if local is not None:
            kv, _ = local.update_and_fetch(kv, mx.zeros((B, 1, L, 0), dtype=x.dtype))
        pooled = self.compressor.consume(*projected, pool, offset, x.dtype)
        sinks = self.attn_sink.astype(q.dtype)
        out = None
        if L > 1:
            out = self._prefill_attention(q, kv, pooled, sinks, offset)
        if out is None:
            out = self._dense_attention(q, kv, pooled, mask, local, pool, sinks, offset)
        return self.rope(out, offset, inverse=True)

    def _prefill_attention(self, q, kv, pooled, sinks, offset):
        if isinstance(offset, mx.array):
            return None
        args = (self.scale, offset, self.config.sliding_window, self.compress_ratio)
        rows = mx.arange(pooled.shape[1], dtype=mx.uint32)
        every = mx.broadcast_to(rows[None, None], (1, q.shape[2], rows.size))
        out = sparse_topk_attention_nax(q, kv, pooled, every, sinks, *args)
        if out is not None:
            return out
        return wsdpa_prefill(q, kv, pooled, sinks, *args)

    def _dense_attention(self, q, kv, pooled, mask, local, pool, sinks, offset):
        batch, _, length, _ = q.shape
        mask = _align_local_mask(mask, kv.shape[2])
        if pooled.shape[1]:
            if pool is not None:
                pool_mask = pool.make_mask(length, offset)
            else:
                pool_mask = mx.arange(pooled.shape[1])[None] < (
                    mx.arange(1, length + 1)[:, None] // self.compress_ratio
                )
            if mask is None and pool_mask is not None:
                mask = mx.ones((batch, 1, length, kv.shape[2]), dtype=mx.bool_)
            kv = mx.concatenate((kv, pooled[:, None]), axis=2)
            mask = _extend_mask(mask, pool_mask, kv.shape[2])
        return scaled_dot_product_attention(
            q, kv, kv, cache=local, scale=self.scale, mask=mask, sinks=sinks
        )


class Indexer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.index_n_heads
        self.dim = config.index_head_dim
        self.topk = config.index_topk
        self.window = config.sliding_window
        self.wq_b = nn.Linear(
            config.hybrid_mla_q_lora_rank, self.heads * self.dim, bias=False
        )
        self.wk = nn.Linear(config.hidden_size, self.dim, bias=False)
        self.k_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.weights_proj = nn.Linear(config.hidden_size, self.heads, bias=False)
        self.rope = nn.RoPE(
            config.hybrid_mla_qk_rope_head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def _project(self, x, qr, cache):
        B, L, _ = x.shape
        offset = cache.offset if cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset
        q = _linear(self.wq_b, qr).reshape(B, L, self.heads, self.dim)
        q = q.transpose(0, 2, 1, 3)
        k = self.k_norm(_linear(self.wk, x))[:, None]
        q, k = self.rope(q, offset), self.rope(k, offset)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((B, 1, L, 0), dtype=k.dtype))
        weights = (
            _linear(self.weights_proj, x).astype(mx.float32)
            * (self.heads * self.dim) ** -0.5
        )
        return q, k, weights

    def select(self, x, qr, cache):
        q, k, weights = self._project(x, qr, cache)
        length, total = q.shape[2], k.shape[2]
        start = total - length
        # Invariant: row p ranks keys [0, p - window] and keeps p - window + 1..p.
        # - Rows below topk + window keep every older key, so they skip the rank.
        # - Slots that hold the key count select nothing.
        positions = mx.arange(start, total)[:, None]
        slots = mx.arange(self.topk)
        older = mx.where(slots <= positions - self.window, slots, total)
        first = min(max(self.topk + self.window - start, 0), length)
        if first < length:
            ranked = self._rank(q[:, :, first:], k, weights[:, first:], start + first)
            older = mx.concatenate([older[:first], ranked])
        recent = positions - self.window + 1 + mx.arange(self.window)
        rows = mx.concatenate([older, mx.where(recent >= 0, recent, total)], axis=-1)
        return rows.astype(mx.uint32)[None, None]

    def verify_rows(self, x, qr, cache, visible):
        q, k, weights = self._project(x, qr, cache)
        length, total = q.shape[2], k.shape[2]
        start = total - length
        scores = mx.concatenate(
            [
                mx.contiguous(q[:, :, r : r + 1]) @ k.swapaxes(-1, -2)
                for r in range(length)
            ]
        )
        scores = mx.maximum(scores, 0).astype(mx.float32) * weights[0][..., None, None]
        scores = scores.sum(axis=1)
        rows = []
        for r in range(length):
            size = start + r + 1
            if size <= self.topk + self.window:
                ids = mx.arange(size)
            else:
                valid = mx.arange(size) <= size - 1 - self.window
                if visible is not None:
                    valid = valid & visible[:size]
                ranked = mx.where(valid, scores[r, :, :size], -mx.inf)
                top = mx.argpartition(-ranked, kth=self.topk - 1, axis=-1)
                recent = mx.arange(size - self.window, size)
                ids = mx.concatenate([top[0, : self.topk], recent])
            rows.append((ids, None if visible is None else visible[ids]))
        return rows

    def _rank(self, q, keys, weights, start):
        weights = weights.astype(q.dtype)
        rows = q.shape[2]
        step = max_rows_per_call(keys.shape[2])
        picks = []
        for lo in range(0, rows, step):
            hi = min(rows, lo + step)
            scores = self._scores(
                q[:, :, lo:hi], keys, weights[:, lo:hi], start + lo - self.window
            )
            picks.append(index_fast.dsa_topk_indices(scores, self.topk)[0, 0])
            if hi < rows:
                # mlx: lazy groups keep every score buffer alive until the graph runs.
                mx.eval(picks[-1])
        return mx.concatenate(picks).astype(mx.int32)

    @staticmethod
    def _scores(q, keys, weights, before):
        if is_nax_available():
            out = indexer_scores_nax(
                q[0].swapaxes(0, 1), keys[0, 0], weights[0], before, keys.shape[2], 1
            )
            if out is not None:
                return out[None, None]
        return index_fast.dsa_indexer_scores(
            q, keys, weights, causal=True, causal_q_offset=before
        )

    def __call__(self, x, qr, mask, cache=None):
        B, L, _ = x.shape
        q, k, weights = self._project(x, qr, cache)
        S = k.shape[2]
        if S <= self.topk + self.window:
            return mask, None
        # Invariant: cache columns include batch left padding.
        # - The causal diagonal of query row j stays at column S-L+j.
        # - The recent window is always kept; only older keys are ranked.
        delta = mx.arange(S)[None, :] - mx.arange(S - L, S)[:, None]
        recent = (delta <= 0) & (delta > -self.window)
        older = delta <= -self.window
        valid = older if mask is None else older & mask
        scores = mx.maximum(q @ k.swapaxes(-1, -2), 0).astype(mx.float32)
        scores = (scores * weights.transpose(0, 2, 1)[..., None]).sum(
            axis=1, keepdims=True
        )
        scores = mx.where(valid, scores, -mx.inf)
        ids = mx.argpartition(-scores, kth=self.topk - 1, axis=-1)[..., : self.topk]
        sparse = mx.put_along_axis(
            mx.zeros(scores.shape, dtype=mx.bool_), ids, mx.array(True), axis=-1
        )
        sparse = (sparse & older) | recent
        sparse = sparse if mask is None else sparse & mask
        if L == 1:
            recent_ids = mx.broadcast_to(
                mx.arange(S - self.window, S), (B, 1, 1, self.window)
            )
            return sparse, mx.concatenate((ids, recent_ids), axis=-1)
        return sparse, None


class MLAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.hybrid_mla_num_attention_heads
        self.nope = config.hybrid_mla_qk_nope_head_dim
        self.rope_dim = config.hybrid_mla_qk_rope_head_dim
        self.latent = config.hybrid_mla_kv_lora_rank
        self.value_dim = config.hybrid_mla_v_head_dim
        qr = config.hybrid_mla_q_lora_rank
        self.scale = (self.nope + self.rope_dim) ** -0.5
        self.q_a_proj = nn.Linear(config.hidden_size, qr, bias=False)
        self.q_a_layernorm = nn.RMSNorm(qr, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            qr, self.heads * (self.nope + self.rope_dim), bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, self.latent + self.rope_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.latent, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(self.nope, self.latent, self.heads)
        self.unembed_out = MultiLinear(self.latent, self.value_dim, self.heads)
        self.wo_b = nn.Linear(
            self.heads * self.value_dim, config.hidden_size, bias=False
        )
        self.o_gate = nn.Linear(qr, self.heads * self.value_dim, bias=False)
        self.vha = VirtualHeads(self.heads, config.vha_rank)
        self.indexer = Indexer(config)
        self.attn_sink = mx.zeros((self.heads,), dtype=mx.float32)
        self.rope = nn.RoPE(self.rope_dim, traditional=True, base=config.rope_theta)
        self.sparse_shapes = (
            self.heads in (32, 64)
            and self.latent == 512
            and self.rope_dim == 64
            and config.index_n_heads in (16, 32, 64)
            and config.index_head_dim == 128
        )

    def __call__(self, x, cache=None):
        B, L, _ = x.shape
        qr = self.q_a_layernorm(_linear(self.q_a_proj, x))
        q = _linear(self.q_b_proj, qr).reshape(B, L, self.heads, -1)
        q_nope, q_pe = mx.split(q.transpose(0, 2, 1, 3), [self.nope], axis=-1)
        kv, k_pe = mx.split(_linear(self.kv_a_proj_with_mqa, x), [self.latent], axis=-1)
        kv, k_pe = self.kv_a_layernorm(kv)[:, None], k_pe[:, None]
        requests = _request_caches(cache, B, L)
        if requests is None:
            out = self._attend(x, qr, q_nope, q_pe, kv, k_pe, cache)
        else:
            out = self._verify(x, qr, q_nope, q_pe, kv, k_pe, requests)
        out = self.vha(out, 1).reshape(B, L, -1)
        return _linear(self.wo_b, out * mx.sigmoid(_linear(self.o_gate, qr)))

    def rollback(self, cache, kept, rejected):
        for c in cache.caches:
            c.trim(rejected)

    def _verify(self, x, qr, q_nope, q_pe, kv, k_pe, requests):
        B, L, _ = x.shape
        queries = _per_row(self.embed_q, q_nope).transpose(0, 2, 1, 3)
        pe_queries, keys, pe_keys, masks = [], [], [], []
        for b, cache in enumerate(requests):
            latent_cache, index_cache = cache[0], cache[1]
            offset = _offset(latent_cache)
            pe_q = self.rope(q_pe[b : b + 1], offset)
            latent, pe_k = latent_cache.update_and_fetch(
                kv[b : b + 1], self.rope(k_pe[b : b + 1], offset)
            )
            padding = getattr(latent_cache, "left_padding", None)
            visible = (
                None if padding is None else mx.arange(latent.shape[2]) >= padding[0]
            )
            picks = self.indexer.verify_rows(
                x[b : b + 1], qr[b : b + 1], index_cache, visible
            )
            for r, (ids, mask) in enumerate(picks):
                pe_queries.append(pe_q[0, :, r])
                keys.append(mx.take(latent[0, 0], ids, axis=0))
                pe_keys.append(mx.take(pe_k[0, 0], ids, axis=0))
                masks.append(mask)

        def bias(start, stop):
            q = mx.stack(pe_queries[start:stop]) * self.scale
            return rowwise_gemm(q, mx.stack(pe_keys[start:stop]), True)

        out = _rows_attention(
            queries.reshape(B * L, self.heads, -1),
            keys,
            masks,
            self.scale,
            self.attn_sink.astype(q_nope.dtype),
            bias,
        )
        out = out.reshape(B, L, self.heads, -1).transpose(0, 2, 1, 3)
        return _per_row(self.unembed_out, out)

    def _attend(self, x, qr, q_nope, q_pe, kv, k_pe, cache):
        B, L, _ = x.shape
        latent_cache = cache[0] if cache is not None else None
        index_cache = cache[1] if cache is not None else None
        offset = latent_cache.offset if latent_cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset
        mask = create_attention_mask(x, latent_cache, return_array=True)
        q_pe, k_pe = self.rope(q_pe, offset), self.rope(k_pe, offset)
        if latent_cache is not None:
            kv, k_pe = latent_cache.update_and_fetch(kv, k_pe)
        if (
            B == 1
            and L > 1
            and self.sparse_shapes
            and x.dtype in (mx.bfloat16, mx.float16)
            and sparse_prefill_available()
        ):
            rows = self.indexer.select(x, qr, index_cache)
            return self._sparse_attention(q_nope, q_pe, kv, k_pe, rows)
        return self._dense_attention(
            x, qr, q_nope, q_pe, kv, k_pe, mask, latent_cache, index_cache
        )

    def _sparse_attention(self, q_nope, q_pe, kv, k_pe, rows):
        q_latent = self.embed_q(q_nope)
        sinks = self.attn_sink.astype(q_nope.dtype)
        out = sparse_mla_attention_nax(
            q_latent, kv, rows, self.scale, q_pe=q_pe, k_pe=k_pe, sinks=sinks
        )
        if out is None:
            out = attention_fast.sparse_mla_attention(
                q_latent, q_pe, kv, k_pe, rows, self.scale, sinks=sinks
            )
        return self.unembed_out(out)

    def _dense_attention(
        self, x, qr, q_nope, q_pe, kv, k_pe, mask, latent_cache, index_cache
    ):
        decode = x.shape[1] == 1
        mask, indices = self.indexer(x, qr, mask, index_cache)
        if indices is not None:
            take = indices[:, :, 0, :, None]
            kv = mx.take_along_axis(kv, take, axis=2)
            k_pe = mx.take_along_axis(k_pe, take, axis=2)
            mask = mx.take_along_axis(mask, indices, axis=-1)
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(mask, pe_scores, -mx.inf)
        if decode:
            q_nope = self.embed_q(q_nope)
            k = v = kv
        else:
            k = self.embed_q(kv, transpose=False)
            v = self.unembed_out(kv)
        out = scaled_dot_product_attention(
            q_nope,
            k,
            v,
            cache=latent_cache,
            scale=self.scale,
            mask=pe_scores,
            sinks=self.attn_sink.astype(q_nope.dtype),
        )
        return self.unembed_out(out) if decode else out


class DecoderLayer(nn.Module):
    def __init__(self, config, attn, ffn):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(self, x, cache=None):
        h, mixes = self.attn_hc.collapse(x, self.attn_norm)
        x = self.attn_hc.expand(self.attn(h, cache), x, mixes)
        h, mixes = self.ffn_hc.collapse(x, self.ffn_norm)
        return self.ffn_hc.expand(self.ffn(h), x, mixes)


class ErnieModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(
                config,
                (MLAAttention(config) if ratio == -2 else HCAAttention(config, i)),
                (
                    MLP(config, config.intermediate_size)
                    if i < config.first_k_dense_replace
                    else LatentMoE(config)
                ),
            )
            for i, ratio in enumerate(config.compress_ratios)
        ]
        self.hc_head = HyperHead(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, inputs, cache=None, inputs_embeds=None, taps=(), tap_rows=None):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        h = mx.broadcast_to(
            h[:, :, None, :], (*h.shape[:2], self.config.hc_mult, h.shape[-1])
        )
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("Expected one cache per decoder layer")
        submit = cache[0] is not None and h.shape[0] * h.shape[1] <= _DECODE_ROWS
        captured = []
        for i, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            h = layer(h, layer_cache)
            if i in taps:
                captured.append(h[:, -(tap_rows or h.shape[1]) :].mean(axis=2))
            if submit:
                mx.async_eval(h, _states(layer_cache))
        if isinstance(cache[0], list):
            # mlx: an fp32 matmul over B rows rounds differently from B one-row calls.
            h = mx.concatenate([self.hc_head(row[None]) for row in h])
        else:
            h = self.hc_head(h)
        return self.norm(h), captured


class MultiMax(nn.Module):
    def __init__(self):
        super().__init__()
        self.multimax_ranges = mx.zeros((4,), dtype=mx.float32)
        self.multimax_ts = mx.zeros((4,), dtype=mx.float32)

    def __call__(self, logits):
        return _multimax(logits, self.multimax_ranges, self.multimax_ts)


@mx.compile
def _multimax(logits, r, t):
    return (
        logits
        + t[0] * mx.maximum(r[0] - logits, 0)
        + t[1] * mx.maximum(logits - r[1], 0)
        + t[2] * mx.maximum(r[2] - logits, 0) ** 2
        + t[3] * mx.maximum(logits - r[3], 0) ** 2
    )


_RENAMES = {
    ".input_layernorm.": ".attn_norm.",
    ".post_attention_layernorm.": ".ffn_norm.",
    ".block_sparse_moe.": ".ffn.",
    ".mlp.": ".ffn.",
    ".self_attn.": ".attn.",
    ".core_attention.": ".",
    ".linear_q_down_proj.": ".wq_a.",
    ".linear_q_up_proj.": ".wq_b.",
    ".linear_kv_proj.": ".wkv.",
    ".q_layernorm.": ".q_norm.",
    ".kv_layernorm.": ".kv_norm.",
    ".compressor.linear_wkv.": ".compressor.wkv.",
    ".compressor.linear_wgate.": ".compressor.wgate.",
    ".attn.gate_proj.": ".attn.o_gate.",
    ".attn.o_proj.": ".attn.wo_b.",
    ".attn.vha_postmix_U": ".attn.vha.U",
    ".attn.vha_postmix_V": ".attn.vha.V",
    ".attn.softmax_offset": ".attn.attn_sink",
}


class LanguageModel(DSparkHost, nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = ErnieModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.multimax = MultiMax()
        self.configure_mtp(False)

    def __call__(self, inputs=None, cache=None, inputs_embeds=None, **kwargs):
        return_hidden = kwargs.get("return_hidden", False)
        prime = (
            self._omlx_dspark_decode_enabled
            and not return_hidden
            and cache is not None
            and inputs is not None
            and inputs.shape[0] == 1
        )
        taps = self.config.dspark_target_layer_ids if return_hidden or prime else ()
        keep = None if return_hidden else self.config.sliding_window
        h, captured = self.model(inputs, cache, inputs_embeds, taps, keep)
        hidden = mx.concatenate(captured, axis=-1) if captured else None
        if prime:
            self.capture_draft_context(inputs[:, -hidden.shape[1] :], hidden, cache)
        logits = None
        if not kwargs.get("skip_logits"):
            logits = self.multimax(project_logits(h, self.lm_head.weight))
        return LanguageModelOutput(
            logits=logits, hidden_states=[hidden] if return_hidden else None
        )

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                CacheList(KVCache(), KVCache())
                if ratio == -2
                else CacheList(
                    RotatingKVCache(max_size=self.config.sliding_window),
                    PoolingCache(ratio),
                )
            )
            for ratio in self.config.compress_ratios
        ]

    def sanitize(self, weights):
        """Map release tensor names to module paths; FP8 is already unpacked."""
        out = {}
        for key, value in weights.items():
            key = re.sub(r"^model\.mtp\.(\d+)\.", r"dspark.layers.\1.", key)
            key = key.replace("model.mtp.", "dspark.")
            if (
                key.endswith(".hc_head_fn")
                or key == "dspark.markov_head.markov_w1.weight"
            ):
                value = value.T
            key = key.replace("model.lm_head.", "multimax.")
            key = key.replace("model.mhc_contract.hc_head_", "model.hc_head.")
            key = key.replace("dspark.mhc_contract.hc_head_", "dspark.mhc_contract.")
            for src, dst in (("self_attention", "attn"), ("mlp", "ffn")):
                prefix = f".{src}_hyper_connection."
                if prefix in key:
                    key = key.replace(prefix, f".{dst}_hc.")
                    if key.endswith("mapping_proj.weight"):
                        key = key.replace("mapping_proj.weight", "fn")
                        value = mx.contiguous(value.T)
                    key = key.replace(f".{dst}_hc.bias", f".{dst}_hc.base")
            for src, dst in _RENAMES.items():
                key = key.replace(src, dst)
            stem, group_proj, suffix = key.partition(".attn.linear_o_group_proj")
            if group_proj:
                key = f"{stem}.attn.wo_a{suffix or '.weight'}"
                value = value.reshape(self.config.o_groups, self.config.o_lora_rank, -1)
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                key = key.replace(f".shared_experts.{src}.", f".shared_experts.{dst}.")
                key = key.replace(f".experts.{src}.", f".switch_mlp.{dst}.")
            out[key] = value

        stages = len(self.config.dspark_target_layer_ids)
        prefixes = [f"model.layers.{i}" for i in range(len(self.layers))]
        for prefix in prefixes + [f"dspark.layers.{i}" for i in range(stages)]:
            for kind in ("attn", "ffn"):
                keys = [f"{prefix}.{kind}_hc.alpha_{p}" for p in ("pre", "post", "res")]
                if keys[0] in out:
                    out[f"{prefix}.{kind}_hc.scale"] = mx.concatenate(
                        [out.pop(k).reshape(-1) for k in keys]
                    )
        for i, ratio in enumerate(self.config.compress_ratios):
            prefix = f"model.layers.{i}"
            key = f"{prefix}.attn.kv_b_proj.weight"
            if ratio == -2 and key in out:
                attn = self.layers[i].attn
                w = out.pop(key).reshape(
                    attn.heads, attn.nope + attn.value_dim, attn.latent
                )
                k, v = mx.split(w, [attn.nope], axis=1)
                out[f"{prefix}.attn.embed_q.weight"] = k.swapaxes(-1, -2)
                out[f"{prefix}.attn.unembed_out.weight"] = v
        return out
