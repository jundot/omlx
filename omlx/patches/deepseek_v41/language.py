# SPDX-License-Identifier: MIT
"""MLX reference port of DeepSeek V4.1 CSA2 and single-pass mHC.

The official inference code is the numerical specification. Unlike its global
shared_attn and module buffers, all persistent state belongs to request caches.
Batch rows are evaluated independently so late admission cannot share history.
"""

import math
from contextlib import nullcontext

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchLinear

from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

from ..deepseek_v4.switch_layers import _block_config, _build_mxfp4_blocks
from .activation import quantize_paired_swiglu_activation, quantize_swiglu_activation
from .cache import DeepseekV41Cache
from .engram import Engram, NgramHash, build_compressed_token_map
from .head import project_logits
from .hyper_connection import (
    fused_hc_post,
    fused_hc_pre_norm,
    fused_hc_projection,
    sinkhorn,
)
from .kernels import packed_index_scores, packed_index_topk, packed_sparse_attention
from .mtp import DSparkMixin
from .quantization import QuantizedProjection, pack_activation, quantize_activation
from .routing import combine_sorted_experts


@mx.compile
def norm(x, weight, eps):
    f = x.astype(mx.float32)
    return (f * mx.rsqrt(mx.mean(f * f, -1, keepdims=True) + eps) * weight).astype(
        x.dtype
    )


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return norm(x, self.weight, self.eps)


@mx.compile
def _rope(x, positions, params, compressed, inverse):
    d, base, original_seq_len, beta_fast, beta_slow, rope_factor = params
    freq = 1 / mx.power(base, mx.arange(0, d, 2).astype(mx.float32) / d)
    if compressed and original_seq_len:

        def correction(rotations):
            return (
                d
                * math.log(original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction(beta_fast)), 0)
        high = min(math.ceil(correction(beta_slow)), d - 1)
        smooth = 1 - mx.clip((mx.arange(d // 2) - low) / max(high - low, 1e-3), 0, 1)
        freq = freq / rope_factor * (1 - smooth) + freq * smooth
    angles = positions.astype(mx.float32)[:, None] * freq
    angles = angles.reshape(1, len(positions), *([1] * (x.ndim - 3)), d // 2)
    if inverse:
        angles = -angles
    tail = x[..., -d:].astype(mx.float32).reshape(*x.shape[:-1], d // 2, 2)
    a, b = tail[..., 0], tail[..., 1]
    rotated = mx.stack(
        [
            a * mx.cos(angles) - b * mx.sin(angles),
            a * mx.sin(angles) + b * mx.cos(angles),
        ],
        -1,
    )
    return mx.concatenate(
        [x[..., :-d], rotated.reshape(*x.shape[:-1], d).astype(x.dtype)], -1
    )


def rope(x, positions, config, compressed, inverse=False):
    params = (
        config.rope_head_dim,
        config.compress_rope_theta if compressed else config.rope_theta,
        config.original_seq_len,
        config.beta_fast,
        config.beta_slow,
        config.rope_factor,
    )
    return _rope(x, positions, params, compressed, inverse)


def candidate_block_ids(scores, lengths, count, block):
    width = scores.shape[-1]
    if not width:
        return mx.zeros((*scores.shape[:-1], 0), dtype=mx.int32)
    padded = mx.pad(
        scores,
        [(0, 0)] * (scores.ndim - 1) + [(0, -width % block)],
        constant_values=-float("inf"),
    )
    grouped = padded.reshape(*scores.shape[:-1], -1, block).max(-1)
    n = grouped.shape[-1]
    grouped = mx.where(
        (lengths > 0) & (mx.arange(n) == (lengths - 1) // block), float("inf"), grouped
    )
    chosen = mx.argsort(-grouped, axis=-1)[..., : min(count, n)]
    chosen = mx.sort(chosen, axis=-1).astype(mx.int32)
    valid = mx.take_along_axis(grouped, chosen, -1) > -float("inf")
    return mx.where(valid, chosen, -1)


def sparse_attention(q, selected, indices, sink, scale):
    scores = (
        mx.einsum("bshd,bskd->bshk", q.astype(mx.float32), selected.astype(mx.float32))
        * scale
    )
    scores = mx.where(indices[:, :, None, :] >= 0, scores, -float("inf"))
    sinks = mx.broadcast_to(sink[None, None, :, None], (*scores.shape[:-1], 1))
    weights = mx.softmax(mx.concatenate([scores, sinks], -1), axis=-1)[..., :-1]
    return mx.einsum("bshk,bskd->bshd", weights, selected.astype(mx.float32)).astype(
        q.dtype
    )


class Compressor(nn.Module):
    def __init__(self, c, ratio):
        super().__init__()
        self.wkv = nn.Linear(c.dim, c.head_dim, bias=False)
        self.norm = RMSNorm(c.head_dim, c.norm_eps)
        self.ratio = ratio
        if ratio > 1:
            self.wgate = nn.Linear(c.dim, c.head_dim, bias=False)

    def __call__(self, x, cache, start):
        r = self.ratio
        if r == 1:
            return self.norm(self.wkv(x))
        kv, gate = self.wkv(x.astype(mx.float32)), self.wgate(x.astype(mx.float32))
        rem = start % r
        if rem:
            kv = mx.concatenate([cache[4][:, :rem], kv], 1)
            gate = mx.concatenate([cache[5][:, :rem], gate], 1)
        verify_state = getattr(cache, "_mtp_verify_state", None)
        if verify_state is not None:
            verify_state["compressor"] = (kv, gate)
        cutoff = kv.shape[1] // r * r
        cache[4], cache[5] = kv[:, cutoff:], gate[:, cutoff:]
        if not cutoff:
            return None
        pooled = mx.sum(
            kv[:, :cutoff].reshape(1, -1, r, kv.shape[-1])
            * mx.softmax(gate[:, :cutoff].reshape(1, -1, r, gate.shape[-1]), axis=2),
            axis=2,
        )
        return self.norm(pooled.astype(x.dtype))


class Indexer(nn.Module):
    def __init__(self, c, layer):
        super().__init__()
        self.wq_b = nn.Linear(
            c.q_lora_rank, c.index_n_heads * c.index_head_dim, bias=False
        )
        self.weights_proj = nn.Linear(c.dim, c.index_n_heads, bias=False)
        if layer in c.kv_source_layers:
            self.wk = nn.Linear(c.head_dim, c.index_head_dim, bias=False)
            self.k_norm = RMSNorm(c.index_head_dim, c.norm_eps)
        self._config, self._layer = c, layer

    def __call__(self, x, qr, latent, cache, shared, start, ratio):
        c, layer = self._config, self._layer
        end = start + x.shape[1]
        if layer in c.kv_source_layers:
            previous = cache[3]
            previous = (
                previous[:, : start // ratio]
                if previous is not None
                else pack_activation(
                    mx.zeros((1, 0, c.index_head_dim), x.dtype), bits=4
                )
            )
            if latent is not None:
                pos = mx.arange(start // ratio, end // ratio) * ratio
                key = pack_activation(
                    rope(self.k_norm(self.wk(latent)), pos, c, True), bits=4
                )
                previous = mx.concatenate([previous, key], 1)
            shared["index_k"] = cache[3] = previous
        key = shared["index_k"]
        q = self.wq_b(qr).reshape(1, x.shape[1], c.index_n_heads, c.index_head_dim)
        q = quantize_activation(rope(q, mx.arange(start, end), c, True), bits=4)
        weights = self.weights_proj(x).astype(mx.float32) * (
            c.index_head_dim**-0.5 * c.index_n_heads**-0.5
        )
        candidates = None
        if 0 <= c.candidate_source_layer < layer:
            blocks = shared["candidates"]
            candidates = blocks[..., None] * c.candidate_block_size + mx.arange(
                c.candidate_block_size
            )
            candidates = mx.where(blocks[..., None] >= 0, candidates, -1).reshape(
                1, x.shape[1], blocks.shape[-1] * c.candidate_block_size
            )
        if candidates is None:
            idx, blocks = packed_index_topk(
                q,
                key,
                weights,
                start,
                ratio,
                c.index_topk,
                block_count=(
                    c.candidate_topk_blocks if layer == c.candidate_source_layer else 0
                ),
                block_size=c.candidate_block_size,
            )
            if layer == c.candidate_source_layer:
                shared["candidates"] = blocks
            return idx
        scores = packed_index_scores(q, key, weights, start, ratio, candidates)
        count = min(c.index_topk, scores.shape[-1])
        order = mx.argsort(-scores, axis=-1)[..., :count].astype(mx.int32)
        valid = mx.take_along_axis(scores, order, axis=-1) > -float("inf")
        idx = mx.take_along_axis(candidates, order, -1)
        # Keep chronological order; invalid slots are masked by attention.
        return mx.sort(mx.where(valid, idx, -1), axis=-1)


class Attention(nn.Module):
    def __init__(self, c, layer):
        super().__init__()
        self._config, self._layer = c, layer
        self.attn_sink = mx.zeros((c.n_heads,), mx.float32)
        self.wq_a = nn.Linear(c.dim, c.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(c.q_lora_rank, c.norm_eps)
        self.wq_b = nn.Linear(c.q_lora_rank, c.n_heads * c.head_dim, bias=False)
        self.wkv = nn.Linear(c.dim, c.head_dim, bias=False)
        self.kv_norm = RMSNorm(c.head_dim, c.norm_eps)
        self.wo_a = nn.Linear(
            c.n_heads * c.head_dim // c.o_groups, c.o_groups * c.o_lora_rank, bias=False
        )
        self.wo_b = nn.Linear(c.o_groups * c.o_lora_rank, c.dim, bias=False)
        if layer in c.kv_source_layers:
            self.compressor = Compressor(c, c.compress_ratios[layer])
        if layer in c.index_source_layers:
            self.indexer = Indexer(c, layer)

    def _input_projections(self, x):
        if all(
            isinstance(p, QuantizedProjection) and p.quantize_input
            for p in (self.wq_a, self.wkv)
        ):
            quantized = quantize_activation(x)
            return (
                self.wq_a.project_quantized(quantized),
                self.wkv.project_quantized(quantized),
            )
        return self.wq_a(x), self.wkv(x)

    def __call__(self, x, cache, shared, start):
        c, layer = self._config, self._layer
        ratio, length = c.compress_ratios[layer], x.shape[1]
        positions = mx.arange(start, start + length)
        query, kv_input = self._input_projections(x)
        qr = self.q_norm(query)
        q = rope(
            self.wq_b(qr).reshape(1, length, c.n_heads, c.head_dim),
            positions,
            c,
            bool(ratio),
        )
        new = pack_activation(rope(self.kv_norm(kv_input), positions, c, bool(ratio)))
        old = cache[1]
        old_len = min(start, c.window_size)
        kv = mx.concatenate([old[:, :old_len], new], 1) if old_len else new
        verify_state = getattr(cache, "_mtp_verify_state", None)
        if verify_state is not None:
            verify_state["window"] = kv
        cache[1] = kv[:, -c.window_size :]
        # List only the causal local window, including the current token.
        local = positions[:, None] - c.window_size + 1 + mx.arange(c.window_size)
        valid = (local >= max(0, start - old_len)) & (local <= positions[:, None])
        idx = mx.where(valid, local - (start - old_len), -1)[None]
        pooled = mx.zeros((1, 0, c.head_dim // 2 + c.head_dim // 16), mx.uint8)
        ci = mx.zeros((1, length, 0), mx.int32)
        if ratio:
            latent = None
            if layer in c.kv_source_layers:
                latent = self.compressor(x, cache, start)
                if cache[2] is None:
                    cache[2] = pack_activation(
                        mx.zeros((1, 0, c.head_dim), x.dtype),
                        bits=4,
                        group_size=16,
                        e4m3_scale=True,
                    )
                shared["kv"] = cache[2][:, : start // ratio]
            if layer in c.index_source_layers:
                shared["idx"] = self.indexer(x, qr, latent, cache, shared, start, ratio)
            if latent is not None:
                compressed = rope(
                    latent,
                    mx.arange(start // ratio, (start + length) // ratio) * ratio,
                    c,
                    True,
                )
                compressed = pack_activation(
                    compressed, bits=4, group_size=16, e4m3_scale=True
                )
                shared["kv"] = cache[2] = mx.concatenate([shared["kv"], compressed], 1)
            ci, pooled = shared["idx"], shared["kv"]
        if (
            length > 8
            and c.n_heads == 64
            and c.head_dim == 512
            and q.dtype == mx.bfloat16
            and glm_fast.has_symbol("deepseek_v41_packed_attention")
        ):
            out = glm_fast.deepseek_v41_packed_attention(
                q.transpose(0, 2, 1, 3),
                kv[:, None],
                pooled,
                ci[:, None].astype(mx.uint32),
                self.attn_sink,
                c.head_dim**-0.5,
                start,
                max(ratio, 1),
                c.window_size,
            ).transpose(0, 2, 1, 3)
        else:
            out = packed_sparse_attention(
                q, kv, pooled, idx, ci, self.attn_sink, c.head_dim**-0.5
            )
        out = rope(out, positions, c, bool(ratio), inverse=True)
        grouped = out.reshape(1, length, c.o_groups, -1)
        weight = self.wo_a.weight.reshape(c.o_groups, c.o_lora_rank, -1)
        return self.wo_b(mx.einsum("bsgd,grd->bsgr", grouped, weight).flatten(-2))


class Expert(nn.Module):
    def __init__(self, c, switched=False):
        super().__init__()

        def linear(a, b):
            return (
                SwitchLinear(a, b, c.n_routed_experts, bias=False)
                if switched
                else nn.Linear(a, b, bias=False)
            )

        self.w1, self.w3 = (
            linear(c.dim, c.moe_inter_dim),
            linear(c.dim, c.moe_inter_dim),
        )
        self.w2 = linear(c.moe_inter_dim, c.dim)
        self._limit = c.swiglu_limit

    @property
    def quantizes_input(self):
        return all(
            isinstance(p, QuantizedProjection) and p.quantize_input
            for p in (self.w1, self.w3)
        )

    def __call__(
        self,
        x,
        indices=None,
        weights=None,
        sorted_indices=False,
        *,
        input_quantized=False,
    ):
        if input_quantized and not self.quantizes_input:
            raise ValueError(
                "Prequantized input requires matching quantized projections"
            )
        projections = (self.w1, self.w3, self.w2)
        if (
            indices is not None
            and weights is not None
            and not sorted_indices
            and x.dtype == mx.bfloat16
            and mx.default_device() != mx.cpu
            and all(
                isinstance(p, QuantizedProjection)
                and (
                    (p.mode in ("mxfp4", "mxfp8") and p.group_size == 32)
                    or (
                        p.mode == "affine"
                        and p.bits in (2, 3, 4, 6, 8)
                        and p.get("biases") is not None
                    )
                )
                and p.quantize_input
                for p in projections
            )
            and 1 <= x.size // self.w1.input_dims <= 8
            and x.size // self.w1.input_dims == indices.size // indices.shape[-1]
            and indices.size > 0
            and weights.size == indices.size
            and glm_fast.has_symbol("deepseek_v41_grouped_expert")
        ):
            # Short DSpark verification blocks share the decode execution
            # pipeline, retaining a distinct quantized input for each token.
            inp = (x if input_quantized else quantize_activation(x)).reshape(
                -1, 1, self.w1.input_dims
            )
            ids = indices.reshape(-1).astype(mx.uint32)
            lhs = mx.arange(ids.size, dtype=mx.uint32) // indices.shape[-1]

            def gather(value, projection, left):
                return mx.gather_qmm(
                    value,
                    projection.weight,
                    projection.scales,
                    projection.get("biases"),
                    lhs_indices=left,
                    rhs_indices=ids,
                    transpose=True,
                    group_size=projection.group_size,
                    bits=projection.bits,
                    mode=projection.mode,
                )

            # Keep these nodes unevaluated for one native execution pipeline.
            gate = gather(inp, self.w1, lhs)
            up = gather(inp, self.w3, lhs)
            y = quantize_swiglu_activation(
                gate, up, weights.reshape(-1), x.dtype, self._limit
            )
            down = gather(y, self.w2, mx.arange(ids.size, dtype=mx.uint32))
            return glm_fast.deepseek_v41_grouped_expert(gate, up, y, down).reshape(
                *indices.shape, 1, x.shape[-1]
            )
        block_plan = None
        block_kind = None
        if sorted_indices:
            kinds = [
                (
                    p._native_block_kind(x, True)
                    if isinstance(p, QuantizedProjection)
                    else None
                )
                for p in projections
            ]
            if kinds[0] is not None and all(kind == kinds[0] for kind in kinds):
                block_kind = kinds[0]
                bm, variant = _block_config(indices.size, block_kind)
                meta, count = _build_mxfp4_blocks(indices, self.w1.num_experts, bm)
                block_plan = (meta, count, variant)

        def project(module, value):
            if indices is None:
                return module(value)
            if isinstance(module, QuantizedProjection):
                return module(value, indices, sorted_indices, block_plan)
            return module(value, indices, sorted_indices=sorted_indices)

        dtype = x.dtype
        pair = None
        if (
            block_plan is not None
            and block_kind == "mxfp4"
            and self.w1.quantize_input == self.w3.quantize_input
            and self.w1.output_dims == self.w3.output_dims
            and glm_fast.has_symbol("deepseek_mxfp4_gather_qmm_pair_concat_blocks")
        ):
            inp = (
                quantize_activation(x)
                if self.w1.quantize_input and not input_quantized
                else x
            )
            pair = glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
                inp,
                self.w1.weight,
                self.w1.scales,
                self.w3.weight,
                self.w3.scales,
                *block_plan,
            )
            gate = pair[..., : self.w1.output_dims]
            up = pair[..., self.w1.output_dims :]
        elif (
            block_kind == "affine"
            and self.w1.bits == self.w3.bits
            and self.w1.group_size == self.w3.group_size
            and self.w1.quantize_input == self.w3.quantize_input
            and self.w1.output_dims == self.w3.output_dims
            and glm_fast.has_symbol("deepseek_affine_gather_qmm_pair_concat_blocks")
        ):
            inp = (
                quantize_activation(x)
                if self.w1.quantize_input and not input_quantized
                else x
            )
            meta, count, variant = block_plan
            pair = glm_fast.deepseek_affine_gather_qmm_pair_concat_blocks(
                inp,
                self.w1.weight,
                self.w1.scales,
                self.w1.biases,
                self.w3.weight,
                self.w3.scales,
                self.w3.biases,
                meta,
                count,
                self.w1.group_size,
                self.w1.bits,
                variant,
            )
            gate = pair[..., : self.w1.output_dims]
            up = pair[..., self.w1.output_dims :]
        elif (
            isinstance(self.w1, QuantizedProjection)
            and isinstance(self.w3, QuantizedProjection)
            and self.w1.quantize_input == self.w3.quantize_input
        ):
            inp = (
                quantize_activation(x)
                if self.w1.quantize_input and not input_quantized
                else x
            )
            gate = self.w1.project_quantized(inp, indices, sorted_indices, block_plan)
            up = self.w3.project_quantized(inp, indices, sorted_indices, block_plan)
        else:
            gate = project(self.w1, x)
            up = project(self.w3, x)
        if (
            isinstance(self.w2, QuantizedProjection)
            and self.w2.quantize_input
            and gate.size
            and gate.shape == up.shape
            and gate.shape[-1] % 32 == 0
            and dtype in (mx.float32, mx.float16, mx.bfloat16)
            and mx.default_device() != mx.cpu
            and (weights is None or weights.size == gate.size // gate.shape[-1])
        ):
            y = (
                quantize_paired_swiglu_activation(pair, weights, dtype, self._limit)
                if pair is not None
                else quantize_swiglu_activation(gate, up, weights, dtype, self._limit)
            )
            return self.w2.project_quantized(y, indices, sorted_indices, block_plan)
        gate, up = gate.astype(mx.float32), up.astype(mx.float32)
        if self._limit:
            gate = mx.minimum(gate, self._limit)
            up = mx.clip(up, -self._limit, self._limit)
        y = nn.silu(gate) * up
        if weights is not None:
            y *= weights[..., None, None]
        return project(self.w2, y.astype(dtype))


class Gate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = mx.zeros((c.n_routed_experts, c.dim))
        self.bias = mx.zeros((c.n_routed_experts,))
        if c.vision_enabled:
            self.bias_vl = mx.zeros((c.n_routed_experts,))
        self._config = c

    def __call__(self, x, image_mask):
        c = self._config
        raw = (x.astype(mx.float32) @ self.weight.astype(mx.float32).T) / c.gate_temp
        scores = (
            mx.softmax(raw, -1)
            if c.score_func == "softmax"
            else (
                mx.sigmoid(raw)
                if c.score_func == "sigmoid"
                else mx.sqrt(mx.logaddexp(raw, 0))
            )
        )
        bias = self.bias
        if image_mask is not None and "bias_vl" in self:
            bias = mx.where(image_mask[..., None], self.bias_vl, bias)
        idx = mx.argsort(-(scores + bias), axis=-1)[..., : c.n_activated_experts]
        weights = mx.take_along_axis(scores, idx, -1)
        if c.norm_topk_prob and c.n_activated_experts > 1:
            weights /= mx.sum(weights, -1, keepdims=True) + 1e-20
        return idx, weights * c.route_scale


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate, self.experts, self.shared_experts = (
            Gate(c),
            Expert(c, True),
            Expert(c),
        )

    def __call__(self, x, image_mask):
        idx, weights = self.gate(x, image_mask)
        routed_quantized = self.experts.quantizes_input
        shared_quantized = self.shared_experts.quantizes_input
        # Quantization is row-local; reuse it before routing duplicates token rows.
        quantized = (
            quantize_activation(x) if routed_quantized or shared_quantized else x
        )
        routed_input = quantized if routed_quantized else x
        if x.shape[1] >= 32 and idx.size >= 64:
            shape = idx.shape
            flat = idx.reshape(-1)
            order = mx.argsort(flat)
            inverse = mx.argsort(order)
            selected = routed_input.reshape(-1, x.shape[-1])[order // shape[-1]][
                :, None, :
            ]
            routed = self.experts(
                selected,
                flat[order],
                weights.reshape(-1)[order],
                sorted_indices=True,
                input_quantized=routed_quantized,
            )
            shared = self.shared_experts(
                quantized if shared_quantized else x, input_quantized=shared_quantized
            )
            combined = combine_sorted_experts(routed, inverse, shared)
            if combined is not None:
                return combined
            routed = routed[inverse].reshape(*shape, x.shape[-1])
        else:
            routed = self.experts(
                routed_input[..., None, None, :],
                idx,
                weights,
                input_quantized=routed_quantized,
            ).squeeze(-2)
            shared = self.shared_experts(
                quantized if shared_quantized else x, input_quantized=shared_quantized
            )
        return (routed.astype(mx.float32).sum(-2) + shared.astype(mx.float32)).astype(
            x.dtype
        )


@mx.compile
def _hc_mix_weights(mixes, scale, base, n, hc_eps, iters):
    pre = mx.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + hc_eps
    post = 2 * mx.sigmoid(mixes[..., n : 2 * n] * scale[1] + base[n : 2 * n])
    comb = (mixes[..., 2 * n :] * scale[2] + base[2 * n :]).reshape(
        *mixes.shape[:-1], n, n
    )
    comb = mx.softmax(comb, -1) + hc_eps
    return pre, post, sinkhorn(comb, hc_eps, iters)


@mx.compile
def _hc_mixes(x, fn, scale, base, n, norm_eps, hc_eps, iters):
    flat = x.flatten(-2).astype(mx.float32)
    mixes = (flat @ fn.T) * mx.rsqrt(mx.mean(flat * flat, -1, keepdims=True) + norm_eps)
    return _hc_mix_weights(mixes, scale, base, n, hc_eps, iters)


def hc_mixes(x, fn, scale, base, c):
    if (
        x.ndim == 4
        and x.shape[1] >= 256
        and x.shape[-2] == c.hc_mult == 4
        and x.shape[-1] > 0
        and x.dtype == mx.bfloat16
        and fn.dtype == mx.float32
        and fn.shape == (24, 4 * x.shape[-1])
        and mx.default_device() == mx.gpu
    ):
        mixes = fused_hc_projection(x, fn, c.norm_eps)
        return _hc_mix_weights(
            mixes, scale, base, c.hc_mult, c.hc_eps, c.hc_sinkhorn_iters
        )
    return _hc_mixes(
        x, fn, scale, base, c.hc_mult, c.norm_eps, c.hc_eps, c.hc_sinkhorn_iters
    )


@mx.compile
def hc_pre(x, pre):
    return mx.sum(x.astype(mx.float32) * pre[..., None], axis=-2).astype(x.dtype)


def hc_pre_norm(x, pre, weight, eps):
    if (
        x.ndim == 4
        and x.size
        and x.shape[-2] == 4
        and x.shape[-1] <= 8192
        and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and pre.dtype == mx.float32
        and pre.shape == x.shape[:-1]
        and weight.shape == (x.shape[-1],)
        and weight.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and mx.default_device() == mx.gpu
    ):
        return fused_hc_pre_norm(x, pre, weight, eps)
    return norm(hc_pre(x, pre), weight, eps)


@mx.compile
def _hc_post_reference(x, residual, post, comb):
    return (
        post[..., None] * x[..., None, :]
        + mx.einsum("bsij,bsid->bsjd", comb, residual.astype(mx.float32))
    ).astype(x.dtype)


def hc_post(x, residual, post, comb):
    if (
        x.ndim == 3
        and x.size
        and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and residual.dtype == x.dtype
        and residual.shape == (*x.shape[:-1], 4, x.shape[-1])
        and post.shape == (*x.shape[:-1], 4)
        and comb.shape == (*x.shape[:-1], 4, 4)
        and post.dtype == mx.float32
        and comb.dtype == mx.float32
        and mx.default_device() == mx.gpu
    ):
        return fused_hc_post(x, residual, post, comb)
    return _hc_post_reference(x, residual, post, comb)


class Block(nn.Module):
    def __init__(self, c, layer):
        super().__init__()
        self._config = c
        self.attn, self.ffn = Attention(c, layer), MoE(c)
        self.attn_norm, self.ffn_norm = (
            RMSNorm(c.dim, c.norm_eps),
            RMSNorm(c.dim, c.norm_eps),
        )
        size = (2 + c.hc_mult) * c.hc_mult
        for kind in ("attn", "ffn"):
            self[f"hc_{kind}_fn"] = mx.zeros((size, c.hc_mult * c.dim))
            self[f"hc_{kind}_base"] = mx.zeros((size,))
            self[f"hc_{kind}_scale"] = mx.ones((3,))
        if layer in c.engram_layer_ids:
            self.engram = Engram(c, list(c.engram_layer_ids).index(layer))

    def __call__(self, h, pre, cache, shared, start, image_mask):
        ap, ao, ac = hc_mixes(
            h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self._config
        )
        h = hc_post(
            self.attn(
                hc_pre_norm(h, pre, self.attn_norm.weight, self.attn_norm.eps),
                cache,
                shared,
                start,
            ),
            h,
            ao,
            ac,
        )
        fp, fo, fc = hc_mixes(
            h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, self._config
        )
        h = hc_post(
            self.ffn(
                hc_pre_norm(h, ap, self.ffn_norm.weight, self.ffn_norm.eps), image_mask
            ),
            h,
            fo,
            fc,
        )
        return h, fp


class LanguageModel(DSparkMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        config.validate()
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [Block(config, i) for i in range(config.n_layers)]
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.preserve_mtp:
            from .dspark import make_stages

            self.mtp = make_stages(config)
        self._hasher = None
        from ..mlx_lm_mtp import get_mtp_depth, is_mtp_active

        self.configure_mtp(config.preserve_mtp and is_mtp_active(), get_mtp_depth())

    def make_dspark_cache(self):
        from .dspark import make_cache

        return make_cache(self._config)

    def forward_spec(self, input_ids, main_hidden, cache, *, temperature=0.0):
        from .dspark import forward_spec

        return forward_spec(
            self, input_ids, main_hidden, cache, temperature=temperature
        )

    def set_tokenizer(self, tokenizer):
        mapping, _ = build_compressed_token_map(tokenizer)
        self.set_token_map(mapping)

    def set_token_map(self, token_map):
        if self._config.engram_layer_ids:
            self._hasher = NgramHash(self._config, token_map)

    def make_cache(self):
        return [
            DeepseekV41Cache(
                self._config.compress_ratios[i]
                if i in self._config.kv_source_layers
                else 0
            )
            for i in range(len(self.layers))
        ]

    def _forward(
        self, input_ids, cache=None, inputs_embeds=None, token_types=None, **kwargs
    ):
        c = self._config
        cache = self.make_cache() if cache is None else cache
        if len(cache) != len(self.layers):
            raise ValueError("DeepSeek V4.1 cache layer count mismatch")
        for i, item in enumerate(cache):
            ratio = c.compress_ratios[i] if i in c.kv_source_layers else 0
            if item.compress_ratio is None:
                item.compress_ratio = ratio  # Upgrade a legacy full snapshot.
            elif item.compress_ratio != ratio:
                raise ValueError("DeepSeek V4.1 cache compression layout mismatch")
        if c.engram_layer_ids and self._hasher is None:
            raise ValueError(
                "DeepSeek V4.1 requires its tokenizer-derived Engram token map"
            )
        masks = cache[0].make_mask(input_ids.shape[1])
        capture = bool(kwargs.get("return_dspark_hidden", False))
        if capture and input_ids.shape[0] != 1:
            raise ValueError(
                "DSpark target capture currently requires one unpadded row"
            )
        captured = {}
        results, rows = [], [[] for _ in self.layers]
        for row in range(input_ids.shape[0]):
            valid = (
                np.ones(input_ids.shape[1], bool)
                if masks is None
                else np.asarray(masks[row])
            )
            positions = np.flatnonzero(valid)
            if not len(positions):
                results.append(mx.zeros((1, input_ids.shape[1], c.vocab_size)))
                for i, item in enumerate(cache):
                    rows[i].append(item.extract(row))
                continue
            begin, end = int(positions[0]), int(positions[-1]) + 1
            if capture and (begin != 0 or end != input_ids.shape[1]):
                raise ValueError("DSpark target capture requires an unpadded row")
            if end - begin != len(positions):
                raise ValueError("Prefill padding must be contiguous")
            rc = [item.extract(row) for item in cache]
            verify_states = kwargs.get("mtp_verify_states")
            if verify_states is not None:
                for item, state in zip(rc, verify_states):
                    item._mtp_verify_state = state
            start = rc[0].size()
            if any(item.size() != start for item in rc):
                raise ValueError("DeepSeek V4.1 cache offsets diverged across layers")
            ids = input_ids[row : row + 1, begin:end]
            image_mask = ids == c.image_token_id if inputs_embeds is not None else None
            h = (
                self.embed(ids)
                if inputs_embeds is None
                else inputs_embeds[row : row + 1, begin:end]
            )
            hashes, history = (None, None)
            if self._hasher is not None:
                hashes, history = self._hasher(ids, rc[0][6], image_mask)
            h = mx.repeat(h[..., None, :], c.hc_mult, -2)
            pre = mx.broadcast_to(
                (mx.arange(c.hc_mult) == 0).astype(mx.float32), h.shape[:-1]
            )
            shared = {}
            prefetch = getattr(self, "_engram_prefetch", None)
            with prefetch.forward() if prefetch is not None else nullcontext():
                if prefetch is not None and c.engram_layer_ids:
                    first = self.layers[c.engram_layer_ids[0]].engram.embed
                    prefetch.submit(first, hashes[:, :, 0])
                for i, layer in enumerate(self.layers):
                    if "engram" in layer:
                        ix = list(c.engram_layer_ids).index(i)
                        if prefetch is not None:
                            mx.async_eval(h, pre)
                        h = layer.engram(h, hashes[:, :, ix], image_mask)
                        if prefetch is not None and ix + 1 < len(c.engram_layer_ids):
                            next_layer = self.layers[c.engram_layer_ids[ix + 1]]
                            prefetch.submit(
                                next_layer.engram.embed, hashes[:, :, ix + 1]
                            )
                    if capture and i in c.dspark_target_layer_ids:
                        captured[i] = mx.mean(h, axis=2)
                    h, pre = layer(h, pre, rc[i], shared, start, image_mask)
                    if prefetch is not None and "engram" in layer:
                        mx.async_eval(h, pre)
                    rc[i][0] = mx.array([start + end - begin], mx.int32)
                    if history is not None and i == 0:
                        rc[i][6] = mx.array(history, mx.int64)
                    for slot in range(1, 7):
                        if rc[i][slot] is None:
                            width = c.index_head_dim if slot == 3 else c.head_dim
                            empty = mx.zeros((1, 0, width), h.dtype)
                            if slot == 1:
                                empty = pack_activation(empty)
                            elif slot == 2:
                                empty = pack_activation(empty, 4, 16, True)
                            elif slot == 3:
                                empty = pack_activation(empty, 4)
                            rc[i][slot] = (
                                mx.zeros((1, 0), mx.int64) if slot == 6 else empty
                            )
                    rows[i].append(rc[i])
            logits = project_logits(self.norm(hc_pre(h, pre)), self.head.weight)
            results.append(
                mx.pad(logits, [(0, 0), (begin, input_ids.shape[1] - end), (0, 0)])
            )
        for i, item in enumerate(cache):
            merged = DeepseekV41Cache.merge(rows[i])
            item.cache = merged.cache
            item.advance(input_ids.shape[1])
        logits = mx.concatenate(results, 0)
        if capture:
            if not c.dspark_target_layer_ids or set(captured) != set(
                c.dspark_target_layer_ids
            ):
                raise ValueError("Incomplete DSpark target hidden capture")
            return logits, mx.concatenate(
                [captured[i] for i in c.dspark_target_layer_ids], axis=-1
            )
        return logits
