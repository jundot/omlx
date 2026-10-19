# SPDX-License-Identifier: MIT
"""MLX reference port of DeepSeek V4.1 CSA2 and single-pass mHC.

The official inference code is the numerical specification. Unlike its global
shared_attn and module buffers, all persistent state belongs to request caches.
Batch rows are evaluated independently so late admission cannot share history.
"""

import math
from contextlib import nullcontext
from functools import cache, lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchLinear

from omlx.custom_kernels.attention import fast as attention_fast
from omlx.custom_kernels.quant_matmul import fast as quant_fast

from ..deepseek_v4.switch_layers import _block_config, _build_mxfp4_blocks
from . import decode_fused, growth
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


def _rope_freqs(d, base, original_seq_len, factor, beta_fast, beta_slow):
    """DeepSeek-V4.1-Flash-1017 inference/model.py precompute_freqs_cis frequencies, fp32.

    Every elementwise step is the reference's own fp32 operation in the reference's order
    (exponent i/d, reciprocal, YaRN ramp and blend). The one transcendental, base**e, is
    taken correctly rounded. The reference evaluates it with CUDA powf, which is not
    correctly rounded, so its accuracy is unspecified; Metal's pow (upstream oMLX computed
    the table on the GPU) is within 1 ulp on these exponents, i.e. at most 0.0015 rad of
    angle at position 200k. The exact value is the closest to the reference formula and the
    same on every platform; the table is built once per parameter set on the host.
    """
    return _rope_freqs_cached(int(d), float(base), int(original_seq_len), float(factor), float(beta_fast), float(beta_slow))


@cache
def _rope_freqs_cached(d, base, original_seq_len, factor, beta_fast, beta_slow):
    exps = (np.arange(0, d, 2, dtype=np.float32) / np.float32(d)).astype(np.float32)
    try:
        import mpmath

        mpmath.mp.prec = 128
        powers = np.array(
            [float(mpmath.power(mpmath.mpf(base), mpmath.mpf(float(e)))) for e in exps],
            np.float64,
        )
    except ImportError:  # float64 pow: < 1 ulp(double), so fp32 rounding is correct
        powers = np.power(np.float64(base), exps.astype(np.float64))
    freqs = (np.float32(1.0) / powers.astype(np.float32)).astype(np.float32)
    if original_seq_len > 0:
        def corrected(rotations):
            return d * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected(beta_fast)), 0)
        high = min(math.ceil(corrected(beta_slow)), d - 1)
        ramp = np.clip(
            ((np.arange(d // 2, dtype=np.float32) - np.float32(low)) / np.float32(max(high - low, 1e-3))).astype(np.float32),
            0, 1,
        ).astype(np.float32)
        smooth = (np.float32(1) - ramp).astype(np.float32)
        freqs = (
            ((freqs / np.float32(factor)).astype(np.float32) * (np.float32(1) - smooth)).astype(np.float32)
            + (freqs * smooth).astype(np.float32)
        ).astype(np.float32)
    return freqs


def _rope_positions(positions):
    if isinstance(positions, range):
        return positions.start, positions.stop, positions.step
    values = np.asarray(positions, np.int64).reshape(-1)  # host sync; callers pass ranges
    if len(values) > 1 and np.all(np.diff(values) == values[1] - values[0]):
        return int(values[0]), int(values[-1]) + int(values[1] - values[0]), int(values[1] - values[0])
    return tuple(int(v) for v in values)


@lru_cache(maxsize=32)   # a decode step or verify block needs one range per RoPE variant
def _rope_table(params, key):
    """cos/sin of the reference's fp32 angles fl32(pos * freq) (torch.outer output is fp32),
    evaluated in float64 and rounded once to fp32: correctly rounded except when the exact
    value lies within ~2^-29 relative of an fp32 midpoint. Computed once per position range
    and shared by every layer of the step."""
    freqs = _rope_freqs(*params)
    pos = np.arange(*key) if len(key) == 3 else np.array(key)
    angles = (pos.astype(np.float32)[:, None] * freqs[None, :]).astype(np.float32).astype(np.float64)
    return mx.array(np.cos(angles).astype(np.float32)), mx.array(np.sin(angles).astype(np.float32))


@mx.compile
def _rope_apply(x, cos, sin, inverse):
    d = cos.shape[-1] * 2
    shape = (1, cos.shape[0], *([1] * (x.ndim - 3)), d // 2)
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    if inverse:
        sin = -sin  # freqs_cis.conj()
    tail = x[..., -d:].astype(mx.float32).reshape(*x.shape[:-1], d // 2, 2)
    a, b = tail[..., 0], tail[..., 1]
    # Complex product in fp32 (fusion/FMA of a*c - b*s is unspecified in the reference).
    rotated = mx.stack([a * cos - b * sin, a * sin + b * cos], -1)
    return mx.concatenate([x[..., :-d], rotated.reshape(*x.shape[:-1], d).astype(x.dtype)], -1)


def rope(x, positions, config, compressed, inverse=False):
    params = (
        config.rope_head_dim,
        config.compress_rope_theta if compressed else config.rope_theta,
        config.original_seq_len if compressed else 0,
        config.rope_factor,
        config.beta_fast,
        config.beta_slow,
    )
    cos, sin = _rope_table(params, _rope_positions(positions))
    return _rope_apply(x, cos, sin, inverse)


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

    def __call__(self, x, qr, latent, cache, shared, start, ratio, latent_start=None):
        c, layer = self._config, self._layer
        end = start + x.shape[1]
        if layer in c.kv_source_layers:
            # CED keeps the query path on the tail while the key path still
            # spans the full chunk; latent_start marks that wider key origin.
            kv_start = start if latent_start is None else int(latent_start)
            kv_end = (
                end
                if latent_start is None
                else kv_start + (0 if latent is None else latent.shape[1])
            )
            previous = cache[3]
            previous = (
                previous[:, : kv_start // ratio]
                if previous is not None
                else pack_activation(
                    mx.zeros((1, 0, c.index_head_dim), x.dtype), bits=4
                )
            )
            if latent is not None:
                pos = range(kv_start // ratio * ratio, kv_end // ratio * ratio, ratio)
                key = pack_activation(
                    rope(self.k_norm(self.wk(latent)), pos, c, True), bits=4
                )
                previous = growth.append(cache, 3, previous, key, kv_start // ratio)
            shared["index_k"] = cache[3] = previous
        key = shared["index_k"]
        q = self.wq_b(qr).reshape(1, x.shape[1], c.index_n_heads, c.index_head_dim)
        q = quantize_activation(rope(q, range(start, end), c, True), bits=4)
        # Reference: weights_proj(x) * scale is a tensor of the activation dtype (bf16).
        weights = (
            self.weights_proj(x)
            * mx.array(c.index_head_dim**-0.5 * c.index_n_heads**-0.5, mx.float32)
        ).astype(x.dtype).astype(mx.float32)
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

    def __call__(self, x, cache, shared, start, ced_kv=None, ced_kv_start=None):
        c, layer = self._config, self._layer
        ratio, length = c.compress_ratios[layer], x.shape[1]
        positions = mx.arange(start, start + length)
        rope_positions = range(start, start + length)
        query, kv_input = self._input_projections(x)
        qr = self.q_norm(query)
        q = rope(
            self.wq_b(qr).reshape(1, length, c.n_heads, c.head_dim),
            rope_positions,
            c,
            bool(ratio),
        )
        new = pack_activation(rope(self.kv_norm(kv_input), rope_positions, c, bool(ratio)))
        old = cache[1]
        # CED clears the stale window whenever bounded replay skips tokens.
        old_len = min(start, c.window_size, 0 if old is None else int(old.shape[1]))
        kv = mx.concatenate([old[:, :old_len], new], 1) if old_len else new
        verify_state = getattr(cache, "_mtp_verify_state", None)
        if verify_state is not None:
            verify_state["window"] = kv
        cache[1] = kv[:, -c.window_size :]
        # List only the causal local window, including the current token.
        if start == 0:
            local = mx.maximum(positions[:, None] - c.window_size + 1, 0)
            local = local + mx.arange(min(length, c.window_size))
        else:
            local = positions[:, None] - c.window_size + 1 + mx.arange(c.window_size)
        valid = (local >= max(0, start - old_len)) & (local <= positions[:, None])
        idx = mx.where(valid, local - (start - old_len), -1)[None]
        pooled = mx.zeros((1, 0, c.head_dim // 2 + c.head_dim // 16), mx.uint8)
        ci = mx.zeros((1, length, 0), mx.int32)
        if ratio:
            latent = None
            if layer in c.kv_source_layers:
                # CED: global KV projects the full encoder-final hidden state
                # (ced_kv) even though queries only attend from the tail.
                kv_x = x if ced_kv is None else ced_kv
                kv_start = start if ced_kv_start is None else int(ced_kv_start)
                latent = self.compressor(kv_x, cache, kv_start)
                if cache[2] is None:
                    cache[2] = pack_activation(
                        mx.zeros((1, 0, c.head_dim), x.dtype),
                        bits=4,
                        group_size=16,
                        e4m3_scale=True,
                    )
                shared["kv"] = cache[2][:, : kv_start // ratio]
            if layer in c.index_source_layers:
                shared["idx"] = self.indexer(
                    x, qr, latent, cache, shared, start, ratio, latent_start=ced_kv_start
                )
            if latent is not None:
                compressed = rope(
                    latent,
                    range(kv_start // ratio * ratio, (kv_start + kv_x.shape[1]) // ratio * ratio, ratio),
                    c,
                    True,
                )
                compressed = pack_activation(
                    compressed, bits=4, group_size=16, e4m3_scale=True
                )
                shared["kv"] = cache[2] = growth.append(
                    cache, 2, shared["kv"], compressed, kv_start // ratio
                )
            ci, pooled = shared["idx"], shared["kv"]
        if (
            length > 8
            and c.n_heads == 64
            and c.head_dim == 512
            and q.dtype == mx.bfloat16
            and attention_fast.has_symbol("packed_sparse_attention")
        ):
            out = attention_fast.packed_sparse_attention(
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
        out = rope(out, rope_positions, c, bool(ratio), inverse=True)
        # DeepSeek-V4.1-Flash model.py Attention.forward (1017, and the current Flash repo):
        # "quantize/dequantize o before einsum", act_quant(o, 32, ue8m0, inplace=True) over
        # each group's values. At df42c109 it fed bf16 o to wo_a directly, as upstream oMLX did.
        grouped = quantize_activation(out.reshape(1, length, c.o_groups, -1))
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
            and quant_fast.has_symbol("grouped_expert")
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
            return quant_fast.grouped_expert(gate, up, y, down).reshape(
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
            and quant_fast.has_symbol("mxfp4_gather_qmm_pair_concat_blocks")
        ):
            inp = (
                quantize_activation(x)
                if self.w1.quantize_input and not input_quantized
                else x
            )
            pair = quant_fast.mxfp4_gather_qmm_pair_concat_blocks(
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
            and quant_fast.has_symbol("affine_gather_qmm_pair_concat_blocks")
        ):
            inp = (
                quantize_activation(x)
                if self.w1.quantize_input and not input_quantized
                else x
            )
            meta, count, variant = block_plan
            pair = quant_fast.affine_gather_qmm_pair_concat_blocks(
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

    def __call__(self, h, pre, cache, shared, start, image_mask, ced_tail=None):
        if (
            ced_tail is None
            and image_mask is None
            and h.shape[0] == 1
            and h.shape[1] == 1
            and getattr(cache, "_mtp_verify_state", None) is None
            and decode_fused.supported(self, self._config)
        ):
            return decode_fused.decode_block(self, h, pre, cache, shared, start, self._config)
        if (
            shared.get("_verify_rows")
            and ced_tail is None
            and image_mask is None
            and getattr(cache, "_mtp_verify_state", None) is not None
        ):
            return decode_fused.decode_block_verify(self, h, pre, cache, shared, start, self._config)
        ced_kv = ced_kv_start = None
        if ced_tail is not None:
            # CED bounded replay: queries, SWA KV and MoE run on the tail
            # only. The midpoint CSA2 layer still projects global KV from the
            # full encoder-final hidden states it receives as ced_kv.
            tail = min(int(ced_tail), h.shape[1])
            full_len = h.shape[1]
            if hasattr(self.attn, "compressor"):
                ap, ao, ac = hc_mixes(
                    h,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self._config,
                )
                x = hc_pre_norm(
                    h, pre, self.attn_norm.weight, self.attn_norm.eps
                )
                ced_kv, ced_kv_start = x, start
                ap, ao, ac = ap[:, -tail:], ao[:, -tail:], ac[:, -tail:]
            else:
                h, pre = h[:, -tail:], pre[:, -tail:]
                ap, ao, ac = hc_mixes(
                    h,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self._config,
                )
                x = hc_pre_norm(
                    h, pre, self.attn_norm.weight, self.attn_norm.eps
                )
            # Bounded replay skipped every position before the tail, so the
            # stored window is non-contiguous with these queries: drop it.
            cache[1] = None
            h, pre, x = h[:, -tail:], pre[:, -tail:], x[:, -tail:]
            if image_mask is not None:
                image_mask = image_mask[:, -tail:]
            start = start + full_len - tail
        else:
            ap, ao, ac = hc_mixes(
                h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self._config
            )
            x = hc_pre_norm(h, pre, self.attn_norm.weight, self.attn_norm.eps)
        h = hc_post(
            self.attn(x, cache, shared, start, ced_kv=ced_kv, ced_kv_start=ced_kv_start),
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
        from ..mlx_lm_mtp import get_mtp_depth, is_mtp_active, is_mtp_depth_fixed

        self.configure_mtp(config.preserve_mtp and is_mtp_active(), get_mtp_depth())
        self._omlx_mtp_depth_fixed = is_mtp_depth_fixed()

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

    def _batched_decode_ok(self, input_ids, cache, masks, inputs_embeds, capture, kwargs):
        """One token for each of 2..MAX_BATCH requests, all at a valid position, through
        layers the fused decode supports: run the layers once for the whole batch."""
        b_n = input_ids.shape[0]
        if not (
            input_ids.shape[1] == 1
            and 2 <= b_n <= decode_fused.MAX_BATCH
            and inputs_embeds is None
            and not capture
            and kwargs.get("mtp_verify_states") is None
            and not kwargs.get("_ced_prefill", False)
            and all(decode_fused.supported(layer, self._config) for layer in self.layers)
        ):
            return False
        if masks is not None and not bool(np.all(np.asarray(masks))):
            return False
        return all(not item.empty() for item in cache)

    def _forward_decode_batched(self, input_ids, cache):
        """_forward for B single-token requests with the weights read once per step.

        Same per-request semantics as the row loop below (positions, caches, Engram history,
        cache bookkeeping). decode_fused.decode_block_nb, Engram (each request's row projected
        on its own) and project_logits give each request its single-request result: bit for
        bit below decode_moe.MMA_MIN_BATCH requests, and within the fp32 summation rounding of
        the simdgroup-matrix GEMVs from there on.
        """
        c = self._config
        b_n = input_ids.shape[0]
        rcs = [[item.extract(row) for item in cache] for row in range(b_n)]
        starts = []
        for rc in rcs:
            start = rc[0].size()
            if any(item.size() != start for item in rc):
                raise ValueError("DeepSeek V4.1 cache offsets diverged across layers")
            starts.append(start)
        hashes, histories = None, None
        if self._hasher is not None:
            per_row = [self._hasher(input_ids[r : r + 1], rcs[r][0][6], None) for r in range(b_n)]
            hashes = np.concatenate([p[0] for p in per_row], 0)
            histories = [p[1] for p in per_row]
        h = self.embed(input_ids)
        h = mx.repeat(h[..., None, :], c.hc_mult, -2)
        pre = mx.broadcast_to((mx.arange(c.hc_mult) == 0).astype(mx.float32), h.shape[:-1])
        shareds = [{} for _ in range(b_n)]
        prefetch = getattr(self, "_engram_prefetch", None)
        with prefetch.forward() if prefetch is not None else nullcontext():
            if prefetch is not None and c.engram_layer_ids:
                prefetch.submit(self.layers[c.engram_layer_ids[0]].engram.embed, hashes[:, :, 0])
            for i, layer in enumerate(self.layers):
                if "engram" in layer:
                    ix = list(c.engram_layer_ids).index(i)
                    if prefetch is not None:
                        mx.async_eval(h, pre)
                    h = layer.engram(h, hashes[:, :, ix], None)
                    if prefetch is not None and ix + 1 < len(c.engram_layer_ids):
                        prefetch.submit(
                            self.layers[c.engram_layer_ids[ix + 1]].engram.embed, hashes[:, :, ix + 1]
                        )
                h, pre = decode_fused.decode_block_nb(
                    layer, h, pre, [rcs[r][i] for r in range(b_n)], shareds, starts, c
                )
                if prefetch is not None and "engram" in layer:
                    mx.async_eval(h, pre)
                for r in range(b_n):
                    rc = rcs[r]
                    rc[i][0] = mx.array([starts[r] + 1], mx.int32)
                    if histories is not None and i == 0:
                        rc[i][6] = mx.array(histories[r], mx.int64)
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
                            rc[i][slot] = mx.zeros((1, 0), mx.int64) if slot == 6 else empty
        logits = project_logits(self.norm(hc_pre(h, pre)), self.head.weight)
        for i, item in enumerate(cache):
            item.adopt_rows([rcs[r][i] for r in range(b_n)])
            item.advance(1)
        return logits

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
        if self._batched_decode_ok(input_ids, cache, masks, inputs_embeds, capture, kwargs):
            return self._forward_decode_batched(input_ids, cache)
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
            # An MTP verify block runs through the fused decode kernels token-exactly
            # (decode_fused.decode_block_verify: each token gets its single-token decode
            # result, so speculation cannot change greedy output) when every layer has them.
            if (
                verify_states is not None
                and inputs_embeds is None
                and 1 <= end - begin <= decode_fused.MAX_BATCH
                and all(decode_fused.supported(layer, c) for layer in self.layers)
            ):
                shared["_verify_rows"] = True
            # Only cache-only scheduler calls may omit decoder logits. A short
            # contiguous suffix retains its existing window rather than
            # pretending that a full replay window was present in this chunk.
            ced_tail = (
                c.window_size
                if c.ced_prefill
                and kwargs.get("_ced_prefill", False)
                and verify_states is None
                and end - begin > c.window_size
                else None
            )
            ced_mid = c.n_layers // 2
            prefetch = getattr(self, "_engram_prefetch", None)
            # The CPU enqueues a whole prefill chunk in ~1s while the GPU runs
            # ~15x longer; with the layer loop fully lazy every intermediate
            # (fp32 hyper-connection streams across hc_mult, gathered sparse
            # attention loads, MoE routes) stays pinned until the final logits
            # eval and the Metal pool hoards every freed size class as
            # IOAccelerator footprint (measured +14.5GB per 2048-token chunk).
            # Eval the running stream at each layer boundary during prefill so
            # intermediates retire as the GPU progresses, and clear the pool at
            # the same points (per-layer widths vary across the ratio switch,
            # CED tail and 2047/2048 chunk widths, so size classes never
            # reuse). Decode/verify widths stay lazy for latency.
            prefill_backpressure = end - begin >= 256
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
                    query_start = start
                    if ced_tail is not None and i > ced_mid:
                        # The midpoint layer already advanced h to the tail.
                        query_start = start + end - begin - ced_tail
                    h, pre = layer(
                        h,
                        pre,
                        rc[i],
                        shared,
                        query_start,
                        image_mask,
                        ced_tail=ced_tail if ced_tail is not None and i >= ced_mid else None,
                    )
                    if prefetch is not None and "engram" in layer:
                        mx.async_eval(h, pre)
                    if prefill_backpressure:
                        mx.eval(h, pre)
                        mx.clear_cache()
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
            # The scheduler discards this tail-only result. Ordinary forward
            # calls still return real logits for every input token.
            results.append(
                mx.pad(
                    logits,
                    [
                        (0, 0),
                        (begin, input_ids.shape[1] - end),
                        (0, 0),
                    ],
                )
            )
        for i, item in enumerate(cache):
            # Rows ran in place on the copies extract() handed out; a batch keeps them as
            # rows rather than re-concatenating every slot of every layer each step.
            item.adopt_rows(rows[i])
            item.advance(input_ids.shape[1])
        logits = mx.concatenate(results, 0)
        if capture:
            if not c.dspark_target_layer_ids or set(captured) != set(
                c.dspark_target_layer_ids
            ):
                raise ValueError("Incomplete DSpark target hidden capture")
            return logits, mx.concatenate(
                [captured[i][:, -logits.shape[1] :] for i in c.dspark_target_layer_ids],
                axis=-1,
            )
        return logits
