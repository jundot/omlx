# Copyright © 2026 Apple Inc.

import logging
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_flatten

from omlx.patches.deepseek_v4.decode_consistency import (
    is_armed as is_dspark_verify_armed,
)
from omlx.patches.deepseek_v4.decode_consistency import matmul as decode_matmul
from omlx.patches.deepseek_v4.indexer_dispatch import (
    disable_native_indexer,
    native_indexer_available,
    native_indexer_disabled,
    native_indexer_shape_eligible,
)
from omlx.patches.deepseek_v4.switch_layers import SwitchGLU
from omlx.patches.deepseek_v4.verify_attention import (
    exact_attention,
    exact_local_scores,
    exact_local_values,
    rowwise_gemm,
)
from omlx.patches.deepseek_v4.wsdpa_attention import wsdpa_prefill, wsdpa_topk_prefill

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import CacheList, PoolingCache, RotatingKVCache
from .hyper_connection import HyperConnection, HyperHead, hc_expand
from .mla import MultiLinear
from .pipeline import PipelineMixin

_DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED = False
_DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED = False


def set_dspark_verify_armed(flag: bool) -> None:
    from omlx.patches.deepseek_v4.decode_consistency import set_armed

    set_armed(flag)


def _project_attention_output(attn: nn.Module, out: mx.array, offset: Any) -> mx.array:
    out = attn.rope(out, offset, inverse=True)

    def prepare(row: mx.array) -> mx.array:
        batch, _, length, _ = row.shape
        row = row.reshape(batch, attn.o_groups, -1, length, attn.head_dim)
        return row.transpose(0, 1, 3, 2, 4).flatten(-2)

    def finish(row: mx.array) -> mx.array:
        return row.transpose(0, 2, 1, 3).flatten(-2)

    def project_a(row: mx.array) -> mx.array:
        return finish(attn.wo_a(prepare(row)))

    if is_dspark_verify_armed():
        prepared = mx.concatenate(
            [prepare(out[:, :, idx : idx + 1]) for idx in range(out.shape[2])],
            axis=2,
        )
        from omlx.patches.deepseek_v4.verify_qmv import (
            exact_verify_multi_qmv,
            multi_eligible,
        )

        if multi_eligible(attn.wo_a, prepared[0]):
            projected = finish(exact_verify_multi_qmv(attn.wo_a, prepared[0])[None])
        else:
            projected = mx.concatenate(
                [
                    finish(attn.wo_a(prepared[:, :, idx : idx + 1]))
                    for idx in range(prepared.shape[2])
                ],
                axis=1,
            )
        return attn.wo_b(projected)
    return attn.wo_b(project_a(out))


def _batched_m1_attention(
    queries: mx.array,
    key_rows: List[mx.array],
    scale: float,
    sinks: mx.array,
) -> mx.array:
    """Attend independent cache views with one decode-consistent kernel."""
    return exact_attention(queries, key_rows, scale, sinks)


def _is_dspark_model(config: Any) -> bool:
    return bool(
        int(getattr(config, "dspark_block_size", 0) or 0)
        and tuple(getattr(config, "dspark_target_layer_ids", ()) or ())
    )


def _materialize_cache_arrays(cache: Optional[Any]) -> None:
    """Detach DeepSeek-V4 cache update graphs from prior decode steps."""
    if cache is None:
        return

    cache_arrays = []
    for layer_cache in cache:
        if layer_cache is None:
            continue
        leaves = getattr(layer_cache, "caches", None) or (layer_cache,)
        for leaf in leaves:
            if leaf is None:
                continue
            for value in vars(leaf).values():
                if isinstance(value, mx.array):
                    cache_arrays.append(value)

    if cache_arrays:
        mx.eval(*cache_arrays)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 1.5
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    num_experts_per_tok: int = 6
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 512
    scoring_func: str = "sqrtsoftplus"
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    # DeepSeek-V4-Flash-0731 embeds a DSpark drafter in mtp.0..N.  The
    # legacy num_nextn_predict_layers field remains set for compatibility,
    # so these fields are the architecture discriminator.
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: List[int] = field(default_factory=list)
    dspark_markov_rank: int = 256
    n_mtp_layers: int = 0
    tie_word_embeddings: bool = False
    topk_method: str = "noaux_tc"
    use_native_ratio128_attention: bool = True
    # DeepSeek-V4-Flash-Vision-Exp uses flat top-level vision fields rather
    # than a nested vision_config. Zero keeps ordinary DeepSeek-V4 text models
    # byte-for-byte on their existing path.
    vision_n_layers: int = 0
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 384
    vision_min_pixels: int = 147456
    vision_max_wh_ratio: int = 8

    def __post_init__(self):
        if not self.compress_ratios:
            n = self.num_hidden_layers
            self.compress_ratios = (
                [0]
                + [4 if i % 2 else 128 for i in range(max(n - 2, 0))]
                + ([0] if n >= 2 else [])
            )
        self.compress_ratios = list(self.compress_ratios[: self.num_hidden_layers])
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError(
                "`compress_ratios` must have one entry per hidden layer, "
                f"got {len(self.compress_ratios)} for {self.num_hidden_layers} layers."
            )
        bad = [r for r in self.compress_ratios if r not in (0, 4, 128)]
        if bad:
            raise ValueError(f"Unsupported DeepSeek-V4 compress ratios: {bad}")


def make_quantization_config(model):
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}

    flat_modules = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    experts = {
        k: mxfp4
        for k, _ in flat_modules
        if ".ffn.switch_mlp." in k and k.endswith("_proj")
    }
    shared_experts = {k: mxfp8 for k, _ in flat_modules if ".ffn.shared_experts." in k}
    attn = {
        k: mxfp8
        for k, _ in flat_modules
        if not k.startswith("vision.")
        and (".attn.w" in k or ".attn.indexer.wq" in k)
    }
    # MTP fusion projections. Lightning checkpoints use e_proj/h_proj;
    # embedded DSpark uses main_proj on stage 0. These ship as e4m3 weight +
    # e8m0 block scale, i.e. mxfp8 after sanitize. Without an explicit entry
    # they fall through to affine and strict loading asks for missing biases.
    mtp_projs = {
        k: mxfp8
        for k, _ in flat_modules
        if k.startswith("mtp.")
        and (k.endswith(".e_proj") or k.endswith(".h_proj") or k.endswith(".main_proj"))
    }

    return {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        **experts,
        **shared_experts,
        **attn,
        **mtp_projs,
    }


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    if func == "sqrtsoftplus":
        return mx.sqrt(nn.softplus(scores))
    raise ValueError(f"Unsupported DeepSeek-V4 scoring function: {func}")


@mx.compile
def _expert_select(
    logits: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> Tuple[mx.array, mx.array]:
    logits = logits.astype(mx.float32)
    scores = _score_func(logits, scoring_func)
    biased = scores + e_score_correction_bias
    inds = mx.argpartition(-biased, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


@mx.compile
def _hash_expert_select(
    input_ids: mx.array,
    logits: mx.array,
    tid2eid: mx.array,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> Tuple[mx.array, mx.array]:
    logits = logits.astype(mx.float32)
    scores = _score_func(logits, scoring_func)
    inds = tid2eid[input_ids]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class LimitedSwiGLU(nn.Module):
    def __init__(self, limit: float, *, fp32: bool = False):
        super().__init__()
        self.limit = limit
        self.fp32 = fp32

    def __call__(self, x, gate):
        if not self.fp32:
            return _limited_swiglu(gate, x, self.limit)
        dtype = x.dtype
        return _limited_swiglu(
            gate.astype(mx.float32),
            x.astype(mx.float32),
            self.limit,
        ).astype(dtype)


class DeepseekV4RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict] = None,
        max_position_embeddings: int = 1048576,
        freq_scale: int = 1,
    ):
        super().__init__()
        self.dims = dims
        self.freq_scale = freq_scale

        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")

        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            original_max_position_embeddings = scaling_config[
                "original_max_position_embeddings"
            ]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(
                        original_max_position_embeddings / (num_rotations * 2 * math.pi)
                    )
                    / (2 * math.log(base))
                )

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type: {rope_type}")

        self._freqs = 1.0 / inv_freq
        self._freqs_cache = {}

    def _get_freqs(self, head_dim: int, inverse: bool):
        key = (head_dim, inverse)
        if key not in self._freqs_cache:
            f = self._freqs
            if self.freq_scale != 1:
                f = f / self.freq_scale
            if inverse:
                f = -f
            nope_pairs = (head_dim - self.dims) // 2
            if nope_pairs > 0:
                f = mx.concatenate([mx.full((nope_pairs,), mx.inf), f])
            self._freqs_cache[key] = f
        return self._freqs_cache[key]

    def __call__(
        self,
        x: mx.array,
        offset: Any = 0,
        inverse: bool = False,
    ) -> mx.array:
        head_dim = x.shape[-1]
        freqs = self._get_freqs(head_dim, inverse)
        offset = offset // self.freq_scale if self.freq_scale != 1 else offset
        return mx.fast.rope(
            x,
            head_dim,
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=freqs,
        )


def _apply_score_mask(scores: mx.array, mask: Optional[mx.array]) -> mx.array:
    if mask is None:
        return scores
    if mask.dtype == mx.bool_:
        return mx.where(mask, scores, mx.finfo(scores.dtype).min)
    return scores + mask.astype(scores.dtype)


def _dspark_rowwise_mm(
    lhs: mx.array,
    rhs: mx.array,
    transpose_rhs: bool,
) -> mx.array:
    """Run each verify row through the same NAX GEMM as decode."""
    return rowwise_gemm(lhs[:, 0], rhs[:, 0], transpose_rhs)[:, None]


def _dspark_ring_mm(
    lhs: mx.array,
    source: mx.array,
    indices: mx.array,
    transpose_rhs: bool,
) -> mx.array:
    """Run an exact local GEMM without materializing every ring snapshot."""
    from omlx.custom_kernels.glm_moe_dsa import fast

    return fast.dspark_ring_gemm(
        mx.contiguous(lhs[:, 0]),
        mx.contiguous(source),
        mx.contiguous(indices),
        transpose_rhs,
    )[:, None]


def _extend_mask(mask: Optional[mx.array], pool_mask: Optional[mx.array], N: int):
    if mask is None:
        return None

    if mask.ndim == 2:
        mask = mask[None, None]
    B, H, L, S = mask.shape

    if pool_mask is None:
        pool_mask = mx.ones((B, H, L, N - S), dtype=mx.bool_)
    elif pool_mask.ndim == 2:
        pool_mask = mx.broadcast_to(pool_mask, (B, H, L, N - S))
    elif pool_mask.ndim == 3:
        pool_mask = mx.broadcast_to(pool_mask[:, None], (B, H, L, N - S))

    full_mask = mx.concatenate([mask, pool_mask], axis=-1)

    return full_mask


def _apply_image_visibility_mask(
    inputs: mx.array,
    mask: Optional[mx.array],
    vocab_size: int,
) -> Tuple[Optional[mx.array], bool]:
    """Add full visibility for image spans in the current cache-aware suffix."""
    batch, length = inputs.shape
    if length <= 1 or not bool(mx.any(inputs >= vocab_size).item()):
        return mask, False

    source_length = mask.shape[-1] if mask is not None else length
    suffix_start = source_length - length
    visible = mx.zeros((batch, length, source_length), dtype=mx.bool_)
    has_image_span = False
    for batch_index, row in enumerate(inputs.tolist()):
        start = 0 if row and int(row[0]) >= vocab_size else None
        for index, token in enumerate(row):
            kind = int(token) - vocab_size
            if kind == 0:
                start = index
            elif kind < 0:
                start = None
            elif kind == 4 and start is not None:
                visible = visible.at[
                    batch_index,
                    start : index + 1,
                    suffix_start + start : suffix_start + index + 1,
                ].add(True)
                has_image_span = True
                start = None
        if start is not None:
            visible = visible.at[
                batch_index,
                start:length,
                suffix_start + start : suffix_start + length,
            ].add(True)
            has_image_span = True

    if not has_image_span:
        return mask, False
    visible = visible[0] if batch == 1 else visible[:, None]
    return (visible if mask is None else mask | visible), True


def _image_embedding_slices(images, offset: int, length: int):
    """Map absolute image blocks to destination/source slices in a suffix."""
    suffix_end = offset + length
    slices = []
    for image in images or ():
        image_end = image.start + len(image.types)
        overlap_start = max(offset, image.start)
        overlap_end = min(suffix_end, image_end)
        if overlap_start < overlap_end:
            slices.append(
                (
                    image,
                    slice(overlap_start - offset, overlap_end - offset),
                    slice(overlap_start - image.start, overlap_end - image.start),
                )
            )
    return slices


def _cache_offset(cache: Optional[Any]) -> int:
    if not cache:
        return 0
    cache_item = cache[0]
    if isinstance(cache_item, CacheList):
        cache_item = cache_item[0]
    offset = getattr(cache_item, "offset", 0)
    if isinstance(offset, mx.array):
        if offset.size != 1:
            raise ValueError("DeepSeek-V4 Vision requires a scalar cache offset.")
        offset = offset.item()
    return int(offset)


@partial(mx.compile, shapeless=True)
def _simple_compress_kv(kv, gate, ape, head_dim):
    weights = mx.softmax(gate.astype(mx.float32) + ape, axis=-2)
    weights = weights.astype(kv.dtype)
    return (kv * weights).sum(axis=-2)


@mx.compile
def _overlap_compress_kv(kv, gate, ape, head_dim):
    B, L, R, D = kv.shape

    gate = gate + ape.astype(gate.dtype)

    kv_0 = mx.zeros((B, 1, R, D // 2), dtype=kv.dtype)
    kv_a, kv_b = mx.split(kv, 2, axis=-1)
    kv_a = mx.concatenate([kv_0, kv_a[:, :-1]], axis=1)
    kv = mx.concatenate([kv_a, kv_b], axis=2)

    gate_0 = mx.full((B, 1, R, D // 2), -mx.inf, dtype=kv.dtype)
    gate_a, gate_b = mx.split(gate, 2, axis=-1)
    gate_a = mx.concatenate([gate_0, gate_a[:, :-1]], axis=1)
    gate = mx.concatenate([gate_a, gate_b], axis=2)

    weights = mx.softmax(gate, axis=-2, precise=True)
    return (kv * weights).sum(axis=-2)


# Pooled-axis tile for the MLX indexer fallback (used when the native
# glm_moe_dsa extension is not built). The native dsa_indexer_scores kernel
# keeps the (heads, L, P) score tensor in registers; the fallback has to
# materialize it, and at ratio 4 with a 512-token chunk P = ctx/4, so the
# intermediate is (1, 64, 512, ctx/4) fp32 — 8.3 GiB at 273k context, read
# five more times. Worse, 64*512*P crosses 2**31 elements at ctx = 256k, the
# boundary where mlx's int32 kernel indexing silently zeros the tail and
# corrupts top-k selection. Tiling keeps each matmul under 2**31 elements;
# the memory estimator separately accounts for lazy evaluation retaining
# multiple score shards at once.
_INDEXER_POOL_TILE = 16384
# mlx kernels index with int32, so a tensor at or past 2**31 elements has its
# tail silently zeroed. Stay a factor of 2 below that. For a 512-token chunk
# with 64 index heads this is reached at P = ctx/4 ~= 32768, i.e. ctx ~= 128k.
_INDEXER_MAX_ELEMS = 2**30
_DEEPSEEK_V4_INDEXER_FALLBACK_WARNED = False
_DEEPSEEK_V4_M2_MMA_SCORE = os.getenv(
    "OMLX_DSV4F_M2_MMA_SCORE", "1"
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_M2_MMA_SCORE_LOGGED = False


def _dsv4f_m2_exact_pairing(config, compress_ratio: int) -> bool:
    """True only for the exact DeepSeek-V4-Flash / Apple M2 Ultra pairing.

    The v25 MMA indexer score kernel is benchmark-proven (and bit-exact
    validated) on precisely this checkpoint/hardware combination; every
    other pairing keeps the Steel kernel.
    """
    if compress_ratio != 4:
        return False
    try:
        if mx.device_info().get("device_name") != "Apple M2 Ultra":
            return False
    except Exception:
        return False
    return (
        config.model_type == "deepseek_v4"
        and config.vocab_size == 129280
        and config.hidden_size == 4096
        and config.moe_intermediate_size == 2048
        and config.num_hidden_layers == 43
        and config.num_attention_heads == 64
        and config.n_routed_experts == 256
        and config.num_experts_per_tok == 6
        and config.max_position_embeddings == 1048576
        and config.index_n_heads == 64
        and config.index_head_dim == 128
        and config.index_topk == 512
    )


def _dsv4f_m2_mma_score_enabled(config, compress_ratio: int) -> bool:
    """Gate for the v25 from-scratch MMA score kernel (1.37x over Steel).

    Exact-fingerprint pairing, env rollback (OMLX_DSV4F_M2_MMA_SCORE=0),
    and an extension build exposing the symbol. The kernel serves bf16 /
    H=64 / D=128 / weights [B, L, H] / non-causal only — all guaranteed by
    the fingerprint and this call site; N >= 64 is re-checked per call.
    """
    if not _DEEPSEEK_V4_M2_MMA_SCORE:
        return False
    return _dsv4f_m2_exact_pairing(config, compress_ratio)


@partial(mx.compile, shapeless=True)
def _indexer_head_reduce(scores, weights, scale):
    """relu -> scale -> head-weight -> head-sum, fused over `scores`.

    `scores` is (B, H, L, P_tile); the reduction is over the head axis (1),
    entirely within each pooled tile, so tiling P never crosses the reduce.
    """
    return (mx.maximum(scores, 0) * scale * weights).sum(axis=1)


@partial(mx.compile, shapeless=True)
def _split_softmax(log_normalizer, logits_a, logits_b, sinks=None):
    if sinks is not None:
        log_normalizer = mx.logaddexp(log_normalizer, sinks)
    weights_a = mx.exp(logits_a - log_normalizer)
    weights_b = mx.exp(logits_b - log_normalizer)
    return weights_a, weights_b


@mx.compile
def _dspark_sparse_exact_attention(
    q_scaled: mx.array,
    local_kv: mx.array,
    pooled_kv: mx.array,
    sinks: mx.array,
) -> mx.array:
    """Fuse exact sparse-attention glue around M=1-equivalent GEMMs."""
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    local_scores = _dspark_rowwise_mm(q_bl, local_kv, True)
    pooled_scores = _dspark_rowwise_mm(q_bl, pooled_kv, True)
    local_scores = local_scores.transpose(0, 2, 1, 3)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)

    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)
    normalizer = mx.logaddexp(
        normalizer,
        mx.logsumexp(pooled_scores, -1, keepdims=True),
    )
    local_weights, pooled_weights = _split_softmax(
        normalizer,
        local_scores,
        pooled_scores,
        sinks[None, :, None, None],
    )

    local_out = _dspark_rowwise_mm(
        local_weights.transpose(0, 2, 1, 3),
        local_kv,
        False,
    )
    pooled_out = _dspark_rowwise_mm(
        pooled_weights.transpose(0, 2, 1, 3),
        pooled_kv,
        False,
    )
    return (local_out + pooled_out).transpose(0, 2, 1, 3)


@mx.compile
def _dspark_ring_sparse_exact_attention(
    q_scaled: mx.array,
    local_source: mx.array,
    local_indices: mx.array,
    pooled_kv: mx.array,
    sinks: mx.array,
) -> mx.array:
    """Apply exact sparse attention directly to physical-ring KV rows."""
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    local_scores = _dspark_ring_mm(q_bl, local_source, local_indices, True)
    pooled_scores = _dspark_rowwise_mm(q_bl, pooled_kv, True)
    local_scores = local_scores.transpose(0, 2, 1, 3)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)

    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)
    normalizer = mx.logaddexp(
        normalizer,
        mx.logsumexp(pooled_scores, -1, keepdims=True),
    )
    local_weights, pooled_weights = _split_softmax(
        normalizer,
        local_scores,
        pooled_scores,
        sinks[None, :, None, None],
    )

    local_out = _dspark_ring_mm(
        local_weights.transpose(0, 2, 1, 3),
        local_source,
        local_indices,
        False,
    )
    pooled_out = _dspark_rowwise_mm(
        pooled_weights.transpose(0, 2, 1, 3),
        pooled_kv,
        False,
    )
    return (local_out + pooled_out).transpose(0, 2, 1, 3)


def _sparse_pooled_ring_attention(
    q: mx.array,
    local_source: mx.array,
    local_indices: mx.array,
    pooled: mx.array,
    topk: mx.array,
    scale: float,
    sinks: mx.array,
) -> mx.array:
    """Select pooled rows and attend without gathering the local KV ring."""
    batch, _, length, head_dim = q.shape
    if length != 1:
        raise ValueError("DSpark physical-ring attention requires L=1 rows.")
    idx = topk[:, None, :, :, None]
    pooled = mx.take_along_axis(
        mx.broadcast_to(
            pooled[:, None, None],
            (batch, 1, length, pooled.shape[1], head_dim),
        ),
        mx.broadcast_to(idx, idx.shape[:-1] + (head_dim,)),
        axis=3,
    ).squeeze(1)
    return _dspark_ring_sparse_exact_attention(
        q * scale,
        local_source,
        local_indices,
        pooled,
        sinks,
    ).astype(q.dtype)


def _native_sparse_attention_available() -> bool:
    """Return whether ratio-128 dispatch may attempt the native kernel."""
    global _DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED

    if _DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED:
        return False
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

        return glm_fast.has_symbol("deepseek_v4_sparse_attention")
    except Exception as exc:
        _DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED = True
        logging.getLogger(__name__).warning(
            "DSV4 native sparse attention unavailable; MLX fallback for "
            "the rest of this process: %s",
            exc,
        )
        return False


def _sparse_pooled_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled: mx.array,
    topk: mx.array,
    local_mask: Optional[mx.array],
    pooled_mask: Optional[mx.array],
    scale: float,
    sinks: Optional[mx.array],
    q_offset: Optional[Union[int, mx.array]] = None,
    compress_ratio: Optional[int] = None,
    local_window: Optional[int] = None,
    decode_consistent: bool = False,
    native_only: bool = False,
    _standard_mask: bool = False,
) -> Optional[mx.array]:
    global _DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED

    B, H, L, D = q.shape
    if (
        q_offset is not None
        and compress_ratio is not None
        and local_window is not None
        and sinks is not None
        and not isinstance(q_offset, mx.array)
        and q.dtype in (mx.float16, mx.bfloat16)
        and topk.dtype == mx.uint32
        and B >= 1
        and H == 64
        and L > 4
        and D == 512
        and local_kv.ndim == 4
        and local_kv.shape[1] == 1
        and local_kv.shape[-1] == D
        and pooled.ndim == 3
        and pooled.shape[-1] == D
        and topk.ndim == 3
    ):
        if _standard_mask and B == 1 and q.dtype == mx.bfloat16 and topk.shape[1] == L:
            out = wsdpa_topk_prefill(
                q,
                local_kv,
                pooled,
                topk,
                sinks,
                scale,
                int(q_offset),
                int(local_window),
                int(compress_ratio),
            )
            if out is not None:
                return out
        if not _DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED:
            try:
                from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

                if glm_fast.has_symbol("deepseek_v4_sparse_attention"):
                    return glm_fast.deepseek_v4_sparse_attention(
                        q,
                        local_kv,
                        pooled,
                        topk[:, None],
                        sinks,
                        scale,
                        int(q_offset),
                        int(compress_ratio),
                        int(local_window),
                    )
            except Exception as exc:
                _DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED = True
                logging.getLogger(__name__).warning(
                    "DSV4 native sparse attention kernel failed; MLX fallback "
                    "for the rest of this process: %s",
                    exc,
                    exc_info=True,
                )

    if native_only:
        return None

    idx = topk[:, None, :, :, None]
    pooled = mx.take_along_axis(
        mx.broadcast_to(pooled[:, None, None], (B, 1, L, pooled.shape[1], D)),
        mx.broadcast_to(idx, idx.shape[:-1] + (D,)),
        axis=3,
    )

    q_scaled = q * scale
    exact_local = decode_consistent and L == 1 and 1 <= B <= 6
    pooled_sq = pooled.squeeze(1)
    if exact_local and local_mask is None and pooled_mask is None and sinks is not None:
        return _dspark_sparse_exact_attention(
            q_scaled,
            local_kv,
            pooled_sq,
            sinks,
        ).astype(q.dtype)
    if exact_local:
        if B == 1:
            query_rows = q
        else:
            query_rows = q[:, :, 0].transpose(1, 0, 2)[None]
        local_rows = [local_kv[idx : idx + 1] for idx in range(B)]
        row_scores = exact_local_scores(query_rows, local_rows, scale)
        local_scores = (
            row_scores if B == 1 else row_scores[0].transpose(1, 0, 2)[:, :, None]
        )
    else:
        local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    local_scores = _apply_score_mask(local_scores, local_mask)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)

    q_bl = q_scaled.transpose(0, 2, 1, 3)
    if decode_consistent and L == 1:
        pooled_scores = _dspark_rowwise_mm(q_bl, pooled_sq, True)
    else:
        pooled_scores = q_bl @ pooled_sq.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    pooled_scores = _apply_score_mask(pooled_scores, pooled_mask)
    normalizer = mx.logaddexp(
        normalizer, mx.logsumexp(pooled_scores, -1, keepdims=True)
    )

    local_weights, pooled_weights = _split_softmax(
        normalizer,
        local_scores,
        pooled_scores,
        sinks[None, :, None, None] if sinks is not None else None,
    )

    if exact_local:
        row_weights = (
            local_weights if B == 1 else local_weights[:, :, 0].transpose(1, 0, 2)[None]
        )
        row_out = exact_local_values(row_weights, local_rows)
        out = row_out if B == 1 else row_out[0].transpose(1, 0, 2)[:, :, None]
    else:
        out = local_weights @ local_kv
    pw_bl = pooled_weights.transpose(0, 2, 1, 3)
    if decode_consistent and L == 1:
        pooled_out = _dspark_rowwise_mm(pw_bl, pooled_sq, False)
    else:
        pooled_out = pw_bl @ pooled_sq
    out = out + pooled_out.transpose(0, 2, 1, 3)
    return out.astype(q.dtype)


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.hash = layer_idx < config.num_hash_layers
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.vocab_size = config.vocab_size
        self.vision = config.vision_n_layers > 0
        self.weight = mx.zeros((self.num_experts, self.hidden_dim))
        if self.hash:
            self.tid2eid = mx.zeros((config.vocab_size, self.top_k), dtype=mx.int32)
        if not self.hash or self.vision:
            self.e_score_correction_bias = mx.zeros(
                (self.num_experts,), dtype=mx.float32
            )
        if self.vision:
            self.e_score_correction_bias_vl = mx.zeros(
                (self.num_experts,), dtype=mx.float32
            )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        logits = decode_matmul(x, self.weight.T)

        image_mask = None
        if self.vision:
            if input_ids is None:
                raise ValueError("DeepSeek-V4 Vision routing requires input_ids.")
            image_mask = input_ids >= self.vocab_size
        if self.hash:
            if input_ids is None:
                raise ValueError("DeepSeek-V4 hash routing requires input_ids.")
            if image_mask is None:
                inds, weights = _hash_expert_select(
                    input_ids,
                    logits,
                    self.tid2eid,
                    self.routed_scaling_factor,
                    self.norm_topk_prob,
                    self.scoring_func,
                )
            else:
                safe_ids = mx.where(image_mask, 0, input_ids)
                scores = _score_func(logits.astype(mx.float32), self.scoring_func)
                inds = self.tid2eid[safe_ids]
                vision_inds = mx.argpartition(
                    -(scores + self.e_score_correction_bias_vl),
                    kth=self.top_k - 1,
                    axis=-1,
                )[..., : self.top_k]
                inds = mx.where(image_mask[..., None], vision_inds, inds)
                weights = mx.take_along_axis(scores, inds, axis=-1)
                if self.scoring_func != "softmax" and self.norm_topk_prob:
                    weights /= weights.sum(axis=-1, keepdims=True) + 1e-20
                weights *= self.routed_scaling_factor
        else:
            bias = self.e_score_correction_bias
            if image_mask is not None:
                bias = mx.where(
                    image_mask[..., None], self.e_score_correction_bias_vl, bias
                )
            inds, weights = _expert_select(
                logits,
                bias,
                self.top_k,
                self.routed_scaling_factor,
                self.norm_topk_prob,
                self.scoring_func,
            )

        return inds, weights


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        intermediate_size: Optional[int] = None,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit
        self.fp32_swiglu = False

    def __call__(self, x: mx.array) -> mx.array:
        paired = False
        if is_dspark_verify_armed():
            from omlx.patches.deepseek_v4.verify_qmv import (
                exact_verify_qmv_pair,
                pair_eligible,
            )

            paired = pair_eligible(self.gate_proj, self.up_proj, x)
        if paired:
            gate, up = exact_verify_qmv_pair(self.gate_proj, self.up_proj, x)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        if self.fp32_swiglu:
            hidden = _limited_swiglu(
                gate.astype(mx.float32),
                up.astype(mx.float32),
                self.swiglu_limit,
            ).astype(x.dtype)
        else:
            hidden = _limited_swiglu(gate, up, self.swiglu_limit)
        return self.down_proj(hidden)


class DeepseekV4MoE(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.gate = MoEGate(config, layer_idx)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=LimitedSwiGLU(config.swiglu_limit),
        )
        self.shared_experts = DeepseekV4MLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            swiglu_limit=config.swiglu_limit,
        )
        self.sharding_group = None

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.gate(x, input_ids)
        y = self.switch_mlp(x, inds, scores=scores)
        if y.ndim == scores.ndim + 1:
            y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class Compressor(nn.Module):

    def __init__(self, config: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.overlap = compress_ratio == 4
        self.out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, self.out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
            freq_scale=compress_ratio,
        )

    def project(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        return self.wkv(x), self.wgate(x)

    def consume(
        self,
        kv: mx.array,
        gate: mx.array,
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
    ) -> mx.array:
        B, _, _ = kv.shape
        if pool_cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
            pool_base = offset
        else:
            ready_kv, ready_gate, pool_base = pool_cache.accumulate_windows(
                kv, gate, offset
            )

        if ready_kv.size == 0:
            new_pooled = mx.zeros((B, 0, self.head_dim), dtype=kv.dtype)
        else:
            compress_func = (
                _overlap_compress_kv if self.overlap else _simple_compress_kv
            )
            kv = mx.unflatten(ready_kv, 1, (-1, self.compress_ratio))
            gate = mx.unflatten(ready_gate, 1, (-1, self.compress_ratio))

            # Overlap (ratio==4) pools each window from its own lane-B plus the
            # previous window's lane-A. _overlap_compress_kv gets lane-A via
            # `kv_a[:, :-1]` (window-axis shift), which collapses to zero-
            # padding when only one window is in view (every decode step, and
            # the first window of any prefill/verify chunk). Prepend the last
            # completed window carried in the cache so the shift sees a real
            # predecessor; drop the prepend's own (zero lane-A) pooled output.
            # Mirrors native DS4's rolling state_kv double buffer
            # (ds4.c compressor_decode_one). The carry is per batch row: rows
            # without a valid prev come back -inf gated so the kernel masks
            # their lane-A exactly like its own first-window padding. rope
            # runs on the current windows only with pool_base, so positions
            # stay aligned.
            prev_kv = prev_gate = None
            if self.overlap and pool_cache is not None:
                prev_kv, prev_gate = pool_cache.prev_for_prepend()
            if prev_kv is not None:
                kv = mx.concatenate([prev_kv, kv], axis=1)
                gate = mx.concatenate([prev_gate, gate], axis=1)
                new_pooled = compress_func(kv, gate, self.ape, self.head_dim)
                new_pooled = new_pooled[:, 1:]
            else:
                new_pooled = compress_func(kv, gate, self.ape, self.head_dim)

            if self.overlap and pool_cache is not None:
                pool_cache.store_prev(kv, gate, dropped=1 if prev_kv is not None else 0)

            new_pooled = self.norm(new_pooled)
            new_pooled = self.rope(
                new_pooled[:, None],
                offset=pool_base,
            ).squeeze(1)

        if pool_cache is not None:
            new_pooled = pool_cache.update_and_fetch(new_pooled)

        return new_pooled

    def __call__(
        self,
        x: mx.array,
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
    ) -> mx.array:
        return self.consume(*self.project(x), pool_cache, offset)


@lru_cache(maxsize=512)
def _rotating_snapshot_indices(
    ring_size: int,
    slots: Tuple[int, ...],
) -> mx.array:
    """Reuse the immutable physical-ring gather plan across model layers."""
    indices = []
    for row in range(len(slots)):
        snapshot = list(range(ring_size))
        for update in range(row + 1):
            snapshot[slots[update]] = ring_size + update
        indices.append(snapshot)
    return mx.array(indices, dtype=mx.uint32)


@dataclass(frozen=True)
class _RotatingVerifyView:
    source: mx.array
    indices: mx.array


def _stage_full_rotating_verify_view(
    cache: Optional[RotatingKVCache],
    kv: mx.array,
) -> Optional[_RotatingVerifyView]:
    """Advance a full ring and retain its M physical row maps.

    The returned source remains immutable while the cache is rebound to its
    final state. Consumers can either gather the snapshots or let the native
    Steel loader resolve the physical row map inside each GEMM.
    """
    if cache is None:
        return None

    logical_size = int(getattr(cache, "_offset", cache.offset))
    full_ring = (
        cache.keys is not None
        and int(cache.keys.shape[2]) == int(cache.max_size)
        and logical_size >= int(cache.max_size)
    )
    if not full_ring:
        return None

    from omlx.patches.mlx_lm_mtp.cache_rollback import (
        stage_functional_rotating_update,
    )

    steps = int(kv.shape[2])
    empty_values = mx.zeros((*kv.shape[:-1], 0), dtype=kv.dtype)
    stage_functional_rotating_update(cache, kv, empty_values)
    ring_size = int(cache.max_size)
    write_idx = int(cache._idx)
    slots = []
    is_batch_cache = hasattr(cache, "rotated")
    if is_batch_cache:
        rotated = bool(cache.rotated)
        rotated_writes = 0
        for _ in range(steps):
            if write_idx == ring_size:
                write_idx = 0
                rotated = True
            if rotated:
                rotated_writes += 1
            slots.append(write_idx)
            write_idx += 1
    else:
        for _ in range(steps):
            if write_idx == ring_size:
                write_idx = int(cache.keep)
            slots.append(write_idx)
            write_idx += 1

    source = mx.concatenate([cache.keys, kv], axis=2)
    index_array = _rotating_snapshot_indices(ring_size, tuple(slots))
    final_keys = mx.take(source, index_array[-1], axis=2)

    cache.keys = final_keys
    cache._idx = write_idx
    cache.offset = cache.offset + steps
    if is_batch_cache:
        cache._offset += steps
        cache.rotated = rotated
        if rotated_writes:
            cache.left_padding = cache.left_padding - rotated_writes
        cache.keys = mx.depends(cache.keys, (cache.left_padding, cache.offset))
    return _RotatingVerifyView(source=source[0, 0], indices=index_array)


def _materialize_rotating_verify_rows(
    view: _RotatingVerifyView,
) -> List[mx.array]:
    snapshots = mx.take(view.source, view.indices, axis=0)
    return [
        snapshots[idx : idx + 1][:, None] for idx in range(int(view.indices.shape[0]))
    ]


def _consume_rotating_verify_rows(
    cache: Optional[RotatingKVCache],
    kv: mx.array,
) -> List[mx.array]:
    """Advance a physical ring once and expose every exact M=1 snapshot."""
    view = _stage_full_rotating_verify_view(cache, kv)
    if view is not None:
        return _materialize_rotating_verify_rows(view)

    steps = int(kv.shape[2])
    if cache is None:
        return [kv[..., : idx + 1, :] for idx in range(steps)]

    empty_values = mx.zeros((*kv.shape[:-1], 0), dtype=kv.dtype)
    rows = []
    for idx in range(steps):
        row_kv, _ = cache.update_and_fetch(
            kv[..., idx : idx + 1, :],
            empty_values[..., idx : idx + 1, :],
        )
        rows.append(row_kv + 0)
    return rows


def _consume_verify_rows(
    compressor: Compressor,
    kv: mx.array,
    gate: mx.array,
    pool_cache: Optional[PoolingCache],
    offset: Union[int, mx.array],
) -> List[mx.array]:
    """Consume one short verify block and expose its M decode snapshots."""
    steps = int(kv.shape[1])
    if pool_cache is None:
        rows = []
        for idx in range(steps):
            rows.append(
                compressor.consume(
                    kv[:, idx : idx + 1],
                    gate[:, idx : idx + 1],
                    None,
                    offset + idx,
                )
            )
        return rows

    remainder = pool_cache.remainder
    old_remainder = int(remainder[0] if isinstance(remainder, list) else remainder)
    first_completion = compressor.compress_ratio - old_remainder - 1
    if first_completion + compressor.compress_ratio < steps:
        # A full DSpark verification block can complete two ratio-4 pooling
        # windows.  Materializing only the final pool would expose the second
        # pooled row to earlier query positions.  Consume those rare blocks
        # one row at a time so both the snapshots and final cache match M=1
        # decode exactly.
        return [
            compressor.consume(
                kv[:, idx : idx + 1],
                gate[:, idx : idx + 1],
                pool_cache,
                offset + idx,
            )
            for idx in range(steps)
        ]

    old_pooled = pool_cache.pooled
    if old_pooled is None:
        old_pooled = mx.zeros((kv.shape[0], 0, compressor.head_dim), dtype=kv.dtype)

    final_pooled = compressor.consume(kv, gate, pool_cache, offset)
    return [
        (final_pooled if idx >= first_completion else old_pooled)
        for idx in range(steps)
    ]


def _stable_topk_indices(scores: mx.array, k: int) -> mx.array:
    """Select top-k with a deterministic position tie-break and order."""
    partition = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    selected = mx.take_along_axis(scores, partition, axis=-1)
    threshold = mx.min(selected, axis=-1, keepdims=True)

    size = scores.shape[-1]
    positions = mx.arange(size, dtype=mx.uint32)
    region = mx.where(
        scores > threshold,
        0,
        mx.where(scores == threshold, 1, 2),
    ).astype(mx.uint32)
    # Keys are unique: all strictly-better scores come first, then cutoff
    # ties in time order, then the remainder.  A second partition therefore
    # has a deterministic selected set even though its internal order is not.
    keys = region * size + positions
    indices = mx.argpartition(keys, kth=k - 1, axis=-1)[..., :k]
    return mx.sort(indices, axis=-1)


class Indexer(nn.Module):
    def __init__(self, config: ModelArgs, compress_ratio: int):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(config, compress_ratio, self.head_dim)
        self.scale = self.head_dim**-0.5
        self._m2_mma_score = _dsv4f_m2_mma_score_enabled(
            config, compress_ratio
        )

    def __call__(
        self,
        x: mx.array,
        q_residual: mx.array,
        position_rope: DeepseekV4RoPE,
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
        compressor_projection: Optional[Tuple[mx.array, mx.array]] = None,
        projected_q: Optional[mx.array] = None,
        projected_weights: Optional[mx.array] = None,
    ):
        B, L, _ = x.shape
        if compressor_projection is None:
            pooled = self.compressor(x, pool_cache, offset)
        else:
            pooled = self.compressor.consume(
                *compressor_projection,
                pool_cache,
                offset,
            )
        if pooled.shape[1] == 0:
            return None

        if projected_q is None:
            q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
            q = q.transpose(0, 2, 1, 3)
            q = position_rope(q, offset)
        else:
            q = projected_q

        pmask = pool_cache.make_mask(L, offset) if pool_cache is not None else None
        k = min(self.index_topk, pooled.shape[1])

        if native_indexer_shape_eligible(
            query_tokens=L,
            pooled_tokens=pooled.shape[1],
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            index_topk=self.index_topk,
            dtype_supported=q.dtype in (mx.float16, mx.bfloat16),
        ):
            try:
                from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

                _have_native = native_indexer_available()
                if not _have_native and not native_indexer_disabled():
                    # Warn only when the kernels are genuinely missing -- the
                    # shape predicates above legitimately skip small/early
                    # chunks, so warning outside this branch cries wolf. The
                    # miss is otherwise completely silent, and the MLX
                    # fallback's prefill slope is ~4x worse at long context.
                    global _DEEPSEEK_V4_INDEXER_FALLBACK_WARNED
                    if not _DEEPSEEK_V4_INDEXER_FALLBACK_WARNED:
                        _DEEPSEEK_V4_INDEXER_FALLBACK_WARNED = True
                        logging.getLogger(__name__).warning(
                            "deepseek_v4: native dsa_indexer_scores/"
                            "dsa_topk_indices unavailable (glm_moe_dsa "
                            "extension not built); falling back to MLX. "
                            "Long-context prefill is several times slower "
                            "(rebuild with OMLX_WITH_CUSTOM_KERNEL=1)."
                        )
                if _have_native:
                    weights = (
                        self.weights_proj(x)
                        if projected_weights is None
                        else projected_weights
                    ).astype(q.dtype) * ((self.n_heads**-0.5) * self.scale)
                    # Fused pooled-ratio causal mask (lossless): the kernel
                    # epilogue writes the finfo(bf16).min sentinel itself
                    # when the mask is the plain 2-D PoolingCache ratio mask,
                    # bit-identical to the mx.where pass it replaces.
                    # Batched 3-D masks keep the where pass; pmask is None
                    # stays unmasked as before.
                    _mask_ratio = 0
                    _mask_q_offset = 0
                    if (
                        pmask is not None
                        and pmask.ndim == 2
                        and isinstance(offset, int)
                        and type(pool_cache).__name__ == "PoolingCache"
                    ):
                        _mask_ratio = int(pool_cache.ratio)
                        _mask_q_offset = int(offset)
                    # v25 from-scratch MMA score kernel: 1.37x over Steel
                    # on the exact M2/DSV4F pairing, bit-exact incl. the
                    # fused pooled-ratio mask (validated across aligned and
                    # unaligned M/N and chunked-prefill offsets). Gated by
                    # fingerprint + OMLX_DSV4F_M2_MMA_SCORE + extension
                    # symbol; every other configuration keeps the Steel
                    # path below unchanged.
                    _use_mma = (
                        self._m2_mma_score
                        and getattr(glm_fast, "_EXT_MMA_SCORE", False)
                        and q.dtype == mx.bfloat16
                        and pooled.shape[1] >= 64
                    )
                    global _DEEPSEEK_V4_M2_MMA_SCORE_LOGGED
                    if _use_mma and not _DEEPSEEK_V4_M2_MMA_SCORE_LOGGED:
                        _DEEPSEEK_V4_M2_MMA_SCORE_LOGGED = True
                        logging.getLogger(__name__).info(
                            "deepseek_v4: DSV4F/M2 v25 MMA indexer score "
                            "kernel active (zero-per-head-barrier, "
                            "1.37x over Steel)"
                        )
                    if _use_mma:
                        scores4 = glm_fast.dsa_indexer_scores_mma(
                            q,
                            pooled[:, None],
                            weights,
                            mask_ratio=_mask_ratio,
                            mask_q_offset=_mask_q_offset,
                        )
                    else:
                        scores4 = glm_fast.dsa_indexer_scores(
                            q,
                            pooled[:, None],
                            weights,
                            causal=False,
                            mask_ratio=_mask_ratio,
                            mask_q_offset=_mask_q_offset,
                        )
                    if _mask_ratio == 0 and pmask is not None:
                        scores4 = mx.where(
                            (pmask[:, None] if pmask.ndim == 3 else pmask[None, None]),
                            scores4,
                            mx.finfo(scores4.dtype).min,
                        )
                    indices = glm_fast.dsa_topk_indices(
                        scores4,
                        self.index_topk,
                        # The bucketed writer appends equal-threshold entries
                        # with atomics, so their membership depends on GPU
                        # scheduling.  ReLU indexer scores contain many exact
                        # zero ties; use the kernel's deterministic scan which
                        # resolves cutoff ties by temporal index.
                        bucketed=False,
                    )[:, 0]
                    return mx.sort(indices, axis=-1)
            except Exception as exc:
                disable_native_indexer()
                logging.getLogger(__name__).warning(
                    "DSV4 native indexer top-k failed; MLX fallback "
                    "(slower prefill) for the rest of this process: %s",
                    exc,
                    exc_info=True,
                )

        weights = (
            self.weights_proj(x) if projected_weights is None else projected_weights
        ).astype(mx.float32) * (self.n_heads**-0.5)
        weights = weights.swapaxes(-1, -2)[..., None]  # (B, H, L, 1)
        qf = q.astype(mx.float32)
        kf = pooled[:, None].swapaxes(-1, -2).astype(mx.float32)  # (B, 1, Dh, P)
        n_pool = pooled.shape[1]
        n_elems = qf.shape[0] * qf.shape[1] * qf.shape[2] * n_pool
        if n_elems < _INDEXER_MAX_ELEMS:
            # Single matmul: fastest, and the intermediate is safely indexable.
            scores = _indexer_head_reduce(qf @ kf, weights, self.scale)
        else:
            # Only past the int32 indexing limit is tiling worth its cost:
            # splitting the pooled axis adds matmul launches and a full-width
            # concatenate (measured: ~1.2x slower slope at 273k), but an
            # intermediate over 2**31 elements is silently zeroed by mlx's
            # int32 kernel indexing, which corrupts top-k selection. Choose
            # the largest tile that stays under the limit so the split is as
            # coarse as correctness allows.
            per_pool = max(1, qf.shape[0] * qf.shape[1] * qf.shape[2])
            tile = max(1024, min(_INDEXER_POOL_TILE, _INDEXER_MAX_ELEMS // per_pool))
            scores = mx.concatenate(
                [
                    _indexer_head_reduce(
                        qf @ kf[..., s : s + tile], weights, self.scale
                    )
                    for s in range(0, n_pool, tile)
                ],
                axis=-1,
            )
        if pmask is not None:
            scores = mx.where(
                pmask if pmask.ndim == 3 else pmask[None],
                scores,
                mx.finfo(scores.dtype).min,
            )
        return _stable_topk_indices(scores, k)


def _batch_indexer_rows(
    indexer: Indexer,
    pooled_rows: List[mx.array],
    projected_q: mx.array,
    projected_weights: mx.array,
) -> List[Optional[mx.array]]:
    """Select each decode row's pooled indices with grouped FP32 GEMMs."""
    global _DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED

    lengths = [int(row.shape[1]) for row in pooled_rows]
    if len(lengths) > 1 and min(lengths) > indexer.index_topk:
        # A ratio-4 boundary makes adjacent verify rows differ by one pooled
        # token. The score reduction is over the fixed 128-wide head, so
        # padding N does not alter any valid dot product. Mask the padded tail
        # before top-k and keep all rows in one native GEMM/top-k dispatch.
        max_length = max(lengths)
        padded_rows = []
        for row, length in zip(pooled_rows, lengths):
            if length < max_length:
                row = mx.concatenate(
                    [
                        row,
                        mx.zeros(
                            (row.shape[0], max_length - length, row.shape[2]),
                            dtype=row.dtype,
                        ),
                    ],
                    axis=1,
                )
            padded_rows.append(row)
        query_batch = projected_q[0].transpose(1, 0, 2)
        pooled_batch = mx.concatenate(padded_rows, axis=0)
        scores = rowwise_gemm(
            query_batch.astype(mx.float32),
            pooled_batch.astype(mx.float32),
            True,
        )
        scores = mx.maximum(scores, 0) * indexer.scale
        weights = projected_weights[0].astype(mx.float32)
        weights = weights * (indexer.n_heads**-0.5)
        scores = (scores * weights[..., None]).sum(axis=1)
        valid = mx.arange(max_length)[None] < mx.array(lengths)[:, None]
        scores = mx.where(valid, scores, mx.finfo(scores.dtype).min)
        indices = None
        if not _DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED:
            try:
                from omlx.custom_kernels.glm_moe_dsa import fast

                if fast.has_symbol("dspark_fp32_topk_indices"):
                    indices = fast.dspark_fp32_topk_indices(
                        scores,
                        indexer.index_topk,
                    )
            except Exception as exc:
                _DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED = True
                logging.getLogger(__name__).warning(
                    "DSV4 native DSpark top-k failed; stable-sort fallback "
                    "for the rest of this process: %s",
                    exc,
                    exc_info=True,
                )
        if indices is None:
            indices = _stable_topk_indices(scores, indexer.index_topk)
        indices = indices[:, None]
        return [indices[idx : idx + 1] for idx in range(len(lengths))]

    results: List[Optional[mx.array]] = []
    start = 0
    while start < len(pooled_rows):
        pooled_length = int(pooled_rows[start].shape[1])
        stop = start + 1
        while (
            stop < len(pooled_rows) and int(pooled_rows[stop].shape[1]) == pooled_length
        ):
            stop += 1

        if pooled_length == 0:
            results.extend([None] * (stop - start))
            start = stop
            continue

        if pooled_length <= indexer.index_topk:
            # Every pooled position is selected, so the exact temporally
            # ordered top-k result is independent of the query scores.
            # Avoid the QK GEMM and two argpartitions on the ratio-128
            # DSpark layers, where this is the common decode case.
            all_indices = mx.arange(pooled_length, dtype=mx.uint32)[None, None]
            results.extend([all_indices] * (stop - start))
            start = stop
            continue

        query_batch = projected_q[0, :, start:stop].transpose(1, 0, 2)
        pooled_batch = mx.concatenate(pooled_rows[start:stop], axis=0)
        scores = rowwise_gemm(
            query_batch.astype(mx.float32),
            pooled_batch.astype(mx.float32),
            True,
        )
        scores = mx.maximum(scores, 0) * indexer.scale
        weights = projected_weights[0, start:stop].astype(mx.float32)
        weights = weights * (indexer.n_heads**-0.5)
        scores = (scores * weights[..., None]).sum(axis=1)
        k = min(indexer.index_topk, pooled_length)
        indices = None
        if not _DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED and k == 512:
            try:
                from omlx.custom_kernels.glm_moe_dsa import fast

                if fast.has_symbol("dspark_fp32_topk_indices"):
                    indices = fast.dspark_fp32_topk_indices(scores, k)
            except Exception as exc:
                _DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED = True
                logging.getLogger(__name__).warning(
                    "DSV4 native DSpark top-k failed; stable-sort fallback "
                    "for the rest of this process: %s",
                    exc,
                    exc_info=True,
                )
        if indices is None:
            indices = _stable_topk_indices(scores, k)
        indices = indices[:, None]
        results.extend(indices[idx : idx + 1] for idx in range(stop - start))
        start = stop

    return results


class LocalAttention(nn.Module):
    """DeepSeek V4 attention with no KV compression."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.dspark = _is_dspark_model(config)
        self.layer_idx = layer_idx
        self.compress_ratio = 0
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.rope_theta,
            None,
            config.max_position_embeddings,
        )

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        *,
        _standard_mask: bool = False,
    ) -> mx.array:
        B, L, _ = x.shape
        offset = cache.offset if cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        sinks = self.attn_sink.astype(q.dtype)
        if is_dspark_verify_armed() and B == 1 and 1 < L <= 6:
            key_rows = _consume_rotating_verify_rows(cache, kv)
            out = _batched_m1_attention(q, key_rows, self.scale, sinks)
            out = _project_attention_output(self, out, offset)
            if self.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self.sharding_group)
            return out
        if cache is not None:
            kv, _ = cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

        if self.dspark and B == 1 and L == 1:
            out = exact_attention(q, [kv], self.scale, sinks)
        else:
            out = None
            if _standard_mask and B == 1 and L > 1:
                out = wsdpa_prefill(
                    q,
                    kv,
                    None,
                    sinks,
                    self.scale,
                    offset,
                    self.config.sliding_window,
                    1,
                )
            if out is None:
                out = scaled_dot_product_attention(
                    q,
                    kv,
                    kv,
                    cache=cache,
                    scale=self.scale,
                    mask=mask,
                    sinks=sinks,
                )
        out = _project_attention_output(self, out, offset)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class CompressedAttention(nn.Module):
    """DeepSeek V4 attention with pooled KV compression."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.dspark = _is_dspark_model(config)
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        # Compressed layers use Yarn-scaled RoPE
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = Compressor(config, self.compress_ratio, self.head_dim)

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        *,
        _standard_mask: bool = False,
    ) -> mx.array:
        B, L, _ = x.shape
        local_cache = cache[0] if cache is not None else None
        pool_cache = cache[1] if cache is not None else None
        offset = local_cache.offset if local_cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        sinks = self.attn_sink.astype(q.dtype)
        if is_dspark_verify_armed() and B == 1 and 1 < L <= 6:
            compressed_kv, compressed_gate = self.compressor.project(x)
            pooled_rows = _consume_verify_rows(
                self.compressor,
                compressed_kv,
                compressed_gate,
                pool_cache,
                offset,
            )
            local_rows = _consume_rotating_verify_rows(local_cache, kv)
            key_rows = [
                (
                    mx.concatenate([row, pooled[:, None]], axis=2)
                    if pooled.shape[1] > 0
                    else row
                )
                for row, pooled in zip(local_rows, pooled_rows)
            ]
            out = _batched_m1_attention(q, key_rows, self.scale, sinks)
            out = _project_attention_output(self, out, offset)
            if self.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self.sharding_group)
            return out
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

        pooled = self.compressor(x, pool_cache, offset)
        pooled_mask = None
        if pooled.shape[1] > 0:
            pooled_mask = (
                pool_cache.make_mask(L, offset) if pool_cache is not None else None
            )
        # The wsdpa and native kernels reconstruct the model's causal/sliding
        # masks from offsets; direct callers with custom masks stay on the
        # reference path.
        out = None
        if _standard_mask and B == 1 and L > 1:
            out = wsdpa_prefill(
                q,
                kv,
                pooled if pooled.shape[1] > 0 else None,
                sinks,
                self.scale,
                offset,
                self.config.sliding_window,
                self.compress_ratio,
            )
        if out is None and (
            self.config.use_native_ratio128_attention
            and self.compress_ratio == 128
            and _standard_mask
            and pooled.shape[1] > 0
            and L > 4
            and not (self.dspark and B == 1 and L == 1)
            and _native_sparse_attention_available()
        ):
            pooled_indices = mx.broadcast_to(
                mx.arange(pooled.shape[1], dtype=mx.uint32)[None, None],
                (B, L, pooled.shape[1]),
            )
            out = _sparse_pooled_attention(
                q,
                kv,
                pooled,
                pooled_indices,
                mask,
                pooled_mask,
                self.scale,
                sinks,
                q_offset=offset,
                compress_ratio=self.compress_ratio,
                local_window=self.config.sliding_window,
                decode_consistent=self.dspark,
                native_only=True,
                _standard_mask=_standard_mask,
            )
        if out is None:
            if pooled.shape[1] > 0:
                kv = mx.concatenate([kv, pooled[:, None]], axis=2)
            mask = _extend_mask(mask, pooled_mask, kv.shape[2])
            if self.dspark and B == 1 and L == 1:
                out = exact_attention(q, [kv], self.scale, sinks)
            else:
                out = scaled_dot_product_attention(
                    q,
                    kv,
                    kv,
                    cache=local_cache,
                    scale=self.scale,
                    mask=mask,
                    sinks=sinks,
                )
        out = _project_attention_output(self, out, offset)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class SparseCompressedAttention(nn.Module):
    """DeepSeek V4 attention with sparse indexed pooled KV compression."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.dspark = _is_dspark_model(config)
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
        self.indexer = Indexer(config, self.compress_ratio)

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        *,
        _standard_mask: bool = False,
    ) -> mx.array:
        B, L, _ = x.shape
        local_cache = cache[0] if cache is not None else None
        comp_cache = cache[1] if cache is not None else None
        idx_cache = cache[2] if cache is not None else None
        offset = local_cache.offset if local_cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        q_residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        sinks = self.attn_sink.astype(q.dtype)
        if is_dspark_verify_armed() and B == 1 and L <= 6:
            compressed_kv, compressed_gate = self.compressor.project(x)
            index_kv, index_gate = self.indexer.compressor.project(x)
            pooled_rows = _consume_verify_rows(
                self.compressor,
                compressed_kv,
                compressed_gate,
                comp_cache,
                offset,
            )
            index_pooled_rows = _consume_verify_rows(
                self.indexer.compressor,
                index_kv,
                index_gate,
                idx_cache,
                offset,
            )
            pooled = pooled_rows[-1]
            ring_view = _stage_full_rotating_verify_view(local_cache, kv)
            local_rows = (
                None
                if ring_view is not None
                else _consume_rotating_verify_rows(local_cache, kv)
            )

            if pooled is None or pooled.shape[1] == 0:
                if local_rows is None:
                    local_rows = _materialize_rotating_verify_rows(ring_view)
                out = _batched_m1_attention(q, local_rows, self.scale, sinks)
            elif pooled.shape[1] <= self.indexer.index_topk:
                if local_rows is None:
                    local_rows = _materialize_rotating_verify_rows(ring_view)
                key_rows = [
                    mx.concatenate([row, row_pooled[:, None]], axis=2)
                    for row, row_pooled in zip(local_rows, pooled_rows)
                ]
                out = _batched_m1_attention(q, key_rows, self.scale, sinks)
            else:
                index_q = self.indexer.wq_b(q_residual).reshape(
                    B,
                    L,
                    self.indexer.n_heads,
                    self.indexer.head_dim,
                )
                index_q = index_q.transpose(0, 2, 1, 3)
                index_q = self.rope(index_q, offset)
                index_weights = self.indexer.weights_proj(x)
                topk_rows = _batch_indexer_rows(
                    self.indexer,
                    index_pooled_rows,
                    index_q,
                    index_weights,
                )
                query_batch = q[0].transpose(1, 0, 2)[:, :, None]
                pooled_batch = mx.broadcast_to(
                    pooled,
                    (L, pooled.shape[1], pooled.shape[2]),
                )
                topk_batch = mx.concatenate(topk_rows, axis=0)
                use_ring_kernel = bool(
                    ring_view is not None
                    and ring_view.source.ndim == 2
                    and ring_view.source.shape[1] == 512
                    and ring_view.indices.ndim == 2
                    and ring_view.indices.shape == (L, 128)
                    and ring_view.indices.dtype == mx.uint32
                )
                if use_ring_kernel:
                    try:
                        from omlx.custom_kernels.glm_moe_dsa import fast

                        use_ring_kernel = (
                            fast.is_native_available()
                            and fast.has_symbol("dspark_ring_gemm")
                        )
                    except Exception:
                        use_ring_kernel = False
                set_dspark_verify_armed(False)
                try:
                    if use_ring_kernel:
                        batch_out = _sparse_pooled_ring_attention(
                            query_batch,
                            ring_view.source,
                            ring_view.indices,
                            pooled_batch,
                            topk_batch,
                            self.scale,
                            sinks,
                        )
                    else:
                        if local_rows is None:
                            local_rows = _materialize_rotating_verify_rows(ring_view)
                        local_batch = mx.concatenate(local_rows, axis=0)
                        batch_out = _sparse_pooled_attention(
                            query_batch,
                            local_batch,
                            pooled_batch,
                            topk_batch,
                            None,
                            None,
                            self.scale,
                            sinks,
                            decode_consistent=self.dspark,
                        )
                finally:
                    set_dspark_verify_armed(True)
                out = batch_out[:, :, 0].transpose(1, 0, 2)[None]

            out = _project_attention_output(self, out, offset)
            if self.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self.sharding_group)
            return out
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

        pooled = self.compressor(x, comp_cache, offset)
        pmask = comp_cache.make_mask(L, offset) if comp_cache is not None else None
        if 0 < pooled.shape[1] <= self.indexer.index_topk:
            index_pooled = self.indexer.compressor(x, idx_cache, offset)
            if index_pooled.shape[1] != pooled.shape[1]:
                raise RuntimeError(
                    "DeepSeek V4 attention/indexer pooling caches diverged"
                )
            topk = mx.broadcast_to(
                mx.arange(pooled.shape[1], dtype=mx.uint32)[None, None],
                (B, L, pooled.shape[1]),
            )
        else:
            topk = self.indexer(x, q_residual, self.rope, idx_cache, offset)
        sparse_mask = None
        if pmask is not None and topk is not None:
            sparse_mask = mx.take_along_axis(
                pmask[None] if pmask.ndim == 2 else pmask,
                topk,
                axis=2,
            )[:, None]

        if pooled.shape[1] == 0:
            if self.dspark and B == 1 and L == 1:
                out = exact_attention(q, [kv], self.scale, sinks)
            else:
                out = None
                if _standard_mask and B == 1 and L > 1:
                    out = wsdpa_prefill(
                        q,
                        kv,
                        None,
                        sinks,
                        self.scale,
                        offset,
                        self.config.sliding_window,
                        self.compress_ratio,
                    )
                if out is None:
                    out = scaled_dot_product_attention(
                        q,
                        kv,
                        kv,
                        cache=local_cache,
                        scale=self.scale,
                        mask=mask,
                        sinks=sinks,
                    )
        elif pooled.shape[1] <= self.indexer.index_topk:
            if self.dspark and B == 1 and L == 1:
                full_kv = mx.concatenate([kv, pooled[:, None]], axis=2)
                out = exact_attention(q, [full_kv], self.scale, sinks)
            else:
                out = None
                if _standard_mask and B == 1 and L > 1:
                    out = wsdpa_prefill(
                        q,
                        kv,
                        pooled,
                        sinks,
                        self.scale,
                        offset,
                        self.config.sliding_window,
                        self.compress_ratio,
                    )
                if out is None:
                    full_kv = mx.concatenate([kv, pooled[:, None]], axis=2)
                    mask = _extend_mask(mask, pmask, full_kv.shape[2])
                    out = scaled_dot_product_attention(
                        q,
                        full_kv,
                        full_kv,
                        cache=local_cache,
                        scale=self.scale,
                        mask=mask,
                        sinks=sinks,
                    )
        else:
            out = _sparse_pooled_attention(
                q,
                kv,
                pooled,
                topk,
                mask,
                sparse_mask,
                self.scale,
                sinks,
                q_offset=offset,
                compress_ratio=self.compress_ratio,
                local_window=self.config.sliding_window,
                decode_consistent=self.dspark,
                _standard_mask=_standard_mask,
            )

        out = _project_attention_output(self, out, offset)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


def v4_attention_factory(config: ModelArgs, layer_idx: int) -> nn.Module:
    """Instantiate the appropriate attention module for a given layer."""
    ratio = config.compress_ratios[layer_idx]
    if ratio == 0:
        return LocalAttention(config, layer_idx)
    if ratio == 128:
        return CompressedAttention(config, layer_idx)
    return SparseCompressedAttention(config, layer_idx)


class DeepseekV4Block(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn = v4_attention_factory(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        input_ids: mx.array,
        *,
        _standard_mask: bool = False,
    ) -> mx.array:
        residual = h
        x, post, comb = self.attn_hc(h)
        attn_input = self.attn_norm(x)
        x = self.attn(
            attn_input,
            mask=mask,
            cache=cache,
            _standard_mask=_standard_mask,
        )
        h = hc_expand(x, residual, post, comb)

        residual = h
        x, post, comb = self.ffn_hc(h)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return hc_expand(x, residual, post, comb)


class DeepseekV4Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV4Block(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HyperHead(config)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        *,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        safe_inputs = mx.where(inputs >= self.vocab_size, 0, inputs)
        h = self.embed_tokens(safe_inputs) if inputs_embeds is None else inputs_embeds
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
        )
        h = mx.contiguous(h)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        first_cache = cache[0]
        mask_cache = (
            first_cache[0] if isinstance(first_cache, CacheList) else first_cache
        )
        mask = create_attention_mask(
            h[:, :, 0, :],
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )
        mask, image_visibility = _apply_image_visibility_mask(
            inputs, mask, self.vocab_size
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            # Optimized kernels may reconstruct only the standard causal mask.
            h = layer(
                h,
                mask,
                layer_cache,
                inputs,
                _standard_mask=not image_visibility,
            )

        _materialize_cache_arrays(cache)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            cache_item = cache[-1]
            if isinstance(cache_item, CacheList):
                cache_item = cache_item[0]
            if cache_item is not None:
                cache_item.keys = mx.depends(cache_item.keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(self.hc_head(h))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._vision_group = mx.distributed.init()
        self._vision_rank = int(self._vision_group.rank())
        self._vision_inputs = None
        self._vision_blocks = {}
        if config.vision_n_layers > 0 and self._vision_rank == 0:
            from omlx.patches.deepseek_v4.vision_model import Aligner, ViT

            self.vision = ViT(config)
            self.aligner = Aligner(config)
            self.image_start = mx.zeros((config.hidden_size,))
            self.image_end = mx.zeros((config.hidden_size,))
            self.image_newline = mx.zeros((config.hidden_size,))
            self.image_pad = mx.zeros((config.hidden_size,))

    @property
    def is_vision_model(self) -> bool:
        return self.args.vision_n_layers > 0

    def set_vision_inputs(self, images) -> None:
        """Install one request's preprocessed images on coordinator only."""

        self._vision_inputs = images if self._vision_rank == 0 else None
        self._vision_blocks = {}

    def _vision_embeddings(self, inputs: mx.array, offset: int = 0) -> mx.array:
        import time

        logger = logging.getLogger(__name__)
        dtype = self.model.embed_tokens.weight.dtype
        shape = (inputs.shape[0], inputs.shape[1], self.args.hidden_size)
        encode_error = None
        if self._vision_rank == 0:
            try:
                if not self._vision_inputs:
                    raise RuntimeError(
                        "image sentinel tokens reached the model without "
                        "rank-zero vision inputs"
                    )
                safe = mx.where(inputs >= self.args.vocab_size, 0, inputs)
                embeddings = self.model.embed_tokens(safe)
                params = mx.stack(
                    [
                        self.image_start,
                        self.image_pad,
                        self.image_pad,
                        self.image_newline,
                        self.image_end,
                    ]
                )
                image_slices = _image_embedding_slices(
                    self._vision_inputs, offset, inputs.shape[1]
                )
                started = time.perf_counter()
                logger.info(
                    "deepseek_v4_vision stage=vision_encode_begin images=%d",
                    len(image_slices),
                )
                for image, destination, source in image_slices:
                    block = self._vision_blocks.get(image.start)
                    if block is None:
                        patches = mx.array(image.patches, dtype=mx.bfloat16)
                        features = self.aligner(
                            self.vision(patches, image.n_vit_h, image.n_vit_w),
                            image.n_vit_h,
                            image.n_vit_w,
                        )
                        features = features[
                            mx.array(image.permutation, dtype=mx.uint32)
                        ]
                        types = mx.array(image.types, dtype=mx.uint32)
                        block = params[types]
                        positions = mx.array(
                            [i for i, kind in enumerate(image.types) if kind == 2],
                            dtype=mx.uint32,
                        )
                        block = block.at[positions].add(features - block[positions])
                        self._vision_blocks[image.start] = block
                    current = embeddings[0, destination]
                    embeddings = embeddings.at[0, destination].add(
                        block[source] - current
                    )
                mx.eval(embeddings)
                logger.info(
                    "deepseek_v4_vision stage=vision_encode_complete images=%d "
                    "elapsed_ms=%.1f",
                    len(image_slices),
                    (time.perf_counter() - started) * 1000,
                )
            except Exception as exc:
                encode_error = exc
                embeddings = mx.zeros(shape, dtype=dtype)
                logger.exception(
                    "deepseek_v4_vision stage=vision_encode_failed rank=0"
                )
        else:
            embeddings = mx.zeros(shape, dtype=dtype)
        suffix_end = offset + inputs.shape[1]
        self._vision_inputs = (
            tuple(
                image
                for image in self._vision_inputs or ()
                if image.start + len(image.types) > suffix_end
            )
            if encode_error is None
            else None
        )
        pending_starts = {image.start for image in self._vision_inputs or ()}
        self._vision_blocks = {
            start: block
            for start, block in self._vision_blocks.items()
            if start in pending_starts
        }
        if self._vision_group.size() > 1:
            # Synchronize success before the large embedding collective so a
            # coordinator-side decode/encode failure cannot strand peers.
            local_status = mx.array(
                int(self._vision_rank == 0 and encode_error is None),
                dtype=mx.int32,
            )
            status = mx.distributed.all_sum(local_status, group=self._vision_group)
            mx.eval(status)
            if int(status.item()) != 1:
                message = "DeepSeek-V4 vision encoding failed on rank zero"
                if encode_error is not None:
                    raise RuntimeError(message) from encode_error
                raise RuntimeError(message)
            embeddings = mx.distributed.all_sum(
                embeddings, group=self._vision_group
            )
            # Rank zero's pipeline stage replaces this value with recv_like().
            # Materialize first so its lazy collective cannot be pruned while
            # peers wait for the embedding broadcast.
            mx.eval(embeddings)
        elif encode_error is not None:
            raise RuntimeError("DeepSeek-V4 vision encoding failed") from encode_error
        logger.info(
            "deepseek_v4_vision stage=multimodal_embeddings shape=%s dtype=%s "
            "sequence_length=%d",
            tuple(embeddings.shape),
            embeddings.dtype,
            int(embeddings.shape[1]),
        )
        return embeddings

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
        multimodal = self.is_vision_model and bool(
            mx.any(inputs >= self.args.vocab_size).item()
        )
        embeddings = (
            self._vision_embeddings(inputs, _cache_offset(cache))
            if multimodal
            else None
        )
        return self.lm_head(self.model(inputs, cache, inputs_embeds=embeddings))

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return not (
                "attn_sink" in k
                or "e_score_correction_bias" in k
                or ".attn_hc." in k
                or ".ffn_hc." in k
                or ".hc_head." in k
            )

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            ratio = layer.attn.compress_ratio
            if ratio == 0:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            elif isinstance(layer.attn, SparseCompressedAttention):
                # local + compressor pool + indexer pool
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=self.args.sliding_window),
                        PoolingCache(ratio),
                        PoolingCache(ratio),
                    )
                )
            else:
                # local + compressor pool
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=self.args.sliding_window),
                        PoolingCache(ratio),
                    )
                )
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers

        new_weights = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    if int(parts[1]) >= n_layers:
                        continue
                except ValueError:
                    pass
            new_weights[k] = v
        weights = new_weights

        new_weights = {}
        for k, v in weights.items():
            if "tid2eid" in k:
                new_weights[k] = v.astype(mx.int32)

            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue

            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ".ffn.experts." in wk
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[k + "s"] = v
                new_weights[wk] = weight.view(mx.uint32)
            elif weight.dtype == mx.uint8:
                new_weights[k + "s"] = mx.repeat(mx.repeat(v, 4, -1), 128, 0)
                new_weights[wk] = weight.view(mx.uint32)
            else:
                new_weights[k] = v
        weights = new_weights

        top_remap = {
            "embed.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        }
        for old, new in top_remap.items():
            if old in weights:
                weights[new] = weights.pop(old)

        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = "model." + k if k.startswith("layers.") else k
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
            skip = False
            for old, new in (
                (".hc_attn.", ".attn_hc."),
                (".hc_ffn.", ".ffn_hc."),
            ):
                if old in nk:
                    candidate = nk.replace(old, new)
                    if candidate in weights or candidate in remapped:
                        skip = True
                        break
                    nk = candidate
            if skip:
                continue
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            remapped[nk] = v
        weights = remapped

        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.ffn.experts"
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                for suffix in ("weight", "scales", "biases"):
                    key0 = f"{prefix}.0.{src}.{suffix}"
                    if key0 in weights:
                        stacked = [
                            weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[
                            f"model.layers.{layer_idx}.ffn.switch_mlp.{dst}.{suffix}"
                        ] = mx.stack(stacked)

        for key, value in list(weights.items()):
            if (
                ".ffn.switch_mlp." not in key
                or not key.endswith((".scales", ".biases"))
                or value.dtype != mx.bfloat16
            ):
                continue
            stem = key.rsplit(".", 1)[0]
            if (
                stem + ".weight" in weights
                and stem + ".scales" in weights
                and stem + ".biases" in weights
                and weights[stem + ".weight"].dtype == mx.uint32
            ):
                weights[key] = value.astype(mx.float16)

        # Reshape wo_a from nn.Linear (2D) to MultiLinear (3D) for all layers
        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.attn.wo_a"
            for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
                if key in weights and weights[key].ndim == 2:
                    weights[key] = weights[key].reshape(
                        self.args.o_groups, self.args.o_lora_rank, -1
                    )

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()
        for layer in self.model.layers:
            layer.attn.sharding_group = group
            layer.attn.wq_b = shard_linear(
                layer.attn.wq_b,
                "all-to-sharded",
                segments=self.args.o_groups,
                group=group,
            )
            shard_inplace(layer.attn.wo_a, "sharded-to-all", group=group)
            layer.attn.attn_sink = mx.split(layer.attn.attn_sink, N)[rank]
            layer.attn.n_heads //= N

            layer.ffn.sharding_group = group
            shard_inplace(
                layer.ffn.shared_experts.gate_proj, "all-to-sharded", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.down_proj, "sharded-to-all", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.up_proj, "all-to-sharded", group=group
            )
            shard_inplace(layer.ffn.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(layer.ffn.switch_mlp.down_proj, "sharded-to-all", group=group)
            shard_inplace(layer.ffn.switch_mlp.up_proj, "all-to-sharded", group=group)
