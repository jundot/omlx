# Copyright © 2026 Apple Inc.

import logging
import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_flatten, tree_map_with_path

from omlx.custom_kernels.nax import is_nax_available
from omlx.patches.deepseek_v4.decode_consistency import (
    is_armed as is_dspark_verify_armed,
)
from omlx.patches.deepseek_v4.decode_consistency import matmul as decode_matmul
from omlx.patches.deepseek_v4.hierarchical_indexer import hierarchical_topk
from omlx.patches.deepseek_v4.indexer_dispatch import (
    disable_native_indexer,
    native_indexer_available,
    native_indexer_disabled,
    native_indexer_shape_eligible,
)
from omlx.patches.deepseek_v4.qkv_prefill_bundle import (
    prefill_qkv_projection_bundle,
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
from .hyper_connection import (
    HyperConnection,
    HyperHead,
    hc_expand,
    hc_merge_branch,
    hc_residual_branch,
)
from .mla import MultiLinear
from .pipeline import PipelineMixin

_DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED = False
_DEEPSEEK_V4_B1_SCALAR_OFFSET = os.getenv(
    "OMLX_DSV4_B1_SCALAR_OFFSET", "1"
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED = False
# Fused sparse decode attention (decode_fast.sparse_attn_decode) replaces the
# composed rowwise-GEMM + logsumexp glue of _dspark_sparse_exact_attention
# with one Metal dispatch. The kernel is batch-row invariant by construction
# (independent per-row reduction), preserving the DSpark decode==verify
# contract. OMLX_DSV4_FUSED_SPARSE=0 forces the composed path.
_DEEPSEEK_V4_FUSED_SPARSE_DECODE_DISABLED = os.environ.get(
    "OMLX_DSV4_FUSED_SPARSE", "1"
).strip().lower() in ("0", "false", "off")
_DEEPSEEK_V4_FUSED_SPARSE_DECODE_FAILED = False
# OMLX_DSV4_RING=0 disables the dspark_ring_gemm verify fast path, forcing
# the per-row _sparse_pooled_attention route (fused or composed). Diagnostic
# escape hatch for A/B benchmarking.
_DEEPSEEK_V4_RING_VERIFY_DISABLED = os.environ.get(
    "OMLX_DSV4_RING", "1"
).strip().lower() in ("0", "false", "off")
# The exact row-wise decode path exists to make one-token target decoding
# bit-identical to DSpark's multi-row verification.  It is unnecessary when
# speculative decoding is disabled and is materially slower on TP-local head
# counts (24/32/40 on the supported 64-head checkpoint).  The MTP patch marks
# attention modules that need the consistency contract at model construction;
# this switch is a startup-time rollback to the legacy always-exact behavior.
_DEEPSEEK_V4_EXACT_DECODE = os.environ.get(
    "OMLX_DSV4_EXACT_DECODE", "0"
).strip().lower() in ("1", "true", "on")

# ``DeepseekV4Model.__call__`` normally materializes every cache update before
# returning.  That barrier is the conservative default for decode and for all
# non-cluster callers.  The distributed TP prefill A/B may temporarily capture
# those exact arrays instead, queue them with ``mx.async_eval``, and complete
# them at its own bounded depth-two boundary.  Thread-local state is required:
# model loading and unrelated engines can execute on other threads while one
# generation thread owns the experiment.
_CACHE_MATERIALIZATION_CAPTURE = threading.local()


def _exact_decode_required(attention: Any, batch: int, length: int) -> bool:
    return bool(
        attention.dspark
        and batch == 1
        and length == 1
        and (
            _DEEPSEEK_V4_EXACT_DECODE
            or getattr(attention, "_omlx_decode_consistent", False)
        )
    )


def _tp_partition_weights(
    group: mx.distributed.Group,
) -> Optional[Tuple[int, ...]]:
    """Validated unequal TP weights, or None for MLX's equal split."""

    raw = os.environ.get("OMLX_TP_SHARD_WEIGHTS", "").strip()
    if not raw:
        return None
    try:
        weights = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError("OMLX_TP_SHARD_WEIGHTS must contain integers") from exc
    if len(weights) != int(group.size()) or any(
        not 1 <= item <= 4096 for item in weights
    ):
        raise ValueError(
            "OMLX_TP_SHARD_WEIGHTS must contain one positive weight per TP rank"
        )
    if len(set(weights)) == 1:
        return None
    return weights


_PROJECTION_OWNER_LOGGED = False


def _projection_owner_rank(group: Any) -> Optional[int]:
    """Return the operator-qualified owner of replicated DS4 projections."""

    if group is None or int(group.size()) != 2:
        return None
    raw = os.environ.get("OMLX_DSV4_PROJECTION_OWNER_RANK", "off").strip().lower()
    if raw in {"", "off", "none", "false", "disabled"}:
        return None
    try:
        owner = int(raw)
    except ValueError:
        return None
    # The physical gate is specific to the signed M3:M5 3:5 placement. A
    # differently ordered or equal cluster must qualify its own owner first.
    if owner not in (0, 1) or _tp_partition_weights(group) != (3, 5):
        return None
    return owner


def _owned_projection_bank(
    x: mx.array,
    modules: Tuple[Any, ...],
    group: Any,
) -> Optional[Tuple[mx.array, ...]]:
    """Compute replicated input projections once and send exact BF16 views.

    Rank zero's physical M3 gate executes the full DS4 projection shapes much
    faster than the M5. Both ranks enter one symmetric all-sum: the owner
    contributes its original packed arrays and the peer contributes exact
    zeros. Unlike selecting the owner's own all-gather slice, this collective
    cannot be optimized away on only one rank.
    """

    owner = _projection_owner_rank(group)
    if owner is None or not modules:
        return None
    rank = int(group.rank())
    widths = tuple(int(module.weight.shape[0]) for module in modules)
    total = sum(widths)
    if any(width < 1 for width in widths) or x.dtype not in (mx.bfloat16, mx.float16):
        return None
    if rank == owner:
        values = tuple(module(x) for module in modules)
        packed = mx.concatenate(values, axis=-1)
    else:
        shape = (*x.shape[:-1], total)
        # A free constant has no edge to this layer, so MLX may enqueue later
        # placeholder reductions ahead of the owner's real projection graphs.
        # Tie the placeholder to the replicated layer input to preserve the
        # same per-layer collective order without adding arithmetic.
        packed = mx.depends(mx.zeros(shape, dtype=x.dtype), x)
    packed = mx.distributed.all_sum(packed, group=group)
    boundaries = []
    cursor = 0
    for width in widths[:-1]:
        cursor += width
        boundaries.append(cursor)
    outputs = tuple(mx.split(packed, boundaries, axis=-1))
    global _PROJECTION_OWNER_LOGGED
    if not _PROJECTION_OWNER_LOGGED:
        _PROJECTION_OWNER_LOGGED = True
        logging.getLogger(__name__).info(
            "DeepSeek V4 replicated projections owned by rank %d "
            "(rank=%d packed_width=%d)",
            owner,
            rank,
            total,
        )
    return outputs


def _validated_ds4_tp_weights(
    args: Any,
    group: mx.distributed.Group,
) -> Optional[Tuple[int, ...]]:
    weights = _tp_partition_weights(group)
    if weights is not None:
        heads_per_group = args.num_attention_heads // args.o_groups
        if sum(weights) != heads_per_group:
            raise ValueError(
                "unequal DS4 TP weights must sum to the heads in each output group"
            )
    return weights


def _validated_ds4_moe_tp_weights(
    args: Any,
    group: mx.distributed.Group,
    outer_weights: Optional[Tuple[int, ...]],
) -> Optional[Tuple[int, ...]]:
    """Optional routed-MoE-only split over a signed unequal outer TP plan."""

    raw = os.environ.get("OMLX_TP_MOE_SHARD_WEIGHTS", "").strip()
    if not raw:
        return outer_weights
    try:
        weights = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError("OMLX_TP_MOE_SHARD_WEIGHTS must contain integers") from exc
    if len(weights) != int(group.size()) or any(
        not 1 <= item <= 4096 for item in weights
    ):
        raise ValueError(
            "OMLX_TP_MOE_SHARD_WEIGHTS must contain one positive weight per TP rank"
        )
    if outer_weights is None:
        # A conservative signed-equal plan may still qualify a mixed layout by
        # declaring the non-routed split explicitly. In that form the plan's
        # 4:4 weight budget safely covers the routed 4:4 banks, while the small
        # attention/shared shift remains inside the existing tolerance.
        if not os.environ.get("OMLX_TP_NON_MOE_SHARD_WEIGHTS", "").strip():
            raise ValueError(
                "OMLX_TP_MOE_SHARD_WEIGHTS requires either a signed unequal "
                "outer plan or OMLX_TP_NON_MOE_SHARD_WEIGHTS"
            )
        expected_total = int(args.num_attention_heads) // int(args.o_groups)
    else:
        expected_total = sum(outer_weights)
    if sum(weights) != expected_total:
        raise ValueError(
            "OMLX_TP_MOE_SHARD_WEIGHTS must sum to the signed outer TP weights"
        )

    intermediate = int(getattr(args, "moe_intermediate_size", 0) or 0)
    total = expected_total
    if intermediate <= 0 or intermediate % total:
        raise ValueError(
            "routed MoE intermediate size is not divisible by the override sum"
        )
    boundaries = [sum(weights[:rank]) for rank in range(1, len(weights))]
    if any(intermediate * boundary % total for boundary in boundaries):
        raise ValueError("routed MoE override does not preserve model boundaries")
    local_widths = [intermediate * weight // total for weight in weights]
    if any(width % 32 for width in local_widths):
        raise ValueError(
            "routed MoE override must preserve 32-value MXFP4 quant groups"
        )
    return weights


def _validated_ds4_non_moe_tp_weights(
    args: Any,
    group: mx.distributed.Group,
    outer_weights: Optional[Tuple[int, ...]],
) -> Optional[Tuple[int, ...]]:
    """Optional attention/shared/vocab split over a conservative outer plan."""

    raw = os.environ.get("OMLX_TP_NON_MOE_SHARD_WEIGHTS", "").strip()
    if not raw:
        return outer_weights
    try:
        weights = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError("OMLX_TP_NON_MOE_SHARD_WEIGHTS must contain integers") from exc
    if len(weights) != int(group.size()) or any(
        not 1 <= item <= 4096 for item in weights
    ):
        raise ValueError(
            "OMLX_TP_NON_MOE_SHARD_WEIGHTS must contain one positive weight per TP rank"
        )
    units = int(args.num_attention_heads) // int(args.o_groups)
    if sum(weights) != units:
        raise ValueError(
            "OMLX_TP_NON_MOE_SHARD_WEIGHTS must sum to the heads in each "
            "DS4 output group"
        )
    intermediate = int(getattr(args, "moe_intermediate_size", 0) or 0)
    if intermediate <= 0 or intermediate % units:
        raise ValueError(
            "DS4 shared-expert intermediate size is not divisible by the "
            "non-MoE override sum"
        )
    boundaries = [sum(weights[:rank]) for rank in range(1, len(weights))]
    if any(intermediate * boundary % units for boundary in boundaries):
        raise ValueError("non-MoE override does not preserve shared-expert boundaries")
    local_widths = [intermediate * weight // units for weight in weights]
    if any(width % 32 for width in local_widths):
        raise ValueError("non-MoE override must preserve 32-value MXFP4 quant groups")
    return None if len(set(weights)) == 1 else weights


def _weighted_segment_slice(
    value: mx.array,
    *,
    axis: int,
    segments: int,
    rank: int,
    weights: Tuple[int, ...],
) -> mx.array:
    """Slice every fused segment by the same exact integer partition."""

    axis %= value.ndim
    if segments < 1 or value.shape[axis] % segments:
        raise ValueError("asymmetric TP segment dimension is not divisible")
    total = sum(weights)
    before = sum(weights[:rank])
    after = before + weights[rank]
    pieces = []
    for segment in mx.split(value, segments, axis=axis):
        length = int(segment.shape[axis])
        if length * before % total or length * after % total:
            raise ValueError(
                "asymmetric TP boundary does not preserve a tensor/quant group"
            )
        start = length * before // total
        stop = length * after // total
        index = [slice(None)] * value.ndim
        index[axis] = slice(start, stop)
        pieces.append(segment[tuple(index)])
    return mx.contiguous(mx.concatenate(pieces, axis=axis))


def _asymmetric_shard_parameters(
    module: nn.Module,
    sharding: str,
    *,
    group: mx.distributed.Group,
    weights: Tuple[int, ...],
    segments: int = 1,
) -> Dict[str, Any]:
    """MLX distributed-linear parameter slicing with unequal rank widths."""

    rank = int(group.rank())

    def shard(path: str, value: Any) -> Any:
        if not isinstance(value, mx.array):
            return value
        if sharding == "all-to-sharded":
            axis = -1 if path.endswith("bias") else max(value.ndim - 2, 0)
        elif sharding == "sharded-to-all":
            if path.endswith("bias"):
                return value
            axis = -1
        else:
            raise ValueError(f"unsupported asymmetric TP sharding {sharding!r}")
        return _weighted_segment_slice(
            value,
            axis=axis,
            segments=segments,
            rank=rank,
            weights=weights,
        )

    return tree_map_with_path(shard, module.parameters())


def _shard_linear_weighted(
    module: nn.Module,
    sharding: str,
    *,
    group: mx.distributed.Group,
    weights: Optional[Tuple[int, ...]],
    segments: int = 1,
) -> nn.Module:
    replacement = shard_linear(
        module,
        sharding,
        segments=segments,
        group=group,
    )
    if weights is not None:
        replacement.update(
            _asymmetric_shard_parameters(
                module,
                sharding,
                group=group,
                weights=weights,
                segments=segments,
            )
        )
    return replacement


def _shard_inplace_weighted(
    module: nn.Module,
    sharding: str,
    *,
    group: mx.distributed.Group,
    weights: Optional[Tuple[int, ...]],
    segments: int = 1,
) -> None:
    if weights is None:
        shard_inplace(module, sharding, segments=segments, group=group)
        return
    module.update(
        _asymmetric_shard_parameters(
            module,
            sharding,
            group=group,
            weights=weights,
            segments=segments,
        )
    )


# Fused decode indexer (glm_moe_dsa.dsa_decode_scores, 64-head instantiation)
# replaces the DSpark verify indexer score pipeline's fp32 cast of the pooled
# context, the full-width rowwise GEMM, and the (rows, 64, P) fp32 score
# sheet with one Metal scan per row that streams K exactly once in its cache
# dtype and accumulates in fp32. Scores stay fp32 with fp32 head weights, so
# the downstream dspark_fp32_topk_indices/_stable_topk_indices selection
# semantics are unchanged. OMLX_DSV4_FUSED_DECODE_INDEXER=0 forces the fp32
# reference path.
_DEEPSEEK_V4_FUSED_DECODE_INDEXER_ENV_DISABLED = os.environ.get(
    "OMLX_DSV4_FUSED_DECODE_INDEXER", "1"
).strip().lower() in ("0", "false", "off")
_DEEPSEEK_V4_FUSED_DECODE_INDEXER_FAILED = False
_DEEPSEEK_V4_NAX_OA_PREFILL = os.getenv(
    "OMLX_DSV4_NAX_OA_PREFILL", "0"
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_NAX_OA_PREFILL_LOGGED = False
_DEEPSEEK_V4_ATTN_FINALIZER_PREFILL = os.getenv(
    "OMLX_DSV4_ATTN_FINALIZER_PREFILL", "0"
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_ATTN_FINALIZER_VERIFY = os.getenv(
    "OMLX_DSV4_ATTN_FINALIZER_VERIFY", "0"
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_ATTN_FINALIZER_PREFILL_LOGGED = False
_DEEPSEEK_V4_OUTPUT_CHAIN_PREFILL = os.getenv(
    "OMLX_DSV4_OUTPUT_CHAIN_PREFILL", "0"
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_OUTPUT_CHAIN_EQUAL_TP = os.getenv(
    "OMLX_DSV4_OUTPUT_CHAIN_EQUAL_TP", "1"
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_OUTPUT_CHAIN_PREFILL_LOGGED = False
_DEEPSEEK_V4_VERIFY_BATCHED_OA_PREPARE = os.getenv(
    "OMLX_DSV4_VERIFY_BATCHED_OA_PREPARE", "0"
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_VERIFY_BATCHED_OA_PREPARE_LOGGED = False
_DEEPSEEK_V4_HC_RESIDUAL_OVERLAP = os.getenv(
    "OMLX_DSV4_HC_RESIDUAL_OVERLAP", "0"
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_CANONICAL_WIDE_PREFILL = os.getenv(
    "OMLX_DSV4_CANONICAL_WIDE_PREFILL", "0"
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_CANONICAL_WIDE_PREFILL_STATE = threading.local()
_DEEPSEEK_V4_CANONICAL_WIDE_PREFILL_LOGGED = False
_DEEPSEEK_V4_QKV_BUNDLE_DECODE = os.getenv(
    "OMLX_DSV4_QKV_BUNDLE_DECODE", "1"
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_QKV_BUNDLE_DECODE_LOGGED = False
_DEEPSEEK_V4_QKV_BUNDLE_ALL_SCHEDULES = os.getenv(
    "OMLX_DSV4_QKV_BUNDLE_ALL_SCHEDULES", "0"
).strip().lower() in ("1", "true", "on", "yes")


def _decode_qkv_projection_bundle(
    attn: nn.Module,
    x: mx.array,
) -> Optional[Tuple[mx.array, ...]]:
    """Exact DS4 attention-input bundle for qualified decode/prefill tiles."""

    prefill = prefill_qkv_projection_bundle(attn, x)
    if prefill is not None:
        return prefill

    if (
        not _DEEPSEEK_V4_QKV_BUNDLE_DECODE
        or getattr(attn, "training", False)
        or is_dspark_verify_armed()
        or tuple(x.shape) != (1, 1, 4096)
        or x.dtype != mx.bfloat16
    ):
        return None
    config = getattr(attn, "config", None)
    try:
        if config is None or not _dsv4f_exact_config(config, 4):
            return None
    except (AttributeError, TypeError, ValueError):
        return None
    ratio = int(getattr(attn, "compress_ratio", -1))
    if ratio not in (0, 4, 128):
        return None
    if ratio in (0, 128):
        if not _DEEPSEEK_V4_QKV_BUNDLE_ALL_SCHEDULES:
            return None
        try:
            if mx.device_info().get("device_name") != "Apple M5 Max":
                return None
        except Exception:
            return None
    indexer = getattr(attn, "indexer", None)
    compressor = getattr(attn, "compressor", None)
    index_compressor = getattr(indexer, "compressor", None)
    common = (getattr(attn, "wq_a", None), getattr(attn, "wkv", None))
    compressor_pair = (
        getattr(compressor, "wkv", None),
        getattr(compressor, "wgate", None),
    )
    index_pair = (
        getattr(index_compressor, "wkv", None),
        getattr(index_compressor, "wgate", None),
    )
    modules = (
        common
        if ratio == 0
        else common + compressor_pair
        if ratio == 128
        else common + compressor_pair + index_pair
    )
    if any(
        module is None or not callable(getattr(module, "get", None))
        for module in modules
    ):
        return None
    q_a, wkv = common
    if not (
        getattr(q_a, "group_size", None) == 32
        and getattr(q_a, "bits", None) == 8
        and getattr(q_a, "mode", None) == "mxfp8"
        and getattr(wkv, "group_size", None) == 32
        and getattr(wkv, "bits", None) == 8
        and getattr(wkv, "mode", None) == "mxfp8"
    ):
        return None
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

        flat = mx.contiguous(x.reshape(1, 4096))
        args = (
            flat,
            q_a["weight"],
            q_a["scales"],
            wkv["weight"],
            wkv["scales"],
        )
        if ratio == 0:
            if not glm_fast.has_symbol("deepseek_v4_qkv_pair_b1"):
                return None
            packed = glm_fast.deepseek_v4_qkv_pair_b1(*args)
            boundaries = (1024,)
        elif ratio == 128:
            if not glm_fast.has_symbol("deepseek_v4_qkv_compressor128_bundle_b1"):
                return None
            compressor_kv, compressor_gate = compressor_pair
            packed = glm_fast.deepseek_v4_qkv_compressor128_bundle_b1(
                *args,
                compressor_kv["weight"],
                compressor_gate["weight"],
            )
            boundaries = (1024, 1536, 2048)
        else:
            if not glm_fast.has_symbol("deepseek_v4_qkv_compressor_bundle_b1"):
                return None
            compressor_kv, compressor_gate = compressor_pair
            index_kv, index_gate = index_pair
            packed = glm_fast.deepseek_v4_qkv_compressor_bundle_b1(
                *args,
                compressor_kv["weight"],
                compressor_gate["weight"],
                index_kv["weight"],
                index_gate["weight"],
            )
            boundaries = (1024, 1536, 2560, 3584, 3840)
        packed = packed.reshape(1, 1, packed.shape[-1])
    except (KeyError, TypeError, ValueError):
        return None
    global _DEEPSEEK_V4_QKV_BUNDLE_DECODE_LOGGED
    if not _DEEPSEEK_V4_QKV_BUNDLE_DECODE_LOGGED:
        _DEEPSEEK_V4_QKV_BUNDLE_DECODE_LOGGED = True
        logging.getLogger(__name__).info(
            "DeepSeek V4 exact B1 Q/KV/compressor bundles active "
            "(ratio=%d, one dispatch)",
            ratio,
        )
    return tuple(mx.split(packed, boundaries, axis=-1))


def _verify_q_a_kv_bank(
    attn: nn.Module,
    x: mx.array,
) -> Optional[Tuple[mx.array, mx.array]]:
    """One target-row-exact MXFP8 dispatch for DS4 verify Q-A + raw KV."""

    if (
        not is_dspark_verify_armed()
        or x.dtype != mx.bfloat16
        or x.ndim != 3
        or x.shape[0] != 1
        or not 2 <= int(x.shape[1]) <= 6
        or int(x.shape[2]) != 4096
    ):
        return None
    modules = (getattr(attn, "wq_a", None), getattr(attn, "wkv", None))
    if any(
        module is None
        or not callable(getattr(module, "get", None))
        or getattr(module, "bits", None) != 8
        or getattr(module, "group_size", None) != 32
        or getattr(module, "mode", None) != "mxfp8"
        or module.get("biases") is not None
        or module.get("weight") is None
        or module.get("scales") is None
        for module in modules
    ):
        return None
    q_a, wkv = modules
    assert q_a is not None and wkv is not None
    if (
        tuple(q_a.weight.shape) != (1024, 1024)
        or tuple(q_a.scales.shape) != (1024, 128)
        or tuple(wkv.weight.shape) != (512, 1024)
        or tuple(wkv.scales.shape) != (512, 128)
    ):
        return None
    weight = getattr(attn, "_omlx_verify_q_a_kv_weight", None)
    scales = getattr(attn, "_omlx_verify_q_a_kv_scales", None)
    if weight is None or scales is None:
        weight = mx.concatenate([q_a.weight, wkv.weight], axis=0)
        scales = mx.concatenate([q_a.scales, wkv.scales], axis=0)
        mx.eval(weight, scales)
        object.__setattr__(attn, "_omlx_verify_q_a_kv_weight", weight)
        object.__setattr__(attn, "_omlx_verify_q_a_kv_scales", scales)
    from omlx.patches.deepseek_v4.verify_qmv import exact_verify_mxfp8_bank

    packed = exact_verify_mxfp8_bank(weight, scales, x)
    q_a_out, kv_out = mx.split(packed, (1024,), axis=-1)
    return q_a_out, kv_out


def _can_overlap_hc_residual(block: nn.Module, h: mx.array) -> bool:
    """Exact M=1024 TP-only gate for scheduling HC beside collectives."""

    if (
        not _DEEPSEEK_V4_HC_RESIDUAL_OVERLAP
        or getattr(block, "training", False)
        or is_dspark_verify_armed()
        or h.ndim != 4
        or h.shape[0] != 1
        or h.shape[1] != 1024
        or h.shape[-1] != 4096
        or h.dtype != mx.bfloat16
    ):
        return False
    attn_group = getattr(getattr(block, "attn", None), "sharding_group", None)
    ffn_group = getattr(getattr(block, "ffn", None), "sharding_group", None)
    return bool(
        attn_group is not None
        and ffn_group is not None
        and int(attn_group.size()) == 2
        and int(ffn_group.size()) == 2
    )


def _can_use_nax_oa_prefill(attn: nn.Module, prepared: mx.array) -> bool:
    """Return true only for the confirmed M5 rank-1 O-A prefill contract."""

    if not _DEEPSEEK_V4_NAX_OA_PREFILL or getattr(attn, "training", False):
        return False
    if tuple(prepared.shape) != (1, 8, 1024, 2560):
        return False
    if prepared.dtype != mx.bfloat16:
        return False

    projection = getattr(attn, "wo_a", None)
    if (
        projection is None
        or not callable(getattr(projection, "get", None))
        or getattr(projection, "group_size", None) != 32
        or getattr(projection, "bits", None) != 8
        or getattr(projection, "mode", None) != "mxfp8"
        or projection.get("biases") is not None
    ):
        return False
    weight = projection.get("weight")
    scales = projection.get("scales")
    if (
        weight is None
        or scales is None
        or tuple(weight.shape) != (8, 1024, 640)
        or tuple(scales.shape) != (8, 1024, 80)
        or weight.dtype != mx.uint32
        or scales.dtype != mx.uint8
    ):
        return False

    try:
        from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

        return bool(
            glm_fast.is_native_available()
            and glm_fast.has_symbol("ds4_projection_mxfp8_qmm")
            and glm_fast.ds4_projection_nax_kernels_built()
            and glm_fast.ds4_projection_nax_device_available()
        )
    except Exception:
        # Capability probing happens before graph construction. An old or
        # classic-only extension keeps the unchanged stock O-A path.
        return False


def _project_attention_oa(attn: nn.Module, prepared: mx.array) -> mx.array:
    """Project one prepared O-A tensor, selecting native only before enqueue."""

    if not _can_use_nax_oa_prefill(attn, prepared):
        return attn.wo_a(prepared)

    from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

    global _DEEPSEEK_V4_NAX_OA_PREFILL_LOGGED
    if not _DEEPSEEK_V4_NAX_OA_PREFILL_LOGGED:
        _DEEPSEEK_V4_NAX_OA_PREFILL_LOGGED = True
        logging.getLogger(__name__).info(
            "deepseek_v4: using exact M5 NAX O-A prefill tile "
            "(M=1024, heads=40, BM64/BN64/BK64; "
            "OMLX_DSV4_NAX_OA_PREFILL=0 disables)"
        )
    projection = attn.wo_a
    return glm_fast.ds4_projection_mxfp8_qmm(
        mx.contiguous(prepared),
        projection["weight"],
        projection["scales"],
        variant=0,
        use_nax=True,
        nax_variant=0,
    )


def _attention_output_chain_native_inputs(
    attn: nn.Module,
    prepared: mx.array,
) -> Optional[Tuple[mx.array, mx.array, mx.array, mx.array]]:
    """Preflight the exact DS4 3:5 M=1024 O-A→BF16→O-B chain."""

    prepared_shape = tuple(prepared.shape)
    legacy_shapes = {
        (1, 8, 1024, 1536),
        (1, 8, 1024, 2560),
        (1, 8, 1024, 4096),
        (1, 8, 2048, 4096),
    }
    equal_shapes = {
        (1, 8, 1024, 2048),
        (1, 8, 2048, 2048),
    }
    if (
        getattr(attn, "training", False)
        or is_dspark_verify_armed()
        or prepared.dtype != mx.bfloat16
        or prepared_shape not in legacy_shapes | equal_shapes
    ):
        return None
    config = getattr(attn, "config", None)
    try:
        if config is None or not _dsv4f_exact_config(config, 4):
            return None
    except (AttributeError, TypeError, ValueError):
        return None

    o_a = getattr(attn, "wo_a", None)
    o_b = getattr(attn, "wo_b", None)
    if not all(
        projection is not None
        and callable(getattr(projection, "get", None))
        and getattr(projection, "group_size", None) == 32
        and getattr(projection, "bits", None) == 8
        and getattr(projection, "mode", None) == "mxfp8"
        and projection.get("biases") is None
        for projection in (o_a, o_b)
    ):
        return None
    assert o_a is not None and o_b is not None
    o_a_weight = o_a.get("weight")
    o_a_scales = o_a.get("scales")
    o_b_weight = o_b.get("weight")
    o_b_scales = o_b.get("scales")
    k = int(prepared.shape[-1])
    if k == 2048:
        if not _DEEPSEEK_V4_OUTPUT_CHAIN_EQUAL_TP:
            return None
        try:
            # Equal TP2's K2048 classic chain is bit-exact and faster on M3
            # Ultra. M5 stock dispatches O-A/O-B through NAX TensorOps; the
            # classic chain changes that arithmetic frontier and is ~3x
            # slower, so that peer must retain its canonical stock graph.
            if str(mx.device_info().get("device_name") or "") != "Apple M3 Ultra":
                return None
        except Exception:
            return None
    elif not _DEEPSEEK_V4_OUTPUT_CHAIN_PREFILL:
        return None
    if (
        o_a_weight is None
        or o_a_scales is None
        or o_b_weight is None
        or o_b_scales is None
        or tuple(o_a_weight.shape) != (8, 1024, k // 4)
        or tuple(o_a_scales.shape) != (8, 1024, k // 32)
        or tuple(o_b_weight.shape) != (4096, 2048)
        or tuple(o_b_scales.shape) != (4096, 256)
        or o_a_weight.dtype != mx.uint32
        or o_a_scales.dtype != mx.uint8
        or o_b_weight.dtype != mx.uint32
        or o_b_scales.dtype != mx.uint8
    ):
        return None
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

        if not (
            glm_fast.is_native_available()
            and glm_fast.has_symbol("ds4_output_projection_chain")
        ):
            return None
    except Exception:
        return None
    return o_a_weight, o_a_scales, o_b_weight, o_b_scales


def _project_attention_output_chain(
    attn: nn.Module,
    prepared: mx.array,
) -> Optional[mx.array]:
    native_inputs = _attention_output_chain_native_inputs(attn, prepared)
    if native_inputs is None:
        return None
    from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

    global _DEEPSEEK_V4_OUTPUT_CHAIN_PREFILL_LOGGED
    if not _DEEPSEEK_V4_OUTPUT_CHAIN_PREFILL_LOGGED:
        _DEEPSEEK_V4_OUTPUT_CHAIN_PREFILL_LOGGED = True
        logging.getLogger(__name__).info(
            "deepseek_v4: using exact BF16 O-A/O-B prefill chain "
            "(M=%d, H=%d, BM64/BK32/BN64; "
            "OMLX_DSV4_OUTPUT_CHAIN_PREFILL=0 disables)",
            prepared.shape[2],
            prepared.shape[-1] // 64,
        )
    o_a_weight, o_a_scales, o_b_weight, o_b_scales = native_inputs
    return glm_fast.ds4_output_projection_chain(
        mx.contiguous(prepared),
        o_a_weight,
        o_a_scales,
        o_b_weight,
        o_b_scales,
        variant=1,
    )


def _stock_attention_qkv_finalizer(
    attn: nn.Module,
    q_raw: mx.array,
    kv_raw: mx.array,
    offset: Any,
) -> Tuple[mx.array, mx.array]:
    """Preserve the established four-operation BF16 finalizer graph."""

    q = mx.fast.rms_norm(q_raw, None, attn.config.rms_norm_eps)
    q = q.transpose(0, 2, 1, 3)
    q = attn.rope(q, offset)
    kv = attn.kv_norm(kv_raw).reshape(q_raw.shape[0], 1, q_raw.shape[1], attn.head_dim)
    kv = attn.rope(kv, offset)
    return q, kv


def _attention_finalizer_native_inputs(
    attn: nn.Module,
    q_raw: mx.array,
    kv_raw: mx.array,
    offset: Any,
) -> Optional[Tuple[mx.array, mx.array, float]]:
    """Preflight both native finalizers before either graph node is created."""

    verify_armed = is_dspark_verify_armed()
    tokens = int(q_raw.shape[1]) if q_raw.ndim == 4 else -1
    verify_route = bool(
        _DEEPSEEK_V4_ATTN_FINALIZER_VERIFY and verify_armed and tokens == 6
    )
    prefill_route = bool(
        _DEEPSEEK_V4_ATTN_FINALIZER_PREFILL
        and not verify_armed
        and tokens in (1024, 2048)
    )
    if (
        not (verify_route or prefill_route)
        or getattr(attn, "training", False)
        or type(offset) is not int
        or not 0 <= offset <= 0x7FFFFFFF
    ):
        return None
    supported_heads = (24, 32, 40, 64)
    if (
        q_raw.ndim != 4
        or q_raw.shape[0] != 1
        or q_raw.shape[1] not in (6, 1024, 2048)
        or q_raw.shape[2] not in supported_heads
        or q_raw.shape[3] != 512
        or q_raw.dtype != mx.bfloat16
        or kv_raw.shape != (1, q_raw.shape[1], 512)
        or kv_raw.dtype != mx.bfloat16
        or getattr(attn, "head_dim", None) != 512
    ):
        return None

    kv_norm = getattr(attn, "kv_norm", None)
    rope = getattr(attn, "rope", None)
    config = getattr(attn, "config", None)
    if (
        kv_norm is None
        or not callable(getattr(kv_norm, "get", None))
        or rope is None
        or not callable(getattr(rope, "_get_freqs", None))
        or config is None
    ):
        return None
    weight = kv_norm.get("weight")
    if weight is None or weight.shape != (512,) or weight.dtype != mx.bfloat16:
        return None

    try:
        eps = float(config.rms_norm_eps)
        if not math.isfinite(eps) or eps <= 0:
            return None
        freqs = rope._get_freqs(512, False)
        if freqs.shape != (256,) or freqs.dtype != mx.float32:
            return None
        from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

        if not (
            glm_fast.is_native_available()
            and glm_fast.has_symbol("ds4_q_head_rms_rope")
            and glm_fast.has_symbol("ds4_kv_rms_rope")
        ):
            return None
    except Exception:
        # All capability and frequency checks precede graph construction. Old
        # extensions and custom RoPE modules retain the stock four-op path.
        return None
    return weight, freqs, eps


def _finalize_attention_qkv(
    attn: nn.Module,
    q_raw: mx.array,
    kv_raw: mx.array,
    offset: Any,
) -> Tuple[mx.array, mx.array]:
    """Select the exact native pair or the unchanged stock graph, once."""

    native_inputs = _attention_finalizer_native_inputs(attn, q_raw, kv_raw, offset)
    if native_inputs is None:
        return _stock_attention_qkv_finalizer(attn, q_raw, kv_raw, offset)

    from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

    weight, freqs, eps = native_inputs
    global _DEEPSEEK_V4_ATTN_FINALIZER_PREFILL_LOGGED
    if not _DEEPSEEK_V4_ATTN_FINALIZER_PREFILL_LOGGED:
        _DEEPSEEK_V4_ATTN_FINALIZER_PREFILL_LOGGED = True
        logging.getLogger(__name__).info(
            "deepseek_v4: using exact BF16 Q/KV RMSNorm+RoPE finalizers (M=%d, H=%d)",
            q_raw.shape[1],
            q_raw.shape[2],
        )

    q_raw = mx.contiguous(q_raw)
    kv_raw = mx.contiguous(kv_raw)
    weight = mx.contiguous(weight)
    freqs = mx.contiguous(freqs)
    q = glm_fast.ds4_q_head_rms_rope(q_raw, freqs, offset, eps)
    kv = glm_fast.ds4_kv_rms_rope(
        kv_raw,
        weight,
        freqs,
        offset,
        eps,
    )
    return q, kv


def set_dspark_verify_armed(flag: bool) -> None:
    from omlx.patches.deepseek_v4.decode_consistency import set_armed

    set_armed(flag)


# Optional ANE prefill backends. They see every eligible call and return None
# to fall through to the normal GPU path (decode, verify, non-fixed shapes).
_ANE_LINEAR_BACKEND = None
_ANE_GROUPED_LINEAR_BACKEND = None
_ANE_MLP_BACKEND = None
_ANE_ATTENTION_INPUT_BACKEND = None


def register_ane_linear_backend(backend) -> None:
    """Install the optional ANE prefill backend for eligible plain linears."""
    global _ANE_LINEAR_BACKEND
    _ANE_LINEAR_BACKEND = backend


def register_ane_grouped_linear_backend(backend) -> None:
    """Install the optional ANE prefill backend for grouped linears."""
    global _ANE_GROUPED_LINEAR_BACKEND
    _ANE_GROUPED_LINEAR_BACKEND = backend


def register_ane_mlp_backend(backend) -> None:
    """Install the optional ANE prefill backend for the shared-expert MLP."""
    global _ANE_MLP_BACKEND
    _ANE_MLP_BACKEND = backend


def register_ane_attention_input_backend(backend) -> None:
    """Install the optional stacked backend for projections that consume x."""
    global _ANE_ATTENTION_INPUT_BACKEND
    _ANE_ATTENTION_INPUT_BACKEND = backend


def _ane_attention_input(attn: nn.Module, x: mx.array) -> dict[str, mx.array]:
    backend = _ANE_ATTENTION_INPUT_BACKEND
    if backend is not None:
        projected = backend(attn, x)
        if projected is not None:
            return projected
    return {}


def _projection_or(
    projected: dict[str, mx.array], name: str, linear: nn.Module, x: mx.array
) -> mx.array:
    value = projected.get(name)
    return linear(x) if value is None else value


def _ane_linear(linear: nn.Module, x: mx.array) -> mx.array:
    backend = _ANE_LINEAR_BACKEND
    if backend is not None:
        out = backend(linear, x)
        if out is not None:
            return out
    return linear(x)


def _ane_grouped_linear(linear: nn.Module, x: mx.array) -> mx.array:
    backend = _ANE_GROUPED_LINEAR_BACKEND
    if backend is not None:
        out = backend(linear, x)
        if out is not None:
            return out
    return linear(x)


_ANE_STACKED_Q_BACKEND = None


def register_ane_stacked_q_backend(backend) -> None:
    """Install the optional ANE backend for the stacked attn+indexer wq_b."""
    global _ANE_STACKED_Q_BACKEND
    _ANE_STACKED_Q_BACKEND = backend


def _ane_stacked_q(
    attn_linear: nn.Module, indexer_linear: nn.Module | None, value: mx.array
):
    """Project the attention q, folding the indexer q into the same hybrid op.

    Returns ``(attn_q_raw, indexer_q_raw)``; the second element is ``None``
    whenever the stacked backend did not run, and the caller must then leave
    the indexer to do its own projection.
    """
    backend = _ANE_STACKED_Q_BACKEND
    if backend is not None and indexer_linear is not None:
        split = backend(attn_linear, value)
        if split is not None:
            return split
    return _ane_linear(attn_linear, value), None


def _prepare_attention_output(attn: nn.Module, row: mx.array) -> mx.array:
    """Map (B,H,M,D) attention rows to grouped O-A input (B,G,M,K)."""

    batch, _, length, _ = row.shape
    row = row.reshape(batch, attn.o_groups, -1, length, attn.head_dim)
    return row.transpose(0, 1, 3, 2, 4).flatten(-2)


def _project_attention_output(attn: nn.Module, out: mx.array, offset: Any) -> mx.array:
    out = attn.rope(out, offset, inverse=True)

    def finish(row: mx.array) -> mx.array:
        return row.transpose(0, 2, 1, 3).flatten(-2)

    if is_dspark_verify_armed():
        if _DEEPSEEK_V4_VERIFY_BATCHED_OA_PREPARE:
            # This view is element-for-element identical to concatenating M
            # independently prepared one-row views, but avoids six slice
            # graphs plus a materializing concatenate in depth-5 verify.
            prepared = _prepare_attention_output(attn, out)
            global _DEEPSEEK_V4_VERIFY_BATCHED_OA_PREPARE_LOGGED
            if not _DEEPSEEK_V4_VERIFY_BATCHED_OA_PREPARE_LOGGED:
                _DEEPSEEK_V4_VERIFY_BATCHED_OA_PREPARE_LOGGED = True
                logging.getLogger(__name__).info(
                    "deepseek_v4: using exact batched O-A verify preparation "
                    "(M=%d; OMLX_DSV4_VERIFY_BATCHED_OA_PREPARE=0 disables)",
                    out.shape[2],
                )
        else:
            prepared = mx.concatenate(
                [
                    _prepare_attention_output(attn, out[:, :, idx : idx + 1])
                    for idx in range(out.shape[2])
                ],
                axis=2,
            )
        from omlx.patches.deepseek_v4.verify_qmv import (
            eligible as qmv_eligible,
        )
        from omlx.patches.deepseek_v4.verify_qmv import (
            exact_verify_multi_qmv,
            exact_verify_qmv,
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
        return (
            exact_verify_qmv(attn.wo_b, projected)
            if qmv_eligible(attn.wo_b, projected)
            else attn.wo_b(projected)
        )
    prepared = _prepare_attention_output(attn, out)
    native_chain = _project_attention_output_chain(attn, prepared)
    if native_chain is not None:
        return native_chain
    if _ANE_GROUPED_LINEAR_BACKEND is not None:
        projected = finish(_ane_grouped_linear(attn.wo_a, prepared))
        return _ane_linear(attn.wo_b, projected)
    return attn.wo_b(finish(_project_attention_oa(attn, prepared)))


def _project_verify_q_b(module: nn.Module, inputs: mx.array) -> mx.array:
    """Use the one-token reduction order for DSpark's six target rows."""

    if is_dspark_verify_armed():
        from omlx.patches.deepseek_v4.verify_qmv import (
            eligible as qmv_eligible,
        )
        from omlx.patches.deepseek_v4.verify_qmv import (
            exact_verify_qmv,
        )

        if qmv_eligible(module, inputs):
            return exact_verify_qmv(module, inputs)
    return _ane_linear(module, inputs)


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


@contextmanager
def _defer_cache_materialization():
    """Capture one forward's cache arrays instead of evaluating them.

    The yielded list belongs to the caller and remains valid after the context
    exits.  Nested use restores the prior capture target, and exceptions cannot
    leave later decode calls in deferred mode.
    """

    previous = getattr(_CACHE_MATERIALIZATION_CAPTURE, "arrays", None)
    captured = []
    _CACHE_MATERIALIZATION_CAPTURE.arrays = captured
    try:
        yield captured
    finally:
        if previous is None:
            try:
                del _CACHE_MATERIALIZATION_CAPTURE.arrays
            except AttributeError:
                pass
        else:
            _CACHE_MATERIALIZATION_CAPTURE.arrays = previous


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

    if not cache_arrays:
        return

    captured = getattr(_CACHE_MATERIALIZATION_CAPTURE, "arrays", None)
    if captured is not None:
        captured.extend(cache_arrays)
        return
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
        k: mxfp8 for k, _ in flat_modules if ".attn.w" in k or ".attn.indexer.wq" in k
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


_DEEPSEEK_V4_ROUTER_TOPK_DECODE = os.getenv(
    "OMLX_DSV4_ROUTER_TOPK_DECODE", "1"
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_ROUTER_TOPK_DECODE_LOGGED = False


@mx.compile
def _router_native_pre(logits: mx.array, bias: mx.array):
    scores = mx.sqrt(nn.softplus(logits.astype(mx.float32)))
    return scores, mx.contiguous(scores + bias)


@mx.compile
def _router_native_post(
    scores: mx.array,
    indices: mx.array,
    routed_scaling_factor: float,
):
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return weights * routed_scaling_factor


def _native_router_select(
    logits: mx.array,
    bias: mx.array,
    routed_scaling_factor: float,
) -> Optional[Tuple[mx.array, mx.array]]:
    if (
        not _DEEPSEEK_V4_ROUTER_TOPK_DECODE
        or tuple(logits.shape) != (1, 1, 256)
        or logits.dtype not in (mx.bfloat16, mx.float16)
        or bias.shape != (256,)
        or bias.dtype != mx.float32
    ):
        return None
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

        if not glm_fast.has_symbol("ds4_router_topk_indices"):
            return None
        scores, biased = _router_native_pre(logits, bias)
        indices = glm_fast.ds4_router_topk_indices(biased.reshape(1, 256)).reshape(
            1, 1, 6
        )
        weights = _router_native_post(
            scores,
            indices,
            routed_scaling_factor,
        )
    except (TypeError, ValueError):
        return None
    global _DEEPSEEK_V4_ROUTER_TOPK_DECODE_LOGGED
    if not _DEEPSEEK_V4_ROUTER_TOPK_DECODE_LOGGED:
        _DEEPSEEK_V4_ROUTER_TOPK_DECODE_LOGGED = True
        logging.getLogger(__name__).info(
            "DeepSeek V4 exact B1 native router top-6 active"
        )
    return indices, weights


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


def _extend_mask(
    mask: Optional[mx.array],
    pool_mask: Optional[mx.array],
    N: int,
    *,
    local_width: Optional[int] = None,
    pooled_width: Optional[int] = None,
):
    if mask is None:
        return None

    if mask.ndim == 2:
        mask = mask[None, None]
    B, H, L, S = mask.shape
    local_width = S if local_width is None else int(local_width)
    pooled_width = N - local_width if pooled_width is None else int(pooled_width)
    if local_width < 0 or pooled_width < 0 or local_width + pooled_width != N:
        raise ValueError("logical local/pooled widths do not match the KV sequence")
    if S > local_width:
        # RotatingKVCache retains the newest local window. A causal mask built
        # from the absolute prompt can therefore be wider than the physical
        # local KV view after restoring a prefix; keep the matching suffix.
        mask = mask[..., S - local_width :]
        S = local_width
    elif S < local_width:
        raise ValueError("local attention mask is shorter than local KV")

    if pool_mask is not None:
        mask_width = int(pool_mask.shape[-1])
        if mask_width > pooled_width:
            # BatchPoolingCache may retain a capacity/physical tail beyond the
            # restored row's logical pooled extent. make_mask marks that tail
            # invalid, but broadcasting it to the shorter logical KV view
            # fails before the values can be ignored. Logical pooled rows are
            # prefix-contiguous, so trim only the physical tail.
            pool_mask = pool_mask[..., :pooled_width]
        elif mask_width < pooled_width:
            raise ValueError("pooled attention validity mask is shorter than pooled KV")

    if pool_mask is None:
        pool_mask = mx.ones((B, H, L, pooled_width), dtype=mx.bool_)
    elif pool_mask.ndim == 2:
        pool_mask = mx.broadcast_to(pool_mask, (B, H, L, pooled_width))
    elif pool_mask.ndim == 3:
        pool_mask = mx.broadcast_to(pool_mask[:, None], (B, H, L, pooled_width))

    full_mask = mx.concatenate([mask, pool_mask], axis=-1)

    return full_mask


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
# Shape/argument rejections from the native kernels surface as ValueError
# (std::invalid_argument) and are raised before any GPU work. They are
# per-call conditions -- e.g. a stale extension binary rejecting unaligned
# prefill tails -- so they must NOT latch the process-wide native-disable
# flags: warn once and fall back for that call only, keeping the native
# path armed for later calls with different shapes. Genuine runtime
# failures (RuntimeError etc.) still latch exactly as before.
_DEEPSEEK_V4_INDEXER_SHAPE_WARNED = False
_DEEPSEEK_V4_SPARSE_ATTN_SHAPE_WARNED = False
_DEEPSEEK_V4_DSPARK_TOPK_SHAPE_WARNED = False
# The original v25 gate was qualified on M2 Ultra first.  Keep its environment
# name as a compatibility alias, while the preferred name reflects the now
# qualified M2 Ultra / M3 Ultra / M5 Max family.
_DEEPSEEK_V4_MMA_SCORE = os.getenv(
    "OMLX_DSV4F_MMA_SCORE", os.getenv("OMLX_DSV4F_M2_MMA_SCORE", "1")
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_MMA_SCORE_LOGGED = False
_DEEPSEEK_V4_NAX_INDEXER_SCORE = os.getenv(
    # Experimental/default-off: the first M5 strict gate retained exact top-k
    # membership but found a one-BF16-ULP score drift in 1/8,208 outputs. Keep
    # Steel until the score sheet itself is bit-exact across the full matrix.
    "OMLX_DSV4F_NAX_INDEXER_SCORE",
    "0",
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_NAX_INDEXER_SCORE_LOGGED = False
_DEEPSEEK_V4_INDEXER_ROW_TP = os.getenv(
    "OMLX_DSV4_INDEXER_ROW_TP", "1"
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_WEIGHTED_INDEXER_ROWS = os.getenv(
    # Default-off: the first live 3:5/30K gate lost 3.1% because padding the
    # all-gather to the larger 640-row shard outweighed the score imbalance.
    "OMLX_DSV4_WEIGHTED_INDEXER_ROWS",
    "0",
).strip().lower() in ("1", "true", "on", "yes")
try:
    _DEEPSEEK_V4_INDEXER_ROW_WEIGHTS = tuple(
        int(value)
        for value in os.getenv("OMLX_DSV4_INDEXER_ROW_WEIGHTS", "").split(",")
        if value.strip()
    )
except ValueError:
    _DEEPSEEK_V4_INDEXER_ROW_WEIGHTS = ()
try:
    _DEEPSEEK_V4_INDEXER_ROW_WEIGHTS_MIN_POOL = max(
        0,
        int(os.getenv("OMLX_DSV4_INDEXER_ROW_WEIGHTS_MIN_POOL", "16000")),
    )
except ValueError:
    _DEEPSEEK_V4_INDEXER_ROW_WEIGHTS_MIN_POOL = 16000
_DEEPSEEK_V4_INDEXER_ROW_WEIGHTS_LOGGED = False
# JACCL's two-rank all-gather lost a completion during the 100K lifetime
# gate.  DS4 row parallelism does not need a general collective here: each
# rank already knows the peer's exact row shape and both need only exchange
# their compact uint32 top-k rows. The physical gate rejected P2P for equal
# rows but found it useful for unpadded 3:5 rows, so the path below requires
# both explicit P2P and weighted-row opt-ins in addition to pure TP2.
_DEEPSEEK_V4_INDEXER_GATHER_P2P = os.getenv(
    # Default-off until a physical two-Mac lifetime gate proves that the
    # stricter ordering does not cost more than 2% at 100K-equivalent rows.
    "OMLX_DSV4_INDEXER_GATHER_P2P",
    "0",
).strip().lower() in ("1", "true", "on", "yes")
_DEEPSEEK_V4_INDEXER_DECISION_TRANSPORT = (
    os.getenv("OMLX_DSV4_INDEXER_DECISION_TRANSPORT", "jaccl").strip().lower()
)
_DEEPSEEK_V4_INDEXER_CONTROL_LOGGED = False
try:
    _DEEPSEEK_V4_INDEXER_ROW_TP_MIN_POOL = max(
        0, int(os.getenv("OMLX_DSV4_INDEXER_ROW_TP_MIN_POOL", "2048"))
    )
except ValueError:
    _DEEPSEEK_V4_INDEXER_ROW_TP_MIN_POOL = 2048


def _indexer_decode_owner_rank(group: Any) -> Optional[int]:
    """Process-local owner selected from the signed performance profiles."""

    if group is None or int(group.size()) < 2:
        return None
    raw = os.getenv("OMLX_DSV4_INDEXER_DECODE_OWNER_RANK", "").strip().lower()
    if not raw or raw in {"auto", "false", "no", "off", "disabled"}:
        return None
    try:
        owner = int(raw)
    except ValueError:
        return None
    return owner if 0 <= owner < int(group.size()) else None


def _broadcast_indexer_indices(
    indices: Optional[mx.array],
    *,
    shape: Tuple[int, ...],
    group: Any,
    owner: int,
) -> mx.array:
    """Broadcast one exact owner-computed top-k decision.

    The shipped path remains JACCL until live A/B promotes the separate control
    channel.  ``control`` is deliberately explicit: once any rank starts a TCP
    packet, falling back locally would split the distributed operation order.
    Prefill row parallelism and DSpark verification never enter this helper's
    owner-decode call sites, so their collective schedules remain unchanged.
    """

    # Startup-fixed on purpose. Changing transport between layers would split
    # the global control/JACCL operation order across ranks.
    transport = _DEEPSEEK_V4_INDEXER_DECISION_TRANSPORT
    if transport in {"control", "tcp"}:
        global _DEEPSEEK_V4_INDEXER_CONTROL_LOGGED
        from omlx.cluster.control_plane import active_rank_control_plane

        control = active_rank_control_plane()
        if control is None:
            raise RuntimeError(
                "DS4 indexer control transport requires an active rank-control plane"
            )
        rank = int(group.rank())
        size = int(group.size())
        if int(control.rank) != rank or int(control.world_size) != size:
            raise RuntimeError(
                "DS4 indexer control transport does not match its tensor group"
            )
        expected_shape = tuple(int(value) for value in shape)
        count = math.prod(expected_shape)
        if count < 1:
            raise RuntimeError("DS4 indexer decision shape is empty")
        expected_size = count * np.dtype(">u4").itemsize
        if not _DEEPSEEK_V4_INDEXER_CONTROL_LOGGED:
            _DEEPSEEK_V4_INDEXER_CONTROL_LOGGED = True
            logging.getLogger(__name__).info(
                "deepseek_v4: sparse decode index decisions use the rank-control "
                "plane (owner=%d, payload=%d bytes)",
                owner,
                expected_size,
            )
        if rank == owner:
            if indices is None or tuple(indices.shape) != expected_shape:
                raise RuntimeError("DS4 indexer owner produced an invalid decision")
            local = indices.astype(mx.uint32).reshape(-1)
            # The control plane is host TCP, so this is the intentional graph
            # boundary: materialize the owner's deterministic top-k before the
            # peer can consume it.  Only 512 uint32 values cross for DS4 decode.
            payload = np.asarray(local, dtype=">u4").tobytes()
        else:
            local = None
            payload = None
        received = control.broadcast_owned_bytes(
            payload,
            source_rank=owner,
            expected_size=expected_size,
        )
        if rank == owner:
            return local.reshape(expected_shape)
        values = np.frombuffer(received, dtype=">u4").astype(np.uint32)
        return mx.array(values, dtype=mx.uint32).reshape(expected_shape)
    if transport not in {"jaccl", "collective", "all_sum"}:
        raise RuntimeError(
            "OMLX_DSV4_INDEXER_DECISION_TRANSPORT must be jaccl or control"
        )

    local = (
        indices.astype(mx.int32)
        if int(group.rank()) == owner and indices is not None
        else mx.zeros(shape, dtype=mx.int32)
    )
    return mx.distributed.all_sum(local, group=group).astype(mx.uint32)


def _balanced_row_ranges(length: int, size: int) -> Tuple[Tuple[int, int], ...]:
    """Contiguous query-row ranges with at most one row of skew."""

    base, remainder = divmod(length, size)
    ranges = []
    start = 0
    for rank in range(size):
        stop = start + base + (1 if rank < remainder else 0)
        ranges.append((start, stop))
        start = stop
    return tuple(ranges)


def _weighted_row_ranges(
    length: int,
    weights: Tuple[int, ...],
) -> Tuple[Tuple[int, int], ...]:
    """Contiguous row ranges proportional to a qualified TP partition.

    Cumulative integer boundaries keep every row exactly once and preserve
    sequence order without a floating-point or host-specific tie break.  The
    signed 3:5 DS4 placement therefore gives 384/640 rows of an M=1024 indexer
    score build to the M3/M5 ranks instead of repeating the old 512/512 split.
    """

    if length < 0 or not weights or any(weight < 1 for weight in weights):
        raise ValueError("row length and TP weights must be valid")
    total = sum(weights)
    ranges = []
    prefix = 0
    for weight in weights:
        start = length * prefix // total
        prefix += weight
        stop = length * prefix // total
        ranges.append((start, stop))
    return tuple(ranges)


def _indexer_row_ranges(
    length: int,
    group: mx.distributed.Group,
    pooled_tokens: Optional[int] = None,
) -> Tuple[Tuple[int, int], ...]:
    """Use explicit long-context compute weights or preserve equal splitting."""

    size = int(group.size())
    explicit = _DEEPSEEK_V4_INDEXER_ROW_WEIGHTS
    explicit_active = (
        len(explicit) == size
        and all(weight > 0 for weight in explicit)
        and pooled_tokens is not None
        and pooled_tokens >= _DEEPSEEK_V4_INDEXER_ROW_WEIGHTS_MIN_POOL
    )
    weights = explicit if explicit_active else None
    if weights is None and _DEEPSEEK_V4_WEIGHTED_INDEXER_ROWS:
        weights = _tp_partition_weights(group)
    if weights is None:
        return _balanced_row_ranges(length, size)
    ranges = _weighted_row_ranges(length, weights)
    if explicit_active:
        global _DEEPSEEK_V4_INDEXER_ROW_WEIGHTS_LOGGED
        if not _DEEPSEEK_V4_INDEXER_ROW_WEIGHTS_LOGGED:
            _DEEPSEEK_V4_INDEXER_ROW_WEIGHTS_LOGGED = True
            logging.getLogger(__name__).info(
                "deepseek_v4: long-context indexer rows use explicit "
                "weights=%s ranges=%s pool=%d (min_pool=%d)",
                weights,
                ranges,
                pooled_tokens,
                _DEEPSEEK_V4_INDEXER_ROW_WEIGHTS_MIN_POOL,
            )
    return ranges


def _gather_indexer_rows(
    local_indices: mx.array,
    total_rows: int,
    group: Any,
    pooled_tokens: Optional[int] = None,
) -> mx.array:
    """Reassemble uneven row shards after exact per-row top-k selection."""

    size = int(group.size())
    ranges = _indexer_row_ranges(total_rows, group, pooled_tokens)
    rank = int(group.rank()) if hasattr(group, "rank") else -1
    row_counts = tuple(stop - start for start, stop in ranges)
    # For unequal TP2 rows, rank zero's reconstruction depends on send before
    # recv while rank one's depends on recv before send. This rank-asymmetric
    # graph order gives JACCL one producer and one consumer per direction with
    # one final transport evaluation instead of two Python/Metal barriers.
    # The payload stays on-device throughout; recv_like's zero is shape
    # metadata and no control-plane/CPU copy participates.
    #
    # Do not extend this ordering to a ring without a separate qualification:
    # the two-phase proof relies on exactly one peer and non-empty row shards.
    if (
        _DEEPSEEK_V4_INDEXER_GATHER_P2P
        and _DEEPSEEK_V4_WEIGHTED_INDEXER_ROWS
        and size == 2
        and rank in (0, 1)
        and all(rows > 0 for rows in row_counts)
        and row_counts[0] != row_counts[1]
    ):
        peer = 1 - rank
        peer_rows = row_counts[peer]
        rows_first = local_indices.swapaxes(0, 1)
        peer_template = mx.zeros(
            (peer_rows, *rows_first.shape[1:]),
            dtype=rows_first.dtype,
        )
        # Both ranks finish their independent score/top-k work before either
        # begins transport. Without this common boundary rank 1's first recv
        # would defer its own score graph until after rank 0 sent, serializing
        # the heterogeneous GPUs and surrendering row parallelism's benefit.
        mx.eval(rows_first)
        sent = mx.distributed.send(rows_first, peer, group=group)
        peer_rows_first = mx.distributed.recv_like(
            peer_template,
            peer,
            group=group,
        )
        parts = (sent, peer_rows_first) if rank == 0 else (peer_rows_first, sent)
        return mx.concatenate(parts, axis=0).swapaxes(0, 1)

    max_rows = max(stop - start for start, stop in ranges)
    rows_first = local_indices.swapaxes(0, 1)
    if rows_first.shape[0] < max_rows:
        rows_first = mx.concatenate(
            [
                rows_first,
                mx.zeros(
                    (max_rows - rows_first.shape[0], *rows_first.shape[1:]),
                    dtype=rows_first.dtype,
                ),
            ],
            axis=0,
        )
    gathered = mx.distributed.all_gather(rows_first, group=group)
    # The normal prompt chunk divides evenly across TP ranks (512 rows over
    # two Macs). ``all_gather`` already concatenates those equal row blocks in
    # rank order, which is the final sequence order. Slicing every block and
    # concatenating them again only schedules another full top-k-index copy
    # (1 MiB per ratio-4 layer at L=K=512).
    if total_rows == max_rows * size and rows_first.shape[0] == max_rows:
        return gathered.swapaxes(0, 1)
    parts = [
        gathered[rank * max_rows : rank * max_rows + (stop - start)]
        for rank, (start, stop) in enumerate(ranges)
    ]
    return mx.concatenate(parts, axis=0).swapaxes(0, 1)


def _dsv4f_exact_config(config, compress_ratio: int) -> bool:
    """Exact DeepSeek-V4-Flash ratio-4 fingerprint shared by tuned kernels."""
    if compress_ratio != 4:
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


def _dsv4f_mma_exact_pairing(config, compress_ratio: int) -> bool:
    """True only for a benchmark-qualified DS4-Flash / Apple GPU pairing.

    The v25 MMA indexer score kernel is benchmark-proven and bit-exact on M2
    Ultra, M3 Ultra, and M5 Max for this precise checkpoint fingerprint.
    Unknown Apple generations and every other model keep the Steel kernel.
    """
    try:
        if mx.device_info().get("device_name") not in {
            "Apple M2 Ultra",
            "Apple M3 Ultra",
            "Apple M5 Max",
        }:
            return False
    except Exception:
        return False
    return _dsv4f_exact_config(config, compress_ratio)


def _dsv4f_m2_exact_pairing(config, compress_ratio: int) -> bool:
    """Compatibility alias for callers written before M3/M5 qualification."""
    return _dsv4f_mma_exact_pairing(config, compress_ratio)


def _dsv4f_mma_score_enabled(config, compress_ratio: int) -> bool:
    """Gate for the exact v25 from-scratch MMA score kernel.

    Exact-fingerprint pairing, env rollback (OMLX_DSV4F_MMA_SCORE=0, with
    OMLX_DSV4F_M2_MMA_SCORE retained as an alias), and an extension build
    exposing the symbol. The kernel serves bf16 /
    H=64 / D=128 / weights [B, L, H] / non-causal only — all guaranteed by
    the fingerprint and this call site; N >= 64 is re-checked per call.
    """
    if not _DEEPSEEK_V4_MMA_SCORE:
        return False
    return _dsv4f_mma_exact_pairing(config, compress_ratio)


def _dsv4f_m2_mma_score_enabled(config, compress_ratio: int) -> bool:
    """Compatibility alias for the original M2-named startup gate."""
    return _dsv4f_mma_score_enabled(config, compress_ratio)


def _dsv4f_nax_indexer_score_enabled(config, compress_ratio: int) -> bool:
    """Startup gate for the lossless M5 TensorOps indexer-score port.

    Hardware detection mirrors ``metal::is_nax_available`` and the exact
    checkpoint fingerprint prevents the H64/D128/ratio-4 tuning from leaking
    into GLM or future DeepSeek variants. The candidate remains default-off;
    ``OMLX_DSV4F_NAX_INDEXER_SCORE=1`` arms it for isolated A/B only until the
    strict score-bit gate passes. Per-call dtype/shape and artifact gates remain
    below.
    """
    return bool(
        _DEEPSEEK_V4_NAX_INDEXER_SCORE
        and _dsv4f_exact_config(config, compress_ratio)
        and is_nax_available()
    )


@partial(mx.compile, shapeless=True)
def _indexer_head_reduce(scores, weights, scale):
    """relu -> scale -> head-weight -> head-sum, fused over `scores`.

    `scores` is (B, H, L, P_tile); the reduction is over the head axis (1),
    entirely within each pooled tile, so tiling P never crosses the reduce.
    """
    return (mx.maximum(scores, 0) * scale * weights).sum(axis=1)


def _foldable_pool_mask_ratio(
    pool_cache: Any,
    pmask: Optional[mx.array],
    *,
    batch_size: int,
    pooled_tokens: int,
    query_offset: Any,
) -> int:
    """Return an exact uniform pooled-mask ratio, or zero for mx.where.

    A singleton ``BatchPoolingCache`` has the same causal mask as
    ``PoolingCache`` only when its one logical pool length fills the physical
    view. Multi-row batches and restored/trimmed physical tails retain their
    explicit 3-D validity mask.
    """

    if pmask is None or not isinstance(query_offset, int):
        return 0
    name = type(pool_cache).__name__
    if name == "PoolingCache":
        if pmask.ndim != 2:
            return 0
    elif name == "BatchPoolingCache":
        lengths = getattr(pool_cache, "_pool_lengths", None)
        if (
            batch_size != 1
            or pmask.ndim != 3
            or not isinstance(lengths, list)
            or lengths != [pooled_tokens]
        ):
            return 0
    else:
        return 0
    try:
        ratio = int(pool_cache.ratio)
    except (AttributeError, TypeError, ValueError):
        return 0
    return ratio if ratio > 0 else 0


@partial(mx.compile, shapeless=True)
def _split_softmax(log_normalizer, logits_a, logits_b, sinks=None):
    if sinks is not None:
        log_normalizer = mx.logaddexp(log_normalizer, sinks)
    weights_a = mx.exp(logits_a - log_normalizer)
    weights_b = mx.exp(logits_b - log_normalizer)
    return weights_a, weights_b


def _fused_sparse_decode_attention(
    q_scaled: mx.array,
    local_kv: mx.array,
    pooled_sq: mx.array,
    sinks: mx.array,
) -> Optional[mx.array]:
    """Single-dispatch fused sparse decode attention; None when unavailable.

    decode_fast.sparse_attn_decode computes the same composition as
    _dspark_sparse_exact_attention in one Metal kernel with an independent
    per-(batch, head) reduction order, so decode (B=1) and verify (B=block)
    stay bitwise row-consistent. Any failure latches the composed path for
    the rest of the process.
    """
    global _DEEPSEEK_V4_FUSED_SPARSE_DECODE_FAILED
    if _DEEPSEEK_V4_FUSED_SPARSE_DECODE_FAILED:
        return None
    try:
        from omlx.custom_kernels.decode_fast import fast as decode_fast

        return decode_fast.sparse_attn_decode(q_scaled, local_kv, pooled_sq, sinks)
    except Exception as exc:
        _DEEPSEEK_V4_FUSED_SPARSE_DECODE_FAILED = True
        logging.getLogger(__name__).warning(
            "DSV4 fused sparse decode attention failed; composed path for "
            "the rest of this process: %s",
            exc,
        )
        return None


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
        and H in (24, 32, 40, 64)
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
        # The separate native sparse kernel is still specialized to the full
        # 64-head model. TP=2/H in {24, 32, 40} either returned through WSDPA
        # above or keeps the exact composed fallback below.
        if H == 64 and not _DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED:
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
            except ValueError as exc:
                # Shape/argument rejection (std::invalid_argument): per-call
                # condition, raised before GPU work. Do NOT latch -- keep
                # the native path armed for later calls.
                global _DEEPSEEK_V4_SPARSE_ATTN_SHAPE_WARNED
                if not _DEEPSEEK_V4_SPARSE_ATTN_SHAPE_WARNED:
                    _DEEPSEEK_V4_SPARSE_ATTN_SHAPE_WARNED = True
                    logging.getLogger(__name__).warning(
                        "DSV4 native sparse attention rejected this shape; "
                        "MLX fallback for this call only (native path "
                        "stays armed for later calls): %s",
                        exc,
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
        if not _DEEPSEEK_V4_FUSED_SPARSE_DECODE_DISABLED:
            fused = _fused_sparse_decode_attention(q_scaled, local_kv, pooled_sq, sinks)
            if fused is not None:
                return fused
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
        self.weight = mx.zeros((self.num_experts, self.hidden_dim))
        if self.hash:
            self.tid2eid = mx.zeros((config.vocab_size, self.top_k), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros(
                (self.num_experts,), dtype=mx.float32
            )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        logits = decode_matmul(x, self.weight.T)

        if self.hash:
            if input_ids is None:
                raise ValueError("DeepSeek-V4 hash routing requires input_ids.")
            inds, weights = _hash_expert_select(
                input_ids,
                logits,
                self.tid2eid,
                self.routed_scaling_factor,
                self.norm_topk_prob,
                self.scoring_func,
            )
        else:
            native = (
                _native_router_select(
                    logits,
                    self.e_score_correction_bias,
                    self.routed_scaling_factor,
                )
                if self.top_k == 6
                and self.norm_topk_prob
                and self.scoring_func == "sqrtsoftplus"
                else None
            )
            if native is None:
                inds, weights = _expert_select(
                    logits,
                    self.e_score_correction_bias,
                    self.top_k,
                    self.routed_scaling_factor,
                    self.norm_topk_prob,
                    self.scoring_func,
                )
            else:
                inds, weights = native

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
        backend = _ANE_MLP_BACKEND
        if backend is not None and not is_dspark_verify_armed():
            hybrid = backend(self, x)
            if hybrid is not None:
                return hybrid
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
        try:
            self.switch_mlp._omlx_dsv4f_exact_config = _dsv4f_exact_config(config, 4)
        except (AttributeError, TypeError, ValueError):
            # Future/partial configs fail closed before the tuned path can
            # create a graph.
            self.switch_mlp._omlx_dsv4f_exact_config = False
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


def _fp32_topk_indices(scores: mx.array, k: int) -> mx.array:
    """Deterministic native FP32 top-k with a per-call safe fallback."""

    global _DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED
    global _DEEPSEEK_V4_DSPARK_TOPK_SHAPE_WARNED

    if not _DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED:
        try:
            from omlx.custom_kernels.glm_moe_dsa import fast

            if fast.has_symbol("dspark_fp32_topk_indices"):
                flat = scores.reshape(-1, scores.shape[-1])
                indices = fast.dspark_fp32_topk_indices(flat, k)
                return indices.reshape(*scores.shape[:-1], k)
        except ValueError as exc:
            # Shape rejection happens before GPU work and may be unique to one
            # tail. Keep the native path armed for later compatible calls.
            if not _DEEPSEEK_V4_DSPARK_TOPK_SHAPE_WARNED:
                _DEEPSEEK_V4_DSPARK_TOPK_SHAPE_WARNED = True
                logging.getLogger(__name__).warning(
                    "DSV4 native FP32 top-k rejected this shape; stable "
                    "fallback for this call only (native path stays armed): %s",
                    exc,
                )
        except Exception as exc:
            _DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED = True
            logging.getLogger(__name__).warning(
                "DSV4 native FP32 top-k failed; stable fallback for the rest "
                "of this process: %s",
                exc,
                exc_info=True,
            )
    return _stable_topk_indices(scores, k)


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
        self._m2_mma_score = _dsv4f_mma_score_enabled(config, compress_ratio)
        self._nax_indexer_score = _dsv4f_nax_indexer_score_enabled(
            config, compress_ratio
        )
        self.row_sharding_group = None

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
        total_rows = L
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

        # Sparse score/top-k selection is independent for every prompt row,
        # but TP used to repeat all of it on every rank. Keep pooled-cache
        # updates replicated, split only the expensive query rows, and gather
        # the compact uint32 top-k result before sharded attention consumes it.
        # Decode and DSpark's pre-projected verify rows retain the original path.
        row_group = self.row_sharding_group
        k = min(self.index_topk, pooled.shape[1])
        decode_owner = _indexer_decode_owner_rank(row_group)
        owner_decode = bool(
            decode_owner is not None
            and B == 1
            and L == 1
            and pooled.shape[1] > self.index_topk
            and projected_q is None
            and projected_weights is None
            and not is_dspark_verify_armed()
        )
        if owner_decode and int(row_group.rank()) != decode_owner:
            return _broadcast_indexer_indices(
                None,
                shape=(B, L, k),
                group=row_group,
                owner=decode_owner,
            )
        row_sharded = bool(
            _DEEPSEEK_V4_INDEXER_ROW_TP
            and row_group is not None
            and int(row_group.size()) > 1
            and B == 1
            and L >= int(row_group.size())
            and pooled.shape[1] >= _DEEPSEEK_V4_INDEXER_ROW_TP_MIN_POOL
            and projected_q is None
            and projected_weights is None
        )
        query_offset = offset
        if row_sharded:
            ranges = _indexer_row_ranges(L, row_group, int(pooled.shape[1]))
            start, stop = ranges[int(row_group.rank())]
            x = x[:, start:stop]
            q_residual = q_residual[:, start:stop]
            L = stop - start
            query_offset = offset + start

        if projected_q is None:
            q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
            q = q.transpose(0, 2, 1, 3)
            q = position_rope(q, query_offset)
        else:
            q = projected_q

        pmask = (
            pool_cache.make_mask(total_rows, offset) if pool_cache is not None else None
        )
        if row_sharded and pmask is not None:
            pmask = pmask[..., start:stop, :]
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
                    # for a uniform PoolingCache ratio mask, bit-identical to
                    # the mx.where pass it replaces. Singleton
                    # BatchPoolingCache is also uniform when its logical pool
                    # fills the physical view; true multi-row/tailed batches
                    # keep the explicit 3-D mask.
                    _mask_ratio = _foldable_pool_mask_ratio(
                        pool_cache,
                        pmask,
                        batch_size=B,
                        pooled_tokens=int(pooled.shape[1]),
                        query_offset=query_offset,
                    )
                    _mask_q_offset = 0
                    if _mask_ratio:
                        _mask_q_offset = int(query_offset)
                    hierarchical_indices = hierarchical_topk(
                        q,
                        pooled,
                        weights,
                        pool_cache,
                        query_offset=(
                            int(query_offset) if isinstance(query_offset, int) else -1
                        ),
                        topk=self.index_topk,
                        ratio=_mask_ratio,
                        kernels=glm_fast,
                    )
                    if hierarchical_indices is not None:
                        indices = (
                            _gather_indexer_rows(
                                hierarchical_indices,
                                total_rows,
                                row_group,
                                int(pooled.shape[1]),
                            )
                            if row_sharded
                            else hierarchical_indices
                        )
                        return (
                            _broadcast_indexer_indices(
                                indices,
                                shape=tuple(indices.shape),
                                group=row_group,
                                owner=decode_owner,
                            )
                            if owner_decode
                            else indices
                        )
                    # v25 from-scratch MMA score kernel: bit-exact and faster
                    # on the qualified M2/M3/M5 DS4F pairings, including the
                    # fused pooled-ratio mask (validated across aligned and
                    # unaligned M/N and chunked-prefill offsets). Gated by
                    # fingerprint + OMLX_DSV4F_MMA_SCORE + extension
                    # symbol; every other configuration keeps the Steel
                    # path below unchanged.
                    _use_mma = (
                        self._m2_mma_score
                        and getattr(glm_fast, "_EXT_MMA_SCORE", False)
                        and q.dtype == mx.bfloat16
                        and pooled.shape[1] >= 64
                    )
                    _use_mma_wm4_wn1 = (
                        _use_mma and glm_fast.dsa_indexer_mma_wm4_wn1_eligible()
                    )
                    # M5 TensorOps score tile: exact DS4F fingerprint at
                    # construction, then the narrow runtime domain here.
                    # In heterogeneous TP, only row-sharded prefill may use a
                    # rank-local kernel: replicated rows stay on Steel on both
                    # ranks so a numerical drift can never split top-k control
                    # flow. The native primitive itself demotes to Steel if the
                    # optional NAX library/pipeline cannot load.
                    _nax_rank_safe = (
                        row_group is None or int(row_group.size()) <= 1 or row_sharded
                    )
                    _use_nax = (
                        self._nax_indexer_score
                        and _nax_rank_safe
                        and getattr(glm_fast, "_EXT_NAX_SCORE", False)
                        and glm_fast.dsa_indexer_nax_kernels_built()
                        and q.dtype == mx.bfloat16
                        and B == 1
                        and q.shape[1] == 64
                        and q.shape[3] == 128
                        and L >= 16
                        and pooled.shape[1] > 512
                        and _mask_ratio == 4
                    )
                    global _DEEPSEEK_V4_NAX_INDEXER_SCORE_LOGGED
                    if _use_nax and not _DEEPSEEK_V4_NAX_INDEXER_SCORE_LOGGED:
                        _DEEPSEEK_V4_NAX_INDEXER_SCORE_LOGGED = True
                        logging.getLogger(__name__).info(
                            "deepseek_v4: DS4F M5/NAX ratio-4 indexer score "
                            "kernel active (TensorOps 16x32 tile)"
                        )
                    global _DEEPSEEK_V4_MMA_SCORE_LOGGED
                    if _use_mma and not _DEEPSEEK_V4_MMA_SCORE_LOGGED:
                        _DEEPSEEK_V4_MMA_SCORE_LOGGED = True
                        logging.getLogger(__name__).info(
                            "deepseek_v4: DSV4F v25 MMA indexer score "
                            "kernel active (zero-per-head-barrier, "
                            "lossless qualified Apple GPU path, partition=%s)",
                            "WM4xWN1" if _use_mma_wm4_wn1 else "WM2xWN2",
                        )
                    if _use_mma:
                        scores4 = glm_fast.dsa_indexer_scores_mma(
                            q,
                            pooled[:, None],
                            weights,
                            mask_ratio=_mask_ratio,
                            mask_q_offset=_mask_q_offset,
                            use_wm4_wn1=_use_mma_wm4_wn1,
                        )
                    else:
                        scores4 = glm_fast.dsa_indexer_scores(
                            q,
                            pooled[:, None],
                            weights,
                            causal=False,
                            mask_ratio=_mask_ratio,
                            mask_q_offset=_mask_q_offset,
                            use_nax=_use_nax,
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
                    indices = mx.sort(indices, axis=-1)
                    indices = (
                        _gather_indexer_rows(
                            indices,
                            total_rows,
                            row_group,
                            int(pooled.shape[1]),
                        )
                        if row_sharded
                        else indices
                    )
                    return (
                        _broadcast_indexer_indices(
                            indices,
                            shape=tuple(indices.shape),
                            group=row_group,
                            owner=decode_owner,
                        )
                        if owner_decode
                        else indices
                    )
            except ValueError as exc:
                # Shape/argument rejection (std::invalid_argument): raised
                # before any GPU work and per-call -- e.g. a stale extension
                # binary rejecting an unaligned prefill tail. Do NOT latch:
                # later calls with different shapes may succeed natively.
                global _DEEPSEEK_V4_INDEXER_SHAPE_WARNED
                if not _DEEPSEEK_V4_INDEXER_SHAPE_WARNED:
                    _DEEPSEEK_V4_INDEXER_SHAPE_WARNED = True
                    logging.getLogger(__name__).warning(
                        "DSV4 native indexer top-k rejected this shape; MLX "
                        "fallback for this call only (native path stays "
                        "armed for later calls): %s",
                        exc,
                    )
            except Exception as exc:
                disable_native_indexer()
                logging.getLogger(__name__).warning(
                    "DSV4 native indexer top-k failed; MLX fallback "
                    "(slower prefill) for the rest of this process: %s",
                    exc,
                    exc_info=True,
                )

        # NOTE: M=1 decode deliberately keeps the fp32 GEMV path below. The
        # fused dsa_decode_scores scan was measured slower than mlx's GEMV
        # for a single row (see _dspark_fused_indexer_scores); the fused
        # route only pays off for multi-row DSpark verify batches.

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
        indices = _fp32_topk_indices(scores, k)
        indices = (
            _gather_indexer_rows(
                indices,
                total_rows,
                row_group,
                int(pooled.shape[1]),
            )
            if row_sharded
            else indices
        )
        return (
            _broadcast_indexer_indices(
                indices,
                shape=tuple(indices.shape),
                group=row_group,
                owner=decode_owner,
            )
            if owner_decode
            else indices
        )


def _dspark_fused_indexer_scores(
    indexer: "Indexer",
    pooled_rows: List[mx.array],
    projected_q: mx.array,
    projected_weights: mx.array,
    row_start: int = 0,
) -> Optional[List[mx.array]]:
    """Score DSpark verify rows with the fused dsa_decode_scores scan.

    One dispatch per row: K is streamed exactly once in its cache dtype with
    fp32 accumulation, so neither the fp32 cast of the pooled context nor the
    (rows, n_heads, P) fp32 score sheet of the reference path exists. Head
    weights stay fp32 (folded with n_heads**-0.5 * scale), so scores match
    the reference reduction semantics: relu(q . k) * scale, weighted by
    projected_weights * n_heads**-0.5, summed over heads.

    Returns one fp32 (1, P_row) score row per pooled row, or None when the
    fused kernel is opted out, ineligible for these shapes (including the
    single-row case, where the fused scan measures slower than the reference
    GEMV), or failed at runtime; the caller then uses the fp32 rowwise-GEMM
    reference path.
    """
    global _DEEPSEEK_V4_FUSED_DECODE_INDEXER_FAILED
    if (
        _DEEPSEEK_V4_FUSED_DECODE_INDEXER_ENV_DISABLED
        or _DEEPSEEK_V4_FUSED_DECODE_INDEXER_FAILED
    ):
        return None
    # Single-row calls (M=1 decode, depth-1 verify) stay on the fp32 GEMV
    # reference: measured at 100K pooled tokens the one-dispatch fused scan
    # is slightly SLOWER than mlx's GEMV (0.95 ms vs 0.89 ms — the h64 scan
    # is issue-bound at B=1), while at 2+ rows the fused route wins
    # (1.2-1.4x at 100K) because the reference pays the fp32 cast and the
    # (rows, 64, P) score sheet per row.
    if len(pooled_rows) < 2:
        return None
    # The fp32-weight instantiation exists for the 64-head DS4 indexer only;
    # 32-head DSA models are GLM and route through their own patch.
    if indexer.n_heads != 64 or getattr(indexer, "head_dim", 128) != 128:
        return None
    if projected_q.dtype not in (mx.float16, mx.bfloat16):
        return None
    if projected_q.ndim != 4 or projected_q.shape[0] != 1:
        return None
    if projected_q.shape[1] != indexer.n_heads:
        return None
    if projected_weights.shape[0] != 1:
        return None
    for row in pooled_rows:
        # The native scan requires S >= 1024. It reads K by strides and
        # re-validates the row layout (contiguous, 16B-aligned rows) itself,
        # so capacity-backed cache slices are consumed in place.
        if row.ndim != 3 or row.shape[2] != 128 or row.shape[1] < 1024:
            return None
        if row.dtype != projected_q.dtype:
            return None

    from omlx.custom_kernels.glm_moe_dsa import fast

    if not fast.has_symbol("dsa_decode_scores"):
        return None
    weight_scale = (indexer.n_heads**-0.5) * indexer.scale
    score_rows: List[mx.array] = []
    try:
        for idx, row in enumerate(pooled_rows):
            q_row = projected_q[0:1, :, row_start + idx : row_start + idx + 1, :]
            w_row = (
                projected_weights[0:1, row_start + idx].astype(mx.float32)
                * weight_scale
            )
            scores = fast.dsa_decode_scores(
                q_row, row[:, None], w_row, fp32_scores=True
            )
            score_rows.append(scores.reshape(1, row.shape[1]))
    except Exception as exc:
        _DEEPSEEK_V4_FUSED_DECODE_INDEXER_FAILED = True
        logging.getLogger(__name__).warning(
            "DSV4 fused decode indexer failed; fp32 rowwise-GEMM fallback "
            "for the rest of this process: %s",
            exc,
            exc_info=True,
        )
        return None
    return score_rows


def _batch_indexer_rows(
    indexer: Indexer,
    pooled_rows: List[mx.array],
    projected_q: mx.array,
    projected_weights: mx.array,
) -> List[Optional[mx.array]]:
    """Select each decode row's pooled indices with grouped FP32 GEMMs."""
    lengths = [int(row.shape[1]) for row in pooled_rows]
    if len(lengths) > 1 and min(lengths) > indexer.index_topk:
        # A ratio-4 boundary makes adjacent verify rows differ by one pooled
        # token. The score reduction is over the fixed 128-wide head, so
        # padding N does not alter any valid dot product. The fused path
        # scores each row at its own length and pads the score tail with the
        # sentinel; the reference path pads the pooled rows instead. Either
        # way the padded tail is masked before top-k and all rows share one
        # native top-k dispatch.
        max_length = max(lengths)
        score_rows = _dspark_fused_indexer_scores(
            indexer, pooled_rows, projected_q, projected_weights
        )
        if score_rows is not None:
            # Per-row fused scores; pad the ±1-token tails with the same
            # finfo.min sentinel the `valid` mask below writes, so the padded
            # positions stay excluded from top-k.
            scores = mx.concatenate(
                [
                    row_scores
                    if row_scores.shape[1] == max_length
                    else mx.concatenate(
                        [
                            row_scores,
                            mx.full(
                                (1, max_length - row_scores.shape[1]),
                                mx.finfo(mx.float32).min,
                                dtype=mx.float32,
                            ),
                        ],
                        axis=1,
                    )
                    for row_scores in score_rows
                ],
                axis=0,
            )
        else:
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
        indices = _fp32_topk_indices(scores, indexer.index_topk)
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

        score_rows = _dspark_fused_indexer_scores(
            indexer, pooled_rows[start:stop], projected_q, projected_weights, start
        )
        if score_rows is not None:
            # Equal-length group: rows stack without any score padding.
            scores = mx.concatenate(score_rows, axis=0)
        else:
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
        indices = _fp32_topk_indices(scores, k)
        indices = indices[:, None]
        results.extend(indices[idx : idx + 1] for idx in range(stop - start))
        start = stop

    return results


def _ragged_verify_sparse_attention(
    query_rows: mx.array,
    local_rows: List[mx.array],
    pooled_rows: List[mx.array],
    topk_rows: List[mx.array],
    scale: float,
    sinks: mx.array,
    *,
    decode_consistent: bool,
) -> mx.array:
    """Evaluate equal-geometry verify subgroups when sparse widths differ.

    At a ratio-4/128 pooling boundary, the first speculative row can select
    511 pooled entries while the following rows select 512. Padding that first
    row would either duplicate a KV entry or change the softmax reduction tree.
    Split the at-most-six verification rows into adjacent equal-geometry
    groups. The boundary row runs through the established B=1 exact path while
    the usual 512-wide tail remains batched. Each group's final pooled view is
    safe for its earlier rows because their top-k indices can only name the
    prefix they observed; no padded/duplicated entry enters the softmax.
    """

    outputs: List[mx.array] = []
    start = 0
    count = min(
        int(query_rows.shape[0]),
        len(local_rows),
        len(pooled_rows),
        len(topk_rows),
    )
    while start < count:
        signature = (
            int(local_rows[start].shape[2]),
            int(topk_rows[start].shape[-1]),
        )
        stop = start + 1
        while stop < count and (
            int(local_rows[stop].shape[2]),
            int(topk_rows[stop].shape[-1]),
        ) == signature:
            stop += 1
        group_size = stop - start
        pooled = pooled_rows[stop - 1]
        pooled_batch = mx.broadcast_to(
            pooled,
            (group_size, pooled.shape[1], pooled.shape[2]),
        )
        output = _sparse_pooled_attention(
            query_rows[start:stop],
            mx.concatenate(local_rows[start:stop], axis=0),
            pooled_batch,
            mx.concatenate(topk_rows[start:stop], axis=0),
            None,
            None,
            scale,
            sinks,
            decode_consistent=decode_consistent,
        )
        if output is None:
            raise RuntimeError("DS4 rowwise verify attention did not produce output")
        outputs.append(output)
        start = stop
    if sum(int(output.shape[0]) for output in outputs) != int(query_rows.shape[0]):
        raise RuntimeError("DS4 rowwise verify cache count is inconsistent")
    return mx.concatenate(outputs, axis=0)


def _b1_cache_offset(cache: Any, batch_size: int) -> Any:
    """Use BatchRotatingKVCache's host absolute offset for a B=1 request.

    The public ``offset`` is an MLX vector because batched rows may differ.
    Its private ``_offset`` is the same absolute position as a Python integer
    while B=1, which lets the exact DS4 WSDPA/native-mask routes dispatch
    without a device synchronization. Multi-row batches retain the vector.
    """

    if cache is None:
        return 0
    if _DEEPSEEK_V4_B1_SCALAR_OFFSET and batch_size == 1:
        host_offset = getattr(cache, "_offset", None)
        if isinstance(host_offset, int) and not isinstance(host_offset, bool):
            return host_offset
    return cache.offset


def _canonical_wide_attention_prefill(
    module: nn.Module,
    x: mx.array,
    cache: Any,
    *,
    standard_mask: bool,
    call: Any,
) -> Optional[mx.array]:
    """Run a 2K outer tile as two exact 1K attention/cache transactions.

    HyperConnection and attention are the chunk-sensitive arithmetic frontier.
    Keeping only those boundaries at 1K lets the surrounding block retain 2K
    routed-MoE/GEMM scheduling without changing the established 1K result.
    """

    if (
        not _DEEPSEEK_V4_CANONICAL_WIDE_PREFILL
        or getattr(_DEEPSEEK_V4_CANONICAL_WIDE_PREFILL_STATE, "active", False)
        or getattr(module, "training", False)
        or is_dspark_verify_armed()
        or not standard_mask
        or cache is None
        or x.ndim != 3
        or tuple(x.shape) != (1, 2048, 4096)
        or x.dtype != mx.bfloat16
        or int(getattr(module, "compress_ratio", 0)) not in (4, 128)
    ):
        return None

    global _DEEPSEEK_V4_CANONICAL_WIDE_PREFILL_LOGGED
    if not _DEEPSEEK_V4_CANONICAL_WIDE_PREFILL_LOGGED:
        _DEEPSEEK_V4_CANONICAL_WIDE_PREFILL_LOGGED = True
        logging.getLogger(__name__).info(
            "deepseek_v4: preserving the 1K HC/attention arithmetic frontier "
            "inside a 2K outer prefill tile"
        )

    outputs = []
    _DEEPSEEK_V4_CANONICAL_WIDE_PREFILL_STATE.active = True
    try:
        for start in (0, 1024):
            part = x[:, start : start + 1024]
            first = cache[0] if hasattr(cache, "caches") else cache
            mask = create_attention_mask(
                part,
                first,
                window_size=module.config.sliding_window,
                return_array=True,
            )
            output = call(
                part,
                mask,
                cache,
                _standard_mask=True,
            )
            # A true outer 1K step queues its cache before the next model call.
            # async_eval preserves that dependency/alias boundary without the
            # host synchronization that erased the wider block's gain.
            mx.async_eval(output, cache.state)
            outputs.append(output)
    finally:
        _DEEPSEEK_V4_CANONICAL_WIDE_PREFILL_STATE.active = False
    return mx.concatenate(outputs, axis=1)


class LocalAttention(nn.Module):
    """DeepSeek V4 attention with no KV compression."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.dspark = _is_dspark_model(config)
        # Set True per model instance by the MTP patch when draft/verify
        # equality is required.  Ordinary decode can use fused SDPA.
        self._omlx_decode_consistent = False
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
        offset = _b1_cache_offset(cache, B)
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        projected = _ane_attention_input(self, x)
        if projected:
            q_a = _projection_or(projected, "wq_a", self.wq_a, x)
            kv_raw = _projection_or(projected, "wkv", self.wkv, x)
        else:
            projection_bank = _decode_qkv_projection_bundle(self, x)
            if projection_bank is None:
                projection_bank = _owned_projection_bank(
                    x,
                    (self.wq_a, self.wkv),
                    self.sharding_group,
                )
            if projection_bank is None:
                verify_bank = _verify_q_a_kv_bank(self, x)
                if verify_bank is None:
                    q_a = self.wq_a(x)
                    kv_raw = self.wkv(x)
                else:
                    q_a, kv_raw = verify_bank
            else:
                q_a, kv_raw = projection_bank
        q_raw = _project_verify_q_b(self.wq_b, self.q_norm(q_a))
        q_raw = q_raw.reshape(B, L, self.n_heads, self.head_dim)
        q, kv = _finalize_attention_qkv(self, q_raw, kv_raw, offset)
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

        if _exact_decode_required(self, B, L):
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
        self._omlx_decode_consistent = False
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
        canonical = _canonical_wide_attention_prefill(
            self,
            x,
            cache,
            standard_mask=_standard_mask,
            call=self.__call__,
        )
        if canonical is not None:
            return canonical
        B, L, _ = x.shape
        local_cache = cache[0] if cache is not None else None
        pool_cache = cache[1] if cache is not None else None
        offset = _b1_cache_offset(local_cache, B)
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        projected = _ane_attention_input(self, x)
        if projected:
            q_a = _projection_or(projected, "wq_a", self.wq_a, x)
            kv_raw = _projection_or(projected, "wkv", self.wkv, x)
            compressed_kv = projected.get("compressor_wkv")
            compressed_gate = projected.get("compressor_wgate")
            compressor_projection = (
                (compressed_kv, compressed_gate)
                if compressed_kv is not None and compressed_gate is not None
                else None
            )
        else:
            projection_bank = _decode_qkv_projection_bundle(self, x)
            if projection_bank is None:
                projection_bank = _owned_projection_bank(
                    x,
                    (
                        self.wq_a,
                        self.wkv,
                        self.compressor.wkv,
                        self.compressor.wgate,
                    ),
                    self.sharding_group,
                )
            if projection_bank is None:
                verify_bank = _verify_q_a_kv_bank(self, x)
                if verify_bank is None:
                    q_a = self.wq_a(x)
                    kv_raw = self.wkv(x)
                else:
                    q_a, kv_raw = verify_bank
                compressor_projection = None
            else:
                q_a, kv_raw, compressed_kv, compressed_gate = projection_bank
                compressor_projection = (compressed_kv, compressed_gate)
        q_raw = _project_verify_q_b(self.wq_b, self.q_norm(q_a))
        q_raw = q_raw.reshape(B, L, self.n_heads, self.head_dim)
        q, kv = _finalize_attention_qkv(self, q_raw, kv_raw, offset)
        sinks = self.attn_sink.astype(q.dtype)
        if is_dspark_verify_armed() and B == 1 and 1 < L <= 6:
            if compressor_projection is None:
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

        pooled = (
            self.compressor(x, pool_cache, offset)
            if compressor_projection is None
            else self.compressor.consume(
                *compressor_projection,
                pool_cache,
                offset,
            )
        )
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
            and not _exact_decode_required(self, B, L)
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
                decode_consistent=self._omlx_decode_consistent,
                native_only=True,
                _standard_mask=_standard_mask,
            )
        if out is None:
            local_width = int(kv.shape[2])
            pooled_width = int(pooled.shape[1])
            if pooled.shape[1] > 0:
                kv = mx.concatenate([kv, pooled[:, None]], axis=2)
            mask = _extend_mask(
                mask,
                pooled_mask,
                kv.shape[2],
                local_width=local_width,
                pooled_width=pooled_width,
            )
            if _exact_decode_required(self, B, L):
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
        self._omlx_decode_consistent = False
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
        canonical = _canonical_wide_attention_prefill(
            self,
            x,
            cache,
            standard_mask=_standard_mask,
            call=self.__call__,
        )
        if canonical is not None:
            return canonical
        B, L, _ = x.shape
        local_cache = cache[0] if cache is not None else None
        comp_cache = cache[1] if cache is not None else None
        idx_cache = cache[2] if cache is not None else None
        offset = _b1_cache_offset(local_cache, B)
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        projected = _ane_attention_input(self, x)
        if projected:
            q_a = _projection_or(projected, "wq_a", self.wq_a, x)
            kv_raw = _projection_or(projected, "wkv", self.wkv, x)
            compressed_kv = projected.get("compressor_wkv")
            compressed_gate = projected.get("compressor_wgate")
            compressor_projection = (
                (compressed_kv, compressed_gate)
                if compressed_kv is not None and compressed_gate is not None
                else None
            )
            index_kv = projected.get("indexer_compressor_wkv")
            index_gate = projected.get("indexer_compressor_wgate")
            index_compressor_projection = (
                (index_kv, index_gate)
                if index_kv is not None and index_gate is not None
                else None
            )
        else:
            projection_bank = _decode_qkv_projection_bundle(self, x)
            if projection_bank is None:
                projection_bank = _owned_projection_bank(
                    x,
                    (
                        self.wq_a,
                        self.wkv,
                        self.compressor.wkv,
                        self.compressor.wgate,
                        self.indexer.compressor.wkv,
                        self.indexer.compressor.wgate,
                    ),
                    self.sharding_group,
                )
            if projection_bank is None:
                verify_bank = _verify_q_a_kv_bank(self, x)
                if verify_bank is None:
                    q_a = self.wq_a(x)
                    kv_raw = self.wkv(x)
                else:
                    q_a, kv_raw = verify_bank
                compressor_projection = None
                index_compressor_projection = None
            else:
                (
                    q_a,
                    kv_raw,
                    compressed_kv,
                    compressed_gate,
                    index_kv,
                    index_gate,
                ) = projection_bank
                compressor_projection = (compressed_kv, compressed_gate)
                index_compressor_projection = (index_kv, index_gate)
        q_residual = self.q_norm(q_a)
        if is_dspark_verify_armed():
            q_raw = _project_verify_q_b(self.wq_b, q_residual)
            indexer_q_raw = None
        else:
            q_raw, indexer_q_raw = _ane_stacked_q(
                self.wq_b,
                getattr(self.indexer, "wq_b", None),
                q_residual,
            )
        q_raw = q_raw.reshape(B, L, self.n_heads, self.head_dim)
        q, kv = _finalize_attention_qkv(self, q_raw, kv_raw, offset)
        sinks = self.attn_sink.astype(q.dtype)
        if is_dspark_verify_armed() and B == 1 and L <= 6:
            if compressor_projection is None:
                compressed_kv, compressed_gate = self.compressor.project(x)
            if index_compressor_projection is None:
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
                index_q = _project_verify_q_b(self.indexer.wq_b, q_residual).reshape(
                    B,
                    L,
                    self.indexer.n_heads,
                    self.indexer.head_dim,
                )
                index_q = index_q.transpose(0, 2, 1, 3)
                index_q = self.rope(index_q, offset)
                index_weights = _projection_or(
                    projected,
                    "indexer_weights",
                    self.indexer.weights_proj,
                    x,
                )
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
                topk_widths = tuple(int(row.shape[-1]) for row in topk_rows)
                ragged_topk = len(set(topk_widths)) > 1
                topk_batch = (
                    None if ragged_topk else mx.concatenate(topk_rows, axis=0)
                )
                use_ring_kernel = bool(
                    not _DEEPSEEK_V4_RING_VERIFY_DISABLED
                    and not ragged_topk
                    and ring_view is not None
                    # The physical-ring kernel has an aligned 64-row MMA
                    # store. TP ranks own fewer query heads, so they must use
                    # the materialized rowwise path until a tail-safe ring
                    # kernel is available.
                    and q.shape[1] == 64
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
                    if ragged_topk:
                        if local_rows is None:
                            local_rows = _materialize_rotating_verify_rows(ring_view)
                        batch_out = _ragged_verify_sparse_attention(
                            query_batch,
                            local_rows,
                            pooled_rows,
                            topk_rows,
                            self.scale,
                            sinks,
                            decode_consistent=self._omlx_decode_consistent,
                        )
                    elif use_ring_kernel:
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
                            decode_consistent=self._omlx_decode_consistent,
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

        pooled = (
            self.compressor(x, comp_cache, offset)
            if compressor_projection is None
            else self.compressor.consume(
                *compressor_projection,
                comp_cache,
                offset,
            )
        )
        pmask = comp_cache.make_mask(L, offset) if comp_cache is not None else None
        if 0 < pooled.shape[1] <= self.indexer.index_topk:
            index_pooled = (
                self.indexer.compressor(x, idx_cache, offset)
                if index_compressor_projection is None
                else self.indexer.compressor.consume(
                    *index_compressor_projection,
                    idx_cache,
                    offset,
                )
            )
            if index_pooled.shape[1] != pooled.shape[1]:
                raise RuntimeError(
                    "DeepSeek V4 attention/indexer pooling caches diverged"
                )
            topk = mx.broadcast_to(
                mx.arange(pooled.shape[1], dtype=mx.uint32)[None, None],
                (B, L, pooled.shape[1]),
            )
        else:
            projected_q = None
            if indexer_q_raw is not None:
                projected_q = self.rope(
                    indexer_q_raw.reshape(
                        B, L, self.indexer.n_heads, self.indexer.head_dim
                    ).transpose(0, 2, 1, 3),
                    offset,
                )
            topk = self.indexer(
                x,
                q_residual,
                self.rope,
                idx_cache,
                offset,
                compressor_projection=index_compressor_projection,
                projected_q=projected_q,
                projected_weights=projected.get("indexer_weights"),
            )
        sparse_mask = None
        if pmask is not None and topk is not None:
            sparse_mask = mx.take_along_axis(
                pmask[None] if pmask.ndim == 2 else pmask,
                topk,
                axis=2,
            )[:, None]

        if pooled.shape[1] == 0:
            if _exact_decode_required(self, B, L):
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
            if _exact_decode_required(self, B, L):
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
                    local_width = int(kv.shape[2])
                    pooled_width = int(pooled.shape[1])
                    full_kv = mx.concatenate([kv, pooled[:, None]], axis=2)
                    mask = _extend_mask(
                        mask,
                        pmask,
                        full_kv.shape[2],
                        local_width=local_width,
                        pooled_width=pooled_width,
                    )
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
                decode_consistent=self._omlx_decode_consistent,
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
        overlap_hc = _can_overlap_hc_residual(self, h)
        residual = h
        fused_hc = self.attn_hc.call_with_norm(h, self.attn_norm)
        if fused_hc is None:
            x, post, comb = self.attn_hc(h)
            attn_input = self.attn_norm(x)
        else:
            x, attn_input, post, comb = fused_hc
        residual_branch = hc_residual_branch(residual, comb) if overlap_hc else None
        x = self.attn(
            attn_input,
            mask=mask,
            cache=cache,
            _standard_mask=_standard_mask,
        )
        h = (
            hc_merge_branch(x, post, residual_branch)
            if residual_branch is not None
            else hc_expand(x, residual, post, comb)
        )

        residual = h
        fused_hc = self.ffn_hc.call_with_norm(h, self.ffn_norm)
        if fused_hc is None:
            x, post, comb = self.ffn_hc(h)
            x = self.ffn_norm(x)
        else:
            _collapsed, x, post, comb = fused_hc
        residual_branch = hc_residual_branch(residual, comb) if overlap_hc else None
        x = self.ffn(x, input_ids)
        return (
            hc_merge_branch(x, post, residual_branch)
            if residual_branch is not None
            else hc_expand(x, residual, post, comb)
        )


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

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        h = self.embed_tokens(inputs)
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

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            # This mask was created above from the model's own cache/window.
            h = layer(h, mask, layer_cache, inputs, _standard_mask=True)

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

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
        return self.lm_head(self.model(inputs, cache))

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
        outer_shard_weights = _validated_ds4_tp_weights(self.args, group)
        shard_weights = _validated_ds4_non_moe_tp_weights(
            self.args, group, outer_shard_weights
        )
        moe_shard_weights = _validated_ds4_moe_tp_weights(
            self.args, group, outer_shard_weights
        )
        for layer in self.model.layers:
            layer.attn.sharding_group = group
            indexer = getattr(layer.attn, "indexer", None)
            if indexer is not None:
                indexer.row_sharding_group = group
            layer.attn.wq_b = _shard_linear_weighted(
                layer.attn.wq_b,
                "all-to-sharded",
                segments=self.args.o_groups,
                group=group,
                weights=shard_weights,
            )
            _shard_inplace_weighted(
                layer.attn.wo_a,
                "sharded-to-all",
                group=group,
                weights=shard_weights,
            )
            # wq_b shards segment-interleaved (segments=o_groups): rank r
            # keeps slice r of *every* head group, not the contiguous r-th
            # block of all heads. Slice the sinks the same way or each one
            # gates the wrong head under TP.
            sinks = layer.attn.attn_sink.reshape(self.args.o_groups, -1)
            if shard_weights is None:
                layer.attn.attn_sink = mx.split(sinks, N, axis=1)[rank].reshape(-1)
                layer.attn.n_heads //= N
            else:
                layer.attn.attn_sink = _weighted_segment_slice(
                    sinks,
                    axis=1,
                    segments=1,
                    rank=rank,
                    weights=shard_weights,
                ).reshape(-1)
                layer.attn.n_heads = self.args.o_groups * shard_weights[rank]

            layer.ffn.sharding_group = group
            _shard_inplace_weighted(
                layer.ffn.shared_experts.gate_proj,
                "all-to-sharded",
                group=group,
                weights=shard_weights,
            )
            _shard_inplace_weighted(
                layer.ffn.shared_experts.down_proj,
                "sharded-to-all",
                group=group,
                weights=shard_weights,
            )
            _shard_inplace_weighted(
                layer.ffn.shared_experts.up_proj,
                "all-to-sharded",
                group=group,
                weights=shard_weights,
            )
            _shard_inplace_weighted(
                layer.ffn.switch_mlp.gate_proj,
                "all-to-sharded",
                group=group,
                weights=moe_shard_weights,
            )
            _shard_inplace_weighted(
                layer.ffn.switch_mlp.down_proj,
                "sharded-to-all",
                group=group,
                weights=moe_shard_weights,
            )
            _shard_inplace_weighted(
                layer.ffn.switch_mlp.up_proj,
                "all-to-sharded",
                group=group,
                weights=moe_shard_weights,
            )
            layer.ffn.switch_mlp._omlx_dsv4f_moe_tp = (
                int(N),
                int(rank),
                tuple(moe_shard_weights or ()),
            )
