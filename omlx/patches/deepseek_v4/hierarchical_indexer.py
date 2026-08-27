# SPDX-License-Identifier: Apache-2.0
"""Certified low-rank screen for DeepSeek-V4's long-context indexer.

This is an opt-in exact accelerator, not approximate retrieval.  A rank-48
PCA screen supplies an upper bound for every pooled key.  Shared candidate
sets are rescored by the promoted BF16 D=128 kernel, and a strict cutoff
certificate proves that no omitted key can enter top-k.  Any unsupported
shape, setup failure, or failed certificate returns ``None`` so the caller
runs the complete exact scan.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("OMLX_DSV4_HIERARCHICAL_INDEXER", "0").strip().lower() in (
    "1",
    "true",
    "on",
    "yes",
)
_MIN_POOL = int(os.getenv("OMLX_DSV4_HIERARCHICAL_MIN_POOL", "16000"))
_REFRESH_POOL = int(os.getenv("OMLX_DSV4_HIERARCHICAL_REFRESH_POOL", "2048"))
_RANK = 48
_GROUP_ROWS = 16
_CANDIDATE_FRACTION = float(
    os.getenv("OMLX_DSV4_HIERARCHICAL_CANDIDATE_FRACTION", "0.30")
)
_TRACE = os.getenv("OMLX_DSV4_HIERARCHICAL_TRACE", "0").strip().lower() in (
    "1",
    "true",
    "on",
    "yes",
)
_NATIVE_UPPER_ENABLED = os.getenv(
    "OMLX_DSV4_HIERARCHICAL_NATIVE_UPPER", "0"
).strip().lower() in ("1", "true", "on", "yes")
_NUMERIC_ABS_GUARD = 0.02
_NUMERIC_REL_GUARD = 0.005
# The native pass rounds the same FP32 terms explicitly but may be compiled
# with a different contraction choice than MLX's elementwise graph. Add a
# small outward-only margin so its upper bound can never fall below the proven
# Python reference due solely to one FP32 ULP.
_NATIVE_OUTWARD_GUARD = 1e-4
_STATE_ATTR = "_omlx_dsv4_hierarchical_indexer_state"
_MISS_BACKOFF_ATTR = "_omlx_dsv4_hierarchical_miss_backoff"
_MISS_BACKOFF_MAX_MULTIPLIER = 8
_SUCCESS_LOGGED = False
_FALLBACK_LOGGED = False
_TRACE_REASONS: set[str] = set()
_GROUP_UPPER_KERNEL = None
_NATIVE_UPPER_FAILED = False


_GROUP_UPPER_SOURCE = r"""
    const uint index = thread_position_in_grid.x;
    const uint rows = uint(params[0]);
    const uint pool = uint(params[1]);
    const uint groups = rows / uint(GROUP_ROWS);
    const uint total = groups * pool;
    if (index >= total) {
        return;
    }

    const uint group = index / pool;
    const uint key = index - group * pool;
    const uint row_start = group * uint(GROUP_ROWS);
    const float kr = key_residual[key];
    const float kn = key_norm[key];
    const float ke = key_error[key];
    const float abs_guard = guards[0];
    const float rel_guard = guards[1];
    const float outward_guard = guards[2];

    float best = -INFINITY;
    #pragma unroll
    for (uint local_row = 0; local_row < uint(GROUP_ROWS); ++local_row) {
        const uint row = row_start + local_row;
        const float score = float(approximate[(uint64_t)row * pool + key]);
        const float rr = row_residual[row];
        const float re = row_error[row];
        const float rn = row_norm[row];
        const float bound0 = rr * kr;
        const float bound1 = re * kn;
        const float bound2 = (rn + re) * ke;
        const float upper = score + bound0 + bound1 + bound2
            + abs_guard + rel_guard * metal::abs(score) + outward_guard;
        best = metal::max(best, upper);
    }
    group_upper[index] = best;
"""


def _group_upper_kernel():
    global _GROUP_UPPER_KERNEL
    if _GROUP_UPPER_KERNEL is None:
        _GROUP_UPPER_KERNEL = mx.fast.metal_kernel(
            name="omlx_dsv4_hierarchical_group_upper",
            input_names=[
                "approximate",
                "row_residual",
                "row_error",
                "row_norm",
                "key_residual",
                "key_norm",
                "key_error",
                "params",
                "guards",
            ],
            output_names=["group_upper"],
            source=_GROUP_UPPER_SOURCE,
            ensure_row_contiguous=True,
        )
    return _GROUP_UPPER_KERNEL


def _native_group_upper(
    approximate: mx.array,
    residual_factor: mx.array,
    coordinate_error_factor: mx.array,
    coordinate_norm_factor: mx.array,
    state: "_LowRankState",
) -> mx.array | None:
    """Fuse the certified FP32 error bound and 16-row maximum in Metal."""

    global _NATIVE_UPPER_FAILED
    if not _NATIVE_UPPER_ENABLED or _NATIVE_UPPER_FAILED:
        return None
    rows, pool = map(int, approximate.shape)
    if rows < _GROUP_ROWS or rows % _GROUP_ROWS != 0 or pool < 1:
        return None
    try:
        params = mx.array([rows, pool], dtype=mx.int32)
        guards = mx.array(
            [_NUMERIC_ABS_GUARD, _NUMERIC_REL_GUARD, _NATIVE_OUTWARD_GUARD],
            dtype=mx.float32,
        )
        (group_upper,) = _group_upper_kernel()(
            inputs=[
                approximate,
                residual_factor,
                coordinate_error_factor,
                coordinate_norm_factor,
                state.key_orthogonal_residual,
                state.key_coordinate_norm,
                state.key_coordinate_error,
                params,
                guards,
            ],
            template=[("T", approximate.dtype), ("GROUP_ROWS", _GROUP_ROWS)],
            grid=(((rows // _GROUP_ROWS) * pool + 255) // 256 * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows // _GROUP_ROWS, pool)],
            output_dtypes=[mx.float32],
        )
        return group_upper
    except Exception:
        _NATIVE_UPPER_FAILED = True
        logger.warning(
            "DS4 native hierarchical upper-bound pass failed; using MLX graph",
            exc_info=True,
        )
        return None


@dataclass
class _LowRankState:
    basis: mx.array
    key_projection: mx.array
    key_orthogonal_residual: mx.array
    key_coordinate_error: mx.array
    key_coordinate_norm: mx.array
    basis_pool_length: int
    projected_pool_length: int


@dataclass(frozen=True)
class _MissBackoff:
    next_pool_length: int
    last_pool_length: int
    streak: int


def _miss_backoff_active(pool_cache: Any, pool_length: int) -> bool:
    """Skip a known-miss layer until its next bounded basis-refresh retry."""

    state = getattr(pool_cache, _MISS_BACKOFF_ATTR, None)
    if not isinstance(state, _MissBackoff):
        return False
    if pool_length < state.last_pool_length:
        # Cache reset/shorter restore: the old request cannot suppress a new
        # prefix's first certificate attempt.
        delattr(pool_cache, _MISS_BACKOFF_ATTR)
        return False
    return pool_length < state.next_pool_length


def _record_certificate_miss(pool_cache: Any, pool_length: int) -> _MissBackoff:
    previous = getattr(pool_cache, _MISS_BACKOFF_ATTR, None)
    streak = (
        previous.streak + 1
        if isinstance(previous, _MissBackoff)
        and pool_length >= previous.next_pool_length
        else 1
    )
    multiplier = min(
        1 << min(streak - 1, 3),
        _MISS_BACKOFF_MAX_MULTIPLIER,
    )
    delay = max(1, _REFRESH_POOL) * multiplier
    state = _MissBackoff(
        next_pool_length=pool_length + delay,
        last_pool_length=pool_length,
        streak=streak,
    )
    setattr(pool_cache, _MISS_BACKOFF_ATTR, state)
    return state


def _clear_certificate_miss(pool_cache: Any) -> None:
    if hasattr(pool_cache, _MISS_BACKOFF_ATTR):
        delattr(pool_cache, _MISS_BACKOFF_ATTR)


def _trace_once(reason: str) -> None:
    """Emit one operator-requested live diagnostic without touching hot paths."""

    if _TRACE and reason not in _TRACE_REASONS:
        _TRACE_REASONS.add(reason)
        print(f"OMLX_DSV4_HIERARCHICAL_TRACE {reason}", flush=True)


def _project_keys(keys: mx.array, basis: mx.array) -> tuple[mx.array, ...]:
    keys_f = keys.astype(mx.float32)
    coordinates = keys_f @ basis
    quantized = coordinates.astype(mx.bfloat16)
    orthogonal_residual = mx.linalg.norm(
        keys_f - coordinates @ basis.T,
        axis=-1,
    )
    coordinate_error = mx.linalg.norm(
        coordinates - quantized.astype(mx.float32),
        axis=-1,
    )
    coordinate_norm = mx.linalg.norm(quantized.astype(mx.float32), axis=-1)
    return quantized, orthogonal_residual, coordinate_error, coordinate_norm


def _build_state(pooled: mx.array) -> _LowRankState:
    # The covariance is only 128x128.  Copying one rank-local pooled cache to
    # NumPy and solving that tiny symmetric system measured ~2 ms after the
    # first BLAS warmup; basis refresh is amortized over many prompt chunks.
    import numpy as np

    keys_f = pooled[0].astype(mx.float32)
    mx.eval(keys_f)
    keys_np = np.asarray(keys_f)
    covariance = keys_np.T @ keys_np
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:_RANK]
    basis = mx.array(eigenvectors[:, order].copy())
    projection, residual, error, norm = _project_keys(keys_f, basis)
    mx.eval(basis, projection, residual, error, norm)
    length = int(pooled.shape[1])
    return _LowRankState(
        basis=basis,
        key_projection=projection,
        key_orthogonal_residual=residual,
        key_coordinate_error=error,
        key_coordinate_norm=norm,
        basis_pool_length=length,
        projected_pool_length=length,
    )


def _state_for_cache(pool_cache: Any, pooled: mx.array) -> _LowRankState:
    length = int(pooled.shape[1])
    state = getattr(pool_cache, _STATE_ATTR, None)
    if not isinstance(state, _LowRankState) or state.projected_pool_length > length:
        state = _build_state(pooled)
    elif length - state.basis_pool_length >= max(1, _REFRESH_POOL):
        state = _build_state(pooled)
    elif state.projected_pool_length < length:
        start = state.projected_pool_length
        projection, residual, error, norm = _project_keys(
            pooled[0, start:length], state.basis
        )
        state = _LowRankState(
            basis=state.basis,
            key_projection=mx.concatenate(
                [state.key_projection, projection], axis=0
            ),
            key_orthogonal_residual=mx.concatenate(
                [state.key_orthogonal_residual, residual], axis=0
            ),
            key_coordinate_error=mx.concatenate(
                [state.key_coordinate_error, error], axis=0
            ),
            key_coordinate_norm=mx.concatenate(
                [state.key_coordinate_norm, norm], axis=0
            ),
            basis_pool_length=state.basis_pool_length,
            projected_pool_length=length,
        )
    setattr(pool_cache, _STATE_ATTR, state)
    return state


def hierarchical_topk(
    q: mx.array,
    pooled: mx.array,
    weights: mx.array,
    pool_cache: Any,
    *,
    query_offset: int,
    topk: int,
    ratio: int,
    kernels: Any,
) -> mx.array | None:
    """Return certified sorted ``[1, L, topk]`` indices, or ``None``.

    The host synchronization at the end is intentional: it is the fail-closed
    certificate boundary.  It occurs only for long prompts behind the opt-in
    gate and replaces, rather than supplements, the full score-sheet eval.
    """

    if not _ENABLED:
        return None
    if (
        pool_cache is None
        or not isinstance(query_offset, int)
        or ratio != 4
        or q.dtype != mx.bfloat16
        or pooled.dtype != mx.bfloat16
        or weights.dtype != mx.bfloat16
        or tuple(q.shape[:2]) != (1, 64)
        or q.shape[3] != 128
        or tuple(weights.shape) != (1, q.shape[2], 64)
        or pooled.ndim != 3
        or pooled.shape[0] != 1
        or pooled.shape[2] != 128
        or q.shape[2] < _GROUP_ROWS
        or q.shape[2] % _GROUP_ROWS != 0
        or pooled.shape[1] < max(_MIN_POOL, topk * 2)
        or not getattr(kernels, "_EXT_MMA_SCORE", False)
    ):
        _trace_once(
            "skip_gate "
            f"q={tuple(q.shape)} pooled={tuple(pooled.shape)} "
            f"weights={tuple(weights.shape)} ratio={ratio} topk={topk} "
            f"query_offset={query_offset} mma="
            f"{bool(getattr(kernels, '_EXT_MMA_SCORE', False))}"
        )
        return None

    pool_length = int(pooled.shape[1])
    if _miss_backoff_active(pool_cache, pool_length):
        backoff = getattr(pool_cache, _MISS_BACKOFF_ATTR)
        _trace_once(
            "certificate_backoff "
            f"streak={backoff.streak} next_pool={backoff.next_pool_length}"
        )
        return None

    try:
        state = _state_for_cache(pool_cache, pooled)
        rows = int(q.shape[2])
        groups = rows // _GROUP_ROWS
        candidate_count = max(
            topk + 1,
            int(math.ceil(pool_length * _CANDIDATE_FRACTION / 256.0)) * 256,
        )
        candidate_count = min(candidate_count, pool_length - 1)
        if candidate_count <= topk:
            _trace_once(
                f"skip_candidate_count pool={pool_length} topk={topk} "
                f"candidates={candidate_count}"
            )
            return None

        q_f = q[0].transpose(1, 0, 2).astype(mx.float32)
        abs_weights = mx.abs(weights[0].astype(mx.float32))
        coordinates = q_f @ state.basis
        quantized_q = coordinates.astype(mx.bfloat16)
        q_orthogonal_residual = mx.linalg.norm(
            q_f - coordinates @ state.basis.T,
            axis=-1,
        )
        q_coordinate_error = mx.linalg.norm(
            coordinates - quantized_q.astype(mx.float32),
            axis=-1,
        )
        q_coordinate_norm = mx.linalg.norm(
            quantized_q.astype(mx.float32), axis=-1
        )

        approximate_bf16 = kernels.dsa_indexer_scores_mma(
            quantized_q.transpose(1, 0, 2)[None],
            state.key_projection[None, None],
            weights,
            mask_ratio=ratio,
            mask_q_offset=query_offset,
        )[0, 0]

        # Exact dot-error bound for orthogonal PCA residuals plus BF16
        # coordinate quantization:
        #   |q.k - q_b.k_b| <= |r_q||r_k| + |e_q||k_b|
        #                         + (|q_b|+|e_q|)|e_k|.
        residual_factor = mx.sum(
            abs_weights * q_orthogonal_residual, axis=1
        )
        coordinate_error_factor = mx.sum(
            abs_weights * q_coordinate_error, axis=1
        )
        coordinate_norm_factor = mx.sum(
            abs_weights * q_coordinate_norm, axis=1
        )
        group_upper = _native_group_upper(
            approximate_bf16,
            residual_factor,
            coordinate_error_factor,
            coordinate_norm_factor,
            state,
        )
        if group_upper is None:
            approximate = approximate_bf16.astype(mx.float32)
            error_bound = (
                residual_factor[:, None] * state.key_orthogonal_residual[None]
                + coordinate_error_factor[:, None]
                * state.key_coordinate_norm[None]
                + (coordinate_norm_factor + coordinate_error_factor)[:, None]
                * state.key_coordinate_error[None]
            )
            upper = (
                approximate
                + error_bound
                + _NUMERIC_ABS_GUARD
                + _NUMERIC_REL_GUARD * mx.abs(approximate)
            )
            group_upper = mx.max(
                upper.reshape(groups, _GROUP_ROWS, pool_length), axis=1
            )
        candidates = mx.sort(
            mx.argpartition(-group_upper, candidate_count - 1, axis=-1)[
                :, :candidate_count
            ],
            axis=-1,
        )
        candidate_keys = mx.take(pooled[0], candidates, axis=0)[:, None]
        q_batch = q[0].reshape(64, groups, _GROUP_ROWS, 128).transpose(
            1, 0, 2, 3
        )
        weight_batch = weights[0].reshape(groups, _GROUP_ROWS, 64)
        exact_scores = kernels.dsa_indexer_scores_mma(
            q_batch,
            candidate_keys,
            weight_batch,
            mask_ratio=0,
            mask_q_offset=0,
        )[:, 0]

        row_ids = mx.arange(rows).reshape(groups, _GROUP_ROWS)
        visible = ((query_offset + row_ids + 1) // ratio)[:, :, None]
        exact_scores = mx.where(
            candidates[:, None] < visible,
            exact_scores,
            mx.finfo(mx.bfloat16).min,
        )
        local_indices = kernels.dsa_topk_indices(
            exact_scores[:, None], topk, bucketed=False
        )[:, 0]
        mapped = mx.sort(
            mx.take_along_axis(
                mx.broadcast_to(
                    candidates[:, None],
                    (groups, _GROUP_ROWS, candidate_count),
                ),
                local_indices,
                axis=-1,
            ),
            axis=-1,
        ).reshape(1, rows, topk)

        cutoff = mx.min(
            mx.take_along_axis(
                exact_scores, local_indices, axis=-1
            ).astype(mx.float32),
            axis=-1,
        )
        selected_upper_floor = mx.min(
            mx.take_along_axis(group_upper, candidates, axis=-1), axis=-1
        )
        certificate = cutoff > selected_upper_floor[:, None]
        mx.eval(mapped, certificate)
        if not bool(mx.all(certificate).item()):
            backoff = _record_certificate_miss(pool_cache, pool_length)
            global _FALLBACK_LOGGED
            if not _FALLBACK_LOGGED:
                _FALLBACK_LOGGED = True
                logger.warning(
                    "DS4 hierarchical index certificate missed; using full scan "
                    "(pool=%d candidates=%d)",
                    pool_length,
                    candidate_count,
                )
            _trace_once(
                f"certificate_miss pool={pool_length} candidates={candidate_count} "
                f"retry={backoff.next_pool_length} streak={backoff.streak}"
            )
            return None

        _clear_certificate_miss(pool_cache)
        global _SUCCESS_LOGGED
        if not _SUCCESS_LOGGED:
            _SUCCESS_LOGGED = True
            logger.info(
                "DS4 certified hierarchical index active: rank=%d pool=%d "
                "candidates=%d group_rows=%d",
                _RANK,
                pool_length,
                candidate_count,
                _GROUP_ROWS,
            )
        _trace_once(
            f"certificate_pass pool={pool_length} candidates={candidate_count}"
        )
        return mapped
    except Exception as exc:
        _trace_once(f"exception={type(exc).__name__}")
        logger.warning(
            "DS4 hierarchical index failed closed; using full scan",
            exc_info=True,
        )
        return None
