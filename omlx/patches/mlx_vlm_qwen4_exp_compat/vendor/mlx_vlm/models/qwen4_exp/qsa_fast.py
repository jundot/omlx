# SPDX-License-Identifier: Apache-2.0
"""Exact gathered QSA for contiguous batch-one text prompts.

The native path reads selected four-token blocks directly from K/V. The MLX
fallback gathers the selected rows and causal tail. Eligible text-only Lightning
MTP verification uses this path too. Batched, padded, and multimodal requests
use mlx-vlm's general implementation.
"""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Callable

import mlx.core as mx

IndexKeyNorm = Callable[[mx.array], mx.array]
IndexRoPE = Callable[[mx.array, mx.array], mx.array]


_NATIVE_QSA_SCORE_DISABLED = False
_NATIVE_QSA_SCORE_PROVEN = False
_NATIVE_QSA_TOPK_DISABLED = False
_NATIVE_QSA_TOPK_PROVEN = False
_NATIVE_QSA_MAIN_DISABLED = False
_NATIVE_QSA_MAIN_PROVEN = False


def _nax_gpu() -> bool:
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:
        return False


def _min_rows(env: str, nax_default: int) -> int:
    raw = os.environ.get(env, "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return nax_default if _nax_gpu() else 0


@functools.lru_cache(maxsize=None)
def _native_score_min_rows() -> int:
    """Query rows from which the native indexer-score kernel engages; below it the
    MLX ops are faster on NAX GPUs (0.27 vs 0.36-0.77 ms per layer at 1-16 rows)."""
    return _min_rows("OMLX_QWEN4_QSA_NATIVE_SCORE_MIN_ROWS", 32)


@functools.lru_cache(maxsize=None)
def _native_topk_min_rows() -> int:
    """Query rows from which the native top-k engages; argpartition ties or wins
    below it on NAX GPUs (0.26-0.31 vs 0.26 ms per layer at one row)."""
    return _min_rows("OMLX_QWEN4_QSA_NATIVE_TOPK_MIN_ROWS", 8)


@functools.lru_cache(maxsize=None)
def _native_main_min_rows() -> int:
    """Query rows from which the native sparse GQA kernel engages; the gathered SDPA
    is faster below it on NAX GPUs (0.7 vs 1.6 ms per layer at verify width)."""
    return _min_rows("OMLX_QWEN4_QSA_NATIVE_MAIN_MIN_ROWS", 24)


def contiguous_causal_query_chunk(key_tokens: int) -> int:
    """Keep long-context score sheets bounded without tiny launch overhead."""

    if key_tokens <= 4096:
        return 32
    if key_tokens <= 16384:
        return 64
    return 128


_TOKEN_MAJOR_MIN_QUERIES = 32
_TOKEN_MAJOR_MAX_TOKENS = 131072


def _gather_kv_rows(kv: mx.array, indices: mx.array) -> mx.array:
    """Gather token rows of the stored ``(B, H, N, D)`` cache: ``(B, S)`` -> ``(B, H, S, D)``,
    ``(B, T, S)`` -> ``(B, T, H, S, D)``. Prefill-width gathers copy token-major below 128k."""

    per_query = indices.shape[1] if indices.ndim == 3 else 1
    if per_query >= _TOKEN_MAJOR_MIN_QUERIES and kv.shape[2] < _TOKEN_MAJOR_MAX_TOKENS:
        return _gather_kv_rows_token_major(kv, indices)
    return _gather_kv_rows_stored(kv, indices)


def _gather_kv_rows_stored(kv: mx.array, indices: mx.array) -> mx.array:
    """Take along the stored token axis: cheapest for few queries, no copy of the cache."""

    batch, heads, _, dim = kv.shape
    flat = indices.astype(mx.int32).reshape(batch, -1)
    if batch == 1:
        rows = mx.take(kv, flat[0], axis=2)
    else:
        rows = mx.stack([mx.take(kv[b], flat[b], axis=1) for b in range(batch)])
    if indices.ndim == 2:
        return rows
    per_query, width = indices.shape[1], indices.shape[2]
    return rows.reshape(batch, heads, per_query, width, dim).transpose(0, 2, 1, 3, 4)


def _gather_kv_rows_token_major(kv: mx.array, indices: mx.array) -> mx.array:
    """Copy the cache token-major and gather flat rows: cheaper for many queries per token."""

    batch, heads, tokens, dim = kv.shape
    rows = kv.transpose(0, 2, 1, 3).reshape(batch * tokens, heads, dim)
    flat = indices.astype(mx.int32).reshape(batch, -1)
    if batch > 1:
        flat = flat + (mx.arange(batch, dtype=mx.int32) * tokens)[:, None]
    gathered = rows[flat.reshape(-1)].reshape(*indices.shape, heads, dim)
    axes = (0, 2, 1, 3) if indices.ndim == 2 else (0, 1, 3, 2, 4)
    return gathered.transpose(*axes)


def _portable_indexer_scores(
    queries: mx.array,
    pooled_keys: mx.array,
    head_dim: int,
) -> mx.array:
    """Current float32 MLX QSA score reference."""

    batch, query_tokens, query_heads, _ = queries.shape
    # Flatten the query-token and index-head axes so MLX emits one FP32 GEMM
    # for the chunk instead of a broadcasted batch of tiny matmuls.  Each
    # output dot product and the following head reduction are unchanged.
    scores = (
        queries.astype(mx.float32).reshape(
            batch, query_tokens * query_heads, head_dim
        )
        @ pooled_keys.astype(mx.float32).swapaxes(-1, -2)
    ).reshape(batch, query_tokens, query_heads, pooled_keys.shape[1])
    return mx.sum(mx.maximum(scores, 0), axis=-2) / math.sqrt(head_dim)


def pool_completed_index_keys(
    index_keys: mx.array,
    index_position_ids: mx.array,
    *,
    compress_ratio: int,
    index_key_norm: IndexKeyNorm,
    apply_index_rope: IndexRoPE,
    start_block: int = 0,
    stop_block: int | None = None,
) -> mx.array:
    """Pool, normalize, and rotate a contiguous range of complete QSA blocks."""

    if index_keys.ndim != 3:
        raise ValueError("QSA raw index keys must have shape [B, S, D]")
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    complete_blocks = index_keys.shape[1] // compress_ratio
    if stop_block is None:
        stop_block = complete_blocks
    if not 0 <= start_block <= stop_block <= complete_blocks:
        raise ValueError(
            "QSA pooled block range must lie within the complete raw-key prefix"
        )
    if index_position_ids.ndim not in {2, 3} or (
        index_position_ids.shape[-1] != index_keys.shape[1]
    ):
        raise ValueError("QSA index positions do not match raw index keys")

    block_count = stop_block - start_block
    raw_start = start_block * compress_ratio
    raw_stop = stop_block * compress_ratio
    pooled = index_keys[:, raw_start:raw_stop].reshape(
        index_keys.shape[0],
        block_count,
        compress_ratio,
        index_keys.shape[-1],
    )
    pooled = mx.mean(pooled.astype(mx.float32), axis=-2).astype(index_keys.dtype)
    pooled = index_key_norm(pooled)
    block_starts = mx.arange(
        raw_start,
        raw_stop,
        compress_ratio,
        dtype=mx.int32,
    )
    pooled_positions = index_position_ids[..., block_starts]
    return apply_index_rope(pooled[:, None], pooled_positions)[:, 0]


def _native_indexer_scores(
    queries: mx.array,
    pooled_keys: mx.array,
    *,
    head_dim: int,
    compress_ratio: int,
    mask_q_offset: int,
) -> mx.array | None:
    """Use the narrow native M3 score ABI or fail closed to the MLX path."""

    global _NATIVE_QSA_SCORE_DISABLED, _NATIVE_QSA_SCORE_PROVEN
    if _NATIVE_QSA_SCORE_DISABLED:
        return None
    if (
        queries.ndim != 4
        or queries.shape[0] != 1
        or queries.shape[-2:] != (4, 128)
        or pooled_keys.ndim != 3
        or pooled_keys.shape[0] != 1
        or pooled_keys.shape[-1] != 128
        or queries.dtype != pooled_keys.dtype
        or queries.dtype not in {mx.float16, mx.bfloat16}
        or head_dim != 128
        or compress_ratio != 4
        or mask_q_offset < 0
    ):
        return None
    if queries.shape[1] < _native_score_min_rows():
        return None

    try:
        from omlx.custom_kernels.glm_moe_dsa import fast

        if not fast.is_native_available() or not fast.has_symbol(
            "qwen4_qsa_indexer_scores"
        ):
            _NATIVE_QSA_SCORE_DISABLED = True
            return None
        # The caller's [B,M,H,D] view transposes back to the GEMM-friendly
        # [B,H,M,D] ABI. The native wrapper only copies when the resulting
        # view is not row-contiguous (for example an offset query chunk).
        scores = fast.qwen4_qsa_indexer_scores(
            queries.transpose(0, 2, 1, 3),
            pooled_keys[:, None],
            mask_ratio=compress_ratio,
            mask_q_offset=mask_q_offset,
        )
        if not _NATIVE_QSA_SCORE_PROVEN:
            # MLX primitives are lazy, so a missing Metal pipeline would not
            # otherwise surface until the enclosing attention graph is
            # evaluated and can no longer fall back. Pay one process-wide
            # synchronization to prove the extension/pipeline pair.
            mx.eval(scores)
            _NATIVE_QSA_SCORE_PROVEN = True
        return scores
    except Exception:
        # A stale binary or rejected ABI should cost one attempt per process.
        # Shape misses were excluded above and remain eligible on later calls.
        _NATIVE_QSA_SCORE_DISABLED = True
        return None


def _native_topk_indices(scores: mx.array, topk: int) -> mx.array | None:
    """Use Qwen's exact FP32 top-k ABI or fail closed to argpartition."""

    global _NATIVE_QSA_TOPK_DISABLED, _NATIVE_QSA_TOPK_PROVEN
    if _NATIVE_QSA_TOPK_DISABLED:
        return None
    if (
        scores.ndim != 3
        or scores.shape[0] != 1
        or scores.shape[1] < 1
        or scores.shape[2] < topk
        or scores.dtype != mx.float32
        or topk != 512
    ):
        return None
    if scores.shape[1] < _native_topk_min_rows():
        return None
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast

        if not fast.is_native_available() or not fast.has_symbol(
            "qwen4_qsa_topk_indices"
        ):
            _NATIVE_QSA_TOPK_DISABLED = True
            return None
        indices = fast.qwen4_qsa_topk_indices(scores, topk=topk).astype(mx.int32)
        if not _NATIVE_QSA_TOPK_PROVEN:
            mx.eval(indices)
            _NATIVE_QSA_TOPK_PROVEN = True
        return indices
    except Exception:
        _NATIVE_QSA_TOPK_DISABLED = True
        return None


def _native_sparse_gqa_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    selected_blocks: mx.array,
    *,
    q_offset: int,
) -> mx.array | None:
    """Consume Qwen's selected rows directly in the exact native GQA kernel."""

    global _NATIVE_QSA_MAIN_DISABLED, _NATIVE_QSA_MAIN_PROVEN
    if _NATIVE_QSA_MAIN_DISABLED:
        return None
    if (
        queries.ndim != 4
        or queries.shape[0] != 1
        or queries.shape[1] != 24
        or queries.shape[-1] != 256
        or keys.ndim != 4
        or values.shape != keys.shape
        or keys.shape[0] != 1
        or keys.shape[1] != 2
        or keys.shape[-1] != 256
        or queries.dtype != keys.dtype
        or queries.dtype != values.dtype
        or queries.dtype not in {mx.float16, mx.bfloat16}
        or selected_blocks.ndim != 3
        or selected_blocks.shape != (1, queries.shape[2], 512)
        or q_offset < 0
        or q_offset + queries.shape[2] > keys.shape[2]
    ):
        return None
    if queries.shape[2] < _native_main_min_rows():
        return None
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast

        if not fast.is_native_available() or not fast.has_symbol(
            "qwen4_qsa_sparse_gqa_attention"
        ):
            _NATIVE_QSA_MAIN_DISABLED = True
            return None
        native_blocks = mx.contiguous(selected_blocks.astype(mx.uint32)[:, None])
        output = fast.qwen4_qsa_sparse_gqa_attention(
            queries,
            keys,
            values,
            native_blocks,
            queries.shape[-1] ** -0.5,
            q_offset,
            key_tile=64,
            dimension_tile=64,
        )
        if not _NATIVE_QSA_MAIN_PROVEN:
            # Prove the rebuilt extension and Metal pipeline before the lazy
            # graph advances cache state past a point where fallback is safe.
            mx.eval(output)
            _NATIVE_QSA_MAIN_PROVEN = True
        return output.transpose(0, 2, 1, 3)
    except Exception:
        _NATIVE_QSA_MAIN_DISABLED = True
        return None


def _decode_qsa_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
) -> mx.array:
    """Run exact unmasked singleton SDPA through the narrow native seam.

    The gathered decode caller has already applied QSA selection, so there is
    no causal or sparse mask left to interpret.  Only the decode_fast ABI's
    explicitly supported shape/dtype contract may use the native primitive;
    missing/stale extensions and every other shape fail closed to MLX SDPA.
    Inputs are made contiguous by the caller, avoiding a lazy layout failure
    after the model caches have advanced.
    """

    try:
        from omlx.custom_kernels.decode_fast import fast

        extension = getattr(fast, "_ext", None)
        supported = getattr(extension, "sdpa_decode_supported", None)
        if (
            bool(getattr(fast, "NATIVE_AVAILABLE", False))
            and supported is not None
            and bool(supported(queries, keys, values))
        ):
            return fast.sdpa_decode(
                queries,
                keys,
                values,
                scale,
                causal=False,
            )
    except Exception:
        # Capability probing is eager and happens before a native primitive is
        # added to the lazy graph, so this fallback cannot leave partial work.
        pass

    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
    )


def contiguous_causal_gathered_qsa_decode(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    index_queries: mx.array,
    pooled_index_keys: mx.array,
    *,
    num_query_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    indexer_head_dim: int,
    compress_ratio: int,
    token_budget: int,
) -> mx.array:
    """Run exact batch-one QSA decode over only the selected K/V rows.

    This is the singleton counterpart to :func:`contiguous_causal_gathered_qsa`.
    The query is the final visible token, so every completed compressed block
    is causal.  QSA chooses ``token_budget / compress_ratio`` complete blocks;
    their token rows are gathered in chronological order and the zero-to-three
    incomplete tail rows are appended.  Main attention therefore remains
    bounded by ``token_budget + compress_ratio - 1`` instead of scanning a
    dense full-length mask.
    """

    if queries.ndim != 4 or queries.shape[:3] != (1, num_query_heads, 1):
        raise ValueError("gathered QSA decode requires [1, H, 1, D] queries")
    if queries.shape[-1] != head_dim:
        raise ValueError("QSA decode queries do not match the configured head dim")
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("QSA decode K/V must be matching rank-four arrays")
    if keys.shape[0] != 1 or keys.shape[1] != num_key_value_heads:
        raise ValueError("QSA decode K/V do not match the configured head count")
    if keys.shape[-1] != head_dim or keys.dtype != queries.dtype:
        raise ValueError("QSA decode K/V dtype or head dim does not match queries")
    if values.dtype != queries.dtype:
        raise ValueError("QSA decode values must match the query dtype")
    if (
        index_queries.ndim != 4
        or index_queries.shape[0] != 1
        or index_queries.shape[1] != 1
        or index_queries.shape[-1] != indexer_head_dim
    ):
        raise ValueError("QSA decode index queries must have shape [1, 1, H, D]")
    if compress_ratio <= 0 or token_budget <= 0 or token_budget % compress_ratio:
        raise ValueError("QSA decode token budget must contain complete blocks")
    if num_query_heads % num_key_value_heads:
        raise ValueError("QSA decode query heads must divide over K/V heads")

    key_tokens = int(keys.shape[2])
    max_blocks = key_tokens // compress_ratio
    block_budget = token_budget // compress_ratio
    if max_blocks <= block_budget:
        raise ValueError("gathered QSA decode requires a sparse block crossover")
    if pooled_index_keys.shape != (1, max_blocks, indexer_head_dim):
        raise ValueError("QSA decode pooled index-key cache has the wrong shape")

    block_scores = _native_indexer_scores(
        index_queries,
        pooled_index_keys,
        head_dim=indexer_head_dim,
        compress_ratio=compress_ratio,
        mask_q_offset=key_tokens - 1,
    )
    if block_scores is None:
        block_scores = _portable_indexer_scores(
            index_queries,
            pooled_index_keys,
            indexer_head_dim,
        )

    selected_blocks = _native_topk_indices(block_scores, block_budget)
    if selected_blocks is None:
        selected_blocks = mx.argpartition(
            block_scores,
            kth=-block_budget,
            axis=-1,
        )[..., -block_budget:].astype(mx.int32)
    # Argpartition/native radix order is not chronological.  Sorting the
    # selected set preserves the official key order for deterministic SDPA.
    selected_blocks = mx.sort(selected_blocks, axis=-1)
    selected_tokens = (
        selected_blocks[..., None] * compress_ratio
        + mx.arange(compress_ratio, dtype=mx.int32)
    ).reshape(1, block_budget * compress_ratio)

    complete_key_len = max_blocks * compress_ratio
    if complete_key_len < key_tokens:
        tail = mx.arange(complete_key_len, key_tokens, dtype=mx.int32)[None]
        selected_tokens = mx.concatenate((selected_tokens, tail), axis=-1)

    selected_keys = _gather_kv_rows(keys, selected_tokens)
    selected_values = _gather_kv_rows(values, selected_tokens)
    output = _decode_qsa_sdpa(
        queries,
        selected_keys,
        selected_values,
        head_dim**-0.5,
    )
    return output.transpose(0, 2, 1, 3)


def contiguous_causal_gathered_qsa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    index_queries: mx.array,
    index_keys: mx.array,
    index_position_ids: mx.array,
    *,
    num_query_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    indexer_head_dim: int,
    compress_ratio: int,
    token_budget: int,
    index_key_norm: IndexKeyNorm,
    apply_index_rope: IndexRoPE,
    pooled_index_keys: mx.array | None = None,
    query_chunk: int | None = None,
) -> mx.array:
    """Run exact QSA over gathered K/V for one contiguous causal prompt.

    ``queries`` and ``keys`` must already carry their main-attention RoPE.
    ``index_queries`` must likewise be normalized and RoPE-rotated.  Raw
    indexer keys remain unrotated because Qwen pools each complete micro-block
    before applying its checkpoint k-norm and the block-start RoPE.
    """

    if queries.ndim != 4 or queries.shape[0] != 1 or queries.shape[2] <= 1:
        raise ValueError(
            "gathered QSA requires rank-four batch-one multi-token queries"
        )
    batch, actual_query_heads, query_tokens, actual_head_dim = queries.shape
    if actual_query_heads != num_query_heads or actual_head_dim != head_dim:
        raise ValueError("QSA queries do not match the configured geometry")
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("QSA keys and values must be matching rank-four arrays")
    if keys.shape[0] != batch or keys.shape[1:] != (
        num_key_value_heads,
        keys.shape[2],
        head_dim,
    ):
        raise ValueError("QSA K/V do not match the configured geometry")
    key_tokens = keys.shape[2]
    if query_tokens > key_tokens:
        raise ValueError("QSA query length cannot exceed cached key length")
    if index_queries.ndim != 4 or index_queries.shape[:2] != (
        batch,
        query_tokens,
    ):
        raise ValueError("QSA index queries do not match the current prompt")
    if index_queries.shape[-1] != indexer_head_dim:
        raise ValueError("QSA index queries have the wrong head dimension")
    if index_keys.shape != (batch, key_tokens, indexer_head_dim):
        raise ValueError("QSA raw index keys do not match cached K/V")
    if (
        index_position_ids.ndim not in {2, 3}
        or index_position_ids.shape[-1] != key_tokens
    ):
        raise ValueError("QSA index positions do not match cached K/V")
    if compress_ratio <= 0 or token_budget <= 0 or token_budget % compress_ratio:
        raise ValueError("QSA token budget must contain complete micro-blocks")
    if num_query_heads % num_key_value_heads:
        raise ValueError("QSA query heads must divide evenly over K/V heads")

    if query_chunk is None:
        query_chunk = contiguous_causal_query_chunk(key_tokens)
        # The direct-index main-attention kernel carries no per-query gathered
        # K/V tensor, so a 256-row score tile stays comfortably bounded and
        # halves Python/Metal dispatch overhead. Preserve the smaller portable
        # tiles whenever the exact production ABI is absent.
        if (
            queries.shape[1:] == (24, query_tokens, 256)
            and keys.shape[1] == 2
            and queries.dtype in {mx.float16, mx.bfloat16}
            and not _NATIVE_QSA_MAIN_DISABLED
        ):
            try:
                from omlx.custom_kernels.glm_moe_dsa import fast

                if fast.is_native_available() and fast.has_symbol(
                    "qwen4_qsa_sparse_gqa_attention"
                ):
                    query_chunk = max(query_chunk, 256)
            except Exception:
                pass
    if query_chunk <= 0:
        raise ValueError("QSA query chunk must be positive")

    ratio = compress_ratio
    max_blocks = key_tokens // ratio
    block_budget = token_budget // ratio
    query_start = key_tokens - query_tokens

    # A contiguous prompt shares the same block bank for every query.  The
    # caller can provide its cache of completed blocks; standalone users still
    # get the exact one-shot construction.
    if max_blocks:
        if pooled_index_keys is None:
            pooled = pool_completed_index_keys(
                index_keys,
                index_position_ids,
                compress_ratio=ratio,
                index_key_norm=index_key_norm,
                apply_index_rope=apply_index_rope,
            )
        else:
            if pooled_index_keys.shape != (batch, max_blocks, indexer_head_dim):
                raise ValueError("QSA pooled index-key cache has the wrong shape")
            pooled = pooled_index_keys
    else:
        if pooled_index_keys is not None and pooled_index_keys.shape != (
            batch,
            0,
            indexer_head_dim,
        ):
            raise ValueError("QSA pooled index-key cache has the wrong shape")
        pooled = None

    outputs: list[mx.array] = []
    groups = num_query_heads // num_key_value_heads
    for start in range(0, query_tokens, query_chunk):
        stop = min(start + query_chunk, query_tokens)
        chunk_tokens = stop - start
        absolute_queries = query_start + mx.arange(start, stop, dtype=mx.int32)
        visible_counts = mx.broadcast_to(
            (absolute_queries + 1)[None], (batch, chunk_tokens)
        )
        complete_counts = visible_counts // ratio

        if max_blocks:
            chunk_index_queries = index_queries[:, start:stop]
            block_scores = _native_indexer_scores(
                chunk_index_queries,
                pooled,
                head_dim=indexer_head_dim,
                compress_ratio=ratio,
                mask_q_offset=query_start + start,
            )
            if block_scores is None:
                block_scores = _portable_indexer_scores(
                    chunk_index_queries,
                    pooled,
                    indexer_head_dim,
                )
                valid_blocks = (
                    mx.arange(max_blocks)[None, None, :]
                    < complete_counts[..., None]
                )
                block_scores = mx.where(
                    valid_blocks,
                    block_scores,
                    mx.finfo(block_scores.dtype).min,
                )

            selected_width = min(max_blocks, block_budget)
            canonical = mx.broadcast_to(
                mx.arange(selected_width, dtype=mx.int32)[None, None],
                (batch, chunk_tokens, selected_width),
            )
            if max_blocks > block_budget:
                ranked = _native_topk_indices(block_scores, block_budget)
                if ranked is None:
                    ranked = mx.argpartition(
                        block_scores,
                        kth=-block_budget,
                        axis=-1,
                    )[..., -block_budget:].astype(mx.int32)
                selected_block_rows = mx.where(
                    (complete_counts <= block_budget)[..., None],
                    canonical,
                    ranked,
                )
            else:
                selected_block_rows = canonical

            # The top-k set is unordered. Restore checkpoint/dense-mask token
            # order before either the portable gathered SDPA or the direct
            # native kernel performs its FP32 online-softmax reduction.
            selected_block_rows = mx.sort(selected_block_rows, axis=-1)

            selected_count = mx.minimum(complete_counts, block_budget)

            native_output = _native_sparse_gqa_attention(
                queries[:, :, start:stop],
                keys,
                values,
                selected_block_rows,
                q_offset=query_start + start,
            )
            if native_output is not None:
                outputs.append(native_output)
                continue

            selected_indices = (
                selected_block_rows[..., None] * ratio
                + mx.arange(ratio, dtype=mx.int32)
            ).reshape(batch, chunk_tokens, selected_width * ratio)
            selected_valid = mx.broadcast_to(
                mx.arange(selected_width)[None, None, :, None]
                < selected_count[..., None, None],
                (batch, chunk_tokens, selected_width, ratio),
            ).reshape(batch, chunk_tokens, selected_width * ratio)
        else:
            selected_indices = mx.zeros(
                (batch, chunk_tokens, 0), dtype=mx.int32
            )
            selected_valid = mx.zeros(
                (batch, chunk_tokens, 0), dtype=mx.bool_
            )

        # The zero-to-three visible tokens after the final complete block are
        # always retained by the published QSA contract.
        tail_width = ratio - 1
        tail = complete_counts[..., None] * ratio + mx.arange(
            tail_width, dtype=mx.int32
        )
        tail_valid = tail < visible_counts[..., None]
        selected_indices = mx.concatenate((selected_indices, tail), axis=-1)
        selected_valid = mx.concatenate((selected_valid, tail_valid), axis=-1)

        safe_selected = mx.where(selected_valid, selected_indices, 0).astype(mx.int32)

        selected_keys = _gather_kv_rows(keys, safe_selected)
        selected_values = _gather_kv_rows(values, safe_selected)

        chunk_queries = queries[:, :, start:stop].transpose(0, 2, 1, 3)
        grouped_queries = chunk_queries.reshape(
            batch,
            chunk_tokens,
            num_key_value_heads,
            groups,
            head_dim,
        )
        scores = (
            grouped_queries.astype(mx.float32)
            @ selected_keys.astype(mx.float32).swapaxes(-1, -2)
        ) / math.sqrt(head_dim)
        scores = mx.where(
            selected_valid[:, :, None, None, :],
            scores,
            mx.finfo(scores.dtype).min,
        )
        probabilities = mx.softmax(scores, axis=-1).astype(chunk_queries.dtype)
        output = probabilities @ selected_values
        outputs.append(
            output.reshape(batch, chunk_tokens, num_query_heads, head_dim)
        )

    return mx.concatenate(outputs, axis=1)


__all__ = [
    "contiguous_causal_gathered_qsa",
    "contiguous_causal_gathered_qsa_decode",
    "contiguous_causal_query_chunk",
    "pool_completed_index_keys",
]
