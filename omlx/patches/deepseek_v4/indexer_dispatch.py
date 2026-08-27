# SPDX-License-Identifier: Apache-2.0
"""Shared DeepSeek V4 native-indexer eligibility and runtime state."""

from __future__ import annotations

import os

_NATIVE_INDEXER_DISABLED = False
_NATIVE_INDEXER_PROVEN = False


def _operator_disabled() -> bool:
    return os.environ.get("OMLX_DSV4_NATIVE_INDEXER", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }


def disable_native_indexer() -> None:
    """Disable the native indexer after a runtime kernel failure."""
    global _NATIVE_INDEXER_DISABLED
    _NATIVE_INDEXER_DISABLED = True


def mark_native_indexer_proven() -> None:
    """Record that both kernels completed one evaluated runtime probe."""

    global _NATIVE_INDEXER_PROVEN
    _NATIVE_INDEXER_PROVEN = True


def native_indexer_proven() -> bool:
    """Whether low-memory admission may rely on the native implementation.

    Symbol presence is enough to *attempt* the native route, but not enough to
    admit a prompt that only fits when its score/top-k buffers stay fused.  A
    stale metallib or first command-buffer failure can still demote the model
    to its much larger MLX fallback.  The pre-load probe marks this only after
    evaluating both kernels successfully.
    """

    return _NATIVE_INDEXER_PROVEN and native_indexer_available()


def native_indexer_disabled() -> bool:
    """Return whether a runtime kernel failure disabled the native path."""
    return _NATIVE_INDEXER_DISABLED or _operator_disabled()


def native_indexer_available() -> bool:
    """Return whether both native indexer kernels are usable in this process."""
    if _NATIVE_INDEXER_DISABLED or _operator_disabled():
        return False
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

        return glm_fast.has_symbol("dsa_indexer_scores") and glm_fast.has_symbol(
            "dsa_topk_indices"
        )
    except Exception:
        return False


def native_indexer_eligible(
    *,
    query_tokens: int,
    pooled_tokens: int,
    n_heads: int,
    head_dim: int,
    index_topk: int,
    dtype_supported: bool,
) -> bool:
    """Match the DeepSeek V4 model's native prefill-indexer dispatch gate.

    The Metal kernel handles partial 64x64 M/N tiles with bounded loads and
    stores. Query and pooled lengths therefore do not need scheduler-visible
    alignment; only the model, dtype, top-k, and runtime-kernel constraints
    remain.
    """
    return native_indexer_available() and native_indexer_shape_eligible(
        query_tokens=query_tokens,
        pooled_tokens=pooled_tokens,
        n_heads=n_heads,
        head_dim=head_dim,
        index_topk=index_topk,
        dtype_supported=dtype_supported,
    )


def native_indexer_memory_safe_eligible(
    *,
    query_tokens: int,
    pooled_tokens: int,
    n_heads: int,
    head_dim: int,
    index_topk: int,
    dtype_supported: bool,
) -> bool:
    """Eligibility for a low-memory estimate, gated on evaluated proof."""

    return native_indexer_proven() and native_indexer_shape_eligible(
        query_tokens=query_tokens,
        pooled_tokens=pooled_tokens,
        n_heads=n_heads,
        head_dim=head_dim,
        index_topk=index_topk,
        dtype_supported=dtype_supported,
    )


def native_indexer_shape_eligible(
    *,
    query_tokens: int,
    pooled_tokens: int,
    n_heads: int,
    head_dim: int,
    index_topk: int,
    dtype_supported: bool,
) -> bool:
    """Return whether shapes and dtype are supported, ignoring availability."""
    return (
        dtype_supported
        # Keep single-token decode on its row-wise fp32 path. The native
        # 64-row score tile is valuable for prefill tails but wasteful for M=1.
        and query_tokens > 1
        and pooled_tokens > index_topk
        and n_heads in (32, 64)
        and head_dim == 128
        and index_topk in (512, 2048)
    )
