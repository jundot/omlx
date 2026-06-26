# SPDX-License-Identifier: Apache-2.0
"""Patch MiniMax M3 sparse attention for batched left-padded caches."""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_omlx_minimax_m3_sparse_left_padding_patch"
_LEFT_PADDING_ATTR = "_omlx_minimax_m3_sparse_left_padding"


def _storage_q_positions(
    q_positions: Any,
    left_padding: Any,
    batch_size: int,
    seq_len: int,
):
    """Map semantic query positions to batched-cache storage positions."""
    if q_positions is None or left_padding is None:
        return q_positions
    if not isinstance(left_padding, mx.array):
        return q_positions

    padding = left_padding.reshape(-1)[:batch_size]
    if padding.size == 0:
        return q_positions

    if not isinstance(q_positions, mx.array):
        return q_positions

    padding = padding.astype(q_positions.dtype)
    if q_positions.ndim == 0:
        return q_positions + padding[0]
    if q_positions.ndim == 1:
        if q_positions.shape[0] == batch_size and seq_len == 1:
            return q_positions + padding
        return q_positions + padding[0]
    if q_positions.ndim == 2:
        return q_positions + padding[:, None]
    if q_positions.ndim == 3:
        return q_positions + padding[None, :, None]
    return q_positions


def _adjust_sparse_positions(self, q_positions, batch_size: int, seq_len: int):
    return _storage_q_positions(
        q_positions,
        getattr(self, _LEFT_PADDING_ATTR, None),
        batch_size,
        seq_len,
    )


def _make_patched_call(original_call):
    def patched_call(self, x, mask=None, cache=None, position_ids=None):
        previous = getattr(self, _LEFT_PADDING_ATTR, None)
        installed = False
        if position_ids is not None and cache is not None:
            left_padding = getattr(cache, "left_padding", None)
            if isinstance(left_padding, mx.array):
                setattr(self, _LEFT_PADDING_ATTR, left_padding)
                installed = True
        try:
            return original_call(
                self, x, mask=mask, cache=cache, position_ids=position_ids
            )
        finally:
            if installed:
                if previous is None:
                    try:
                        delattr(self, _LEFT_PADDING_ATTR)
                    except AttributeError:
                        pass
                else:
                    setattr(self, _LEFT_PADDING_ATTR, previous)

    setattr(patched_call, _PATCH_MARKER, True)
    return patched_call


def _make_patched_build_sparse_mask(original):
    def patched_build_sparse_mask(
        self,
        idx_queries,
        idx_keys,
        q_start,
        mask=None,
        return_block_indices=False,
        build_token_mask=True,
        q_positions=None,
    ):
        q_positions = _adjust_sparse_positions(
            self, q_positions, idx_queries.shape[0], idx_queries.shape[2]
        )
        return original(
            self,
            idx_queries,
            idx_keys,
            q_start,
            mask,
            return_block_indices,
            build_token_mask,
            q_positions,
        )

    setattr(patched_build_sparse_mask, _PATCH_MARKER, True)
    return patched_build_sparse_mask


def _make_patched_build_sparse_decode_indices(original):
    def patched_build_sparse_decode_indices(
        self,
        idx_queries,
        idx_keys,
        q_start,
        q_positions=None,
    ):
        q_positions = _adjust_sparse_positions(
            self, q_positions, idx_queries.shape[0], idx_queries.shape[2]
        )
        return original(self, idx_queries, idx_keys, q_start, q_positions)

    setattr(patched_build_sparse_decode_indices, _PATCH_MARKER, True)
    return patched_build_sparse_decode_indices


def _make_patched_sparse_decode_attention(original):
    def patched_sparse_decode_attention(
        self,
        queries,
        keys,
        values,
        topk_idx,
        topk_valid,
        q_start,
        *,
        topk_all_valid=False,
        q_positions=None,
    ):
        q_positions = _adjust_sparse_positions(
            self, q_positions, queries.shape[0], queries.shape[2]
        )
        return original(
            self,
            queries,
            keys,
            values,
            topk_idx,
            topk_valid,
            q_start,
            topk_all_valid=topk_all_valid,
            q_positions=q_positions,
        )

    setattr(patched_sparse_decode_attention, _PATCH_MARKER, True)
    return patched_sparse_decode_attention


def apply_minimax_m3_sparse_attention_patch() -> bool:
    """Patch mlx-vlm MiniMax M3 sparse attention once.

    MiniMax M3 keeps sparse-index keys in the same left-padded storage layout as
    batched KV cache, but sparse query positions are semantic token positions.
    With unequal prompt lengths, sparse block selection compares semantic query
    positions against storage key positions and can mask valid current-row keys.
    """
    try:
        from .mlx_vlm_minimax_m3_compat import (
            apply_mlx_vlm_minimax_m3_compat_patch,
        )

        apply_mlx_vlm_minimax_m3_compat_patch()
    except Exception as e:  # noqa: BLE001
        logger.debug("minimax_m3_sparse_attention: compat patch failed: %s", e)

    try:
        from mlx_vlm.models.minimax_m3_vl import language as minimax_language
    except ImportError:
        logger.debug("minimax_m3_sparse_attention: mlx-vlm MiniMax M3 not available")
        return False

    attention_cls = getattr(minimax_language, "MiniMaxAttention", None)
    if attention_cls is None:
        logger.debug("minimax_m3_sparse_attention: MiniMaxAttention class not found")
        return False

    current_call = attention_cls.__dict__.get("__call__")
    if getattr(current_call, _PATCH_MARKER, False):
        return False

    attention_cls.__call__ = _make_patched_call(current_call)
    attention_cls._build_sparse_mask = _make_patched_build_sparse_mask(
        attention_cls.__dict__["_build_sparse_mask"]
    )
    attention_cls._build_sparse_decode_indices = (
        _make_patched_build_sparse_decode_indices(
            attention_cls.__dict__["_build_sparse_decode_indices"]
        )
    )
    attention_cls._sparse_decode_attention = _make_patched_sparse_decode_attention(
        attention_cls.__dict__["_sparse_decode_attention"]
    )
    logger.info("MiniMax M3 sparse attention left-padding patch applied")
    return True


__all__ = ["apply_minimax_m3_sparse_attention_patch", "_storage_q_positions"]
