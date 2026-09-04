# SPDX-License-Identifier: Apache-2.0
"""Storage-only delta encoding for cumulative PoolingCache snapshots."""

from __future__ import annotations

import logging
from typing import Any

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)

POOLING_CACHE_DELTA_CLASS = "PoolingCacheDelta"
POOLING_CACHE_DELTA_FORMAT_VERSION = "4"


def compact_pooling_cache_snapshot(
    extracted: list[dict[str, Any]],
    token_count: int,
    block_size: int,
) -> list[dict[str, Any]]:
    """Replace cumulative PoolingCache tensors with the current block delta.

    PoolingCache is non-sliceable because its buffer and gate are required for
    continued decoding, but its ``pooled`` tensor is append-only. Boundary
    snapshots therefore only need the newly appended pooled rows. The absolute
    row range is recorded so restore can reject incomplete or out-of-order
    delta chains instead of silently constructing a corrupt cache.

    The input is modified in place and returned for convenience. If an
    unfamiliar state shape or length is encountered, that sub-cache remains a
    legacy full snapshot.
    """
    if token_count <= 0 or block_size <= 0:
        return extracted

    for layer in extracted:
        if layer.get("class_name") != "CacheList":
            continue

        sub_states = layer.get("state")
        class_names = layer.get("sub_class_names")
        meta_state = layer.get("meta_state")
        if (
            not class_names
            and isinstance(meta_state, (list, tuple))
            and len(meta_state) >= 1
        ):
            class_names = meta_state[0]
        sub_meta_states = (
            meta_state[1]
            if isinstance(meta_state, (list, tuple)) and len(meta_state) >= 2
            else []
        )
        if not isinstance(sub_states, (list, tuple)) or not isinstance(
            class_names, (list, tuple)
        ):
            continue

        delta_ranges: dict[str, list[int]] = {}
        compacted_states = list(sub_states)
        for sub_idx, (class_name, state) in enumerate(
            zip(class_names, sub_states, strict=False)
        ):
            if class_name != "PoolingCache" or not isinstance(state, (list, tuple)):
                continue
            if len(state) not in (3, 5) or sub_idx >= len(sub_meta_states):
                continue

            try:
                ratio = int(sub_meta_states[sub_idx])
                if ratio <= 0:
                    continue
                pooled = state[2]
                pooled_length = int(pooled.shape[1])
                expected_end = token_count // ratio
                expected_start = max(0, token_count - block_size) // ratio
            except (AttributeError, TypeError, ValueError, IndexError):
                continue

            if pooled_length != expected_end:
                logger.debug(
                    "Keeping full PoolingCache snapshot at layer %s sub-cache "
                    "%d: pooled=%s expected=%s ratio=%s",
                    layer.get("layer_idx"),
                    sub_idx,
                    pooled_length,
                    expected_end,
                    ratio,
                )
                continue

            # mx.contiguous, not a bare slice: a retained view keeps its
            # whole parent buffer alive, so an in-memory snapshot would pin
            # the entire cumulative pooled tensor instead of this block's
            # delta -- the compaction saves nothing and total retention goes
            # quadratic in context length. The SSD path escapes this only
            # because serialization copies the bytes out. (PoolingCache
            # .extract() copies for the same reason, cache_extras.py:930.)
            delta = pooled[:, expected_start:expected_end]
            if HAS_MLX and hasattr(mx, "contiguous"):
                delta = mx.contiguous(delta)
            compacted_states[sub_idx] = (
                state[0],
                state[1],
                delta,
                *state[3:],
            )
            delta_ranges[str(sub_idx)] = [expected_start, expected_end]

        if delta_ranges:
            layer["state"] = compacted_states
            layer["pooling_delta_ranges"] = delta_ranges

    return extracted
