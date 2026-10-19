# SPDX-License-Identifier: Apache-2.0
"""Preserve both padding bounds in mlx-vlm's Qwen GDN mask."""

from functools import wraps

import mlx.core as mx


def apply_qwen35_ssm_mask_patch() -> bool:
    from mlx_vlm.models.qwen3_5 import language

    original = language._create_qwen3_5_ssm_mask
    if getattr(original, "_omlx_padding_bounds", False):
        return False

    @wraps(original)
    def create_ssm_mask(h, cache):
        if getattr(cache, "lengths", None) is None:
            return original(h, cache)

        left_info = language._qwen3_5_left_padding_info(cache)
        lengths_min = language._qwen3_5_lengths_info(cache)
        if (
            lengths_min is not None
            and lengths_min >= h.shape[1]
            and (left_info is None or left_info[1] <= 0)
        ):
            return None

        # ArraysCache.make_mask selects one bound when both are present.
        positions = mx.arange(h.shape[1])
        mask = positions < cache.lengths[:, None]
        if cache.left_padding is not None:
            mask = mask & (positions >= cache.left_padding[:, None])
        return mask

    create_ssm_mask._omlx_padding_bounds = True
    language._create_qwen3_5_ssm_mask = create_ssm_mask
    return True
