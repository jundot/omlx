# SPDX-License-Identifier: Apache-2.0
"""Patch qwen3_5 GatedDeltaNet to mirror mlx-lm fixes that mlx-vlm lacks.

mlx-lm's qwen3_5.py GatedDeltaNet:
- Calls ``cache.advance(S)`` after writing ``cache[1] = state`` so
  ArraysCache.left_padding / lengths get decremented between prefill
  chunks (correct SSM mask under batched, varying-length prompts).
- Wraps the conv state assignment in ``mx.contiguous`` so the cached
  slice has a sane memory layout for the next decode step.

mlx-vlm e41cd25's qwen3_5/language.py and qwen3_5_moe (which subclasses
the same Qwen3_5GatedDeltaNet) miss both. Once omlx routes decode
through mlx-vlm directly, those gaps surface as wrong attention masks
and degraded output for Qwen3.5 / Qwen3.6 under concurrent requests
with different prompt lengths.

This patch wraps the GatedDeltaNet ``__call__`` of both libraries:
- always calls ``cache.advance(S)`` post-forward (no-op if upstream
  already does — but skip if upstream source already contains
  ``.advance(`` to avoid double-advance)
- forces ``cache[0] = mx.contiguous(cache[0])`` after the original
  forward so the conv state matches mlx-lm's layout
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)


# Track whether the class-level patch has been applied
_patched_classes: set[int] = set()


def _patch_class(cls: Any, label: str) -> bool:
    """Monkey-patch a GatedDeltaNet-like class once.

    Returns True if patched (or already patched), False on no-op (e.g.
    upstream already has advance() AND a contiguous cache write).
    """
    if id(cls) in _patched_classes:
        return True

    # Forward compatibility: skip advance-wrapping if upstream already
    # calls cache.advance() inside __call__. We still wrap to apply the
    # contiguous fix; the wrapped call only adds advance() when the
    # original lacks it.
    try:
        source = inspect.getsource(cls.__call__)
        upstream_has_advance = ".advance(" in source
        upstream_has_contiguous = "mx.contiguous" in source
    except (OSError, TypeError):
        upstream_has_advance = False
        upstream_has_contiguous = False

    if upstream_has_advance and upstream_has_contiguous:
        logger.debug(
            f"GatedDeltaNet patch: {label} already has advance() and contiguous, skipping"
        )
        _patched_classes.add(id(cls))
        return False

    original_call = cls.__call__

    # Locate the ``cache`` parameter so we can extract it whether the caller
    # passes it positionally or as a keyword. mlx-vlm's qwen3_5_moe layer
    # passes it positionally (``self.linear_attn(h, mask, cache, ...)``)
    # while mlx-lm passes it as a keyword.
    cache_param_idx: Any = None
    try:
        params = list(inspect.signature(original_call).parameters)
        if "cache" in params:
            # ``self`` is index 0 in the bound signature when we forward
            # *args verbatim from the patched method.
            cache_param_idx = params.index("cache")
    except (ValueError, TypeError):
        cache_param_idx = None

    def patched_call(*args, **kwargs):
        # Original call always runs verbatim. Any failure in our
        # post-processing must NOT break the model's forward pass —
        # log a warning and return the unmodified result instead.
        result = original_call(*args, **kwargs)

        try:
            cache = kwargs.get("cache")
            if cache is None and cache_param_idx is not None and len(args) > cache_param_idx:
                cache = args[cache_param_idx]
            if cache is None:
                return result

            # Force conv state into a contiguous layout. mx.contiguous
            # is a no-op when the array is already contiguous, so this
            # is safe even after upstream adopts the same fix.
            cs = cache[0]
            if cs is not None and getattr(cs, "size", 0) > 0:
                cache[0] = mx.contiguous(cs)

            if not upstream_has_advance:
                # ``inputs`` is always the first positional arg after ``self``.
                inputs = args[1] if len(args) > 1 else kwargs.get("inputs")
                if inputs is not None:
                    cache.advance(inputs.shape[1])
        except Exception as e:
            logger.warning(
                f"GatedDeltaNet patch post-fix failed for {label}: {e}. "
                "Continuing with original forward result — model may regress on "
                "concurrent batched + varying-length prompts."
            )
        return result

    cls.__call__ = patched_call
    _patched_classes.add(id(cls))
    logger.info(f"GatedDeltaNet patch applied: {label}")
    return True


def apply_gated_delta_advance_patch(model: Any = None) -> bool:
    """Patch every importable GatedDeltaNet variant.

    The ``model`` argument is accepted for backward compatibility but
    is not used: the patch is applied at the class level, so a single
    call is enough regardless of which model instance is loaded.

    Returns True if at least one class was (or had already been)
    patched, False if neither library exposed a target.
    """
    any_patched = False

    # mlx-lm path (current upstream already includes both fixes, so
    # this is mostly a backward-compat shim for older mlx-lm releases)
    try:
        from mlx_lm.models.qwen3_5 import GatedDeltaNet as _LMGdn

        _patch_class(_LMGdn, "mlx_lm.models.qwen3_5.GatedDeltaNet")
        any_patched = True
    except ImportError:
        logger.debug("mlx_lm.models.qwen3_5 not importable")

    # mlx-vlm path (qwen3_5_moe reuses Qwen3_5GatedDeltaNet, so one
    # patch covers both qwen3_5 and qwen3_5_moe / Qwen3.6)
    try:
        from mlx_vlm.models.qwen3_5.language import (
            Qwen3_5GatedDeltaNet as _VLMGdn,
        )

        _patch_class(_VLMGdn, "mlx_vlm.models.qwen3_5.language.Qwen3_5GatedDeltaNet")
        any_patched = True
    except ImportError:
        logger.debug("mlx_vlm.models.qwen3_5.language not importable")

    return any_patched
