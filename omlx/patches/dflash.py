# SPDX-License-Identifier: Apache-2.0
"""DFlash speculative decoding integration for omlx.

Wraps dflash-mlx's block-diffusion speculative decoding into the omlx
scheduler's generation loop. Provides:
  - load_dflash_draft(): resolve and load a DFlash draft model
  - install_dflash_hooks(): install target-model speculative hooks

The actual draft-verify-accept cycle runs inside dflash-mlx's runtime via
the hooks installed on the target model; omlx does not drive it per-step.

All dflash-mlx imports are guarded so omlx works without it installed.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import guard: dflash-mlx is optional
# ---------------------------------------------------------------------------

_HAS_DFLASH = False
try:
    from dflash_mlx.generate import resolve_optional_draft_ref
    from dflash_mlx.runtime import (
        configure_full_attention_split,
        load_draft_bundle,
        pack_target_model_weights,
    )

    # detect_target_family may not be exported in older dflash-mlx releases
    try:
        from dflash_mlx.runtime import detect_target_family
    except ImportError:
        detect_target_family = None

    _HAS_DFLASH = True
except ImportError:
    detect_target_family = None


# ---------------------------------------------------------------------------
# Local helpers (for API compatibility across dflash-mlx versions)
# ---------------------------------------------------------------------------


def _detect_target_family(target_model: Any) -> str:
    """Detect whether target model uses hybrid GDN or pure attention.

    Reimplements dflash_mlx.runtime.detect_target_family for older releases.
    """
    if detect_target_family is not None:
        return detect_target_family(target_model)

    # Fallback implementation
    try:
        # Navigate to the inner model.layers
        wrapper = target_model
        if hasattr(wrapper, "model"):
            wrapper = wrapper.model
        if hasattr(wrapper, "language_model"):
            wrapper = wrapper.language_model
        if hasattr(wrapper, "model"):
            inner = wrapper.model
        else:
            inner = wrapper

        has_linear = any(
            hasattr(layer, "linear_attn") or getattr(layer, "is_linear", False)
            for layer in inner.layers
        )
        return "hybrid_gdn" if has_linear else "pure_attention"
    except Exception:
        return "pure_attention"


# ---------------------------------------------------------------------------
# Draft model loading
# ---------------------------------------------------------------------------


def load_dflash_draft(
    target_model_ref: str,
    draft_ref_override: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Resolve and load a DFlash draft model for the given target.

    Args:
        target_model_ref: HuggingFace model name or path of the target.
        draft_ref_override: Optional explicit draft model reference.

    Returns:
        (draft_model, resolved_ref) or (None, None) if no draft found.
    """
    if not _HAS_DFLASH:
        logger.warning("DFlash: dflash-mlx package not installed")
        return None, None

    resolved_ref = resolve_optional_draft_ref(target_model_ref, draft_ref_override)
    if not resolved_ref:
        logger.debug(f"DFlash: no draft registered for {target_model_ref}")
        return None, None

    try:
        draft_model, _ = load_draft_bundle(resolved_ref, lazy=True)
        return draft_model, resolved_ref
    except Exception as e:
        logger.error(f"DFlash: failed to load draft model {resolved_ref}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Target model hook installation
# ---------------------------------------------------------------------------


def install_dflash_hooks(target_model: Any) -> bool:
    """Install speculative decoding hooks on the target model.

    For hybrid GDN models (Qwen3.5), installs:
      - Speculative linear cache hook (RecurrentRollbackCache)
      - Split full-attention hook (JIT SDPA 2-pass for long-context verify)
      - Exact small projection padding
      - Weight packing (gate_up, qkv fused projections)

    For pure-attention models (Qwen3), hooks are a no-op (no GDN layers).

    Args:
        target_model: A loaded mlx-lm model instance.

    Returns:
        True if hooks were installed, False if not applicable.
    """
    if not _HAS_DFLASH:
        return False

    try:
        family = _detect_target_family(target_model)
    except Exception:
        family = "pure_attention"

    if family == "hybrid_gdn":
        from dflash_mlx.runtime import _install_target_speculative_hooks

        _install_target_speculative_hooks(target_model)
        configure_full_attention_split(target_model, enabled=True)
        try:
            pack_target_model_weights(target_model, validate=True)
        except Exception as e:
            logger.warning(f"DFlash: weight packing skipped: {e}")
        logger.info("DFlash: speculative hooks installed (hybrid GDN)")
        return True
    else:
        logger.info(f"DFlash: target family={family}, no GDN hooks needed")
        return False


