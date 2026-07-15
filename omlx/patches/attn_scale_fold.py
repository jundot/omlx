"""Identity I-I: Row-constant fold — pre-multiply attention scale at load time.

For any attention module with a `scale` attribute (head_dim**-0.5), the runtime
`queries *= scale` multiply is eliminated by folding the constant into:
  - q_norm.weight (architectures with per-head QK normalization, e.g. Qwen3.x)
  - q_proj.scales (quantized, architectures without q_norm)

After folding, module.scale is set to 1.0.  The attention forward already
passes `scale=self.scale` to SDPA, so no other code changes are needed — the
kernel receives scale=1.0 and the per-token elementwise multiply is gone.

Applied once at model load time via apply_attn_scale_fold(model).
"""
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _fold_one(module: nn.Module, name: str) -> bool:
    """Fold module.scale into q_norm.weight or q_proj.scales.

    Returns True if the fold was applied.
    """
    scale = getattr(module, "scale", None)
    if not isinstance(scale, (int, float)) or float(scale) == 1.0:
        return False

    scale = float(scale)

    # Preferred: fold into q_norm.weight (Qwen3.x and similar architectures).
    # q_norm is applied after q_proj, so multiplying its learnable weight by
    # scale bakes in the constant without touching the quantized weights.
    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        w = getattr(q_norm, "weight", None)
        if isinstance(w, mx.array):
            q_norm.weight = w * scale
            mx.eval(q_norm.weight)
            module.scale = 1.0
            logger.debug(
                "I-I scale fold: %s — scale=%.6f → q_norm.weight", name, scale
            )
            return True

    # Fallback: no q_norm — fold directly into q_proj quantized scales.
    # Valid when scale is applied directly to the output of the projection
    # (D_α Ŵ with all α_i = scale).
    q_proj = getattr(module, "q_proj", None)
    if isinstance(q_proj, nn.QuantizedLinear):
        # Use object.__setattr__ to bypass Module bookkeeping on frozen layers.
        object.__setattr__(q_proj, "scales", q_proj.scales * scale)
        mx.eval(q_proj.scales)
        module.scale = 1.0
        logger.debug(
            "I-I scale fold: %s — scale=%.6f → q_proj.scales", name, scale
        )
        return True

    return False


def apply_attn_scale_fold(model: nn.Module) -> int:
    """Traverse model and fold attention scale constants into weights.

    Returns the number of attention modules where the fold was applied.
    """
    n = 0
    for name, module in model.named_modules():
        if _fold_one(module, name):
            n += 1

    if n:
        logger.info(
            "attn_scale_fold (I-I): folded attention scale into %d modules "
            "(queries *= scale eliminated)",
            n,
        )
    return n
