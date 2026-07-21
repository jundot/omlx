# SPDX-License-Identifier: Apache-2.0
"""YaRN RoPE frequency correction for Qwen3.5/3.6 MRoPE models.

mlx-vlm's ``MRoPERotaryEmbedding`` computes plain inverse frequencies and
ignores the ``type: "yarn"`` field in ``rope_parameters``.  This patch wraps
``Qwen3_5RotaryEmbedding.__init__`` so that, when YaRN parameters are active,
the base ``_inv_freq`` is recomputed with YaRN's interpolation/extrapolation
blend — the same math as mlx-lm's ``YarnRoPE`` but applied in inv_freq space.

MRoPE sectioning (temporal / height / width position IDs) is orthogonal to
the frequency correction and is unaffected.

Limitation: the ``mscale`` attention amplitude correction is stored as
``self.attention_scaling`` and takes effect in the slow Python path, but the
fused Metal ``apply_rotary`` kernel bypasses it.  For typical YaRN factors
(2–4×) the mscale is ≈1.07–1.14, so the impact is minor.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

_CLASS_PATCHED = False
_ACTIVE_YARN_PARAMS: dict[str, Any] | None = None


def set_yarn_params(params: dict[str, Any] | None) -> None:
    global _ACTIVE_YARN_PARAMS
    _ACTIVE_YARN_PARAMS = params


def _compute_yarn_inv_freq(
    inv_freq: Any,
    dim: int,
    base: float,
    factor: float,
    orig_max_pos: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Any:
    """Apply YaRN frequency correction to plain inv_freq (in inv_freq space).

    Returns a new array with the same shape as *inv_freq*.
    """
    import mlx.core as mx

    half_dim = dim // 2

    def _correction_dim(num_rotations: float) -> float:
        return (
            dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    low = max(math.floor(_correction_dim(beta_fast)), 0)
    high = min(math.ceil(_correction_dim(beta_slow)), half_dim - 1)

    denom = max(high - low, 1e-4)
    ramp = mx.clip(
        (mx.arange(half_dim, dtype=mx.float32) - low) / denom, 0, 1
    )
    freq_mask = 1.0 - ramp

    return inv_freq * (freq_mask + (1.0 - freq_mask) / factor)


def _yarn_mscale(
    factor: float,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
) -> float:
    """Compute the YaRN attention amplitude correction factor."""
    if factor <= 1.0:
        return 1.0

    def _ms(s: float, m: float) -> float:
        return 0.1 * m * math.log(s) + 1.0 if s > 1.0 else 1.0

    return _ms(factor, mscale) / _ms(factor, mscale_all_dim)


def apply_qwen35_yarn_rope_patch() -> bool:
    """Install the YaRN ``__init__`` wrapper on ``Qwen3_5RotaryEmbedding``.

    The wrapper is a no-op when ``_ACTIVE_YARN_PARAMS`` is ``None``, so
    non-YaRN models are unaffected.  Call :func:`set_yarn_params` before
    model load to activate.

    Safe to call repeatedly — idempotent via ``_CLASS_PATCHED``.
    """
    global _CLASS_PATCHED
    if _CLASS_PATCHED:
        return True

    try:
        from mlx_vlm.models.qwen3_5 import language as lang_module
    except ImportError:
        return False

    cls = getattr(lang_module, "Qwen3_5RotaryEmbedding", None)
    if cls is None:
        return False

    if getattr(cls.__init__, "_omlx_yarn_patched", False):
        _CLASS_PATCHED = True
        return True

    original_init = cls.__init__

    def _yarn_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)

        params = _ACTIVE_YARN_PARAMS
        if params is None:
            return

        import mlx.core as mx

        factor = params["factor"]
        yarn_inv_freq = _compute_yarn_inv_freq(
            self._inv_freq,
            self.dim,
            self.base,
            factor,
            params["orig_max_pos"],
            params.get("beta_fast", 32.0),
            params.get("beta_slow", 1.0),
        )
        self._inv_freq = yarn_inv_freq

        mscale = _yarn_mscale(
            factor,
            params.get("mscale", 1.0),
            params.get("mscale_all_dim", 0.0),
        )
        if mscale != 1.0:
            self.attention_scaling = mscale

        self.eval_cached_arrays()

    _yarn_init._omlx_yarn_patched = True  # type: ignore[attr-defined]
    cls.__init__ = _yarn_init  # type: ignore[method-assign]
    _CLASS_PATCHED = True
    return True
