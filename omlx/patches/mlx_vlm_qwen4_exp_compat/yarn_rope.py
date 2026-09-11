# SPDX-License-Identifier: Apache-2.0
"""YaRN RoPE scaling for the vendored Qwen4-Exp (Qwen3.8-Flash-Next) stack.

Qwen's official recipe for extending Flash-Next beyond its native 262,144
token context (up to 1,000,000) is static YaRN declared in
``text_config.rope_parameters`` — the same schema vLLM, SGLang and
TokenSpeed consume::

    {
        "rope_type": "yarn",
        "factor": 2.0,                            # 4.0 for the full 1M
        "original_max_position_embeddings": 262144,
        "rope_theta": 10000000,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25
    }

The pinned mlx-vlm ``MRoPERotaryEmbedding`` builds plain
``compute_inv_freq(dim, base)`` frequencies and never dispatches on the
rope type, so without this module the recipe is a silent no-op.
:func:`maybe_apply_yarn` applies the HF-Transformers yarn frequency
correction to the module's ``_inv_freq`` table in place and sets
``attention_scaling`` to the yarn mscale. Both the fused Metal apply
kernel (which re-reads ``inv_freq`` on every call) and the eager cos/sin
path then serve the corrected frequencies, and because the QSA indexer
shares the attention module's rotary instance, sparse block selection
stays consistent with the main attention.

Fused-path caveat: the Metal mrope kernel computes ``cos/sin`` from
``inv_freq`` directly and has no ``attention_scaling`` input. When the
mscale is not 1.0 (the default for factor > 1), the fused path is
disabled so the eager path — which multiplies ``attention_scaling`` into
cos/sin — handles every apply. With ``mscale_all_dim`` cancelling the
temperature the fused kernel stays active.

Known interaction: ``omlx.patches.specprefill`` rebuilds positions with
``manual_rope(base=...)`` unless the wrapped rope stores a ``_freqs``
table, which MRoPE modules do not. Keep SpecPrefill disabled for
yarn-scaled Qwen4-Exp models; the frequency correction would be lost.

Frequency formulas mirror HF Transformers ``ROPE_INIT_FUNCTIONS["yarn"]``
(identical to mlx-lm's ``YarnRoPE``), inverted into the ``_inv_freq``
(1/wavelength) convention MRoPE stores.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_REQUIRED_YARN_KEYS = ("factor", "original_max_position_embeddings")


def _correction_dim(
    num_rotations: float,
    dim: int,
    base: float,
    original_max_position_embeddings: int,
) -> float:
    return (
        dim * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
    ) / (2 * math.log(base))


def _correction_range(
    beta_fast: float,
    beta_slow: float,
    dim: int,
    base: float,
    original_max_position_embeddings: int,
) -> tuple[int, int]:
    low = math.floor(
        _correction_dim(beta_fast, dim, base, original_max_position_embeddings)
    )
    high = math.ceil(
        _correction_dim(beta_slow, dim, base, original_max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def _mscale(scale: float, mscale: float) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_inv_freq(
    dim: int,
    base: float,
    factor: float,
    original_max_position_embeddings: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> mx.array:
    """YaRN-corrected inverse frequencies for *dim* rotary dimensions.

    Returns the ``dim // 2``-entry ``1/wavelength`` table in the layout
    ``MRoPERotaryEmbedding._inv_freq`` uses.
    """
    freq_extra = base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
    freq_inter = factor * freq_extra
    low, high = _correction_range(
        beta_fast, beta_slow, dim, base, original_max_position_embeddings
    )
    ramp_high = float(high)
    if low == high:
        ramp_high += 0.001  # prevent the ramp singularity, as HF does
    ramp = mx.clip(
        (mx.arange(dim // 2, dtype=mx.float32) - float(low)) / (ramp_high - float(low)),
        0.0,
        1.0,
    )
    freq_mask = 1.0 - ramp
    wavelengths = (freq_inter * freq_extra) / (
        freq_inter * freq_mask + freq_extra * (1.0 - freq_mask)
    )
    return 1.0 / wavelengths


def yarn_attention_scaling(
    factor: float,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
) -> float:
    """YaRN attention temperature (multiplies cos/sin), per HF/vLLM."""
    return _mscale(factor, mscale) / _mscale(factor, mscale_all_dim)


def maybe_apply_yarn(rotary_emb: Any, rope_parameters: dict | None) -> bool:
    """Apply YaRN scaling to *rotary_emb* in place when the config selects it.

    Returns True when the correction was applied. A ``default`` (or
    missing) rope type is a no-op; any other type raises instead of
    silently serving unscaled positions past the trained horizon.
    """
    if not isinstance(rope_parameters, dict):
        return False
    rope_type = (
        rope_parameters.get("type") or rope_parameters.get("rope_type") or "default"
    )
    if rope_type == "default":
        return False
    if rope_type != "yarn":
        raise ValueError(
            f"Unsupported Qwen4-Exp RoPE type {rope_type!r}: this stack "
            "implements 'default' and 'yarn'"
        )
    missing = [key for key in _REQUIRED_YARN_KEYS if rope_parameters.get(key) is None]
    if missing:
        raise ValueError(
            "rope_parameters with rope type 'yarn' must include "
            + ", ".join(repr(key) for key in missing)
        )

    factor = float(rope_parameters["factor"])
    original = int(rope_parameters["original_max_position_embeddings"])
    dim = int(rotary_emb.dim)
    base = float(rotary_emb.base)
    rotary_emb._inv_freq = yarn_inv_freq(
        dim,
        base,
        factor,
        original,
        beta_fast=float(rope_parameters.get("beta_fast", 32)),
        beta_slow=float(rope_parameters.get("beta_slow", 1)),
    )
    scaling = yarn_attention_scaling(
        factor,
        mscale=float(rope_parameters.get("mscale", 1)),
        mscale_all_dim=float(rope_parameters.get("mscale_all_dim", 0)),
    )
    rotary_emb.attention_scaling = scaling
    if scaling != 1.0:
        # The fused Metal mrope kernel ignores attention_scaling; drop to
        # the eager path, which bakes it into cos/sin.
        rotary_emb.fused_apply = False
        rotary_emb._compiled_apply = None
    rotary_emb.eval_cached_arrays()
    logger.info(
        "Qwen4-Exp YaRN RoPE active: dim=%d base=%g factor=%g original=%d "
        "attention_scaling=%.4f",
        dim,
        base,
        factor,
        original,
        scaling,
    )
    return True
