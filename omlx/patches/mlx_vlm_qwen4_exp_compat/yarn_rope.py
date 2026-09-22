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

Fused-path scaling: the upstream Metal mrope kernel computes ``cos/sin``
from ``inv_freq`` directly and has no ``attention_scaling`` input, so at
mscale != 1.0 this module derives a scaling-aware copy and installs it
into the rotary module's ``_compiled_apply`` registry (both position
layouts the dispatcher accepts). Every ``apply_rotary`` consumer — main
attention, gathered-QSA prefill, MTP verify and the shared indexer —
keeps the single-pass fused kernel. Modules outside the kernel envelope
(no Metal, missing position selector, non half-split pairing) fall back
to the eager path, which multiplies ``attention_scaling`` into cos/sin.
With ``mscale_all_dim`` cancelling the temperature there is nothing to
fold and the upstream fused kernel stays active untouched.

Operator override: the oMLX per-model setting ``yarn_context_length``
takes precedence over the checkpoint config. ``configure_yarn_runtime``
binds the target before model construction (set by the loader through
``configure_qwen4_exp_runtime``); ``resolve_rope_parameters`` derives the
recipe fields from the model's native ``max_position_embeddings`` —
factor = target / native, original = native, betas/mscale at Qwen's
defaults — without ever mutating the checkpoint on disk. With no override
bound, the checkpoint's own ``rope_parameters`` pass through unchanged.

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
from functools import cache
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_REQUIRED_YARN_KEYS = ("factor", "original_max_position_embeddings")

_YARN_RUNTIME_CONTEXT_LENGTH: int | None = None


def configure_yarn_runtime(context_length: int | None) -> None:
    """Bind the operator-selected YaRN target before model construction.

    Mirrors the ``configure_ple_runtime`` / ``configure_mtp_runtime``
    module-state pattern: the loader sets this once per model load (None
    clears it) and every attention layer constructed afterwards resolves
    against the same value.
    """
    global _YARN_RUNTIME_CONTEXT_LENGTH
    _YARN_RUNTIME_CONTEXT_LENGTH = int(context_length) if context_length else None


def get_yarn_runtime_context_length() -> int | None:
    return _YARN_RUNTIME_CONTEXT_LENGTH


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


# ---------------------------------------------------------------------------
# Scaling-aware fused MRoPE apply kernel
# ---------------------------------------------------------------------------
#
# Derived from the pinned mlx-vlm ``rope_utils._mrope_apply_kernel``
# (half-split pairing only) with one structural change: an
# ``attention_scaling`` input multiplied into the in-kernel cos/sin,
# matching the eager path's ``cos = mx.cos(emb) * attention_scaling``.
# Installed per instance through ``_compiled_apply``, whose ``.get(ndim)``
# dispatch ``apply_rotary`` consults before it would ever build the
# upstream unscaled kernel.


@cache
def _yarn_mrope_apply_kernel(rotary_dim: int, position_ndim: int):
    """Metal mrope apply with the YaRN mscale folded into cos/sin."""
    if not mx.metal.is_available():
        return None

    if position_ndim == 2:
        position_expr = "position_ids[b * q_len + t]"
        selector_source = ""
    else:
        position_expr = "position_ids[(axis * q_bsz + b) * q_len + t]"
        selector_source = "int axis = int(position_selector[freq_idx]);"

    source = f"""
        uint elem = thread_position_in_grid.x;

        const int half_dim = {rotary_dim // 2};
        const int q_bsz = x_shape[0];
        const int q_heads = x_shape[1];
        const int q_len = x_shape[2];
        const int q_dim = x_shape[3];
        const int slots = half_dim + q_dim - {rotary_dim};
        const int work_size = q_bsz * q_heads * q_len * slots;

        if (elem >= uint(work_size)) {{
            return;
        }}

        int local = int(elem);
        int slot = local % slots;
        int tmp = local / slots;
        int t = tmp % q_len;
        tmp = tmp / q_len;
        int h = tmp % q_heads;
        int b = tmp / q_heads;
        int base = ((b * q_heads + h) * q_len + t) * q_dim;

        if (slot >= half_dim) {{
            int pass_d = {rotary_dim} + slot - half_dim;
            int pass_idx = base + pass_d;
            x_out[pass_idx] = x[pass_idx];
            return;
        }}

        int freq_idx = slot;
        int d = freq_idx;
        int pair_d = d + half_dim;
        {selector_source}
        float pos = static_cast<float>({position_expr});
        float angle = pos * static_cast<float>(inv_freq[freq_idx]);
        float scale = static_cast<float>(attention_scaling[0]);
        float c = metal::cos(angle) * scale;
        float s = metal::sin(angle) * scale;

        int idx = base + d;
        float xv = static_cast<float>(x[idx]);
        float xp = static_cast<float>(x[base + pair_d]);
        x_out[idx] = static_cast<T>(xv * c - xp * s);
        x_out[base + pair_d] = static_cast<T>(xp * c + xv * s);
    """

    return mx.fast.metal_kernel(
        name=f"omlx_yarn_mrope_apply_{rotary_dim}_{position_ndim}d",
        input_names=[
            "x",
            "position_ids",
            "inv_freq",
            "position_selector",
            "attention_scaling",
        ],
        output_names=["x_out"],
        source=source,
    )


def _yarn_fast_mrope_apply(
    kernel, q, k, position_ids, inv_freq, position_selector, attention_scaling
):
    def apply_one(x):
        half_dim = inv_freq.shape[0]
        slots = half_dim + x.shape[-1] - half_dim * 2
        work_size = x.shape[0] * x.shape[1] * x.shape[2] * slots
        (out,) = kernel(
            inputs=[x, position_ids, inv_freq, position_selector, attention_scaling],
            template=[("T", x.dtype)],
            grid=(work_size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )
        return out

    return apply_one(q), apply_one(k)


@cache
def _compiled_yarn_mrope_apply(rotary_dim: int, position_ndim: int):
    kernel = _yarn_mrope_apply_kernel(rotary_dim, position_ndim)
    if kernel is None:
        return None

    @mx.compile
    def apply(q, k, position_ids, inv_freq, position_selector, attention_scaling):
        return _yarn_fast_mrope_apply(
            kernel, q, k, position_ids, inv_freq, position_selector, attention_scaling
        )

    return apply


def _install_scaled_fused_apply(rotary_emb: Any, scaling: float) -> bool:
    """Keep the fused apply path under YaRN with a scaling-aware kernel.

    Replaces ``rotary_emb._compiled_apply`` with closures that bind the
    YaRN mscale for both position layouts the dispatcher accepts (2-D
    text-only, 3-D multimodal). ``apply_rotary`` consults the registry
    before it would ever build the upstream unscaled kernel, so every
    consumer of this instance — main attention, gathered-QSA prefill,
    MTP verify and the shared indexer — stays on the single-pass kernel.

    Returns False when the module sits outside the kernel envelope (no
    Metal, missing position selector, non half-split pairing, or a kernel
    build failure); the caller then keeps the historical eager fallback.
    """
    from mlx_vlm.models.rope_utils import _HALF_SPLIT

    if not rotary_emb.fused_apply:
        return False
    if rotary_emb.pairing != _HALF_SPLIT or rotary_emb.position_selector is None:
        return False
    try:
        scaling_array = mx.array([scaling], dtype=mx.float32)
        registry = {}
        for ndim in (2, 3):
            compiled = _compiled_yarn_mrope_apply(rotary_emb.dim, ndim)
            if compiled is None:
                return False

            def _apply(
                q,
                k,
                position_ids,
                inv_freq,
                position_selector,
                *,
                _compiled=compiled,
                _scaling=scaling_array,
            ):
                return _compiled(
                    q, k, position_ids, inv_freq, position_selector, _scaling
                )

            registry[ndim] = _apply
        mx.eval(scaling_array)
    except Exception:
        logger.debug(
            "YaRN scaling-aware fused mrope kernel unavailable; using the eager path",
            exc_info=True,
        )
        return False
    rotary_emb._compiled_apply = registry
    return True


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
    if scaling != 1.0 and not _install_scaled_fused_apply(rotary_emb, scaling):
        # Outside the fused kernel envelope: keep the historical fallback —
        # the eager path multiplies attention_scaling into cos/sin.
        rotary_emb.fused_apply = False
        rotary_emb._compiled_apply = None
    rotary_emb.eval_cached_arrays()
    logger.info(
        "Qwen4-Exp YaRN RoPE active: dim=%d base=%g factor=%g original=%d "
        "attention_scaling=%.4f (%s)",
        dim,
        base,
        factor,
        original,
        scaling,
        "fused mrope kernel" if rotary_emb.fused_apply else "eager rope path",
    )
    return True


def resolve_rope_parameters(
    rope_parameters: dict | None,
    native_max_position_embeddings: int,
) -> dict | None:
    """Merge the operator YaRN override onto the checkpoint rope config.

    Returns the rope-parameters mapping :func:`maybe_apply_yarn` should
    consume: the runtime target (when bound) wins over any
    checkpoint-declared scaling; with no target the checkpoint config
    passes through unchanged. Checkpoint-provided beta/mscale keys are
    preserved under the override.
    """
    target = _YARN_RUNTIME_CONTEXT_LENGTH
    if not target:
        return rope_parameters
    native = int(native_max_position_embeddings or 0)
    if native <= 0:
        raise ValueError(
            "YaRN override requires a positive native max_position_embeddings, "
            f"got {native_max_position_embeddings!r}"
        )
    if not isinstance(rope_parameters, dict):
        raise ValueError(
            "YaRN override requires the checkpoint rope_parameters mapping "
            "(rope_theta/mrope_section/partial_rotary_factor), got "
            f"{rope_parameters!r}"
        )
    factor = target / native
    if factor <= 1.0:
        logger.warning(
            "YaRN target %d does not exceed the native context %d; serving "
            "the checkpoint rope configuration unscaled",
            target,
            native,
        )
        return rope_parameters
    if factor > 4.0:
        logger.warning(
            "YaRN factor %.2f exceeds Qwen's published recipe maximum "
            "(4.0 = 1M tokens over the native 262144 context); quality past "
            "the published horizon is unvalidated",
            factor,
        )
    resolved = dict(rope_parameters)
    resolved.pop("rope_type", None)
    resolved["type"] = "yarn"
    resolved["factor"] = factor
    resolved["original_max_position_embeddings"] = native
    return resolved
