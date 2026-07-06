# SPDX-License-Identifier: Apache-2.0
"""Fused decode-step patch for mlx-lm's Qwen3.5/3.6 ``GatedDeltaNet``.

Wraps ``mlx_lm.models.qwen3_5.GatedDeltaNet.__call__`` (shared by the
``qwen3_5`` and ``qwen3_5_moe`` model types — ``qwen3_5_moe.Model``
subclasses ``qwen3_5.Model``) with a fast path that runs a single fused
Metal kernel (see ``kernel.py``) covering the conv1d/SiLU/split/q-k-RMSNorm/
gating "glue" AND the delta-rule scan of a T==1 decode step in one dispatch
(see ``kernel.py``'s module docstring for why a two-kernel split — fused
glue kernel + the existing unmodified scan kernel — was tried first and
measured to fall short of the perf gate, and why merging into one dispatch
was the fix).

Composes with the Native-MTP patch (``omlx.patches.mlx_lm_mtp``), which also
replaces ``GatedDeltaNet.__call__`` (for sanitize correctness, independent of
whether MTP decoding is active) and may run before or after this one across
repeated model loads in the same process. This patch always captures
whatever ``__call__`` is *currently* installed as its fallback and always
re-applies itself as long as it isn't already the topmost layer — so it
self-heals regardless of application order, mirroring the MTP patch's own
idempotency pattern. The fast path only ever engages for the plain T==1,
no-mask, warm-cache, non-distributed, non-training case; everything else
(prefill, n_confirmed draft/verify splits, first token of a sequence,
padded/heterogeneous batches, sharded layers, non-Metal/CPU) falls straight
through to that captured fallback, so composing with MTP or DFlash's
lifecycle wrapping never loses correctness -- it just skips the fast path.
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_omlx_qwen3_5_fused_decode_patch"


def apply_qwen3_5_fused_decode_patch() -> bool:
    """Patch ``mlx_lm.models.qwen3_5.GatedDeltaNet.__call__``. Idempotent.

    Returns True when (re-)applied, False when the current ``__call__`` is
    already this patch's own installed function.
    """
    try:
        from mlx_lm.models import qwen3_5 as q35
    except ImportError:
        logger.debug("qwen3_5_fused_decode: mlx_lm.models.qwen3_5 not importable")
        return False

    from .kernel import fused_gdn_decode_step, is_available

    cls = q35.GatedDeltaNet
    current_call = cls.__dict__.get("__call__")
    if getattr(current_call, _PATCH_MARKER, False):
        return False

    fallback_call = current_call

    def _fused_forward(self, inputs, cache):
        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        y, conv_state_out, delta_state_out = fused_gdn_decode_step(
            qkv,
            cache[0],
            self.conv1d.weight,
            b,
            a,
            self.dt_bias,
            self.A_log,
            cache[1],
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
        )
        cache[0] = conv_state_out
        cache[1] = delta_state_out
        cache.advance(S)

        out = self.norm(y, z)
        return self.out_proj(out.reshape(B, S, -1))

    def patched_call(self, inputs, mask=None, cache=None):
        B, S, _ = inputs.shape
        if (
            S == 1
            and mask is None
            and cache is not None
            and cache[0] is not None
            and cache[1] is not None
            and getattr(cache, "lengths", None) is None
            and self.sharding_group is None
            and not self.training
            and is_available()
        ):
            return _fused_forward(self, inputs, cache)
        return fallback_call(self, inputs, mask, cache)

    setattr(patched_call, _PATCH_MARKER, True)
    cls.__call__ = patched_call
    logger.info(
        "qwen3_5_fused_decode: patched mlx_lm.models.qwen3_5.GatedDeltaNet.__call__"
    )
    return True


__all__ = ["apply_qwen3_5_fused_decode_patch"]
