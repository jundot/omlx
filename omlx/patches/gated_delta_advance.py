# SPDX-License-Identifier: Apache-2.0
"""Replace mlx-vlm Qwen3_5GatedDeltaNet.__call__ body with the mlx-lm version.

As of mlx-vlm 191d7c8 (target), upstream ships ``cache.advance(S)`` on its
own, so the original ``ed7884c`` fix is no longer carried by this patch. Two
items remain that upstream still does not have, both of which require touching
the body mid-function:

- ``9dcefa5`` "break shared-buffer memory leak in GatedDeltaNet cache" — wrap
  the ``cache[0]`` write in ``mx.contiguous`` and add the ``cache.lengths is
  not None`` per-element slicing branch.

- Drop the mlx-vlm silent fallbacks (``conv_state.shape[0] != B`` ⇒ zeros,
  same shape for state and mask) that mask real bugs in favor of mlx-lm
  semantics.

The mlx-vlm-specific ``gdn_sink`` parameter is preserved: when given, append
the same tuple the rollback path consumes.

Patch target (current upstream):
- mlx_vlm.models.qwen3_5.language.Qwen3_5GatedDeltaNet (commit 191d7c8)

mlx-lm's GatedDeltaNet already has both fixes, so we leave it untouched.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.gated_delta import gated_delta_update

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)


_patched_classes: set[int] = set()


def _build_replacement_call():
    """Construct the new __call__ matching mlx-lm semantics with mlx-vlm
    signature (gdn_sink) preserved."""

    def __call__(
        self,
        inputs: "mx.array",
        mask: Optional["mx.array"] = None,
        cache: Optional[Any] = None,
        gdn_sink: Optional[list] = None,
    ) -> "mx.array":
        B, S, _ = inputs.shape

        # Optional sharding group (mlx-lm only — mlx-vlm class has no such
        # attribute, so guard with getattr).
        sharding_group = getattr(self, "sharding_group", None)
        if sharding_group is not None:
            from mlx_lm.models.gated_delta import sum_gradients  # type: ignore

            inputs = sum_gradients(sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        # Use -1 for the v-head axis (mlx-vlm style) — equivalent to
        # num_v_heads but tolerant if config drifts.
        z = self.in_proj_z(inputs).reshape(B, S, -1, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)

        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            lengths = getattr(cache, "lengths", None)
            if lengths is not None:
                ends = mx.clip(lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        # mlx-vlm-only: capture pre-update tensors for speculative-cache
        # rollback. Layout matches Qwen3_5GatedDeltaNet upstream.
        if gdn_sink is not None:
            gdn_sink.append(
                (
                    q,
                    k,
                    v,
                    a,
                    b,
                    self.A_log,
                    self.dt_bias,
                    state,
                    mask,
                    conv_input,
                    self.conv_kernel_size,
                )
            )

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        if sharding_group is not None:
            out = mx.distributed.all_sum(out, group=sharding_group)

        return out

    return __call__


def _patch_class(cls: Any, label: str) -> bool:
    """Replace cls.__call__ with the mlx-lm-equivalent body once."""
    if id(cls) in _patched_classes:
        return True
    cls.__call__ = _build_replacement_call()
    _patched_classes.add(id(cls))
    logger.info(f"GatedDeltaNet patch applied (body replacement): {label}")
    return True


def apply_gated_delta_advance_patch(model: Any = None) -> bool:
    """Patch mlx-vlm Qwen3_5GatedDeltaNet to mirror mlx-lm semantics.

    The ``model`` argument is accepted for backward compatibility but
    not used: the patch is class-level. Returns True if the target class
    was importable and (re-)patched.
    """
    if not HAS_MLX:
        logger.debug("mlx not importable; skipping GatedDeltaNet patch")
        return False

    any_patched = False
    try:
        from mlx_vlm.models.qwen3_5.language import (
            Qwen3_5GatedDeltaNet as _VLMGdn,
        )

        _patch_class(_VLMGdn, "mlx_vlm.models.qwen3_5.language.Qwen3_5GatedDeltaNet")
        any_patched = True
    except ImportError:
        logger.debug("mlx_vlm.models.qwen3_5.language not importable")

    return any_patched
