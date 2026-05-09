# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-vlm Qwen3_5Attention to use plain RoPE on text-only inputs.

Background
----------
mlx-lm's Qwen3.5/3.6 path uses ``Qwen3NextAttention`` which applies plain
1D RoPE with ``self.rope(x, offset=cache.offset)``. That path is correct
under prefix-cache restore (verified by Qwen3.6-35B-A3B-oQ4 cached-length
sweep 12/12 PASS on the mlx-lm engine).

mlx-vlm's ``Qwen3_5Attention`` always uses multimodal RoPE (mRoPE) with
``apply_multimodal_rotary_pos_emb``, even on text-only inputs. mRoPE is
mathematically equivalent to plain RoPE when all three position-id
sections carry identical values, but the mRoPE numerics under prefix-cache
restore on the mlx-vlm path produce greedy divergence (10/12 FAIL on the
same sweep).

This patch replaces ``Qwen3_5Attention.__call__`` with a body that
- detects text-only inputs (position_ids sections all equal, or no
  position_ids passed) and applies plain RoPE matching the mlx-lm flow
- preserves the original mRoPE branch when position_ids carries
  genuinely multimodal positions (so vision input still works)

Patch target (current upstream): mlx-vlm e41cd25
- ``mlx_vlm.models.qwen3_5.language.Qwen3_5Attention``

The companion file ``gated_delta_advance.py`` patches the GatedDeltaNet
of the same module.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)


_patched_classes: set[int] = set()
_call_counter = {"plain": 0, "mrope": 0, "forced": 0}


# Thread-local force-plain flag.  VLMModelAdapter sets this around forward
# calls during text-only decode so the patch can skip the per-layer
# ``mx.all(...).item()`` probes inside ``_is_text_only_position_ids``.
# Each .item() forces a GPU→CPU sync; with 40-64 layers and two probes
# per call, that adds 5-10 ms per generated token (Qwen tg slowdown
# observed in `/tmp/omlx_bench/results_omlx*.json`).
_local = threading.local()


def _is_force_text_only() -> bool:
    return bool(getattr(_local, "force_text_only", False))


class force_text_only_rope:
    """Context manager that forces ``use_plain=True`` in patched Qwen3_5Attention.

    Reentry-safe via depth counter so nested decode calls inside the same
    thread (e.g. chunked prefill that loops the language model) don't drop
    the flag prematurely.  Single-thread executor in ``engine_core`` makes
    races impossible, but threading.local keeps it safe under any caller.
    """

    def __enter__(self):
        _local.depth = getattr(_local, "depth", 0) + 1
        _local.force_text_only = True
        return self

    def __exit__(self, exc_type, exc, tb):
        depth = getattr(_local, "depth", 1) - 1
        _local.depth = depth
        if depth <= 0:
            _local.force_text_only = False
            _local.depth = 0
        return False


def _rotate_half(x):
    """Same as mlx-vlm's rotate_half — split last dim in half, [-x2, x1]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def _is_text_only_position_ids(position_ids: "mx.array") -> bool:
    """Return True if position_ids is a text-only mRoPE tensor (all 3 sections
    identical), or doesn't have the multimodal triplet shape at all.

    Triggers a small mx.eval via .item() once per attention call, but this
    is an O(L) reduction so the overhead per layer is negligible compared
    to the matmul cost."""
    if position_ids.ndim < 3 or position_ids.shape[0] != 3:
        return True
    p0 = position_ids[0]
    p1 = position_ids[1]
    p2 = position_ids[2]
    same_01 = mx.all(p0 == p1).item()
    if not bool(same_01):
        return False
    same_12 = mx.all(p1 == p2).item()
    return bool(same_12)


def _build_replacement_call():
    """Construct the replacement Qwen3_5Attention.__call__."""

    def __call__(
        self,
        x: "mx.array",
        mask: Optional["mx.array"] = None,
        cache: Optional[Any] = None,
        position_ids: Optional["mx.array"] = None,
    ) -> "mx.array":
        B, L, D = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        # Decide RoPE path. Default = plain (text-only); fall back to mRoPE
        # only when position_ids carries genuinely different multimodal
        # positions across the 3 sections.
        #
        # Fast path: when VLMModelAdapter wraps a text-only decode with
        # force_text_only_rope(), skip the .item() probes entirely.
        forced = _is_force_text_only()
        use_plain = True
        if not forced and position_ids is not None:
            try:
                use_plain = _is_text_only_position_ids(position_ids)
            except Exception as e:
                logger.warning(
                    f"Qwen3_5Attention patch: position_ids check failed ({e}); "
                    "falling back to mRoPE path."
                )
                use_plain = False

        # Track usage so it's visible in logs whether the plain/mRoPE/forced
        # branch is being exercised.
        if forced:
            _call_counter["forced"] += 1
            if _call_counter["forced"] in (1, 100, 1000):
                logger.debug(
                    f"[qwen3_5-attn-patch] forced-plain call #{_call_counter['forced']} "
                    f"B,L={B},{L}"
                )
        elif use_plain:
            _call_counter["plain"] += 1
            if _call_counter["plain"] in (1, 100, 1000):
                logger.debug(
                    f"[qwen3_5-attn-patch] plain-rope call #{_call_counter['plain']} "
                    f"B,L={B},{L} cache_offset="
                    f"{getattr(cache, 'offset', '-') if cache is not None else '-'}"
                )
        else:
            _call_counter["mrope"] += 1
            if _call_counter["mrope"] in (1, 100, 1000):
                logger.debug(
                    f"[qwen3_5-attn-patch] mRoPE call #{_call_counter['mrope']} "
                    f"B,L={B},{L}"
                )

        if use_plain:
            # Plain RoPE branch — mirror mlx-lm Qwen3NextAttention behavior.
            #
            # cache.offset can be int (single-batch) or mx.array (batched
            # with per-element offsets); both are handled by mx.arange
            # broadcasting up to a point. For per-element offsets we'd need
            # additional logic, but the immediate goal is to match the
            # mlx-lm engine path which is also int-offset for our targets.
            offset = (
                cache.offset
                if cache is not None and hasattr(cache, "offset")
                else 0
            )

            # Build positions [offset, offset+L) and compute freqs once.
            # rotary_emb.inv_freq has shape (rotary_dim/2,), where rotary_dim
            # = int(head_dim * partial_rotary_factor) and was set in
            # Qwen3_5RotaryEmbedding.__init__.
            inv_freq = self.rotary_emb.inv_freq  # (rotary_dim/2,)

            if isinstance(offset, mx.array):
                # offset may be an int 0-d array; squeeze to scalar by item
                # if possible. For multi-element offset arrays, fall back
                # to mRoPE since plain RoPE here doesn't model per-batch
                # offsets.
                if offset.ndim == 0:
                    offset_val = int(offset.item())
                else:
                    # Cannot uniformly serve different offsets per batch
                    # element with this simple path — defer to mRoPE.
                    cos, sin = self.rotary_emb(values, position_ids)
                    from mlx_vlm.models.qwen3_5.language import (
                        apply_multimodal_rotary_pos_emb,
                    )
                    queries, keys = apply_multimodal_rotary_pos_emb(
                        queries, keys, cos, sin
                    )
                    if cache is not None:
                        keys, values = cache.update_and_fetch(keys, values)
                    from mlx_vlm.models.base import scaled_dot_product_attention

                    output = scaled_dot_product_attention(
                        queries,
                        keys,
                        values,
                        cache=cache,
                        scale=self.scale,
                        mask=mask,
                    )
                    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    return self.o_proj(output * mx.sigmoid(gate))
            else:
                offset_val = int(offset)

            positions = mx.arange(offset_val, offset_val + L).astype(mx.float32)
            # freqs: (L, rotary_dim/2)
            freqs = positions[:, None] * inv_freq[None, :].astype(mx.float32)
            # emb: (L, rotary_dim) — duplicate to match rotate_half layout
            emb = mx.concatenate([freqs, freqs], axis=-1)
            cos = mx.cos(emb)[None, None, :, :]  # (1, 1, L, rotary_dim)
            sin = mx.sin(emb)[None, None, :, :]

            rotary_dim = cos.shape[-1]
            q_rot = queries[..., :rotary_dim]
            q_pass = queries[..., rotary_dim:]
            k_rot = keys[..., :rotary_dim]
            k_pass = keys[..., rotary_dim:]

            dtype = queries.dtype
            q_rot = ((q_rot * cos) + (_rotate_half(q_rot) * sin)).astype(dtype)
            k_rot = ((k_rot * cos) + (_rotate_half(k_rot) * sin)).astype(dtype)

            queries = mx.concatenate([q_rot, q_pass], axis=-1)
            keys = mx.concatenate([k_rot, k_pass], axis=-1)
        else:
            # multimodal mRoPE branch — original mlx-vlm flow
            from mlx_vlm.models.qwen3_5.language import (
                apply_multimodal_rotary_pos_emb,
            )

            cos, sin = self.rotary_emb(values, position_ids)
            queries, keys = apply_multimodal_rotary_pos_emb(
                queries, keys, cos, sin
            )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        from mlx_vlm.models.base import scaled_dot_product_attention

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))

    return __call__


def _patch_class(cls: Any, label: str) -> bool:
    if id(cls) in _patched_classes:
        return True
    cls.__call__ = _build_replacement_call()
    _patched_classes.add(id(cls))
    logger.info(f"Qwen3_5Attention patch applied (body replacement): {label}")
    return True


def apply_qwen3_5_attention_patch(model: Any = None) -> bool:
    """Patch mlx-vlm Qwen3_5Attention to use plain RoPE on text-only inputs.

    Returns True if the target class was importable and (re-)patched.
    """
    if not HAS_MLX:
        logger.debug("mlx not importable; skipping Qwen3_5Attention patch")
        return False

    try:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5Attention as _VLMAttn

        _patch_class(
            _VLMAttn, "mlx_vlm.models.qwen3_5.language.Qwen3_5Attention"
        )
        return True
    except ImportError:
        logger.debug("mlx_vlm.models.qwen3_5.language not importable")
        return False
