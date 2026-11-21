# SPDX-License-Identifier: Apache-2.0
"""Take a Qwen3.5/3.6 prefill tail snapshot inside the chunk that passes it.

The prefill stops once at a tail position (the start of the generation prompt)
so the recurrent state there can be stored for the next turn of the
conversation. That costs one more model pass for a handful of tokens: on
Qwen3.8-27B with a 128K-token cached document, the 6-token tail chunk takes
~170 ms (all weights are read again and every attention layer runs again).

With this patch a Gated DeltaNet layer whose cache carries a capture offset
``c`` runs its projections and causal conv once over the whole chunk, splits
only the recurrence at ``c``, and keeps the conv window and recurrent state
after the first ``c`` tokens. The scheduler then stores the tail snapshot from
those, and the chunk does not stop at the tail. Attention layers need nothing:
their KV is sliced at the tail when the prompt is stored.

Fail-closed: only unmasked single-row prefill chunks are split, and the
scheduler only asks for it when every stateful cache is an ``ArraysCache``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_PATCHED = False
_CAPTURE_AT = "_omlx_capture_at"
_CAPTURED = "_omlx_captured"


def _capture_forward(self, inputs, cache, c, lang):
    B, S, _ = inputs.shape
    mixed_qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, -1, self.head_v_dim)
    b, a = self._project_gates(inputs)
    conv_state = cache[0]
    if conv_state is None or conv_state.shape[0] != B:
        conv_state = mx.zeros(
            (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
        )
    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    width = self.conv_kernel_size - 1
    conv_at_c = mx.contiguous(conv_input[:, c : c + width])
    cache.update_window(0, conv_input, width, lengths=cache.lengths)
    conv_out = nn.silu(self.conv1d(conv_input))
    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]
    q, k = self._normalize_qk(q, k)
    # Resolved at call time: oMLX rebinds the module's prefill kernel.
    update = lang.gated_delta_update
    out_head, _ = update(
        q[:, :c], k[:, :c], v[:, :c], a[:, :c], b[:, :c], self.A_log, self.dt_bias,
        mask=None, use_kernel=not self.training, cache=cache,
    )
    state_at_c = cache[1]
    out_tail, _ = update(
        q[:, c:], k[:, c:], v[:, c:], a[:, c:], b[:, c:], self.A_log, self.dt_bias,
        mask=None, use_kernel=not self.training, cache=cache,
    )
    out = mx.concatenate([out_head, out_tail], axis=1)
    if hasattr(cache, "advance"):
        cache.advance(S)
        lang._qwen3_5_advance_lengths_info(cache, S)
    setattr(cache, _CAPTURED, (conv_at_c, state_at_c))
    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))


def apply_qwen35_gdn_tail_capture_patch() -> bool:
    """Patch mlx-vlm's Qwen3.5 Gated DeltaNet; True when the patch is in place."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from mlx_vlm.models.qwen3_5 import language as lang
    except Exception:  # noqa: BLE001
        return False
    cls = getattr(lang, "Qwen3_5GatedDeltaNet", None)
    if cls is None or not hasattr(lang, "_qwen3_5_advance_lengths_info"):
        return False
    original = cls.__call__

    def __call__(self, inputs, mask=None, cache=None):
        c = getattr(cache, _CAPTURE_AT, None) if cache is not None else None
        if c is not None:
            setattr(cache, _CAPTURE_AT, None)
            B, S, _ = inputs.shape
            if mask is None and B == 1 and 0 < c < S:
                return _capture_forward(self, inputs, cache, c, lang)
        return original(self, inputs, mask=mask, cache=cache)

    cls.__call__ = __call__
    _PATCHED = True
    logger.info("Qwen3.5 GDN tail capture on")
    return True


def tail_capture_ready() -> bool:
    return _PATCHED


def model_capture_layers(model: Any) -> int:
    """Patched Gated DeltaNet layers in ``model`` (0 when the patch is off).

    The scheduler only skips a tail stop when every stateful cache belongs to
    one of these: other recurrent layers would not keep the state.
    """
    if not _PATCHED or model is None:
        return 0
    cached = getattr(model, "_omlx_tail_capture_layers", None)
    if cached is not None:
        return cached
    try:
        from mlx_vlm.models.qwen3_5 import language as lang

        cls = lang.Qwen3_5GatedDeltaNet
        # oMLX's VLMModelAdapter reaches the language model twice (as
        # _language_model and through the full VLM), so count each layer once.
        seen: set[int] = set()
        for root in (
            model,
            getattr(model, "_language_model", None),
            getattr(model, "language_model", None),
        ):
            if root is not None and hasattr(root, "modules"):
                seen.update(id(m) for m in root.modules() if isinstance(m, cls))
        count = len(seen)
    except Exception:  # noqa: BLE001
        count = 0
    try:
        model._omlx_tail_capture_layers = count
    except Exception:  # noqa: BLE001
        pass
    return count


def request_tail_capture(stateful_caches: list[Any], offset: int) -> bool:
    """Ask every stateful cache to keep its state ``offset`` tokens into the next chunk."""
    if not _PATCHED or offset <= 0 or not stateful_caches:
        return False
    if not all(type(c).__name__ == "ArraysCache" for c in stateful_caches):
        return False
    for c in stateful_caches:
        setattr(c, _CAPTURE_AT, offset)
        setattr(c, _CAPTURED, None)
    return True


def captured_snapshot_cache(prompt_cache: list[Any]) -> list[Any] | None:
    """The cache list as it was at the captured offset, for the tail snapshot.

    Stateful layers become shallow copies holding the captured arrays; every
    other entry is passed through (the snapshot keeps stateful layers only).
    Returns None, and clears the capture, if any stateful layer has none.
    """
    out: list[Any] = []
    missing = False
    for c in prompt_cache:
        if type(c).__name__ != "ArraysCache":
            out.append(c)
            continue
        captured = getattr(c, _CAPTURED, None)
        setattr(c, _CAPTURED, None)
        setattr(c, _CAPTURE_AT, None)
        if captured is None:
            missing = True
            continue
        proxy = copy.copy(c)
        proxy.cache = [captured[0], captured[1]]
        out.append(proxy)
    return None if missing else out
