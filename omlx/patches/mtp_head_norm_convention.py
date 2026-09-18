# SPDX-License-Identifier: Apache-2.0
"""Convention detection for the MTP head's pre-fc RMSNorm gammas.

Qwen3-Next RMSNorm gammas are zero-centered in a raw-HF checkpoint; the
MLX convention stores them shifted by +1. Sanitizers decide per key from
the tensor's own mean (``mean < 0.5`` means raw-HF), which works because
a raw gamma sits near 0 and a shifted one near 1.

``pre_fc_norm_hidden`` and ``pre_fc_norm_embedding`` break that test in
one band. The head damps the target hidden state and the next-token
embedding before ``fc`` fuses them, so a CONVERTED pre-fc gamma can land
well below 0.5 — measured +0.4937 / +0.2734 on Qwen3.6-35B-A3B and
+0.8428 / +0.5393 on Qwen3.8-27B, the latter clearing the cutoff by
0.04. Reading those as raw-HF shifts an already-shifted weight, which
corrupts the first projection of every draft and roughly halves MTP
acceptance while leaving the backbone untouched. mlx-vlm 0.7.0 masked
this by only sanitizing non-MLX checkpoints (``if not is_mlx_format`` in
``utils.py``); 0.7.1 sanitizes unconditionally, so the misfire went live.

The sign still decides the unambiguous cases:

* ``mean < 0``    -- raw-HF. A converted gamma is raw + 1, and raw pre-fc
  gammas measure about -0.8..-0.2, so converted ones are positive.
* ``mean >= 0.5`` -- already converted, as the legacy cutoff says.
* otherwise       -- ambiguous. Defer to the head's per-layer norms
  (``input_layernorm``, ``post_attention_layernorm``, ``q_norm``,
  ``k_norm``), which ARE magnitude-discriminable.

Only the ambiguous band changes behaviour, so mixed bundles (JANG MXFP4
Qwen3.6 keeps ``mtp.norm`` shifted while per-layer head norms stay
raw-HF) classify exactly as before.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# Gammas the plain magnitude cutoff cannot classify on its own.
FC_NORM_SUFFIXES = (
    ".pre_fc_norm_hidden.weight",
    ".pre_fc_norm_embedding.weight",
)

# Head norms whose magnitude reliably reveals the checkpoint's convention.
_LAYER_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)

_RAW_HF_CUTOFF = 0.5


def is_fc_norm(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in FC_NORM_SUFFIXES)


def is_oq_tracked_tensor(value: Any) -> bool:
    # oQ streaming plan discovery hands out no-data placeholders whose mean
    # is unreadable; they can neither vote nor be classified here.
    return value.__class__.__name__ == "_TrackedTensor" and hasattr(value, "_clone")


def _mean(value: Any) -> float:
    import mlx.core as mx

    return float(mx.mean(value.astype(mx.float32)).item())


def head_layer_norms_are_raw_hf(
    weights: Mapping[str, Any],
) -> Optional[bool]:
    """Majority convention of the MTP head's per-layer norms.

    ``True`` for raw-HF, ``False`` for already-shifted MLX, ``None`` when
    no per-layer head norm can be read. A tie counts as already-shifted:
    the shift is the destructive direction, so it needs a real majority.
    """
    votes = []
    for key, value in weights.items():
        if "mtp." not in key or getattr(value, "ndim", None) != 1:
            continue
        if not any(key.endswith(suffix) for suffix in _LAYER_NORM_SUFFIXES):
            continue
        if is_oq_tracked_tensor(value):
            continue
        try:
            votes.append(_mean(value) < _RAW_HF_CUTOFF)
        except Exception:
            continue
    if not votes:
        return None
    return sum(votes) * 2 > len(votes)


def fc_norm_is_raw_hf(value: Any, head_verdict: Optional[bool]) -> bool:
    """Whether a pre-fc gamma still needs the +1 shift.

    ``head_verdict`` comes from :func:`head_layer_norms_are_raw_hf`. The
    caller must have excluded oQ tracked tensors, whose mean is
    unreadable.
    """
    mean = _mean(value)
    if mean < 0.0:
        return True
    if mean >= _RAW_HF_CUTOFF:
        return False
    # Ambiguous band: a converted gamma can live here. With no head
    # evidence, keep the legacy cutoff's answer rather than inventing one.
    return True if head_verdict is None else head_verdict
