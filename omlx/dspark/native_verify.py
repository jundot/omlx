"""Target-verification mathematics shared by native dSpark handlers.

Adapted in part from ARahim3/mlx-dspark (MIT); see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import mlx.core as mx

from .native_sampling import sample_probs, truncate_probs


def speculative_sample_accept(
    target_logits,
    draft: list[int],
    draft_probs,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> tuple[int, int]:
    """Exact Leviathan/Chen rejection sampling for one verified block."""
    length = len(draft)
    target_probs = truncate_probs(
        mx.softmax(target_logits / temperature, axis=-1), top_p, top_k
    )
    rows = mx.arange(length)
    indices = mx.array(draft)
    target_selected = target_probs[rows, indices]
    draft_selected = draft_probs[rows, indices]
    uniforms = mx.random.uniform(shape=(length,))
    accepted = uniforms < mx.minimum(
        1.0, target_selected / mx.maximum(draft_selected, 1e-9)
    )
    count = int(mx.cumprod(accepted.astype(mx.int32)).sum().item())
    if count < length:
        residual = mx.maximum(target_probs[count] - draft_probs[count], 0.0)
        residual = residual / mx.maximum(residual.sum(), 1e-9)
        replacement = int(mx.random.categorical(mx.log(residual + 1e-20)).item())
    else:
        replacement = int(sample_probs(target_probs[length]).item())
    return count, replacement
