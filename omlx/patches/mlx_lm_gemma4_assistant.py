# SPDX-License-Identifier: Apache-2.0
"""The gemma4 assistant draft step, shared by the mlx-lm and mlx-vlm paths.

Google ships the gemma4 draft head as a separate ``gemma4_assistant``
model, and the only implementation of it lives in mlx-vlm. Serving a
merged checkpoint through mlx-lm therefore needs mlx-vlm installed —
which is true of gemma4 and of nothing else in ``mlx_lm_mtp``.

Which model needs mlx-vlm is load-bearing, not tidiness.
``cluster.autoconfigure.required_imports`` tells a rank what to install by
scanning each patch package for third-party imports, and it scans the whole
package directory. An ``import mlx_vlm`` anywhere under ``mlx_lm_mtp`` would
therefore ask every rank serving any model — a plain llama included — to
bring mlx-vlm, and over-asking blocks a cluster that would have worked.
Living here instead, imported from the dispatcher under a gemma4 guard,
asks exactly the ranks that need it. Deferring the import to dodge that
scan would hide the requirement and put the failure back in the middle of
a load, which is the bug the scan exists to prevent.

Both engines drive the head identically, so ``draft_step`` lives here too
and both call it. Everything it needs a host to provide — ``mtp``,
``model.embed_tokens``, and the stash a ``return_hidden`` forward leaves
behind — the two hosts already expose under the same names.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# Resolved once and kept. The rejection path then costs a global read rather
# than the import machinery. It stays deferred rather than module-level for
# the reason in the docstring above: an ``import mlx_vlm`` at module scope
# would make importing this module fail outright without mlx-vlm, which is
# the failure ``warn_if_unavailable`` exists to replace with a sentence.
_slice_after_reject = None


def build_draft_model(assistant_config: dict) -> Any:
    """Size the assistant head from the config oQ merged into the checkpoint."""
    from mlx_vlm.speculative.drafters.gemma4_assistant import (
        Gemma4AssistantDraftModel,
        ModelConfig,
    )

    return Gemma4AssistantDraftModel(ModelConfig.from_dict(assistant_config))


def slice_shared_kv_after_reject(shared_kv: dict, rejected: int) -> dict:
    """Drop the rejected tail from a captured K/V stash."""
    global _slice_after_reject

    if _slice_after_reject is None:
        from mlx_vlm.speculative.mtp import (  # noqa: SLF001
            _slice_shared_kv_after_reject,
        )

        _slice_after_reject = _slice_shared_kv_after_reject
    return _slice_after_reject(shared_kv, rejected)


def query_position(host: Any) -> int:
    """Committed length, from the cache stashed at the last verify forward.

    Host-side ints only, never the per-row ``offset`` array, so the chain
    cycle stays sync-free. Rotating caches expose ``_offset`` (absolute
    processed length, where ``_idx`` is a ring index), batched plain caches
    ``_idx`` (storage length, equal to committed under the singleton gating
    MTP requires), and stock caches an int ``offset``. ``trim`` and the
    rollback undo adjust all three, so a post-rollback read is committed
    length rather than verify length.
    """
    for cache in getattr(host, "_omlx_mtp_cache_ref", None) or []:
        rotating = getattr(cache, "_offset", None)
        if isinstance(rotating, int):
            return rotating
        offset = getattr(cache, "offset", None)
        if isinstance(offset, int):
            return offset
        idx = getattr(cache, "_idx", None)
        if isinstance(idx, int):
            return idx
    raise RuntimeError("gemma4 mtp_forward: no cache offset available")


def draft_step(host: Any, hidden_states, next_token_ids, return_hidden: bool = False):
    """One assistant-head draft against the host's stashed backbone K/V.

    The head is single-position: only the last (hidden, token) pair matters,
    so a multi-row history fold slices to the final row. ``hidden_states``
    is the backbone's hidden on the first call of a cycle and the head's own
    ``post_projection`` output on chain steps — both backbone-width, which
    is what ``draft_block`` feeds back as ``h_prev``. The head is stateless,
    so there is no cache to thread.
    """
    drafter = host.mtp
    # nn.quantize() swaps embed_tokens for a QuantizedEmbedding after
    # construction, so a bind taken in __init__ can point at a random-init
    # module — which drafts garbage and reads as a ~10% accept rate.
    if drafter._input_embed is not host.model.embed_tokens:
        drafter.bind(host)

    shared_kv = getattr(host, "_omlx_mtp_shared_kv", None)
    if not shared_kv:
        raise RuntimeError(
            "gemma4 mtp_forward called without a prior return_hidden "
            "backbone forward (no shared K/V stash)"
        )

    h = hidden_states[:, -1:, :]
    tok_embed = drafter._input_embed(next_token_ids[:, -1:])
    tok_embed = tok_embed * drafter._input_embed_scale
    inputs_embeds = mx.concatenate([tok_embed.astype(h.dtype), h], axis=-1)

    # The stash was captured at the verify forward, which ran before
    # rollback trimmed the cache, so on a rejection its tail still holds the
    # rejected rows. The difference is exactly how many to drop.
    valid_len = query_position(host)
    rejected = getattr(host, "_omlx_mtp_kv_offset", valid_len) - valid_len
    if rejected > 0:
        shared_kv = slice_shared_kv_after_reject(shared_kv, rejected)

    drafter._kv_valid_len = valid_len
    # The head's query position is the hidden's own token — the last
    # committed slot, not the next one.
    position_ids = mx.array([[max(valid_len - 1, 0)]])
    head_hidden, logits = drafter(inputs_embeds, shared_kv, position_ids)
    if return_hidden:
        return logits, head_hidden
    return logits


def warn_if_unavailable(model_name: str) -> bool:
    """Say up front when a merged head cannot be attached, and why.

    Without this the load reaches ``Model.__init__``, fails to import the
    drafter, and the operator sees a ModuleNotFoundError from inside model
    construction rather than a sentence naming the missing package.
    """
    try:
        import mlx_vlm.speculative.drafters.gemma4_assistant  # noqa: F401
    except ImportError:
        logger.warning(
            "Gemma 4 assistant MTP for %s needs mlx-vlm, which supplies the "
            "draft head; speculative decoding will stay inactive",
            model_name,
        )
        return False
    return True
