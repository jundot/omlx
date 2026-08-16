# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP head attachment for the mlx-vlm Gemma 4 VLM path.

Gemma 4 ships its MTP head as a separate ``gemma4_assistant`` checkpoint
(mlx-vlm loads it via ``load_drafter`` for the vlm_mtp round-loop). This
patch instead supports **merged single-checkpoint** models: the assistant
weights live under ``language_model.mtp.*`` in the main checkpoint and the
assistant config is embedded at ``text_config.mtp_assistant_config``. With
``model_settings.mtp_enabled`` the model then runs through the Lightning
MTP cycle in ``mlx_lm_mtp.batch_generator`` — batched decode, paged /
prefix / SSD caches and rollback all unchanged.

How the assistant head differs from the Qwen3.5 MTPModule:

* It is **stateless** — no KV cache of its own (``make_mtp_cache`` returns
  ``[]``). Its four decoder layers attend over the backbone's last
  sliding-attention and last full-attention layers' K/V (Gemma 4 KV
  sharing, ``kv_shared_only=True``).
* Drafting holds the query position constant and feeds the head's
  ``post_projection`` output back as the next chain step's hidden input
  (mirrors HF ``SinglePositionMultiTokenCandidateGenerator``).

To feed it inside the Lightning cycle, the patched ``__call__`` captures
``shared_kv_sink`` during every ``return_hidden=True`` backbone forward
(the verify / post-init forwards) and stashes it with the live cache list;
``mtp_forward`` then reads the query position from the current cache
offset, so post-rollback folds automatically see the committed length and
mask the rejected tail via ``kv_valid_len``.

Known approximation: after a rejection, the stashed sliding-window bank
still contains the rejected draft rows. The full-attention bank masks them
exactly by index; a rotated sliding ring can mask up to ``depth`` wrong
slots out of ``sliding_window`` until the next verify refreshes the stash.
This only affects draft quality (accept rate), never output correctness —
the verify forward guarantees the output distribution.

Apply ordering: must run *before* ``mlx_vlm.utils.load(...)`` so the
patched ``LanguageModel.__init__`` attaches the head for weight binding.
``maybe_apply_pre_load_patches`` handles this for inference loads.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_APPLIED = False


def apply() -> bool:
    """Apply the mlx-vlm Gemma 4 runtime MTP patches. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True

    try:
        from mlx_vlm.models.gemma4 import config as g4_config
        from mlx_vlm.models.gemma4 import language as g4_lang
        from mlx_vlm.models.gemma4_unified import config as g4_unified_config
        from mlx_vlm.speculative.drafters.gemma4_assistant import (  # noqa: F401
            Gemma4AssistantDraftModel,
        )
    except Exception as e:
        logger.debug(f"mlx_vlm.gemma4 not importable for MTP runtime: {e}")
        return False

    _patch_text_config(g4_config)
    # Gemma4 unified reuses Gemma4's LanguageModel but declares its own
    # TextConfig subclass. Retain the embedded assistant config there too so
    # the shared language model can attach the head during loading.
    _patch_text_config(g4_unified_config)
    _patch_vlm_language_model(g4_lang)
    # VLMModelAdapter pass-throughs (mtp / mtp_forward / make_mtp_cache /
    # rollback_speculative_cache) are shared with the Qwen3.5 runtimes;
    # the helper is idempotent.
    from .qwen35_vlm_runtime import _patch_vlm_model_adapter

    _patch_vlm_model_adapter()

    # Small-L verify forwards otherwise pay gemma4's unfused multi-token
    # attention (head_dim 256/512 has no fused SDPA above L=1); the
    # decomposed route keeps shallow-depth speculation profitable.
    from ..gemma4_verify_attention import apply as apply_verify_attention

    apply_verify_attention()

    _APPLIED = True
    logger.info("mlx-vlm Gemma 4 runtime MTP patch applied")
    return True


# ---------------------------------------------------------------------------
# TextConfig — retain the embedded assistant config across from_dict.
# ---------------------------------------------------------------------------


def _patch_text_config(g4_config: Any) -> None:
    """Wrap ``TextConfig.from_dict`` so ``mtp_assistant_config`` survives.

    mlx-vlm's ``BaseModelConfig.from_dict`` filters params by the dataclass
    signature, dropping the embedded assistant config dict. Without it the
    patched ``LanguageModel.__init__`` cannot size or attach the head.
    """
    cls = g4_config.TextConfig
    if getattr(cls, "_omlx_mtp_from_dict_patched", False):
        return

    original_from_dict = cls.from_dict.__func__  # unwrap classmethod

    def patched_from_dict(cls_inner, params):
        instance = original_from_dict(cls_inner, params)
        if params:
            instance.mtp_assistant_config = params.get("mtp_assistant_config")
        else:
            instance.mtp_assistant_config = None
        return instance

    cls.from_dict = classmethod(patched_from_dict)
    cls._omlx_mtp_from_dict_patched = True


# ---------------------------------------------------------------------------
# LanguageModel — attach head, stash shared K/V, add mtp_forward/cache.
# ---------------------------------------------------------------------------


def _patch_vlm_language_model(g4_lang: Any) -> None:
    cls = g4_lang.LanguageModel
    if "_omlx_mtp_runtime_patched" in cls.__dict__:
        return

    from mlx_vlm.models.base import LanguageModelOutput
    from mlx_vlm.speculative.drafters.gemma4_assistant import (
        Gemma4AssistantDraftModel,
        ModelConfig as Gemma4AssistantConfig,
    )
    from mlx_vlm.speculative.mtp import _slice_shared_kv_after_reject  # noqa: SLF001

    original_init = cls.__init__
    original_call = cls.__call__

    def __init__(self, config):
        from . import is_mtp_attach_enabled
        from ..mlx_lm_mtp import get_mtp_depth, is_mtp_active

        original_init(self, config)
        asst_cfg = getattr(config, "mtp_assistant_config", None)
        attach = bool(asst_cfg) and bool(is_mtp_attach_enabled())
        self._omlx_mtp_decode_enabled = bool(attach and is_mtp_active())
        if attach:
            drafter_config = Gemma4AssistantConfig.from_dict(asst_cfg)
            self.mtp = Gemma4AssistantDraftModel(drafter_config)
            # Binds the backbone's embed_tokens (+ scale) for the fused
            # [token_embed, hidden] input and resolves the head's tied
            # lm_head fn. Function refs read weights at call time, so
            # binding before load_weights is safe.
            self.mtp.bind(self)
        if self._omlx_mtp_decode_enabled:
            # The chain cycle applies the backbone's final RMSNorm to the
            # verify hidden rows (HEAD_HIDDEN_POST_NORM) — exactly the
            # ``speculative_draft_hidden`` variant this drafter consumes.
            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = get_mtp_depth()

    def __call__(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs):
        """Backbone forward with MTP-cycle shared-K/V capture.

        For ``return_hidden=True`` forwards on an MTP-enabled instance
        (the Lightning verify / post-init calls), capture the shared K/V
        banks the assistant head drafts against and remember the live
        cache list for position bookkeeping. ``gdn_states=[]`` (Gemma 4
        has no SSM state) routes ``_chain_rollback`` to the stock
        ``rollback_speculative_cache``.
        """
        return_hidden = bool(kwargs.get("return_hidden", False))
        if not (return_hidden and getattr(self, "_omlx_mtp_decode_enabled", False)):
            return original_call(self, inputs, inputs_embeds, mask, cache, **kwargs)

        kwargs.pop("n_confirmed", None)  # mlx-vlm rollback is post-hoc
        sink = kwargs.pop("shared_kv_sink", None)
        if sink is None:
            sink = {}
        out = original_call(
            self,
            inputs,
            inputs_embeds,
            mask,
            cache,
            shared_kv_sink=sink,
            **kwargs,
        )
        self._omlx_mtp_shared_kv = sink
        self._omlx_mtp_cache_ref = cache
        # Committed length at capture time. A later rollback lowers the
        # live cache counters below this, and mtp_forward slices the
        # rejected tail off the stashed banks by the difference.
        self._omlx_mtp_kv_offset = _mtp_query_position(self)
        return LanguageModelOutput(
            logits=out.logits,
            hidden_states=out.hidden_states,
            gdn_states=[],
            shared_kv_states=sink,
        )

    def _mtp_query_position(self) -> int:
        """Current committed length = the drafter's constant query position.

        Read from the live cache list (stashed at the last verify forward)
        so post-rollback folds see the committed offset, not the verify
        length. Uses the caches' host-side int counters — never the
        per-row ``offset`` mx.array — to keep the chain cycle sync-free:
        ``BatchRotatingKVCache._offset`` is the absolute processed length
        (its ``_idx`` is a ring index), ``BatchKVCache._idx`` is the
        storage length (== committed length; singleton MTP activation
        guarantees zero left padding), and plain caches keep an int
        ``offset``. All three are adjusted by ``trim`` / the rollback
        undo, so post-rollback reads are committed-only.
        """
        for c in self._omlx_mtp_cache_ref or []:
            rotating_offset = getattr(c, "_offset", None)
            if isinstance(rotating_offset, int):
                return rotating_offset
            offset = getattr(c, "offset", None)
            if isinstance(offset, int):
                return offset
            idx = getattr(c, "_idx", None)
            if isinstance(idx, int):
                return idx
        raise RuntimeError("gemma4 mtp_forward: no cache offset available")

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        """Assistant-head forward under the Lightning chain contract.

        The head is single-position (no history fold): only the last
        (hidden, token) pair matters, so multi-row history folds slice to
        the final position. ``hidden_states`` is either the trunk-normed
        backbone hidden (first fold call) or the head's own
        ``post_projection`` output (chain steps) — both 5376-dim, matching
        ``draft_block``'s ``h_prev`` feedback. ``mtp_cache`` is unused
        (stateless head).
        """
        del mtp_cache, logits_keep  # stateless head; output is 1 position
        drafter = self.mtp
        # Re-bind when the backbone embed module was swapped after the
        # __init__-time bind — nn.quantize() replaces embed_tokens with a
        # QuantizedEmbedding AFTER model construction, and a stale binding
        # keeps a random-init nn.Embedding (garbage drafts, ~10% accept).
        if drafter._input_embed is not self.model.embed_tokens:
            drafter.bind(self)
        shared_kv = getattr(self, "_omlx_mtp_shared_kv", None)
        if not shared_kv:
            raise RuntimeError(
                "gemma4 mtp_forward called without a prior "
                "return_hidden backbone forward (no shared K/V stash)"
            )

        h = hidden_states[:, -1:, :]
        ids = next_token_ids[:, -1:]
        tok_embed = drafter._input_embed(ids) * drafter._input_embed_scale
        inputs_embeds = mx.concatenate([tok_embed.astype(h.dtype), h], axis=-1)

        # Committed length (post-rollback). The stash was captured at the
        # verify forward, so on a rejection it still carries the rejected
        # draft rows in the tail — slice them off (verify captures are
        # temporal-ordered, mirrors _mtp_rounds' post-reject slicing).
        valid_len = _mtp_query_position(self)
        rejected = getattr(self, "_omlx_mtp_kv_offset", valid_len) - valid_len
        if rejected > 0:
            shared_kv = _slice_shared_kv_after_reject(shared_kv, rejected)

        # The drafter's constant query position is the position of the
        # hidden's token — the last committed slot, not the next one
        # (mirrors _mtp_draft_position).
        drafter._kv_valid_len = valid_len
        position_ids = mx.array([[max(valid_len - 1, 0)]])
        head_hidden, logits = drafter(inputs_embeds, shared_kv, position_ids)
        if return_hidden:
            return logits, head_hidden
        return logits

    def make_mtp_cache(self):
        """The assistant head keeps no state — nothing to clone or trim."""
        return []

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls._omlx_mtp_runtime_patched = True
