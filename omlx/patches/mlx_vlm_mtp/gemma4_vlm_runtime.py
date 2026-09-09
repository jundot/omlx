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

    # MLX 0.32.2 covers Gemma4's head-dim-256 small-L shapes natively. Global
    # head-dim-512 verify still needs the custom fused route to keep
    # shallow-depth speculation profitable.
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

    from ..mlx_lm_gemma4_assistant import query_position

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
        self._omlx_mtp_kv_offset = query_position(self)
        return LanguageModelOutput(
            logits=out.logits,
            hidden_states=out.hidden_states,
            gdn_states=[],
            shared_kv_states=sink,
        )

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        """Drive the assistant head; the mlx-lm path drives it the same way."""
        del mtp_cache, logits_keep  # stateless head; output is 1 position
        from ..mlx_lm_gemma4_assistant import draft_step

        return draft_step(self, hidden_states, next_token_ids, return_hidden)

    def make_mtp_cache(self):
        """The assistant head keeps no state — nothing to clone or trim."""
        return []

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls._omlx_mtp_runtime_patched = True
