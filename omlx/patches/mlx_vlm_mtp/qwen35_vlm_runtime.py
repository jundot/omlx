# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP head attachment for the mlx-vlm Qwen3.5 (dense) VLM path.

Mirror of ``qwen35_moe_vlm_runtime.py`` for the dense Qwen3.5/3.6 family
(model_type=qwen3_5, e.g. Qwen3.6-27B). The MoE variant was wired up in
PR 1180; this companion handles dense VLM checkpoints that ship MTP
heads (mtp_num_hidden_layers > 0).

It adds:

* a Multi-Token Prediction head (``MTPModule``) to
  ``mlx_vlm.models.qwen3_5.language.LanguageModel`` when the config
  declares ``mtp_num_hidden_layers > 0`` and the process-wide MTP active
  flag is on;
* a ``return_hidden=True`` mode on ``LanguageModel.__call__`` that
  returns ``(logits, pre_norm_hidden, gdn_states)``.

Outer ``Model.sanitize`` is already patched separately by
``qwen35_vlm_model.py`` (MTP-key preservation + norm +1 shift), so no
sanitize work is needed here.

The decoder-graph classes (``Qwen3_5DecoderLayer``, ``Qwen3_5Attention``,
``Qwen3_5MLP``, ``Qwen3_5GatedDeltaNet``) are not modified. SSM rollback
on draft rejection uses mlx-vlm's stock
``LanguageModel.rollback_speculative_cache(...)`` which already exists
and consumes the ``gdn_states`` returned from this patched ``__call__``.

Apply ordering: this patch must run *before* ``mlx_vlm.utils.load(...)``
so the patched ``LanguageModel.__init__`` runs, and *before*
``omlx/patches/gated_delta_advance.py`` overrides
``Qwen3_5GatedDeltaNet.__call__``. ``maybe_apply_pre_load_patches`` in
``omlx/utils/model_loading.py`` calls ``apply_mlx_vlm_mtp_runtime_patch``
ahead of both, satisfying the ordering for inference. The oQ path in
``omlx/oq.py:_measure_sensitivity`` also calls it before
``vlm_load_model`` for sensitivity measurement.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_APPLIED = False


def apply() -> bool:
    """Apply the mlx-vlm Qwen3.5 (dense) runtime MTP patches. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True

    try:
        from mlx_vlm.models.qwen3_5 import config as q35_config
        from mlx_vlm.models.qwen3_5 import language as q35_lang
    except Exception as e:
        logger.debug(f"mlx_vlm.qwen3_5 not importable for MTP runtime: {e}")
        return False

    _patch_text_config(q35_config)
    _register_mtp_classes_for_vlm(q35_lang)
    _patch_vlm_language_model(q35_lang)
    # VLMModelAdapter pass-throughs are installed by the MoE runtime patch
    # too; the function is idempotent so calling it twice is safe.
    _patch_vlm_model_adapter()

    _APPLIED = True
    logger.info("mlx-vlm Qwen3.5 (dense) runtime MTP patch applied")
    return True


# ---------------------------------------------------------------------------
# TextConfig — retain mtp_num_hidden_layers as instance attribute.
# ---------------------------------------------------------------------------

def _patch_text_config(q35_config: Any) -> None:
    """Wrap ``TextConfig.from_dict`` so ``mtp_num_hidden_layers`` survives.

    mlx-vlm's ``BaseModelConfig.from_dict`` filters incoming params by the
    dataclass signature, dropping any key that isn't a declared field —
    including ``mtp_num_hidden_layers``. Without it the MTP head can't be
    sized; with it, ``LanguageModel.__init__`` knows to attach a head.
    """
    cls = q35_config.TextConfig
    if getattr(cls, "_omlx_mtp_from_dict_patched", False):
        return

    original_from_dict = cls.from_dict.__func__  # unwrap classmethod

    def patched_from_dict(cls_inner, params):
        instance = original_from_dict(cls_inner, params)
        if params:
            instance.mtp_num_hidden_layers = int(
                params.get("mtp_num_hidden_layers", 0) or 0
            )
        else:
            instance.mtp_num_hidden_layers = 0
        return instance

    cls.from_dict = classmethod(patched_from_dict)
    cls._omlx_mtp_from_dict_patched = True


# ---------------------------------------------------------------------------
# MTPDecoderLayer + MTPModule — dense VLM classes.
# ---------------------------------------------------------------------------

def _register_mtp_classes_for_vlm(q35_lang: Any) -> None:
    """Attach ``MTPDecoderLayer`` / ``MTPModule`` to the mlx-vlm qwen3_5
    language module. Dense uses ``Qwen3_5MLP`` (no MoE branch)."""
    if hasattr(q35_lang, "MTPModule"):
        return

    Attention = q35_lang.Qwen3_5Attention
    MLP = q35_lang.Qwen3_5MLP
    from mlx_vlm.models.qwen3_5.language import create_attention_mask

    class MTPDecoderLayer(nn.Module):
        """Full-attention transformer layer used inside the dense MTP head."""

        def __init__(self, args):
            super().__init__()
            self.self_attn = Attention(args)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

        def __call__(self, x, mask=None, cache=None, position_ids=None):
            r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
            h = x + r
            return h + self.mlp(self.post_attention_layernorm(h))

    class MTPModule(nn.Module):
        """Multi-Token Prediction head (mlx-lm PR 990) for dense VLM Qwen3.5/3.6.

        Predicts token t+2 by fusing the backbone pre-norm hidden state at
        position t with the embedding of the sampled main token t+1.
        """

        def __init__(self, args):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.pre_fc_norm_embedding = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.layers = [
                MTPDecoderLayer(args) for _ in range(args.mtp_num_hidden_layers)
            ]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, hidden_states, next_token_ids, embed_tokens, cache=None):
            embeds = embed_tokens(next_token_ids)
            e = self.pre_fc_norm_embedding(embeds)
            h = self.pre_fc_norm_hidden(hidden_states)
            fused = self.fc(mx.concatenate([e, h], axis=-1))

            if cache is None:
                cache = [None] * len(self.layers)

            mask = create_attention_mask(fused, cache[0] if cache else None)
            for layer, c in zip(self.layers, cache):
                fused = layer(fused, mask, c)

            return self.norm(fused)

    q35_lang.MTPDecoderLayer = MTPDecoderLayer
    q35_lang.MTPModule = MTPModule


# ---------------------------------------------------------------------------
# LanguageModel — wrap __init__, support return_hidden, add mtp_forward/cache.
# ---------------------------------------------------------------------------

def _patch_vlm_language_model(q35_lang: Any) -> None:
    cls = q35_lang.LanguageModel
    if "_omlx_mtp_runtime_patched" in cls.__dict__:
        return

    from mlx_lm.models.cache import KVCache

    original_init = cls.__init__
    original_call = cls.__call__

    def __init__(self, args, config=None):
        original_init(self, args, config)
        # Always attach MTPModule when the config declares MTP heads, so
        # mlx-vlm's load_weights (which skips Model.sanitize for is_mlx_format
        # checkpoints) can place the persisted mtp.* tensors. Whether MTP
        # speculative decode is actually invoked at inference time is gated
        # downstream by ``mlx_lm_mtp.batch_generator._is_mtp_eligible``,
        # which checks the process-wide ``is_mtp_active`` flag.
        # Without this unconditional attach, mtp_enabled=False would fail
        # VLM load with "Received N parameters not in model" and the engine
        # pool would permanently downgrade the entry to BatchedEngine —
        # losing vision support.
        n_mtp = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
        if n_mtp > 0:
            self.mtp = q35_lang.MTPModule(args)

    def __call__(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs):
        """Backbone forward with optional MTP-cycle return shape.

        With ``return_hidden=True``, returns the triple
        ``(logits, pre_norm_hidden, gdn_states)`` for the speculative
        decode cycle. ``n_confirmed`` is accepted and discarded — the
        mlx-vlm path uses post-hoc ``rollback_speculative_cache`` instead
        of a confirmed/draft split.
        """
        return_hidden = kwargs.pop("return_hidden", False)
        kwargs.pop("n_confirmed", None)
        if not return_hidden:
            return original_call(self, inputs, inputs_embeds, mask, cache, **kwargs)

        # Passing any non-None ``capture_layer_ids`` makes stock
        # ``LanguageModel.__call__`` allocate ``hidden_sink`` AND ``gdn_sink``,
        # both of which the MTP cycle needs.
        last_layer_idx = len(self.model.layers) - 1
        out = original_call(
            self,
            inputs,
            inputs_embeds,
            mask,
            cache,
            capture_layer_ids=[last_layer_idx],
            **kwargs,
        )
        hidden_pre_norm = out.hidden_states[0]
        return out.logits, hidden_pre_norm, out.gdn_states

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache):
        mtp_out = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
        )
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(mtp_out)
        return self.lm_head(mtp_out)

    def make_mtp_cache(self):
        if hasattr(self, "mtp"):
            return [KVCache() for _ in self.mtp.layers]
        return []

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls._omlx_mtp_runtime_patched = True


# ---------------------------------------------------------------------------
# VLMModelAdapter — add MTP pass-through methods at runtime.
# ---------------------------------------------------------------------------

def _patch_vlm_model_adapter() -> None:
    """Extend ``omlx.models.vlm.VLMModelAdapter`` with MTP plumbing.

    Same setup as the MoE runtime patch — idempotent, so calling from
    both dense and MoE apply() is safe.
    """
    try:
        from omlx.models.vlm import VLMModelAdapter
    except Exception as e:
        logger.debug(f"VLMModelAdapter not importable: {e}")
        return

    if getattr(VLMModelAdapter, "_omlx_mtp_adapter_patched", False):
        return

    @property
    def mtp(self):
        return getattr(self._language_model, "mtp", None)

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache):
        return self._language_model.mtp_forward(
            hidden_states, next_token_ids, mtp_cache
        )

    def make_mtp_cache(self):
        if hasattr(self._language_model, "make_mtp_cache"):
            return self._language_model.make_mtp_cache()
        return []

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        return self._language_model.rollback_speculative_cache(
            caches, gdn_states, accepted, block_size
        )

    VLMModelAdapter.mtp = mtp
    VLMModelAdapter.mtp_forward = mtp_forward
    VLMModelAdapter.make_mtp_cache = make_mtp_cache
    VLMModelAdapter.rollback_speculative_cache = rollback_speculative_cache
    VLMModelAdapter._omlx_mtp_adapter_patched = True
