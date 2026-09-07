# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP head attachment for the mlx-vlm Qwen3.5 (dense) VLM path.

Mirror of ``qwen35_moe_vlm_runtime.py`` for the dense Qwen3.5/3.6 family
(model_type=qwen3_5, e.g. Qwen3.6-27B). The MoE variant was wired up in
PR 1180; this companion handles dense VLM checkpoints that ship MTP
heads (mtp_num_hidden_layers > 0).

It adds:

* a Multi-Token Prediction head (``MTPModule``) to
  ``mlx_vlm.models.qwen3_5.language.LanguageModel`` when the config
  declares ``mtp_num_hidden_layers > 0`` and the checkpoint has MTP
  weights to bind;
* a ``return_hidden=True`` mode on ``LanguageModel.__call__`` that
  returns ``(logits, pre_norm_hidden, gdn_states)``.

Outer ``Model.sanitize`` is already patched separately by
``qwen35_vlm_model.py`` (MTP-key preservation + norm +1 shift). This patch
also normalizes root ``mtp.*`` keys at weight binding because mlx-vlm skips
``Model.sanitize`` for MLX-format shards.

The decoder-graph classes (``Qwen3_5DecoderLayer``, ``Qwen3_5Attention``,
``Qwen3_5MLP``, ``Qwen3_5GatedDeltaNet``) are not modified. SSM rollback
on draft rejection uses mlx-vlm's stock
``LanguageModel.rollback_speculative_cache(...)`` which already exists
and consumes the ``gdn_states`` returned from this patched ``__call__``.

Apply ordering: this patch must run *before* ``mlx_vlm.utils.load(...)``
so the patched ``LanguageModel.__init__`` runs. ``maybe_apply_pre_load_patches`` in
``omlx/utils/model_loading.py`` calls ``apply_mlx_vlm_mtp_runtime_patch``
before loading the model, satisfying the ordering for inference. The oQ path in
``omlx/oq.py:_measure_sensitivity`` also calls it before
``vlm_load_model`` for sensitivity measurement.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..mlx_lm_mtp import prompt_priming

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
    _patch_inner_model_capture(q35_lang)
    # VLMModelAdapter pass-throughs are installed by the MoE runtime patch
    # too; the function is idempotent so calling it twice is safe.
    _patch_vlm_model_adapter()
    _patch_vlm_outer_model_load_weights()

    _APPLIED = True
    logger.info("mlx-vlm Qwen3.5 (dense) runtime MTP patch applied")
    return True


def _patch_vlm_outer_model_load_weights() -> None:
    """Normalize MTP paths and repair norms in ``Model.load_weights`` payloads.

    Third-party MLX checkpoints can retain the head at root ``mtp.*`` while
    the runtime attaches it at ``language_model.mtp``. Older oQ conversions
    also stored the head's q_norm/k_norm/mtp.norm one below the correct MLX
    value. Both operations are idempotent-safe on canonical checkpoints.
    """
    try:
        from mlx_vlm.models import qwen3_5 as q35_outer
    except Exception as e:
        logger.debug(f"mlx_vlm outer qwen3_5 not importable: {e}")
        return

    cls = q35_outer.Model
    if getattr(cls, "_omlx_mtp_norm_repair_patched", False):
        return

    original_load_weights = cls.load_weights

    def load_weights(self, weights, strict=True):
        from ..mlx_lm_mtp.norm_repair import repair_legacy_head_norms

        weights = _remap_root_mtp_weights(self, weights)
        try:
            weights, _ = repair_legacy_head_norms(weights)
        except Exception:
            logger.warning("MTP head-norm repair failed", exc_info=True)
        return original_load_weights(self, weights, strict=strict)

    cls.load_weights = load_weights
    cls._omlx_mtp_norm_repair_patched = True


def _remap_root_mtp_weights(model: Any, weights: Any) -> Any:
    """Map legacy root ``mtp.*`` weights onto the VLM language model.

    Some third-party MLX Qwen VLM checkpoints use canonical
    ``language_model.*`` paths for the backbone but retain the MTP head at
    root ``mtp.*``. mlx-vlm skips ``Model.sanitize`` for MLX-format shards,
    so the normal sanitizer remap never runs and strict loading rejects the
    head. Only rewrite when the runtime MTP module is actually attached.
    """
    language_model = getattr(model, "language_model", None)
    if language_model is None or getattr(language_model, "mtp", None) is None:
        return weights

    if not isinstance(weights, list):
        return weights

    keys = {key for key, _ in weights if isinstance(key, str)}
    remapped = []
    remapped_count = 0
    for key, value in weights:
        if isinstance(key, str) and key.startswith("mtp."):
            target = "language_model." + key
            if target in keys:
                raise ValueError(
                    "Checkpoint contains both root and canonical MTP weights: "
                    f"{key!r} and {target!r}"
                )
            key = target
            remapped_count += 1
        remapped.append((key, value))

    if not remapped_count:
        return weights

    logger.info(
        "Remapped %d root mtp.* weight(s) to language_model.mtp.*",
        remapped_count,
    )
    return remapped


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
        from . import is_mtp_attach_enabled
        from ..mlx_lm_mtp import is_mtp_active

        if type(self) is not cls:
            # Subclasses (e.g. the qwen4_exp vendor LanguageModel, which
            # inherits from this class) own their MTP wiring; running the
            # Qwen3.5 attach here would bolt a q35 MTPModule onto a
            # foreign architecture (issue #2972, reverse leg).
            return original_init(self, args, config)
        original_init(self, args, config)
        # Attach MTPModule when the config declares MTP heads so mlx-vlm's
        # load_weights (which skips Model.sanitize for is_mlx_format
        # checkpoints) can place the persisted mtp.* tensors. Whether MTP
        # speculative decode is actually invoked at inference time is gated
        # downstream by ``mlx_lm_mtp.batch_generator._is_mtp_eligible``,
        # which checks the per-instance ``_omlx_mtp_decode_enabled`` marker.
        #
        # Gated by ``is_mtp_attach_enabled()`` so checkpoints that declare
        # mtp_num_hidden_layers > 0 but ship no mtp.* weights (unsloth
        # Qwen3.6 UD MLX builds, issue #1426) don't fail strict load_weights
        # with "Missing N parameters" and silently downgrade to LLM.
        n_mtp = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
        attach_enabled = bool(is_mtp_attach_enabled())
        self._omlx_mtp_decode_enabled = bool(
            n_mtp > 0 and attach_enabled and is_mtp_active()
        )
        if n_mtp > 0 and attach_enabled:
            self.mtp = q35_lang.MTPModule(args)
        if self._omlx_mtp_decode_enabled:
            # Depth-k chained drafting works on this path: mtp_forward
            # supports return_hidden below, and rollback uses mlx-vlm's
            # stock rollback_speculative_cache (native partial accepts).
            from ..mlx_lm_mtp import get_mtp_depth

            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = get_mtp_depth()
            # Prompt-priming capture runs inside the inner Qwen3_5Model
            # forward, which has no reference back to this LanguageModel
            # (the mtp module / make_mtp_cache live here). A weakref avoids
            # a tracked module cycle in the nn.Module tree.
            self.model._omlx_mtp_prime_host = weakref.ref(self)

    def __call__(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs):
        """Backbone forward with optional MTP-cycle return shape.

        With ``return_hidden=True``, returns ``LanguageModelOutput`` with
        pre-norm hidden states for the speculative decode cycle. ``n_confirmed``
        is accepted and discarded — the mlx-vlm path uses post-hoc
        ``rollback_speculative_cache`` instead of a confirmed/draft split.
        """
        if type(self) is not cls:
            # Subclasses (qwen4_exp vendor) reach here via super().__call__
            # and implement their own return_hidden / capture contract.
            # Reshaping their kwargs here collapses the raw hyper-stream
            # hidden that Qwen4 Lightning MTP requires (issue #2972).
            return original_call(self, inputs, inputs_embeds, mask, cache, **kwargs)
        return_hidden = kwargs.pop("return_hidden", False)
        return_shared_kv = kwargs.pop("return_shared_kv", False)
        kwargs.pop("n_confirmed", None)
        if not return_hidden:
            return original_call(self, inputs, inputs_embeds, mask, cache, **kwargs)

        # Passing any non-None ``capture_layer_ids`` makes stock
        # ``LanguageModel.__call__`` allocate ``hidden_sink`` AND ``gdn_sink``,
        # both of which the MTP cycle needs. Pop any existing value from kwargs
        # to avoid "got multiple values for keyword argument" when the caller
        # already passed capture_layer_ids.
        kwargs.pop("capture_layer_ids", None)
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
        from mlx_vlm.models.base import LanguageModelOutput

        hidden_pre_norm = out.hidden_states[0]
        return LanguageModelOutput(
            logits=out.logits,
            hidden_states=[hidden_pre_norm],
            gdn_states=out.gdn_states,
            shared_kv_states={} if return_shared_kv else None,
        )

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        """MTP-head forward (see mlx_lm_mtp.qwen35_model for the depth-k
        chain contract: return_hidden yields the head's post-norm hidden for
        chaining; logits_keep limits the lm_head to the last N positions)."""
        mtp_out = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
        )
        logits_source = mtp_out
        if logits_keep and logits_source.shape[1] > logits_keep:
            logits_source = logits_source[:, -logits_keep:, :]
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(logits_source)
        else:
            logits = self.lm_head(logits_source)
        if return_hidden:
            return logits, mtp_out
        return logits

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
# Inner Qwen3_5Model — prompt-priming capture on prefill/decode forwards.
# ---------------------------------------------------------------------------

def _patch_inner_model_capture(q35_lang: Any) -> None:
    """Wrap ``Qwen3_5Model.__call__`` to fold prompt chunks into the MTP head.

    The inner model's return value is the trunk-normed hidden for every
    position of the forward — exactly what the head history fold needs — and
    scheduler prefill reaches it via the outer ``LanguageModel.__call__``
    delegate, so this single wrap covers external prefill, chunked prefill
    and the seam decode steps.

    Skips: image/recursive calls (``inputs_embeds``), MTP verify and other
    capture forwards (any of ``capture_layer_ids`` / ``hidden_sink`` /
    ``gdn_sink``), unknown extra positional call shapes, and hosts without
    an MTP head (weakref unset). All real gating lives in
    ``prompt_priming.maybe_capture`` and fails safe to unprimed.
    """
    cls = q35_lang.Qwen3_5Model
    if getattr(cls, "_omlx_mtp_prime_capture_patched", False):
        return

    original_call = cls.__call__

    def __call__(
        self, inputs, inputs_embeds=None, mask=None, cache=None, *args, **kwargs
    ):
        out = original_call(
            self, inputs, inputs_embeds, mask, cache, *args, **kwargs
        )
        if (
            inputs_embeds is None
            and cache is not None
            and not args
            and kwargs.get("capture_layer_ids") is None
            and kwargs.get("hidden_sink") is None
            and kwargs.get("gdn_sink") is None
        ):
            host_ref = getattr(self, "_omlx_mtp_prime_host", None)
            host = host_ref() if host_ref is not None else None
            if host is not None:
                try:
                    prompt_priming.maybe_capture(host, inputs, out, cache)
                except Exception:
                    logger.debug(
                        "MTP prompt-priming capture failed", exc_info=True
                    )
        return out

    cls.__call__ = __call__
    cls._omlx_mtp_prime_capture_patched = True


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

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        # Forward the depth-k chain kwargs only when set so language models
        # whose mtp_forward predates them (MoE runtime) keep working.
        if return_hidden or logits_keep:
            return self._language_model.mtp_forward(
                hidden_states,
                next_token_ids,
                mtp_cache,
                return_hidden=return_hidden,
                logits_keep=logits_keep,
            )
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
