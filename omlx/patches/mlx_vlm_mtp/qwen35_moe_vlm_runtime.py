# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP head attachment for the mlx-vlm Qwen3.5-MoE VLM path.

This module is the mlx-vlm-side companion to ``omlx/patches/mlx_lm_mtp``.
It adds:

* a Multi-Token Prediction head (``MTPModule``) to
  ``mlx_vlm.models.qwen3_5_moe.language.LanguageModel`` when the model
  config declares ``mtp_num_hidden_layers > 0`` and the process-wide MTP
  active flag is on;
* a ``return_hidden=True`` mode on ``LanguageModel.__call__`` that returns
  ``(logits, pre_norm_hidden, gdn_states)`` — everything the MTP
  draft/verify cycle needs without touching the forward path of any
  decoder block;
* a ``sanitize`` extension that keeps ``mtp.*`` weights and converts the
  raw HF per-expert MoE layout to the ``switch_mlp.*`` layout mlx-vlm
  uses at runtime;
* matching pass-through methods on ``omlx.models.vlm.VLMModelAdapter`` so
  the engine can call ``model.mtp_forward(...)`` / ``model.make_mtp_cache()``
  and inspect ``model.mtp`` directly.

Critically, **no decoder-graph classes are modified**. The MTP draft
rollback for SSM/conv state is delegated to mlx-vlm's own
``LanguageModel.rollback_speculative_cache(...)``, which already exists
in upstream and consumes the ``gdn_states`` we return. This keeps the
diff against mlx-vlm small (LanguageModel constructor + __call__ wrap
+ sanitize) and avoids the brittle cross-stack n_confirmed plumbing
that an earlier iteration of this patch attempted.

Module-level apply ordering is significant: this patch must be applied
*before* the model loads (so the patched ``__init__`` runs) and
*before* ``omlx/patches/gated_delta_advance.py`` overrides
``Qwen3_5GatedDeltaNet.__call__``. The current loader (``omlx/utils/model_loading.py``)
calls ``apply_mlx_vlm_mtp_runtime_patch()`` in ``maybe_apply_pre_load_patches``
which satisfies both.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_APPLIED = False


def apply() -> bool:
    """Apply the mlx-vlm Qwen3.5-MoE runtime MTP patches. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True

    try:
        from mlx_vlm.models.qwen3_5_moe import config as q35moe_config
        from mlx_vlm.models.qwen3_5_moe import language as q35moe_lang
        from mlx_vlm.models.qwen3_5_moe import qwen3_5_moe as q35moe_outer
    except Exception as e:
        logger.debug(f"mlx_vlm.qwen3_5_moe not importable for MTP runtime: {e}")
        return False

    _patch_text_config(q35moe_config)
    _register_mtp_classes_for_vlm(q35moe_lang)
    _patch_vlm_language_model(q35moe_lang)
    _patch_vlm_outer_model_sanitize(q35moe_outer)
    _patch_vlm_model_adapter()

    _APPLIED = True
    logger.info("mlx-vlm Qwen3.5-MoE runtime MTP patch applied")
    return True


# ---------------------------------------------------------------------------
# TextConfig — retain mtp_num_hidden_layers as instance attribute.
# ---------------------------------------------------------------------------

def _patch_text_config(q35moe_config: Any) -> None:
    """Wrap ``TextConfig.from_dict`` so ``mtp_num_hidden_layers`` survives.

    mlx-vlm's ``BaseModelConfig.from_dict`` filters incoming params by the
    dataclass signature, dropping any key that isn't a declared field —
    including ``mtp_num_hidden_layers``. Without it the MTP head can't be
    sized; with it, ``LanguageModel.__init__`` knows to attach a head.
    """
    cls = q35moe_config.TextConfig
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
# MTPDecoderLayer + MTPModule — VLM-classes-based.
# ---------------------------------------------------------------------------

def _register_mtp_classes_for_vlm(q35moe_lang: Any) -> None:
    """Attach ``MTPDecoderLayer`` / ``MTPModule`` classes to the mlx-vlm
    qwen3_5_moe.language module so the language model can instantiate them
    later. The MTP head uses full attention only (no GatedDeltaNet)."""
    if hasattr(q35moe_lang, "MTPModule"):
        return

    from mlx_vlm.models.qwen3_5.language import (
        Qwen3_5Attention as MoeAttention,
        Qwen3_5MLP as MoeMLP,
        create_attention_mask,
    )

    SparseMoeBlock = q35moe_lang.Qwen3_5MoeSparseMoeBlock

    class MTPDecoderLayer(nn.Module):
        """Full-attention transformer layer used inside the VLM MTP head.

        Unlike the regular DecoderLayer (which switches between linear and
        full attention based on layer_idx), the MTP head only uses full
        attention. Honors MoE config when ``num_experts > 0``.
        """

        def __init__(self, args):
            super().__init__()
            self.self_attn = MoeAttention(args)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            if int(getattr(args, "num_experts", 0) or 0) > 0:
                self.mlp = SparseMoeBlock(args)
            else:
                self.mlp = MoeMLP(args.hidden_size, args.intermediate_size)

        def __call__(self, x, mask=None, cache=None, position_ids=None):
            r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
            h = x + r
            return h + self.mlp(self.post_attention_layernorm(h))

    class MTPModule(nn.Module):
        """Multi-Token Prediction head (mlx-lm PR 990) for VLM Qwen3.5-MoE.

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

    q35moe_lang.MTPDecoderLayer = MTPDecoderLayer
    q35moe_lang.MTPModule = MTPModule


# ---------------------------------------------------------------------------
# LanguageModel — wrap __init__, support return_hidden, add mtp_forward/cache.
# ---------------------------------------------------------------------------

def _patch_vlm_language_model(q35moe_lang: Any) -> None:
    cls = q35moe_lang.LanguageModel
    if "_omlx_mtp_runtime_patched" in cls.__dict__:
        return

    from mlx_lm.models.cache import KVCache

    original_init = cls.__init__
    original_call = cls.__call__

    def __init__(self, args, config=None):
        original_init(self, args, config)
        n_mtp = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
        try:
            from ..mlx_lm_mtp import is_mtp_active
            active = bool(is_mtp_active())
        except Exception:
            active = False
        if n_mtp > 0 and active:
            self.mtp = q35moe_lang.MTPModule(args)

    def __call__(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs):
        """Backbone forward with optional MTP-cycle return shape.

        With ``return_hidden=True``, returns the triple
        ``(logits, pre_norm_hidden, gdn_states)``:
        - ``pre_norm_hidden`` is the last-layer activation BEFORE the final
          RMSNorm; the MTP head fuses it with the next-token embedding.
        - ``gdn_states`` is the list of per-layer (q, k, v, a, b, A_log,
          dt_bias, state, mask, conv_input, conv_kernel_size) tuples
          captured by ``Qwen3_5GatedDeltaNet`` when a non-None
          ``capture_layer_ids`` is in flight. ``LanguageModel.rollback_speculative_cache``
          consumes this on draft rejection.

        ``n_confirmed`` is accepted and discarded — the mlx-vlm path does
        not need a confirmed/draft split because rollback is done after
        the fact via ``rollback_speculative_cache``.
        """
        return_hidden = kwargs.pop("return_hidden", False)
        kwargs.pop("n_confirmed", None)
        if not return_hidden:
            return original_call(self, inputs, inputs_embeds, mask, cache, **kwargs)

        # Passing any non-None ``capture_layer_ids`` makes stock
        # ``LanguageModel.__call__`` allocate ``hidden_sink`` AND ``gdn_sink``,
        # both of which we need.
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

    BatchGenerator's MTP path needs ``model.mtp_forward``,
    ``model.make_mtp_cache``, ``model.rollback_speculative_cache`` and
    ``model.mtp`` (used for eligibility detection). The adapter delegates
    each to the wrapped LanguageModel via ``getattr`` so callers see a
    uniform interface regardless of engine type.

    Notes on ``__call__``: the stock adapter already passes ``**kwargs``
    straight through to ``self._language_model(...)``, so a tuple return
    from the patched LanguageModel propagates correctly — its final
    ``hasattr(result, 'logits')`` check returns False for a tuple and
    returns the tuple as-is. No ``__call__`` wrap needed.
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


# ---------------------------------------------------------------------------
# Outer Model.sanitize — keep mtp.* keys and stack per-expert (incl. quant
# metadata) MoE MTP weights.
# ---------------------------------------------------------------------------

def _patch_vlm_outer_model_sanitize(q35moe_outer: Any) -> None:
    cls = q35moe_outer.Model
    if "_omlx_mtp_runtime_sanitize_patched" in cls.__dict__:
        return

    def _stack_per_expert(weights, prefix, num_experts):
        for n in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "scales", "biases"):
                key0 = f"{prefix}.experts.0.{n}.{suffix}"
                if key0 not in weights:
                    continue
                weights[f"{prefix}.switch_mlp.{n}.{suffix}"] = mx.stack(
                    [
                        weights.pop(f"{prefix}.experts.{e}.{n}.{suffix}")
                        for e in range(num_experts)
                    ]
                )

    def _unfuse_layer_experts(weights, prefix):
        gate_up_key = f"{prefix}.experts.gate_up_proj"
        if gate_up_key not in weights:
            return False
        gate_up = weights.pop(gate_up_key)
        gate_w, up_w = mx.split(gate_up, 2, axis=-2)
        weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_w
        weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_w
        down_key = f"{prefix}.experts.down_proj"
        if down_key in weights:
            weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(down_key)
        return True

    def sanitize(self, weights):
        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Backbone MoE: convert fused gate_up_proj → switch_mlp.{gate,up,down}.
        for l in range(self.config.text_config.num_hidden_layers):
            _unfuse_layer_experts(
                weights, f"model.language_model.layers.{l}.mlp"
            )

        # MTP MoE layers: discover via weight keys.
        # Two possible prefixes:
        #   ``mtp.layers.N.mlp...``  (raw HF format)
        #   ``language_model.mtp.layers.N.mlp...``  (already-MLX oQ output)
        def _discover_mtp_layers(prefix_root: str):
            return sorted({
                int(k[len(prefix_root):].split(".")[0])
                for k in weights
                if k.startswith(prefix_root)
                and k[len(prefix_root):].split(".")[0].isdigit()
            })

        num_experts = int(getattr(self.config.text_config, "num_experts", 0) or 0)
        for prefix_root in ("mtp.layers.", "language_model.mtp.layers."):
            for layer_idx in _discover_mtp_layers(prefix_root):
                prefix = f"{prefix_root}{layer_idx}.mlp"
                if f"{prefix}.switch_mlp.gate_proj.weight" in weights:
                    continue
                if not _unfuse_layer_experts(weights, prefix):
                    _stack_per_expert(weights, prefix, num_experts)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        )

        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and getattr(v, "shape", (1,))[-1] != 1
            for k, v in weights.items()
        )

        sanitized = {}
        for key, value in weights.items():
            if "model.language_model" in key:
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("model.visual"):
                key = key.replace("model.visual", "vision_tower")
            elif key.startswith("lm_head"):
                key = key.replace("lm_head", "language_model.lm_head")
            elif key.startswith("mtp."):
                key = "language_model." + key

            if "conv1d.weight" in key and value.shape[-1] != 1:
                # Use the module-level mx.moveaxis so it goes through the
                # streaming-discovery monkey-patch (in ``omlx.oq``) when
                # called with a ``_TrackedTensor`` placeholder. The instance
                # method on _TrackedTensor doesn't exist.
                value = mx.moveaxis(value, 2, 1)
            if has_unsanitized_conv1d and any(
                key.endswith(sfx) for sfx in norm_keys
            ):
                if value.ndim == 1:
                    value = value + 1.0

            sanitized[key] = value

        return sanitized

    cls.sanitize = sanitize
    cls._omlx_mtp_runtime_sanitize_patched = True
