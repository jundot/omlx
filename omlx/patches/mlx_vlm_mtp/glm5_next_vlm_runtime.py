# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP head attachment for the mlx-vlm GLM-5.3-Flash (glm5_next) path.

Companion to ``omlx/patches/mlx_lm_mtp`` for an mlx-vlm-hosted architecture,
in the same shape as ``qwen35_moe_vlm_runtime`` and ``gemma4_vlm_runtime``.

``zai-org/GLM-5.3-Flash`` declares ``num_nextn_predict_layers: 1`` and stores
the MTP head DeepSeek-V3 style, as one extra decoder layer at
``model.language_model.layers.<num_hidden_layers>.*`` rather than ``mtp.*``.
Its fusion tensors are ``eh_proj`` / ``enorm`` / ``hnorm`` and its output norm
is ``shared_head.norm``, matching glm_moe_dsa.

The nextn layer carries no ``hc_attn_*`` / ``hc_ffn_*`` tensors while every
regular layer does, so the MTP block is a plain pre-norm decoder layer with
additive residuals. It does not use HyperConnection and runs on the ordinary
``(B, S, D)`` hidden state, not the backbone's ``(B, S, hc_mult, D)``.

Cache families, from ``LanguageModel.make_cache``:

* linear-attention layers use ``ArraysCache(size=2)`` holding conv state and
  gated-delta recurrent state. Neither is trimmable, so a verify forward
  records each layer's block inputs and entry state in ``gdn_sink`` and
  ``rollback_speculative_cache`` replays the accepted prefix through the same
  fused kernel.
* sparse-attention layers use ``CacheList(KVCache(), PoolingCache(...))``.
  Both members trim, the pooling half through the cross-boundary undo log in
  ``deepseek_v4/cache_extras.py``, so rollback only trims them.

Apply this before ``mlx_vlm.utils.load`` so the patched ``__init__`` runs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_APPLIED = False

# Source-side prefixes for the nextn MTP layer. glm5_next checkpoints use the
# VLM-nested form; the other two are accepted so a text-only re-export or a
# hand-rebased checkpoint still binds.
_NEXTN_PREFIXES = (
    "model.language_model.layers.{i}.",
    "language_model.model.layers.{i}.",
    "model.layers.{i}.",
)

# layers.45.<suffix> -> mtp.<n>.<renamed>. Anything not listed lands under
# ``block.`` (the decoder layer itself).
_MTP_FUSION_KEYS = {
    "eh_proj.weight": "eh_proj.weight",
    "enorm.weight": "enorm.weight",
    "hnorm.weight": "hnorm.weight",
    "shared_head.norm.weight": "norm.weight",
}

# Dropped outright: the MTP head reuses the trunk's embedding and lm_head.
_MTP_DROP_PREFIXES = ("shared_head.head.", "embed_tokens.")


def apply() -> bool:
    """Apply the glm5_next runtime MTP patches. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True

    try:
        from omlx.patches.mlx_vlm_glm5_next_compat import (
            apply_mlx_vlm_glm5_next_compat_patch,
        )

        # Registers mlx_vlm.models.glm5_next from oMLX's vendor tree and,
        # importantly, installs oMLX's PoolingCache into mlx_lm.models.cache.
        apply_mlx_vlm_glm5_next_compat_patch()
        from mlx_vlm.models.glm5_next import config as g5_config
        from mlx_vlm.models.glm5_next import language as g5_lang
    except Exception as e:  # noqa: BLE001
        logger.debug(f"mlx_vlm.glm5_next not importable for MTP runtime: {e}")
        return False

    try:
        from ..mlx_lm_mtp import cache_rollback

        # PoolingCache's undo log reads cache_rollback's armed flag.
        cache_rollback.apply()
    except Exception:  # noqa: BLE001
        logger.debug("cache_rollback.apply() failed", exc_info=True)

    _patch_text_config(g5_config)
    _register_mtp_classes(g5_lang)
    _patch_linear_attention(g5_lang)
    _patch_decoder_layer(g5_lang)
    _patch_model_call(g5_lang)
    _patch_language_model(g5_lang)
    _patch_vlm_model_adapter()

    _APPLIED = True
    logger.info("mlx-vlm GLM-5.3 (glm5_next) runtime MTP patch applied")
    return True


# ---------------------------------------------------------------------------
# TextConfig: keep num_nextn_predict_layers through from_dict.
# ---------------------------------------------------------------------------


def _patch_text_config(g5_config: Any) -> None:
    """``BaseModelConfig.from_dict`` drops keys that aren't declared fields.

    ``num_nextn_predict_layers`` IS declared on glm5_next's TextConfig
    (config.py:63, default 0), so unlike the Qwen3.5 case this is only a
    belt-and-braces re-read for checkpoints that nest it oddly.
    """
    cls = g5_config.TextConfig
    if getattr(cls, "_omlx_mtp_from_dict_patched", False):
        return

    original_from_dict = cls.from_dict.__func__

    def patched_from_dict(cls_inner, params):
        instance = original_from_dict(cls_inner, params)
        if params:
            n = params.get("num_nextn_predict_layers")
            if n is None:
                n = (params.get("text_config") or {}).get(
                    "num_nextn_predict_layers", 0
                )
            instance.num_nextn_predict_layers = int(n or 0)
        return instance

    cls.from_dict = classmethod(patched_from_dict)
    cls._omlx_mtp_from_dict_patched = True


# ---------------------------------------------------------------------------
# MTP block classes.
# ---------------------------------------------------------------------------


def _register_mtp_classes(g5_lang: Any) -> None:
    if hasattr(g5_lang, "Glm5NextMTPBlock"):
        return

    Glm5NextSparseAttention = g5_lang.Glm5NextSparseAttention
    Glm5NextMoE = g5_lang.Glm5NextMoE
    Glm5NextMLP = g5_lang.Glm5NextMLP

    class _MTPDecoderLayer(nn.Module):
        """``Glm5NextDecoderLayer`` minus HyperConnection.

        The checkpoint's layer 45 has no ``hc_attn_*``/``hc_ffn_*`` tensors,
        so the block uses plain additive residuals and stays 3-D. Attention
        is always the sparse/indexer variant. Layer 45 ships q_a/q_b/kv_a/
        kv_b/indexer, never the linear-attention tensor set.
        """

        def __init__(self, config: Any):
            super().__init__()
            self.self_attn = Glm5NextSparseAttention(config)
            n_routed = getattr(config, "n_routed_experts", None)
            self.mlp = Glm5NextMoE(config) if n_routed else Glm5NextMLP(config)
            self.input_layernorm = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        def __call__(self, x, mask=None, cache=None):
            x = x + self.self_attn(self.input_layernorm(x), mask, cache)
            return x + self.mlp(self.post_attention_layernorm(x))

    class Glm5NextMTPBlock(nn.Module):
        """enorm/hnorm fusion + one plain decoder layer + shared_head norm.

        ``norm`` holds the checkpoint's ``shared_head.norm``; ``mtp_forward``
        applies it and the shared ``lm_head`` so ``logits_keep`` can shrink
        the vocab matmul to the positions actually sampled.
        """

        def __init__(self, config: Any):
            super().__init__()
            dim = config.hidden_size
            self.enorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.hnorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.eh_proj = nn.Linear(2 * dim, dim, bias=False)
            self.norm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.block = _MTPDecoderLayer(config)

        def __call__(self, h, embed_tokens, input_ids, mask, cache):
            e = self.enorm(embed_tokens(input_ids))
            x = self.eh_proj(mx.concatenate([e, self.hnorm(h)], axis=-1))
            return self.block(x, mask, cache)

    g5_lang._MTPDecoderLayer = _MTPDecoderLayer
    g5_lang.Glm5NextMTPBlock = Glm5NextMTPBlock


# ---------------------------------------------------------------------------
# Linear attention: record verify-block inputs for rollback.
# ---------------------------------------------------------------------------


def _patch_linear_attention(g5_lang: Any) -> None:
    """Capture each KDA layer's verify-block inputs for post-hoc rollback.

    ``batch_generator._call_backbone`` selects the mlx-vlm rollback path only
    when the forward returns ``gdn_states``, so the sink is also the routing
    signal. Recording the block inputs and the entry state lets
    ``rollback_speculative_cache`` replay the accepted prefix on the fused
    kernel, which is cheaper on the verify path than capturing per step.

    Replaces ``__call__`` rather than wrapping it because the recorded
    tensors are locals of the stock body.

    Follows the approach in Blaizzy/mlx-vlm#2044.
    """
    cls = g5_lang.Glm5NextLinearAttention
    if getattr(cls, "_omlx_mtp_capture_patched", False):
        return

    _l2norm = g5_lang._l2norm
    linear_forward = g5_lang.linear_forward
    gated_delta_update = g5_lang.gated_delta_update

    def __call__(self, inputs, mask=None, cache=None, gdn_sink=None):
        B, S, _ = inputs.shape
        has_right_padding = cache is not None and cache.lengths is not None
        if has_right_padding:
            mask = mx.arange(S)[None] < cache.lengths[:, None]
        if self.fuse_in:
            q_o, k_o, v_o, fa_o, ga_o, b_o = self._fused_in_proj(inputs)
            mixed = mx.concatenate([q_o, k_o, v_o], axis=-1)
        else:
            mixed = mx.concatenate(
                [self.q_proj(inputs), self.k_proj(inputs), self.v_proj(inputs)],
                axis=-1,
            )
            fa_o = self.forget_gate.f_a_proj(inputs)
            ga_o = self.g_a_proj(inputs)
            b_o = self.b_proj(inputs)
        if mask is not None and mask.dtype == mx.bool_:
            mixed = mx.where(mask[..., None], mixed, 0)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        conv_input = mx.concatenate([conv_state, mixed], axis=1)
        if cache is not None:
            state_size = self.conv_kernel_size - 1
            if has_right_padding:
                valid_lengths = mx.sum(mask, axis=-1).astype(mx.int32)
                state_indices = valid_lengths[:, None] + mx.arange(state_size)[None]
                state_indices = mx.broadcast_to(
                    state_indices[..., None], (B, state_size, self.conv_dim)
                )
                cache[0] = mx.contiguous(
                    mx.take_along_axis(conv_input, state_indices, axis=1)
                )
            else:
                cache[0] = mx.contiguous(conv_input[:, -state_size:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.qkv_dim, 2 * self.qkv_dim], axis=-1)
        q = q.reshape(B, S, self.num_heads, self.head_dim)
        k = k.reshape(B, S, self.num_heads, self.head_dim)
        v = v.reshape(B, S, self.num_heads, self.head_dim)

        fg = self.forget_gate
        a = linear_forward(fg.f_b_proj, fa_o).reshape(
            B, S, self.num_heads, self.head_dim
        )
        in_dtype = q.dtype
        q = (_l2norm(q.astype(mx.float32)) * (self.head_dim**-0.5)).astype(in_dtype)
        k = _l2norm(k.astype(mx.float32)).astype(in_dtype)

        state = cache[1] if cache is not None else None
        A_log = fg.A_log.reshape(self.num_heads, 1)
        dt_bias = fg.dt_bias.reshape(self.num_heads, self.head_dim)
        lower_bound = fg.safe_gate_lower_bound
        if gdn_sink is not None:
            # Entry state + block inputs; rollback replays the accepted prefix.
            gdn_sink.append(
                (
                    q, k, v, a, b_o, A_log, dt_bias, state,
                    conv_input, self.conv_kernel_size, lower_bound,
                )
            )
        out, state = gated_delta_update(
            q, k, v, a, b_o, A_log, dt_bias,
            state=state,
            mask=mask if mask is not None and mask.dtype == mx.bool_ else None,
            lower_bound=lower_bound,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)

        gate = linear_forward(self.g_b_proj, ga_o).reshape(
            B, S, self.num_heads, self.head_dim
        )
        out = self.o_norm(out, gate).reshape(B, S, -1)
        return linear_forward(self.o_proj, out)

    cls.__call__ = __call__
    cls._omlx_mtp_capture_patched = True


def _patch_decoder_layer(g5_lang: Any) -> None:
    """Thread ``gdn_sink`` from the model down to the KDA layers only."""
    cls = g5_lang.Glm5NextDecoderLayer
    if getattr(cls, "_omlx_mtp_sink_patched", False):
        return

    original_call = cls.__call__

    def __call__(self, x, mask=None, cache=None, gdn_sink=None):
        if gdn_sink is None or not self.is_linear:
            return original_call(self, x, mask, cache)
        # Mirror the stock body, passing the sink into the KDA attention. The
        # FFN half is untouched (stateless, no recurrent state to roll back).
        residual = x
        xc, post, comb = self.attn_hc(x)
        r = self.self_attn(
            self.input_layernorm(xc), mask, cache, gdn_sink=gdn_sink
        )
        x = g5_lang.hc_expand(r, residual, post, comb)
        return self._ffn_block(x)

    cls.__call__ = __call__
    cls._omlx_mtp_sink_patched = True


# ---------------------------------------------------------------------------
# Glm5NextModel: expose the pre-norm hidden state.
# ---------------------------------------------------------------------------


def _patch_model_call(g5_lang: Any) -> None:
    """Add ``return_raw_hidden`` to ``Glm5NextModel.__call__``.

    The MTP head fuses the trunk's hidden state with the next token's
    embedding. Stock ``__call__`` only returns ``self.norm(h)``; the raw
    (pre-final-norm) activation is what chains cleanly across draft steps,
    and ``hnorm`` inside the block normalises whichever variant it gets.

    The hidden is taken after ``h.mean(axis=2)``, where the HyperConnection
    streams are already collapsed there, so it is an ordinary
    ``(B, S, D)`` tensor.
    """
    cls = g5_lang.Glm5NextModel
    if getattr(cls, "_omlx_mtp_call_patched", False):
        return

    create_attention_mask = g5_lang.create_attention_mask
    create_ssm_mask = g5_lang.create_ssm_mask

    def __call__(self, inputs, cache=None, inputs_embeds=None,
                 return_raw_hidden: bool = False, gdn_sink=None,
                 hidden_sink=None):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_cache = cache[self.fa_idx]
        fa_mask = create_attention_mask(
            h, fa_cache[0] if fa_cache else None, return_array=True
        )
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        h = mx.broadcast_to(
            h[:, :, None, :], (h.shape[0], h.shape[1], self.hc_mult, h.shape[2])
        )
        h = mx.contiguous(h)

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            if gdn_sink is not None and layer.is_linear:
                h = layer(h, mask=mask, cache=c, gdn_sink=gdn_sink)
            else:
                h = layer(h, mask=mask, cache=c)

        # Collapse the mHC streams first: everything downstream (the final
        # norm, the lm_head, and the nextn head) consumes the ordinary
        # (B, S, D) hidden, not the 4-D (B, S, hc_mult, D) internal form.
        h = h.mean(axis=2)
        if hidden_sink is not None:
            hidden_sink.append(h)  # pre-final-norm hidden for the nextn head
        out = self.norm(h)
        if return_raw_hidden:
            return out, h
        return out

    cls.__call__ = __call__
    cls._omlx_mtp_call_patched = True


# ---------------------------------------------------------------------------
# LanguageModel: attach head, return_hidden, mtp_forward, rollback, sanitize.
# ---------------------------------------------------------------------------


def _patch_language_model(g5_lang: Any) -> None:
    cls = g5_lang.LanguageModel
    if "_omlx_mtp_runtime_patched" in cls.__dict__:
        return

    from mlx_lm.models.cache import KVCache

    create_attention_mask = g5_lang.create_attention_mask

    original_init = cls.__init__
    original_call = cls.__call__
    original_sanitize = cls.sanitize

    def __init__(self, args, config=None):
        from . import is_mtp_attach_enabled
        from ..mlx_lm_mtp import get_mtp_depth, is_mtp_active

        original_init(self, args, config)

        n_mtp = int(getattr(args, "num_nextn_predict_layers", 0) or 0)
        attach_enabled = bool(is_mtp_attach_enabled())
        self._omlx_mtp_decode_enabled = bool(
            n_mtp > 0 and attach_enabled and is_mtp_active()
        )
        if n_mtp > 0 and attach_enabled:
            # A list (not a submodule attr) so weight paths are mtp.<i>.*,
            # matching how glm_moe_dsa binds its head.
            self.mtp = [g5_lang.Glm5NextMTPBlock(args) for _ in range(n_mtp)]
        if self._omlx_mtp_decode_enabled:
            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = get_mtp_depth()
            self._omlx_mtp_head_clone = False
            # Same 8-of-N routing economics as GLM-5.2: each extra verify row
            # pulls a nearly disjoint expert set, so the adaptive depth
            # controller needs a high marginal-cost prior or it over-drafts.
            self._omlx_mtp_marginal_ms = 35.0

    def __call__(self, inputs=None, inputs_embeds=None, cache=None, mask=None,
                 **kwargs):
        """Backbone forward, optionally returning the raw hidden for MTP.

        Note the argument order: glm5_next is ``(inputs, inputs_embeds,
        cache, mask)``, NOT the ``(inputs, inputs_embeds, mask, cache)`` used
        by the Qwen VLM runtimes. Positional forwarding between the two is a
        silent-corruption trap.
        """
        from mlx_vlm.models.base import LanguageModelOutput

        return_hidden = kwargs.pop("return_hidden", False)
        # Post-hoc rollback: the split-at-n_confirmed contract is not used on
        # this path (see _patch_linear_attention). Accept and discard, as the
        # Qwen VLM runtime does.
        kwargs.pop("n_confirmed", None)
        if inputs is None:
            inputs = kwargs.get("input_ids")

        if not return_hidden:
            out = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
            nlk = kwargs.get("num_logits_to_keep", 0)
            if nlk:
                out = out[:, -nlk:, :]
            if self.args.tie_word_embeddings:
                logits = self.model.embed_tokens.as_linear(out)
            else:
                logits = g5_lang.linear_forward(self.lm_head, out)
            return LanguageModelOutput(logits=logits)

        # MTP verify forward. Returning gdn_states is what routes
        # batch_generator._call_backbone onto the mlx-vlm rollback path
        # (rollback_speculative_cache); without it the engine silently takes
        # the mlx-lm path and never rewinds the KDA recurrent state.
        gdn_sink: list = []
        hidden_sink: list = []
        out = self.model(
            inputs, cache=cache, inputs_embeds=inputs_embeds,
            gdn_sink=gdn_sink, hidden_sink=hidden_sink,
        )
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = g5_lang.linear_forward(self.lm_head, out)
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=gdn_sink,
            shared_kv_states={},
        )

    def rollback_speculative_cache(self, caches, gdn_states, accepted,
                                   block_size):
        """Rewind every cache after a speculative round. Ported from PR #2044.

        KDA layers hold recurrent state with no trim semantics, so the
        accepted prefix is replayed through the same fused kernel from the
        stashed entry state. Sparse layers just trim: oMLX's own PoolingCache
        carries a cross-boundary undo log, so unlike the upstream PR there is
        no need to hand-roll the indexer pool rewind here.
        """
        gated_delta_update = g5_lang.gated_delta_update

        def _is_recurrent(cache) -> bool:
            # Match the family, not one class. glm5_next's linear layers use
            # ArraysCache or SizedArraysCache depending on the batch path, and
            # a SizedArraysCache falling through to the trim branch leaves its
            # recurrent state holding rejected tokens: the KV caches rewind,
            # the KDA state does not, and the drift compounds every round.
            # scheduler.py groups the same two names for this reason.
            return type(cache).__name__ in ("ArraysCache", "SizedArraysCache")

        if isinstance(accepted, int):
            acc = [int(accepted)]
        elif isinstance(accepted, mx.array):
            acc = [int(x) for x in accepted.reshape(-1).tolist()]
        else:
            acc = [int(x) for x in accepted]
        max_a = max(acc) if acc else 0
        n = max_a + 1
        trim = int(block_size) - n

        gdn_idx = 0
        for c in caches:
            if c is None:
                continue
            if _is_recurrent(c):
                if gdn_states is None or gdn_idx >= len(gdn_states):
                    logger.warning(
                        "glm5_next rollback: missing gdn state for KDA layer %d",
                        gdn_idx,
                    )
                    return 0
                (q_, k_, v_, a_, b_, A_log_, dt_bias_, init_state,
                 conv_input, K, lb) = gdn_states[gdn_idx]
                gdn_idx += 1
                _, state_n = gated_delta_update(
                    q_[:, :n], k_[:, :n], v_[:, :n], a_[:, :n], b_[:, :n],
                    A_log_, dt_bias_, state=init_state, lower_bound=lb,
                )
                c[1] = state_n
                c[0] = conv_input[:, n : n + K - 1]
                # The verify forward advanced this cache by the whole block.
                # Restoring the state does not undo that, and unlike the
                # sparse caches there is no trim() here to do it, so rewind
                # explicitly or the recurrent layers drift ahead of the KV
                # layers by `trim` on every partial accept.
                if trim > 0 and getattr(c, "offset", 0) >= trim:
                    c.offset -= trim
            elif trim > 0:
                if not c.is_trimmable():
                    logger.warning(
                        "glm5_next rollback: %s not trimmable", type(c).__name__
                    )
                    continue
                c.trim(trim)
        return max_a

    def make_mtp_cache(self):
        """One ``[KVCache, PoolingCache]`` pair per MTP block.

        The head's attention is the sparse/indexer variant, so it needs the
        same cache shape ``make_cache`` builds for a sparse backbone layer:
        latent KV plus the indexer's pooling cache. A flat list (rather than
        a CacheList) keeps each cache's ``offset``/``trim`` directly visible
        to the chain's trim helper, matching glm_moe_dsa.
        """
        if not hasattr(self, "mtp"):
            return None
        from mlx_lm.models.cache import PoolingCache

        caches = []
        for blk in self.mtp:
            caches.append(KVCache())
            caches.append(PoolingCache(blk.block.self_attn.indexer.index_kpool))
        return caches

    def mtp_forward(self, h, input_ids, cache=None, return_hidden: bool = False,
                    logits_keep: int = 0):
        """Run the MTP block(s), then shared_head norm + shared lm_head.

        ``h`` is the trunk's raw hidden for the first fold, or the head's own
        output for chained draft steps; ``hnorm`` normalises either.
        """
        if not hasattr(self, "mtp"):
            raise RuntimeError("mtp_forward called on a model with no MTP head")
        if cache is None:
            cache = [None] * (2 * len(self.mtp))

        mask = create_attention_mask(h, cache[0], return_array=True)

        last = None
        for i, blk in enumerate(self.mtp):
            pair = cache[2 * i : 2 * i + 2]
            if pair and pair[0] is None:
                pair = None
            h = blk(h, self.model.embed_tokens, input_ids, mask, pair)
            last = blk

        logits_source = h
        if logits_keep and logits_source.shape[1] > logits_keep:
            logits_source = logits_source[:, -logits_keep:]
        normed = last.norm(logits_source)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = g5_lang.linear_forward(self.lm_head, normed)
        if return_hidden:
            return logits, h
        return logits

    def sanitize(self, weights):
        """Preserve the nextn MTP layer and rebind it under ``mtp.<i>.*``.

        Stock ``glm5_next.LanguageModel.sanitize`` drops keys containing
        ``mtp.``, then calls ``DSV32Model.sanitize`` whose own MTP filter keys
        on ``parts[1] == "layers"`` and so never matches the VLM-nested
        ``model.language_model.layers.<n>.*``. Without this the head survives
        under a layer index the model does not build and strict
        ``load_weights`` rejects it.

        Lifts the nextn tensors out, rewrites them onto layer 0, runs them
        through the stock sanitize in isolation so they receive the same FP8
        dequant, expert stacking, indexer fusion and ``kv_b_proj`` absorption
        as the backbone, then rebinds them under ``mtp.<i>.*``.
        """
        n_mtp = int(getattr(self.args, "num_nextn_predict_layers", 0) or 0)
        n_main = int(getattr(self.args, "num_hidden_layers", 0) or 0)
        if n_mtp <= 0 or n_main <= 0:
            return original_sanitize(self, weights)

        # Three-way split. ``already`` is the case that only shows up on a
        # checkpoint this patch already converted: the head is stored as
        # ``mtp.*``, and stock ``glm5_next.LanguageModel.sanitize`` opens with
        #     weights = {k: v for k, v in weights.items() if "mtp." not in k}
        # so handing those tensors to it deletes the whole head and strict
        # load_weights then fails with "Missing N parameters: ...mtp.0...".
        # Raw HF checkpoints never hit this because their head is nextn-named.
        nextn, backbone, already = {}, {}, {}
        for k, v in weights.items():
            if _is_mtp_named(k):
                already[k] = v
                continue
            idx, rest = _match_nextn(k, n_main, n_mtp)
            if idx is None:
                backbone[k] = v
            elif not rest.startswith(_MTP_DROP_PREFIXES):
                nextn.setdefault(idx, {})[rest] = v

        out = original_sanitize(self, backbone)

        if already:
            # Re-merge untouched, but apply the same fp32 rule the stock pass
            # applies to the backbone so the head's router tensors match.
            for k, v in already.items():
                if (
                    (k.endswith("mlp.gate.weight")
                     or k.endswith("e_score_correction_bias"))
                    and hasattr(v, "dtype")
                    and mx.issubdtype(v.dtype, mx.floating)
                    and v.dtype != mx.float32
                ):
                    v = v.astype(mx.float32)
                out[k] = v
            logger.info(
                "glm5_next sanitize: preserved %d pre-converted mtp.* tensors",
                len(already),
            )
        if not nextn:
            logger.info("glm5_next sanitize: no nextn MTP layer in checkpoint")
            return out

        for idx, tensors in sorted(nextn.items()):
            fusion = {}
            block = {}
            for rest, v in tensors.items():
                if rest in _MTP_FUSION_KEYS:
                    fusion[_MTP_FUSION_KEYS[rest]] = v
                else:
                    block[f"model.layers.0.{rest}"] = v
            # Run the block through the stock pass as if it were layer 0.
            cooked = original_sanitize(self, block)
            for k, v in cooked.items():
                marker = ".layers.0."
                if marker not in k:
                    continue
                out[f"mtp.{idx}.block.{k.split(marker, 1)[1]}"] = v
            for k, v in fusion.items():
                out[f"mtp.{idx}.{k}"] = v

        logger.info(
            "glm5_next sanitize: bound %d nextn MTP layer(s) as mtp.*", len(nextn)
        )
        return out

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.rollback_speculative_cache = rollback_speculative_cache
    cls.sanitize = sanitize
    cls._omlx_mtp_runtime_patched = True


def _is_mtp_named(key: str) -> bool:
    """True for a head already stored under ``mtp.*``, i.e. a converted model."""
    return key.startswith("mtp.") or ".mtp." in key


def _match_nextn(key: str, n_main: int, n_mtp: int):
    """Return ``(mtp_index, suffix)`` if *key* is a nextn tensor, else (None, '')."""
    for i in range(n_mtp):
        for tmpl in _NEXTN_PREFIXES:
            prefix = tmpl.format(i=n_main + i)
            if key.startswith(prefix):
                return i, key[len(prefix) :]
    return None, ""


# ---------------------------------------------------------------------------
# VLMModelAdapter pass-throughs.
# ---------------------------------------------------------------------------


def _patch_vlm_model_adapter() -> None:
    """Expose the MTP surface through omlx's VLM adapter.

    ``qwen35_moe_vlm_runtime`` installs the same set when it runs first, so
    each attribute is added only when absent and this module works in either
    order.
    """
    try:
        from omlx.models.vlm import VLMModelAdapter
    except Exception as e:  # noqa: BLE001
        logger.debug(f"VLMModelAdapter not importable: {e}")
        return

    if getattr(VLMModelAdapter, "_omlx_glm5_mtp_adapter_patched", False):
        return

    if not hasattr(VLMModelAdapter, "mtp"):

        @property
        def mtp(self):
            return getattr(self._language_model, "mtp", None)

        VLMModelAdapter.mtp = mtp

    if not hasattr(VLMModelAdapter, "mtp_forward"):

        def mtp_forward(self, hidden_states, next_token_ids, mtp_cache,
                        return_hidden: bool = False, logits_keep: int = 0):
            return self._language_model.mtp_forward(
                hidden_states, next_token_ids, mtp_cache,
                return_hidden=return_hidden, logits_keep=logits_keep,
            )

        VLMModelAdapter.mtp_forward = mtp_forward

    if not hasattr(VLMModelAdapter, "make_mtp_cache"):

        def make_mtp_cache(self):
            if hasattr(self._language_model, "make_mtp_cache"):
                return self._language_model.make_mtp_cache()
            return []

        VLMModelAdapter.make_mtp_cache = make_mtp_cache


    if not hasattr(VLMModelAdapter, "rollback_speculative_cache"):

        def rollback_speculative_cache(self, caches, gdn_states, accepted,
                                       block_size):
            return self._language_model.rollback_speculative_cache(
                caches, gdn_states, accepted, block_size
            )

        VLMModelAdapter.rollback_speculative_cache = rollback_speculative_cache
    VLMModelAdapter._omlx_glm5_mtp_adapter_patched = True
