# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (glm5_next, mlx-vlm vendored) native Lightning MTP head.

JANG-MTP checkpoints (GLM-5.3-Flash-JANG-MTP) declare
``num_nextn_predict_layers: 1`` and ship the draft as one extra decoder
layer (index ``num_hidden_layers``) with the DeepSeek-V3 nextn layout:
RMSNorm'd trunk hidden + next-token embedding (``hnorm``/``enorm``)
concatenated, projected back to ``hidden_size`` (``eh_proj``), run
through one full sparse-attention + 288-expert MoE decoder layer, and
read out through ``shared_head.norm`` + the shared ``lm_head``. That is
the same head shape as GLM-5.2's glm_moe_dsa nextn layer — this patch
mirrors ``glm_moe_dsa_model`` onto the vendored mlx-vlm module.

The compat patch (``mlx_vlm_glm5_next_compat``) registers
``mlx_vlm.models.glm5_next``; this patch sits on top and adds:

- ``Glm5NextMTPBlock``: enorm/hnorm/eh_proj fusion + a full
  ``Glm5NextDecoderLayer`` (synthesized sparse config entry).
- ``LanguageModel.__init__`` wrap: attach ``self.mtp`` when the load-time
  MTP flag is active; stamp the chain/depth markers that
  ``batch_generator._resolve_mtp_chain_depth`` reads.
- ``LanguageModel.__call__`` wrap: ``return_hidden=True`` returns the
  post-norm trunk hidden (GLM-5.2 convention — ``hnorm`` re-normalises
  inside the block) plus logits from ONE backbone pass.
- ``mtp_forward`` / ``make_mtp_cache`` / ``get_mtp_module``: the
  chain-cycle contract; the draft cache is CacheList(KVCache +
  PoolingCache), mirroring the trunk's sparse-attention layers.
- Linear-attention rollback: ``Glm5NextLinearAttention.__call__`` is
  replaced with an ``n_confirmed``-aware body that stashes the
  pre-forward recurrent state plus the verify window's projected inputs
  (``rollback_state`` / ``_mtp_draft_stash``, the qwen35 unsplit-verify
  pattern); ``mtp_partial_rollback`` replays the accepted prefix through
  the factored ``_process_chunk`` on rejection. DSA layers trim through
  PoolingCache's cross-boundary undo. ``mtp_clamp_accept`` bounds the
  accept count to what every layer can roll back (DeepSeek-V4 pattern).
- Prompt-priming capture: ``Glm5NextModel.__call__`` wrap folds prompt
  chunks into the head cache via ``prompt_priming.maybe_capture``.

Apply order: the compat patch must register the module first; this
patch's ``apply()`` no-ops when ``mlx_vlm.models.glm5_next`` is absent.
"""

from __future__ import annotations

import logging
import sys
import weakref
from typing import Any

logger = logging.getLogger(__name__)


def _is_our_method(cls: Any, attr: str, marker: str) -> bool:
    existing = cls.__dict__.get(attr)
    return getattr(existing, marker, False)


def _language_mod(glm: Any):
    return getattr(glm, "language", None) or __import__(
        "mlx_vlm.models.glm5_next.language", fromlist=["LanguageModel"]
    )


def apply() -> bool:
    """Apply the glm5_next MTP patches when the vendored module is present."""
    glm = sys.modules.get("mlx_vlm.models.glm5_next")
    if glm is None or not hasattr(glm, "LanguageModel"):
        logger.debug(
            "glm5_next module not registered; skipping MTP patch (expected "
            "for non-GLM-5.3 models)"
        )
        return False

    # The draft's DSA indexer needs PoolingCache (same class the trunk's
    # sparse layers use). The compat patch injects it too; this is a
    # belt-and-suspenders re-run for direct apply() orderings.
    try:
        from ..deepseek_v4 import _inject_cache_extras

        _inject_cache_extras()
    except Exception:
        logger.debug("PoolingCache injection unavailable", exc_info=True)

    lang = _language_mod(glm)
    _patch_linear_attention(lang)
    _patch_decoder_layer(lang)
    _patch_trunk_model(lang)
    _patch_language_model(glm)
    if not hasattr(glm.LanguageModel, "_omlx_mtp_patched"):
        glm.LanguageModel._omlx_mtp_patched = "patch"
        logger.info("GLM-5.3 (glm5_next) MTP model patch applied")
    return True


# ---------------------------------------------------------------------------
# Module-path counterpart of the wrapper sanitize's weight-key special map.
# ---------------------------------------------------------------------------

_MTP_QUANT_SPECIAL = {
    "eh_proj": "eh_proj",
    "enorm": "enorm",
    "hnorm": "hnorm",
    "shared_head.norm": "norm",
}


def remap_mtp_quant_overrides(
    params: dict[str, Any], n_main: int, n_mtp: int
) -> None:
    """Copy per-module quantization overrides to the runtime MTP paths.

    ``sanitize`` renames the nextn layer's config keys from
    ``model.layers.<n_main + i>.*`` to ``mtp.<i>.*`` but per-module
    quantization lookups key on runtime module paths, so the copies must
    ride along (same trick as glm_moe_dsa's ``_remap_mtp_quant_overrides``;
    mutating ``params`` in place is enough — it is the same dict the
    loader closes over).
    """
    quant = params.get("quantization") if isinstance(params, dict) else None
    if not isinstance(quant, dict):
        return
    for k, v in list(quant.items()):
        for i in range(n_mtp):
            prefix = f"model.layers.{n_main + i}."
            if not k.startswith(prefix):
                continue
            rest = k[len(prefix) :]
            if rest.startswith(("shared_head.head", "embed_tokens")):
                nk = None
            elif rest in _MTP_QUANT_SPECIAL:
                nk = f"mtp.{i}.{_MTP_QUANT_SPECIAL[rest]}"
            else:
                nk = f"mtp.{i}.block.{rest}"
            if nk is not None and nk not in quant:
                quant[nk] = v
            break


def _head_pool_adapter(pool: Any):
    """Expose the head PoolingCache through a token-offset contract.

    The chain's ``_mtp_head_trim_to`` counts history in TOKENS while the
    PoolingCache's native ``offset`` counts completed pool WINDOWS — mixing
    the units would make trims silently miss. The adapter keeps the same
    underlying object (state lives there) and routes every method through,
    but reports ``offset`` in tokens (``pool_len * ratio + remainder``) so
    ``_mtp_head_trim_to``'s token-space arithmetic stays correct.
    """
    ratio = int(pool.ratio)

    class _HeadPoolAdapter:
        def __init__(self):
            self._pool = pool

        def __getattr__(self, name):
            # Everything else (trim, is_trimmable, size, state, ...) is the
            # PoolingCache's own token-space API — delegate verbatim.
            return getattr(pool, name)

        @property
        def offset(self):
            return int(pool.size() * ratio + pool.remainder)

    return _HeadPoolAdapter()


def _mtp_layer_config(args: Any):
    """Synthesize the draft layer's config entry.

    The checkpoint's ``layer_types`` / ``mlp_layer_types`` lists span the
    trunk layers only; the JANG draft is a sparse-attention + MoE layer
    (DSA indexer weights + switch_mlp banks present). Extend both lists
    by one sparse entry on a copy so ``Glm5NextDecoderLayer`` can
    construct at index ``num_hidden_layers``.
    """
    import copy

    cfg = copy.copy(args)
    n = args.num_hidden_layers
    layer_types = list(getattr(args, "layer_types", None) or [])
    mlp_types = list(getattr(args, "mlp_layer_types", None) or [])
    while len(layer_types) < n + 1:
        layer_types.append("deepseek_sparse_attention")
    while len(mlp_types) < n + 1:
        mlp_types.append("sparse")
    cfg.layer_types = layer_types
    cfg.mlp_layer_types = mlp_types
    return cfg


# ---------------------------------------------------------------------------
# Glm5NextLinearAttention — n_confirmed-aware unsplit verify + rollback stash.
# Mirrors qwen35_model's GatedDeltaNet replacement: the whole verify window
# runs as ONE chunk (cheaper than splitwise across 33 linear layers), with
# zero-copy pre-forward state refs plus the projected inputs stashed on the
# ArraysCache. On rejection, mtp_partial_rollback replays the kept prefix
# through _process_chunk from those refs.
# ---------------------------------------------------------------------------


def _patch_linear_attention(lang: Any) -> None:
    cls = getattr(lang, "Glm5NextLinearAttention", None)
    if cls is None:
        logger.debug("Glm5NextLinearAttention missing; skip linear patch")
        return
    if getattr(cls, "_omlx_mtp_patched", False):
        return

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.glm5_next.gated_delta import gated_delta_update
    from mlx_vlm.models.glm5_next.linear import (
        fused_quantized_matmul,
        linear_forward,
    )

    def _l2norm(x, eps: float = 1e-6):
        return x * mx.rsqrt((x * x).sum(-1, keepdims=True) + eps)

    def _process_chunk(
        self,
        mixed,
        fa_o,
        ga_o,
        b_o,
        conv_state,
        ssm_state,
        mask,
    ):
        """Recurrent core from the projected inputs to the layer output.

        ``mixed`` is the fused q|k|v concat (B, S, 3*qkv_dim) ALREADY
        gated by the boolean mask (the caller applies mx.where); the rest
        are the forget-gate / output-gate projections. Returns
        ``(out, new_conv_state, new_ssm_state)`` without touching any
        cache — the caller owns cache writes so the verify path can stash
        and the replay path can run on snapshots.
        """
        B, S, _ = mixed.shape
        conv_dim = self.conv_dim
        if conv_state is None:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, conv_dim), dtype=mixed.dtype
            )
        conv_input = mx.concatenate([conv_state, mixed], axis=1)
        new_conv_state = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :, :])
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

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b_o,
            fg.A_log.reshape(self.num_heads, 1),
            fg.dt_bias.reshape(self.num_heads, self.head_dim),
            state=ssm_state,
            mask=mask if mask is not None and mask.dtype == mx.bool_ else None,
            lower_bound=fg.safe_gate_lower_bound,
        )
        gate = linear_forward(self.g_b_proj, ga_o).reshape(
            B, S, self.num_heads, self.head_dim
        )
        out = self.o_norm(out, gate).reshape(B, S, -1)
        return linear_forward(self.o_proj, out), new_conv_state, state

    def __call__(
        self,
        inputs,
        mask=None,
        cache=None,
        n_confirmed: int = 0,
    ):
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

        conv_state = (
            cache[0]
            if cache is not None and cache[0] is not None
            else None
        )
        ssm_state = (
            cache[1]
            if cache is not None and cache[1] is not None
            else None
        )
        # Right-padding batches (continuous batching) read conv_state from
        # the take_along_axis write path below, which concatenates it — a
        # fresh cache's empty slot must look like zeros there, exactly as
        # the stock body's guard does.
        conv_state_for_write = (
            conv_state
            if conv_state is not None
            else mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        )

        if n_confirmed > 0 and n_confirmed < S:
            # MTP verify forward (single chunk, qwen35-style): keep zero-copy
            # pre-forward state refs plus the projected inputs so a
            # rejection can replay the accepted prefix through
            # _process_chunk. Full accepts drop the refs (no replay cost).
            out, conv_f, ssm_f = _process_chunk(
                self, mixed, fa_o, ga_o, b_o, conv_state, ssm_state, mask
            )
            if cache is not None:
                cache.rollback_state = (conv_state, ssm_state)
                cache._mtp_draft_stash = (mixed, fa_o, ga_o, b_o, mask)
        else:
            out, conv_f, ssm_f = _process_chunk(
                self, mixed, fa_o, ga_o, b_o, conv_state, ssm_state, mask
            )

        if cache is not None:
            if has_right_padding:
                state_size = self.conv_kernel_size - 1
                valid_lengths = mx.sum(mask, axis=-1).astype(mx.int32)
                state_indices = valid_lengths[:, None] + mx.arange(state_size)[None]
                state_indices = mx.broadcast_to(
                    state_indices[..., None], (B, state_size, self.conv_dim)
                )
                cache[0] = mx.contiguous(
                    mx.take_along_axis(
                        mx.concatenate([conv_state_for_write, mixed], axis=1),
                        state_indices,
                        axis=1,
                    )
                )
            else:
                cache[0] = conv_f
            cache[1] = ssm_f
            cache.advance(S)

        return out

    cls._process_chunk = _process_chunk
    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__
    cls._omlx_mtp_patched = True


def _patch_decoder_layer(lang: Any) -> None:
    """Pass ``n_confirmed`` through to the linear-attention layer only.

    Sparse-attention layers ignore it (their rollback is trim-based).
    """
    cls = getattr(lang, "Glm5NextDecoderLayer", None)
    if cls is None or getattr(cls, "_omlx_mtp_patched", False):
        return

    import mlx.core as mx
    from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand

    original_call = cls.__call__

    def __call__(self, x, mask=None, cache=None, n_confirmed: int = 0):
        if n_confirmed and self.is_linear:
            residual = x
            xc, post, comb = self.attn_hc(x)
            r = self.self_attn(
                self.input_layernorm(xc), mask, cache, n_confirmed=n_confirmed
            )
            x = hc_expand(r, residual, post, comb)
            if self.compile_ffn and x.shape[0] == 1 and x.shape[1] == 1:
                if self._ffn_c is None:
                    self._ffn_c = mx.compile(self._ffn_block)
                out = self._ffn_c(x)
            else:
                out = self._ffn_block(x)
            if getattr(self, "_stream_eval", False):
                mx.eval(out)
                mx.clear_cache()
            return out
        return original_call(self, x, mask, cache)

    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__
    cls._omlx_mtp_patched = True


def _patch_trunk_model(lang: Any) -> None:
    """n_confirmed plumbing + prompt-priming capture on the trunk model.

    The stock ``Glm5NextModel.__call__`` loops the layers with
    ``layer(h, mask, c)``; this wrap re-runs the loop itself when
    ``n_confirmed`` is set so the linear layers get the verify-aware
    call, and folds prompt chunks into the MTP head cache through
    ``prompt_priming.maybe_capture`` otherwise.
    """
    cls = getattr(lang, "Glm5NextModel", None)
    if cls is None or getattr(cls, "_omlx_mtp_patched", False):
        return

    import mlx.core as mx
    import numpy as np
    from mlx_vlm.models.base import create_attention_mask, create_ssm_mask

    original_call = cls.__call__

    def __call__(self, inputs, cache=None, inputs_embeds=None, n_confirmed: int = 0, skip_capture: bool = False):
        if n_confirmed:
            # Verify forward: run the layer loop with n_confirmed reaching
            # the linear-attention layers (their unsplit-verify stash). The
            # expert-prefetch pilot is skipped here — verify windows are
            # S=k+1>1 and the stock loop only runs the pilot in decode (S=1);
            # the next decode fold re-arms it.
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
            for i, (layer, c) in enumerate(zip(self.layers, cache)):
                mask = ssm_mask if layer.is_linear else fa_mask
                h = layer(h, mask=mask, cache=c, n_confirmed=n_confirmed)
            h = h.mean(axis=2)
            return self.norm(h)

        out = original_call(self, inputs, cache=cache, inputs_embeds=inputs_embeds)
        if inputs_embeds is None and cache is not None and not skip_capture:
            host_ref = getattr(self, "_omlx_mtp_prime_host", None)
            host = host_ref() if host_ref is not None else None
            if host is not None:
                try:
                    from . import prompt_priming

                    prompt_priming.maybe_capture(host, inputs, out, cache)
                except Exception:
                    logger.debug("MTP prompt-priming capture failed", exc_info=True)
        return out

    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__
    cls._omlx_mtp_patched = True


# ---------------------------------------------------------------------------
# LanguageModel — head attach, return_hidden, chain contract, rollback.
# ---------------------------------------------------------------------------


def _patch_language_model(glm: Any) -> None:
    cls = glm.LanguageModel
    init_wrapped = getattr(cls, "_omlx_mtp_init_wrapped", False)
    call_owned = _is_our_method(cls, "__call__", "_omlx_mtp_call_marker")
    if init_wrapped and call_owned:
        return

    original_init = cls.__init__
    original_call = cls.__call__

    def __init__(self, args, config=None):
        original_init(self, args, config)
        n_mtp = int(getattr(args, "num_nextn_predict_layers", 0) or 0)
        from . import is_mtp_active

        mtp_decode_enabled = bool(n_mtp > 0 and is_mtp_active())
        self._omlx_mtp_decode_enabled = mtp_decode_enabled
        if mtp_decode_enabled:
            self.mtp = [_make_mtp_block(glm, _mtp_layer_config(args), args)]
            from . import get_mtp_depth

            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = get_mtp_depth()
            self._omlx_mtp_head_clone = False
            # Head input is the trunk's POST-final-norm hidden (same as
            # GLM-5.2's validated nextn head: hnorm re-normalises inside
            # the block; folded history and decode folds must match).
            self._omlx_mtp_head_prenorm = False
            # The return_hidden wrapper already applies the trunk's final
            # norm (GLM-5.2 head-input convention), so the chain's
            # ``_trunk_norm_module`` must be the identity — without this
            # marker the chain would norm the already-normed hidden twice.
            self._omlx_mtp_head_hidden_normed = True
            # Marginal-cost prior for the adaptive depth controller: the
            # draft's own MoE pulls a near-disjoint expert set per extra
            # verify row (same regime as GLM-5.2's 8-of-256 routing).
            self._omlx_mtp_marginal_ms = 35.0
            # Prompt-priming capture runs inside the trunk forward, which
            # has no reference back to this module; a weakref avoids a
            # tracked-module cycle.
            self.model._omlx_mtp_prime_host = weakref.ref(self)
            quant = getattr(args, "quantization", None)
            if isinstance(quant, dict):
                remap_mtp_quant_overrides(quant, int(args.num_hidden_layers), n_mtp)

    def __call__(
        self,
        inputs=None,
        inputs_embeds=None,
        cache=None,
        mask=None,
        **kwargs,
    ):
        return_hidden = bool(kwargs.pop("return_hidden", False))
        n_confirmed = int(kwargs.pop("n_confirmed", 0) or 0)
        if not return_hidden and not n_confirmed:
            return original_call(
                self,
                inputs,
                inputs_embeds=inputs_embeds,
                cache=cache,
                mask=mask,
                **kwargs,
            )
        if inputs is None:
            inputs = kwargs.get("input_ids")
        # One backbone pass, both products: the trunk returns the
        # post-norm hidden (the head's input variant); logits project from
        # the same pass's tail.
        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            n_confirmed=n_confirmed,
            skip_capture=bool(return_hidden or n_confirmed),
        )
        nlk = kwargs.get("num_logits_to_keep", 0)
        out_tail = out[:, -nlk:, :] if nlk else out
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out_tail)
        else:
            from mlx_vlm.models.glm5_next.language import linear_forward

            logits = linear_forward(self.lm_head, out_tail)
        from mlx_vlm.models.base import LanguageModelOutput

        if return_hidden:
            return LanguageModelOutput(logits=logits, hidden_states=out)
        return LanguageModelOutput(logits=logits)

    def get_mtp_module(self):
        return getattr(self, "mtp", None)

    def make_mtp_cache(self):
        """One CacheList(KVCache + PoolingCache) per draft block.

        Mirrors ``LanguageModel.make_cache``'s sparse-attention entry so
        the DSA indexer's pooling window advances identically in the draft.
        """
        from mlx_lm.models.cache import KVCache, PoolingCache

        caches = []
        for block in getattr(self, "mtp", None) or []:
            # FLAT list, not CacheList: the chain's ``_mtp_head_trim_to``
            # reads ``offset`` on each entry directly (CacheList has none,
            # so speculative rows would never trim and the head cache
            # would grow without bound). ``mtp_forward`` re-slices the
            # pairs per block below. Same layout as glm_moe_dsa's head
            # cache (flat KVCache pairs).
            caches.append(KVCache())
            caches.append(
                _head_pool_adapter(
                    PoolingCache(block.block.self_attn.indexer.index_kpool)
                )
            )
        return caches

    def mtp_forward(
        self,
        h,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        """Run the draft block(s) + shared_head norm + shared lm_head.

        ``h`` is the trunk's post-norm hidden for the history fold, or the
        head's own raw output for chained draft steps — ``hnorm`` inside
        the block normalises either. ``return_hidden`` returns the block's
        raw residual output (the next chain step's ``h``).
        """
        import mlx.core as mx

        if cache is None:
            cache = [None] * len(self.mtp)

        # Arm the PoolingCache undo log around the fold: the chain folds
        # committed history AND speculative drafts through this same
        # entry, and a rejected cycle trims the speculative tail via
        # ``_mtp_head_trim_to`` — which needs the undo stash those folds
        # left behind (it is only recorded while the flag is armed).
        from . import cache_rollback

        cache_rollback.set_undo_armed(True)
        try:
            last_block = None
            for i, block in enumerate(self.mtp):
                # Flat head-cache layout (see make_mtp_cache): entry 2*i is the
                # draft's KVCache, 2*i+1 its PoolingCache.
                layer_cache = (
                    cache[2 * i : 2 * i + 2]
                    if isinstance(cache, list) and len(cache) > 2 * i + 1
                    else (cache[i] if i < len(cache) else None)
                )
                h = block(h, self.model.embed_tokens, input_ids, layer_cache)
                last_block = block
        finally:
            cache_rollback.set_undo_armed(False)

        logits_source = h
        if logits_keep and logits_source.shape[1] > logits_keep:
            logits_source = logits_source[:, -logits_keep:, :]
        normed = last_block.norm(logits_source)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            from mlx_vlm.models.glm5_next.language import linear_forward

            logits = linear_forward(self.lm_head, normed)
        if return_hidden:
            return logits, h
        return logits

    def _cache_can_trim(c, n: int) -> bool:
        """Non-mutating check that ``c.trim(n)`` (or its replay) will
        succeed. A failed trim on one layer after earlier layers already
        trimmed desynchronises per-layer lengths (the hazard
        ``_restore_or_trim_caches`` documents).
        """
        subs = getattr(c, "caches", None)
        if subs is not None:
            return all(_cache_can_trim(sub, n) for sub in subs)
        if getattr(c, "_mtp_draft_stash", None) is not None:
            # Unsplit-verify replay: keep = accepted + 1 <= stash rows.
            mixed = c._mtp_draft_stash[0]
            return c.rollback_state is not None and bool(
                mixed.shape[1] >= n
            )
        if getattr(c, "rollback_state", None) is not None:
            # Bare snapshot semantics (single-position restore).
            return n == 1
        remainder = getattr(c, "remainder", None)
        if remainder is not None:  # PoolingCache / BatchPoolingCache
            rem_min = remainder if isinstance(remainder, int) else min(remainder)
            if n <= rem_min:
                return True
            can_undo = getattr(c, "_can_undo", None)
            return bool(can_undo and can_undo(n))
        is_trimmable = getattr(c, "is_trimmable", None)
        if callable(is_trimmable) and is_trimmable():
            return True
        return False

    def mtp_clamp_accept(self, cache, accepted: int, num_drafts: int) -> int:
        """Largest m' <= accepted whose rollback every layer supports.

        The 12 DSA layers' PoolingCache windows bound a partial rollback
        (same shape as DeepSeek-V4's compressed attention); the 33
        linear-attention layers replay from the verify stash. Emitting
        fewer verified drafts than the acceptance test allowed is always
        correct — the skipped ones are re-derived next cycle.
        """
        for m in range(accepted, -1, -1):
            n = num_drafts - m
            if n <= 0 or all(_cache_can_trim(c, n) for c in cache):
                return m
        return 0

    def _layer_cache_of(self, idx: int):
        return self.model.layers[idx]

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Roll the trunk caches back to ``accepted`` drafts after a
        depth-k verify forward over ``[confirmed, d1..dk]``.

        Sparse layers are CacheList(KVCache + PoolingCache) and trim
        through PoolingCache's cross-boundary undo. Linear-attention
        layers are ArraysCache: the unsplit-verify stash
        (``rollback_state`` + ``_mtp_draft_stash``) replays the kept
        prefix — the confirmed token plus the accepted drafts — through
        the factored ``_process_chunk``, one small recurrent kernel per
        layer, paid only on rejections. All layers are validated before
        any mutation.
        """
        n = num_drafts - accepted
        if n <= 0:
            return True
        if not all(_cache_can_trim(c, n) for c in cache):
            return False
        linear_idx = [
            i for i, layer in enumerate(self.model.layers) if layer.is_linear
        ]
        for i, c in zip(linear_idx, [cache[j] for j in linear_idx]):
            stash = getattr(c, "_mtp_draft_stash", None)
            if stash is None:
                continue
            mixed, fa_o, ga_o, b_o, mask = stash
            conv_0, ssm_0 = c.rollback_state
            keep = 1 + accepted
            layer = self.model.layers[i]
            _, conv_m, ssm_m = layer.self_attn._process_chunk(
                mixed[:, :keep],
                fa_o[:, :keep],
                ga_o[:, :keep],
                b_o[:, :keep],
                conv_0,
                ssm_0,
                mask[:, :keep] if mask is not None else None,
            )
            c[0] = conv_m
            c[1] = ssm_m
            c.rollback_state = None
            c._mtp_draft_stash = None
        for c in cache:
            if getattr(c, "_mtp_draft_stash", None) is not None:
                continue  # handled above
            subs = getattr(c, "caches", None)
            candidates = subs if subs is not None else [c]
            for sub in candidates:
                rollback = getattr(sub, "rollback_state", None)
                if rollback is not None:
                    conv_snap, ssm_snap = rollback
                    sub[0] = conv_snap
                    sub[1] = ssm_snap
                    sub.rollback_state = None
                    continue
                if hasattr(sub, "trim"):
                    trimmed = sub.trim(n)
                    if trimmed != n:
                        logger.warning(
                            "glm5_next MTP rollback trim shortfall on %s",
                            type(sub).__name__,
                        )
                        return False
        return True

    if not init_wrapped:
        cls.__init__ = __init__
        cls._omlx_mtp_init_wrapped = True
    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__
    cls.get_mtp_module = get_mtp_module
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_forward = mtp_forward
    cls.mtp_clamp_accept = mtp_clamp_accept
    cls.mtp_partial_rollback = mtp_partial_rollback


def _identity_hc(hc_mult: int):
    # Parameter-free HyperConnection identity for the MTP draft block.
    # The JANG draft ships no attn_hc/ffn_hc weights: the reference nextn
    # head is a plain pre-norm residual layer. Streams are broadcast
    # identical, so collapse-to-stream-0 plus post=1/comb=I gives
    # out = branch(norm(x)) + x exactly; the trailing mean is a no-op.
    import mlx.core as _mx
    import mlx.nn as _nn

    class _IdentityHC(_nn.Module):
        def __init__(self):
            super().__init__()
            self._h = int(hc_mult)

        def __call__(self, x):
            b, s, _, d = x.shape
            xc = _mx.contiguous(x[:, :, 0, :])
            post = _mx.ones((b, s, self._h), dtype=_mx.float32)
            comb = _mx.broadcast_to(_mx.eye(self._h, dtype=_mx.float32), (b, s, self._h, self._h))
            return xc, post, comb

    return _IdentityHC()


def _make_mtp_block(glm: Any, layer_config: Any, args: Any):
    import mlx.nn as nn

    lang = _language_mod(glm)
    Glm5NextDecoderLayer = lang.Glm5NextDecoderLayer

    class Glm5NextMTPBlock(nn.Module):
        """One MTP layer: enorm/hnorm/eh_proj fusion + a full glm5_next
        decoder layer.

        ``norm`` holds the checkpoint's ``shared_head.norm`` — applied to
        the block output before the shared lm_head (mtp_forward does the
        lm_head so ``logits_keep`` can shrink the vocab matmul).
        """

        def __init__(self):
            super().__init__()
            dim = args.hidden_size
            eps = args.rms_norm_eps
            self._hc_mult = int(getattr(args, "hc_mult", 4) or 4)
            self.enorm = nn.RMSNorm(dim, eps=eps)
            self.hnorm = nn.RMSNorm(dim, eps=eps)
            self.eh_proj = nn.Linear(2 * dim, dim, bias=False)
            self.norm = nn.RMSNorm(dim, eps=eps)
            self.block = Glm5NextDecoderLayer(
                layer_config, layer_config.num_hidden_layers
            )
            self.block.attn_hc = _identity_hc(self._hc_mult)
            self.block.ffn_hc = _identity_hc(self._hc_mult)

        def __call__(self, h, embed_tokens, input_ids, cache):
            import mlx.core as mx

            e = self.enorm(embed_tokens(input_ids))
            x = self.eh_proj(mx.concatenate([e, self.hnorm(h)], axis=-1))
            # The decoder layer stack runs on the hyper-connection layout
            # (B, L, hc_mult, D): broadcast the fused input out and fold
            # the streams back with a mean afterwards, exactly like the
            # trunk's Glm5NextModel.__call__ does around its layer loop.
            x = mx.broadcast_to(
                x[:, :, None, :],
                (x.shape[0], x.shape[1], self._hc_mult, x.shape[2]),
            )
            x = mx.contiguous(x)
            out = self.block(x, mask=None, cache=cache)
            return out.mean(axis=2)

    return Glm5NextMTPBlock()


