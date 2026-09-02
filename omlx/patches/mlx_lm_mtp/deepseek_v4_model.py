# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch for Blaizzy/mlx-lm#15 — DeepSeek-V4-Flash native MTP.

PR 15 is currently DRAFT. The shape mirrors PR 990 (Qwen3.5/3.6) but the
MTP head architecture is heavier: each ``MTPBlock`` wraps a full
``DeepseekV4Block`` plus per-block ``HyperHead`` and projection layers
(``e_proj``, ``h_proj``, ``enorm``, ``hnorm``, ``norm``).

oMLX already injects the DeepSeek-V4 base model itself via
``omlx/patches/deepseek_v4/`` (which lands the model class into
``sys.modules['mlx_lm.models.deepseek_v4']``); this patch sits on top and
adds the MTP head + ``mtp_forward`` / ``make_mtp_cache``. Apply order:
caller (``patches/mlx_lm_mtp/__init__.py``) runs ``apply()`` after the
base DeepSeek-V4 patch has registered the module.

The DeepSeek-V4 backbone has 4D hidden states (``B, S, hc_mult, hidden``)
because of the Hyper-head broadcasting. Both the patched ``__call__``
(with ``return_hidden=True``) and ``mtp_forward`` accept / produce 4D
tensors; the ``BatchGenerator`` MTP dispatch handles the dimension
difference via a small adapter (see ``batch_generator._slice_hidden``).
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

from . import deepseek_v4_dspark, prompt_priming

logger = logging.getLogger(__name__)


def _is_our_method(cls: Any, attr: str, marker: str) -> bool:
    """True iff ``cls.<attr>`` carries our marker. Mirror of the helper
    in qwen35_model — used for self-healing idempotency so a dflash
    overwrite of ``__call__`` in between two Native-MTP loads doesn't
    leave the class stuck on dflash's signature (issue #1388)."""
    existing = cls.__dict__.get(attr)
    return getattr(existing, marker, False)


def apply() -> bool:
    """Apply PR 15 model-side patches when the DeepSeek-V4 base patch is active.

    Self-healing: re-runs sub-patches when class state has drifted (see
    qwen35_model.apply for the same pattern).
    """
    dsv4 = sys.modules.get("mlx_lm.models.deepseek_v4")
    if dsv4 is None or not hasattr(dsv4, "Model"):
        # Base DeepSeek-V4 patch hasn't registered the module yet. This
        # branch only hits when MTP is enabled on a non-DeepSeek model —
        # log and skip cleanly.
        logger.debug(
            "DeepSeek-V4 module not registered; skipping MTP patch (this is "
            "expected for non-DeepSeek models)"
        )
        return False

    _patch_model_args(dsv4)
    _register_mtp_block(dsv4)
    deepseek_v4_dspark.register(dsv4)
    _patch_deepseek_v4_model_call(dsv4)
    _patch_model(dsv4)

    if not hasattr(dsv4.Model, "_omlx_mtp_patched"):
        dsv4.Model._omlx_mtp_patched = "patch"
        logger.info("DeepSeek-V4 MTP model patch applied (PR 15)")
    return True


# ---------------------------------------------------------------------------
# ModelArgs — extend compress_ratios to cover MTP layers.
# ---------------------------------------------------------------------------


def _patch_model_args(dsv4: Any) -> None:
    """Wrap ``ModelArgs.from_dict`` so MTP layers get a default compress_ratio.

    PR 15 widens the compress_ratios list from ``num_hidden_layers`` to
    ``num_hidden_layers + num_nextn_predict_layers`` (default 0, no
    compression). The original ``__post_init__`` raises if the list length
    doesn't match num_hidden_layers; we extend the list before that check
    by wrapping ``from_dict``.
    """
    args_cls = dsv4.ModelArgs
    if "_omlx_mtp_args_patched" in args_cls.__dict__:
        return

    original_from_dict = args_cls.from_dict.__func__

    def patched_from_dict(cls, params):
        # Build args via the base ``from_dict`` (which runs ``__post_init__``
        # and may truncate ``compress_ratios`` back to ``num_hidden_layers``).
        # Then re-extend the ratio list to cover MTP layers so MTPBlock's
        # ``DeepseekV4Block(..., layer_idx=n_main+i)`` lookup succeeds.
        args = original_from_dict(cls, params)
        n_main = int(getattr(args, "num_hidden_layers", 0) or 0)
        is_dspark = deepseek_v4_dspark.is_dspark_config(args)
        n_mtp = (
            deepseek_v4_dspark.stage_count(args)
            if is_dspark
            else int(getattr(args, "num_nextn_predict_layers", 0) or 0)
        )
        if n_mtp > 0 and hasattr(args, "compress_ratios"):
            source_ratios = list(params.get("compress_ratios") or ())
            ratios = list(args.compress_ratios)
            if is_dspark and len(source_ratios) >= n_main + n_mtp:
                ratios = source_ratios[: n_main + n_mtp]
            if len(ratios) < n_main + n_mtp:
                ratios = ratios + [0] * (n_main + n_mtp - len(ratios))
            args.compress_ratios = ratios
        return args

    args_cls.from_dict = classmethod(patched_from_dict)
    args_cls._omlx_mtp_args_patched = True


# ---------------------------------------------------------------------------
# MTPBlock — register on the module.
# ---------------------------------------------------------------------------


def _register_mtp_block(dsv4: Any) -> None:
    """Define ``MTPBlock`` and attach it to the module."""
    if hasattr(dsv4, "MTPBlock"):
        return

    import mlx.core as mx
    import mlx.nn as nn

    DeepseekV4Block = dsv4.DeepseekV4Block
    HyperHead = dsv4.HyperHead

    class MTPBlock(nn.Module):
        """One MTP layer in DeepSeek-V4's stack.

        Fuses the previous-layer hidden ``h`` (4D, broadcast to
        ``hc_mult`` Hyper-head copies) with the embedding of the
        next-position token ``input_ids``, then runs a full
        ``DeepseekV4Block`` over the result. The block's own
        ``hc_head`` collapses Hyper-head copies back to ``hidden_size``
        before the shared lm_head produces logits.
        """

        def __init__(self, config, layer_idx: int):
            super().__init__()
            dim = config.hidden_size
            self.block = DeepseekV4Block(config, layer_idx)
            self.e_proj = nn.Linear(dim, dim, bias=False)
            self.h_proj = nn.Linear(dim, dim, bias=False)
            self.enorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.hnorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.norm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.hc_head = HyperHead(config)

        def __call__(
            self,
            h,
            embed_tokens,
            input_ids,
            mask,
            cache,
        ):
            e = embed_tokens(input_ids)
            e = self.enorm(e)
            h_norm = self.hnorm(h)
            x = self.e_proj(e)[:, :, None, :] + self.h_proj(h_norm)
            x = mx.contiguous(x)
            # MTP masks are created by the patched model from this cache/window.
            x = self.block(x, mask, cache, input_ids, _standard_mask=True)
            return x

    dsv4.MTPBlock = MTPBlock


# ---------------------------------------------------------------------------
# DeepseekV4Model — return_raw_hidden support.
# ---------------------------------------------------------------------------


def _patch_deepseek_v4_model_call(dsv4: Any) -> None:
    """Replace ``DeepseekV4Model.__call__`` to optionally return the raw 4D hidden."""
    cls = dsv4.DeepseekV4Model
    if _is_our_method(cls, "__call__", "_omlx_mtp_call_marker"):
        return

    original_call = getattr(cls, "_omlx_mtp_base_call", cls.__call__)
    cls._omlx_mtp_base_call = original_call

    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    CacheList = dsv4.CacheList
    _materialize_cache_arrays = dsv4._materialize_cache_arrays

    def __call__(
        self,
        inputs,
        cache=None,
        return_raw_hidden: bool = False,
        return_dspark_hidden: bool = False,
        *,
        inputs_embeds=None,
    ):
        if inputs_embeds is not None:
            return original_call(
                self,
                inputs,
                cache,
                inputs_embeds=inputs_embeds,
            )
        h = self.embed_tokens(inputs)
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
        )
        h = mx.contiguous(h)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        first_cache = cache[0]
        mask_cache = (
            first_cache[0] if isinstance(first_cache, CacheList) else first_cache
        )
        mask = create_attention_mask(
            h[:, :, 0, :],
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        target_ids = tuple(getattr(self.args, "dspark_target_layer_ids", ()) or ())
        target_id_set = set(target_ids)
        dspark_hidden = {}
        for layer_idx, (layer, layer_cache) in enumerate(
            zip(self.pipeline_layers, cache)
        ):
            h = layer(h, mask, layer_cache, inputs, _standard_mask=True)
            if return_dspark_hidden and layer_idx in target_id_set:
                dspark_hidden[layer_idx] = h.mean(axis=2)

        _materialize_cache_arrays(cache)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            cache_item = cache[-1]
            if isinstance(cache_item, CacheList):
                cache_item = cache_item[0]
            if cache_item is not None:
                cache_item.keys = mx.depends(cache_item.keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        out = self.norm(self.hc_head(h))
        if return_dspark_hidden:
            if len(dspark_hidden) != len(target_ids):
                raise RuntimeError(
                    "DeepSeek DSpark target tap mismatch: "
                    f"captured={len(dspark_hidden)}, expected={len(target_ids)}"
                )
            return out, mx.concatenate(
                [dspark_hidden[layer_idx] for layer_idx in target_ids],
                axis=-1,
            )
        if return_raw_hidden:
            return out, h
        return out

    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__


# ---------------------------------------------------------------------------
# Model — wrap __init__, replace __call__, add mtp_forward / make_mtp_cache,
# replace sanitize with the PR 15 body that handles MTP weight remapping.
# ---------------------------------------------------------------------------


def _patch_model(dsv4: Any) -> None:
    cls = dsv4.Model
    init_wrapped = getattr(cls, "_omlx_mtp_init_wrapped", False)
    call_owned = _is_our_method(cls, "__call__", "_omlx_mtp_call_marker")
    if init_wrapped and call_owned:
        return

    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    # oMLX's DeepSeek-V4 fork uses a ``CacheList`` of (RotatingKVCache,
    # PoolingCache, [PoolingCache]) instead of upstream's
    # ``DeepseekV4Cache`` wrapper. The make_mtp_cache + mtp_forward bodies
    # below use the oMLX-side cache layout to stay compatible with the
    # already-patched DeepseekV4Model.__call__ above.
    CacheList = dsv4.CacheList
    PoolingCache = dsv4.PoolingCache
    RotatingKVCache = dsv4.RotatingKVCache
    SparseCompressedAttention = getattr(dsv4, "SparseCompressedAttention", None)
    materialize_cache_arrays = dsv4._materialize_cache_arrays

    original_init = cls.__init__
    original_call = getattr(cls, "_omlx_mtp_base_call", cls.__call__)
    cls._omlx_mtp_base_call = original_call

    def __init__(self, config):
        original_init(self, config)
        is_dspark = deepseek_v4_dspark.is_dspark_config(config)
        n_mtp = (
            deepseek_v4_dspark.stage_count(config)
            if is_dspark
            else int(getattr(config, "num_nextn_predict_layers", 0) or 0)
        )
        # See qwen35_model._patch_model: gated on the MTP active-flag so
        # mtp_enabled=False produces a model indistinguishable from stock.
        from . import is_mtp_active

        mtp_decode_enabled = bool(n_mtp > 0 and is_mtp_active())
        self._omlx_mtp_decode_enabled = mtp_decode_enabled
        self._omlx_dspark_decode_enabled = bool(mtp_decode_enabled and is_dspark)
        if mtp_decode_enabled:
            n_main = config.num_hidden_layers
            if is_dspark:
                self.mtp = [
                    dsv4.DSparkBlock(config, n_main + i, i, n_mtp) for i in range(n_mtp)
                ]
            else:
                self.mtp = [dsv4.MTPBlock(config, n_main + i) for i in range(n_mtp)]
            # Depth-k chained drafting is available: mtp_forward supports
            # return_hidden below, backbone rollback goes through
            # mtp_partial_rollback, and the head cache is a RotatingKVCache
            # (not exactly trimmable once rotated) so the chain runs its
            # speculative head steps on a per-cycle clone.
            from . import get_mtp_depth

            self._omlx_mtp_chain = True
            depth = get_mtp_depth()
            if is_dspark:
                depth = min(depth, int(config.dspark_block_size))
                self._omlx_mtp_head_clone = False
                # DSpark context caches are singleton and committed-only. The
                # existing row-wise extract/merge path does not model them.
                self._omlx_mtp_rowwise_unsupported = True
                logger.info(
                    "DeepSeek speculative backend selected: embedded DSpark "
                    "(%d stages, draft width %d)",
                    n_mtp,
                    depth,
                )
            else:
                self._omlx_mtp_head_clone = True
            self._omlx_mtp_depth = depth

    def __call__(
        self,
        inputs,
        cache=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
        skip_lm_head: bool = False,
    ):
        if (
            getattr(self, "is_vision_model", False)
            and not getattr(self, "_omlx_mtp_decode_enabled", False)
            and not return_hidden
            and not skip_lm_head
        ):
            return original_call(self, inputs, cache)
        if skip_lm_head:
            # Chunked prefill discards per-chunk logits (the prompt's final
            # token is scored by the first decode step instead). Run the
            # hidden pass and preserve capture side effects, but skip the
            # full-vocabulary projection for every chunk.
            if (
                getattr(self, "_omlx_dspark_decode_enabled", False)
                and not n_confirmed
                and cache is not None
            ):
                h, h_aux = self.model(inputs, cache, return_dspark_hidden=True)
                try:
                    deepseek_v4_dspark.capture_prompt(self, inputs, h_aux, cache)
                except Exception:
                    logger.debug("DeepSeek DSpark prompt capture failed", exc_info=True)
                return None
            if not n_confirmed and prompt_priming.capture_eligible(self, cache):
                h, h_raw = self.model(inputs, cache, return_raw_hidden=True)
                try:
                    prompt_priming.maybe_capture(self, inputs, h_raw, cache)
                except Exception:
                    logger.debug("MTP prompt-priming capture failed", exc_info=True)
                return None
            self.model(inputs, cache)
            return None
        # ``n_confirmed`` is part of the patched-backbone interface:
        # batch_generator._call_backbone passes n_confirmed=1 during MTP
        # verify cycles. It only matters for models with module-level
        # recurrent state (Qwen3.5's GatedDeltaNet splits its forward to
        # snapshot state after the confirmed prefix). DeepSeek-V4 keeps all
        # decode state in cache objects (RotatingKVCache / PoolingCache)
        # and draft rejection rolls back via cache.trim in
        # _restore_or_trim_caches, so the argument is accepted and unused.
        if return_hidden:
            if getattr(self, "_omlx_dspark_decode_enabled", False):
                h, h_aux = self.model(inputs, cache, return_dspark_hidden=True)
                return self.lm_head(h), h_aux
            h, h_raw = self.model(inputs, cache, return_raw_hidden=True)
            return self.lm_head(h), h_raw
        if (
            getattr(self, "_omlx_dspark_decode_enabled", False)
            and not n_confirmed
            and cache is not None
        ):
            h, h_aux = self.model(inputs, cache, return_dspark_hidden=True)
            try:
                deepseek_v4_dspark.capture_prompt(self, inputs, h_aux, cache)
            except Exception:
                logger.debug("DeepSeek DSpark prompt capture failed", exc_info=True)
            return self.lm_head(h)
        if not n_confirmed and prompt_priming.capture_eligible(self, cache):
            # Prompt-priming capture needs the head-input hidden (the raw 4D
            # Hyper-stream activation), which the stock branch discards
            # inside self.model — same compute, one extra returned tensor.
            h, h_raw = self.model(inputs, cache, return_raw_hidden=True)
            try:
                prompt_priming.maybe_capture(self, inputs, h_raw, cache)
            except Exception:
                logger.debug("MTP prompt-priming capture failed", exc_info=True)
            return self.lm_head(h)
        h = self.model(inputs, cache)
        return self.lm_head(h)

    def make_mtp_cache(self):
        """Build per-MTP-block caches. Mirrors ``Model.make_cache`` but for the
        MTP stack. PR 15's MTP layers default to ``compress_ratio=0`` so the
        common case is a plain RotatingKVCache, but we honor the same
        SparseCompressedAttention / CacheList layout as the backbone for any
        config that ever assigns a non-zero compress_ratio to MTP layers.
        """
        if not hasattr(self, "mtp"):
            return None
        if getattr(self, "_omlx_dspark_decode_enabled", False):
            return [dsv4.DSparkContextCache(self.args.sliding_window) for _ in self.mtp]
        caches = []
        sw = self.args.sliding_window
        for mtp_block in self.mtp:
            attn = mtp_block.block.attn
            ratio = getattr(attn, "compress_ratio", 0)
            if ratio == 0:
                caches.append(RotatingKVCache(max_size=sw))
            elif SparseCompressedAttention is not None and isinstance(
                attn, SparseCompressedAttention
            ):
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=sw),
                        PoolingCache(ratio),
                        PoolingCache(ratio),
                    )
                )
            else:
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=sw),
                        PoolingCache(ratio),
                    )
                )
        return caches

    def dspark_append_context(
        self,
        main_hidden,
        cache,
        *,
        start_offset=None,
    ):
        """Project target taps once and append committed K/V to every stage."""
        if not getattr(self, "_omlx_dspark_decode_enabled", False):
            raise RuntimeError("DSpark context requested on a Lightning MTP model")
        first = self.mtp[0]
        main_x = first.main_norm(first.main_proj(main_hidden))
        for stage, stage_cache in zip(self.mtp, cache):
            stage.attn.append_context(
                main_x,
                stage_cache,
                start_offset=start_offset,
            )
        return main_x

    def dspark_forward(
        self,
        main_hidden,
        anchor_ids,
        cache=None,
        *,
        draft_length=None,
    ):
        """Run one parallel DSpark proposal block.

        ``main_hidden`` contains concatenated target taps for newly committed
        target inputs. ``anchor_ids`` is the newest target-confirmed token;
        remaining query positions use the checkpoint's noise token.
        """
        if not getattr(self, "_omlx_dspark_decode_enabled", False):
            raise RuntimeError("DSpark forward requested on a Lightning MTP model")
        if cache is None:
            cache = self.make_mtp_cache()
        self.dspark_append_context(main_hidden, cache)

        width = int(draft_length or self._omlx_mtp_depth)
        max_width = int(self.args.dspark_block_size)
        width = max(1, min(width, max_width))
        anchor_ids = anchor_ids.reshape(anchor_ids.shape[0], -1)[:, -1:]
        draft_ids = mx.full(
            (anchor_ids.shape[0], width),
            int(self.args.dspark_noise_token_id),
            dtype=anchor_ids.dtype,
        )
        draft_ids = mx.concatenate([anchor_ids, draft_ids[:, 1:]], axis=1)

        hidden = self.model.embed_tokens(draft_ids)
        hidden = mx.broadcast_to(
            hidden[:, :, None, :],
            (
                hidden.shape[0],
                hidden.shape[1],
                self.args.hc_mult,
                hidden.shape[-1],
            ),
        )
        hidden = mx.contiguous(hidden)
        for stage_idx, (stage, stage_cache) in enumerate(zip(self.mtp, cache)):
            hidden = stage(
                hidden,
                draft_ids,
                stage_cache,
                output_width=width if stage_idx + 1 == len(self.mtp) else None,
            )

        final = self.mtp[-1]
        head_hidden = final.hc_head(hidden)
        from omlx.patches.deepseek_v4.verify_qmv import dspark_head_gemv

        logits = dspark_head_gemv(self.lm_head, final.norm(head_hidden))
        return logits[:, :width], head_hidden[:, :width]

    def dspark_markov(self, token_ids):
        """Return DSpark's previous-token logit bias and rank-R embedding."""
        head = self.mtp[-1].markov_head
        embedding = head.markov_w1(token_ids)
        from omlx.patches.deepseek_v4.verify_qmv import dspark_head_gemv

        return dspark_head_gemv(head.markov_w2, embedding), embedding

    def dspark_calibration_forward(self, target_hiddens, input_ids):
        """Exercise all DSpark linears for oQe imatrix collection."""
        if not getattr(self, "_omlx_dspark_decode_enabled", False):
            return None
        width = min(int(self.args.dspark_block_size), max(1, input_ids.shape[1] - 1))
        context = target_hiddens[:, :-1]
        anchor = input_ids[:, -1:]
        logits, _ = self.dspark_forward(
            context,
            anchor,
            self.make_mtp_cache(),
            draft_length=width,
        )
        mx.eval(logits)
        return logits

    def mtp_take_primed(self, cache, main_token):
        if not getattr(self, "_omlx_dspark_decode_enabled", False):
            return None
        return deepseek_v4_dspark.take_primed(self, cache, main_token)

    def mtp_forward(
        self,
        h,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        """Run the chained MTP blocks + final hc_head/norm/lm_head on a 4D hidden.

        Mirrors PR 15: each MTP block fuses ``h`` with the embedded
        ``input_ids`` through its ``e_proj``/``h_proj`` projection and
        passes the result through a ``DeepseekV4Block``. The last block's
        ``hc_head`` collapses the Hyper-head dimension before ``norm`` and
        the shared ``lm_head`` produce logits.

        ``return_hidden`` additionally returns the last block's raw 4D
        output — the chain feeds it back as the next draft step's ``h``
        (the same contract the block sees when consuming trunk hidden).
        ``logits_keep`` limits the hc_head/norm/lm_head tail to the last N
        positions (0 = all); the chain's history+draft fold only needs the
        final position and the vocab is large enough that it matters.
        """
        if getattr(self, "_omlx_dspark_decode_enabled", False):
            logits, hidden = self.dspark_forward(
                h,
                input_ids,
                cache,
                draft_length=input_ids.shape[1],
            )
            if logits_keep and logits.shape[1] > logits_keep:
                logits = logits[:, -logits_keep:]
                hidden = hidden[:, -logits_keep:]
            if return_hidden:
                return logits, hidden
            return logits

        if cache is None:
            cache = [None] * len(self.mtp)

        first_cache = cache[0]
        mask_cache = (
            first_cache[0] if isinstance(first_cache, CacheList) else first_cache
        )
        mask_input = h[:, :, 0, :] if h.ndim == 4 else h
        mask = create_attention_mask(
            mask_input,
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        last_block = None
        for mtp_block, layer_cache in zip(self.mtp, cache):
            h = mtp_block(h, self.model.embed_tokens, input_ids, mask, layer_cache)
            last_block = mtp_block

        materialize_cache_arrays(cache)

        logits_source = h
        if logits_keep and logits_source.shape[1] > logits_keep:
            logits_source = logits_source[:, -logits_keep:]
        out = last_block.hc_head(logits_source)
        out = last_block.norm(out)
        logits = self.lm_head(out)
        if return_hidden:
            return logits, h
        return logits

    def _cache_can_trim(c, n: int) -> bool:
        """Non-mutating check that ``c.trim(n)`` will succeed.

        Needed because a failed ``trim`` on one layer after earlier layers
        already trimmed leaves per-layer lengths desynchronised (the hazard
        ``_restore_or_trim_caches`` documents). PoolingCaches expose the
        exact undo feasibility; rotating caches accept the trim while the
        armed undo (or an unrotated buffer) covers it.
        """
        if isinstance(c, CacheList):
            return all(_cache_can_trim(sub, n) for sub in c.caches)
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
        # Rotated RotatingKVCache: the cache_rollback undo stash covers the
        # last armed multi-token write.
        undo = getattr(c, "_mtp_undo", None)
        if undo is not None:
            return undo[1].shape[2] >= n
        return False

    def mtp_clamp_accept(self, cache, accepted: int, num_drafts: int) -> int:
        """Largest ``m' <= accepted`` whose rollback every layer supports.

        Emitting fewer verified drafts than the acceptance test allowed is
        always correct (the skipped ones are re-derived next cycle); it
        just wastes a little verified work. This keeps the chain alive when
        a PoolingCache can't replay a longer confirmed prefix (its pooled
        windows can't be reconstructed inside ``trim``).
        """
        for m in range(accepted, -1, -1):
            n = num_drafts - m
            if n <= 0 or all(_cache_can_trim(c, n) for c in cache):
                return m
        return 0

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Trim the verify window back to ``accepted`` drafts on every layer."""
        n = num_drafts - accepted
        if n <= 0:
            return True
        if not all(_cache_can_trim(c, n) for c in cache):
            return False
        for c in cache:
            if c.trim(n) != n:
                logger.warning(
                    "DeepSeek-V4 MTP rollback trim shortfall on %s",
                    type(c).__name__,
                )
                return False
        return True

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Combined oMLX-base + PR 15 sanitize.

        oMLX's stock sanitize strips ``mtp.*`` and remaps the FP4 expert
        weights / Hyper-head names. PR 15 keeps ``mtp.*`` when an MTP head
        is present, nests block-internal weights under ``.block.``, and
        stacks routed expert weights for MTP layers as well as backbone
        layers.
        """
        n_layers = self.args.num_hidden_layers
        is_dspark = bool(getattr(self, "_omlx_dspark_decode_enabled", False))
        has_mtp = hasattr(self, "mtp")
        has_mtp_weights = any(k.startswith("mtp.") for k in weights)
        # Disable MTP module if weights are absent (e.g. quantized checkpoints
        # that stripped them). Mirrors PR 15's graceful fallback.
        if has_mtp and not has_mtp_weights:
            try:
                del self.mtp
            except AttributeError:
                pass
            has_mtp = False
            self._omlx_mtp_decode_enabled = False
            self._omlx_dspark_decode_enabled = False

        new_weights: Dict[str, Any] = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                if not has_mtp:
                    continue
                new_weights[k] = v
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    if int(parts[1]) >= n_layers:
                        continue
                except ValueError:
                    pass
            new_weights[k] = v
        weights = new_weights

        # FP4 dequant pre-pass (oMLX-specific). Identical to the
        # un-patched body — safe to keep as-is.
        new_weights = {}
        for k, v in weights.items():
            if "tid2eid" in k:
                new_weights[k] = v.astype(mx.int32)

            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue

            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ".ffn.experts." in wk
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[k + "s"] = v
                new_weights[wk] = weight.view(mx.uint32)
            elif weight.dtype == mx.uint8:
                new_weights[k + "s"] = mx.repeat(mx.repeat(v, 4, -1), 128, 0)
                new_weights[wk] = weight.view(mx.uint32)
            else:
                new_weights[k] = v
        weights = new_weights

        top_remap = {
            "embed.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        }
        for old, new in top_remap.items():
            if old in weights:
                weights[new] = weights.pop(old)

        # Block-internal weight key remapping. Legacy Lightning MTP nests
        # stage weights under ``mtp.<idx>.block``; 0731 DSpark stages are
        # represented directly at ``mtp.<idx>`` to match the checkpoint.
        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        mtp_block_subs = (
            "attn.",
            "ffn.",
            "attn_norm.",
            "ffn_norm.",
            "hc_attn_",
            "hc_ffn_",
            "hc_attn.",
            "hc_ffn.",
        )
        for k, v in weights.items():
            nk = "model." + k if k.startswith("layers.") else k
            # MTP block: nest block-internal weights under .block.
            if nk.startswith("mtp."):
                parts = nk.split(".", 2)  # ["mtp", "<idx>", "<rest>"]
                if len(parts) == 3:
                    rest = parts[2]
                    if not is_dspark and any(
                        rest.startswith(s) for s in mtp_block_subs
                    ):
                        nk = f"mtp.{parts[1]}.block.{rest}"
                    for param in ("fn", "base", "scale"):
                        if rest == f"hc_head_{param}":
                            nk = f"mtp.{parts[1]}.hc_head.{param}"
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
            skip = False
            for old, new in (
                (".hc_attn.", ".attn_hc."),
                (".hc_ffn.", ".ffn_hc."),
            ):
                if old in nk:
                    candidate = nk.replace(old, new)
                    if candidate in weights or candidate in remapped:
                        skip = True
                        break
                    nk = candidate
            if skip:
                continue
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            remapped[nk] = v
        weights = remapped

        # Stack routed expert weights for backbone layers.
        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.ffn.experts"
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                for suffix in ("weight", "scales", "biases"):
                    key0 = f"{prefix}.0.{src}.{suffix}"
                    if key0 in weights:
                        stacked = [
                            weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[
                            f"model.layers.{layer_idx}.ffn.switch_mlp.{dst}.{suffix}"
                        ] = mx.stack(stacked)

        # Reshape wo_a from nn.Linear (2D) to MultiLinear (3D) for all layers.
        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.attn.wo_a"
            for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
                if key in weights and weights[key].ndim == 2:
                    weights[key] = weights[key].reshape(
                        self.args.o_groups, self.args.o_lora_rank, -1
                    )

        # Same wo_a 2D -> 3D reshape for the MTP block's attention. The
        # ndim==2 gate keeps this idempotent for checkpoints that already
        # store the 3D MultiLinear layout (e.g. oQ output).
        if has_mtp:
            n_mtp = (
                deepseek_v4_dspark.stage_count(self.args)
                if is_dspark
                else self.args.num_nextn_predict_layers
            )
            block_part = "" if is_dspark else ".block"
            for mtp_idx in range(n_mtp):
                prefix = f"mtp.{mtp_idx}{block_part}.attn.wo_a"
                for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
                    if key in weights and weights[key].ndim == 2:
                        weights[key] = weights[key].reshape(
                            self.args.o_groups, self.args.o_lora_rank, -1
                        )

        # Stack routed expert weights for MTP layers (PR 15).
        if has_mtp:
            n_mtp = (
                deepseek_v4_dspark.stage_count(self.args)
                if is_dspark
                else self.args.num_nextn_predict_layers
            )
            block_part = "" if is_dspark else ".block"
            for mtp_idx in range(n_mtp):
                prefix = f"mtp.{mtp_idx}{block_part}.ffn.experts"
                for src, dst in (
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ):
                    for suffix in ("weight", "scales", "biases"):
                        key0 = f"{prefix}.0.{src}.{suffix}"
                        if key0 in weights:
                            stacked = [
                                weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                                for e in range(self.args.n_routed_experts)
                            ]
                            weights[
                                f"mtp.{mtp_idx}{block_part}.ffn.switch_mlp."
                                f"{dst}.{suffix}"
                            ] = mx.stack(stacked)

        # Preserve the base DeepSeek sanitizer's affine SwitchGLU dtype
        # normalization. MLX affine MoE kernels require FP16 scale/bias
        # metadata when the packed weight is uint32; this applies equally to
        # backbone and oQ/oQe-quantized DSpark experts.
        for key, value in list(weights.items()):
            if (
                ".ffn.switch_mlp." not in key
                or not key.endswith((".scales", ".biases"))
                or value.dtype != mx.bfloat16
            ):
                continue
            stem = key.rsplit(".", 1)[0]
            if (
                stem + ".weight" in weights
                and stem + ".scales" in weights
                and stem + ".biases" in weights
                and weights[stem + ".weight"].dtype == mx.uint32
            ):
                weights[key] = value.astype(mx.float16)

        return weights

    if not init_wrapped:
        cls.__init__ = __init__
        cls._omlx_mtp_init_wrapped = True
    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.dspark_append_context = dspark_append_context
    cls.dspark_forward = dspark_forward
    cls.dspark_markov = dspark_markov
    cls.dspark_calibration_forward = dspark_calibration_forward
    cls.mtp_take_primed = mtp_take_primed
    cls.mtp_clamp_accept = mtp_clamp_accept
    cls.mtp_partial_rollback = mtp_partial_rollback
    cls.sanitize = sanitize
