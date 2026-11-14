# SPDX-License-Identifier: Apache-2.0
"""MiniCPM5-2B DSpark speculative decoding adaptation (llama backbone).

This is the **single-file adaptation** for running a MiniCPM5-2B-DSpark
draft model against a MiniCPM5-2B / MiniCPM5-2B-oQ4e target through the
oMLX DFlash block-drafter path.  It keeps the upstream ``jundot/omlx``
source files byte-identical and puts every change here.

Why llama needs this module
---------------------------
The MiniCPM5 target is a plain text-only ``llama`` model served through
mlx-vlm's VLM engine (``omlx_force_engine: "vlm"``).  Its ``LanguageModel``
does NOT expose the speculative-decoding hooks that the oMLX batch generator
expects from a block-drafter target:

  - ``capture_layer_ids`` / ``return_hidden`` / ``hidden_sink`` on the
    backbone ``__call__``, so the drafter can consume per-layer hidden
    states at ``target_layer_ids``;
  - ``rollback_speculative_cache`` and ``mtp_partial_rollback``, so a
    rejected draft can be rolled back (llama keeps only KV / RotatingKV
    caches, so committing is a pure trim).

This module monkey-patches ``mlx_vlm.models.llama.language`` in place.  It
also extends the oMLX ``_DepthController`` with DSpark's **block-level
adaptive feedback** (official ``_dflash_next_block_size``): DSpark keeps
proposing down to a minimum block width instead of parking to standard
decoding, and decides the next block width from the recent accept/draft
lengths.  Upstream's ``_post_init_mtp`` already offers a
``make_mtp_depth_controller`` extension point, so the DSpark controller is
injected there without editing ``batch_generator.py``.

Integration
-----------
Call :func:`apply` from a pre-load hook in ``utils/model_loading.py`` when
a llama model carries a bundled DSpark drafter (see the sibling
``minimax_m3_dspark.py`` / ``deepseek_v4_dspark.py`` precedents).  The
two required upstream edits are:

  - ``utils/model_loading.py``: add ``"llama"`` to
    ``_DFLASH_BATCHED_MODEL_TYPES["vlm"]`` and merge the pre-quantized
    per-path ``quantization_config`` into ``quantization`` (MiniCPM5 ships
    mixed bit-width layers).
  - ``model_discovery.py``: honor ``omlx_force_engine: "vlm"`` so the
    text-only llama loads on the VLM engine.

The module is idempotent: re-applying after a partial application or a
later load is a no-op, mirroring the ``apply_mlx_lm_mtp_patch`` self-healing
contract.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Marker placed on the patched llama classes so re-apply is a no-op.
_LLAMA_PATCHED = "_omlx_minicpm5_llama_patched"
# Marker on the batch_generator module so the _DepthController extension is
# installed exactly once.
_DEPTH_PATCHED = "_omlx_minicpm5_depth_patched"
# Marker on the attach_drafter wrapper so it is installed exactly once.
_ATTACH_PATCHED = "_omlx_minicpm5_attach_patched"
# Marker for the _post_init_mtp rewrite (distinct from _DEPTH_PATCHED so the
# depth-controller extension and the post_init routing can be installed
# independently).
_POST_INIT_PATCHED = "_omlx_minicpm5_post_init_patched"

# DSpark keeps proposing down to this minimum block width (official default).
_MIN_BLOCK_SIZE = 4


# ---------------------------------------------------------------------------
# 1. llama LanguageModel capture + rollback hooks
# ---------------------------------------------------------------------------
def _patch_llama_language() -> bool:
    """Add speculative-decoding hooks to the mlx-vlm llama ``LanguageModel``.

    Monkey-patches ``LlamaModel.__call__`` and ``LanguageModel.__call__`` so
    the backbone can expose per-layer hidden states for the drafter, and adds
    ``rollback_speculative_cache`` / ``mtp_partial_rollback`` so a rejected
    draft is rolled back.  Returns True when the patch is (or was already)
    active.
    """
    try:
        import mlx_vlm.models.llama.language as lang
    except Exception as exc:  # pragma: no cover - environment without mlx-vlm
        logger.debug("mlx_vlm llama not importable: %s", exc)
        return False

    if getattr(lang, _LLAMA_PATCHED, False):
        return True

    LlamaModel = lang.LlamaModel
    LanguageModel = lang.LanguageModel
    LanguageModelOutput = lang.LanguageModelOutput

    # --- LlamaModel.__call__: optional per-layer hidden capture ---
    _orig_llama_call = LlamaModel.__call__

    def _llama_call(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: Optional[list] = None,
        hidden_sink: Optional[list] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        _mask_fn = _resolve_attention_mask_fn()
        fa_mask = _mask_fn(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = _mask_fn(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for idx, (layer, c) in enumerate(zip(self.layers, cache)):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, cache=c)
            if hidden_sink is not None and idx in capture_set:
                hidden_sink.append(h)
        return self.norm(h)

    # --- LanguageModel.__call__: capture + return_hidden ---
    _orig_lm_call = LanguageModel.__call__

    def _lm_call(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        capture_layer_ids: Optional[list] = None,
        return_hidden: bool = False,
        hidden_sink: Optional[list] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings

        if capture_layer_ids is not None or return_hidden:
            requested = sorted(set(capture_layer_ids or []))
            sink: list = []
            # qwen3_5 route: run the block-drafter forward inside one
            # ``SpeculativeCacheTransaction`` so a rejected block rolls back
            # through the shared ``cache_state`` machinery instead of a
            # hand-rolled trim loop.  The transaction rides on
            # ``gdn_states`` (llama has no GatedDeltaNet state, so the field
            # is repurposed for the transaction) and ``_call_backbone_impl``
            # / ``_rollback_partial`` commit or abort it.
            transaction = None
            try:
                from mlx_vlm.speculative.cache_state import start_speculative_cache

                transaction = start_speculative_cache(cache or [], inputs.shape[1])
            except Exception:
                transaction = None
            try:
                out = self.model(
                    inputs,
                    cache,
                    inputs_embeds,
                    capture_layer_ids=requested,
                    hidden_sink=sink,
                )
            except BaseException:
                if transaction is not None:
                    transaction.abort()
                raise
            by_layer = dict(zip(requested, sink))
            if self.args.tie_word_embeddings:
                logits = self.model.embed_tokens.as_linear(out)
            else:
                logits = self.lm_head(out)
            # ``[captures..., last]`` convention: the drafter needs each
            # captured layer hidden plus the final (post-norm) hidden, which
            # the batch generator's ``_call_backbone_impl`` uses as the MTP
            # head's pre-norm hidden.  This matches what the official code
            # already expects (``captured = hidden[:-1]``, ``hidden[-1]``).
            return LanguageModelOutput(
                logits=logits,
                hidden_states=[by_layer[i] for i in requested] + [out],
                gdn_states=transaction,
            )

        out = self.model(inputs, cache, inputs_embeds)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    # --- rollback hooks (llama has no GatedDeltaNet state) ---
    def _rollback_speculative_cache(
        self,
        caches: list,
        gdn_states: Any,
        accepted: Any,
        block_size: int,
    ) -> int:
        """Roll back a speculative block through the shared transaction.

        qwen3_5 route: when the forward was wrapped in a
        ``SpeculativeCacheTransaction`` (see ``_lm_call``), delegate to the
        stock ``cache_state.rollback_speculative_cache`` exactly like
        qwen3_5's ``LanguageModel.rollback_speculative_cache`` does.  That
        handles partial accepts (``accepted + 1`` positions kept) and aborts
        on reject.  A legacy/non-transaction state falls back to the simple
        trim loop (plain llama with no transaction, e.g. the 1-token
        post-init forward where the caller passes ``0, 1``).
        """
        try:
            from mlx_vlm.speculative.cache_state import (
                SpeculativeCacheTransaction,
                rollback_speculative_cache as _rollback_transaction,
            )
        except Exception:
            SpeculativeCacheTransaction = ()
            _rollback_transaction = None

        if (
            _rollback_transaction is not None
            and isinstance(gdn_states, SpeculativeCacheTransaction)
        ):
            return _rollback_transaction(caches, gdn_states, accepted, block_size)

        # Fallback (no transaction): plain trim by block rejection.
        del gdn_states
        if isinstance(accepted, int):
            accepted = mx.array([accepted], dtype=mx.int32)
        elif not isinstance(accepted, mx.array):
            accepted = mx.array(accepted, dtype=mx.int32)
        max_accepted = int(accepted.max().item())
        trim = block_size - max_accepted - 1
        if trim > 0:
            for cache in caches:
                if cache is not None and hasattr(cache, "trim"):
                    cache.trim(trim)
        return max_accepted

    def _mtp_partial_rollback(
        self, caches, accepted, num_drafts
    ) -> bool:
        try:
            if isinstance(accepted, int):
                accepted = mx.array([accepted], dtype=mx.int32)
            elif not isinstance(accepted, mx.array):
                accepted = mx.array(accepted, dtype=mx.int32)
            max_accepted = int(accepted.max().item())
            trim = num_drafts - max_accepted
            if trim > 0:
                for cache in caches:
                    if cache is not None and hasattr(cache, "trim"):
                        cache.trim(trim)
            return True
        except Exception:
            return False

    # Rebind: the module-level names must be closed over so the patch can be
    # applied after the module is imported (the monkey-patch replaces the
    # class attribute, not the local reference).
    lang.LlamaModel.__call__ = _llama_call
    lang.LanguageModel.__call__ = _lm_call
    lang.LanguageModel.rollback_speculative_cache = _rollback_speculative_cache
    lang.LanguageModel.mtp_partial_rollback = _mtp_partial_rollback
    setattr(lang, _LLAMA_PATCHED, True)
    logger.info("MiniCPM5 llama capture/rollback hooks installed")
    return True


# Forward reference: create_attention_mask imported lazily in _patch_llama_language.
create_attention_mask_fn = None


def _resolve_attention_mask_fn() -> Any:
    global create_attention_mask_fn
    if create_attention_mask_fn is None:
        from mlx_vlm.models.base import create_attention_mask
        create_attention_mask_fn = create_attention_mask
    return create_attention_mask_fn


# ---------------------------------------------------------------------------
# 2. DSpark block-level adaptive depth control
# ---------------------------------------------------------------------------
def dspark_initial_depth(block_total: int, min_block_size: int = _MIN_BLOCK_SIZE) -> int:
    """DSpark's initial proposal count (official initial block 4 -> 3)."""
    if block_total <= 1:
        return max(0, block_total - 1)
    return max(1, min(block_total, max(2, int(min_block_size))) - 1)


def dspark_next_block_size(
    block_total: int,
    accept_lens: List[int],
    draft_lens: List[int],
    min_block_size: int = _MIN_BLOCK_SIZE,
    log: Optional[logging.Logger] = None,
) -> int:
    """Official DSpark adaptive block width (pure port of ``_dflash_next_block_size``).

    ``accept_lens`` / ``draft_lens`` are **block widths** (the official
    convention: ``draft_lens`` records how many tokens were verified this
    cycle, not how many were proposed).  Returns the next **block width**;
    the controller converts to depth (``block - 1``).  Mirrors mlx-vlm
    ``_dflash_next_block_size`` exactly: back off below 0.30 accept or
    mean_accept < 2.0, hold at 0.50-0.85, grow at >=0.85 with >=0.75
    full-hit rate, never below ``min_block_size``.
    """
    if block_total <= 1:
        return block_total
    recent: List[Tuple[float, int]] = [
        (float(a), int(d))
        for a, d in zip(accept_lens[-8:], draft_lens[-8:])
        if int(d) > 0
    ]
    if not recent:
        if min_block_size >= 2:
            # No history: start at the configured floor (official initial).
            return min(block_total, max(2, int(min_block_size)))
        return block_total
    current = min(block_total, max(2, recent[-1][1] + 1))
    min_total = min(block_total, max(2, int(min_block_size)))
    drafted = sum(d for _, d in recent)
    accepted = sum(a for a, _ in recent)
    accept_rate = accepted / max(1e-6, drafted)
    mean_accept = accepted / len(recent)
    if accept_rate < 0.30 or mean_accept < 2.0:
        if current >= 8:
            return max(min_total, min(block_total, current // 2))
        result = max(min_total, min(block_total, current - 2))
        if log is not None:
            log.debug(
                "_dspark_next_block_size shrink: recent=%s rate=%.2f current=%d "
                "min_total=%d -> %d",
                recent, accept_rate, current, min_total, result,
            )
        return result
    if accept_rate < 0.50:
        return max(min_total, min(block_total, current - 2))
    full_hits = sum(1 for a, d in recent if a >= d)
    full_hit_rate = full_hits / len(recent)
    if accept_rate >= 0.85 and full_hit_rate >= 0.75:
        return min(block_total, current + 2)
    return min(block_total, current)


def _patch_depth_controller() -> bool:
    """Extend ``_DepthController`` in ``omlx.patches.mlx_lm_mtp.batch_generator``.

    Adds DSpark block-level adaptive control as an opt-in mode.  The base
    class stays untouched for the score-based controller; the DSpark branch
    is activated by ``dspark_mode=True`` and injected through the existing
    ``make_mtp_depth_controller`` extension point.
    """
    try:
        from omlx.patches.mlx_lm_mtp import batch_generator as bg
    except Exception as exc:  # pragma: no cover
        logger.debug("batch_generator not importable: %s", exc)
        return False

    DepthController = bg._DepthController
    if getattr(bg, _DEPTH_PATCHED, False):
        return True

    # --- extend __init__ ---
    _orig_init = DepthController.__init__

    def _init(self, max_depth, marginal_ms=None, exit_margin=None, seed=None,
              dspark_mode: bool = False, dspark_block_size: Optional[int] = None,
              dspark_min_block_size: int = _MIN_BLOCK_SIZE):
        _orig_init(self, max_depth, marginal_ms, exit_margin, seed)
        self.dspark_mode = bool(dspark_mode)
        self.dspark_block_size = int(dspark_block_size or self.max_depth + 1)
        self.dspark_min_block_size = max(2, int(dspark_min_block_size))
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []
        if self.dspark_mode:
            # DSpark starts at the official initial block (4 -> 3 proposals).
            self.cur = dspark_initial_depth(
                self.dspark_block_size, self.dspark_min_block_size
            )
            # DSpark drives the depth directly from block-level feedback, so
            # the score-based warmup sweep / probe / exit machinery must not
            # override ``cur`` or misclassify these cycles as warmup.
            self._warmup = []
            self.probe_left = 0
            self.exit_streak = 0
            self._streak_spec = 0

    # --- extend observe ---
    _orig_observe = DepthController.observe

    def _observe(self, used, accepted, cycle_ms, time_sample=True):
        if self.dspark_mode:
            self.cycles += 1
            used = max(0, min(int(used), self.max_depth))
            accepted = max(0, min(int(accepted), used))
            # DSpark adaptive control works on BLOCK WIDTHS: the drafter
            # proposes ``used`` tokens, so the block that was verified spans
            # ``used + 1`` positions (anchor + drafts).  Record the block
            # width and convert the official block width back to depth.
            self.accept_lens.append(int(accepted))
            self.draft_lens.append(int(used) + 1)
            self.accept_lens = self.accept_lens[-8:]
            self.draft_lens = self.draft_lens[-8:]
            next_block = dspark_next_block_size(
                self.dspark_block_size,
                self.accept_lens,
                self.draft_lens,
                min_block_size=self.dspark_min_block_size,
                log=logger,
            )
            self.cur = max(0, next_block - 1)
            return
        return _orig_observe(self, used, accepted, cycle_ms, time_sample)

    # --- should_exit: DSpark never parks to standard decoding ---
    _orig_should_exit = DepthController.should_exit

    def _should_exit(self) -> bool:
        if self.dspark_mode:
            return False
        return _orig_should_exit(self)

    DepthController.__init__ = _init
    DepthController.observe = _observe
    DepthController.should_exit = _should_exit
    setattr(bg, _DEPTH_PATCHED, True)
    logger.info("MiniCPM5 DSpark depth controller installed")
    return True


def make_dspark_depth_controller(depth: int) -> Any:
    """Factory for a DSpark-adaptive ``_DepthController``.

    Returned to ``_post_init_mtp`` through the model's
    ``make_mtp_depth_controller`` hook so the block-drafter path uses
    DSpark's block-level feedback instead of the score-based controller.

    ``depth`` is the chain depth the batch generator resolved
    (``_omlx_mtp_depth``), which for a MiniCPM5 block drafter is the
    ``dflash_block_size`` ceiling, so the controller's block width is
    ``depth + 1`` (the official DFlash convention).
    """
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    block_size = max(2, int(depth) + 1)
    return bg._DepthController(
        depth,
        dspark_mode=True,
        dspark_block_size=block_size,
        dspark_min_block_size=_MIN_BLOCK_SIZE,
    )


def _patch_post_init_mtp() -> bool:
    """Route MiniCPM5 (llama) block-drafter cycles to the DSpark controller.

    Upstream ``_post_init_mtp`` creates the depth controller via
    ``_dspark_host(model).make_mtp_depth_controller`` when an embedded
    DSpark host exists, else the score-based ``_DepthController``.  MiniCPM5
    carries its DSpark as a *separate* drafter (DFlashDrafter), so
    ``_dspark_host`` is None and the score-based controller would be used.

    This rewrites the controller-creation block of ``_post_init_mtp`` (via
    ``inspect.getsource`` + ``exec``) so a llama model flagged with
    ``_omlx_minicpm5_dspark`` gets the DSpark-adaptive controller, while
    ``_dspark_host`` / the embedded-DSpark path stay untouched.
    """
    import inspect

    try:
        from omlx.patches.mlx_lm_mtp import batch_generator as bg
    except Exception as exc:  # pragma: no cover
        logger.debug("batch_generator not importable: %s", exc)
        return False

    if getattr(bg, _POST_INIT_PATCHED, False):
        return True

    try:
        src = inspect.getsource(bg._post_init_mtp)
    except (OSError, TypeError) as exc:
        logger.warning("cannot read _post_init_mtp source: %s", exc)
        return False

    # Replace the controller-creation expression so a MiniCPM5 llama model
    # picks the DSpark-adaptive controller.  The marker attribute is set on
    # the model by bind_dspark_drafter(); upstream's _dspark_host stays None
    # for llama so embedded-DSpark paths are never entered.
    old = (
        '            factory = getattr(\n'
        '                _dspark_host(gen_batch.model), "make_mtp_depth_controller", None\n'
        '            )\n'
        '            if factory is not None:\n'
        '                state.controller = factory(depth)\n'
    )
    new = (
        '            factory = getattr(\n'
        '                _dspark_host(gen_batch.model), "make_mtp_depth_controller", None\n'
        '            )\n'
        '            if factory is None:\n'
        '                _m5candidates = [gen_batch.model]\n'
        '                _m5lm = getattr(gen_batch.model, "language_model", None)\n'
        '                if _m5lm is not None and _m5lm is not gen_batch.model:\n'
        '                    _m5candidates.append(_m5lm)\n'
        '                for _m5cand in _m5candidates:\n'
        '                    _m5fac = getattr(\n'
        '                        _m5cand, "_omlx_minicpm5_make_mtp_depth_controller", None\n'
        '                    )\n'
        '                    if _m5fac is not None:\n'
        '                        factory = _m5fac\n'
        '                        break\n'
        '            if factory is not None:\n'
        '                state.controller = factory(depth)\n'
    )
    if old not in src:
        # Upstream may have shifted indentation/whitespace; fail loud rather
        # than silently shipping a score-based controller.
        logger.warning("_post_init_mtp controller block not found; MiniCPM5 "
                       "will use the score-based controller")
        return False

    src = src.replace(old, new)
    # exec in the batch_generator module namespace so all module-level
    # helpers stay in scope.
    ns = bg.__dict__
    exec(compile(src, "<minicpm5_post_init_mtp>", "exec"), ns)
    setattr(bg, _POST_INIT_PATCHED, True)
    logger.info("MiniCPM5 _post_init_mtp DSpark controller routing installed")
    return True


# ---------------------------------------------------------------------------
# 3. Drafter detection and attachment
# ---------------------------------------------------------------------------
def bundled_dspark_path(model_path) -> Optional[Any]:
    """``<model>/<draft-dir>`` when it holds a MiniCPM5 DSpark drafter."""
    from pathlib import Path
    import json

    path = Path(model_path)
    # MiniCPM5 keeps the DSpark drafter either in a ``dspark`` subdir or as a
    # sibling ``MiniCPM5-2B-DSpark`` directory.  Detect via config.json.
    candidates = [
        path / "dspark",
        path / "DSpark",
        path.parent / (path.name + "-DSpark"),
    ]
    for cand in candidates:
        cfg = cand / "config.json"
        if not cfg.is_file():
            continue
        try:
            config = json.loads(cfg.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        archs = set(config.get("architectures") or ())
        if "Qwen3DSparkModel" in archs or str(config.get("model_type")) == "qwen3":
            if any(cand.glob("*.safetensors")):
                return cand
    return None


def is_dspark_drafter(drafter: Optional[Any]) -> bool:
    """True iff a block drafter carries a Markov head (DSpark)."""
    if drafter is None:
        return False
    markov_head = getattr(getattr(drafter, "model", None), "markov_head", None)
    return markov_head is not None


def bind_dspark_drafter(target_model: Any, drafter: Any) -> None:
    """Mark a llama target as carrying the MiniCPM5 DSpark drafter.

    Called after ``attach_drafter`` so the block-drafter path uses the
    DSpark-adaptive depth controller instead of the score-based one.
    """
    if not is_dspark_drafter(drafter):
        return
    language_model = getattr(target_model, "language_model", target_model)
    # ``attach_drafter`` already set ``_omlx_mtp_chain`` and
    # ``_omlx_mtp_depth = drafter.depth``; we only add the MiniCPM5 marker
    # so ``_post_init_mtp`` routes this model's cycles to the DSpark-adaptive
    # controller instead of the score-based one.
    language_model._omlx_minicpm5_make_mtp_depth_controller = (
        make_dspark_depth_controller
    )
    language_model._omlx_minicpm5_dspark = True
    logger.info(
        "MiniCPM5 DSpark drafter bound to %s (block depth %d)",
        type(target_model).__name__,
        int(getattr(drafter, "depth", 1) or 1),
    )


def _patch_attach_drafter() -> bool:
    """Wrap ``attach_drafter`` so MiniCPM5 DSpark markers are stamped.

    Upstream ``attach_drafter`` (in ``omlx.speculative.dflash``) sets the
    chain/depth markers but knows nothing about DSpark's block-level depth
    control.  This wraps it: after the drafter is attached, if it carries a
    Markov head (DSpark), the target's language model gets the
    ``_omlx_minicpm5_make_mtp_depth_controller`` marker that routes
    ``_post_init_mtp`` to the DSpark-adaptive controller.
    """
    try:
        from omlx.speculative.dflash import attach_drafter as _orig_attach
    except Exception as exc:  # pragma: no cover
        logger.debug("speculative.dflash not importable: %s", exc)
        return False

    if getattr(_orig_attach, _ATTACH_PATCHED, False):
        return True

    def _attach(language_model: Any, drafter: Any) -> None:
        _orig_attach(language_model, drafter)
        # The target's language model is what carries the chain markers;
        # wrap the whole target too so the marker is visible to
        # ``_post_init_mtp`` regardless of which object it inspects.
        bind_dspark_drafter(language_model, drafter)

    # Rebind the module function and any consumer that imported it by name
    # (e.g. ``omlx.engine.vlm`` binds ``attach_drafter`` at import time).
    try:
        import omlx.speculative.dflash as _df_mod
        _df_mod.attach_drafter = _attach
    except Exception:
        pass
    try:
        import omlx.engine.vlm as _vlm_mod
        if hasattr(_vlm_mod, "attach_drafter"):
            _vlm_mod.attach_drafter = _attach
    except Exception:
        pass
    setattr(_orig_attach, _ATTACH_PATCHED, True)
    logger.info("MiniCPM5 attach_drafter wrapper installed")
    return True


def apply() -> bool:
    """Install every MiniCPM5 DSpark hook.  Idempotent.  Returns active."""
    _resolve_attention_mask_fn()
    llama_ok = _patch_llama_language()
    depth_ok = _patch_depth_controller()
    post_ok = _patch_post_init_mtp()
    attach_ok = _patch_attach_drafter()
    return llama_ok and depth_ok and post_ok and attach_ok


__all__ = [
    "apply",
    "bind_dspark_drafter",
    "bundled_dspark_path",
    "dspark_initial_depth",
    "dspark_next_block_size",
    "is_dspark_drafter",
    "make_dspark_depth_controller",
]
