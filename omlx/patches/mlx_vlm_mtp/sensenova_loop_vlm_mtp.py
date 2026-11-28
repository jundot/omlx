# SPDX-License-Identifier: Apache-2.0
"""SenseNova-Loop-9B built-in MTP head adapted to the VLM drafter interface.

Adapts the loop9b built-in ``MTPModule`` (attached to ``language_model.mtp``
by ``_patch_loop_mtp_attach``) to the mlx-vlm VLM drafter interface
(``draft_block`` / ``prefill_from_target_hidden`` /
``accept_verified_tokens`` / ``reset`` / ``set_shared_kv`` / ``bind`` /
``config``), so it runs the ``vlm_mtp_enabled=True`` ``MTPBatch`` path.

Background:
- loop9b MTP weights are oQ4e quantized (``model-mtp.safetensors``: U32
  packed + BF16 scales/biases). The stock ``Qwen3_5MTPDraftModel`` uses
  non-quantized ``nn.Linear`` and cannot load quantized weights.
- loop9b ``MTPModule`` (registered by qwen35_vlm_runtime, quantization-aware,
  loaded via ``_remap_root_mtp_weights`` + ``quant_predicate``) already hangs
  on ``language_model.mtp``; it is the **only** module that correctly holds
  the quantized MTP weights.
- So this patch does not reload weights; it wraps the already-loaded
  ``language_model.mtp`` as a VLM drafter, reusing its
  ``__call__(hidden, next_ids, embed_tokens, cache)`` forward.

This patch monkey-patches ``load_vlm_mtp_drafter`` (or provides a dedicated
loader) so that when ``vlm_mtp_draft_model`` points at loop9b itself (or the
special marker ``"__builtin__"``), it builds ``LoopMTPDrafter`` from the
target model's ``language_model.mtp``.

All changes are monkey-patches; the VLM internals are untouched.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

logger = logging.getLogger(__name__)

_APPLIED = False

# SenseNova-Loop model types that carry a built-in MTP head.
_SENSENOVA_LOOP_MODEL_TYPES = ("sensenova_loop", "sensenova_loop_text")

# Marker for "use the target's built-in MTP head".
_BUILTIN_MARKERS = ("__builtin__", "builtin", "self", "loop9b_self")


# ---------------------------------------------------------------------------
# LoopMTPDrafter — VLM drafter interface over loop9b's MTPModule.
# ---------------------------------------------------------------------------
class LoopMTPDrafter(nn.Module):
    """Wrap loop9b's already-loaded MTPModule into the VLM drafter API.

    The drafter does NOT own the MTP weights: it holds a reference to the
    target ``language_model.mtp`` (a quantized-aware MTPModule) and reuses its
    forward via ``self._mtp(hidden, next_ids, embed_tokens, cache)``. The
    target's embedding / LM head are bound by ``bind(target_model)``.

    Implements the subset of the VLM drafter protocol exercised by
    ``mlx_vlm.speculative.mtp._mtp_rounds``:

      * ``config``            (has ``block_size``)
      * ``reset(model)``      (per-row state reset)
      * ``set_shared_kv(...)``
      * ``prefill_from_target_hidden(...)``
      * ``draft_block(...)``
      * ``accept_verified_tokens(...)``
      * ``abort_draft_round()``
      * ``draft_eval_state()``

    ``make_mtp_cache`` is kept for parity but is not required by the VLM
    path (the VLM loop builds its own per-row drafter caches via
    ``reset``).
    """

    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = False
    supports_ragged_batch_acceptance = True

    def __init__(self, mtp_module: nn.Module, config: Any):
        super().__init__()
        self.mtp = mtp_module
        self.config = config
        text_config = getattr(config, "text_config", None)
        if text_config is None:
            text_config = config
        self.hidden_size = int(getattr(text_config, "hidden_size", 4096))

        # Bound target modules (set by bind()).
        self._input_embed: Optional[Callable] = None
        self._input_embed_scale: float = 1.0
        self._lm_head_fn: Optional[Callable] = None
        self._greedy_argmax_fn: Optional[Callable] = None
        self._dtype: Any = mx.float32

        # Per-row mutable state (reset() re-initializes).
        self._cache: List[Any] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._next_position: Any = 0
        self._round_appended: int = 0
        self._draft_round: int = 0
        self._kv_valid_len: Any = 0
        self._position: Any = 0

        # Batch acceptance tracking (used by _mtp_next_block_size).
        self.accept_lens: List[int] = []

    # -- bind / reset ------------------------------------------------------
    def bind(self, target_model: nn.Module) -> "LoopMTPDrafter":
        """Bind target embeddings and LM head for draft logits."""
        lm = getattr(target_model, "language_model", target_model)
        model = getattr(lm, "model", None)
        embed = model
        if embed is not None:
            embed = getattr(embed, "embed_tokens", None)
        if embed is None:
            embed = getattr(lm, "embed_tokens", None)
        self._input_embed = embed
        self._input_embed_scale = float(getattr(model, "embed_scale", 1.0))
        args = getattr(lm, "args", None)
        if (
            args is not None
            and getattr(args, "tie_word_embeddings", False)
            and embed is not None
            and hasattr(embed, "as_linear")
        ):
            self._lm_head_fn = embed.as_linear
        if self._lm_head_fn is None:
            self._lm_head_fn = getattr(lm, "lm_head", None)
        if self._lm_head_fn is None and embed is not None:
            self._lm_head_fn = getattr(embed, "as_linear", None)
        # Pre-resolve the as_linear wrapper once (hot path: _lm_head_fn_call
        # runs per drafted token). Store the raw callable; keep the
        # hasattr(fn, "as_linear") branch only as a defensive fallback.
        fn = self._lm_head_fn
        self._lm_head_linear = getattr(fn, "as_linear", fn) if fn is not None else None
        self._dtype = getattr(model, "dtype", mx.float32)
        return self

    def reset(self, target_model: nn.Module, left_padding: Any = None) -> None:
        """Reset per-row drafter state and (re)bind to the target."""
        del left_padding
        # Keep the mtp module reference; create fresh caches.
        self._cache = self.make_mtp_cache()
        self._seed_token = None
        self._seed_hidden = None
        self._next_position = 0
        self._round_appended = 0
        self._draft_round = 0
        self._kv_valid_len = 0
        self._position = 0
        self.accept_lens = []
        # If not already bound, bind now (draft_block requires it).
        if self._input_embed is None or self._lm_head_fn is None:
            self.bind(target_model)

    def make_mtp_cache(self) -> List[Any]:
        """Fresh KVCache list for each MTP layer (mirrors lm make_mtp_cache)."""
        try:
            from mlx_vlm.models.cache import KVCache
        except Exception:  # pragma: no cover
            from mlx_vlm.models.base import KVCache  # type: ignore
        layers = getattr(self.mtp, "layers", [])
        return [KVCache() for _ in layers]

    # -- shared KV (no-op for built-in head; the target owns the KV) --------
    def set_shared_kv(
        self,
        shared_kv_states: dict,
        kv_offset: Any,
        position: Any = None,
        kv_valid_len: Any = None,
    ) -> None:
        del shared_kv_states
        if kv_valid_len is None:
            kv_valid_len = kv_offset
        self._kv_valid_len = kv_valid_len
        if position is None:
            position = kv_valid_len
        self._position = position

    def draft_eval_state(self) -> Any:
        return [self._seed_token, self._seed_hidden, self._cache]

    # -- position helpers (mirror Qwen3_5MTPDraftModel) ---------------------
    def _draft_start_position(self) -> Any:
        return self._next_position

    def _position_ids(self, step: int = 0, length: int = 1) -> mx.array:
        start = self._draft_start_position()
        pos = mx.arange(length, dtype=mx.int32) + step
        if isinstance(start, int):
            return (pos + start)[None, :]
        if isinstance(start, mx.array):
            return start.astype(mx.int32)[:, None] + pos[None, :]
        return mx.array(start, dtype=mx.int32)[:, None] + pos[None, :]

    # -- forward helpers ----------------------------------------------------
    def _mtp_forward(
        self,
        hidden: mx.array,
        next_token_ids: mx.array,
        cache: List[Any],
    ) -> mx.array:
        """Reuse loop9b's MTPModule forward (lm-style).

        ``self.mtp.__call__(hidden_states, next_token_ids, embed_tokens,
        cache)`` returns the post-norm hidden of the MTP head, which is then
        projected through the target LM head for draft logits.
        """
        if self._input_embed is None:
            raise RuntimeError(
                "LoopMTPDrafter.bind(target_model) must be called before "
                "draft_block() so the drafter can use the target embeddings."
            )
        embed = self._input_embed
        out = self.mtp(hidden, next_token_ids, embed, cache)
        return out

    def _forward_tokens(
        self,
        tokens: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype,
    ) -> mx.array:
        steps = int(tokens.shape[1])
        # Pass raw token ids, not a pre-computed embedding: loop9b's
        # MTPModule.__call__ does ``embeds = embed_tokens(next_token_ids)``
        # internally, so handing it an already-embedded tensor would
        # double-embed and corrupt the fused fc input.
        h = self._mtp_forward(hidden, tokens.astype(token_dtype), self._cache)
        self._next_position = self._next_position + steps
        return h

    def _forward_token(
        self,
        tok: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype,
    ) -> mx.array:
        return self._forward_tokens(tok, hidden, token_dtype)

    # -- logits / sampling --------------------------------------------------
    def _lm_head_fn_call(self, hidden: mx.array) -> mx.array:
        if self._lm_head_fn is None:
            raise RuntimeError("LoopMTPDrafter: lm_head not bound")
        # _lm_head_linear is pre-resolved in bind() (hot path: runs per
        # drafted token). Fall back to resolving as_linear defensively in
        # case bind() was not the code path that set _lm_head_fn.
        fn = self._lm_head_linear
        if fn is None:
            fn = self._lm_head_fn
            if hasattr(fn, "as_linear"):
                fn = fn.as_linear
        return fn(hidden)

    def _greedy_token(self, hidden: mx.array) -> mx.array:
        if self._greedy_argmax_fn is not None:
            token = self._greedy_argmax_fn(hidden)
            if token is not None:
                if token.ndim == 1:
                    token = token[:, None]
                return token
        token = mx.argmax(self._lm_head_fn_call(hidden), axis=-1)
        if token.ndim == 1:
            token = token[:, None]
        return token

    def _set_seed_from_hidden(
        self,
        hidden: mx.array,
        sampler: Callable[[mx.array], mx.array],
        greedy: bool,
    ) -> None:
        if greedy:
            self._seed_token = self._greedy_token(hidden)
        else:
            tok = sampler(self._lm_head_fn_call(hidden))
            if getattr(tok, "ndim", 1) == 1:
                tok = tok[:, None]
            self._seed_token = tok
        self._seed_hidden = hidden

    # -- VLM drafter protocol ----------------------------------------------
    def prefill_from_target_hidden(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        bonus_token: Any,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        if input_ids.shape[1] == 0:
            return
        bonus = self._as_col_token(bonus_token, token_dtype)
        shifted = mx.concatenate([input_ids[:, 1:].astype(token_dtype), bonus], axis=1)
        self._next_position = 0
        h = self._forward_tokens(shifted, hidden[:, : shifted.shape[1], :], token_dtype)
        self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)

    @staticmethod
    def _as_col_token(tok: Any, token_dtype: mx.Dtype) -> mx.array:
        """Normalize a single token to shape [B, 1] for drafter ops."""
        arr = (
            mx.array([[tok]], dtype=token_dtype)
            if isinstance(tok, int)
            else (tok if hasattr(tok, "ndim") else mx.array(tok, dtype=token_dtype))
        )
        if arr.ndim == 0:
            return arr[None, None]
        if arr.ndim == 1:
            return arr[:, None]
        return arr[:, -1:] if arr.shape[1] > 1 else arr

    def draft_block(
        self,
        last_bonus: Any,
        hidden: mx.array,
        cache: Any,
        block_size: int,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> mx.array:
        del cache
        if self._input_embed is None or self._lm_head_fn is None:
            raise RuntimeError(
                "LoopMTPDrafter.bind(target_model) must be called before "
                "draft_block() so the drafter can use the target embeddings."
            )
        tok = self._as_col_token(last_bonus, token_dtype)

        h_prev = hidden
        tokens: List[mx.array] = []
        self._round_appended = 0

        if self._seed_token is not None and self._seed_hidden is not None:
            tok = self._as_col_token(self._seed_token, token_dtype)
            h_prev = self._seed_hidden
            tokens.append(tok)
            self._seed_token = None
            self._seed_hidden = None

        lm_head_call = self._lm_head_fn_call
        while len(tokens) < block_size - 1:
            h_prev = self._forward_token(tok, h_prev, token_dtype)
            self._round_appended += 1
            if greedy:
                tok = self._greedy_token(h_prev)
            else:
                tok = self._as_col_token(sampler(lm_head_call(h_prev)), token_dtype)
            tokens.append(tok)

        self._draft_round += 1
        return mx.concatenate(tokens, axis=1)

    def accept_verified_tokens(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: int,
        new_tokens: List[int],
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        keep_appended = min(int(accepted), self._round_appended)
        trim = self._round_appended - keep_appended
        if trim > 0:
            for cache in self._cache:
                if hasattr(cache, "trim"):
                    cache.trim(trim)
            self._next_position = self._next_position - trim

        token_chunks: List[mx.array] = []
        hidden_chunks: List[mx.array] = []
        for draft_idx in range(keep_appended, int(accepted)):
            token_chunks.append(draft_tokens[:, draft_idx : draft_idx + 1])
            hidden_chunks.append(verify_hidden[:, draft_idx : draft_idx + 1, :])
        if new_tokens:
            token_chunks.append(mx.array([[int(new_tokens[-1])]], dtype=token_dtype))
            hidden_chunks.append(verify_hidden[:, int(accepted) : int(accepted) + 1, :])
        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            h = self._forward_tokens(tokens, hiddens, token_dtype)
            self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)
        self._round_appended = 0
        self.accept_lens.append(int(accepted))

    def abort_draft_round(self) -> None:
        """Drop any partial draft state after a verify abort."""
        self._round_appended = 0
        self._seed_token = None
        self._seed_hidden = None

    # -- batch acceptance (delegates to per-row) ----------------------------
    def accept_verified_tokens_batch(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: List[int],
        new_tokens: List[List[int]],
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        if len(accepted) <= 1:
            self.accept_verified_tokens(
                verify_hidden,
                draft_tokens,
                int(accepted[0]),
                new_tokens[0],
                sampler,
                token_dtype,
                greedy,
            )
            return
        # RoWWise fallback: the built-in head is single-row; loop over rows.
        for i in range(len(accepted)):
            self.accept_verified_tokens(
                verify_hidden[i : i + 1],
                draft_tokens[i : i + 1],
                int(accepted[i]),
                new_tokens[i],
                sampler,
                token_dtype,
                greedy,
            )

    def filter_batch(self, keep: Any) -> None:
        if not isinstance(keep, mx.array):
            keep = mx.array(keep, dtype=mx.int32)
        if self._seed_token is not None:
            self._seed_token = self._seed_token[keep]
        if self._seed_hidden is not None:
            self._seed_hidden = self._seed_hidden[keep]


# ---------------------------------------------------------------------------
# Built-in drafter loading.
# ---------------------------------------------------------------------------
def build_loop_mtp_drafter(target_model: nn.Module) -> Optional[LoopMTPDrafter]:
    """Build a LoopMTPDrafter from the target model's attached MTPModule.

    Returns None (with a warning) if the target has no built-in MTP head.
    """
    lm = getattr(target_model, "language_model", target_model)
    mtp = getattr(lm, "mtp", None)
    if mtp is None:
        logger.warning(
            "loop9b built-in MTP drafter requested but %s has no 'mtp' module; "
            "vlm_mtp toggle will be ignored",
            type(lm).__name__,
        )
        return None
    config = getattr(lm, "args", None)
    if config is None:
        config = getattr(lm, "config", None)
    if config is None:
        # Minimal config with just block_size / hidden_size.
        hidden = 4096
        config = type("LoopConfig", (), {"block_size": 3, "hidden_size": hidden})()
    # _dflash_block_total reads draft_model.config.block_size when the
    # caller passes no draft_block_size. TextConfig may not declare it;
    # default to 3 (the optimal DFlash2/MTP block size for loop9b).
    if getattr(config, "block_size", None) is None:
        try:
            config.block_size = 3
        except Exception:
            config = type("LoopConfig", (), {"block_size": 3, "hidden_size": 4096})()
    drafter = LoopMTPDrafter(mtp, config)
    drafter.bind(target_model)
    logger.info(
        "loop9b built-in MTP drafter built: layers=%d hidden=%d",
        len(getattr(mtp, "layers", [])),
        drafter.hidden_size,
    )
    return drafter


# ---------------------------------------------------------------------------
# Patch: intercept load_vlm_mtp_drafter for built-in markers.
# ---------------------------------------------------------------------------
_BUILTIN_TARGET: Optional[nn.Module] = None



# ---------------------------------------------------------------------------
# Loop LanguageModel.__call__: bypass speculative_verify ValueError.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Loop LanguageModel.__init__: attach MTPModule (bypass subclass guard).
# ---------------------------------------------------------------------------
def _patch_loop_mtp_attach() -> None:
    """Attach ``MTPModule`` to loop9b ``LanguageModel.__init__``.

    ``qwen35_vlm_runtime._patch_vlm_language_model.__init__`` has an
    ``if type(self) is not cls: return original_init(...)`` guard; loop9b's
    ``LanguageModel`` subclasses qwen3_5 ``LanguageModel``, so it is skipped
    and the MTP head is never attached. Attach it here for loop9b.
    """
    try:
        from mlx_vlm.models.sensenova_loop import language as sl_lang
        from mlx_vlm.models.qwen3_5 import language as q35_lang
    except Exception as e:  # pragma: no cover
        logger.debug("loop MTP attach patch skipped: %s", e)
        return

    cls = sl_lang.LanguageModel
    if getattr(cls, "_sl_mtp_attached", False):
        return

    # qwen35_vlm_runtime registered MTPDecoderLayer/MTPModule on q35_lang.
    if not hasattr(q35_lang, "MTPModule"):
        logger.debug("MTPModule not registered (qwen35_vlm_runtime not applied)")
        return

    original_init = cls.__init__

    def _init(self, args, config=None):
        from . import is_mtp_attach_enabled
        from ..mlx_lm_mtp import is_mtp_active

        original_init(self, args, config)
        n_mtp = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
        # Gated by ``is_mtp_attach_enabled()`` so checkpoints that declare
        # mtp_num_hidden_layers > 0 but ship no mtp.* weights (unsloth
        # Qwen3.6 UD MLX builds, issue #1426) don't fail strict load_weights
        # with "Missing N parameters" and silently downgrade to LLM.
        attach_enabled = bool(is_mtp_attach_enabled())
        self._omlx_mtp_decode_enabled = bool(
            n_mtp > 0 and attach_enabled and is_mtp_active()
        )
        if n_mtp > 0 and attach_enabled:
            # loop9b LanguageModel subclasses qwen3_5; use its MTPModule
            # (same attention/mlp structure). Keep the original condition
            # (n_mtp > 0 and attach_enabled) — not _omlx_mtp_decode_enabled —
            # so the module is attached even when is_mtp_active() is False
            # (decode stays off but self.mtp exists for other callers).
            self.mtp = q35_lang.MTPModule(args)
        if self._omlx_mtp_decode_enabled:
            self._omlx_mtp_chain = True
            self._omlx_mtp_batch_rollback = True

    cls.__init__ = _init
    setattr(cls, "_sl_mtp_attached", True)
    logger.debug("loop9b LanguageModel MTPModule attached")


# ---------------------------------------------------------------------------
# Loop LanguageModel: mtp_forward / make_mtp_cache.
# ---------------------------------------------------------------------------
def _patch_loop_mtp_forward() -> None:
    """Add ``mtp_forward`` / ``make_mtp_cache`` to loop9b ``LanguageModel``.

    Mirror ``qwen35_vlm_runtime`` but for loop9b (subclasses qwen3_5
    LanguageModel).
    """
    try:
        from mlx_vlm.models.sensenova_loop import language as sl_lang
    except Exception as e:  # pragma: no cover
        logger.debug("loop mtp_forward patch skipped: %s", e)
        return

    cls = sl_lang.LanguageModel
    if getattr(cls, "_sl_mtp_forward_patched", False):
        return

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        """MTP-head forward (depth-k chain contract)."""
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

    mtp_forward._omlx_lm_head_logits = True
    mtp_forward._omlx_head_cache_only = True
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    setattr(cls, "_sl_mtp_forward_patched", True)
    logger.debug("loop9b LanguageModel mtp_forward/make_mtp_cache added")


def set_builtin_target(target_model: Optional[nn.Module]) -> None:
    """Set the target model used to serve built-in MTP drafter requests."""
    global _BUILTIN_TARGET
    _BUILTIN_TARGET = target_model


def _patch_vlm_mtp_drafter_loader() -> None:
    """Patch ``omlx.speculative.vlm_mtp.load_vlm_mtp_drafter`` so that a
    built-in marker (or a path that resolves to the target model itself) is
    served by ``build_loop_mtp_drafter``.

    The patched loader needs access to the target model; it reads the current
    engine's target via a module-level back-reference set by
    ``set_builtin_target``.
    """
    try:
        import omlx.speculative.vlm_mtp as vlm_mtp
    except Exception as e:  # pragma: no cover
        logger.debug("vlm_mtp patch skipped (module unavailable): %s", e)
        return

    original = vlm_mtp.load_vlm_mtp_drafter

    def _patched(path: str):
        if _BUILTIN_TARGET is not None and (
            path in _BUILTIN_MARKERS
            or (isinstance(path, str) and path.lower() in _BUILTIN_MARKERS)
        ):
            drafter = build_loop_mtp_drafter(_BUILTIN_TARGET)
            if drafter is not None:
                return vlm_mtp.VLMMTPDrafter(drafter, "mtp", "<loop9b-builtin>")
        return original(path)

    if getattr(vlm_mtp.load_vlm_mtp_drafter, "_sl_mtp_loader_patched", False):
        return
    vlm_mtp.load_vlm_mtp_drafter = _patched
    vlm_mtp.load_vlm_mtp_drafter._sl_mtp_loader_patched = True
    logger.debug("vlm_mtp.load_vlm_mtp_drafter patched for loop9b builtin")


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def apply() -> bool:
    """Apply the loop9b built-in MTP VLM drafter patch (idempotent).

    Also registers the loop9b LanguageModel runtime adaptations:
    (1) bypass the ``speculative_verify`` ValueError, (2) attach MTPModule
    to loop9b (qwen35_vlm_runtime's ``type(self) is not cls`` guard skips
    loop9b subclasses), and (3) add ``mtp_forward``/``make_mtp_cache``. These
    used to live in ``sensenova_loop_vlm_runtime`` (deleted); they belong to
    the MTP file only, so DFlash2 and MTP no longer share a runtime file.
    """
    global _APPLIED
    if _APPLIED:
        return True
    # qwen35_vlm_runtime registers MTPDecoderLayer/MTPModule on qwen3_5;
    # loop9b's LanguageModel subclasses qwen3_5 and borrows those classes.
    try:
        from omlx.patches.mlx_vlm_mtp.qwen35_vlm_runtime import apply as apply_q35
        apply_q35()
    except Exception as e:  # pragma: no cover
        logger.debug("qwen35_vlm_runtime.apply failed: %s", e)
    # The loop9b 56-slot execution plan visits SSM slots multiple times per
    # block, breaking the upstream ArraysCache speculation assumptions
    # (history capacity, strict validation, state indexing). These live in
    # the shared module (used by both DFlash2 and MTP) so neither drafter
    # depends on the other. Install them for the MTP path (idempotent,
    # do NOT install DFlash2's separate call patch; the shared module's
    # _patch_loop_language_model_call covers both with independent flags.
    try:
        from omlx.patches.sensenova_loop_spec_common import (
            _patch_loop_language_model_call,
            _patch_loop_verify_capacity,
            _patch_loop_validate_speculation,
        )
        _patch_loop_verify_capacity()
        _patch_loop_validate_speculation()
    except Exception as e:  # pragma: no cover
        logger.debug("loop multi-visit spec patches (MTP) failed: %s", e)
    # LanguageModel.__call__ bypass lives in the shared module; MTP uses
    # its own guard flag so DFlash2 and MTP patches coexist independently.
    _patch_loop_language_model_call("_sl_spec_verify_bypassed")
    _patch_loop_mtp_attach()
    _patch_loop_mtp_forward()
    _patch_vlm_mtp_drafter_loader()
    _APPLIED = True
    logger.info("sensenova_loop_vlm_mtp.apply() applied")
    return True
