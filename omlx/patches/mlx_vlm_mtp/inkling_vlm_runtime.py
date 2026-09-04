# SPDX-License-Identifier: Apache-2.0
"""Runtime Lightning MTP for the vendored Inkling VLM.

Single-checkpoint runtime for the checkpoint's multi-depth MTP head
(``mtp_config.num_nextn_predict_layers``, 8 on Inkling Small), each block
being embed/hidden RMSNorms + a 2H->H input projection + one dense
``InklingDecoderLayer``.

The cycle semantics follow vLLM's Inkling MTP (vllm-project/vllm#48768),
NOT mlx-vlm's separate-checkpoint ``speculative/drafters/inkling_mtp``
drafter. The drafter only ever folds block 0 and restores a pre-round
snapshot, so blocks 1..7 run every draft step with an EMPTY cache — that
collapses deep-depth agreement (measured 0.68/0.45/0.19/0.00 per depth).
The reference semantics keep EVERY block stateful over the whole
accepted sequence, and a cycle is k chained passes over one uniform
token window (measured 0.87/0.76/0.67/0.61 with the same head weights):

* pass j runs block j over the window, consuming pass j-1's full-window
  hidden output and the token stream left-shifted by one with the newly
  sampled draft appended; positions stay in trunk coordinates;
* block j's cache row at slot p is valid only while its input token
  index p+1+j stays committed, so each fold trims every active block
  back to the uniform window start
  ``w0 = min_j min(end_j, F_new-1-j)`` and refolds the gap from a small
  ring of recent (token, trunk-hidden) pairs — the incremental
  equivalent of vLLM's snapshot re-prefill of ``num_rejected-1`` slots;
* the embedding side is DOUBLE-normed:
  ``block.embed_norm(backbone.embed_norm(embed(ids)))``. Feeding the raw
  embedding drops depth-0 agreement 0.87 -> 0.58 (vLLM documents the
  same cliff);
* head logits skip the trunk final norm by default
  (``chain_hidden_post_norm: false`` — raw block output through the LM
  head with the muP divide). ``OMLX_INKLING_MTP_FINAL_NORM=trunk``
  restores the trunk-norm readout (measured equivalent).

All cycle bookkeeping lives on the ``_InklingMTPCacheList`` instance the
batch generator threads through the cycle, so request switches isolate
naturally. ``mtp_begin_cycle`` (called by the generator before the fold)
resets the pass counter; there is no per-cycle head clone
(``_omlx_mtp_head_clone`` False) — provisional draft rows append to the
persistent per-block caches and the next fold's trim removes whatever
verify rejected. Head-block conv caches keep a ROLLING input stash
(``_omlx_stash_rows``) so those rewinds can reach across cycles.

Backbone verify rollback is unchanged: the vendored conv stashes its
padded input every forward and ``rollback_speculative_cache`` reslices
the state at the accepted boundary while trimming the KV subs.

The head consumes PRE-norm trunk hidden (per-block ``hidden_norm``,
``chain_hidden_post_norm: false``), signalled to the fold via
``_omlx_mtp_head_prenorm``.

Prompt priming captures (token, pre-norm hidden) pairs into a sliding
window ring during prefill (no head forwards), then ``mtp_take_primed``
folds all active blocks chained over that window at activation
(``OMLX_INKLING_MTP_PRIME_WINDOW``, default 1024: sliding head blocks
warm fully past 512 rows; the two global blocks cover their full
rel_extent=1024 band — only band-outside long-range keys are lost).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_APPLIED = False
_SANITIZE_APPLIED = False

# Ring of recent committed (token, trunk-hidden) pairs kept for gap
# refolds. Bounds the fold window; beyond it deep blocks accept a stale
# clamp (draft quality only — verify guarantees output exactness).
_RING = 64
# Rolling conv-input stash coverage for head blocks: max rewind = ring
# + max chain depth margin (K-1 state rows are added at cache creation).
_STASH_ROWS = _RING + 16
# Keepalive: when any reachable block's committed rows lag the frontier
# by this many pairs, the next fold refolds ALL reachable blocks over the
# ring window (committed-only chained passes). Bounds reactivation cost
# and keeps the controller's deep-depth probes honest — without it a
# depth probe after a shallow cruise pays the whole gap refold (polluting
# t_est) and drafts from stale rows (polluting the accept estimates).
_KEEPALIVE_LAG = 32

_ATTN = {
    "wq_du": "q_proj",
    "wk_dv": "k_proj",
    "wv_dv": "v_proj",
    "wr_du": "r_proj",
    "wo_ud": "o_proj",
}


class _InklingMTPCacheList(list):
    """Persistent per-block MTP cache list + cycle bookkeeping.

    All mutable cycle state lives here (not on the model): the batch
    generator creates a fresh list per request, so request switches and
    ``_drop_mtp_state`` isolate the bookkeeping automatically.

    Logical coordinates count committed PAIRS (pair p = (trunk hidden at
    token p, token p+1)). ``frontier`` is the committed pair count F;
    ``pos_base`` is the logical index of every block's buffer row 0 (all
    active blocks share one buffer origin — priming folds them together
    and unprimed activation starts them all empty). Block j's logical
    end is ``pos_base + kv_j.offset``; its rows past ``F - 1 - j`` were
    computed from draft tokens and are provisional until the next fold's
    trim.
    """

    frontier: int = 0
    # Per-block logical index of buffer row 0. Blocks usually share one
    # origin, but a keepalive RESET restarts a hopelessly lagging block
    # at the current window start, moving only its base — attention is
    # buffer-relative (shift-invariant distances), so bases may diverge.
    base = None  # list[int]
    # -1 => the next mtp_forward call is the cycle fold (pass 0);
    # mtp_begin_cycle resets it, chain calls increment it.
    chain_step: int = -1
    cycle_depth: int = 1
    active_max: int = 1
    sconv_k: int = 4
    fold_keepalive: bool = False
    # Per-block committed-valid row counts (logical). Rows past this were
    # written from draft tokens and stay provisional until refolded —
    # tracked explicitly because a block skipped by lower-depth cycles
    # keeps stale trailing rows that ``frontier`` growth would otherwise
    # falsely legitimize.
    valid_rows = None  # list[int]
    ring_tok = None  # (1, r) uint32, committed t_{p+1}, tail of history
    ring_hid = None  # (1, r, H) trunk pre-norm hidden at p
    win_tok = None  # (1, L) current cycle token stream
    win_hid = None  # (1, L, H) previous pass full-window output
    clamp_logged: bool = False


@dataclass
class _InklingPrimeCtx:
    """Sliding-window prompt capture (no head forwards until activation)."""

    tok_chunks: List[Any] = field(default_factory=list)  # (1, s) each
    hid_chunks: List[Any] = field(default_factory=list)  # (1, s, H) each
    total: int = 0
    pending_hidden: Optional[Any] = None
    expected_offset: int = 0
    valid: bool = True


def _prime_window() -> int:
    try:
        return int(os.environ.get("OMLX_INKLING_MTP_PRIME_WINDOW", "1024"))
    except ValueError:
        return 1024


_FINAL_NORM_MODE: Optional[str] = None


def _final_norm_mode() -> str:
    global _FINAL_NORM_MODE
    if _FINAL_NORM_MODE is None:
        mode = os.environ.get("OMLX_INKLING_MTP_FINAL_NORM", "none").lower()
        _FINAL_NORM_MODE = mode if mode in ("none", "trunk") else "none"
    return _FINAL_NORM_MODE


def _map_transformer_block(sub: str, v: mx.array) -> dict:
    """Checkpoint ``transformer_block.<sub>`` -> InklingDecoderLayer names.

    Port of the upstream drafter's mapping (drafters/inkling_mtp)."""
    out = {}
    p = "transformer_block."
    from mlx_vlm.models.inkling.inkling import _to_mlx_sconv_weight

    if sub.startswith("attn."):
        name, leaf = sub[len("attn.") :].rsplit(".", 1)
        if name in _ATTN:
            out[p + f"self_attn.{_ATTN[name]}.{leaf}"] = v
        elif name in ("q_norm", "k_norm"):
            out[p + f"self_attn.{name}.{leaf}"] = v
        elif name in ("k_sconv", "v_sconv"):
            if leaf == "weight":
                v = _to_mlx_sconv_weight(v)
            out[p + f"self_attn.{name}.conv.{leaf}"] = v
        elif name == "rel_logits_proj":
            out[p + "self_attn.rel_proj"] = v
        else:
            out[p + "self_attn." + name + "." + leaf] = v
    elif sub == "attn_norm.weight":
        out[p + "input_layernorm.weight"] = v
    elif sub == "mlp_norm.weight":
        out[p + "post_attention_layernorm.weight"] = v
    elif sub == "attn_sconv.weight":
        out[p + "attn_sconv.conv.weight"] = _to_mlx_sconv_weight(v)
    elif sub == "mlp_sconv.weight":
        out[p + "mlp_sconv.conv.weight"] = _to_mlx_sconv_weight(v)
    elif sub.startswith("mlp."):
        from mlx_vlm.models.inkling.inkling import _split_gate_up

        m = sub[len("mlp.") :]
        mp = p + "mlp."
        if m == "w13_dn.weight":
            g, u = _split_gate_up(v)
            out[mp + "gate_proj.weight"] = g
            out[mp + "up_proj.weight"] = u
        elif m == "w2_md.weight":
            out[mp + "down_proj.weight"] = v
        elif m in ("gate.global_scale", "global_scale"):
            out[mp + "global_scale"] = v
        else:
            out[mp + m] = v
    else:
        out[p + sub] = v
    return out


def _mtp_sanitize_hook(key: str, value: mx.array) -> dict:
    """Map a raw ``model.mtp.layers.N.*`` checkpoint key onto the attached
    ``language_model.mtp.blocks.N.*`` parameter names."""
    if key.startswith("language_model.mtp."):
        return {key: value}
    for prefix in ("model.mtp.layers.", "mtp.layers."):
        if key.startswith(prefix):
            i, sub = key[len(prefix) :].split(".", 1)
            base = f"language_model.mtp.blocks.{i}."
            if sub in (
                "embed_norm.weight",
                "hidden_norm.weight",
                "input_proj.weight",
            ):
                return {base + sub: value}
            if sub.startswith("transformer_block."):
                tb = sub[len("transformer_block.") :]
                return {
                    base + nk: nv for nk, nv in _map_transformer_block(tb, value).items()
                }
            return {base + sub: value}
    return {}


def apply_sanitize() -> bool:
    """Install the sanitize hook that keeps and remaps mtp.* weights.

    Called from ``apply_mlx_vlm_mtp_patch`` (oQ preserve_mtp path) and
    from ``apply()`` below. Requires the inkling compat vendor namespace.
    """
    global _SANITIZE_APPLIED
    if _SANITIZE_APPLIED:
        return True
    try:
        from omlx.patches.mlx_vlm_inkling_compat import (
            apply_mlx_vlm_inkling_compat_patch,
        )

        apply_mlx_vlm_inkling_compat_patch()
        import mlx_vlm.models.inkling.inkling as inkling_mod
    except Exception as e:
        logger.debug(f"inkling module not importable for MTP sanitize: {e}")
        return False
    inkling_mod._OMLX_MTP_SANITIZE_HOOK = _mtp_sanitize_hook
    _SANITIZE_APPLIED = True
    return True


def apply() -> bool:
    """Apply the inkling runtime Lightning MTP patches. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True

    if not apply_sanitize():
        return False
    try:
        import mlx_vlm.models.inkling as inkling_pkg
        import mlx_vlm.models.inkling.language as inkling_lang
    except Exception as e:
        logger.debug(f"inkling language module not importable: {e}")
        return False

    _patch_model_config(inkling_pkg)
    _register_mtp_classes(inkling_lang)
    _patch_language_model(inkling_lang)
    _patch_inner_model_capture(inkling_lang)

    # Shared VLMModelAdapter pass-throughs (idempotent).
    from .qwen35_vlm_runtime import _patch_vlm_model_adapter

    _patch_vlm_model_adapter()

    _APPLIED = True
    logger.info("mlx-vlm inkling runtime MTP patch applied")
    return True


def _patch_model_config(inkling_pkg: Any) -> None:
    """Carry the top-level ``mtp_config`` block onto the TextConfig.

    Two hops are required: ``BaseModelConfig.from_dict`` filters by
    dataclass signature (dropping ``mtp_config``), and mlx-vlm's
    ``load_model`` REBUILDS ``text_config`` afterwards via
    ``update_module_configs`` → ``TextConfig.from_dict``, discarding any
    attribute stamped on the first instance. So the ModelConfig wrap
    injects the flattened keys into the raw ``text_config`` dict (the
    same object update_module_configs reads), and the TextConfig wrap
    lifts them onto every rebuilt instance."""
    cls = inkling_pkg.ModelConfig
    if not getattr(cls, "_omlx_mtp_from_dict_patched", False):
        original_from_dict = cls.from_dict.__func__

        def patched_from_dict(cls_inner, params):
            mtp_cfg = (params or {}).get("mtp_config") or {}
            depth = int(
                mtp_cfg.get("num_nextn_predict_layers")
                or mtp_cfg.get("num_mtp_layers")
                or 0
            )
            if depth and isinstance(params.get("text_config"), dict):
                params["text_config"].setdefault("mtp_num_hidden_layers", depth)
                params["text_config"].setdefault(
                    "mtp_local_layer_ids", mtp_cfg.get("local_layer_ids")
                )
            return original_from_dict(cls_inner, params)

        cls.from_dict = classmethod(patched_from_dict)
        cls._omlx_mtp_from_dict_patched = True

    text_cls = inkling_pkg.TextConfig
    if not getattr(text_cls, "_omlx_mtp_from_dict_patched", False):
        original_text_from_dict = text_cls.from_dict.__func__

        def patched_text_from_dict(cls_inner, params):
            instance = original_text_from_dict(cls_inner, params)
            params = params or {}
            instance.mtp_num_hidden_layers = int(
                params.get("mtp_num_hidden_layers", 0) or 0
            )
            instance.mtp_local_layer_ids = params.get("mtp_local_layer_ids")
            return instance

        text_cls.from_dict = classmethod(patched_text_from_dict)
        text_cls._omlx_mtp_from_dict_patched = True


def _register_mtp_classes(inkling_lang: Any) -> None:
    if hasattr(inkling_lang, "InklingMTPModule"):
        return

    from dataclasses import replace

    InklingDecoderLayer = inkling_lang.InklingDecoderLayer
    qkvr_fusion_enabled = inkling_lang.qkvr_fusion_enabled

    class InklingMTPBlock(nn.Module):
        def __init__(self, config, layer_idx: int, *, qkvr_fused: bool):
            super().__init__()
            self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.input_proj = nn.Linear(
                2 * config.hidden_size, config.hidden_size, bias=False
            )
            self.transformer_block = InklingDecoderLayer(
                config, layer_idx, qkvr_fused=qkvr_fused
            )

    class InklingMTPModule(nn.Module):
        """Per-depth MTP blocks (upstream drafter layout, trunk-bound)."""

        def __init__(self, text_config):
            super().__init__()
            n = int(getattr(text_config, "mtp_num_hidden_layers", 0) or 0)
            local = set(getattr(text_config, "mtp_local_layer_ids", None) or [])
            layer_config = replace(
                text_config,
                num_hidden_layers=n,
                layer_types=[
                    "hybrid_sliding" if i in local else "hybrid" for i in range(n)
                ],
                mlp_layer_types=["dense"] * n,
                local_layer_ids=None,
            )
            self.blocks = [
                InklingMTPBlock(
                    layer_config,
                    i,
                    qkvr_fused=qkvr_fusion_enabled(text_config, i, mtp=True),
                )
                for i in range(n)
            ]

        def __call__(
            self,
            hidden_states: mx.array,
            next_token_ids: mx.array,
            embed_tokens,
            cache,
            block_idx: int = 0,
        ) -> mx.array:
            # block_idx defaults to 0 for generic callers that use the
            # qwen-style 4-arg head signature (oQe's MTP-head imatrix
            # pass); the chain routing in mtp_forward always passes it.
            block = self.blocks[block_idx]
            token_embed = embed_tokens(next_token_ids)
            h = mx.concatenate(
                [block.hidden_norm(hidden_states), block.embed_norm(token_embed)],
                axis=-1,
            )
            h = block.input_proj(h)
            block_cache = cache[block_idx] if cache is not None else None
            return block.transformer_block(h, cache=block_cache)

    inkling_lang.InklingMTPBlock = InklingMTPBlock
    inkling_lang.InklingMTPModule = InklingMTPModule


def _patch_language_model(inkling_lang: Any) -> None:
    cls = inkling_lang.LanguageModel
    if "_omlx_mtp_runtime_patched" in cls.__dict__:
        return

    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

    original_init = cls.__init__
    original_call = cls.__call__

    def __init__(self, config):
        from . import is_mtp_attach_enabled
        from ..mlx_lm_mtp import is_mtp_active

        original_init(self, config)
        n_mtp = int(getattr(config, "mtp_num_hidden_layers", 0) or 0)
        attach_enabled = bool(is_mtp_attach_enabled())
        self._omlx_mtp_decode_enabled = bool(
            n_mtp > 0 and attach_enabled and is_mtp_active()
        )
        if n_mtp > 0 and attach_enabled:
            self.mtp = inkling_lang.InklingMTPModule(config)
        if self._omlx_mtp_decode_enabled:
            import weakref

            from ..mlx_lm_mtp import get_mtp_depth

            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = min(int(get_mtp_depth()), n_mtp)
            # No per-cycle clone: provisional draft rows append to the
            # persistent per-block caches and the next fold trims them
            # (vLLM re-prefill equivalent). The generator's
            # _mtp_head_trim_to no-ops on CacheList entries (no .offset).
            self._omlx_mtp_head_clone = False
            # Row-wise batch MTP threads per-row extract/merge state the
            # multi-block window bookkeeping does not model.
            self._omlx_mtp_rowwise_unsupported = True
            # The head normalizes its hidden input per block
            # (chain_hidden_post_norm=false) — fold must NOT trunk-norm.
            self._omlx_mtp_head_prenorm = True
            # Prompt-priming capture runs inside InklingModel.__call__,
            # which has no reference back to this LanguageModel; without
            # priming the head enters decode cold (its block-0 cache never
            # saw the prompt) and the adaptive controller parks the
            # request at depth 0 before the fold can warm it up.
            self.model._omlx_mtp_prime_host = weakref.ref(self)

    def __call__(self, inputs=None, cache=None, **kwargs):
        # Prompt-priming capture must only see prefill forwards. Verify /
        # activation forwards pass return_hidden=True; capturing those
        # breaks the ctx offset-contiguity chain (the v1 primed=0 bug),
        # so gate the inner-model capture hook on this flag.
        self.model._omlx_prime_capture_ok = not kwargs.get("return_hidden", False)
        out = original_call(self, inputs, cache=cache, **kwargs)
        if (
            kwargs.get("return_hidden")
            and cache is not None
            and getattr(self, "_omlx_mtp_decode_enabled", False)
            and inputs is not None
            and getattr(out, "gdn_states", None) is None
        ):
            # Rollback marker: the conv-input stashes written during this
            # forward cover exactly these verify positions.
            out.gdn_states = {"verify_len": int(inputs.shape[1])}
        return out

    def _mtp_readout(self, h_last):
        # chain_hidden_post_norm=false: raw block output through the LM
        # head (muP divide + vocab clip live in _logits_from_norm).
        if _final_norm_mode() == "trunk":
            return self.speculative_logits_from_hidden(h_last)
        return self._logits_from_norm(h_last)

    def _mtp_run_block(self, block_idx, block_cache, hid_win, tok_win):
        block = self.mtp.blocks[block_idx]
        # Double-normed embedding side: backbone embed_norm (model.embed)
        # then the block's own embed_norm. Raw embeddings collapse
        # depth-0 agreement 0.87 -> 0.58.
        emb = self.model.embed(tok_win)
        x = mx.concatenate(
            [block.hidden_norm(hid_win), block.embed_norm(emb)], axis=-1
        )
        x = block.input_proj(x)
        return block.transformer_block(x, cache=block_cache)

    def mtp_begin_cycle(self, mtp_cache, depth):
        if not isinstance(mtp_cache, _InklingMTPCacheList) or not len(mtp_cache):
            return
        mtp_cache.cycle_depth = max(1, min(int(depth), len(self.mtp.blocks)))
        mtp_cache.chain_step = -1

    def _mtp_chained_fold(self, cache, hid_win, tok_win, blocks_n):
        """Committed-only chained fold (the priming shape): pass j covers
        the window minus its last j slots, token stream shifted by j."""
        win = hid_win
        length = int(tok_win.shape[1])
        for j in range(blocks_n):
            cols = length - j
            if cols <= 0:
                break
            win = self._mtp_run_block(j, cache[j], win[:, :cols], tok_win[:, j:])
        return win

    def _mtp_keepalive(self, cache, blocks_n):
        """Refold every reachable block over the ring window so none lags
        the frontier. Blocks whose committed rows cannot reach the window
        start (stale below it, unrewindable, or holed) are RESET: their
        cache restarts fresh at the window start with its own base."""
        f = cache.frontier
        k = cache.sconv_k
        ring_len = int(cache.ring_tok.shape[1])
        w0 = max(min(cache.valid_rows[:blocks_n]), f - ring_len, 0)
        for j in range(blocks_n):
            kv, conv = cache[j][0], cache[j][1]
            end_j = cache.base[j] + kv.offset
            r = end_j - w0
            stash = getattr(conv, "_omlx_verify_xp", None)
            cap = 0
            if stash:
                cap = min(int(x.shape[1]) for x in stash.values()) - (k - 1)
            if r == 0 and cache.valid_rows[j] >= w0:
                continue
            if r > 0 and r <= cap and cache.valid_rows[j] >= w0:
                for idx, roll in stash.items():
                    trimmed = roll[:, : roll.shape[1] - r, :]
                    conv[idx] = trimmed[:, -(k - 1) :, :]
                    stash[idx] = trimmed
                kv.trim(r)
                continue
            # Reset: stale rows below the window, a hole, or an
            # unrewindable gap. Fresh start at w0 in this block's own
            # base frame (draft quality only — verify stays exact).
            if kv.offset:
                kv.trim(kv.offset)
            for idx in range(len(getattr(conv, "cache", []) or [])):
                conv[idx] = None
            if stash:
                stash.clear()
            cache.base[j] = w0
        length = f - w0
        self._mtp_chained_fold(
            cache,
            cache.ring_hid[:, -length:],
            cache.ring_tok[:, -length:],
            blocks_n,
        )
        for j in range(blocks_n):
            cache.valid_rows[j] = max(cache.valid_rows[j], f - 1 - j)
        cache.fold_keepalive = True

    def _mtp_fold(self, cache, hidden_states, next_token_ids):
        """Cycle pass 0: trim provisional rows, refold the gap, run block 0
        over the uniform window ``[w0, F_new)``."""
        n = int(next_token_ids.shape[1])
        if cache.valid_rows is None:
            cache.valid_rows = [0] * len(cache)
        if cache.base is None:
            cache.base = [0] * len(cache)

        # Keepalive check on the OLD frontier, before the ring moves.
        reach = max(1, min(int(cache.active_max), len(cache)))
        if reach > 1 and cache.frontier > 0 and cache.ring_tok is not None:
            lag = max(
                (cache.frontier - 1 - j) - cache.valid_rows[j]
                for j in range(reach)
            )
            if lag >= _KEEPALIVE_LAG:
                self._mtp_keepalive(cache, reach)

        tok_row = next_token_ids.reshape(1, n)
        if cache.ring_tok is None:
            cache.ring_tok = tok_row
            cache.ring_hid = hidden_states
        else:
            cache.ring_tok = mx.concatenate([cache.ring_tok, tok_row], axis=1)
            cache.ring_hid = mx.concatenate(
                [cache.ring_hid, hidden_states], axis=1
            )
        if cache.ring_tok.shape[1] > _RING:
            cache.ring_tok = cache.ring_tok[:, -_RING:]
            cache.ring_hid = cache.ring_hid[:, -_RING:]

        f_new = cache.frontier + n
        active = max(1, min(cache.cycle_depth, len(cache)))
        k = cache.sconv_k
        w0 = f_new - 1
        floor = f_new - int(cache.ring_tok.shape[1])
        for j in range(active):
            kv, conv = cache[j][0], cache[j][1]
            w0 = min(w0, cache.valid_rows[j])
            stash = getattr(conv, "_omlx_verify_xp", None)
            if kv.offset:
                cap = 0
                if stash:
                    cap = min(int(x.shape[1]) for x in stash.values()) - (k - 1)
                floor = max(floor, cache.base[j] + kv.offset - cap)
        if floor > w0:
            # Ring/stash-bounded clamp: deep blocks keep some stale
            # provisional rows. Draft quality only — verify stays exact.
            # Rare with keepalive bounding the lags.
            if not cache.clamp_logged:
                logger.debug(
                    "inkling MTP fold clamped: w0 %d -> %d", w0, floor
                )
                cache.clamp_logged = True
            w0 = floor
        for j in range(active):
            kv, conv = cache[j][0], cache[j][1]
            r = cache.base[j] + kv.offset - w0
            if r <= 0:
                continue
            stash = conv._omlx_verify_xp
            for idx, roll in stash.items():
                trimmed = roll[:, : roll.shape[1] - r, :]
                conv[idx] = trimmed[:, -(k - 1) :, :]
                stash[idx] = trimmed
            kv.trim(r)

        length = f_new - w0
        win_tok = cache.ring_tok[:, -length:]
        win_hid = cache.ring_hid[:, -length:]
        out = self._mtp_run_block(0, cache[0], win_hid, win_tok)
        for j in range(active):
            # Conservative committed bound: the window's trailing j rows
            # of pass j come from draft tokens; everything before is
            # committed after this cycle's passes.
            cache.valid_rows[j] = max(cache.valid_rows[j], f_new - 1 - j)
        cache.win_tok = win_tok
        cache.win_hid = out
        cache.chain_step = 1
        cache.frontier = f_new
        # Materialize the cycle bookkeeping (KV buffers, conv states, stash
        # rolls, ring) without a host sync. Nothing else ever evaluates the
        # stash rolls or trimmed states, so left lazy they accumulate a
        # deepening cross-cycle graph that every eval re-walks (measured
        # 26-59 ms/cycle vs 9 ms materialized). Covers last cycle's chain
        # appends too — this runs after the trims that consume them.
        evals = [cache.ring_hid, cache.ring_tok]
        for j in range(active):
            kv, conv = cache[j][0], cache[j][1]
            if kv.keys is not None:
                evals.append(kv.keys)
                evals.append(kv.values)
            slots = getattr(conv, "cache", None)
            if isinstance(slots, list):
                evals.extend(a for a in slots if a is not None)
            stash = getattr(conv, "_omlx_verify_xp", None)
            if stash:
                evals.extend(stash.values())
        mx.async_eval(evals)
        return out

    def _mtp_chain(self, cache, next_token_ids, step):
        """Cycle pass j: shift the token stream left, append the newest
        draft, run block j chained on pass j-1's window output."""
        block_idx = min(step, len(cache) - 1)
        cache.chain_step = step + 1
        cache.win_tok = mx.concatenate(
            [cache.win_tok[:, 1:], next_token_ids.reshape(1, 1)], axis=1
        )
        out = self._mtp_run_block(
            block_idx, cache[block_idx], cache.win_hid, cache.win_tok
        )
        cache.win_hid = out
        return out

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        step = getattr(mtp_cache, "chain_step", -1)
        if step < 0:
            head_hidden = self._mtp_fold(mtp_cache, hidden_states, next_token_ids)
        else:
            head_hidden = self._mtp_chain(mtp_cache, next_token_ids, step)
        # Only the last slot is ever consumed (fold logits_keep=1, chain
        # samples logits[:, -1]) — readout one row, not the window.
        logits = self._mtp_readout(head_hidden[:, -1:, :])
        if return_hidden:
            return logits, head_hidden
        return logits

    def make_mtp_cache(self):
        if not hasattr(self, "mtp"):
            return _InklingMTPCacheList()
        cache = _InklingMTPCacheList(
            CacheList(KVCache(), ArraysCache(4)) for _ in self.mtp.blocks
        )
        k = int(getattr(self.config, "sconv_kernel_size", 4) or 4)
        cache.sconv_k = k
        cache.valid_rows = [0] * len(cache)
        cache.base = [0] * len(cache)
        # Keepalive reach: the deepest block the controller can ever use.
        cache.active_max = max(
            1, min(int(getattr(self, "_omlx_mtp_depth", 1)), len(cache))
        )
        for cl in cache:
            # Rolling conv-input stash (vendored conv honors this) so the
            # fold prologue can rewind provisional rows across cycles.
            cl[1]._omlx_stash_rows = _STASH_ROWS + k - 1
        return cache

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        """Trim KV subs and reslice conv states at the accepted boundary.

        Replaces the vendored snapshot/replay rollback (an extra backbone
        forward per cycle) with an O(1) reslice from the conv-input
        stashes the vendored ``InklingShortConvolution`` records."""
        if isinstance(accepted, mx.array):
            accepted = int(accepted.max().item()) if accepted.size else 0
        elif not isinstance(accepted, int):
            accepted = max(int(a) for a in accepted)
        verify_len = None
        if isinstance(gdn_states, dict):
            verify_len = gdn_states.get("verify_len")
        if verify_len is None:
            verify_len = int(block_size)
        keep = accepted + 1
        trim_n = int(verify_len) - keep
        for c in caches:
            if c is None:
                continue
            kv, conv = c[0], c[1]
            if trim_n > 0 and hasattr(kv, "trim"):
                kv.trim(trim_n)
            stash = getattr(conv, "_omlx_verify_xp", None)
            if not stash:
                continue
            for idx, xp in stash.items():
                k_minus_1 = int(xp.shape[1]) - int(verify_len)
                if k_minus_1 <= 0:
                    continue
                conv[idx] = xp[:, keep : keep + k_minus_1, :]
        return accepted

    def mtp_take_primed(self, cache, main_tok):
        """Consume the sliding-window prompt capture at MTP activation.

        Folds all active blocks chained over the captured window (pass j
        covers slots ``[0, W_eff-1-j]`` with the token stream shifted by
        j — every input is committed, so no draft handling) and returns
        the seeded ``(cache_list, hist_offset)`` for the generator."""
        from ..mlx_lm_mtp import prompt_priming

        ctx = getattr(self, prompt_priming._CTX_ATTR, None)
        prompt_priming.drop_ctx(self)
        if not isinstance(ctx, _InklingPrimeCtx):
            return None
        if not (ctx.valid and ctx.total > 0 and ctx.pending_hidden is not None):
            return None
        offset = prompt_priming._activation_offset(cache)
        if offset is None or ctx.expected_offset != offset - 1:
            logger.debug(
                "inkling MTP priming discarded: seam offset mismatch (%s vs %s)",
                ctx.expected_offset,
                offset,
            )
            return None
        try:
            seam_tok = main_tok.reshape(1, 1).astype(ctx.tok_chunks[0].dtype)
            tok = mx.concatenate(ctx.tok_chunks + [seam_tok], axis=1)
            hid = mx.concatenate(ctx.hid_chunks + [ctx.pending_hidden], axis=1)
            f = int(tok.shape[1])
            w_eff = min(f, _prime_window())
            tok = tok[:, -w_eff:]
            hid = hid[:, -w_eff:]
            head_cache = self.make_mtp_cache()
            if not len(head_cache):
                return None
            active = head_cache.active_max
            self._mtp_chained_fold(head_cache, hid, tok, active)
            head_cache.frontier = f
            head_cache.base = [f - w_eff] * len(head_cache)
            head_cache.valid_rows = [
                max(0, f - 1 - j) if j < active else 0
                for j in range(len(head_cache))
            ]
            r = min(_RING, w_eff)
            head_cache.ring_tok = tok[:, -r:]
            head_cache.ring_hid = hid[:, -r:]
            evals = []
            for cl in head_cache:
                if cl[0].offset:
                    evals.extend(cl[0].state)
            if evals:
                mx.async_eval(evals)
            return head_cache, f
        except Exception as exc:
            logger.debug("inkling MTP priming fold failed: %s", exc)
            return None

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls._mtp_readout = _mtp_readout
    cls._mtp_run_block = _mtp_run_block
    cls._mtp_chained_fold = _mtp_chained_fold
    cls._mtp_keepalive = _mtp_keepalive
    cls._mtp_fold = _mtp_fold
    cls._mtp_chain = _mtp_chain
    cls.mtp_begin_cycle = mtp_begin_cycle
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_take_primed = mtp_take_primed
    cls.rollback_speculative_cache = rollback_speculative_cache
    cls._omlx_mtp_runtime_patched = True


def _inkling_prime_capture(host, inputs, hidden, cache) -> None:
    """Accumulate (token, pre-norm hidden) pairs into a sliding window.

    Unlike the generic ``prompt_priming.maybe_capture`` this runs NO head
    forwards during prefill — ``mtp_take_primed`` folds all blocks in one
    chained pass at activation. Chunks past the priming window slide out
    instead of invalidating the context, so long prompts stay primeable.
    Contiguity gating mirrors the generic capture (anchor offset chain);
    any mismatch drops the context and degrades to an unprimed entry.
    """
    from ..mlx_lm_mtp import prompt_priming

    if prompt_priming._suppressed() or not prompt_priming.priming_enabled():
        return
    if cache is None or not prompt_priming._host_eligible(host):
        return
    if inputs is None or getattr(inputs, "ndim", 0) != 2 or inputs.shape[0] != 1:
        return
    anchor = prompt_priming._anchor(cache)
    if anchor is None:
        return
    seq_len = int(inputs.shape[1])
    offset_after = anchor.offset

    ctx = getattr(host, prompt_priming._CTX_ATTR, None)
    if ctx is not None and (
        not isinstance(ctx, _InklingPrimeCtx)
        or not ctx.valid
        or ctx.expected_offset != offset_after - seq_len
    ):
        prompt_priming.drop_ctx(host)
        ctx = None
    if ctx is None:
        if seq_len <= 1:
            # A lone decode step cannot start a prompt timeline.
            return
        ctx = _InklingPrimeCtx()
        setattr(host, prompt_priming._CTX_ATTR, ctx)

    if ctx.pending_hidden is not None:
        if seq_len > 1:
            hid_chunk = mx.concatenate(
                [ctx.pending_hidden, hidden[:, :-1]], axis=1
            )
        else:
            hid_chunk = ctx.pending_hidden
        tok_chunk = inputs
    else:
        if seq_len <= 1:
            ctx.pending_hidden = hidden[:, -1:]
            ctx.expected_offset = offset_after
            return
        hid_chunk = hidden[:, :-1]
        tok_chunk = inputs[:, 1:]
    ctx.tok_chunks.append(tok_chunk)
    ctx.hid_chunks.append(hid_chunk)
    ctx.total += int(tok_chunk.shape[1])
    window = _prime_window()
    while ctx.tok_chunks and ctx.total - int(ctx.tok_chunks[0].shape[1]) >= window:
        ctx.total -= int(ctx.tok_chunks.pop(0).shape[1])
        ctx.hid_chunks.pop(0)
    ctx.pending_hidden = hidden[:, -1:]
    ctx.expected_offset = offset_after
    mx.async_eval([hid_chunk, ctx.pending_hidden])


def _patch_inner_model_capture(inkling_lang: Any) -> None:
    """Wrap ``InklingModel.__call__`` to capture prompt chunks for priming.

    The LanguageModel always calls the inner model with
    ``skip_final_norm=True``, so the captured hidden is the PRE-norm trunk
    output — the convention this head consumes. Image prompts arrive via
    ``input_embeddings`` and are skipped (same limitation as qwen); all
    real gating lives in ``_inkling_prime_capture`` and fails safe to an
    unprimed entry.
    """
    cls = inkling_lang.InklingModel
    if getattr(cls, "_omlx_mtp_prime_capture_patched", False):
        return

    original_call = cls.__call__

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        skip_final_norm=False,
        **kwargs,
    ):
        out = original_call(
            self,
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            skip_final_norm=skip_final_norm,
            **kwargs,
        )
        if (
            input_embeddings is None
            and cache is not None
            and skip_final_norm
            and inputs is not None
            and getattr(self, "_omlx_prime_capture_ok", False)
        ):
            host_ref = getattr(self, "_omlx_mtp_prime_host", None)
            host = host_ref() if host_ref is not None else None
            if host is not None:
                try:
                    _inkling_prime_capture(host, inputs, out, cache)
                except Exception:
                    logger.debug(
                        "Inkling MTP prompt-priming capture failed", exc_info=True
                    )
        return out

    cls.__call__ = __call__
    cls._omlx_mtp_prime_capture_patched = True


__all__ = ["apply", "apply_sanitize"]
