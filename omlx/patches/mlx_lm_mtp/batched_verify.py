"""Batched depth-1 MTP (always-advance) for multi-row decode steps.

Solves "MTP deactivation under concurrency": with >=2 requests sharing a
batched decode step, the engine falls back to plain batched decode and the
MTP head idles. This patch runs a rapid-mlx-style always-advance depth-1
cycle on the whole row batch:

  cycle 1 (bootstrap): plain (B,1) frontier forward (identical cost to the
      standard step), stash its logits/hidden as the skip state, no draft.

  steady cycle:
    1. tokens emitted this step were already fed+verified last cycle; frontier
       logits/hidden come from that verify (skip state, zero-cost)
    2. per-row MTP head fold of the pending committed pairs -> draft D
    3. next primary P' = greedy(frontier logits)
    4. ONE (B, 2) verify forward [P', D] with n_confirmed=1
    5. accept iff greedy(verify pos-0 logits) == D  (per row)
    6. rejected rows: full-attn KV per-row right-roll (prepare/finalize) +
       GDN per-row exact state restore (stash replay like
       mtp_partial_rollback); accepted rows advance 2 positions
    7. skip logits/hidden = pos-1 (accept) / pos-0 (reject) per row
    8. deferred draft D queued for accepted rows, emitted right after P'

Head caches start fresh (no prompt priming in this MVP) and build history via
the pending folds — cold-head drafts simply get rejected, which costs one
verify position but never correctness.

Gate: OMLX_MTP_BATCHED_VERIFY=1 (default off). Eligibility: B>=2, no logits
processors, greedy samplers (probe fingerprint), model exposes the MTP
surface, omlx row-wise MTP state absent. Any surprise falls back to the
standard path; every cycle leaves the backbone cache advanced exactly by the
tokens it emitted, so fallback is always safe.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

_ENV = "OMLX_MTP_BATCHED_VERIFY"

_stats = {
    "cycles": 0, "bootstraps": 0, "drafts": 0, "accepts": 0,
    "reject_rollbacks": 0, "gdn_exact_rows": 0, "gdn_pollution_rows": 0,
    "fallbacks": 0, "head_ms": 0.0, "verify_ms": 0.0, "rollback_ms": 0.0,
}


def _enabled() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Greedy sampler fingerprinting: temp==0 samplers are argmax — probe on fixed
# logits and require argmax equality on every probe. Cached per sampler id.
# ---------------------------------------------------------------------------

def _make_probes():
    a = mx.zeros((1, 32))
    b = mx.zeros((1, 32))
    b[0, 7] = 5.0
    c = mx.linspace(-3.0, 3.0, 32)[None, :]
    return (a, b, c)


_PROBES = None


def _sampler_is_greedy(sampler: Any, cache: Dict[int, bool]) -> bool:
    key = id(sampler)
    if key in cache:
        return cache[key]
    global _PROBES
    if _PROBES is None:
        _PROBES = _make_probes()
    ok = bool(getattr(sampler, "_omlx_deterministic", False))
    cache[key] = ok
    return ok


def _mtp_host(model: Any) -> Optional[Any]:
    for cand in (model, getattr(model, "language_model", None),
                 getattr(model, "_language_model", None)):
        if cand is not None and all(
            hasattr(cand, m) for m in ("mtp_forward", "make_mtp_cache", "mtp")
        ):
            return cand
    return None


class _RowCtx:
    __slots__ = ("uid", "head_cache", "hist", "last_d", "carry", "_dbg_h5")

    def __init__(self, uid: Any, head_cache: List[Any]):
        self.uid = uid
        self.head_cache = head_cache
        self.hist = 0
        # entry for the accepted draft of the previous cycle:
        # (hidden at that draft's predecessor (1,H), draft token (1,)) or None
        self.last_d: Optional[Tuple[mx.array, mx.array]] = None
        # priming carry (hidden at last prompt token) pending the seam fold
        self.carry: Optional[mx.array] = None
        self._dbg_h5 = None


class _BatchedVerifyCtx:
    def __init__(self):
        self.rows: Dict[Any, _RowCtx] = {}
        self.skip_logits: Optional[mx.array] = None  # (B, V)
        self.skip_hidden: Optional[mx.array] = None  # (B, H)
        self.greedy_cache: Dict[int, bool] = {}
        self.host: Any = None


def _eligible(gen_batch: Any) -> bool:
    if not _enabled():
        return False
    uids = getattr(gen_batch, "uids", None)
    if not uids or len(uids) < 2:
        return False
    if any(getattr(gen_batch, "logits_processors", None) or []):
        return False
    if getattr(gen_batch, "_omlx_mtp_batch_state", None) is not None:
        return False
    if _mtp_host(gen_batch.model) is None:
        return False
    ctx = getattr(gen_batch, "_omlx_batched_verify_ctx", None)
    if ctx is not None and getattr(ctx, "disabled", False):
        return False  # runtime self-check banned this batch composition
    prompt_cache = getattr(gen_batch, "prompt_cache", None)
    if not prompt_cache or not hasattr(prompt_cache[0], "left_padding"):
        return False
    return True


# ---------------------------------------------------------------------------
# Rejection rollback helpers
# ---------------------------------------------------------------------------

def _trim_rejected_rows(gen_batch: Any, reject_mask: List[bool]) -> bool:
    """Per-row KV rollback via extract→trim→re-merge (the row-MTP path's
    proven mechanism). A name-filtered partial roll proved unsafe: qwen3_5
    full-attn layers use RotatingKVCache variants the filter missed, leaving
    layers at inconsistent lengths (broadcast-shape crashes)."""
    reject_idx = [i for i, r in enumerate(reject_mask) if r]
    if not reject_idx:
        return True
    try:
        from .batch_generator import _replace_cache_rows
    except Exception as exc:
        logger.warning("[batched-verify] import _replace_cache_rows: %s", exc)
        return False
    replacements: Dict[int, List[Any]] = {}
    for r in reject_idx:
        row_caches = gen_batch.extract_cache(r)
        for c in row_caches:
            trim = getattr(c, "trim", None)
            if callable(trim):
                try:
                    if getattr(c, "is_trimmable", lambda: True)():
                        trim(1)
                except Exception:
                    pass
        replacements[r] = row_caches
    try:
        _replace_cache_rows(gen_batch, replacements)
        return True
    except Exception as exc:
        logger.warning("[batched-verify] trim/re-merge failed: %s", exc)
        return False


def _replace_row(arr: mx.array, r: int, row: mx.array) -> mx.array:
    return mx.concatenate([arr[:r], row, arr[r + 1 :]], axis=0)


def _restore_gdn_rows(model: Any, prompt_cache: List[Any],
                      reject_mask: List[bool]) -> bool:
    """Exact per-row GDN state restore: replay the stashed confirmed chunk
    from the pre-forward state (per-row mtp_partial_rollback)."""
    host = _mtp_host(model)
    layers = getattr(getattr(host, "model", None), "layers", None)
    if layers is None or len(layers) != len(prompt_cache):
        return False
    reject_idx = [i for i, r in enumerate(reject_mask) if r]
    if not reject_idx:
        return True
    ok = True
    for layer, c in zip(layers, prompt_cache):
        if not getattr(layer, "is_linear", False):
            continue
        roll_state = getattr(c, "rollback_state", None)
        stash = getattr(c, "_mtp_draft_stash", None)
        proc = getattr(getattr(layer, "linear_attn", None), "_process_chunk", None)
        if roll_state is None or stash is None or not callable(proc):
            ok = False
            continue
        try:
            conv_0, ssm_0 = roll_state
            qkv_s, a_s, b_s = stash
            state = c.state
            conv, ssm = state[0], state[1]
            for r in reject_idx:
                sl = slice(r, r + 1)
                _, conv_m, ssm_m = proc(
                    qkv_s[sl, :1], a_s[sl, :1], b_s[sl, :1],
                    conv_0[sl], ssm_0[sl], None,
                )
                if conv.shape[0] == 1:
                    conv, ssm = conv_m, ssm_m
                else:
                    conv = _replace_row(conv, r, conv_m)
                    ssm = _replace_row(ssm, r, ssm_m)
            c.state = (conv, ssm) + tuple(state[2:])
            _stats["gdn_exact_rows"] += len(reject_idx)
        except Exception as exc:
            logger.warning("[batched-verify] GDN restore failed: %s", exc)
            ok = False
    return ok


class _Fallback(Exception):
    pass


def _ensure_mutable_states(prompt_cache: List[Any]) -> None:
    """Normalize tuple state containers back to lists.

    The GDN layer writes its fast weights via ``cache[0] = conv_f`` every
    forward (ArraysCache.__setitem__ → ``self.cache[idx] = ...``). Some
    cache-management paths (SSD-restore wrappers / snapshot round-trips)
    leave the state container as a tuple, which turns that routine write
    into ``'tuple' object does not support item assignment`` — killing the
    stream with a 500. Convert in place before any forward; a list is a
    behavioral superset for every reader.
    """
    for c in prompt_cache or []:
        inner = getattr(c, "_inner", None) or c
        holder = getattr(inner, "cache", None)
        if isinstance(holder, tuple):
            try:
                inner.cache = list(holder)
            except Exception:
                pass


def _model_forward(model: Any, inputs: mx.array, cache: List[Any],
                   n_confirmed: int = 0):
    kwargs: Dict[str, Any] = {"cache": cache, "return_hidden": True}
    if n_confirmed:
        kwargs["n_confirmed"] = n_confirmed
    result = model(inputs, **kwargs)
    # mlx-vlm path (qwen4_exp/flash): LanguageModelOutput dataclass
    if hasattr(result, "logits") and getattr(result, "hidden_states", None) is not None:
        hd = result.hidden_states
        if isinstance(hd, list):
            hd = hd[-1]
        return result.logits, hd
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[1]
    raise _Fallback("model forward returned no hidden states")


def _sample_rows(gen_batch: Any, logits: mx.array) -> mx.array:
    """Deterministic sampling per row through the batch's own samplers
    (omlx wraps temp=0 with token-suppression — raw argmax would diverge)."""
    samplers = list(getattr(gen_batch, "samplers", None) or [])
    B = logits.shape[0]
    if not samplers:
        fb = getattr(gen_batch, "fallback_sampler", None)
        if fb is not None:
            return fb(logits)
        return mx.argmax(logits, axis=-1)
    rows = []
    for i in range(B):
        s_i = samplers[i] or gen_batch.fallback_sampler
        rows.append(s_i(logits[i : i + 1]))
    return mx.concatenate(rows, axis=0)


def _norm_lp(logits: mx.array) -> mx.array:
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


# Head hidden convention: mirror _chain_next_drafts — when the model's
# return_hidden is pre-norm and the head does not norm internally, apply the
# trunk final norm before feeding the MTP head.
try:  # pragma: no cover - import wiring depends on patch load order
    from .batch_generator import _trunk_norm_module as _bg_trunk_norm
    from . import prompt_priming as _pp

    _HEAD_HIDDEN_POST_NORM = bool(getattr(_pp, "HEAD_HIDDEN_POST_NORM", True))
except Exception:  # noqa: BLE001
    _bg_trunk_norm = None
    _HEAD_HIDDEN_POST_NORM = True


def _head_norm(model: Any, host: Any, hidden: mx.array) -> mx.array:
    """Mirror _chain_next_drafts exactly: HEAD_HIDDEN_POST_NORM=True means
    the head consumes POST-norm hidden while return_hidden yields pre-norm
    (qwen3_5 path), so the trunk final norm must be applied."""
    if _bg_trunk_norm is None:
        return hidden
    head_prenorm = getattr(model, "_omlx_mtp_head_prenorm", False) or getattr(
        getattr(model, "_language_model", None), "_omlx_mtp_head_prenorm", False
    )
    if _HEAD_HIDDEN_POST_NORM and not head_prenorm and hidden.ndim == 3:
        try:
            return _bg_trunk_norm(model)(hidden)
        except Exception:
            return hidden
    return hidden


# ---------------------------------------------------------------------------
# Emission helpers (mirror mlx_lm GenerationBatch.next bookkeeping)
# ---------------------------------------------------------------------------

def _emit_one(gen_batch: Any, i: int, uid: Any, token: int,
              lp: Any, responses: List[Any], keep: List[int],
              finished: List[int]) -> None:
    finish_reason = None
    match_sequence = None
    current_state = None
    gen_batch._num_tokens[i] += 1
    gen_batch.tokens[i].append(token)
    if gen_batch._num_tokens[i] >= gen_batch.max_tokens[i]:
        finish_reason = "length"
    (gen_batch._matcher_states[i], match_sequence, current_state) = (
        gen_batch.state_machines[i].match(gen_batch._matcher_states[i], token)
    )
    if match_sequence is not None and current_state is None:
        finish_reason = "stop"
    responses.append(
        gen_batch.Response(
            uid=uid, token=token, logprobs=lp, finish_reason=finish_reason,
            current_state=current_state, match_sequence=match_sequence,
            prompt_cache=gen_batch.extract_cache(i) if finish_reason else None,
            all_tokens=gen_batch.tokens[i] if finish_reason else None,
        )
    )
    if finish_reason is not None:
        finished.append(i)
    else:
        keep.append(i)


def _finish_cycle(gen_batch: Any, ctx: _BatchedVerifyCtx, B: int,
                  keep: List[int], finished: List[int]) -> None:
    live_uids = set(gen_batch.uids)
    for uid in list(ctx.rows):
        if uid not in live_uids:
            del ctx.rows[uid]
    if len(keep) < B:
        gen_batch.filter(keep)
    if len(getattr(gen_batch, "uids", [])) != B:
        # Row set changed (join/leave) — drop skip state so the next cycle
        # re-frontiers with a plain forward (safe seam).
        ctx.skip_logits = None
        ctx.skip_hidden = None


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------

def _batched_next(gen_batch: Any) -> List[Any]:
    ctx: _BatchedVerifyCtx = getattr(
        gen_batch, "_omlx_batched_verify_ctx", None
    )
    if ctx is None:
        ctx = _BatchedVerifyCtx()
        ctx.host = _mtp_host(gen_batch.model)
        gen_batch._omlx_batched_verify_ctx = ctx

    host = ctx.host
    uids = list(gen_batch.uids)
    B = len(uids)

    for uid in uids:
        if uid not in ctx.rows:
            row = None
            # Consume a merge-time adoption (populated by the extend hook —
            # the stash at merge time belongs to the just-prefilled row).
            adopted = _ADOPT_QUEUE.pop(0) if _ADOPT_QUEUE else None
            if adopted is not None:
                cache_i, carry_i = adopted
                row = _RowCtx(uid, cache_i)
                row.carry = carry_i
                _stats["primed_adoptions"] = (
                    _stats.get("primed_adoptions", 0) + 1
                )
            if row is None:
                row = _RowCtx(uid, host.make_mtp_cache())
            ctx.rows[uid] = row

    # Tokens to emit this step (sampled last cycle; fed iff skip state is
    # None, in which case the bootstrap forward below feeds them now).
    emit_tokens = gen_batch._next_tokens
    emit_lps = gen_batch._next_logprobs
    if emit_tokens is None:
        raise _Fallback("no _next_tokens at cycle entry")
    gen_batch._current_tokens = emit_tokens
    gen_batch._current_logprobs = emit_lps
    _ensure_mutable_states(gen_batch.prompt_cache)
    mx.eval(emit_tokens, emit_lps)
    tokens_list = [int(t) for t in emit_tokens.tolist()]

    responses: List[Any] = []
    keep: List[int] = []
    finished: List[int] = []

    # ---- bootstrap cycle: plain forward, no speculation ----
    # Also re-bootstrap whenever the skip state's row count no longer matches
    # the batch (row join/leave since the last cycle) — after a join no row's
    # next token has been fed, so a plain forward is exactly right.
    if ctx.skip_hidden is None or ctx.skip_hidden.shape[0] != B:
        logits_f, hidden_f = _model_forward(
            gen_batch.model, emit_tokens[:, None], gen_batch.prompt_cache
        )
        logits_f = logits_f[:, -1, :]
        hidden_f = hidden_f[:, -1, :]
        lp_f = _norm_lp(logits_f)
        # The one sampler call of the bootstrap: the primary that cycle 2's
        # verify will feed & emit (state-exact: 1 call ↔ 1 emitted token).
        p_next = _sample_rows(gen_batch, logits_f)
        p_next_lp = mx.take_along_axis(
            lp_f, p_next[:, None].astype(mx.int32), axis=-1
        ).squeeze(-1)

        for i, uid in enumerate(uids):
            _emit_one(gen_batch, i, uid, tokens_list[i], emit_lps[i],
                      responses, keep, finished)
            row = ctx.rows.get(uid)
            if row is None:
                continue
            # Primed seam: fold (carry = hidden at last prompt token, first
            # generated token) — the entry the prefill capture could not
            # know (the token didn't exist yet).
            if row.carry is not None:
                try:
                    ph = _head_norm(gen_batch.model, host,
                                    row.carry[None, :, :])
                    host.mtp_forward(
                        ph, emit_tokens[i : i + 1].reshape(1, 1),
                        row.head_cache, logits_keep=1,
                    )
                except Exception as exc:
                    logger.debug("[batched-verify] seam fold failed: %s", exc)
                row.carry = None
            # the emit token's own entry folds at the next steady cycle's
            # pre-draft step (hidden_f is stored as skip_hidden)
        ctx.skip_logits = logits_f
        ctx.skip_hidden = hidden_f
        gen_batch._next_tokens = p_next
        gen_batch._next_logprobs = list(lp_f)
        gen_batch._next_one_lp = p_next_lp
        mx.async_eval(gen_batch._next_tokens)
        _stats["bootstraps"] += 1
        _finish_cycle(gen_batch, ctx, B, keep, finished)
        return responses

    # ---- steady cycle (sampler-exact-once) ----
    # Verify feeds [P', D] where P' IS the token emitted this cycle (sampled
    # once at the previous cycle). Per cycle the sampler is called exactly as
    # many times as tokens are sampled: accept → sampler(v0)(=D) +
    # sampler(v1)(=next primary); reject → sampler(v0)(=next primary). Every
    # call maps to exactly one later-emitted token, in emission order, so
    # stateful suppression/processors stay consistent with plain decode.
    hidden_f = ctx.skip_hidden  # hidden at P' (P' = emit_tokens)
    p_next = emit_tokens

    # per-row head fold -> drafts. The fold chain must cover the CURRENT
    # cycle's emitted token before drafting, so the head predicts the token
    # AFTER it — the same slot the accept oracle (verify pos-0) scores.
    # (dump 2026-08-28: the old chain predicted one position behind.)
    t_head = time.perf_counter()
    drafts: List[int] = []
    for i, uid in enumerate(uids):
        row = ctx.rows[uid]
        pairs_h = []
        pairs_t = []
        if row.last_d is not None:
            pairs_h.append(row.last_d[0])
            pairs_t.append(row.last_d[1])
        pairs_h.append(hidden_f[i : i + 1])          # hidden at emit's pred
        pairs_t.append(emit_tokens[i : i + 1])       # the emit token itself
        try:
            ph = _head_norm(
                gen_batch.model, host, mx.concatenate(pairs_h)[None, :, :]
            )
            ht = mx.concatenate(pairs_t).reshape(1, -1)
            head_logits = host.mtp_forward(
                ph, ht, row.head_cache, logits_keep=1,
            )
            # Draft through the SAME sampler the oracle uses (token
            # suppression flips top picks; a raw-argmax draft mismatches
            # whenever suppression bans the head's first choice).
            d_logits = head_logits[:, -1, :]
            samplers_l = getattr(gen_batch, "samplers", None) or []
            if samplers_l:
                s_i = samplers_l[i] or gen_batch.fallback_sampler
                d_tok = s_i(d_logits)
            else:
                d_tok = gen_batch.fallback_sampler(d_logits)
            drafts.append(int(d_tok[0].item()))
            try:
                row._dbg_h5 = [int(x) for x in mx.argsort(
                    head_logits[0, -1, :])[-5:].tolist()]
            except Exception:
                row._dbg_h5 = None
        except Exception as exc:
            logger.warning("[batched-verify] head draft failed uid=%s: %s",
                           uid, exc)
            raise _Fallback("head draft failure") from exc
        row.last_d = None
    _stats["head_ms"] += time.perf_counter() - t_head
    d_next = mx.array(drafts, dtype=emit_tokens.dtype)

    # (B, 2) verify forward
    t_v = time.perf_counter()
    verify_inputs = mx.stack([p_next, d_next], axis=1)
    v_logits, v_hidden = _model_forward(
        gen_batch.model, verify_inputs, gen_batch.prompt_cache, n_confirmed=1
    )
    _stats["verify_ms"] += time.perf_counter() - t_v
    v0 = v_logits[:, 0, :]  # distribution of the true token after P'
    v1 = v_logits[:, 1, :]  # distribution after D (valid on accept only)
    v0_lp_full = _norm_lp(v0)
    sample0 = _sample_rows(gen_batch, v0)  # THE first sampler call
    accept = sample0 == d_next
    accept_list = [bool(x) for x in accept.tolist()]

    # DEBUG DUMP (first 50 cycles): draft vs oracle top-5 per rejected row.
    if _stats["cycles"] < 50:
        top5 = mx.argsort(v0, axis=-1)[..., -5:]
        top5_l = top5.tolist()
        for i, uid in enumerate(uids):
            if accept_list[i]:
                continue
            truth = int(sample0[i].item())
            draft = int(d_next[i].item())
            am0 = int(mx.argmax(v0[i]).item())
            row5 = [int(x) for x in top5_l[i]]
            rowc = ctx.rows.get(uid)
            h5 = rowc._dbg_h5 if (rowc is not None and rowc._dbg_h5) else []
            logger.info(
                "[bv-dump] c%d uid=%s draft=%d truth=%d argmax0=%d "
                "h5=%s o5=%s ov=%d", _stats["cycles"], uid, draft, truth,
                am0, h5, row5, len(set(h5) & set(row5)),
            )

    # rollback rejected rows (cache: P' kept, D dropped)
    reject_mask = [not a for a in accept_list]
    if any(reject_mask):
        t_r = time.perf_counter()
        gdn_ok = _restore_gdn_rows(
            gen_batch.model, gen_batch.prompt_cache, reject_mask
        )
        # SAFETY GATE: without exact GDN rollback the rejected rows' state is
        # polluted (output-equivalence risk). Flash/qwen4_exp currently lacks
        # the stash → permanently ban this batch from the speculative path.
        if not gdn_ok:
            ctx.disabled = True
            raise _Fallback("GDN exact rollback unavailable (no stash)")
        kv_ok = _trim_rejected_rows(gen_batch, reject_mask)
        _stats["rollback_ms"] += time.perf_counter() - t_r
        _stats["reject_rollbacks"] += 1
        if not gdn_ok:
            _stats["gdn_pollution_rows"] += sum(reject_mask)

    # second sampler call ONLY for accepted rows (state-exact)
    samplers = list(getattr(gen_batch, "samplers", None) or [])
    v1_lp_full = _norm_lp(v1)
    next_tokens_list = []
    next_lps = []
    for i, uid in enumerate(uids):
        if accept_list[i]:
            if samplers:
                s_i = samplers[i] or gen_batch.fallback_sampler
                s1 = s_i(v1[i : i + 1])
            else:
                s1 = gen_batch.fallback_sampler(v1[i : i + 1])
            tok1 = int(s1[0].item())
            lp1 = float(
                mx.take_along_axis(
                    v1_lp_full[i : i + 1],
                    mx.array([[tok1]], dtype=mx.int32), axis=-1,
                ).squeeze().item()
            )
        else:
            tok1 = int(sample0[i].item())
            lp1 = float(
                mx.take_along_axis(
                    v0_lp_full[i : i + 1],
                    mx.array([[tok1]], dtype=mx.int32), axis=-1,
                ).squeeze().item()
            )
        next_tokens_list.append(tok1)
        next_lps.append(lp1)

    # skip hidden = hidden at the newest committed position (pred of the
    # next cycle's primary)
    acc = mx.array(accept_list)
    ctx.skip_hidden = mx.where(acc[:, None], v_hidden[:, 1, :],
                               v_hidden[:, 0, :])
    ctx.skip_logits = None  # no longer sampled from

    # bookkeeping + emissions
    for i, uid in enumerate(uids):
        row = ctx.rows[uid]
        _emit_one(gen_batch, i, uid, tokens_list[i], emit_lps[i],
                  responses, keep, finished)
        if uid not in ctx.rows:
            continue
        row.last_d = (
            (v_hidden[i : i + 1, 0, :], d_next[i : i + 1])
            if accept_list[i] else None
        )
        row.hist += 2 if accept_list[i] else 1

    # deferred drafts (accepted rows that survived the primary emission)
    for i, uid in enumerate(uids):
        if i in finished or uid not in ctx.rows or not accept_list[i]:
            continue
        d_i = int(d_next[i].item())
        d_lp = float(
            mx.take_along_axis(
                v0_lp_full[i : i + 1],
                mx.array([[d_i]], dtype=mx.int32), axis=-1,
            ).squeeze().item()
        )
        _emit_one(gen_batch, i, uid, d_i, d_lp, responses, keep, finished)

    # next step's tokens: the exact-once sampled primaries (un-fed; the next
    # verify — or a fallback's plain forward — will feed them)
    gen_batch._next_tokens = mx.array(next_tokens_list,
                                      dtype=emit_tokens.dtype)
    gen_batch._next_logprobs = next_lps
    mx.async_eval(gen_batch._next_tokens)

    _stats["cycles"] += 1
    _stats["drafts"] += B
    _stats["accepts"] += sum(accept_list)

    _finish_cycle(gen_batch, ctx, B, keep, finished)

    if _stats["cycles"] % 20 == 0:
        logger.info(
            "[batched-verify] cycles=%d accepts=%.1f%% gdn_exact=%d "
            "pollution=%d fallbacks=%d head=%.0fms verify=%.0fms roll=%.0fms",
            _stats["cycles"],
            100.0 * _stats["accepts"] / max(_stats["drafts"], 1),
            _stats["gdn_exact_rows"], _stats["gdn_pollution_rows"],
            _stats["fallbacks"], _stats["head_ms"] * 1000,
            _stats["verify_ms"] * 1000, _stats["rollback_ms"] * 1000,
        )
    return responses




# ---------------------------------------------------------------------------
# B-aware per-request head priming: capture prefill hiddens per row and fold
# them into per-uid MTP head caches. The stock priming capture is
# singleton-only (drops B>1 forwards), which is exactly why the MTP head has
# no history under continuous batching.
# ---------------------------------------------------------------------------

_PRIMED_HEADS: Dict[Any, Dict[str, Any]] = {}
# Merge-time adoption queue: (head_cache, carry) popped from the native
# priming stash at GenerationBatch.extend — the moment the just-prefilled
# row joins the decode batch and the stash still belongs to it.
_ADOPT_QUEUE: List[Tuple[List[Any], Any]] = []


class _HeadPrimingProxy:
    """Wraps the model during PromptProcessingBatch.prompt to capture
    per-row (trunk-normed hidden, token) pairs from the chunked prefill
    forwards and fold them into per-uid head caches immediately."""

    def __init__(self, real_model, uids, token_lists):
        self._real = real_model
        self._uids = list(uids)
        self._lengths = [len(t) for t in token_lists]
        self._consumed = [0] * len(self._uids)
        self._carry: List[Optional[mx.array]] = [None] * len(self._uids)
        self._host = _mtp_host(real_model)

    def _fold_row(self, i, tokens_row, hiddens_pred):
        uid = self._uids[i]
        entry = _PRIMED_HEADS.get(uid)
        if entry is None:
            entry = {"cache": self._host.make_mtp_cache(), "carry": None}
            _PRIMED_HEADS[uid] = entry
        try:
            ph = _head_norm(self._real, self._host, hiddens_pred[None, :, :])
            self._host.mtp_forward(
                ph, tokens_row.reshape(1, -1), entry["cache"], logits_keep=1,
            )
            _stats["primed_folds"] = _stats.get("primed_folds", 0) + 1
        except Exception as exc:
            _stats["priming_fold_fails"] = _stats.get("priming_fold_fails", 0) + 1
            if _stats["priming_fold_fails"] <= 3:
                logger.warning("[batched-verify] priming fold failed uid=%s: %s",
                               uid, exc)
            _PRIMED_HEADS.pop(uid, None)

    def __call__(self, inputs, cache=None, **kwargs):
        out = self._real(inputs, cache=cache, return_hidden=True, **kwargs)
        if (
            self._host is None
            or not isinstance(out, tuple)
            or len(out) < 2
            or getattr(inputs, "ndim", 0) != 2
            or inputs.shape[0] != len(self._uids)
        ):
            return out
        hidden = out[1]
        if hidden is None or hidden.ndim != 3:
            return out
        n = inputs.shape[1]
        for i in range(len(self._uids)):
            remaining = self._lengths[i] - self._consumed[i]
            valid = min(n, max(0, remaining))
            if valid <= 0:
                continue
            tokens_row = inputs[i, :valid]
            preds = [h for h in (self._carry[i],) if h is not None]
            if valid > 1:
                preds.append(hidden[i, : valid - 1, :])
            if preds:
                hiddens_pred = (
                    preds[0] if len(preds) == 1 else mx.concatenate(preds, 0)
                )
                if hiddens_pred.shape[0] == valid:
                    self._fold_row(i, tokens_row, hiddens_pred)
            self._carry[i] = hidden[i, valid - 1 : valid, :]
            self._consumed[i] += valid
        if len(_PRIMED_HEADS) > 128:
            for uid in list(_PRIMED_HEADS)[: len(_PRIMED_HEADS) - 128]:
                _PRIMED_HEADS.pop(uid, None)
        return out


def _install_prefill_priming_capture() -> None:
    try:
        from mlx_lm.generate import PromptProcessingBatch
    except Exception:
        return
    if getattr(PromptProcessingBatch, "_omlx_bv_priming_patched", False):
        return
    original_prompt = PromptProcessingBatch.prompt

    def patched_prompt(self, tokens, *args, **kwargs):
        # Priming capture DISABLED by default: the 08-28 decisive test showed
        # the captured entries poison the head (26.5% -> 43-50% without).
        # Opt back in via OMLX_BV_PRIMING=1 after the capture is fixed.
        priming_on = os.environ.get("OMLX_BV_PRIMING", "0").strip().lower() in ("1", "true", "on")
        if not priming_on or not _enabled() or _mtp_host(self.model) is None or not tokens:
            return original_prompt(self, tokens, *args, **kwargs)
        proxy = _HeadPrimingProxy(self.model, self.uids, tokens)
        real = self.model
        self.model = proxy
        try:
            return original_prompt(self, tokens, *args, **kwargs)
        finally:
            self.model = real
            for i, uid in enumerate(proxy._uids):
                entry = _PRIMED_HEADS.get(uid)
                if entry is not None:
                    entry["carry"] = proxy._carry[i]

    patched_prompt._omlx_bv_priming_patched = True
    PromptProcessingBatch.prompt = patched_prompt


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def install() -> bool:
    try:
        from mlx_lm.generate import GenerationBatch
    except Exception as exc:
        logger.warning("[batched-verify] install failed: %s", exc)
        return False
    if getattr(GenerationBatch, "_omlx_batched_verify_installed", False):
        return True

    chained_next = GenerationBatch.next  # omlx patched_next if active

    _diag = {"logged": False}

    def _why_not(gb):
        if len(getattr(gb, "uids", []) or []) < 2: return "B<2"
        if any(getattr(gb, "logits_processors", None) or []): return "logits_processors"
        if getattr(gb, "_omlx_mtp_batch_state", None) is not None: return "omlx_row_state"
        if _mtp_host(gb.model) is None: return "no_mtp_surface"
        pc = getattr(gb, "prompt_cache", None)
        if not pc or not hasattr(pc[0], "left_padding"): return "cache_not_batched"
        return "sampler_not_greedy"

    def patched_next(self, *args, **kwargs):
        try:
            if _eligible(self):
                return _batched_next(self)
            if not _diag["logged"] and len(getattr(self, "uids", []) or []) >= 2:
                _diag["logged"] = True
                logger.warning("[batched-verify] eligible-check fail reason: %s",
                               _why_not(self))
        except _Fallback as exc:
            _stats["fallbacks"] += 1
            ctx = getattr(self, "_omlx_batched_verify_ctx", None)
            if ctx is not None:
                ctx.skip_logits = None
                ctx.skip_hidden = None
            try:
                _ensure_mutable_states(self.prompt_cache)
            except Exception:
                pass
            logger.warning("[batched-verify] fallback: %s", exc)
        except Exception as exc:  # never break generation
            _stats["fallbacks"] += 1
            logger.warning("[batched-verify] unexpected error: %s", exc,
                           exc_info=True)
        return chained_next(self, *args, **kwargs)

    # Tag samplers at creation: temp==0 → deterministic (omlx may wrap with
    # token suppression — probing the closure is unreliable, tagging is exact).
    try:
        from omlx.utils import sampling as _sampling

        if not getattr(_sampling.make_sampler, "_omlx_tagged", False):
            _orig_ms = _sampling.make_sampler

            def _tagged_make_sampler(temp: float = 0.0, *a, **kw):
                fn = _orig_ms(temp=temp, *a, **kw)
                try:
                    fn._omlx_deterministic = (float(temp) == 0.0)
                except Exception:
                    pass
                return fn

            _tagged_make_sampler._omlx_tagged = True
            _sampling.make_sampler = _tagged_make_sampler
            # scheduler (and any other module) imported make_sampler by name
            # at import time — rebind those references too or the tag never
            # lands on production samplers.
            import omlx.scheduler as _sch
            import omlx.engine as _eng

            for _m in (_sch, _eng):
                if getattr(_m, "omlx_make_sampler", None) is _orig_ms:
                    _m.omlx_make_sampler = _tagged_make_sampler
    except Exception:
        logger.debug("sampler tagging unavailable", exc_info=True)

    _install_prefill_priming_capture()

    # Merge-time adoption: consume the native priming stash when a
    # prefill-finished row joins the generation batch.
    try:
        _orig_extend = GenerationBatch.extend

        def patched_extend(self, *a, **kw):
            out = _orig_extend(self, *a, **kw)
            try:
                from . import prompt_priming as _pp

                host_x = _mtp_host(self.model)
                cand = _pp._find_ctx(host_x) if host_x else None
                if (
                    cand is not None
                    and getattr(cand, "valid", False)
                    and getattr(cand, "folded", 0) > 0
                ):
                    _ADOPT_QUEUE.append((cand.mtp_cache, cand.pending_hidden))
                    _pp.drop_ctx(host_x)
                    if len(_ADOPT_QUEUE) > 8:
                        _ADOPT_QUEUE.pop(0)
            except Exception:
                pass
            return out

        if not getattr(GenerationBatch.extend, "_omlx_bv_extend", False):
            patched_extend._omlx_bv_extend = True
            GenerationBatch.extend = patched_extend
    except Exception:
        logger.debug("extend hook unavailable", exc_info=True)

    GenerationBatch.next = patched_next
    GenerationBatch._omlx_batched_verify_installed = True
    logger.info("[batched-verify] installed (env %s, default off)", _ENV)
    return True


def stats() -> Dict[str, Any]:
    return dict(_stats)
