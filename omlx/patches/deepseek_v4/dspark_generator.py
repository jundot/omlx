# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative decoding inside mlx-lm GenerationBatch (c=1 singleton, v1).

Skeleton mirrors omlx.patches.mlx_lm_mtp.batch_generator: lazy activation in
next(), queue-based emission, reconcile-by-reprefill on ownership change,
fallback demotion to the original step on any doubt. Draft rings run in ramp
mode (empty at activation, filled from verify-forward taps; base offset =
cache length at activation) so no re-prefill is ever needed.
Eligibility v1: singleton, deepseek_v4, no logits_processors, greedy or exact
temp=1.0/top_p=1.0. Enabled via OMLX_DSPARK_SPEC=<model_dir>.
"""
from __future__ import annotations
import logging, os, sys, time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional, Tuple
logger = logging.getLogger(__name__)

class _Fallback(Exception): pass
_RUNNER = None
_DD = None

def _dd():
    global _DD
    if _DD is None:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dspark_draft.py")
        sp = importlib.util.spec_from_file_location("omlx_dspark_draft_rt", p)
        m = importlib.util.module_from_spec(sp)
        sys.modules["omlx_dspark_draft_rt"] = m
        sp.loader.exec_module(m)
        _DD = m
    return _DD

@dataclass
class _State:
    uid: Any = None
    queue: Deque[Tuple[int, Any, str]] = field(default_factory=deque)
    C: int = 0
    base: int = 0
    anchor: int = -1
    sample: bool = False
    cyc: int = 0; acc: int = 0; drafted: int = 0; resto: int = 0
    emits: dict = field(default_factory=lambda: {"init": 0, "draft": 0, "bonus": 0, "verify": 0, "step": 0})
    t0: float = field(default_factory=time.perf_counter)

def _log_stats(st, finish):
    tot = sum(st.emits.values()); dt = time.perf_counter() - st.t0
    logger.info("DSPARK[%s] finish=%s tokens=%d cycles=%d tok/cycle=%.2f accept=%d/%d "
                "emits=%s skips=%d resto=%d wall=%.2fs (%.2f tok/s)",
                st.uid, finish, tot, st.cyc, tot / st.cyc if st.cyc else 0.0,
                st.acc, st.drafted, st.emits, getattr(st, "skips", 0), st.resto, dt, tot / dt if dt > 0 else 0.0)

def _drop(gb, why):
    if getattr(gb, "_omlx_dspark_state", None) is not None:
        logger.info("dspark drop (%s)", why)
        try: delattr(gb, "_omlx_dspark_state")
        except AttributeError: pass

def _cache_of(gb):
    for n in ("prompt_cache", "cache", "_cache", "caches"):
        c = getattr(gb, n, None)
        if c is not None: return n, c
    raise _Fallback("no cache attr on GenerationBatch")

def _anchor_of(gb) -> int:
    v = getattr(gb, "_next_tokens", None)
    if v is None: raise _Fallback("_next_tokens absent")
    try:
        if hasattr(v, "tolist"):
            x = v.tolist()
            while isinstance(x, list): x = x[0]
            return int(x)
        if isinstance(v, (list, tuple)):
            x = v[0]
            if hasattr(x, "tolist"):
                x = x.tolist()
                while isinstance(x, list): x = x[0]
            return int(x)
        return int(v)
    except Exception as e:
        raise _Fallback(f"_next_tokens unreadable: {e!r}")

def _sampler_of(gb):
    if getattr(gb, "samplers", None) and gb.samplers[0] is not None:
        return gb.samplers[0]
    return getattr(gb, "fallback_sampler", None)

def _eligible(gb) -> bool:
    if not os.environ.get("OMLX_DSPARK_SPEC"): return False
    uids = getattr(gb, "uids", None)
    if not uids or len(uids) != 1: return False
    m = getattr(gb, "model", None)
    if m is None or type(getattr(m, "model", None)).__module__ != "mlx_lm.models.deepseek_v4":
        return False
    if getattr(m, "mtp", None) is not None:
        return False  # native DSpark/MTP head attached; it owns speculation
    if getattr(gb, "logits_processors", None) and gb.logits_processors and gb.logits_processors[0]:
        return False
    s = _sampler_of(gb)
    if s is None: return True
    t = getattr(s, "temp", None)
    if t == 0.0: return True
    tp = getattr(s, "top_p", 1.0)
    if (t == 1.0 and (tp is None or tp == 0.0 or 0.0 < tp <= 1.0)
            and getattr(s, "top_k", 0) in (0, None) and getattr(s, "min_p", 0.0) in (0.0, None)):
        return True
    return False

def _runner(gb):
    global _RUNNER
    m = gb.model
    if _RUNNER is not None and _RUNNER.model is m: return _RUNNER
    _t0 = time.perf_counter()
    dd = _dd()
    mdir = os.environ["OMLX_DSPARK_SPEC"]
    dsv4 = sys.modules[type(m.model).__module__]
    hcm = sys.modules[type(m.model.layers[5].attn_hc).__module__]
    eos = 1
    try:
        import json
        gc = os.path.join(mdir, "generation_config.json")
        if os.path.exists(gc):
            e = json.load(open(gc)).get("eos_token_id", 1)
            eos = e[0] if isinstance(e, list) else int(e)
    except Exception: pass
    tokns = type("T", (), {"eos_token_id": eos})
    R = dd.SpecRunner(mdir, m, tokns, dsv4, hcm, thr=0.6)
    _RUNNER = R
    logger.info("dspark: SpecRunner built for %s (eos=%d) in %.1fs (one-time)", mdir, eos, time.perf_counter() - _t0)
    return R

def _attn_ramp(R, dd, L, xc, a, base):
    import mlx.core as mx
    BLK, WIN, EPS, rmsn = dd.BLK, dd.WIN, dd.EPS, dd.rmsn
    q = rmsn(dd.q8m(xc, *L.wqa), L.qn)
    q = dd.q8m(q, *L.wqb).reshape(1, BLK, 64, 512)
    q = mx.fast.rms_norm(q, None, EPS).transpose(0, 2, 1, 3)
    q = R.D.ROPE(q, a)
    kv = R.D.ROPE(rmsn(dd.q8m(xc, *L.wkv), L.kvn).reshape(1, 1, BLK, 512), a)
    lo = max(0, a - WIN); s_loc = max(0, lo - base); e_loc = a - base
    if L.ckv is not None and e_loc > s_loc:
        keys = mx.concatenate([L.ckv[:, :, s_loc:e_loc, :], kv], axis=2)
    else:
        keys = kv
    sc = (q.astype(mx.float32) * L.scale) @ keys.swapaxes(-1, -2).astype(mx.float32)
    snk = mx.broadcast_to(L.sink.reshape(1, 64, 1, 1), (1, 64, BLK, 1))
    w = mx.softmax(mx.concatenate([sc, snk], axis=-1), axis=-1)[..., :-1]
    o = (w @ keys.astype(mx.float32)).astype(xc.dtype)
    o = R.D.ROPE(o, a, inverse=True)
    o = o.reshape(1, 8, 8, BLK, 512).transpose(0, 1, 3, 2, 4).reshape(1, 8, BLK, 4096)
    o = mx.quantized_matmul(o, L.woa[0], L.woa[1], None, transpose=True, group_size=32, bits=8, mode="mxfp8")
    return dd.q8m(o.transpose(0, 2, 1, 3).reshape(1, BLK, 8192), *L.wob)

def _draft_ramp(R, dd, anchor_id, a, base, sample):
    import mlx.core as mx
    BLK, NOISE, rmsn = dd.BLK, dd.NOISE, dd.rmsn
    D = R.D; S2 = D.stages[2]
    x = D.inner.embed_tokens(mx.array([[anchor_id, NOISE, NOISE, NOISE, NOISE]]))
    x = mx.contiguous(mx.broadcast_to(x[:, :, None, :], (1, BLK, 4, 4096)))
    for L in D.stages:
        res = x; xc, post, comb = L.attn_hc(x)
        x = D.hcm.hc_expand(_attn_ramp(R, dd, L, rmsn(xc, L.attn_norm), a, base), res, post, comb)
        res = x; xc, post, comb = L.ffn_hc(x)
        x = D.hcm.hc_expand(D._moe(L, rmsn(xc, L.ffn_norm)), res, post, comb)
    xh = rmsn(S2.hh(x), S2.norm_w)
    bl = R.model.lm_head(xh).astype(mx.float32)[0]
    prev = mx.array(anchor_id); ids_l, confs, pds = [], [], []
    for i in range(BLK):
        e = S2.W1[prev]
        lg = bl[i] + e @ S2.W2T
        confs.append(mx.sigmoid(mx.concatenate([xh[0, i].astype(mx.float32), e]) @ R.CW))
        if sample:
            pds.append(mx.softmax(lg, axis=-1))
            prev = mx.random.categorical(lg[None])[0]
        else:
            prev = mx.argmax(lg)
        ids_l.append(prev)
    ia = mx.stack(ids_l); cf = mx.stack(confs)
    PD = mx.stack(pds) if sample else None
    mx.eval(*([ia, cf] + ([PD] if sample else [])))
    return [int(v) for v in ia.tolist()], [float(v) for v in cf.tolist()], PD

def _run_cycle(gb, st):
    import mlx.core as mx
    R = _runner(gb); dd = _dd()
    BLK = dd.BLK; EOS = R.eos
    _, cache = _cache_of(gb)
    D = R.D; cr = R.cr
    if getattr(st, "cool", 0) > 0:
        st.cool -= 1
        st.skips = getattr(st, "skips", 0) + 1
        D.REC[0] = True
        vlog = gb.model(mx.array([[st.anchor]]), cache=cache)
        D.REC[0] = False
        if st.sample:
            PT = mx.softmax(vlog[0, -1].astype(mx.float32), axis=-1)
            nxt = int(mx.random.categorical(mx.log(PT + 1e-30)[None])[0].item())
        else:
            t_ = mx.argmax(vlog[0, -1]); mx.eval(t_); nxt = int(t_.item())
        D.extend_rings(D.take_taps(), st.C)
        st.queue.append((nxt, None, "step"))
        st.cyc += 1; st.C += 1; st.anchor = nxt
        return
    d, cf, PD = _draft_ramp(R, dd, st.anchor, st.C, st.base, st.sample)
    ell = R.policy(cf)
    if ell == 0 and os.environ.get("OMLX_DSPARK_COOLDOWN"):
        st.cool = getattr(st, "cool", 0) + 1
    st.drafted += ell
    snaps = _snapshot_async(cache) if ell > 0 else None
    _dsm = R.D.dsv4
    _sp = getattr(_dsm, "_DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED", False)
    if ell + 1 >= 5:
        _dsm._DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED = True
    D.REC[0] = True; cr.set_undo_armed(ell > 0)
    vlog = gb.model(mx.array([[st.anchor] + d[:ell]]), cache=cache)
    cr.set_undo_armed(False); D.REC[0] = False
    _dsm._DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED = _sp
    if st.sample:
        PT = mx.softmax(vlog[0].astype(mx.float32), axis=-1)
        if getattr(st, "top_p", 1.0) < 1.0:
            PT = _nucleus(PT, st.top_p)
        if ell > 0:
            idx = mx.array(d[:ell])
            ptd = mx.take_along_axis(PT[:ell], idx[:, None], axis=-1)[:, 0]
            pdd = mx.take_along_axis(PD[:ell], idx[:, None], axis=-1)[:, 0]
            us = mx.random.uniform(shape=(ell,))
            mx.eval(ptd, pdd, us)
            ptl, pdl, usl = ptd.tolist(), pdd.tolist(), us.tolist()
        else:
            ptl = pdl = usl = []
        k = 0
        while k < ell and usl[k] < min(1.0, ptl[k] / max(pdl[k], 1e-30)): k += 1
        if k < ell:
            resid = mx.maximum(PT[k] - PD[k], 0)
            src = mx.where(resid.sum() > 1e-9, resid, PT[k])
            nxt_a = mx.random.categorical(mx.log(src + 1e-30)[None])[0]
        else:
            nxt_a = mx.random.categorical(mx.log(PT[ell] + 1e-30)[None])[0]
        nxt = -1
    else:
        tv = mx.argmax(vlog[0], axis=-1); mx.eval(tv)
        tv = [int(v) for v in tv.tolist()]
        k = 0
        while k < ell and tv[k] == d[k] and d[k] != EOS: k += 1
        nxt = tv[k] if k < ell else tv[ell]
        nxt_a = None
    mh = D.take_taps()
    if k < ell:
        need = ell - k
        got = [lf.trim(need) for lf in dd._leaves(cache)]
        if all(g == need for g in got):
            D.extend_rings(mh[:, :k + 1], st.C, do_eval=False)
        else:
            st.resto += 1
            if snaps is None:
                raise _Fallback("trim failed with snapshot elided (OMLX_DSPARK_SNAPSHOT=1 restores per-cycle insurance)")
            dd._restore(snaps)
            for t in D.TAPS: t.clear()
            if k + 1 >= 5:
                _dsm._DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED = True
            D.REC[0] = True
            rl = gb.model(mx.array([[st.anchor] + d[:k]]), cache=cache)
            D.REC[0] = False; mx.eval(rl)
            _dsm._DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED = _sp
            D.extend_rings(D.take_taps(), st.C)
    else:
        D.extend_rings(mh, st.C, do_eval=False)
    _tails = [L.ckv for L in D.stages if L.ckv is not None]
    if st.sample and nxt < 0:
        mx.eval(nxt_a, *_tails)
        nxt = int(nxt_a.item())
    else:
        mx.eval(*_tails)
    for i in range(k):
        st.queue.append((d[i], None, "draft"))
        if d[i] == EOS: break
    if not st.queue or st.queue[-1][0] != EOS:
        st.queue.append((nxt, None, "bonus" if (ell > 0 and k == ell) else ("verify" if ell > 0 else "step")))
    st.cyc += 1; st.acc += k
    st.C += k + 1; st.anchor = st.queue[-1][0]

def _prepare(gb) -> Optional["_State"]:
    st = getattr(gb, "_omlx_dspark_state", None)
    if st is not None and getattr(gb, "uids", None) and gb.uids[0] == st.uid:
        return st
    if st is not None: _drop(gb, "stale-owner")
    R = _runner(gb)
    _cache_of(gb)
    anchor = _anchor_of(gb)
    C = len(gb.tokens[0])
    st = _State(uid=gb.uids[0], C=C, base=C, anchor=anchor)
    s = _sampler_of(gb)
    st.sample = bool(s is not None and getattr(s, "temp", None) == 1.0)
    _tp = getattr(s, "top_p", 1.0) if s is not None else 1.0
    st.top_p = 1.0 if (_tp is None or _tp == 0.0) else float(_tp)
    R.D.reset_rings()
    for t in R.D.TAPS: t.clear()
    st.queue.append((anchor, None, "init"))
    gb._omlx_dspark_state = st
    logger.info("DSPARK activated uid=%s C=%d sample=%s", st.uid, C, st.sample)
    return st

def _emit(gb, token_id, lp, st):
    Response = type(gb).Response
    finish = None
    gb.tokens[0].append(token_id)
    gb._num_tokens[0] += 1
    if gb._num_tokens[0] >= gb.max_tokens[0]: finish = "length"
    new_state, match_sequence, current_state = gb.state_machines[0].match(gb._matcher_states[0], token_id)
    gb._matcher_states[0] = new_state
    if match_sequence is not None and current_state is None: finish = "stop"
    if finish is not None:
        pc = gb.extract_cache(0); at = gb.tokens[0]
        _log_stats(st, finish)
        try: delattr(gb, "_omlx_dspark_state")
        except AttributeError: pass
        resp = Response(uid=gb.uids[0], token=token_id, logprobs=lp, finish_reason=finish,
                        current_state=current_state, match_sequence=match_sequence,
                        prompt_cache=pc, all_tokens=at)
        gb.filter([])
        return [resp]
    return [Response(uid=gb.uids[0], token=token_id, logprobs=lp, finish_reason=None,
                     current_state=current_state, match_sequence=match_sequence,
                     prompt_cache=None, all_tokens=None)]

def _pop_emit(gb, st):
    tid, lp, src = st.queue.popleft()
    st.emits[src] = st.emits.get(src, 0) + 1
    return _emit(gb, tid, lp, st)

def _reconcile(gb, st) -> bool:
    import mlx.core as mx
    toks = gb.tokens[0] if getattr(gb, "tokens", None) else None
    if not toks: return False
    try:
        make_cache = sys.modules["mlx_lm.generate"]._make_cache
        new_cache = make_cache(gb.model, [0], None)
    except Exception as e:
        logger.warning("dspark reconcile: cache rebuild unavailable: %r", e); return False
    ids = mx.array([list(toks)]); lg = None
    for s0 in range(0, ids.shape[1], 512):
        lg = gb.model(ids[:, s0:s0 + 512], cache=new_cache)
    mx.eval(lg)
    name, _ = _cache_of(gb)
    setattr(gb, name, new_cache)
    nt = st.queue[0][0] if st.queue else int(mx.argmax(lg[0, -1]).item())
    row = lg[0, -1].astype(mx.float32)
    lp = row - mx.logsumexp(row)   # correct dist for queue[0]'s position in both branches
    mx.eval(lp)
    v = getattr(gb, "_next_tokens", None)
    try:
        gb._next_tokens = mx.array([nt], dtype=v.dtype) if (v is not None and hasattr(v, "dtype")) else mx.array([nt])
        gb._next_logprobs = [lp]
    except Exception as e:
        logger.warning("dspark reconcile: state set failed: %r", e); return False
    return True

def _reconcile_and_drop(gb, why):
    st = getattr(gb, "_omlx_dspark_state", None)
    if st is None: return
    try:
        ok = _reconcile(gb, st)
        logger.info("dspark reconcile(%s) ok=%s", why, ok)
    except Exception as e:
        logger.warning("dspark reconcile(%s) failed: %r", why, e)
    _drop(gb, why)

def enable() -> bool:
    try:
        from mlx_lm.generate import GenerationBatch
    except ImportError:
        logger.warning("dspark: mlx_lm.generate not importable"); return False
    if getattr(GenerationBatch, "_omlx_dspark_patched", False): return True
    orig_next = GenerationBatch.next
    orig_extend = GenerationBatch.extend
    orig_filter = GenerationBatch.filter
    def patched_next(self, *a, **kw):
        st = getattr(self, "_omlx_dspark_state", None)
        if st is not None and st.queue and getattr(self, "uids", None) and self.uids[0] == st.uid:
            return _pop_emit(self, st)
        if _eligible(self):
            try:
                st = _prepare(self)
                if st is not None:
                    if not st.queue:
                        _run_cycle(self, st)
                    if st.queue:
                        return _pop_emit(self, st)
                    raise _Fallback("empty queue after cycle")
            except _Fallback as e:
                logger.info("dspark fallback -> standard: %s", e)
                _reconcile_and_drop(self, "fallback")
            except Exception as e:
                logger.warning("dspark error -> standard: %r", e)
                _reconcile_and_drop(self, "error")
        elif getattr(self, "_omlx_dspark_state", None) is not None:
            _reconcile_and_drop(self, "ineligible")
        return orig_next(self, *a, **kw)
    def patched_extend(self, batch, *a, **kw):
        _reconcile_and_drop(self, "extend")
        _drop(batch, "donor-extend")
        return orig_extend(self, batch, *a, **kw)
    def patched_filter(self, keep, *a, **kw):
        r = orig_filter(self, keep, *a, **kw)
        st = getattr(self, "_omlx_dspark_state", None)
        if st is not None and (not getattr(self, "uids", None) or self.uids[0] != st.uid):
            _drop(self, "filter-owner-change")
        return r
    GenerationBatch.next = patched_next
    GenerationBatch.extend = patched_extend
    GenerationBatch.filter = patched_filter
    GenerationBatch._omlx_dspark_patched = True
    logger.info("dspark: GenerationBatch wraps installed")
    return True


_SNAP_ELIDE_LOGGED = False

def _snapshot_async(cache):
    """Detached cache snapshot; async_eval keeps FIFO ordering ahead of verify writes (no host stall)."""
    import os as _os
    if _os.environ.get("OMLX_DSPARK_SNAPSHOT", "0") != "1":
        global _SNAP_ELIDE_LOGGED
        if not _SNAP_ELIDE_LOGGED:
            logger.info("dspark: per-cycle cache snapshot ELIDED (restore fired 0x in project history; OMLX_DSPARK_SNAPSHOT=1 restores)")
            _SNAP_ELIDE_LOGGED = True
        return None
    import mlx.core as mx
    dd = _dd()
    snaps, todo = [], []
    for lf in dd._leaves(cache):
        d = {}
        for k, v in vars(lf).items():
            if isinstance(v, mx.array):
                v2 = v + 0; d[k] = v2; todo.append(v2)
            elif isinstance(v, list):
                d[k] = [(x + 0) if isinstance(x, mx.array) else x for x in v]
                todo += [x for x in d[k] if isinstance(x, mx.array)]
            else:
                d[k] = v
        snaps.append((lf, d))
    mx.async_eval(*todo)
    return snaps


def _nucleus(P, p):
    """Top-p truncate + renormalize per row, in-graph.

    Lossless-speculation note: applying the nucleus to the TARGET only is
    exact — draft tokens outside the target nucleus get q=0 and auto-reject;
    residual max(q-p,0)/Z and bonus/correction sampling already use the
    truncated q. The draft distribution needs no masking for correctness.
    """
    import mlx.core as mx
    order = mx.argsort(-P, axis=-1)
    Ps = mx.take_along_axis(P, order, axis=-1)
    cs = mx.cumsum(Ps, axis=-1)
    keep_sorted = (cs - Ps) < p            # include the crossing token
    inv = mx.argsort(order, axis=-1)
    keep = mx.take_along_axis(keep_sorted, inv, axis=-1)
    Pm = mx.where(keep, P, mx.zeros_like(P))
    return Pm / mx.maximum(Pm.sum(axis=-1, keepdims=True), 1e-30)
