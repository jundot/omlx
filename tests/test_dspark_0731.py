# SPDX-License-Identifier: Apache-2.0
"""Tests for the embedded DeepSeek-V4-Flash-0731 DSpark path (OMLX_DSPARK_SPEC).

Unit tests run anywhere mlx is installed. Integration tests need the real
checkpoint on a large-memory Apple Silicon box: set
OMLX_DSPARK_TEST_MODEL=/path/to/DeepSeek-V4-Flash-0731 (~6 min, M3 Ultra 512GB).
They encode the acceptance gates this path shipped under:
  - greedy spec == greedy plain, token for token (lossless canary)
  - ramp-mode drafts equal warm-mode drafts for anchors >=128 past the prompt
    (window-locality: no drafter prefill, no TTFT cost)
  - same-seed temperature-1.0 transcripts identical between synchronous and
    async cache snapshots, with equal restore counts (rejection-path safety)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

_PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "omlx", "patches", "deepseek_v4")


def _load(alias, fname):
    spec = importlib.util.spec_from_file_location(alias, os.path.join(_PKG, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


dd = _load("dspark_draft_t", "dspark_draft.py")
dg = _load("dspark_generator_t", "dspark_generator.py")

# ------------------------- unit: policies -------------------------

def test_static_policy_cutoff():
    p = dd.make_static_policy(0.6)
    assert p([0.9, 0.9, 0.5, 0.9, 0.9]) == 2
    assert p([0.5, 0.9, 0.9, 0.9, 0.9]) == 0
    assert p([0.9] * dd.BLK) == dd.BLK

def test_joint_policy_monotone():
    cf = [0.9, 0.8, 0.7, 0.6, 0.5]
    ells = [dd.make_joint_policy(p)(cf) for p in (0.55, 0.45, 0.35, 0.25)]
    assert ells == sorted(ells)
    assert dd.make_joint_policy(0.99)([0.5] * dd.BLK) == 0

# ------------------------- unit: eligibility -------------------------

class _Inner:
    pass
_Inner.__module__ = "mlx_lm.models.deepseek_v4"

def _model(mtp=None):
    m = types.SimpleNamespace(model=_Inner())
    if mtp is not None:
        m.mtp = mtp
    return m

def _gb(**kw):
    d = dict(uids=["u0"], model=_model(), logits_processors=[[]],
             samplers=[None], fallback_sampler=None)
    d.update(kw)
    return types.SimpleNamespace(**d)

def test_eligibility_matrix(monkeypatch):
    monkeypatch.setenv("OMLX_DSPARK_SPEC", "/x")
    assert dg._eligible(_gb())                                   # greedy singleton
    t1 = types.SimpleNamespace(temp=1.0, top_p=1.0, top_k=0, min_p=0.0)
    assert dg._eligible(_gb(samplers=[t1]))                      # exact temp-1.0
    t95 = types.SimpleNamespace(temp=1.0, top_p=0.95, top_k=0, min_p=0.0)
    assert dg._eligible(_gb(samplers=[t95]))                     # agentic top-p
    t7 = types.SimpleNamespace(temp=0.7, top_p=0.9, top_k=0, min_p=0.0)
    assert not dg._eligible(_gb(samplers=[t7]))                  # general sampling: defer
    assert not dg._eligible(_gb(uids=["a", "b"]))                # not singleton
    proc = lambda toks, logits: logits
    assert not dg._eligible(_gb(logits_processors=[[proc]]))     # processors: defer
    monkeypatch.delenv("OMLX_DSPARK_SPEC")
    assert not dg._eligible(_gb())                               # env-gated off

def test_coexistence_defers_to_native_head(monkeypatch):
    monkeypatch.setenv("OMLX_DSPARK_SPEC", "/x")
    assert not dg._eligible(_gb(model=_model(mtp=object())))

# ------------------------- unit: async snapshot copies -------------------------

def test_snapshot_async_copy_semantics(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    leaf = types.SimpleNamespace(keys=mx.arange(8).reshape(2, 4), meta=7,
                                 stack=[mx.ones((3,)), "tag"])
    stub = types.SimpleNamespace(_leaves=lambda cache: [leaf])
    monkeypatch.setattr(dg, "_DD", stub)
    monkeypatch.delenv("OMLX_DSPARK_SNAPSHOT", raising=False)
    assert dg._snapshot_async(object()) is None   # default contract: elided (commit 13)
    monkeypatch.setenv("OMLX_DSPARK_SNAPSHOT", "1")
    (lf, d), = dg._snapshot_async(object())
    assert lf is leaf and d["meta"] == 7
    assert d["keys"] is not leaf.keys and bool(mx.array_equal(d["keys"], leaf.keys))
    assert d["stack"][0] is not leaf.stack[0]
    assert bool(mx.array_equal(d["stack"][0], leaf.stack[0])) and d["stack"][1] == "tag"

# ------------------------- integration -------------------------

MODEL = os.environ.get("OMLX_DSPARK_TEST_MODEL", "")
needs_model = pytest.mark.skipif(
    not os.path.isdir(MODEL), reason="set OMLX_DSPARK_TEST_MODEL to the 0731 checkpoint dir")

@pytest.fixture(scope="module")
def rig():
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    from omlx.models.llm import MLXLanguageModel
    try:
        lm = MLXLanguageModel(MODEL)
    except TypeError:
        lm = MLXLanguageModel(model_name=MODEL)
    lm.load()
    model = getattr(lm, "model", None) or getattr(lm, "_model", None)
    tok = getattr(lm, "tokenizer", None) or getattr(lm, "_tokenizer", None)
    mx.eval(model.parameters())
    dsv4 = sys.modules[type(model.model).__module__]
    hcm = sys.modules[type(model.model.layers[5].attn_hc).__module__]
    # NOTE: SpecRunner init requants model.lm_head to mxfp8; both arms of every
    # test below share that head, so comparisons stay apples-to-apples.
    R = dd.SpecRunner(MODEL, model, tok, dsv4, hcm, thr=0.6)
    return types.SimpleNamespace(mx=mx, model=model, tok=tok, R=R)

def _plain_greedy(rig, pt, n):
    mx, model = rig.mx, rig.model
    cache = model.make_cache(); lg = None
    ids = mx.array([pt])
    for s0 in range(0, ids.shape[1], 512):
        lg = model(ids[:, s0:s0 + 512], cache=cache)
    out = []
    while len(out) < n:
        t = mx.argmax(lg[0, -1]); mx.eval(t)
        tt = int(t.item()); out.append(tt)
        if tt == rig.R.eos: break
        lg = model(mx.array([[tt]]), cache=cache)
    return out

def _prefill_spec(rig, pt):
    mx, model, D = rig.mx, rig.model, rig.R.D
    D.reset_rings()
    for t in D.TAPS: t.clear()
    cache = model.make_cache(); lg = None
    ids = mx.array([pt])
    D.REC[0] = True
    for s0 in range(0, ids.shape[1], 512):
        lg = model(ids[:, s0:s0 + 512], cache=cache)
    D.REC[0] = False; mx.eval(lg)
    D.extend_rings(D.take_taps(), 0)
    return cache, lg

def _finish_cycle(rig, cache, C, anchor, d, k, ell, snaps):
    """Shared post-accept bookkeeping: rings, trims, restores."""
    mx, R = rig.mx, rig.R
    D = R.D
    mh = D.take_taps()
    resto = 0
    if k < ell:
        need = ell - k
        got = [lf.trim(need) for lf in dd._leaves(cache)]
        if all(g == need for g in got):
            D.extend_rings(mh[:, :k + 1], C)
        else:
            resto = 1
            dd._restore(snaps)
            for t in D.TAPS: t.clear()
            D.REC[0] = True
            rl = rig.model(mx.array([[anchor] + d[:k]]), cache=cache)
            D.REC[0] = False; mx.eval(rl)
            D.extend_rings(D.take_taps(), C)
    else:
        D.extend_rings(mh, C)
    return resto

@needs_model
def test_greedy_l1_reverification(rig):
    """Greedy contract: spec output must be a valid greedy trajectory under
    L=1 teacher-forced re-verification, allowing only margin-bounded flips.

    Bit-identity to plain L=1 decode is NOT the contract: batched verify
    (k+1 rows) has a different matmul reduction order than one-token decode,
    which flips sub-nat near-ties (observed in server A/Bs; the sampled path
    carries a distributional certificate for the same reason). Plain decode
    rides along as an instrument control and must re-verify exactly.
    """
    mx, model, R = rig.mx, rig.model, rig.R
    pt = rig.tok.encode("Write a Python function that merges two sorted lists, with a docstring.")

    plain = _plain_greedy(rig, pt, 200)

    cache, lg = _prefill_spec(rig, pt)
    C = len(pt); anchor = int(mx.argmax(lg[0, -1]).item()); out = [anchor]
    while len(out) < 200 and out[-1] != R.eos:
        d, cf, _pd = R._draft(anchor, C, False)
        ell = R.policy(cf)
        snaps = dd._snapshot(cache) if ell > 0 else None
        R.D.REC[0] = True; R.cr.set_undo_armed(ell > 0)
        vlog = model(mx.array([[anchor] + d[:ell]]), cache=cache)
        R.cr.set_undo_armed(False); R.D.REC[0] = False
        tv = mx.argmax(vlog[0], axis=-1); mx.eval(tv)
        tv = [int(v) for v in tv.tolist()]
        k = 0
        while k < ell and tv[k] == d[k] and d[k] != R.eos: k += 1
        nxt = tv[k] if k < ell else tv[ell]
        _finish_cycle(rig, cache, C, anchor, d, k, ell, snaps)
        for tkn in d[:k]:
            out.append(tkn)
            if tkn == R.eos: break
        if out[-1] != R.eos: out.append(nxt)
        C += k + 1; anchor = out[-1]

    def _audit(seq):
        c2 = model.make_cache(); lg2 = None
        ids = mx.array([pt])
        for s0 in range(0, ids.shape[1], 512):
            lg2 = model(ids[:, s0:s0 + 512], cache=c2)
        mm = []
        for i, tkn in enumerate(seq):
            row = lg2[0, -1]
            t1 = mx.argmax(row); mx.eval(t1); t1 = int(t1.item())
            if t1 != tkn:
                mm.append((i, float((row[t1] - row[tkn]).item())))
            if tkn == R.eos: break
            lg2 = model(mx.array([[tkn]]), cache=c2)
        return mm

    assert _audit(plain) == [], "instrument control: plain decode must re-verify exactly"
    mm = _audit(out)
    assert len(mm) <= max(3, len(out) // 50), f"too many greedy flips: {mm[:8]}"
    bad = [(i, m) for i, m in mm if m > 2.0]
    assert not bad, f"non-near-tie divergence (margin > 2.0 nats): {bad[:5]}"

@needs_model
def test_ramp_matches_warm_past_window(rig):
    mx, model, R, tok = rig.mx, rig.model, rig.R, rig.tok
    D = R.D
    pt = tok.encode("Write a Python class implementing a small LRU cache with docstrings.")
    g = _plain_greedy(rig, pt, 300)
    TOK = pt + g; p0 = len(pt)
    def rings(base):
        D.reset_rings()
        for t in D.TAPS: t.clear()
        c = model.make_cache(); ids = mx.array([TOK]); D.REC[0] = True
        for s0 in range(0, ids.shape[1], 512):
            mx.eval(model(ids[:, s0:s0 + 512], cache=c))
        D.REC[0] = False
        D.extend_rings(D.take_taps()[:, base:], base)
    def sweep(base):
        ks = {}
        for a in range(p0, len(TOK) - dd.BLK):
            outs, _cf, _pd = dg._draft_ramp(R, dd, TOK[a], a, base, False)
            k = 0
            while k < dd.BLK and outs[k] == TOK[a + 1 + k]: k += 1
            ks[a] = k
        return ks
    rings(0);  warm = sweep(0)
    rings(p0); ramp = sweep(p0)
    late = [a for a in warm if a - p0 >= 128]
    assert late, "transcript too short for late anchors"
    mism = [a for a in late if warm[a] != ramp[a]]
    assert not mism, f"{len(mism)}/{len(late)} late anchors diverged"

def _sampled_run(rig, pt, n, desync):
    mx, model, R = rig.mx, rig.model, rig.R
    D = R.D
    mx.random.seed(0)
    cache, lg = _prefill_spec(rig, pt)
    C = len(pt)
    anchor = int(mx.random.categorical(lg[0, -1][None])[0].item())
    out = [anchor]; resto = 0
    while len(out) < n and out[-1] != R.eos:
        d, cf, PD = R._draft(anchor, C, True)
        ell = R.policy(cf)
        snaps = (dg._snapshot_async(cache) if desync else dd._snapshot(cache)) if ell > 0 else None
        D.REC[0] = True; R.cr.set_undo_armed(ell > 0)
        vlog = model(mx.array([[anchor] + d[:ell]]), cache=cache)
        R.cr.set_undo_armed(False); D.REC[0] = False
        PT = mx.softmax(vlog[0].astype(mx.float32), axis=-1)
        if ell > 0:
            idx = mx.array(d[:ell])
            ptd = mx.take_along_axis(PT[:ell], idx[:, None], -1)[:, 0]
            pdd = mx.take_along_axis(PD[:ell], idx[:, None], -1)[:, 0]
            us = mx.random.uniform(shape=(ell,))
            mx.eval(ptd, pdd, us)
            ptl, pdl, usl = ptd.tolist(), pdd.tolist(), us.tolist()
        else:
            ptl = pdl = usl = []
        k = 0
        while k < ell and usl[k] < min(1.0, ptl[k] / max(pdl[k], 1e-30)): k += 1
        if k < ell:
            resid = mx.maximum(PT[k] - PD[k], 0)
            if desync:
                src = mx.where(resid.sum() > 1e-9, resid, PT[k])
            else:
                ssum = resid.sum(); mx.eval(ssum)
                src = resid if float(ssum.item()) > 1e-9 else PT[k]
            nxt_a = mx.random.categorical(mx.log(src + 1e-30)[None])[0]
        else:
            nxt_a = mx.random.categorical(mx.log(PT[ell] + 1e-30)[None])[0]
        resto += _finish_cycle(rig, cache, C, anchor, d, k, ell, snaps)
        mx.eval(nxt_a)
        nxt = int(nxt_a.item())
        for tkn in d[:k]:
            out.append(tkn)
            if tkn == R.eos: break
        if out[-1] != R.eos: out.append(nxt)
        C += k + 1; anchor = out[-1]
    return out, resto

@needs_model
def test_sync_async_snapshot_equivalence_temp1(rig):
    pt = rig.tok.encode("Write a Python class implementing an LRU cache with full docstrings and detailed inline comments.")
    out_s, r_s = _sampled_run(rig, pt, 200, desync=False)
    out_a, r_a = _sampled_run(rig, pt, 200, desync=True)
    assert out_s == out_a, "same-seed transcripts diverged between snapshot modes"
    assert r_s == r_a, f"restore counts differ: {r_s} vs {r_a}"


@needs_model
def test_topp_acceptance_certificate(rig):
    """Top-p contract: output ≡ nucleus-sampled target distribution.

    With acceptance min(1, qt(x)/p(x)) against the nucleus-truncated target qt,
    the per-proposal accept probability is Σ_x min(p(x), qt(x)); pooled over a
    run, realized acceptance must match that prediction within 2σ (binomial).
    The draft stays unmasked — proposals outside the target nucleus get qt=0
    and auto-reject, which the same identity prices in. Also checks nucleus
    rows renormalize to unit mass.
    """
    import math
    mx, model, R = rig.mx, rig.model, rig.R
    TP = 0.95
    probe = mx.softmax(mx.random.normal((4, 4096)).astype(mx.float32), axis=-1)
    mass = dg._nucleus(probe, TP).sum(axis=-1); mx.eval(mass)
    assert abs(float(mass.min().item()) - 1.0) < 1e-4
    assert abs(float(mass.max().item()) - 1.0) < 1e-4

    pt = rig.tok.encode("Write a detailed 800-word essay on the history of lighthouses.")
    mx.random.seed(0)
    cache, lg = _prefill_spec(rig, pt)
    C = len(pt)
    p0 = dg._nucleus(mx.softmax(lg[0, -1].astype(mx.float32), axis=-1)[None], TP)[0]
    anchor = int(mx.random.categorical(mx.log(p0 + 1e-30)[None])[0].item())
    out = [anchor]; EXP = []; ACC = []
    while len(out) < 200 and out[-1] != R.eos:
        d, cf, PD = R._draft(anchor, C, True)
        ell = R.policy(cf)
        snaps = dd._snapshot(cache) if ell > 0 else None
        R.D.REC[0] = True; R.cr.set_undo_armed(ell > 0)
        vlog = model(mx.array([[anchor] + d[:ell]]), cache=cache)
        R.cr.set_undo_armed(False); R.D.REC[0] = False
        PTn = dg._nucleus(mx.softmax(vlog[0].astype(mx.float32), axis=-1), TP)
        if ell > 0:
            idx = mx.array(d[:ell])
            ptd = mx.take_along_axis(PTn[:ell], idx[:, None], -1)[:, 0]
            pdd = mx.take_along_axis(PD[:ell], idx[:, None], -1)[:, 0]
            ex = mx.minimum(PD[:ell], PTn[:ell]).sum(axis=-1)
            us = mx.random.uniform(shape=(ell,))
            mx.eval(ptd, pdd, ex, us)
            ptl, pdl, exl, usl = ptd.tolist(), pdd.tolist(), ex.tolist(), us.tolist()
        else:
            ptl = pdl = exl = usl = []
        k = 0
        while k < ell and usl[k] < min(1.0, ptl[k] / max(pdl[k], 1e-30)): k += 1
        for j in range(min(k + 1, ell)):
            EXP.append(exl[j]); ACC.append(1 if j < k else 0)
        if k < ell:
            resid = mx.maximum(PTn[k] - PD[k], 0)
            ssum = resid.sum(); mx.eval(ssum)
            src = resid if float(ssum.item()) > 1e-9 else PTn[k]
            nxt = int(mx.random.categorical(mx.log(src + 1e-30)[None])[0].item())
        else:
            nxt = int(mx.random.categorical(mx.log(PTn[ell] + 1e-30)[None])[0].item())
        _finish_cycle(rig, cache, C, anchor, d, k, ell, snaps)
        for tkn in d[:k]:
            out.append(tkn)
            if tkn == R.eos: break
        if out[-1] != R.eos: out.append(nxt)
        C += k + 1; anchor = out[-1]
    assert len(EXP) >= 60, "run too short to certify"
    r = sum(ACC) / len(ACC); e = sum(EXP) / len(EXP)
    sig = math.sqrt(sum(x * (1 - x) for x in EXP)) / len(EXP)
    assert abs(r - e) <= 2 * max(sig, 1e-9), \
        "acceptance dev %+.3f exceeds 2 sigma=%.3f (n=%d)" % (r - e, 2 * sig, len(EXP))


def test_dspark_error_circuit_breaker(monkeypatch):
    import importlib.util, os, sys
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
        "omlx", "patches", "deepseek_v4", "dspark_generator.py"))
    spec = importlib.util.spec_from_file_location("dg_breaker_test", p)
    dg = importlib.util.module_from_spec(spec); sys.modules["dg_breaker_test"] = dg
    spec.loader.exec_module(dg)
    monkeypatch.setenv("OMLX_DSPARK_ERROR_LIMIT", "3")
    dg._DSPARK_ERRORS = 0; dg._DSPARK_DISABLED = False
    dg._note_activation_error(); dg._note_activation_error()
    assert not dg._DSPARK_DISABLED
    dg._note_activation_error()
    assert dg._DSPARK_DISABLED
    monkeypatch.setenv("OMLX_DSPARK_ERROR_LIMIT", "0")
    dg._DSPARK_ERRORS = 0; dg._DSPARK_DISABLED = False
    for _ in range(10): dg._note_activation_error()
    assert not dg._DSPARK_DISABLED
