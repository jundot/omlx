import json, time, gc
from types import SimpleNamespace
import mlx.core as mx
GS, EPS, LIMIT, NOISE, BLK, WIN = 32, 1e-6, 10.0, 128799, 5, 128
TGT = (40, 41, 42)

def q8m(x, W, S):
    try:    return mx.quantized_matmul(x, W, S, None, transpose=True, group_size=GS, bits=8, mode="mxfp8")
    except TypeError: return mx.quantized_matmul(x, W, S, None, transpose=True, group_size=GS, bits=8)
def gq4(x, W, S, idx):
    try:    return mx.gather_qmm(x, W, S, None, rhs_indices=idx, transpose=True, group_size=GS, bits=4, mode="mxfp4")
    except TypeError: return mx.gather_qmm(x, W, S, None, rhs_indices=idx, transpose=True, group_size=GS, bits=4)
def cq8(w, s):
    if w.dtype != mx.uint8: w = w.view(mx.uint8)
    if s.dtype != mx.uint8: s = s.view(mx.uint8)
    W = w.view(mx.uint32); S = mx.repeat(mx.repeat(s, 4, axis=-1), 128, axis=0)
    mx.eval(W, S); return W, S
def rmsn(x, w): return mx.fast.rms_norm(x, w, EPS)
def ser(fn, it=12, warm=3):
    for _ in range(warm): mx.eval(fn())
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(it): mx.eval(fn())
    mx.synchronize(); return (time.perf_counter() - t0) / it * 1000

class DSpark:
    def __init__(self, model_dir, model, dsv4, hcm):
        self.model, self.dsv4, self.hcm = model, dsv4, hcm
        self.inner = model.model; margs = model.args
        self.ROPE = dsv4.DeepseekV4RoPE(margs.qk_rope_head_dim, margs.rope_theta, None, margs.max_position_embeddings)
        self.margs = margs
        wmap = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
        raw = {}
        for f in sorted({v for k, v in wmap.items() if k.startswith("mtp.")}):
            d = mx.load(f"{model_dir}/{f}")
            for k, v in d.items():
                if k.startswith("mtp."): raw[k] = v
            del d
        def K(n): v = raw.get(n); assert v is not None, n; return v
        def KS(st, frag):
            ks = [k for k in raw if k.startswith(f"mtp.{st}.") and frag in k]
            assert ks, (st, frag); return raw[sorted(ks)[0]]
        hcfg = SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=margs.hc_sinkhorn_iters,
                               hc_eps=1e-6, rms_norm_eps=1e-6, hidden_size=4096)
        def mk_hc(fn, base, sc):
            h = hcm.HyperConnection(hcfg)
            h.fn = fn.astype(mx.float32); h.base = base.astype(mx.float32); h.scale = sc.astype(mx.float32)
            h.eval(); return h
        self.stages = []
        for s in range(3):
            L = SimpleNamespace()
            L.wqa = cq8(K(f"mtp.{s}.attn.wq_a.weight"), K(f"mtp.{s}.attn.wq_a.scale"))
            L.wqb = cq8(K(f"mtp.{s}.attn.wq_b.weight"), K(f"mtp.{s}.attn.wq_b.scale"))
            L.wkv = cq8(K(f"mtp.{s}.attn.wkv.weight"),  K(f"mtp.{s}.attn.wkv.scale"))
            L.wob = cq8(K(f"mtp.{s}.attn.wo_b.weight"), K(f"mtp.{s}.attn.wo_b.scale"))
            Wa, Sa = cq8(K(f"mtp.{s}.attn.wo_a.weight"), K(f"mtp.{s}.attn.wo_a.scale"))
            L.woa = (Wa.reshape(8, 1024, -1), Sa.reshape(8, 1024, -1))
            L.qn = K(f"mtp.{s}.attn.q_norm.weight"); L.kvn = K(f"mtp.{s}.attn.kv_norm.weight")
            L.sink = K(f"mtp.{s}.attn.attn_sink").astype(mx.float32)
            L.attn_norm = K(f"mtp.{s}.attn_norm.weight"); L.ffn_norm = K(f"mtp.{s}.ffn_norm.weight")
            L.attn_hc = mk_hc(K(f"mtp.{s}.hc_attn_fn"), K(f"mtp.{s}.hc_attn_base"), K(f"mtp.{s}.hc_attn_scale"))
            L.ffn_hc  = mk_hc(K(f"mtp.{s}.hc_ffn_fn"),  K(f"mtp.{s}.hc_ffn_base"),  K(f"mtp.{s}.hc_ffn_scale"))
            L.gw = K(f"mtp.{s}.ffn.gate.weight"); L.gb = K(f"mtp.{s}.ffn.gate.bias").astype(mx.float32)
            def stack(name, s=s):
                Ws, Ss = [], []
                for e in range(256):
                    w = K(f"mtp.{s}.ffn.experts.{e}.{name}.weight"); sc = K(f"mtp.{s}.ffn.experts.{e}.{name}.scale")
                    Ws.append(w.view(mx.uint32)); Ss.append(sc if sc.dtype == mx.uint8 else sc.view(mx.uint8))
                W = mx.stack(Ws); S = mx.stack(Ss); mx.eval(W, S); return W, S
            L.GW, L.UW, L.DW = stack("w1"), stack("w3"), stack("w2")
            L.S1 = cq8(K(f"mtp.{s}.ffn.shared_experts.w1.weight"), K(f"mtp.{s}.ffn.shared_experts.w1.scale"))
            L.S3 = cq8(K(f"mtp.{s}.ffn.shared_experts.w3.weight"), K(f"mtp.{s}.ffn.shared_experts.w3.scale"))
            L.S2 = cq8(K(f"mtp.{s}.ffn.shared_experts.w2.weight"), K(f"mtp.{s}.ffn.shared_experts.w2.scale"))
            L.scale = margs.head_dim ** -0.5; L.ckv = None
            self.stages.append(L)
        self.M = SimpleNamespace(proj=cq8(K("mtp.0.main_proj.weight"), K("mtp.0.main_proj.scale")),
                                 norm=K("mtp.0.main_norm.weight"))
        S2 = self.stages[2]
        S2.hh = hcm.HyperHead(hcfg)
        S2.hh.fn = K("mtp.2.hc_head_fn").astype(mx.float32)
        S2.hh.base = K("mtp.2.hc_head_base").astype(mx.float32)
        S2.hh.scale = K("mtp.2.hc_head_scale").astype(mx.float32); S2.hh.eval()
        S2.norm_w = K("mtp.2.norm.weight")
        S2.W1 = KS(2, "markov_w1")  # native bf16 on disk; fp32 upcast removed (S49: dtau +0.000, -0.7 ms/cycle)
        S2.W2T = KS(2, "markov_w2").T
        mx.eval(S2.W1, S2.W2T)
        raw.clear(); gc.collect(); mx.clear_cache()
        layers = self.inner.layers
        self.BlockCls = type(layers[0]); self._bc = self.BlockCls.__call__
        TAPI = {id(layers[i]): j for j, i in enumerate(TGT)}
        self.TAPS = [[], [], []]; self.REC = [False]
        bc, TAPS, REC = self._bc, self.TAPS, self.REC
        def _tap(slf, h, mask, cache, inputs):
            out = bc(slf, h, mask, cache, inputs)
            if REC[0]:
                j = TAPI.get(id(slf))
                if j is not None: TAPS[j].append(out.mean(axis=2))
            return out
        self.BlockCls.__call__ = _tap
    def untap(self): self.BlockCls.__call__ = self._bc
    def take_taps(self):
        mh = mx.concatenate([mx.concatenate(t, axis=1) for t in self.TAPS], axis=-1)
        for t in self.TAPS: t.clear()
        return mh
    def reset_rings(self):
        for L in self.stages: L.ckv = None
    def extend_rings(self, mh, start, do_eval=True):
        mainx = rmsn(q8m(mh, *self.M.proj), self.M.norm)
        for L in self.stages:
            new = self.ROPE(rmsn(q8m(mainx, *L.wkv), L.kvn).reshape(1, 1, -1, 512), start)
            L.ckv = new if L.ckv is None else mx.concatenate([L.ckv, new], axis=2)
        if do_eval:
            mx.eval(*[L.ckv for L in self.stages])
    def _attn(self, L, xc, a):
        q = rmsn(q8m(xc, *L.wqa), L.qn)
        q = q8m(q, *L.wqb).reshape(1, BLK, 64, 512)
        q = mx.fast.rms_norm(q, None, EPS).transpose(0, 2, 1, 3)
        q = self.ROPE(q, a)
        kv = rmsn(q8m(xc, *L.wkv), L.kvn).reshape(1, 1, BLK, 512)
        kv = self.ROPE(kv, a)
        keys = mx.concatenate([L.ckv[:, :, max(0, a - WIN):a, :], kv], axis=2)
        sc = (q.astype(mx.float32) * L.scale) @ keys.swapaxes(-1, -2).astype(mx.float32)
        snk = mx.broadcast_to(L.sink.reshape(1, 64, 1, 1), (1, 64, BLK, 1))
        w = mx.softmax(mx.concatenate([sc, snk], axis=-1), axis=-1)[..., :-1]
        o = (w @ keys.astype(mx.float32)).astype(xc.dtype)
        o = self.ROPE(o, a, inverse=True)
        o = o.reshape(1, 8, 8, BLK, 512).transpose(0, 1, 3, 2, 4).reshape(1, 8, BLK, 4096)
        o = mx.quantized_matmul(o, L.woa[0], L.woa[1], None, transpose=True, group_size=GS, bits=8, mode="mxfp8")
        return q8m(o.transpose(0, 2, 1, 3).reshape(1, BLK, 8192), *L.wob)
    def _moe(self, L, xc):
        d = self.dsv4
        inds, wts = d._expert_select(xc @ L.gw.T, L.gb, self.margs.num_experts_per_tok,
                                    self.margs.routed_scaling_factor, True, "sqrtsoftplus")
        x5 = mx.expand_dims(xc, (-2, -3))
        h = d._limited_swiglu(gq4(x5, *L.GW, inds), gq4(x5, *L.UW, inds), LIMIT)
        y = gq4(h, *L.DW, inds).squeeze(-2)
        y = (y * wts[..., None].astype(y.dtype)).sum(-2)
        sh = d._limited_swiglu(q8m(xc, *L.S1), q8m(xc, *L.S3), LIMIT)
        return y + q8m(sh, *L.S2)
    def draft(self, anchor_id, a):
        S2 = self.stages[2]
        x = self.inner.embed_tokens(mx.array([[anchor_id, NOISE, NOISE, NOISE, NOISE]]))
        x = mx.contiguous(mx.broadcast_to(x[:, :, None, :], (1, BLK, 4, 4096)))
        for L in self.stages:
            res = x; xc, post, comb = L.attn_hc(x)
            x = self.hcm.hc_expand(self._attn(L, rmsn(xc, L.attn_norm), a), res, post, comb)
            res = x; xc, post, comb = L.ffn_hc(x)
            x = self.hcm.hc_expand(self._moe(L, rmsn(xc, L.ffn_norm)), res, post, comb)
        xh = rmsn(S2.hh(x), S2.norm_w)
        base = self.model.lm_head(xh).astype(mx.float32)[0]
        prev = mx.array(anchor_id); ids_l = []
        for i in range(BLK):
            lg = base[i] + S2.W1[prev] @ S2.W2T
            prev = mx.argmax(lg); ids_l.append(prev)
        ids_a = mx.stack(ids_l); mx.eval(ids_a)
        return [int(v) for v in ids_a.tolist()]

def plain_generate(model, ptoks, n, eos):
    cache = model.make_cache(); ids = mx.array([ptoks]); lg = None
    for s0 in range(0, ids.shape[1], 512): lg = model(ids[:, s0:s0+512], cache=cache)
    out = []; t0 = time.perf_counter()
    for _ in range(n):
        nid = mx.argmax(lg[0, -1]); mx.eval(nid)
        t = int(nid.item()); out.append(t)
        if t == eos: break
        lg = model(mx.array([[t]]), cache=cache)
    return out, len(out) / (time.perf_counter() - t0)

# ---- SpecRunner: c=1 DSpark speculative decoding (greedy + lossless sampled) ----
def _leaves(cache):
    out = []
    for lc in cache:
        subs = getattr(lc, "caches", None)
        out.extend(subs if subs else (lc,))
    return out
def _snapshot(cache):
    snaps, todo = [], []
    for lf in _leaves(cache):
        d = {}
        for k, v in vars(lf).items():
            if isinstance(v, mx.array):
                v2 = v + 0; d[k] = v2; todo.append(v2)
            elif isinstance(v, list):
                d[k] = [(x + 0) if isinstance(x, mx.array) else x for x in v]
                todo += [x for x in d[k] if isinstance(x, mx.array)]
            else: d[k] = v
        snaps.append((lf, d))
    mx.eval(*todo); return snaps
def _restore(snaps):
    for lf, d in snaps:
        lf.__dict__.clear(); lf.__dict__.update(d)

class SpecRunner:
    """Lossless c=1 speculative decoding for DeepSeek-V4-Flash-0731's DSpark module.
    greedy: exact-match verify (bitwise up to near-tie chunk numerics).
    sampled: temp-1.0 rejection sampling (distributionally lossless, Eq.8-audited)."""
    def __init__(self, model_dir, model, tok, dsv4, hcm, thr=0.6, policy=None):
        self.model, self.tok, self.thr = model, tok, thr
        self.D = DSpark(model_dir, model, dsv4, hcm)
        wmap = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
        ck = next(k for k in wmap if k.startswith("mtp.2.") and "confidence" in k)
        sh = mx.load(f"{model_dir}/{wmap[ck]}")
        self.CW = sh[ck].astype(mx.float32).reshape(-1); mx.eval(self.CW); del sh
        from omlx.patches.mlx_lm_mtp import cache_rollback as _cr
        _cr.apply(); self.cr = _cr
        self.eos = tok.eos_token_id
        self.policy = policy if policy is not None else make_static_policy(thr)
    def _draft(self, anchor_id, a, sample):
        D = self.D; S2 = D.stages[2]
        x = D.inner.embed_tokens(mx.array([[anchor_id, NOISE, NOISE, NOISE, NOISE]]))
        x = mx.contiguous(mx.broadcast_to(x[:, :, None, :], (1, BLK, 4, 4096)))
        for L in D.stages:
            res = x; xc, post, comb = L.attn_hc(x)
            x = D.hcm.hc_expand(D._attn(L, rmsn(xc, L.attn_norm), a), res, post, comb)
            res = x; xc, post, comb = L.ffn_hc(x)
            x = D.hcm.hc_expand(D._moe(L, rmsn(xc, L.ffn_norm)), res, post, comb)
        xh = rmsn(S2.hh(x), S2.norm_w)
        base = self.model.lm_head(xh).astype(mx.float32)[0]
        prev = mx.array(anchor_id); ids_l, confs, pds = [], [], []
        for i in range(BLK):
            e = S2.W1[prev]
            lg = base[i] + e @ S2.W2T
            confs.append(mx.sigmoid(mx.concatenate([xh[0, i].astype(mx.float32), e]) @ self.CW))
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
    def generate(self, ptoks, n, sample=False, seed=0, max_wall=600.0):
        model, D, cr, EOS_, thr = self.model, self.D, self.cr, self.eos, self.thr
        if sample: mx.random.seed(seed)
        D.reset_rings()
        for t in D.TAPS: t.clear()
        cache = model.make_cache(); ids = mx.array([ptoks]); lg = None
        D.REC[0] = True
        for s0 in range(0, ids.shape[1], 512): lg = model(ids[:, s0:s0+512], cache=cache)
        D.REC[0] = False; mx.eval(lg)
        D.extend_rings(D.take_taps(), 0)
        C = len(ptoks)
        row = lg[0, -1]
        anchor = int((mx.random.categorical(row[None])[0] if sample else mx.argmax(row)).item())
        out = [anchor]
        s = dict(cyc=0, acc=0, prop=0, resto=0, slow=0, real=0, tstd=0, exp=0.0, wall_hit=0, slowlog=[])
        t_start = time.perf_counter()
        while len(out) < n and out[-1] != EOS_:
            tc0 = time.perf_counter()
            d, cf, PD = self._draft(anchor, C, sample)
            ell = self.policy(cf)
            s["prop"] += ell
            snaps = _snapshot(cache) if ell > 0 else None
            D.REC[0] = True; cr.set_undo_armed(ell > 0)
            vlog = model(mx.array([[anchor] + d[:ell]]), cache=cache)
            cr.set_undo_armed(False); D.REC[0] = False
            if sample:
                PT = mx.softmax(vlog[0].astype(mx.float32), axis=-1)
                if ell > 0:
                    idx = mx.array(d[:ell])
                    ptd = mx.take_along_axis(PT[:ell], idx[:, None], axis=-1)[:, 0]
                    pdd = mx.take_along_axis(PD[:ell], idx[:, None], axis=-1)[:, 0]
                    us = mx.random.uniform(shape=(ell,))
                    em = mx.minimum(PD[:ell], PT[:ell]).sum(axis=-1)
                    mx.eval(ptd, pdd, us, em)
                    ptl, pdl, usl, eml = ptd.tolist(), pdd.tolist(), us.tolist(), em.tolist()
                else:
                    ptl = pdl = usl = eml = []
                k = 0
                while k < ell and usl[k] < min(1.0, ptl[k] / max(pdl[k], 1e-30)): k += 1
                if k < ell:
                    resid = mx.maximum(PT[k] - PD[k], 0)
                    ssum = resid.sum(); mx.eval(ssum)
                    src = resid if float(ssum.item()) > 1e-9 else PT[k]
                    nxt = int(mx.random.categorical(mx.log(src + 1e-30)[None])[0].item())
                else:
                    nxt = int(mx.random.categorical(mx.log(PT[ell] + 1e-30)[None])[0].item())
                tested = k + (1 if k < ell else 0)
                s["exp"] += sum(eml[:tested]); s["real"] += k; s["tstd"] += tested
            else:
                tv = mx.argmax(vlog[0], axis=-1); mx.eval(tv)
                tv = [int(v) for v in tv.tolist()]
                k = 0
                while k < ell and tv[k] == d[k] and d[k] != EOS_: k += 1
                nxt = tv[k] if k < ell else tv[ell]
            mh = D.take_taps()
            if k < ell:
                need = ell - k
                got = [lf.trim(need) for lf in _leaves(cache)]
                if all(g == need for g in got):
                    D.extend_rings(mh[:, :k + 1], C)
                else:
                    s["resto"] += 1; _restore(snaps)
                    for t in D.TAPS: t.clear()
                    D.REC[0] = True
                    rl = model(mx.array([[anchor] + d[:k]]), cache=cache)
                    D.REC[0] = False; mx.eval(rl)
                    D.extend_rings(D.take_taps(), C)
            else:
                D.extend_rings(mh, C)
            for tkn in d[:k]:
                out.append(tkn)
                if tkn == EOS_: break
            if out[-1] != EOS_: out.append(nxt)
            s["cyc"] += 1; s["acc"] += k; C += k + 1; anchor = out[-1]
            cw = time.perf_counter() - tc0
            if cw > 1.0:
                s["slow"] += 1
                if len(s["slowlog"]) < 5: s["slowlog"].append((s["cyc"], round(cw * 1000)))
            if time.perf_counter() - t_start > max_wall:
                s["wall_hit"] = 1; break
        tps = len(out) / (time.perf_counter() - t_start)
        s["tau"] = s["acc"] / s["cyc"] if s["cyc"] else 0.0
        s["eq8"] = (s["real"] / s["tstd"], s["exp"] / s["tstd"]) if s["tstd"] else (0.0, 0.0)
        return out, tps, s


# ---- gating policies (S28: calibration ~wash; joint product self-adapts with depth) ----
DEFAULT_STS = [1.5, 1.75, 1.15, 0.6, 1.0]
def _cal(c, t):
    import math
    c = min(max(c, 1e-4), 1 - 1e-4)
    return 1.0 / (1.0 + math.exp(-math.log(c / (1 - c)) / t))
def make_static_policy(thr):
    return lambda cf: next((i for i, c in enumerate(cf) if c < thr), BLK)
def make_joint_policy(p=0.35, temps=None):
    T = temps or DEFAULT_STS
    def pol(cf):
        a, ell = 1.0, 0
        for i in range(BLK):
            a *= _cal(cf[i], T[i])
            if a < p: break
            ell = i + 1
        return ell
    return pol
