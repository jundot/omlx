"""Numeric delta of the fused Qwen4 GDN verify arm vs the stock forward on real layer weights.
usage: .venv/bin/python gdn_layer_parity.py [model dir] [layer]  (default: bench oQ4e pack, layer 0)"""
import json, os, sys
import mlx.core as mx
import mlx.nn as nn

M = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.omlx-bench/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp")
LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 0
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedDeltaNet
from mlx_vlm.models.qwen3_5 import language as q35
from omlx.patches import qwen35_gdn_prework as pm

cfg = json.load(open(f"{M}/config.json"))
tc = TextConfig.from_dict(cfg.get("text_config", cfg))
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
prefix = f"language_model.model.layers.{LAYER}.linear_attn."
keys = [k for k in idx if k.startswith(prefix)]
shards = {idx[k] for k in keys}
w = {}
for s in shards:
    d = mx.load(f"{M}/{s}")
    w.update({k[len(prefix):]: d[k] for k in keys if idx[k] == s})

mod = Qwen4ExpGatedDeltaNet(tc)
for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
    wq, sc, bi = w[name + ".weight"], w[name + ".scales"], w[name + ".biases"]
    rows, packed = wq.shape
    K = sc.shape[1] * (2560 // sc.shape[1]) if name != "out_proj" else 6144
    K = getattr(mod, name).weight.shape[1] if hasattr(getattr(mod, name), "weight") else K
    K = {"out_proj": 6144}.get(name, 2560)
    bits = packed * 32 // K
    group = K // sc.shape[1]
    ql = nn.QuantizedLinear(K, rows, bias=False, group_size=group, bits=bits)
    ql.weight, ql.scales, ql.biases = wq, sc, bi
    setattr(mod, name, ql)
mod.conv1d.weight = w["conv1d.weight"]
mod.A_log, mod.dt_bias, mod.norm.weight = w["A_log"], w["dt_bias"], w["norm.weight"]
mod.eval()
print("static eligible:", pm._qwen4_decode_static_eligible(mod), "bits:", {n: (getattr(mod, n).bits, getattr(mod, n).group_size) for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")})

cls = q35.Qwen3_5GatedDeltaNet
orig = cls.__call__ if not getattr(cls, "_omlx_gdn_prework_patched", False) else None
assert orig is not None, "patch already applied; run in a fresh process"
assert pm.apply_qwen35_gdn_prework_patch()


class Cache:
    def __init__(self, c, r):
        self._s = {0: c, 1: r}; self.lengths = None; self.left_padding = None; self.n = 0
    def __getitem__(self, i): return self._s[i]
    def __setitem__(self, i, v): self._s[i] = v
    def advance(self, n): self.n += n


def run(fn, S, seed, fused):
    ks = mx.random.split(mx.random.key(seed), 3)
    x = mx.random.normal((1, S, 2560), key=ks[0]).astype(mx.bfloat16)
    conv = (mx.random.normal((1, 3, 10240), key=ks[1]) * 0.5).astype(mx.bfloat16)
    rec = (mx.random.normal((1, 48, 128, 128), key=ks[2]) * 0.05).astype(mx.float32)
    cache = Cache(conv, rec); sink = []
    if fused:
        out = fn(mod, x, cache=cache, gdn_sink=sink)
    else:
        out = fn(mod, x, mask=None, cache=cache, gdn_sink=sink, target_verify=True)
    mx.eval(out, cache[0], cache[1], *[t for t in sink[0] if isinstance(t, mx.array)])
    return out, cache, sink[0]


def cmp(name, a, b):
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    d = mx.abs(a32 - b32); n = int((d > 0).sum().item())
    rel = (d.max() / (mx.abs(b32).max() + 1e-30)).item()
    print(f"    {name:20s} differing={n}/{a.size}  max|d|={d.max().item():.3g}  max|d|/max|ref|={rel:.3g}")
    return n


NSEEDS = int(os.environ.get("NSEEDS", "1"))
for S in (1, 2, 3, 4, 8):
    total = 0; elems = 0
    for seed in range(NSEEDS):
        pm._QWEN4_VERIFY_ENGAGED_LOGGED = False
        o1, c1, s1 = run(cls.__call__, S, 100 + S + 1000 * seed, True)
        assert pm._QWEN4_VERIFY_ENGAGED_LOGGED, "fused arm did not engage"
        o0, c0, s0 = run(orig, S, 100 + S + 1000 * seed, False)
        pairs = [("out", o1, o0), ("conv_state", c1[0], c0[0]), ("recurrent", c1[1], c0[1]),
                 ("q", s1[0], s0[0]), ("k", s1[1], s0[1]), ("v", s1[2], s0[2]), ("inter_states", s1[11], s0[11])]
        if NSEEDS == 1:
            print(f"S={S}")
            for name, a, b in pairs: total += cmp(name, a, b)
        else:
            for name, a, b in pairs:
                d = (a.astype(mx.float32) != b.astype(mx.float32)).sum().item(); total += int(d); elems += a.size
    print(f"S={S}: {'BIT-EXACT' if total == 0 else 'differs'} over {NSEEDS} seeds, {elems} elements, {total} differing")
