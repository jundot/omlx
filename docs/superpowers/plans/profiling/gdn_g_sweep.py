"""Sweep every finite bf16 value of the GDN gate input `a` through the fused kernel's g/beta and
compare with mlx-vlm's _compute_g_beta on a real layer's A_log/dt_bias. usage: gdn_g_sweep.py [layer]"""
import json, os, sys
import mlx.core as mx
from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta
from omlx.patches import qwen35_gdn_prework as pm

M = os.path.expanduser("~/.omlx-bench/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp")
GDN_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 11  # index among the GDN layers
cfg = json.load(open(f"{M}/config.json")); tcfg = cfg.get("text_config", cfg)
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
gdn_layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in idx if ".linear_attn.A_log" in k})
LAYER = gdn_layers[GDN_IDX]
p = f"language_model.model.layers.{LAYER}.linear_attn."
d = mx.load(f"{M}/{idx[p + 'A_log']}")
A_log, dt_bias = d[p + "A_log"], d[p + "dt_bias"]
HK, HV, DK, DV, C = 16, 48, 128, 128, 10240
S = 8
# every bf16 bit pattern with |a| finite and < 2^7 (the model's a range is a few units)
bits = mx.arange(0, 65536, dtype=mx.uint32).astype(mx.uint16)
vals = bits.view(mx.bfloat16)
import numpy as np
keep = np.asarray(mx.isfinite(vals) & (mx.abs(vals) < 128))
vals = mx.array(np.asarray(vals.astype(mx.float32))[keep]).astype(mx.bfloat16)
n = int(vals.size)
per = S  # each row carries one value replicated across all heads
vals = mx.concatenate([vals, mx.zeros((per - n % per,), dtype=mx.bfloat16)]) if n % per else vals
calls = int(vals.size) // per
print(f"layer {LAYER}: sweeping {n} bf16 values of a in {calls} kernel calls")
inv = DK**-0.5
q_scale = mx.array(inv * inv, dtype=mx.bfloat16); k_scale = mx.array(inv, dtype=mx.bfloat16)
qkv = mx.zeros((1, S, C), dtype=mx.bfloat16); conv_state = mx.zeros((1, 3, C), dtype=mx.bfloat16)
conv_w = mx.zeros((C, 4, 1), dtype=mx.bfloat16)
g_mis = b_mis = 0; examples = []
for c in range(calls):
    a = mx.broadcast_to(vals[c * per:(c + 1) * per].reshape(1, S, 1), (1, S, HV))
    a = mx.contiguous(a)
    b = a  # sweep beta on the same values
    out = pm.qwen4_verify_prework_fused(qkv, conv_state, conv_w, q_scale, k_scale, b, a, A_log, dt_bias, HK, HV, DK, DV)
    g_k, beta_k = out[4], out[5]
    g_r, beta_r = _compute_g_beta(A_log, a, b, dt_bias)
    mx.eval(g_k, beta_k, g_r, beta_r)
    gm = (g_k != g_r); bm = (beta_k != beta_r)
    g_mis += int(gm.sum().item()); b_mis += int(bm.sum().item())
    if gm.any().item() and len(examples) < 6:
        i = int(mx.argmax(gm.reshape(-1)).item())
        examples.append((float(a.reshape(-1)[i].item()), i % HV, float(g_k.reshape(-1)[i].item()), float(g_r.reshape(-1)[i].item())))
print(f"g mismatches {g_mis}/{n * HV}  beta mismatches {b_mis}/{n * HV}")
for e in examples: print("  a=%g head=%d kernel=%.9g ref=%.9g" % e)
