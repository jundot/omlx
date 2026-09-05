#!/usr/bin/env python
"""Fase L2 arm aggregation: medians and pairwise deltas for A/C/E plus
the B/D sizing points (all 2k decode 48, budget=0)."""
import json
import statistics as st
from pathlib import Path

ROOT = Path("bench/results/fasel/l2")

def load(name):
    p = ROOT / "arms" / (name + ".json")
    if not p.exists():
        return None
    return json.loads(p.read_text())

def row(d):
    if d is None:
        return None
    pin = d.get("pin") or {}
    return {
        "tok": d.get("tok_s"),
        "ttft": d.get("ttft_s"),
        "toks": len(d.get("tokens") or []),
        "metal_dec": d.get("metal_peak_decode_gib"),
        "metal_pre": d.get("metal_peak_prefill_gib"),
        "phys_max": d.get("phys_lifetime_max_gib"),
        "pinned_gib": round((pin.get("pinned_bytes") or 0) / 1024**3, 3),
        "pinned_exp": pin.get("pinned_experts"),
        "pages": pin.get("pinned_pages_estimate"),
        "fp": pin.get("profile_fingerprint_match"),
        "pin_ms": pin.get("pin_load_time_ms"),
        "fb": d.get("ctx_fallback_to_legacy"),
        "rr95": ((d.get("read_stats") or {}).get("lat_us") or {}).get("p95"),
    }

def med(rows, key):
    vals = [r[key] for r in rows if r and r.get(key) is not None]
    return st.median(vals) if vals else None

arms = {n: [load(n + str(i)) for i in (1, 2, 3)] for n in "ace"}
arms.update({"b": [load("b1")], "d": [load("d1")]})

print("%-5s %8s %8s %6s %6s %6s %8s %7s %7s %8s" % (
    "arm", "tok_s", "ttft_s", "tokN", "mdec", "mpre", "physmax", "pinGiB", "pinExp", "rr95"))
rows_all = {n: [row(r) for r in rs] for n, rs in arms.items()}
for n, rs in sorted(rows_all.items()):
    r0 = rs[0]
    if r0 is None:
        print("%-5s MISSING" % n)
        continue
    print("%-5s %8.3f %8.2f %6d %6.2f %6.2f %8.2f %7.3f %7d %8.2f" % (
        n.upper(), med(rs, "tok") or 0, med(rs, "ttft") or 0, r0.get("toks") or 0,
        med(rs, "metal_dec") or 0, med(rs, "metal_pre") or 0, med(rs, "phys_max") or 0,
        med(rs, "pinned_gib") or 0, med(rs, "pinned_exp") or 0, med(rs, "rr95") or 0))

print("--- decision deltas (medians) ---")
def dt(a, b, key):
    va, vb = med(rows_all[a], key), med(rows_all[b], key)
    if not va or not vb:
        return None
    return (va - vb) / vb * 100.0

for key in ("tok", "ttft"):
    print("C vs A  %s: %+.2f%%" % (key, dt("c", "a", key)))
    print("E vs A  %s: %+.2f%%" % (key, dt("e", "a", key)))
    print("C vs E  %s: %+.2f%%" % (key, dt("c", "e", key)))
print("B sizing tok: %.3f (A %.3f)" % (med(rows_all["b"], "tok") or 0, med(rows_all["a"], "tok") or 0))
print("D sizing tok: %.3f (A %.3f)" % (med(rows_all["d"], "tok") or 0, med(rows_all["a"], "tok") or 0))
a0 = (arms["a"][0] or {}).get("tokens")
ta = [r["tokens"] for r in arms["c"] if r and r.get("tokens")]
te = [r["tokens"] for r in arms["e"] if r and r.get("tokens")]
print("tokens A==C:", all(t == a0 for t in ta) if (ta and a0) else "n/a")
print("tokens A==E:", all(t == a0 for t in te) if (te and a0) else "n/a")

print("--- per-run detail (A/C/E) ---")
for n in "ace":
    for i, r0 in enumerate(rows_all[n], 1):
        if r0 is None:
            print(n + str(i), "MISSING")
            continue
        print("%s%s tok=%.3f ttft=%.1f fp=%s fb=%s rr95=%s pin=%s" % (
            n, i, r0["tok"], r0["ttft"], r0["fp"], r0["fb"], r0["rr95"], r0["pin_ms"]))
