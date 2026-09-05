#!/usr/bin/env python
"""Fase L4A aggregation over a dual-tier memtrace JSONL.

Answers:
 1. which event creates the Metal peak (max peak/active per event);
 2. whether the peak scales with hot/cold ratio or with positions;
 3. whether the peak is retained at the layer boundary or inside the
    layer build (layer_exit vs the next layer's first dual_tier.enter);
 4. whether gate/up/down share the same profile.
"""
import json
import statistics as st
import sys
from collections import defaultdict

def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows

def main(path):
    rows = load(path)
    if not rows:
        print("empty trace")
        return
    events = defaultdict(list)
    for r in rows:
        events[r.get("event")].append(r)
    print("events:", {k: len(v) for k, v in sorted(events.items())})

    print("--- per-event peak (GiB) ---")
    for ev, rs in sorted(events.items()):
        if "dual_tier" not in ev and "ctx" not in ev and "glu" not in ev:
            continue
        pk = max((r.get("peak") or 0 for r in rs), default=0)
        act = max((r.get("active") or 0 for r in rs), default=0)
        fp = max((r.get("footprint") or 0 for r in rs), default=0)
        print("%-34s n=%6d peak=%7.2f active=%7.2f footpr=%7.2f" % (
            ev, len(rs), pk / 1024**3, act / 1024**3, fp / 1024**3))

    le = events.get("dual_tier.layer_exit", [])
    if le:
        def corr(xs, ys):
            n = len(xs)
            if n < 3:
                return None
            mx, my = st.mean(xs), st.mean(ys)
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            vx = sum((x - mx) ** 2 for x in xs)
            vy = sum((y - my) ** 2 for y in ys)
            if vx <= 0 or vy <= 0:
                return None
            return cov / (vx * vy) ** 0.5

        xs_hot = [r.get("hot_positions") or 0 for r in le]
        xs_cold = [r.get("cold_positions") or 0 for r in le]
        xs_pos = [r.get("positions") or 0 for r in le]
        ys = [(r.get("peak") or 0) / 1024**3 for r in le]
        print("--- correlation of layer_exit peak with ---")
        print("  hot_positions: %.2f" % (corr(xs_hot, ys) if corr(xs_hot, ys) is not None else -9))
        print("  cold_positions: %.2f" % (corr(xs_cold, ys) if corr(xs_cold, ys) is not None else -9))
        print("  positions:     %.2f" % (corr(xs_pos, ys) if corr(xs_pos, ys) is not None else -9))
        a_hot = st.mean([(r.get("hot_bank_bytes") or 0) / 1024**2 for r in le])
        a_cold = st.mean([(r.get("cold_bank_bytes") or 0) / 1024**2 for r in le])
        m_hot = max((r.get("hot_bank_bytes") or 0) / 1024**2 for r in le)
        m_cold = max((r.get("cold_bank_bytes") or 0) / 1024**2 for r in le)
        print("  mean hot_bank %.1f MiB / cold_bank %.1f MiB; max %.1f / %.1f" % (
            a_hot, a_cold, m_hot, m_cold))

    ex = events.get("dual_tier.layer_exit", [])
    en = events.get("dual_tier.enter", [])
    if ex and en:
        pk_ex = st.median([(r.get("peak") or 0) / 1024**3 for r in ex])
        pk_en = st.median([(r.get("peak") or 0) / 1024**3 for r in en])
        act_ex = st.median([(r.get("active") or 0) / 1024**3 for r in ex])
        act_en = st.median([(r.get("active") or 0) / 1024**3 for r in en])
        print("--- boundary retention (medians) ---")
        print("  layer_exit:  peak %.2f GiB active %.2f GiB" % (pk_ex, act_ex))
        print("  next enter:  peak %.2f GiB active %.2f GiB" % (pk_en, act_en))
        print("  active retained across boundary: %.1f%%" % (
            (act_en / act_ex * 100.0) if act_ex else 0))

    by_proj = defaultdict(list)
    for r in rows:
        if "dual_tier" in (r.get("event") or ""):
            by_proj[r.get("proj")].append(r)
    print("--- per-projection peak (GiB) ---")
    for p, rs in sorted(by_proj.items()):
        pk = max((r.get("peak") or 0 for r in rs), default=0)
        print("  %-12s n=%6d peak=%7.2f" % (p, len(rs), pk / 1024**3))

if __name__ == "__main__":
    main(sys.argv[1])
