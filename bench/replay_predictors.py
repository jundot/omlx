#!/usr/bin/env python3
"""Offline replay of expert-routing predictors against memtrace traces.

V2-6 phase 1: before training or integrating a prerouter, measure what
cheap predictors achieve on the recorded decode streams. Gate from the
plan: integrate a trained head only if it beats prev-token by +10pts
recall; this harness answers whether a *transition/lookup* predictor
already gets there for free.

Predictors evaluated per MoE layer:
  prev       — predict token t+1's set = token t's set (current runtime
               heuristic; recall ~0.67-0.8 measured).
  union_k    — union of the last k tokens' sets (k=2,3); recall up but
               prediction size grows (staging cost).
  markov     — per-expert bigram table P(e' | e in routed_t); score each
               candidate by summed counts over the current set, take the
               top ``width`` experts (width sweeps top_k..3*top_k).
  trans      — exact-set transition table routed_t -> most common
               routed_t+1; falls back to prev on unseen sets.

Recall = |pred ∩ actual| / |actual|. Also reports mean prediction size
(the staged-fetch budget a predictor would consume) and per-layer
spread — late layers are typically harder than early ones.

Usage:
  python bench/replay_predictors.py bench/results/v2_traces/v41_jundot.jsonl \
      --topk 6 [--widths 6 8 12 18] [--train-frac 0.7]
"""

import argparse
import json
from collections import Counter, defaultdict


def load_token_sets(path, topk, phase="decode"):
    """Reconstruct per-token routed sets per layer from routing events.

    The generic path emits one identical event per projection call
    (~3x per token-layer); the V4.1 path emits one. Multi-position
    events (verify/prefill-tail inside the decode phase) carry unions
    larger than topk and are dropped. Token boundaries are detected by
    the layer index wrapping.
    """
    events = []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("event") != "routing" or e.get("phase") != phase:
                continue
            experts = e.get("experts") or []
            if len(experts) > topk:
                continue  # union event, not per-token routing
            events.append((int(e["layer"]), frozenset(experts)))
    events.sort(key=lambda x: x[0]) if False else None  # keep seq order
    # Dedupe consecutive identical (layer, set) — the per-projection calls.
    dedup = []
    for layer, s in events:
        if dedup and dedup[-1] == (layer, s):
            continue
        dedup.append((layer, s))
    # Group into tokens: a new token starts when layer index does not
    # increase (wrap or restart). Layers may be non-contiguous (GLM
    # starts at layer 3).
    layers_order = []
    seen = set()
    for layer, _ in dedup:
        if layer not in seen:
            seen.add(layer)
            layers_order.append(layer)
    n_layers = len(layers_order)
    tokens = []
    cur = {}
    prev_layer = None
    for layer, s in dedup:
        if prev_layer is not None and layer <= prev_layer:
            if len(cur) >= n_layers // 2:  # tolerate partial tails
                tokens.append(cur)
            cur = {}
        cur[layer] = s
        prev_layer = layer
    if cur:
        tokens.append(cur)
    return tokens, layers_order


def recall(pred, actual):
    if not actual:
        return 0.0
    return len(pred & actual) / len(actual)


def eval_prev(tokens, layers):
    out = {}
    for L in layers:
        hits = tot = 0
        seq = [t.get(L) for t in tokens]
        seq = [s for s in seq if s is not None]
        for i in range(1, len(seq)):
            hits += len(seq[i - 1] & seq[i])
            tot += len(seq[i])
        out[L] = hits / tot if tot else 0.0
    return out


def eval_union_k(tokens, layers, k):
    out = {}
    sizes = {}
    for L in layers:
        hits = tot = size_sum = 0
        seq = [t.get(L) for t in tokens]
        seq = [s for s in seq if s is not None]
        for i in range(k, len(seq)):
            pred = frozenset().union(*seq[i - k : i])
            hits += len(pred & seq[i])
            tot += len(seq[i])
            size_sum += len(pred)
        out[L] = hits / tot if tot else 0.0
        sizes[L] = size_sum / max(1, len(seq) - k)
    return out, sizes


def eval_markov(tokens, layers, widths, train_frac):
    """Per-layer bigram P(next | cur in set); top-`width` by summed count."""
    out = {w: {} for w in widths}
    for L in layers:
        seq = [t.get(L) for t in tokens]
        seq = [s for s in seq if s is not None]
        n_train = max(2, int(len(seq) * train_frac))
        bigram = defaultdict(Counter)
        for i in range(1, n_train):
            for e in seq[i - 1]:
                bigram[e].update(seq[i])
        for w in widths:
            hits = tot = 0
            for i in range(n_train, len(seq)):
                scores = Counter()
                for e in seq[i - 1]:
                    scores.update(bigram[e])
                pred = {e for e, _ in scores.most_common(w)}
                if not pred:
                    pred = seq[i - 1]
                hits += len(pred & seq[i])
                tot += len(seq[i])
            out[w][L] = hits / tot if tot else 0.0
    return out


def eval_trans(tokens, layers, train_frac):
    out = {}
    for L in layers:
        seq = [t.get(L) for t in tokens]
        seq = [s for s in seq if s is not None]
        n_train = max(2, int(len(seq) * train_frac))
        table = defaultdict(Counter)
        for i in range(1, n_train):
            table[seq[i - 1]][seq[i]] += 1
        hits = tot = fallback = 0
        for i in range(n_train, len(seq)):
            best = table.get(seq[i - 1])
            pred = best.most_common(1)[0][0] if best else seq[i - 1]
            if best is None:
                fallback += 1
            hits += len(pred & seq[i])
            tot += len(seq[i])
        out[L] = (hits / tot if tot else 0.0, fallback)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--widths", type=int, nargs="+", default=None)
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()
    widths = args.widths or sorted({args.topk, args.topk + 2, args.topk * 2, args.topk * 3})

    tokens, layers = load_token_sets(args.trace, args.topk)
    print(f"{args.trace}: {len(tokens)} tokens, {len(layers)} MoE layers")

    prev = eval_prev(tokens, layers)
    base = sum(prev.values()) / len(prev)
    print(f"\nprev-token recall (runtime heuristic): {base:.3f}")
    print(f"  per-layer min/max: {min(prev.values()):.3f} / {max(prev.values()):.3f}")

    for k in (2, 3):
        u, sizes = eval_union_k(tokens, layers, k)
        print(
            f"union last-{k}: recall {sum(u.values())/len(u):.3f} "
            f"(+{sum(u.values())/len(u)-base:+.3f}), mean pred size {sum(sizes.values())/len(sizes):.1f}"
        )

    markov = eval_markov(tokens, layers, widths, args.train_frac)
    for w in widths:
        m = sum(markov[w].values()) / len(markov[w])
        print(f"markov top-{w}: recall {m:.3f} ({m-base:+.3f} vs prev)")

    trans = eval_trans(tokens, layers, args.train_frac)
    t = sum(v[0] for v in trans.values()) / len(trans)
    fb = sum(v[1] for v in trans.values()) / max(1, len(tokens))
    print(f"exact-set transition: recall {t:.3f} ({t-base:+.3f}), fallback {fb:.1%}")


if __name__ == "__main__":
    main()
