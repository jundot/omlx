#!/usr/bin/env python3
"""Train + evaluate offline prerouters on hstrace captures (V2-6).

Each capture row is (layer, hidden_state, routed_set) for one decode
token at one MoE layer. Two prediction directions matter:

  next-tok   feature: layer L, token t  ->  target: layer L, token t+1
             (lead ~= one full token — the prefetch window the runtime
             needs to hide a decode-miss read entirely)
  +k layers  feature: layer L, token t  ->  target: layer L+k, token t
             (lead ~= k layer calls — hides only part of a read, but
             temporally closer features usually predict better)

Model: per-(layer,direction) multi-label ridge probe — closed form, no
torch dependency. Score = x @ W, predicted set = top ``width``.

Reported: mean recall = |pred ∩ actual| / topk on held-out tokens,
vs the prev-token baseline at the same horizon.

Gate (plan V2-6): integrate only if the head beats prev-token by
>= +10pts recall at comparable prediction width.

Usage:
  python bench/train_prerouter.py bench/results/v2_hstrace/qwen_jundot \
      --topk 10 [--width-mults 1 1.5 2] [--train-frac 0.7]
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def load(prefix):
    idx = [json.loads(l) for l in open(prefix + ".jsonl")]
    head = idx[0]
    rows, dim = head["rows"], head["dim"]
    entries = idx[1:]
    assert len(entries) == rows, (len(entries), rows)
    X = np.fromfile(prefix + ".bin", dtype=np.float16).reshape(rows, dim)
    # Per-layer token sequences in call order: by_layer[L][t].
    by_layer = defaultdict(list)
    for i, e in enumerate(entries):
        by_layer[e["layer"]].append((i, frozenset(e["experts"])))
    layers = sorted(by_layer)
    return X, by_layer, layers


def ridge_fit(X, Y, lam=1.0):
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d, dtype=np.float64)
    B = X.T @ Y
    return np.linalg.solve(A, B)


def _targets(n_cols, seqs):
    """Multi-hot target matrix: seqs = list of expert sets."""
    n_exp = max(e for s in seqs for e in s) + 1
    Y = np.zeros((n_cols, n_exp), dtype=np.float64)
    for r, s in enumerate(seqs):
        Y[r, list(s)] = 1.0
    return Y


def eval_ridge_multi(X, pairs, widths, train_frac, lam):
    """pairs: ordered [(feat_row, target_set)] — one direction's data
    for a single source layer. One ridge fit serves every width."""
    n = len(pairs)
    n_train = max(8, int(n * train_frac))
    if n - n_train < 4:
        return None
    tr_rows = np.array([p[0] for p in pairs[:n_train]])
    Ytr = _targets(n_train, [p[1] for p in pairs[:n_train]])
    W = ridge_fit(X[tr_rows].astype(np.float64), Ytr, lam)
    hits = {w: 0 for w in widths}
    tot = 0
    for row, actual in pairs[n_train:]:
        scores = X[row].astype(np.float64) @ W
        order = np.argsort(-scores)
        for w in widths:
            hits[w] += len(set(order[:w].tolist()) & actual)
        tot += len(actual)
    return {w: (h / tot if tot else None) for w, h in hits.items()}


def eval_prev(pairs):
    """Same-horizon prev baseline: predict the previous visit's target."""
    hits = tot = 0
    for i in range(1, len(pairs)):
        pred, actual = pairs[i - 1][1], pairs[i][1]
        hits += len(pred & actual)
        tot += len(actual)
    return hits / tot if tot else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--width-mults", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument(
        "--export",
        metavar="OUT.npz",
        help="train +1-layer heads on ALL data and write an "
        "expert_prerouter.npz sidecar (W<target_layer> matrices)",
    )
    ap.add_argument(
        "--export-width",
        type=int,
        default=0,
        help="predicted-set size baked into the artifact (default 2*topk)",
    )
    args = ap.parse_args()

    X, by_layer, layers = load(args.prefix)
    n_tok = min(len(by_layer[L]) for L in layers)
    print(f"{args.prefix}: {len(layers)} layers, dim={X.shape[1]}, {n_tok} tokens/layer")

    # Direction A: same layer, next token (lead ~1 token).
    directions = [("same-layer next-tok", "ntok", 0)]
    # Direction B: layer L -> layer L+k, same token (lead ~k layers).
    for k in (1, 2, 4, 8):
        directions.append((f"+{k} layer(s) same-tok", "xlayer", k))

    for name, kind, k in directions:
        prev_r = []
        head_r = {wm: [] for wm in args.width_mults}
        for i, L in enumerate(layers):
            seq = by_layer[L]
            if kind == "ntok":
                pairs = [
                    (seq[t][0], seq[t + 1][1]) for t in range(len(seq) - 1)
                ]
            else:
                j = i + k
                if j >= len(layers):
                    continue
                tgt = by_layer[layers[j]]
                m = min(len(seq), len(tgt))
                pairs = [(seq[t][0], tgt[t][1]) for t in range(m)]
            p = eval_prev(pairs)
            if p is not None:
                prev_r.append(p)
            widths = [
                max(args.topk, int(args.topk * wm)) for wm in args.width_mults
            ]
            rr = eval_ridge_multi(X, pairs, widths, args.train_frac, args.lam)
            if rr is not None:
                for wm, w in zip(args.width_mults, widths):
                    if rr[w] is not None:
                        head_r[wm].append(rr[w])
        base = np.mean(prev_r) if prev_r else float("nan")
        line = f"{name:22s} prev {base:.3f} | ridge:"
        for wm in args.width_mults:
            v = head_r[wm]
            line += (
                f"  {wm}x {np.mean(v):.3f} ({np.mean(v) - base:+.3f})"
                if v
                else f"  {wm}x n/a"
            )
        print(line)

    if args.export:
        export(args, X, by_layer, layers)


def export(args, X, by_layer, layers):
    """W[target] learns (hidden[target-1], token t) -> routed[target, t].

    The runtime predicts the NEXT layer while still inside the current
    one, so a target's head consumes the previous MoE layer's router
    input. Layers without a predecessor simply ship no matrix — the
    runtime falls back to prev-token for them.
    """
    out = {}
    for i, tgt in enumerate(layers[1:], start=1):
        src = layers[i - 1]
        s_seq, t_seq = by_layer[src], by_layer[tgt]
        m = min(len(s_seq), len(t_seq))
        rows = np.array([s_seq[t][0] for t in range(m)])
        Y = _targets(m, [t_seq[t][1] for t in range(m)])
        W = ridge_fit(X[rows].astype(np.float64), Y, args.lam)
        out[f"W{int(tgt)}"] = W.astype(np.float16)
    width = args.export_width or 2 * args.topk
    out["meta_width"] = np.array(width)
    np.savez(args.export, **out)
    print(f"exported {len(out) - 1} heads (width={width}) -> {args.export}")


if __name__ == "__main__":
    main()
