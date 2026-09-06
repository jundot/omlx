"""LRC analysis (SRP/SCH) over expert-routing traces (Fase I3).

Offline routing-consistency metrics adapted from "Not All Models Suit Expert
Offloading: On Local Routing Consistency of Mixture-of-Expert Models"
(arXiv:2505.16056), computed from a trace written by
OMLX_EXPERT_STREAMING_TRACE=<path> (one JSONL row per MoE layer call).

Metrics (per layer, aggregated):

- SCH(cache_size) — Segment Cache Best Hit Rate: the best achievable hit
  rate for an offline cache of `cache_size` expert slots that may consult
  the future. Simulated with Belady's rule adapted to batch requests: after
  serving a call, keep the `cache_size` experts of (cache ∪ needed) whose
  next use lies farthest in the future. A call needs |uniq| experts; hits
  counted against the cache contents before the call.

- SRP(group_size, segment) — Segment Routing Best Performance: how well one
  FIXED group of `group_size` experts covers a contiguous `segment`-call
  window's routing needs. Two coverages reported:
    demand — demand-weighted (group = top-G by frequency in the segment);
    distinct — |group ∩ union| / |union| (any in-union group of size G
    scores the same; reported for reference).

High SCH means an expert cache + pins can actually pay on this model; low
SCH (GLM-like) means caching cannot beat the page cache and bytes/token
levers matter more. Cache sizes around 2x active experts are the paper's
sweet spot; use --cache-sizes to sweep.

Usage:
    OMLX_EXPERT_STREAMING_TRACE=/tmp/qwen.jsonl .venv/bin/python bench/bench_expert_streaming.py --model qwen ...
    .venv/bin/python bench/lrc_analysis.py --trace /tmp/qwen.jsonl --out bench/results/lrc_qwen.json
"""

import argparse
import bisect
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_trace(path: str) -> dict[int, list[frozenset[int]]]:
    """Rows -> {layer: [needed-set per call, in call order]}."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("call", 0))
    per_layer: dict[int, list[frozenset[int]]] = defaultdict(list)
    for r in rows:
        per_layer[int(r["layer"])].append(frozenset(int(e) for e in r["uniq"]))
    return dict(per_layer)


def _occurrences(seq: list[frozenset[int]]) -> dict[int, list[int]]:
    """expert -> sorted list of call indices where it is needed."""
    occ: dict[int, list[int]] = defaultdict(list)
    for i, need in enumerate(seq):
        for e in need:
            occ[e].append(i)
    return occ


def sch(seq: list[frozenset[int]], cache_size: int) -> tuple[float, int, int]:
    """Oracle-cached hit rate over the whole sequence (classic Belady).

    On each call the needed experts found in the cache hit; missing ones are
    inserted while evicting the cached expert whose next use lies farthest in
    the future — but only when the incoming expert is strictly sooner (an
    expert never needed again never displaces one that is). Returns
    (hit_rate, total_hits, total_needed)."""
    if not seq:
        return 0.0, 0, 0

    occ = _occurrences(seq)
    horizon = len(seq)

    def next_use(e: int, i: int) -> int:
        positions = occ.get(e)
        if not positions:
            return horizon
        j = bisect.bisect_right(positions, i)
        return positions[j] if j < len(positions) else horizon

    cache: set[int] = set()
    hits = 0
    needed_total = 0
    for i, need in enumerate(seq):
        hits += len(cache & need)
        needed_total += len(need)
        for e in sorted(need - cache):
            if len(cache) >= cache_size:
                victim = max(cache, key=lambda c: next_use(c, i))
                if next_use(victim, i) > next_use(e, i):
                    cache.discard(victim)
                    cache.add(e)
            else:
                cache.add(e)
    return hits / needed_total, hits, needed_total


def srp(
    seq: list[frozenset[int]], group_size: int, segment: int
) -> list[tuple[float, float]]:
    """Per-segment (demand_coverage, distinct_coverage) of the best fixed
    group of `group_size` experts. Group chosen by segment demand (top-G
    frequency); distinct coverage is group-size bound for reference."""
    out = []
    for start in range(0, len(seq), segment):
        window = seq[start : start + segment]
        demand: Counter = Counter()
        union: set[int] = set()
        for need in window:
            demand.update(need)
            union |= need
        if not union:
            continue
        group = {e for e, _ in demand.most_common(group_size)}
        demand_cov = sum(demand[e] for e in group) / sum(demand.values())
        distinct_cov = min(group_size, len(union)) / len(union)
        out.append((demand_cov, distinct_cov))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, help="JSONL trace from OMLX_EXPERT_STREAMING_TRACE")
    ap.add_argument(
        "--cache-sizes", default="8,16,32,64,128", help="comma-separated expert-slot budgets"
    )
    ap.add_argument(
        "--group-sizes", default="16,32", help="comma-separated fixed-group sizes for SRP"
    )
    ap.add_argument("--segment", type=int, default=128, help="calls per SRP segment")
    ap.add_argument("--out", default=None, help="write a JSON result here")
    args = ap.parse_args()

    per_layer = load_trace(args.trace)
    if not per_layer:
        print("empty trace", file=sys.stderr)
        sys.exit(2)
    calls_per_layer = {len(v) for v in per_layer.values()}
    print(
        f"{len(per_layer)} layers, {min(calls_per_layer)}-{max(calls_per_layer)} calls/layer"
    )

    sizes = [int(s) for s in args.cache_sizes.split(",") if s.strip()]
    groups = [int(g) for g in args.group_sizes.split(",") if g.strip()]

    result: dict = {"trace": args.trace, "sch": {}, "srp": {}}
    for size in sizes:
        rates = [sch(seq, size)[0] for seq in per_layer.values()]
        agg = statistics.mean(rates)
        result["sch"][str(size)] = {"mean": agg, "per_layer_min": min(rates)}
        print(f"SCH(S={size:>4}): mean {agg:6.1%}  min-layer {min(rates):6.1%}")
    for g in groups:
        covs = [srp(seq, g, args.segment) for seq in per_layer.values()]
        flat_demand = [d for segs in covs for d, _ in segs]
        flat_distinct = [u for segs in covs for _, u in segs]
        if flat_demand:
            result["srp"][str(g)] = {
                "segment": args.segment,
                "demand_mean": statistics.mean(flat_demand),
                "demand_min": min(flat_demand),
                "distinct_mean": statistics.mean(flat_distinct),
            }
            print(
                f"SRP(G={g:>3}, seg={args.segment}): demand {statistics.mean(flat_demand):6.1%} "
                f"(min {min(flat_demand):6.1%})  distinct {statistics.mean(flat_distinct):6.1%}"
            )

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
