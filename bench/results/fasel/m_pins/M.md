# Fase M — repeat of L2 pins with fixed wiring and stage attribution

Protocol: 2k single-request decode-48, budget 0, PROFILE=1, gate-tokens,
min-free-gb 16, interleaved A (no pins) vs C (decode profile 512 MiB,
pin-sync). The shared box degraded mid-window (13.4->6.4 GB free, other
users): m_a3/m_a4 and m_c3 rode the degraded window.

## M1 wiring proven on real arms
- m_c* arms report pin_sync_effective=true, pin_regime_effective=decode,
  pin load 19.5 ms at engine load, pins_applied_before_first_request.
- Tokens 48/48 identical to the K8 reference on both arms (m_a1, m_c3).

## Stage attribution (decode phase, medians; us)

| arm | e2e p50 | preadv p50 | preadv p95 | queue p95 | tail p50 | req peak |
|-----|---------|-----------|------------|-----------|----------|----------|
| A   | 742     | 117       | 1968       | 457       | 708      | 10       |
| C   | 737     | 77        | 1290       | 389       | 708      | 10       |

- preadv p50 72-77 us: the decode working set is page-cache resident; the
  SSD serves only the tail (p95 1-3.4 ms).
- component e2e (~740 us) is dominated by the multi-run WINDOW overhead
  (queue p95 ~390-1280 us + tail ~700 us), not by SSD service time.
- Pool telemetry (m_c2): 241,362 submissions -> 241,362 started/completed,
  queue_delay_max 23 ms, active_us_max 24 ms — the pool stays balanced and
  every submitted run executes; observed active_peak_delta 16 == pool
  capacity (deep overlap across projections).
- run_sizes aggregated in bounded buckets; dropped_samples 2,055,176
  (reservoir capacity 2048 per metric) — memory stays bounded on 241k
  runs.

## Decision (L2 repeat, per plan gates)
- Clean same-window pairs (a1/c1, a2/c2): decode delta -1.4% / -0.5% —
  inside window drift, no >=5% gain.
- The +7.49% median over the whole run is load drift: m_a3 (1.562) and
  m_a4 (2.507) ran under the degraded window.
- p95 demand (page-cache reads at ~2 ms) does not drop 10% under pins.
- decision tree row applies: demand served from page cache (p50 ~77 us)
  -> pins add wired memory without moving the latency -> close additional
  residency; pin/LRU budgets stay closed. Same conclusion as Fase L2,
  now with per-stage attribution.

## effective_config bug fixed during the repeat
- run() shadowed the module helper (_effective_config = _effective_config(
  ...)), the swallowed exception left the block None. Fixed with a
  separate binding; comparator tests + a fresh arm confirm the block.
