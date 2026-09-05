#!/bin/bash
set -u
ROOT="/Volumes/SSD 4TB/DEV/omlx"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/bench/results/fasek/split4"
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== split4 start $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"

run_arm() {
  local arm=$1 plen=$2
  shift 2
  local dir="$OUT/$arm"
  mkdir -p "$dir"
  echo "--- $arm start $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)] env=[$*]" >> "$LOG"
  env $* "$PY" -m bench.bench_expert_streaming --model qwen --budget 0 --decode 48 --prompt-len "$plen" --single-request --gate-tokens --cold-tier 3 --hot-fraction 0.25 --out-dir "$dir" --out "$dir/result.json" >> "$LOG" 2>&1
  echo "--- $arm done rc=$? $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
}
# 2k pair: bridge ON (default) vs OFF (RUN_MERGE_GAP=0) under the real split
run_arm c1 2k
run_arm c2 2k OMLX_EXPERT_STREAMING_RUN_MERGE_GAP=0
# 8k pair with memtrace on the bridged arm (K6 bank_bytes coherence)
run_arm c3 8k OMLX_EXPERT_STREAMING_MEMTRACE=$OUT/c3/memtrace.jsonl
run_arm c4 8k OMLX_EXPERT_STREAMING_RUN_MERGE_GAP=0
echo "=== split4 done $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
echo ALLDONE
