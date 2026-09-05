#!/bin/bash
set -u
ROOT="/Volumes/SSD 4TB/DEV/omlx"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/bench/results/fasek/f1ab"
LOG="$OUT/run.log"
echo "=== f1ab 2k refill start $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"

run_arm() {
  local arm=$1 rep=$2
  shift 2
  local dir="$OUT/$arm$rep"
  mkdir -p "$dir"
  echo "--- $arm rep $rep (2k) start $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)] env=[$*]" >> "$LOG"
  env $* "$PY" -m bench.bench_expert_streaming --model qwen --budget 0 --decode 48 --prompt-len 2k --single-request --gate-tokens --min-free-gb 16 --out-dir "$dir" --out "$dir/result.json" >> "$LOG" 2>&1
  echo "--- $arm rep $rep done rc=$? $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
}
run_arm u 1
run_arm v 1 OMLX_EXPERT_STREAMING_DECODE_UNION_ROWS=0
run_arm u 2
run_arm v 2 OMLX_EXPERT_STREAMING_DECODE_UNION_ROWS=0
echo "=== f1ab 2k refill done $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
echo ALLDONE
