#!/bin/bash
set -u
ROOT="/Volumes/SSD 4TB/DEV/omlx"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/bench/results/fasek/mergeab"
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== mergeab start $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"

run_arm() {
  local arm=$1 rep=$2
  shift 2
  local dir="$OUT/$arm$rep"
  mkdir -p "$dir"
  echo "--- $arm rep $rep start $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)] env=[$*]" >> "$LOG"
  env $* "$PY" -m bench.bench_expert_streaming --model qwen --budget 0 --decode 48 --prompt-len 2k --single-request --out-dir "$dir" --out "$dir/result.json" >> "$LOG" 2>&1
  echo "--- $arm rep $rep done rc=$? $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
}
run_arm m1 1
run_arm m0 1 OMLX_EXPERT_STREAMING_RUN_MERGE_GAP=0
run_arm m1 2
run_arm m0 2 OMLX_EXPERT_STREAMING_RUN_MERGE_GAP=0
run_arm m1 3
run_arm m0 3 OMLX_EXPERT_STREAMING_RUN_MERGE_GAP=0
echo "=== mergeab done $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
echo ALLDONE
